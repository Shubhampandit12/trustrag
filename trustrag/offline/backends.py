"""Deterministic, torch-free stand-ins for the real TrustRAG model backends.

These let the FULL pipeline run end-to-end on a laptop (no GPU, no torch,
no transformers, no faiss) while producing HONEST artifacts: every score is a
real, spread-out function of the actual text, so the abstention gate trains on
meaningful signals instead of constants.

Design constraints:
  * numpy / std-lib only (sklearn allowed elsewhere; not needed here).
  * Fully deterministic. Token hashing uses hashlib (NOT Python's salted
    builtin ``hash``), so embeddings are identical across processes/runs.
  * No use of the wall clock or unseeded randomness.

Interface parity (see the real modules for the exact signatures being matched):
  * OfflineEmbedder   -> trustrag.retrieve.embedder.Embedder
  * OfflineReranker   -> trustrag.retrieve.reranker.Reranker
  * OfflineNLI        -> trustrag.nli.NLIScorer
  * make_offline_judge-> a stand-in for eval.judge.make_llm_judge's callable
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np

from trustrag.schemas import Chunk, ScoredChunk

__all__ = [
    "OfflineEmbedder",
    "OfflineReranker",
    "OfflineNLI",
    "make_offline_judge",
]

# --------------------------------------------------------------------------- #
# Shared, dependency-free text utilities                                      #
# --------------------------------------------------------------------------- #

# A small English stopword set. These carry little lexical signal, so dropping
# them sharpens both retrieval overlap and NLI grounding estimates.
_STOPWORDS = frozenset(
    """
    a an and are as at be been being but by for from had has have he her hers him
    his i if in into is it its me my no nor not of off on once only or our ours out
    over own s so t than that the their theirs them then there these they this those
    to too was we were what when where which while who whom why will with you your
    yours do does did doing done can could should would may might must shall
    """.split()
)

# Word tokens: runs of ascii alphanumerics on the lowercased text. Numbers are
# kept as tokens (e.g. "330") so they participate in overlap/embeddings; the NLI
# number logic separately re-extracts numeric values from the raw text.
_WORD_RE = re.compile(r"[a-z0-9]+")

# Numeric literals (mirrors eval.crag_scorer._NUM_RE): optional sign, digits with
# thousands separators, optional decimal part.
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")

# Negation / refutation cues. Matched on lowercased raw text so contractions
# ("isn't", "don't") and refutation verbs ("refutes") are caught intact.
_NEG_RE = re.compile(
    r"\b(?:not|no|never|none|cannot|can't|isn't|wasn't|aren't|weren't|"
    r"didn't|doesn't|don't|without|false|incorrect|untrue|wrong|"
    r"denies?|denied|refut\w*|contrary|contradict\w*)\b"
)


def _tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Lowercase, split on non-alnum, optionally drop stopwords.

    Deterministic and pure. Numbers survive as their own tokens.
    """
    if not text:
        return []
    toks = _WORD_RE.findall(text.lower())
    if drop_stopwords:
        toks = [t for t in toks if t not in _STOPWORDS]
    return toks


def _stable_bucket(token: str, dim: int) -> int:
    """Hash a token into [0, dim) deterministically (process-independent)."""
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % dim


def _extract_numbers(text: str) -> list[float]:
    """Parse numeric literals from raw text into floats (commas stripped)."""
    out: list[float] = []
    for m in _NUM_RE.finditer(text or ""):
        raw = m.group(0).replace(",", "")
        if raw in {"", "-", ".", "-."}:
            continue
        try:
            out.append(float(raw))
        except ValueError:
            continue
    return out


def _num_close(a: float, b: float, tol: float = 0.01) -> bool:
    """Relative-tolerance numeric match (robust around zero)."""
    if a == b:
        return True
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom <= tol


def _token_f1(pred_tokens: Sequence[str], gold_tokens: Sequence[str]) -> float:
    """SQuAD-style multiset token F1 between two token sequences."""
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# Abstention / "I don't know" surface forms (kept in sync with eval.crag_scorer).
_ABSTAIN_RE = re.compile(
    r"\bi\s*do\s*n['’]?t\s+know\b"
    r"|\bi\s+cannot\s+(?:find|answer|determine)\b"
    r"|\bi\s+can['’]?t\s+(?:find|answer|determine)\b"
    r"|\bunable\s+to\s+(?:find|answer|determine)\b"
    r"|\bnot\s+enough\s+(?:information|evidence|context)\b"
    r"|\bno\s+(?:relevant\s+)?(?:information|answer|evidence)\b"
    r"|\bcannot\s+be\s+(?:found|determined|answered)\b",
    re.IGNORECASE,
)


def _is_abstention(pred: str) -> bool:
    """True if the prediction is empty or an explicit refusal / IDK."""
    if pred is None or not pred.strip():
        return True
    return bool(_ABSTAIN_RE.search(pred))


# --------------------------------------------------------------------------- #
# 1. OfflineEmbedder  (matches trustrag.retrieve.embedder.Embedder)           #
# --------------------------------------------------------------------------- #


class OfflineEmbedder:
    """Hashing bag-of-words embeddings with sublinear term frequency.

    Each token is hashed to a fixed bucket and its sublinear count accumulated;
    the row is then L2-normalized. Cosine similarity between a query and a
    passage therefore tracks (weighted) lexical overlap -- a faithful, if crude,
    stand-in for a dense encoder. Deterministic and torch-free.
    """

    def __init__(self, dim: int = 256, sublinear_tf: bool = True):
        # `model_name` kept for surface parity with the real Embedder.
        self.model_name = "offline-hashing-bow"
        self.dim = int(dim)
        self.sublinear_tf = bool(sublinear_tf)

    def _embed_one(self, text: str, out: np.ndarray) -> None:
        counts = Counter(_tokenize(text))
        if not counts:
            return  # empty / all-stopword text -> zero vector (handled by caller)
        for token, tf in counts.items():
            weight = (1.0 + math.log(tf)) if self.sublinear_tf else float(tf)
            out[_stable_bucket(token, self.dim)] += weight

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """(N, dim) float32, L2-normalized rows. Empty text -> zero row."""
        n = len(texts)
        mat = np.zeros((n, self.dim), dtype="float32")
        for i, text in enumerate(texts):
            self._embed_one(text or "", mat[i])
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        np.divide(mat, norms, out=mat, where=norms > 0.0)  # guard div-by-zero
        return mat

    def embed_query(self, query: str) -> np.ndarray:
        """(1, dim) float32, same scheme as passages (no instruction prefix)."""
        return self.embed_passages([query or ""])


# --------------------------------------------------------------------------- #
# 2. OfflineReranker  (matches trustrag.retrieve.reranker.Reranker)           #
# --------------------------------------------------------------------------- #


class OfflineReranker:
    """BM25-lite lexical reranker.

    Scores each candidate by Okapi BM25 against the query, using the candidate
    set itself as the local corpus for IDF. Produces real-valued, well-spread
    scores (relevant chunks score high, off-topic chunks score ~0), which makes
    ``rerank_top1`` / ``rerank_margin`` genuinely informative gate features.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.model_name = "offline-bm25"
        self.k1 = float(k1)
        self.b = float(b)

    def rerank(self, query: str, candidates: list[Chunk], top_n: int = 5) -> list[ScoredChunk]:
        if not candidates:
            return []

        docs = [_tokenize(c.text) for c in candidates]
        n_docs = len(docs)
        doc_lens = [len(d) for d in docs]
        total_len = sum(doc_lens)
        avgdl = (total_len / n_docs) if n_docs else 0.0

        # Document frequency of each term across the candidate corpus.
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))

        q_terms = set(_tokenize(query))

        scores: list[float] = []
        for tokens, dl in zip(docs, doc_lens):
            tf = Counter(tokens)
            s = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                n_qi = df[term]
                # Smoothed IDF, floored at >0 via ln(1 + .) so it is monotone
                # in rarity and never turns a match into a penalty.
                idf = math.log(1.0 + (n_docs - n_qi + 0.5) / (n_qi + 0.5))
                denom = f + self.k1 * (1.0 - self.b + self.b * (dl / avgdl if avgdl else 0.0))
                s += idf * (f * (self.k1 + 1.0)) / denom
            scores.append(s)

        # Stable, deterministic sort: highest score first, ties keep input order.
        order = sorted(range(n_docs), key=lambda i: scores[i], reverse=True)
        ranked = order[: max(0, top_n)]
        return [ScoredChunk(chunk=candidates[i], score=float(scores[i])) for i in ranked]


# --------------------------------------------------------------------------- #
# 3. OfflineNLI  (matches trustrag.nli.NLIScorer)                             #
# --------------------------------------------------------------------------- #


class OfflineNLI:
    """Lexical stand-in for an MNLI entailment/contradiction scorer.

    Per reranked evidence chunk:
      * entailment  = fraction of the answer's content tokens present in the
        chunk. An answer copied from context scores ~1; an ungrounded guess ~0.
      * contradiction = evidence is on-topic BUT conflicts with the answer, via
        a differing numeric value or a negation/refutation cue. Scaled by how
        on-topic the chunk is, so unrelated chunks never register contradiction.
    Both probs are in [0, 1]. Deterministic, torch-free.
    """

    # Confidence assigned to each conflict kind once the chunk is on-topic.
    _NUMERIC_CONFLICT = 0.85
    _NEGATION_CONFLICT = 0.55

    def __init__(self, model_name: str = "offline-lexical-nli"):
        self.model_name = model_name

    def score(self, answer: str, reranked: list[ScoredChunk]) -> tuple[list[float], list[float]]:
        """Return (entail_probs, contradict_probs), one per reranked chunk."""
        if not reranked or not answer or not answer.strip():
            return [], []

        ans_tokens = set(_tokenize(answer))
        ans_nums = _extract_numbers(answer)

        entail: list[float] = []
        contradict: list[float] = []

        for sc in reranked:
            ctext = sc.chunk.text or ""
            chunk_tokens = set(_tokenize(ctext))

            # --- entailment: grounding of the answer in this chunk ---
            if ans_tokens:
                shared = ans_tokens & chunk_tokens
                topic_overlap = len(shared) / len(ans_tokens)
            else:
                topic_overlap = 0.0
            entail.append(round(min(1.0, topic_overlap), 6))

            # --- contradiction: on-topic conflict signal ---
            conflict = 0.0
            if topic_overlap > 0.0:
                chunk_nums = _extract_numbers(ctext)
                if ans_nums and chunk_nums:
                    # Fire a numeric conflict unless the answer's KEY value (its
                    # last number — the payload, e.g. "$4.2 billion") has a close
                    # match in the chunk. Using `any` over all pairs would let an
                    # incidental shared number (a year/quarter) mask a real value
                    # conflict, so we anchor on the answer's key number.
                    key_ans = ans_nums[-1]
                    if not any(_num_close(key_ans, c) for c in chunk_nums):
                        conflict = max(conflict, self._NUMERIC_CONFLICT)
                if _NEG_RE.search(ctext.lower()):
                    conflict = max(conflict, self._NEGATION_CONFLICT)
            contradict.append(round(min(1.0, topic_overlap * conflict), 6))

        return entail, contradict


# --------------------------------------------------------------------------- #
# 4. make_offline_judge  (stand-in for the external LLM judge)                #
# --------------------------------------------------------------------------- #


def make_offline_judge(
    *,
    threshold: float = 0.6,
    numeric_tolerance: float = 0.01,
):
    """Build a deterministic fuzzy judge for the CRAG scorer residue.

    Returns a callable ``judge(pred, gold, alts) -> {correct|incorrect|missing}``
    matching the signature the scorer expects for its optional ``llm_judge``.
    It stands in for an external LLM so evaluation runs fully offline.

    Logic (conservative):
      * blank / IDK-like prediction              -> "missing"
      * numeric match against any gold/alt        -> "correct"
      * token-F1 against any gold/alt >= threshold-> "correct"
      * otherwise                                 -> "incorrect"
    """
    thr = float(threshold)
    tol = float(numeric_tolerance)

    def judge(pred: str, gold: str, alts: Iterable[str] = ()) -> str:
        if _is_abstention(pred):
            return "missing"

        golds = [g for g in [gold, *(alts or [])] if g and str(g).strip()]
        if not golds:
            # No reference to compare against; stay conservative.
            return "incorrect"

        pred_tokens = _tokenize(pred)
        pred_nums = _extract_numbers(pred)

        best_f1 = 0.0
        numeric_mismatch = False  # both sides numeric on some gold, but no match
        for g in golds:
            g = str(g)
            # Numeric agreement is a strong, precise correctness signal.
            gold_nums = _extract_numbers(g)
            if pred_nums and gold_nums:
                if any(_num_close(p, q, tol) for p in pred_nums for q in gold_nums):
                    return "correct"
                # Numbers present on both sides but none agree: the core fact is
                # wrong. Don't let shared prose (e.g. "billion") rescue it via F1.
                numeric_mismatch = True
            best_f1 = max(best_f1, _token_f1(pred_tokens, _tokenize(g)))

        if numeric_mismatch:
            return "incorrect"
        return "correct" if best_f1 >= thr else "incorrect"

    return judge

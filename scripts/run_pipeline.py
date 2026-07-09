"""Batch inference over a CRAG split -> preds/{system}.jsonl.

With --no-gate (abstention OFF) this produces the answers + features used to
train the gate. With a fitted gate it produces the served predictions.

Usage:
  python scripts/run_pipeline.py --split dev_fit --no-gate --out preds/dev_fit.jsonl
  python scripts/run_pipeline.py --split test --out preds/test.jsonl
"""
from __future__ import annotations

import argparse
import json

from trustrag.config import load_config, resolve_path
from trustrag.data.crag_loader import record_to_pages, stream_records
from trustrag.data.make_splits import load_split_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev_fit", "dev_calib", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-gate", action="store_true", help="abstention OFF (label generation)")
    ap.add_argument("--limit", type=int, default=None, help="cap #questions (dev slice iteration)")
    args = ap.parse_args()

    cfg = load_config()
    ids = load_split_ids(resolve_path(cfg.data.splits_dir), args.split)
    from trustrag.pipeline import build_pipeline
    pipe = build_pipeline(cfg, with_gate=not args.no_gate, with_nli=True)

    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as fout:
        for rec in stream_records(resolve_path(cfg.data.raw_path)):
            if rec.interaction_id not in ids:
                continue
            pages = record_to_pages(rec)
            ans = pipe.answer(rec.query, pages, query_time=rec.query_time)
            fout.write(json.dumps({
                "interaction_id": rec.interaction_id,
                "query": rec.query, "gold": rec.answer, "alt_ans": rec.alt_ans,
                "domain": rec.domain, "question_type": rec.question_type,
                "is_finance": rec.is_finance,
                "pred": ans.text, "raw_answer": ans.raw_answer,
                "abstained": ans.abstained, "confidence": ans.confidence,
                "features": ans.features, "citations": ans.citations,
            }) + "\n")
            n += 1
            if args.limit and n >= args.limit:
                break
    print(f"wrote {n} predictions -> {out_path}")


if __name__ == "__main__":
    main()

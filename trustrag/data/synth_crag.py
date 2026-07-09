"""Generate a synthetic CRAG-shaped dataset for offline execution.

Produces records in the exact real CRAG schema so stream_records() works unchanged.
Controlled difficulty ensures downstream features spread and the gate has signal:
  - answerable-easy (45%): gold sentence strongly overlaps the query
  - answerable-hard (20%): gold is buried; a distractor overlaps more
  - false_premise (15%): query embeds a false premise; pages contradict it
  - unanswerable (20%): needed fact absent from all pages

Usage: python -m trustrag.data.synth_crag
"""
from __future__ import annotations

import bz2
import json
from pathlib import Path

import numpy as np

# ============ FACT BANK ============
# (entity, attribute, value, domain)
FACTS = [
    ("Acme Corp", "Q3 2023 revenue", "$4.2 billion", "finance"),
    ("Globex Industries", "2022 net profit", "$890 million", "finance"),
    ("First National Bank", "founding year", "1952", "finance"),
    ("TechVault Inc", "market cap January 2024", "$12.8 billion", "finance"),
    ("Pacific Trading", "CEO since 2021", "Sarah Chen", "finance"),
    ("MetaWave Holdings", "annual dividend yield", "3.2%", "finance"),

    ("Manchester City", "2022-23 Premier League goals scored", "94", "sports"),
    ("Serena Williams", "total Grand Slam singles titles", "23", "sports"),
    ("Usain Bolt", "100m world record time", "9.58 seconds", "sports"),
    ("Golden State Warriors", "2022 NBA Championship opponent", "Boston Celtics", "sports"),
    ("Lionel Messi", "World Cup wins as of 2023", "1", "sports"),
    ("India cricket team", "2023 ODI World Cup final opponent", "Australia", "sports"),

    ("The Beatles", "final studio album", "Let It Be", "music"),
    ("Taylor Swift", "Eras Tour start year", "2023", "music"),
    ("Beethoven", "total numbered symphonies", "9", "music"),
    ("BTS", "first Billboard Hot 100 number-one single", "Dynamite", "music"),
    ("Pink Floyd", "Dark Side of the Moon release year", "1973", "music"),
    ("Ed Sheeran", "debut album name", "Plus (stylized as +)", "music"),

    ("Oppenheimer", "2023 film director", "Christopher Nolan", "movie"),
    ("Parasite", "Oscar Best Picture year", "2020", "movie"),
    ("Avatar: The Way of Water", "worldwide box office gross", "$2.32 billion", "movie"),
    ("The Godfather", "original release year", "1972", "movie"),
    ("Spider-Man: No Way Home", "opening weekend US gross", "$260 million", "movie"),
    ("Barbie", "2023 film lead actress", "Margot Robbie", "movie"),

    ("Mount Everest", "height in meters", "8,849 meters", "open"),
    ("Python programming language", "creator", "Guido van Rossum", "open"),
    ("James Webb Space Telescope", "launch year", "2021", "open"),
    ("Tokyo", "2020 Olympic Games held in year", "2021", "open"),
    ("ChatGPT", "initial public release month", "November 2022", "open"),
    ("Mars", "number of moons", "2", "open"),
]

QUESTION_TYPES = ["simple", "comparison", "aggregation", "multi_hop",
                  "false_premise", "set", "post_processing", "multi_turn"]

DIFFICULTIES = ["answerable_easy", "answerable_hard", "false_premise", "unanswerable"]
DIFF_WEIGHTS = [0.45, 0.20, 0.15, 0.20]

# Distractor sentences per domain (on-topic but don't contain the answer)
DISTRACTORS = {
    "finance": [
        "The stock market saw increased volatility in the second quarter.",
        "Analysts predict a moderate growth in the tech sector for next year.",
        "Federal Reserve maintained interest rates at their December meeting.",
        "ESG investing continues to gain traction among institutional investors.",
    ],
    "sports": [
        "The tournament attracted record viewership across all platforms.",
        "Several athletes broke personal records at the qualifying rounds.",
        "The coaching staff underwent significant changes during the off-season.",
        "Stadium renovations are expected to be completed by next season.",
    ],
    "music": [
        "The album debuted at the top of several international charts.",
        "Critics praised the innovative production techniques used in the recording.",
        "The world tour sold out within minutes of tickets going on sale.",
        "Streaming numbers for the genre have increased significantly this year.",
    ],
    "movie": [
        "The film received mixed reviews from international critics.",
        "Production costs escalated due to extensive visual effects work.",
        "The sequel was announced shortly after the premiere event.",
        "Several A-list actors were considered for the lead role before casting.",
    ],
    "open": [
        "Research in this area has accelerated in recent years.",
        "The technology has applications across multiple industries.",
        "Public interest surged following the announcement.",
        "Experts remain divided on the long-term implications.",
    ],
}


def _make_page_html(text: str) -> str:
    return f"<html><body><p>{text}</p></body></html>"


def _gold_sentence(entity: str, attribute: str, value: str) -> str:
    """Generate the sentence containing the gold answer."""
    templates = [
        f"{entity} has {attribute} of {value}.",
        f"The {attribute} of {entity} is {value}.",
        f"According to records, {entity}'s {attribute} stands at {value}.",
        f"{entity} reported {attribute} at {value}.",
    ]
    # deterministic pick based on entity hash
    idx = hash(entity) % len(templates)
    return templates[idx]


def _make_query(entity: str, attribute: str) -> str:
    templates = [
        f"What is the {attribute} of {entity}?",
        f"What is {entity}'s {attribute}?",
        f"Can you tell me the {attribute} for {entity}?",
    ]
    idx = hash(entity + attribute) % len(templates)
    return templates[idx]


def generate_synthetic_crag(out_path: str, n: int = 600, seed: int = 13) -> None:
    rng = np.random.default_rng(seed)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Assign difficulties
    diffs = rng.choice(DIFFICULTIES, size=n, p=DIFF_WEIGHTS)

    records = []
    for i in range(n):
        diff = diffs[i]
        # Pick a random fact
        fact_idx = int(rng.integers(0, len(FACTS)))
        entity, attribute, value, domain = FACTS[fact_idx]
        gold_sent = _gold_sentence(entity, attribute, value)

        # Assign question_type
        if diff == "false_premise":
            qtype = "false_premise"
        else:
            qtype = QUESTION_TYPES[int(rng.integers(0, len(QUESTION_TYPES) - 1))]  # exclude false_premise

        # Build pages (3-5 pages, each with 2-4 sentences as HTML)
        n_pages = int(rng.integers(3, 6))
        pages = []
        dist_pool = list(DISTRACTORS.get(domain, DISTRACTORS["open"]))

        if diff == "answerable_easy":
            # Gold sentence in page 0, with strong query overlap
            query = _make_query(entity, attribute)
            answer = value
            alt_ans = [f"It is {value}", value.replace("$", "")]

            # Page 0: gold + 1 distractor
            p0_text = f"{gold_sent} {dist_pool[0]}"
            pages.append({"page_url": f"http://example.com/p{i}_0",
                          "page_name": f"Page about {entity}",
                          "page_result": _make_page_html(p0_text)})
            # Remaining pages: distractors only
            for pi in range(1, n_pages):
                d_idx = pi % len(dist_pool)
                pages.append({"page_url": f"http://example.com/p{i}_{pi}",
                              "page_name": f"Related article {pi}",
                              "page_result": _make_page_html(dist_pool[d_idx])})

        elif diff == "answerable_hard":
            # Gold sentence buried in a later page; page 0 has a stronger-overlap distractor
            query = _make_query(entity, attribute)
            answer = value
            alt_ans = [value]
            # Page 0: distractor that shares many query tokens
            misleading = f"{entity} was discussed in relation to {attribute} but no figure was confirmed."
            p0_text = f"{misleading} {dist_pool[0]}"
            pages.append({"page_url": f"http://example.com/p{i}_0",
                          "page_name": f"Analysis of {entity}",
                          "page_result": _make_page_html(p0_text)})
            # Gold goes in page 2 or 3
            gold_page = min(2, n_pages - 1)
            for pi in range(1, n_pages):
                if pi == gold_page:
                    p_text = f"{dist_pool[pi % len(dist_pool)]} {gold_sent}"
                else:
                    p_text = dist_pool[pi % len(dist_pool)]
                pages.append({"page_url": f"http://example.com/p{i}_{pi}",
                              "page_name": f"Source {pi}",
                              "page_result": _make_page_html(p_text)})

        elif diff == "false_premise":
            # Query embeds a false premise; pages contain the TRUE contradicting fact
            # e.g. "When did X win award Y?" when X never won Y
            other_fact_idx = (fact_idx + 1) % len(FACTS)
            other_entity = FACTS[other_fact_idx][0]
            query = f"When did {other_entity} achieve {entity}'s {attribute} of {value}?"
            answer = f"The premise is false. {other_entity} did not achieve {entity}'s {attribute}."
            alt_ans = ["The premise is false", "false premise"]
            # Pages contain the TRUE fact about entity (contradicts the premise)
            contradiction = f"{entity} holds {attribute} of {value}, not {other_entity}."
            for pi in range(n_pages):
                if pi == 0:
                    p_text = contradiction
                else:
                    p_text = dist_pool[pi % len(dist_pool)]
                pages.append({"page_url": f"http://example.com/p{i}_{pi}",
                              "page_name": f"Fact check {pi}",
                              "page_result": _make_page_html(p_text)})

        else:  # unanswerable
            # The needed fact is absent from all pages
            query = _make_query(entity, attribute)
            answer = value  # gold for scoring, but NOT in pages
            alt_ans = [value]
            # Pages: only distractors (no gold sentence anywhere)
            for pi in range(n_pages):
                p_text = dist_pool[pi % len(dist_pool)]
                pages.append({"page_url": f"http://example.com/p{i}_{pi}",
                              "page_name": f"Article {pi}",
                              "page_result": _make_page_html(p_text)})

        records.append({
            "interaction_id": f"synth_{i:04d}",
            "query": query,
            "answer": answer,
            "alt_ans": alt_ans,
            "query_time": "2024-03-15 10:00:00",
            "domain": domain,
            "question_type": qtype,
            "static_or_dynamic": "static" if rng.random() < 0.7 else "dynamic",
            "search_results": pages,
        })

    # Write as bz2 JSONL
    with bz2.open(path, "wt", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    # Print summary
    from collections import Counter
    domain_counts = Counter(r["domain"] for r in records)
    type_counts = Counter(r["question_type"] for r in records)
    print(f"Generated {len(records)} synthetic CRAG records -> {path}")
    print(f"  domains: {dict(domain_counts)}")
    print(f"  types  : {dict(type_counts)}")


if __name__ == "__main__":
    from trustrag.config import load_config, resolve_path
    cfg = load_config()
    out = resolve_path(cfg.data.raw_path)
    generate_synthetic_crag(str(out))

"""
Retrieval eval: precision@k on hand-labeled queries, for each retrieval mode.

Each query in tests/data/labeled_queries.json says what a relevant song looks
like (allowed genres and/or feature ranges) instead of listing song ids, since
nobody can hand-label 81k songs. A result counts as relevant if it meets every
criterion. Modes compared:
  dense    - cosine search over song cards only
  feature  - FeatureScorer only, using the pseudo-relevance profile
  hybrid   - the full HybridRetriever (RRF of both)

Run from the project root (needs the built index):
    python scripts/eval_retrieval.py            # summary table
    python scripts/eval_retrieval.py --show 3   # also print each query's top 3 hybrid songs
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.data import load_catalog  # noqa: E402
from src.rag.feature_scorer import FeatureScorer  # noqa: E402
from src.rag.retrieve import HybridRetriever  # noqa: E402

LABELS = Path("tests/data/labeled_queries.json")


def is_relevant(row: Dict, label: Dict) -> bool:
    genres = label.get("genres_any")
    if genres and not set(row["genres"]) & set(genres):
        return False
    return all(lo <= row[feature] <= hi for feature, (lo, hi) in label.get("ranges", {}).items())


def precision(indices: List[int], rows: List[Dict], label: Dict) -> float:
    return sum(is_relevant(rows[i], label) for i in indices) / len(indices) if indices else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--show", type=int, default=0, help="print this many hybrid results per query")
    args = parser.parse_args()

    labels = json.loads(LABELS.read_text())
    rows = load_catalog().to_dict("records")
    retriever = HybridRetriever.from_disk()

    print(f"precision@{args.k} on {len(labels)} labeled queries\n")
    print(f"{'query':<45} {'dense':>6} {'feature':>8} {'hybrid':>7}")
    totals = {"dense": [], "feature": [], "hybrid": []}
    for label in labels:
        q = label["query"]
        dense_hits = retriever.dense(q, retriever.dense_n)
        profile, _ = retriever.build_profile(None, dense_hits)
        modes = {
            "dense": [i for i, _ in dense_hits[: args.k]],
            "feature": [i for i, _ in FeatureScorer.rank(retriever.scorer.scores(profile), args.k)],
            "hybrid": [h.index for h in retriever.retrieve(query=q, k=args.k).hits],
        }
        scores = {m: precision(idx, rows, label) for m, idx in modes.items()}
        for m, s in scores.items():
            totals[m].append(s)
        print(f"{q[:45]:<45} {scores['dense']:>6.1f} {scores['feature']:>8.1f} {scores['hybrid']:>7.1f}")
        for i in modes["hybrid"][: args.show]:
            r = rows[i]
            mark = "+" if is_relevant(r, label) else "-"
            print(f"    {mark} {r['title']} — {r['artist']} [{', '.join(r['genres'])}]")

    print("-" * 69)
    print(f"{'mean':<45} {np.mean(totals['dense']):>6.2f} {np.mean(totals['feature']):>8.2f} {np.mean(totals['hybrid']):>7.2f}")


if __name__ == "__main__":
    main()

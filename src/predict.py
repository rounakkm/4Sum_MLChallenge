"""Run the full inference pipeline: blocking -> scoring -> thresholding -> TSV output.

Writes both required output files:
  - candidate_pairs.tsv   (blocking stage output, before the matcher narrows it down)
  - matching_results.tsv  (final matches after thresholding -- scored on the leaderboard)

Usage:
    python -m src.predict --data-dir dataset/test --split test --model model.pkl \
        --threshold 0.8 --output-dir output
"""
import argparse
import pickle
from pathlib import Path

import pandas as pd

from .data import load_all
from .blocking import generate_all_candidates
from .features import pair_features, FEATURE_NAMES


def score_candidates(s1_df, s2_df, s3_df, candidates, model):
    """Return dict: source1_entity_id -> list of (candidate_id, probability)."""
    lookup2 = s2_df.set_index("entity_id")
    lookup3 = s3_df.set_index("entity_id")
    s1_lookup = s1_df.set_index("entity_id")

    results = {}
    for eid, cand_ids in candidates.items():
        name_a = s1_lookup.loc[eid, "business_name"]
        addr_a = s1_lookup.loc[eid, "business_address"]
        country_a = s1_lookup.loc[eid, "country"]

        rows, ids = [], []
        for cid in cand_ids:
            if cid.startswith("S2-") and cid in lookup2.index:
                row = lookup2.loc[cid]
            elif cid.startswith("S3-") and cid in lookup3.index:
                row = lookup3.loc[cid]
            else:
                continue
            feats = pair_features(
                name_a, addr_a, country_a,
                row["business_name"], row["business_address"], row["country"],
            )
            rows.append(feats)
            ids.append(cid)

        if not rows:
            results[eid] = []
            continue

        X = pd.DataFrame(rows, columns=FEATURE_NAMES)
        probs = model.predict_proba(X)[:, 1]
        results[eid] = list(zip(ids, probs))
    return results


def apply_threshold(scored, threshold):
    return {eid: [cid for cid, p in pairs if p >= threshold] for eid, pairs in scored.items()}


def write_tsv(predictions, out_path, id_col, list_col):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"{id_col}\t{list_col}\n")
        for eid, ids in predictions.items():
            # dedupe while preserving order, per the "no duplicate IDs" rule
            seen, deduped = set(), []
            for i in ids:
                if i not in seen:
                    seen.add(i)
                    deduped.append(i)
            f.write(f"{eid}\t{','.join(deduped)}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset/test")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default="model.pkl")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    s1_df, s2_df, s3_df, _ = load_all(args.data_dir, split=args.split)

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    candidates = generate_all_candidates(s1_df, s2_df, s3_df, top_k=args.top_k)
    write_tsv(candidates, Path(args.output_dir) / "candidate_pairs.tsv",
              "source1_entity_id", "candidate_entity_ids")

    scored = score_candidates(s1_df, s2_df, s3_df, candidates, model)
    predictions = apply_threshold(scored, args.threshold)
    write_tsv(predictions, Path(args.output_dir) / "matching_results.tsv",
              "source1_entity_id", "matched_entity_ids")

    print(f"Wrote candidate_pairs.tsv and matching_results.tsv to {args.output_dir}")


if __name__ == "__main__":
    main()

"""Train the pairwise matching model.

Builds a labeled training set from blocking candidates (plus any ground-truth
positives blocking missed, so the model always sees every true match at least
once), extracts similarity features, and fits a LightGBM classifier.

Usage:
    python -m src.train --data-dir dataset/train --model-out model.pkl
"""
import argparse
import pickle

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from .data import load_all
from .blocking import generate_all_candidates
from .features import pair_features, FEATURE_NAMES


def build_training_set(s1_df, s2_df, s3_df, gt, top_k=20):
    lookup2 = s2_df.set_index("entity_id")
    lookup3 = s3_df.set_index("entity_id")
    s1_lookup = s1_df.set_index("entity_id")

    candidates = generate_all_candidates(s1_df, s2_df, s3_df, top_k=top_k)

    rows, labels = [], []
    for eid, cand_ids in candidates.items():
        true_ids = gt.get(eid, set())
        name_a = s1_lookup.loc[eid, "business_name"]
        addr_a = s1_lookup.loc[eid, "business_address"]
        country_a = s1_lookup.loc[eid, "country"]

        # Union with true_ids so the model always trains on every true match,
        # even ones the blocking stage failed to surface (fix blocking later
        # if this union set is large -- it signals a recall gap).
        for cid in set(cand_ids) | true_ids:
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
            labels.append(1 if cid in true_ids else 0)

    X = pd.DataFrame(rows, columns=FEATURE_NAMES)
    y = pd.Series(labels)
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset/train")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--model-out", default="model.pkl")
    args = parser.parse_args()

    s1_df, s2_df, s3_df, gt = load_all(args.data_dir, split="train")
    X, y = build_training_set(s1_df, s2_df, s3_df, gt, top_k=args.top_k)
    print(f"Training examples: {len(X)}  positives: {y.sum()}  negatives: {(y == 0).sum()}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    with open(args.model_out, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {args.model_out}")


if __name__ == "__main__":
    main()

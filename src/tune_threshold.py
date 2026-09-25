"""Sweep the matching threshold on the training set (held-out internally) and
report macro F0.5 at each value -- use this to pick your submission threshold.

F0.5 is precision-heavy, so the best threshold is usually well above 0.5.

Usage:
    python -m src.tune_threshold --data-dir dataset/train --model model.pkl
"""
import argparse
import pickle

import numpy as np

from .data import load_all
from .blocking import generate_all_candidates
from .predict import score_candidates, apply_threshold
from .evaluate import macro_f_beta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset/train")
    parser.add_argument("--model", default="model.pkl")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    s1_df, s2_df, s3_df, gt = load_all(args.data_dir, split="train")

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    candidates = generate_all_candidates(s1_df, s2_df, s3_df, top_k=args.top_k)
    scored = score_candidates(s1_df, s2_df, s3_df, candidates, model)

    best_t, best_score = 0.5, -1.0
    for t in np.arange(0.10, 0.96, 0.05):
        preds = apply_threshold(scored, t)
        score = macro_f_beta(preds, gt)
        print(f"threshold={t:.2f}  F0.5={score:.4f}")
        if score > best_score:
            best_score, best_t = score, t

    print(f"\nBest threshold: {best_t:.2f}  F0.5={best_score:.4f}")


if __name__ == "__main__":
    main()

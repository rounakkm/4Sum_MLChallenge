"""Train the pairwise matching model using cleaned training data.

Uses balanced sampling across US and India, dual-channel blocking,
fast feature extraction into NumPy arrays, and LightGBM classification
optimized for macro F0.5.

Usage:
    python -m src.train --data-dir dataset/train --model-out model.pkl
"""

import argparse
import gc
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .data import load_all
from .blocking import FastBlockingIndex
from .normalize import normalize_name, normalize_address, normalize_country
from .features import extract_features_into_array, get_address_numbers, FEATURE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_training_set(s1_df, s2_df, s3_df, gt, sample_size=15000, max_cands=20, neg_ratio=4):
    """Build a balanced, high-quality training set from ground truth and blocking."""
    logger.info(f"Building training set (sample_size={sample_size})...")

    # Sample S1 entities stratified by matches vs singletons
    eids_with_matches = [e for e, m in gt.items() if m]
    eids_singletons = [e for e, m in gt.items() if not m]

    n_with = min(int(sample_size * 0.94), len(eids_with_matches))
    n_single = min(sample_size - n_with, len(eids_singletons))

    rng = np.random.RandomState(42)
    chosen_with = set(rng.choice(eids_with_matches, n_with, replace=False))
    chosen_single = set(rng.choice(eids_singletons, n_single, replace=False))
    chosen_eids = chosen_with | chosen_single

    s1_sample = s1_df[s1_df['entity_id'].isin(chosen_eids)].copy().reset_index(drop=True)
    logger.info(f"Sampled {len(s1_sample)} S1 entities ({len(chosen_with)} matched, {len(chosen_single)} singletons)")

    needed_s2 = set()
    needed_s3 = set()
    for eid in s1_sample['entity_id']:
        for m in gt.get(eid, set()):
            if m.startswith('S2-'):
                needed_s2.add(m)
            elif m.startswith('S3-'):
                needed_s3.add(m)

    logger.info(f"True positive target IDs: S2={len(needed_s2)}, S3={len(needed_s3)}")

    s2_pos = s2_df[s2_df['entity_id'].isin(needed_s2)]
    s3_pos = s3_df[s3_df['entity_id'].isin(needed_s3)]

    s2_noise = s2_df[~s2_df['entity_id'].isin(needed_s2)].sample(n=min(50000, len(s2_df)), random_state=42)
    s3_noise = s3_df[~s3_df['entity_id'].isin(needed_s3)].sample(n=min(50000, len(s3_df)), random_state=42)

    s2_sub = pd.concat([s2_pos, s2_noise]).reset_index(drop=True)
    s3_sub = pd.concat([s3_pos, s3_noise]).reset_index(drop=True)

    # Build blocking indices (which automatically normalizes and stores records)
    logger.info("Building blocking indices...")
    idx2 = FastBlockingIndex(max_postings=2000)
    idx2.add_records(s2_sub)

    idx3 = FastBlockingIndex(max_postings=2000)
    idx3.add_records(s3_sub)

    # Pre-normalize S1 records
    rec_s1 = {}
    for row in s1_sample.itertuples(index=False):
        eid = row.entity_id
        na = normalize_name(row.business_name)
        aa = normalize_address(row.business_address)
        ca = normalize_country(row.country)
        rec_s1[eid] = (na, aa, ca, set(na.split()), set(aa.split()), get_address_numbers(aa))

    # Pre-allocate feature list
    feature_tuples = []  # list of tuples: (na, aa, ca, nb, ab, cb, na_tok, nb_tok, aa_tok, ab_tok, na_nums, nb_nums, label)

    logger.info("Generating candidate pairs...")
    for eid in tqdm(s1_sample['entity_id'], desc="Candidate generation"):
        if eid not in rec_s1:
            continue
        na, aa, ca, na_tok, aa_tok, na_nums = rec_s1[eid]
        true_ids = gt.get(eid, set())

        # Retrieve blocking candidates
        c2 = idx2.query(na, aa, max_candidates=max_cands)
        c3 = idx3.query(na, aa, max_candidates=max_cands)
        cand_ids = set(c2 + c3)

        # 1. Positives
        for cid in true_ids:
            rec_target = idx2.records.get(cid) if cid.startswith('S2-') else idx3.records.get(cid)
            if rec_target is not None:
                nb, ab, cb, nb_tok, ab_tok, nb_nums = rec_target
                feature_tuples.append((na, aa, ca, nb, ab, cb, na_tok, nb_tok, aa_tok, ab_tok, na_nums, nb_nums, 1))

        # 2. Negatives
        non_match_cands = list(cand_ids - true_ids)
        if non_match_cands:
            max_neg = max(neg_ratio * max(1, len(true_ids)), 3)
            if len(non_match_cands) > max_neg:
                non_match_cands = non_match_cands[:max_neg]

            for cid in non_match_cands:
                rec_target = idx2.records.get(cid) if cid.startswith('S2-') else idx3.records.get(cid)
                if rec_target is not None:
                    nb, ab, cb, nb_tok, ab_tok, nb_nums = rec_target
                    feature_tuples.append((na, aa, ca, nb, ab, cb, na_tok, nb_tok, aa_tok, ab_tok, na_nums, nb_nums, 0))

    n_samples = len(feature_tuples)
    n_feats = len(FEATURE_NAMES)
    logger.info(f"Allocating fast NumPy feature array ({n_samples} x {n_feats})...")

    X_arr = np.empty((n_samples, n_feats), dtype=np.float32)
    y_arr = np.empty(n_samples, dtype=np.int32)

    for i in range(n_samples):
        na, aa, ca, nb, ab, cb, na_tok, nb_tok, aa_tok, ab_tok, na_nums, nb_nums, label = feature_tuples[i]
        extract_features_into_array(X_arr, i, na, aa, ca, nb, ab, cb, na_tok, nb_tok, aa_tok, ab_tok, na_nums, nb_nums)
        y_arr[i] = label

    pos_count = int(np.sum(y_arr))
    neg_count = n_samples - pos_count
    logger.info(f"Feature matrix ready: {n_samples} samples ({pos_count} positives, {neg_count} negatives)")
    return X_arr, y_arr


def train_and_evaluate(X, y, model_out="model.pkl"):
    """Fit LightGBM model, tune threshold for F0.5, and save to disk."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    logger.info("Training LightGBM classifier...")
    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=8,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
    )

    # Validation threshold tuning
    val_probs = model.predict_proba(X_val)[:, 1]
    best_t = 0.85
    best_f05 = 0.0

    logger.info("Validating thresholds on held-out split:")
    for t in np.arange(0.50, 0.98, 0.05):
        preds = (val_probs >= t).astype(int)
        tp = ((preds == 1) & (y_val == 1)).sum()
        fp = ((preds == 1) & (y_val == 0)).sum()
        fn = ((preds == 0) & (y_val == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (prec + rec) > 0 else 0
        logger.info(f"  Threshold {t:.2f} -> Precision={prec:.4f}, Recall={rec:.4f}, F0.5={f05:.4f}")
        if f05 > best_f05:
            best_f05 = f05
            best_t = t

    logger.info(f"Optimal threshold: {best_t:.2f} (F0.5 = {best_f05:.4f})")

    # Feature importance
    importances = sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda x: -x[1])
    logger.info("Top 12 features by importance:")
    for name, imp in importances[:12]:
        logger.info(f"  {name}: {imp}")

    # Save model and metadata
    model_data = {
        "model": model,
        "best_threshold": float(best_t),
        "feature_names": FEATURE_NAMES,
    }
    with open(model_out, "wb") as f:
        pickle.dump(model_data, f)
    logger.info(f"Model and metadata saved to {model_out}")
    return model, best_t


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM entity resolution model.")
    parser.add_argument("--data-dir", default="dataset/train", help="Directory with cleaned training files.")
    parser.add_argument("--sample-size", type=int, default=15000, help="Number of S1 entities to sample.")
    parser.add_argument("--model-out", default="model.pkl", help="Output path for model pickle.")
    args = parser.parse_args()

    t0 = time.time()
    logger.info("Loading training data...")
    s1_df, s2_df, s3_df, gt = load_all(args.data_dir, split="train")

    X, y = build_training_set(s1_df, s2_df, s3_df, gt, sample_size=args.sample_size)
    model, best_t = train_and_evaluate(X, y, model_out=args.model_out)
    logger.info(f"Training completed successfully in {time.time()-t0:.1f}s!")


if __name__ == "__main__":
    main()

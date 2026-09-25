"""End-to-end entity resolution pipeline.

Single entry point that handles:
  1. Data loading and cleaning
  2. Blocking / candidate generation
  3. Feature extraction and model training (with validation)
  4. Threshold optimization for F0.5
  5. Test inference and output generation

Designed for large-scale data (millions of records) with memory-efficient
processing using sampling and chunked operations.

Usage:
    python -m src.pipeline [--sample-size N] [--top-k K] [--output-dir DIR]
"""
import argparse
import gc
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .data import load_source, load_ground_truth, _find_source_file, _find_gt_file
from .blocking import generate_candidates_blocking, generate_all_candidates
from .features import pair_features, FEATURE_NAMES
from .evaluate import macro_f_beta, score_entity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_training_set_sampled(s1_df, s2_df, s3_df, gt,
                                sample_size=50000, max_candidates=30,
                                neg_ratio=5):
    """Build training set with smart sampling for large datasets.

    Samples a subset of S1 entities for training. For each sampled entity:
      - Generates blocking candidates
      - Creates positive pairs from ground truth
      - Creates negative pairs from blocking candidates that are NOT matches

    Args:
        sample_size: Number of S1 entities to sample for training
        max_candidates: Max blocking candidates per entity
        neg_ratio: Max negative pairs per positive pair
    """
    logger.info(f"Building training set (sample_size={sample_size})...")

    # Sample S1 entities — stratify by having matches or not
    has_match = s1_df["entity_id"].isin(
        [eid for eid, ids in gt.items() if ids]
    )
    n_with = min(int(sample_size * 0.95), has_match.sum())
    n_without = min(sample_size - n_with, (~has_match).sum())

    s1_with = s1_df[has_match].sample(n=n_with, random_state=42)
    s1_without = s1_df[~has_match].sample(n=n_without, random_state=42)
    s1_sample = pd.concat([s1_with, s1_without]).reset_index(drop=True)

    logger.info(f"Sampled {len(s1_sample)} S1 entities "
                f"({n_with} with matches, {n_without} singletons)")

    # Generate blocking candidates for the sampled entities
    candidates = generate_all_candidates(
        s1_sample, s2_df, s3_df, max_candidates=max_candidates
    )

    # Build lookup tables
    lookup2 = s2_df.set_index("entity_id")
    lookup3 = s3_df.set_index("entity_id")
    s1_lookup = s1_sample.set_index("entity_id")

    rows, labels = [], []
    blocking_recall_hits = 0
    blocking_recall_total = 0

    for eid in tqdm(s1_sample["entity_id"], desc="Building features"):
        true_ids = gt.get(eid, set())
        cand_ids = set(candidates.get(eid, []))

        name_a = s1_lookup.loc[eid, "business_name"]
        addr_a = s1_lookup.loc[eid, "business_address"]
        country_a = s1_lookup.loc[eid, "country"]

        # Track blocking recall
        if true_ids:
            blocking_recall_total += len(true_ids)
            blocking_recall_hits += len(true_ids & cand_ids)

        # Positive examples: ground truth matches
        for cid in true_ids:
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
            labels.append(1)

        # Negative examples: blocking candidates that are NOT true matches
        neg_cands = list(cand_ids - true_ids)
        # Limit negatives per entity
        max_neg = max(neg_ratio * len(true_ids), 3) if true_ids else 3
        if len(neg_cands) > max_neg:
            rng = np.random.RandomState(hash(eid) % (2**31))
            neg_cands = list(rng.choice(neg_cands, max_neg, replace=False))

        for cid in neg_cands:
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
            labels.append(0)

    X = pd.DataFrame(rows, columns=FEATURE_NAMES)
    y = pd.Series(labels)

    if blocking_recall_total > 0:
        recall = blocking_recall_hits / blocking_recall_total
        logger.info(f"Blocking recall (on sample): {recall:.4f} "
                    f"({blocking_recall_hits}/{blocking_recall_total})")

    return X, y


def train_model(X, y, val_size=0.15):
    """Train a LightGBM classifier optimized for F0.5 (precision-heavy)."""
    logger.info(f"Training set: {len(X)} examples, "
                f"{y.sum()} positives ({100*y.mean():.1f}%), "
                f"{(y==0).sum()} negatives")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=42, stratify=y
    )

    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=8,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
    )

    # Report validation metrics
    val_probs = model.predict_proba(X_val)[:, 1]
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        preds = (val_probs >= t).astype(int)
        tp = ((preds == 1) & (y_val == 1)).sum()
        fp = ((preds == 1) & (y_val == 0)).sum()
        fn = ((preds == 0) & (y_val == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (prec + rec) > 0 else 0
        logger.info(f"  threshold={t:.1f}: P={prec:.4f} R={rec:.4f} F0.5={f05:.4f}")

    # Feature importance
    importances = sorted(
        zip(FEATURE_NAMES, model.feature_importances_),
        key=lambda x: -x[1]
    )
    logger.info("Top features:")
    for fname, imp in importances[:10]:
        logger.info(f"  {fname}: {imp}")

    return model


def tune_threshold_on_validation(s1_df, s2_df, s3_df, gt, model,
                                  val_size=10000, max_candidates=30):
    """Tune the matching threshold on a held-out validation set."""
    logger.info(f"Tuning threshold on {val_size} validation entities...")

    # Sample validation entities
    s1_val = s1_df.sample(n=min(val_size, len(s1_df)), random_state=123)

    candidates = generate_all_candidates(
        s1_val, s2_df, s3_df, max_candidates=max_candidates
    )

    # Score all candidates
    scored = score_all_candidates(s1_val, s2_df, s3_df, candidates, model)

    # Sweep thresholds
    best_t, best_score = 0.5, -1.0
    for t in np.arange(0.20, 0.95, 0.05):
        predictions = {}
        for eid, pairs in scored.items():
            predictions[eid] = [cid for cid, p in pairs if p >= t]

        # Build GT subset for validation entities
        gt_val = {eid: gt.get(eid, set()) for eid in s1_val["entity_id"]}
        score = macro_f_beta(predictions, gt_val)
        logger.info(f"  threshold={t:.2f}: F0.5={score:.4f}")

        if score > best_score:
            best_score = score
            best_t = t

    logger.info(f"Best threshold: {best_t:.2f} (F0.5={best_score:.4f})")
    return best_t


def score_all_candidates(s1_df, s2_df, s3_df, candidates, model):
    """Score all candidate pairs using the trained model.

    Returns dict: source1_entity_id -> list of (candidate_id, probability).
    """
    lookup2 = s2_df.set_index("entity_id")
    lookup3 = s3_df.set_index("entity_id")
    s1_lookup = s1_df.set_index("entity_id")

    results = {}

    for eid in tqdm(s1_df["entity_id"], desc="Scoring candidates",
                    disable=len(s1_df) < 1000):
        cand_ids = candidates.get(eid, [])
        if not cand_ids:
            results[eid] = []
            continue

        name_a = s1_lookup.loc[eid, "business_name"]
        addr_a = s1_lookup.loc[eid, "business_address"]
        country_a = s1_lookup.loc[eid, "country"]

        feat_rows, valid_ids = [], []
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
            feat_rows.append(feats)
            valid_ids.append(cid)

        if not feat_rows:
            results[eid] = []
            continue

        X = pd.DataFrame(feat_rows, columns=FEATURE_NAMES)
        probs = model.predict_proba(X)[:, 1]
        results[eid] = list(zip(valid_ids, probs.tolist()))

    return results


def write_tsv(predictions, out_path, id_col, list_col):
    """Write predictions to a TSV file with deduplication."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write(f"{id_col}\t{list_col}\n")
        for eid, ids in predictions.items():
            seen, deduped = set(), []
            for i in (ids if isinstance(ids, list) else list(ids)):
                if i not in seen:
                    seen.add(i)
                    deduped.append(i)
            f.write(f"{eid}\t{','.join(deduped)}\n")


def run_pipeline(args):
    """Main pipeline execution."""
    start_time = time.time()
    project_dir = Path(args.project_dir)
    train_dir = project_dir / "dataset" / "train"
    test_dir = project_dir / "dataset" / "test"
    output_dir = project_dir / "output"
    model_path = project_dir / "model.pkl"

    # ====== PHASE 1: Load Training Data ======
    logger.info("=" * 60)
    logger.info("PHASE 1: Loading training data")
    logger.info("=" * 60)

    s1_train = load_source(_find_source_file(train_dir, "train", 1))
    s2_train = load_source(_find_source_file(train_dir, "train", 2))
    s3_train = load_source(_find_source_file(train_dir, "train", 3))
    gt = load_ground_truth(_find_gt_file(train_dir, "train"))

    logger.info(f"Train S1: {len(s1_train)}, S2: {len(s2_train)}, S3: {len(s3_train)}")
    logger.info(f"Ground truth: {len(gt)} entities")

    # ====== PHASE 2: Build Training Set & Train Model ======
    logger.info("=" * 60)
    logger.info("PHASE 2: Training model")
    logger.info("=" * 60)

    X, y = build_training_set_sampled(
        s1_train, s2_train, s3_train, gt,
        sample_size=args.sample_size,
        max_candidates=args.top_k,
        neg_ratio=args.neg_ratio,
    )

    model = train_model(X, y)

    # Save model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_path}")

    # ====== PHASE 3: Tune Threshold ======
    logger.info("=" * 60)
    logger.info("PHASE 3: Tuning threshold on validation set")
    logger.info("=" * 60)

    threshold = tune_threshold_on_validation(
        s1_train, s2_train, s3_train, gt, model,
        val_size=args.val_size,
        max_candidates=args.top_k,
    )

    # Free training data memory
    del s2_train, s3_train, gt
    gc.collect()

    # ====== PHASE 4: Inference on Test Data ======
    logger.info("=" * 60)
    logger.info("PHASE 4: Running inference on test data")
    logger.info("=" * 60)

    s1_test = load_source(test_dir / "test_source1.tsv")
    s2_test = load_source(test_dir / "test_source2.tsv")
    s3_test = load_source(test_dir / "test_source3.tsv")

    logger.info(f"Test S1: {len(s1_test)}, S2: {len(s2_test)}, S3: {len(s3_test)}")

    # Process test data in chunks to manage memory
    chunk_size = args.chunk_size
    n_chunks = (len(s1_test) + chunk_size - 1) // chunk_size

    all_candidates = {}
    all_predictions = {}

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(s1_test))
        s1_chunk = s1_test.iloc[start_idx:end_idx].reset_index(drop=True)

        logger.info(f"Processing test chunk {chunk_idx+1}/{n_chunks} "
                    f"(entities {start_idx+1}-{end_idx})")

        # Generate candidates for this chunk
        candidates = generate_all_candidates(
            s1_chunk, s2_test, s3_test, max_candidates=args.top_k
        )
        all_candidates.update(candidates)

        # Score candidates
        scored = score_all_candidates(
            s1_chunk, s2_test, s3_test, candidates, model
        )

        # Apply threshold
        for eid, pairs in scored.items():
            all_predictions[eid] = [cid for cid, p in pairs if p >= threshold]

        gc.collect()

    # Ensure every S1 test entity has an entry
    for eid in s1_test["entity_id"]:
        if eid not in all_predictions:
            all_predictions[eid] = []
        if eid not in all_candidates:
            all_candidates[eid] = []

    # ====== PHASE 5: Write Output Files ======
    logger.info("=" * 60)
    logger.info("PHASE 5: Writing output files")
    logger.info("=" * 60)

    write_tsv(all_candidates, output_dir / "candidate_pairs.tsv",
              "source1_entity_id", "candidate_entity_ids")
    write_tsv(all_predictions, output_dir / "matching_results.tsv",
              "source1_entity_id", "matched_entity_ids")

    # Stats
    matched = sum(1 for v in all_predictions.values() if v)
    total_matches = sum(len(v) for v in all_predictions.values())
    total_candidates = sum(len(v) for v in all_candidates.values())

    elapsed = time.time() - start_time
    logger.info(f"Done in {elapsed/60:.1f} minutes!")
    logger.info(f"Total S1 entities: {len(all_predictions)}")
    logger.info(f"Entities with matches: {matched}")
    logger.info(f"Total match pairs: {total_matches}")
    logger.info(f"Total candidate pairs: {total_candidates}")
    logger.info(f"Threshold used: {threshold:.2f}")
    logger.info(f"Output written to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Entity Resolution Pipeline — Amazon ML Challenge 2026"
    )
    parser.add_argument(
        "--project-dir", default=".",
        help="Root project directory (default: current dir)"
    )
    parser.add_argument(
        "--sample-size", type=int, default=50000,
        help="Number of S1 entities to sample for training (default: 50000)"
    )
    parser.add_argument(
        "--top-k", type=int, default=30,
        help="Max blocking candidates per source (default: 30)"
    )
    parser.add_argument(
        "--neg-ratio", type=int, default=5,
        help="Max negative examples per positive (default: 5)"
    )
    parser.add_argument(
        "--val-size", type=int, default=10000,
        help="Validation set size for threshold tuning (default: 10000)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=50000,
        help="Test data chunk size for memory management (default: 50000)"
    )
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()

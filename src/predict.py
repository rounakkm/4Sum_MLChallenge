"""High-performance streaming inference pipeline for Amazon ML Challenge 2026.

Generates both required deliverable files:
  1. output/matching_results.tsv   (final matches after thresholding -- scored on leaderboard)
  2. output/candidate_pairs.tsv    (candidate set from blocking stage)

Optimized for 1.73M+ test records:
  - Country-partitioned execution (France, US, India processed independently)
  - Pre-cached normalized records in FastBlockingIndex
  - Fast string pre-filter (token sort ratio >= 38 or street number match)
  - Zero-allocation feature matrix via direct float32 NumPy array writes
  - Streaming TSV writes with immediate flushing

Usage:
    python -m src.predict --data-dir dataset/test --model model.pkl --output-dir output
"""

import argparse
import gc
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

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


def run_country_inference(s1_df, s2_df, s3_df, model, threshold, cand_file, match_file,
                          max_candidates=20, batch_size=20000):
    """Run candidate retrieval, feature computation, and scoring for one country subset."""
    n_s1 = len(s1_df)
    if n_s1 == 0:
        return 0, 0

    logger.info(f"Indexing S2 ({len(s2_df):,} records) and S3 ({len(s3_df):,} records)...")
    idx2 = FastBlockingIndex(max_postings=2000)
    idx2.add_records(s2_df)

    idx3 = FastBlockingIndex(max_postings=2000)
    idx3.add_records(s3_df)

    total_candidates_written = 0
    total_matches_written = 0

    n_feats = len(FEATURE_NAMES)
    t0_country = time.time()

    # Pre-normalize S1 records
    logger.info(f"Pre-normalizing {n_s1:,} S1 records for country...")
    s1_eids = s1_df['entity_id'].values
    s1_names_raw = s1_df['business_name'].values
    s1_addrs_raw = s1_df['business_address'].values
    s1_cntry_raw = s1_df['country'].values

    s1_processed = []
    for i in range(n_s1):
        na = normalize_name(str(s1_names_raw[i] or ""))
        aa = normalize_address(str(s1_addrs_raw[i] or ""))
        ca = normalize_country(str(s1_cntry_raw[i] or ""))
        na_split = na.split()
        na_tok = set(na_split)
        aa_tok = set(aa.split())
        na_nums = set(get_address_numbers(aa))
        s1_processed.append((s1_eids[i], na, aa, ca, na_split, na_tok, aa_tok, na_nums))

    # Process S1 entities in batches
    for start_idx in range(0, n_s1, batch_size):
        t0_batch = time.time()
        end_idx = min(start_idx + batch_size, n_s1)
        batch_slice = s1_processed[start_idx:end_idx]

        batch_candidates = {}
        valid_pairs_meta = []  # (eid, cid)
        pair_feature_data = []

        # 1. Candidate retrieval and fast pre-filter
        for eid, na, aa, ca, na_split, na_tok, aa_tok, na_nums in batch_slice:
            c2 = idx2.query(na, aa, max_candidates=max_candidates, name_tokens=na_split, s1_nums=na_nums)
            c3 = idx3.query(na, aa, max_candidates=max_candidates, name_tokens=na_split, s1_nums=na_nums)

            # Deduplicate candidates while preserving order
            seen = set()
            cand_list = []
            for cid in c2 + c3:
                if cid not in seen:
                    seen.add(cid)
                    cand_list.append(cid)

            batch_candidates[eid] = cand_list

            # Pre-filter candidate pairs for full feature evaluation
            for cid in cand_list:
                rec_target = idx2.records.get(cid) if cid.startswith("S2-") else idx3.records.get(cid)
                if rec_target is not None:
                    nb, ab, nb_nums = rec_target
                    name_sim = fuzz.token_sort_ratio(na, nb)
                    num_match = bool(na_nums and nb_nums and na_nums.intersection(nb_nums))

                    if name_sim >= 35 or num_match:
                        nb_tok = set(nb.split())
                        ab_tok = set(ab.split())
                        pair_feature_data.append((na, aa, ca, nb, ab, ca, na_tok, nb_tok, aa_tok, ab_tok, na_nums, set(nb_nums)))
                        valid_pairs_meta.append((eid, cid))

        # 2. Extract features directly into pre-allocated NumPy array
        n_pairs = len(pair_feature_data)
        eid_matches = {item[0]: [] for item in batch_slice}

        if n_pairs > 0:
            X_batch = np.empty((n_pairs, n_feats), dtype=np.float32)
            for i in range(n_pairs):
                na, aa, ca, nb, ab, cb, na_tok, nb_tok, aa_tok, ab_tok, na_nums, nb_nums = pair_feature_data[i]
                extract_features_into_array(X_batch, i, na, aa, ca, nb, ab, cb, na_tok, nb_tok, aa_tok, ab_tok, na_nums, nb_nums)

            # Predict probabilities
            probs = model.predict_proba(X_batch)[:, 1]

            for (eid, cid), prob in zip(valid_pairs_meta, probs):
                if prob >= threshold:
                    eid_matches[eid].append(cid)

        # 3. Stream write results to disk
        for item in batch_slice:
            eid = item[0]
            cands = batch_candidates.get(eid, [])
            matches = eid_matches.get(eid, [])

            cand_file.write(f"{eid}\t{','.join(cands)}\n")
            match_file.write(f"{eid}\t{','.join(matches)}\n")

            total_candidates_written += len(cands)
            total_matches_written += len(matches)

        cand_file.flush()
        match_file.flush()

        dt_batch = time.time() - t0_batch
        logger.info(
            f"  Batch {end_idx:,}/{n_s1:,} ({100*end_idx/n_s1:.1f}%) | "
            f"{n_pairs:,} pairs scored in {dt_batch:.1f}s ({len(batch_slice)/dt_batch:.0f} entities/s) | "
            f"Matches: {total_matches_written:,}"
        )

    logger.info(f"Finished country in {time.time()-t0_country:.1f}s.")
    del idx2, idx3, s1_processed
    gc.collect()
    return total_candidates_written, total_matches_written


def run_inference(args):
    """Main inference execution."""
    t_start = time.time()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cand_path = output_dir / "candidate_pairs.tsv"
    match_path = output_dir / "matching_results.tsv"

    logger.info("=" * 65)
    logger.info("ENTITY RESOLUTION INFERENCE PIPELINE")
    logger.info("=" * 65)

    # Load model
    logger.info(f"Loading trained model from {args.model}...")
    with open(args.model, "rb") as f:
        model_obj = pickle.load(f)

    if isinstance(model_obj, dict):
        model = model_obj["model"]
        threshold = args.threshold if args.threshold is not None else model_obj.get("best_threshold", 0.85)
    else:
        model = model_obj
        threshold = args.threshold if args.threshold is not None else 0.85

    logger.info(f"Active matching probability threshold: {threshold:.3f}")

    # Load test data
    logger.info("Loading test data...")
    s1_df, s2_df, s3_df, _ = load_all(data_dir, split=args.split)
    logger.info(f"Test records loaded: S1={len(s1_df):,}, S2={len(s2_df):,}, S3={len(s3_df):,}")

    cand_exists = cand_path.exists() and cand_path.stat().st_size > 0
    match_exists = match_path.exists() and match_path.stat().st_size > 0

    seen_eids = set()
    if cand_exists and match_exists:
        logger.info("Checking existing output files for resume...")
        with open(match_path, "r", encoding="utf-8") as f:
            first_line = f.readline()
            for line in f:
                parts = line.split("\t")
                if parts and parts[0]:
                    seen_eids.add(parts[0])
        logger.info(f"Found {len(seen_eids):,} previously processed entities.")

    write_header = not (cand_exists and match_exists and len(seen_eids) > 0)
    file_mode = "a" if not write_header else "w"

    with open(cand_path, file_mode, encoding="utf-8") as f_cand, open(match_path, file_mode, encoding="utf-8") as f_match:
        if write_header:
            f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
            f_match.write("source1_entity_id\tmatched_entity_ids\n")

        s1_countries = s1_df['country'].fillna('').map(normalize_country)
        s2_countries = s2_df['country'].fillna('').map(normalize_country)
        s3_countries = s3_df['country'].fillna('').map(normalize_country)

        unique_countries = sorted(s1_countries.unique())
        total_cands = 0
        total_matches = 0

        for c in unique_countries:
            s1_sub = s1_df[s1_countries == c].reset_index(drop=True)
            s1_sub_eids = set(s1_sub['entity_id'])

            if seen_eids and s1_sub_eids.issubset(seen_eids):
                logger.info(f"Skipping already processed country: '{c.upper()}' ({len(s1_sub):,} entities).")
                continue

            logger.info("-" * 65)
            logger.info(f"PROCESSING COUNTRY: '{c.upper()}' ({len(s1_sub):,} S1 entities)")
            logger.info("-" * 65)

            s2_sub = s2_df[s2_countries == c].reset_index(drop=True)
            s3_sub = s3_df[s3_countries == c].reset_index(drop=True)

            c_cands, c_matches = run_country_inference(
                s1_sub, s2_sub, s3_sub, model, threshold, f_cand, f_match,
                max_candidates=args.max_candidates, batch_size=args.batch_size
            )
            total_cands += c_cands
            total_matches += c_matches
            del s2_sub, s3_sub
            gc.collect()

    elapsed = time.time() - t_start
    logger.info("=" * 65)
    logger.info(f"INFERENCE COMPLETE in {elapsed/60:.2f} minutes ({elapsed:.1f}s)!")
    logger.info(f"Total S1 entities evaluated: {len(s1_df):,}")
    logger.info(f"Output files generated:")
    logger.info(f"  - {cand_path}")
    logger.info(f"  - {match_path}")
    logger.info("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Run inference pipeline for Amazon ML Challenge 2026.")
    parser.add_argument("--data-dir", default="dataset/test", help="Path to test dataset directory.")
    parser.add_argument("--split", default="test", help="Split name ('test').")
    parser.add_argument("--model", default="model.pkl", help="Path to trained model pickle.")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold (defaults to model's best_threshold).")
    parser.add_argument("--max-candidates", type=int, default=20, help="Max candidates per source per entity.")
    parser.add_argument("--batch-size", type=int, default=20000, help="Inference batch size.")
    parser.add_argument("--output-dir", default="output", help="Directory to save submission TSVs.")
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()

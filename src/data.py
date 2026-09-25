"""Loading TSV source files and ground truth for the entity resolution challenge.

Handles both cleaned (6-column with landmark/zip) and raw (4-column) data files.
Automatically detects cleaned vs raw files and normalizes column schema.
"""
import pandas as pd
from pathlib import Path


def load_source(path):
    """Load one source file (source1/source2/source3) as a DataFrame of strings.

    Handles both cleaned files (with landmark/zip columns) and raw files.
    Always returns a DataFrame with at least: entity_id, business_name,
    business_address, country.
    """
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    # Ensure consistent column set
    required = ["entity_id", "business_name", "business_address", "country"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {path}")
    return df


def load_ground_truth(path):
    """Load ground truth into a dict: source1_entity_id -> set(matched ids)."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    gt = {}
    for _, row in df.iterrows():
        raw = row["matched_entity_ids"]
        ids = [i.strip() for i in raw.split(",") if i.strip()] if raw else []
        gt[row["source1_entity_id"]] = set(ids)
    return gt


def _find_source_file(data_dir, split, source_num):
    """Find the correct source file, preferring cleaned versions for training."""
    data_dir = Path(data_dir)
    # Try cleaned version first (for training data)
    cleaned_path = data_dir / f"{split}_source{source_num}_cleaned.tsv"
    if cleaned_path.exists():
        return cleaned_path
    # Fall back to raw version
    raw_path = data_dir / f"{split}_source{source_num}.tsv"
    if raw_path.exists():
        return raw_path
    raise FileNotFoundError(
        f"Could not find source{source_num} file in {data_dir}. "
        f"Tried: {cleaned_path}, {raw_path}"
    )


def _find_gt_file(data_dir, split):
    """Find the ground truth file, preferring cleaned versions."""
    data_dir = Path(data_dir)
    cleaned = data_dir / f"{split}_ground_truth_cleaned.tsv"
    if cleaned.exists():
        return cleaned
    raw = data_dir / f"{split}_ground_truth.tsv"
    if raw.exists():
        return raw
    raise FileNotFoundError(f"Could not find ground truth file in {data_dir}")


def load_all(data_dir, split="train"):
    """Load source1/2/3 (+ ground truth if split == 'train') for a given directory.

    Automatically finds cleaned or raw files.
    """
    data_dir = Path(data_dir)
    s1 = load_source(_find_source_file(data_dir, split, 1))
    s2 = load_source(_find_source_file(data_dir, split, 2))
    s3 = load_source(_find_source_file(data_dir, split, 3))
    gt = None
    if split == "train":
        gt = load_ground_truth(_find_gt_file(data_dir, split))
    return s1, s2, s3, gt

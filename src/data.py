"""Loading TSV source files and ground truth for the entity resolution challenge."""
import pandas as pd
from pathlib import Path


def load_source(path):
    """Load one source file (source1/source2/source3) as a DataFrame of strings."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return df


def load_ground_truth(path):
    """Load train_ground_truth.tsv into a dict: source1_entity_id -> set(matched ids)."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    gt = {}
    for _, row in df.iterrows():
        raw = row["matched_entity_ids"]
        ids = [i for i in raw.split(",") if i] if raw else []
        gt[row["source1_entity_id"]] = set(ids)
    return gt


def load_all(data_dir, split="train"):
    """Load source1/2/3 (+ ground truth if split == 'train') for a given directory.

    Expects files named like ``{split}_source1.tsv`` inside ``data_dir``,
    matching the naming convention in the problem statement
    (dataset/train/train_source1.tsv, dataset/test/test_source1.tsv, ...).
    """
    data_dir = Path(data_dir)
    s1 = load_source(data_dir / f"{split}_source1.tsv")
    s2 = load_source(data_dir / f"{split}_source2.tsv")
    s3 = load_source(data_dir / f"{split}_source3.tsv")
    gt = None
    if split == "train":
        gt = load_ground_truth(data_dir / f"{split}_ground_truth.tsv")
    return s1, s2, s3, gt

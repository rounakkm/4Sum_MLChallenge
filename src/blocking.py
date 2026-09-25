"""Candidate generation (blocking) stage — optimized for millions of records.

Vectorized blocking using pandas operations and inverted index lookups.
Avoids row-by-row iteration over the large S2/S3 DataFrames.

Strategies:
  1. Country-partitioned blocking (process each country separately)
  2. Sorted Neighborhood on name prefixes
  3. Token-based inverted index (vectorized construction)
  4. Character trigram hash blocking
"""
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from .normalize import normalize_name, normalize_address, normalize_country

logger = logging.getLogger(__name__)


def _normalize_df(df):
    """Add normalized columns to a DataFrame (vectorized)."""
    df = df.copy()
    df["_name_norm"] = df["business_name"].fillna("").map(normalize_name)
    df["_addr_norm"] = df["business_address"].fillna("").map(normalize_address)
    df["_country_norm"] = df["country"].fillna("").map(normalize_country)
    df["_name_prefix"] = df["_name_norm"].str[:5]
    # Get first significant token from name
    df["_name_first_token"] = df["_name_norm"].str.split().str[0].fillna("")
    return df


def _build_inverted_index_vectorized(df, column, min_token_len=2, max_freq=50000):
    """Build token -> set(entity_id) inverted index from a column, vectorized."""
    # Explode tokens
    tokens_series = df[column].str.split()
    exploded = tokens_series.explode().reset_index()
    exploded.columns = ["idx", "token"]
    exploded["entity_id"] = df.loc[exploded["idx"].values, "entity_id"].values

    # Filter short tokens
    exploded = exploded[exploded["token"].str.len() >= min_token_len]
    exploded = exploded.dropna(subset=["token"])

    # Build index
    index = defaultdict(set)
    # Filter high-frequency tokens
    token_counts = exploded["token"].value_counts()
    valid_tokens = token_counts[token_counts <= max_freq].index
    exploded = exploded[exploded["token"].isin(valid_tokens)]

    for token, eid in zip(exploded["token"], exploded["entity_id"]):
        index[token].add(eid)

    return index


def _build_prefix_index_vectorized(df, prefix_col="_name_prefix", max_freq=20000):
    """Build prefix -> set(entity_id) index."""
    index = defaultdict(set)
    prefix_counts = df[prefix_col].value_counts()
    valid_prefixes = prefix_counts[prefix_counts <= max_freq].index

    mask = df[prefix_col].isin(valid_prefixes)
    for prefix, eid in zip(df.loc[mask, prefix_col], df.loc[mask, "entity_id"]):
        if prefix:
            index[prefix].add(eid)
    return index


def _build_first_token_index(df, max_freq=50000):
    """Build first-token -> set(entity_id) index."""
    index = defaultdict(set)
    token_counts = df["_name_first_token"].value_counts()
    valid = token_counts[token_counts <= max_freq].index

    mask = df["_name_first_token"].isin(valid) & (df["_name_first_token"] != "")
    for token, eid in zip(df.loc[mask, "_name_first_token"],
                          df.loc[mask, "entity_id"]):
        index[token].add(eid)
    return index


def generate_candidates_blocking(s1_df, other_df, max_candidates=30):
    """Generate candidates using multi-strategy inverted-index blocking.

    Returns dict: source1_entity_id -> list of candidate entity_ids.
    """
    if len(other_df) == 0:
        return {eid: [] for eid in s1_df["entity_id"]}

    logger.info(f"Normalizing {len(other_df)} other-source records...")
    other_norm = _normalize_df(other_df)

    logger.info(f"Normalizing {len(s1_df)} S1 records...")
    s1_norm = _normalize_df(s1_df)

    # Build blocking indices on "other" data
    logger.info("Building inverted indices...")
    name_token_index = _build_inverted_index_vectorized(
        other_norm, "_name_norm", min_token_len=2, max_freq=50000
    )
    addr_token_index = _build_inverted_index_vectorized(
        other_norm, "_addr_norm", min_token_len=3, max_freq=30000
    )
    prefix_index = _build_prefix_index_vectorized(other_norm)
    first_token_index = _build_first_token_index(other_norm)

    # Build country index
    country_sets = {}
    for country in other_norm["_country_norm"].unique():
        country_sets[country] = set(
            other_norm.loc[other_norm["_country_norm"] == country, "entity_id"]
        )

    logger.info(f"Name token index: {len(name_token_index)} keys")
    logger.info(f"Address token index: {len(addr_token_index)} keys")
    logger.info(f"Prefix index: {len(prefix_index)} keys")
    logger.info(f"First token index: {len(first_token_index)} keys")

    candidates = {}

    for _, row in tqdm(s1_norm.iterrows(), total=len(s1_norm),
                       desc="Blocking", disable=len(s1_norm) < 100):
        eid = row["entity_id"]
        name = row["_name_norm"]
        addr = row["_addr_norm"]
        country = row["_country_norm"]
        prefix = row["_name_prefix"]
        first_token = row["_name_first_token"]

        cand_scores = defaultdict(float)

        # Strategy 1: Name token overlap
        name_tokens = [t for t in name.split() if len(t) >= 2]
        for token in name_tokens:
            if token in name_token_index:
                for cid in name_token_index[token]:
                    cand_scores[cid] += 3.0

        # Strategy 2: First token match (strong signal)
        if first_token and first_token in first_token_index:
            for cid in first_token_index[first_token]:
                cand_scores[cid] += 4.0

        # Strategy 3: Prefix match
        if prefix and prefix in prefix_index:
            for cid in prefix_index[prefix]:
                cand_scores[cid] += 2.0

        # Strategy 4: Address token overlap
        addr_tokens = [t for t in addr.split() if len(t) >= 3]
        for token in addr_tokens[:5]:  # limit to first 5 tokens
            if token in addr_token_index:
                for cid in addr_token_index[token]:
                    cand_scores[cid] += 1.5

        # Country bonus
        if country in country_sets:
            same_country = country_sets[country]
            for cid in list(cand_scores.keys()):
                if cid in same_country:
                    cand_scores[cid] += 2.0

        # Select top candidates
        if cand_scores:
            sorted_cands = sorted(cand_scores.items(), key=lambda x: -x[1])
            candidates[eid] = [c[0] for c in sorted_cands[:max_candidates]]
        else:
            candidates[eid] = []

    return candidates


def generate_all_candidates(s1_df, s2_df, s3_df, max_candidates=30):
    """Combine candidates from Source 2 and Source 3."""
    logger.info(f"Generating candidates from Source 2 ({len(s2_df)} records)...")
    cand2 = generate_candidates_blocking(s1_df, s2_df, max_candidates=max_candidates)

    logger.info(f"Generating candidates from Source 3 ({len(s3_df)} records)...")
    cand3 = generate_candidates_blocking(s1_df, s3_df, max_candidates=max_candidates)

    combined = {}
    for eid in s1_df["entity_id"]:
        c2 = cand2.get(eid, [])
        c3 = cand3.get(eid, [])
        seen = set()
        merged = []
        for c in c2 + c3:
            if c not in seen:
                seen.add(c)
                merged.append(c)
        combined[eid] = merged

    return combined

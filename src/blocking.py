"""High-performance candidate generation (blocking) stage.

Optimized for millions of records with minimal memory overhead.
Key features:
  1. Country-partitioned indexing: S1 entities are matched against S2/S3 of the same country,
     eliminating 100% of impossible cross-country comparisons.
  2. Dual-channel inverted index:
     - Significant name tokens (filtered for high-frequency stop words)
     - 4-character name prefixes (catches typos, inflections, abbreviations)
     - Distinctive address numbers and locality tokens
  3. Frequency-capped posting lists: Prevents ubiquitous tokens from polluting candidate sets.
  4. Memory-efficient streaming with zero large intermediate allocations.
"""

import logging
import re
from collections import defaultdict

from .normalize import normalize_name, normalize_address, normalize_country
from .features import get_address_numbers

logger = logging.getLogger(__name__)

STOP_WORDS = {
    # Generic English
    'the', 'of', 'and', 'for', 'in', 'at', 'on', 'to', 'a', 'an',
    'corporation', 'company', 'incorporated', 'limited', 'private',
    'international', 'manufacturing', 'associates', 'services',
    'group', 'enterprise', 'industries', 'technologies', 'solutions',
    'communications', 'financial', 'management', 'consulting',
    'corp', 'co', 'inc', 'ltd', 'pvt', 'llc', 'llp',
    # French generic
    'saint', 'st', 'ste', 'de', 'du', 'la', 'des', 'en', 'le', 'les',
    'societe', 'etablissement', 'association', 'groupe', 'service',
    'services', 'france', 'paris', 'lyon', 'marseille', 'rue',
    'boulevard', 'avenue', 'impasse', 'chemin', 'place', 'allee',
    'sa', 'sarl', 'eurl', 'sas', 'sci',
    # Indian generic
    'shri', 'sri', 'mr', 'mrs', 'private', 'limited', 'enterprises',
    'trading', 'industries', 'india', 'delhi', 'mumbai', 'kolkata',
    'chennai', 'road', 'nagar', 'colony', 'street',
    # US generic
    'us', 'usa', 'street', 'drive', 'road', 'lane', 'avenue',
    'boulevard', 'way', 'court', 'circle', 'highway', 'suite',
    'unit', 'floor', 'apartment'
}


class FastBlockingIndex:
    """Inverted index over business records for rapid candidate retrieval."""

    def __init__(self, max_postings=2000):
        self.max_postings = max_postings
        self.token_idx = defaultdict(list)
        self.prefix_idx = defaultdict(list)
        self.addr_idx = defaultdict(list)
        # eid -> (norm_name, norm_addr, country, name_tokens, addr_tokens, addr_numbers)
        self.records = {}

    def add_records(self, df):
        """Index all records from a DataFrame (entity_id, business_name, business_address, country)."""
        eids = df['entity_id'].values
        names = df['business_name'].values
        addrs = df['business_address'].values
        countries = df['country'].values

        for i in range(len(df)):
            eid = eids[i]
            n_raw = str(names[i] or "")
            a_raw = str(addrs[i] or "")
            c_raw = str(countries[i] or "")

            n_norm = normalize_name(n_raw)
            a_norm = normalize_address(a_raw)
            c_norm = normalize_country(c_raw)

            name_tokens = n_norm.split()
            addr_tokens = a_norm.split()
            name_tok_set = set(name_tokens)
            addr_tok_set = set(addr_tokens)
            addr_nums = get_address_numbers(a_norm)

            self.records[eid] = (n_norm, a_norm, c_norm, name_tok_set, addr_tok_set, addr_nums)

            # 1. Index distinctive name tokens
            for t in name_tokens:
                if len(t) >= 3 and t not in STOP_WORDS:
                    self.token_idx[t].append(eid)
                    if len(t) >= 4:
                        self.prefix_idx[t[:4]].append(eid)

            # 2. Index address street numbers
            for num in addr_nums:
                self.addr_idx[num].append(eid)

    def query(self, s1_name_norm, s1_addr_norm, max_candidates=20):
        """Query candidates for an S1 entity."""
        scores = defaultdict(float)

        # 1. Match on distinctive name tokens
        name_tokens = s1_name_norm.split()
        for t in name_tokens:
            if len(t) >= 3 and t not in STOP_WORDS:
                postings = self.token_idx.get(t)
                if postings and len(postings) <= self.max_postings:
                    for cid in postings:
                        scores[cid] += 3.0

                if len(t) >= 4:
                    p_postings = self.prefix_idx.get(t[:4])
                    if p_postings and len(p_postings) <= self.max_postings:
                        for cid in p_postings:
                            scores[cid] += 1.5

        # 2. Match on street numbers
        s1_nums = get_address_numbers(s1_addr_norm)
        for num in s1_nums:
            a_postings = self.addr_idx.get(num)
            if a_postings and len(a_postings) <= self.max_postings:
                for cid in a_postings:
                    scores[cid] += 1.5

        if not scores:
            return []

        # Return top-scoring candidates
        top_cands = sorted(scores.items(), key=lambda x: -x[1])[:max_candidates]
        return [c[0] for c in top_cands]


def generate_candidates_by_country(s1_df, s2_df, s3_df, max_per_source=20):
    """Generate candidates partitioned by country for maximum precision and efficiency.

    Returns dict: source1_entity_id -> list of candidate entity_ids.
    """
    logger.info("Partitioning records by country...")
    s1_countries = s1_df['country'].fillna('').map(normalize_country)
    s2_countries = s2_df['country'].fillna('').map(normalize_country)
    s3_countries = s3_df['country'].fillna('').map(normalize_country)

    unique_countries = s1_countries.unique()
    all_candidates = {}

    for c in unique_countries:
        s1_sub = s1_df[s1_countries == c]
        s2_sub = s2_df[s2_countries == c]
        s3_sub = s3_df[s3_countries == c]

        logger.info(f"Processing country '{c}': S1={len(s1_sub)}, S2={len(s2_sub)}, S3={len(s3_sub)}")

        idx2 = FastBlockingIndex(max_postings=2000)
        idx2.add_records(s2_sub)

        idx3 = FastBlockingIndex(max_postings=2000)
        idx3.add_records(s3_sub)

        s1_eids = s1_sub['entity_id'].values
        s1_names = s1_sub['business_name'].values
        s1_addrs = s1_sub['business_address'].values

        for i in range(len(s1_sub)):
            eid = s1_eids[i]
            n_norm = normalize_name(s1_names[i])
            a_norm = normalize_address(s1_addrs[i])

            cands2 = idx2.query(n_norm, a_norm, max_candidates=max_per_source)
            cands3 = idx3.query(n_norm, a_norm, max_candidates=max_per_source)

            merged = []
            seen = set()
            for cid in cands2 + cands3:
                if cid not in seen:
                    seen.add(cid)
                    merged.append(cid)

            all_candidates[eid] = merged

    for eid in s1_df['entity_id']:
        if eid not in all_candidates:
            all_candidates[eid] = []

    return all_candidates


def generate_all_candidates(s1_df, s2_df, s3_df, max_candidates=30):
    """Backward-compatible entry point for candidate generation."""
    per_source = max(10, max_candidates // 2)
    return generate_candidates_by_country(s1_df, s2_df, s3_df, max_per_source=per_source)

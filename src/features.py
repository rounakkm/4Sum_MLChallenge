"""Pairwise similarity features for the matching model.

Focused on signals that separate true matches from *near-miss* non-matches
(e.g. same chain/brand in two different cities) since the F0.5 metric
penalizes false merges 2x harder than missed matches.
"""
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from .normalize import normalize_name, normalize_address

FEATURE_NAMES = [
    "name_levenshtein_ratio",
    "name_token_sort_ratio",
    "name_jaro_winkler",
    "name_jaccard",
    "addr_levenshtein_ratio",
    "addr_token_sort_ratio",
    "addr_jaro_winkler",
    "addr_jaccard",
    "country_match",
    "name_len_diff",
    "addr_len_diff",
]


def _tokens(text):
    return set(text.split())


def _jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def pair_features(name_a, addr_a, country_a, name_b, addr_b, country_b):
    na, nb = normalize_name(name_a), normalize_name(name_b)
    aa, ab = normalize_address(addr_a), normalize_address(addr_b)

    return {
        "name_levenshtein_ratio": fuzz.ratio(na, nb) / 100.0,
        "name_token_sort_ratio": fuzz.token_sort_ratio(na, nb) / 100.0,
        "name_jaro_winkler": JaroWinkler.normalized_similarity(na, nb),
        "name_jaccard": _jaccard(_tokens(na), _tokens(nb)),
        "addr_levenshtein_ratio": fuzz.ratio(aa, ab) / 100.0,
        "addr_token_sort_ratio": fuzz.token_sort_ratio(aa, ab) / 100.0,
        "addr_jaro_winkler": JaroWinkler.normalized_similarity(aa, ab),
        "addr_jaccard": _jaccard(_tokens(aa), _tokens(ab)),
        "country_match": float(str(country_a).strip().lower() == str(country_b).strip().lower()),
        "name_len_diff": abs(len(na) - len(nb)),
        "addr_len_diff": abs(len(aa) - len(ab)),
    }

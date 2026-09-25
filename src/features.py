"""Pairwise similarity features for the matching model.

Rich feature set combining multiple string similarity metrics for
business names and addresses. Features are designed to separate true
matches from near-miss non-matches.

F0.5 metric penalizes false merges 2x harder than missed matches,
so features include several precision-oriented signals including
street/building number matching.
"""

import re
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from .normalize import normalize_name, normalize_address, normalize_country, _char_ngrams

FEATURE_NAMES = [
    # Name similarity features (0-6)
    "name_levenshtein_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_jaro_winkler",
    "name_jaccard",
    "name_partial_ratio",
    "name_ngram_overlap",
    # Address similarity features (7-12)
    "addr_levenshtein_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_jaro_winkler",
    "addr_jaccard",
    "addr_partial_ratio",
    # Country features (13)
    "country_match",
    # Length-based features (14-17)
    "name_len_diff",
    "addr_len_diff",
    "name_len_ratio",
    "addr_len_ratio",
    # Token overlap features (18-21)
    "name_token_overlap_count",
    "name_token_overlap_ratio",
    "addr_token_overlap_count",
    "addr_token_overlap_ratio",
    # Combined score (22)
    "name_addr_combined_score",
    # Street number verification (23)
    "addr_number_match",
]


def _tokens(text):
    return set(text.split()) if text else set()


def _jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _overlap_count(a, b):
    return len(a & b) if (a and b) else 0


def _overlap_ratio(a, b):
    if not a or not b:
        return 0.0
    min_len = min(len(a), len(b))
    return len(a & b) / min_len if min_len else 0.0


def _len_ratio(a, b):
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def _ngram_overlap(a, b, n=3):
    ng_a = _char_ngrams(a, n)
    ng_b = _char_ngrams(b, n)
    if not ng_a and not ng_b:
        return 1.0
    if not ng_a or not ng_b:
        return 0.0
    return len(ng_a & ng_b) / len(ng_a | ng_b)


def get_address_numbers(addr):
    """Extract numeric tokens from address."""
    return set(re.findall(r"\b\d+\b", addr)) if addr else set()


def pair_features_fast(na, aa, ca, nb, ab, cb, na_tokens, nb_tokens, aa_tokens, ab_tokens, na_nums=None, nb_nums=None):
    """Ultra-fast feature extraction using pre-normalized strings and token sets."""
    name_lev = fuzz.ratio(na, nb) / 100.0
    addr_lev = fuzz.ratio(aa, ab) / 100.0

    if na_nums is None:
        na_nums = get_address_numbers(aa)
    if nb_nums is None:
        nb_nums = get_address_numbers(ab)

    if na_nums and nb_nums:
        num_match = 1.0 if (na_nums & nb_nums) else 0.0
    else:
        num_match = 0.5

    return {
        "name_levenshtein_ratio": name_lev,
        "name_token_sort_ratio": fuzz.token_sort_ratio(na, nb) / 100.0,
        "name_token_set_ratio": fuzz.token_set_ratio(na, nb) / 100.0,
        "name_jaro_winkler": JaroWinkler.normalized_similarity(na, nb),
        "name_jaccard": _jaccard(na_tokens, nb_tokens),
        "name_partial_ratio": fuzz.partial_ratio(na, nb) / 100.0,
        "name_ngram_overlap": _ngram_overlap(na, nb, 3),
        "addr_levenshtein_ratio": addr_lev,
        "addr_token_sort_ratio": fuzz.token_sort_ratio(aa, ab) / 100.0,
        "addr_token_set_ratio": fuzz.token_set_ratio(aa, ab) / 100.0,
        "addr_jaro_winkler": JaroWinkler.normalized_similarity(aa, ab),
        "addr_jaccard": _jaccard(aa_tokens, ab_tokens),
        "addr_partial_ratio": fuzz.partial_ratio(aa, ab) / 100.0,
        "country_match": 1.0 if ca == cb else 0.0,
        "name_len_diff": abs(len(na) - len(nb)),
        "addr_len_diff": abs(len(aa) - len(ab)),
        "name_len_ratio": _len_ratio(na, nb),
        "addr_len_ratio": _len_ratio(aa, ab),
        "name_token_overlap_count": _overlap_count(na_tokens, nb_tokens),
        "name_token_overlap_ratio": _overlap_ratio(na_tokens, nb_tokens),
        "addr_token_overlap_count": _overlap_count(aa_tokens, ab_tokens),
        "addr_token_overlap_ratio": _overlap_ratio(aa_tokens, ab_tokens),
        "name_addr_combined_score": 0.6 * name_lev + 0.4 * addr_lev,
        "addr_number_match": num_match,
    }


def extract_features_into_array(arr, row_idx, na, aa, ca, nb, ab, cb, na_tokens, nb_tokens, aa_tokens, ab_tokens, na_nums, nb_nums):
    """Writes features directly into row_idx of pre-allocated float32 NumPy array for zero allocation overhead."""
    name_lev = fuzz.ratio(na, nb) / 100.0
    addr_lev = fuzz.ratio(aa, ab) / 100.0

    arr[row_idx, 0] = name_lev
    arr[row_idx, 1] = fuzz.token_sort_ratio(na, nb) / 100.0
    arr[row_idx, 2] = fuzz.token_set_ratio(na, nb) / 100.0
    arr[row_idx, 3] = JaroWinkler.normalized_similarity(na, nb)
    arr[row_idx, 4] = _jaccard(na_tokens, nb_tokens)
    arr[row_idx, 5] = fuzz.partial_ratio(na, nb) / 100.0
    arr[row_idx, 6] = _ngram_overlap(na, nb, 3)
    arr[row_idx, 7] = addr_lev
    arr[row_idx, 8] = fuzz.token_sort_ratio(aa, ab) / 100.0
    arr[row_idx, 9] = fuzz.token_set_ratio(aa, ab) / 100.0
    arr[row_idx, 10] = JaroWinkler.normalized_similarity(aa, ab)
    arr[row_idx, 11] = _jaccard(aa_tokens, ab_tokens)
    arr[row_idx, 12] = fuzz.partial_ratio(aa, ab) / 100.0
    arr[row_idx, 13] = 1.0 if ca == cb else 0.0
    arr[row_idx, 14] = abs(len(na) - len(nb))
    arr[row_idx, 15] = abs(len(aa) - len(ab))
    arr[row_idx, 16] = _len_ratio(na, nb)
    arr[row_idx, 17] = _len_ratio(aa, ab)
    arr[row_idx, 18] = _overlap_count(na_tokens, nb_tokens)
    arr[row_idx, 19] = _overlap_ratio(na_tokens, nb_tokens)
    arr[row_idx, 20] = _overlap_count(aa_tokens, ab_tokens)
    arr[row_idx, 21] = _overlap_ratio(aa_tokens, ab_tokens)
    arr[row_idx, 22] = 0.6 * name_lev + 0.4 * addr_lev

    if na_nums and nb_nums:
        arr[row_idx, 23] = 1.0 if (na_nums & nb_nums) else 0.0
    else:
        arr[row_idx, 23] = 0.5


def pair_features(name_a, addr_a, country_a, name_b, addr_b, country_b):
    """Backward-compatible pair_features returning dict."""
    na, nb = normalize_name(name_a), normalize_name(name_b)
    aa, ab = normalize_address(addr_a), normalize_address(addr_b)
    ca, cb = normalize_country(country_a), normalize_country(country_b)
    return pair_features_fast(na, aa, ca, nb, ab, cb, _tokens(na), _tokens(nb), _tokens(aa), _tokens(ab))

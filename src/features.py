"""Pairwise similarity features for the matching model.

Rich feature set combining multiple string similarity metrics for
business names and addresses. Features are designed to separate true
matches from near-miss non-matches.

F0.5 metric penalizes false merges 2x harder than missed matches,
so features include several precision-oriented signals.
"""
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from .normalize import normalize_name, normalize_address, normalize_country, _char_ngrams

FEATURE_NAMES = [
    # Name similarity features
    "name_levenshtein_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_jaro_winkler",
    "name_jaccard",
    "name_partial_ratio",
    "name_ngram_overlap",
    # Address similarity features
    "addr_levenshtein_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_jaro_winkler",
    "addr_jaccard",
    "addr_partial_ratio",
    # Country features
    "country_match",
    # Length-based features
    "name_len_diff",
    "addr_len_diff",
    "name_len_ratio",
    "addr_len_ratio",
    # Token overlap features
    "name_token_overlap_count",
    "name_token_overlap_ratio",
    "addr_token_overlap_count",
    "addr_token_overlap_ratio",
    # Combined features
    "name_addr_combined_score",
]


def _tokens(text):
    """Get token set from text."""
    return set(text.split()) if text else set()


def _jaccard(a, b):
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _overlap_count(a, b):
    """Number of shared tokens."""
    if not a or not b:
        return 0
    return len(a & b)


def _overlap_ratio(a, b):
    """Ratio of shared tokens to min of set sizes."""
    if not a or not b:
        return 0.0
    min_len = min(len(a), len(b))
    if min_len == 0:
        return 0.0
    return len(a & b) / min_len


def _len_ratio(a, b):
    """Length ratio (0 to 1, 1 = same length)."""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def _ngram_overlap(a, b, n=3):
    """Character n-gram overlap ratio."""
    ng_a = _char_ngrams(a, n)
    ng_b = _char_ngrams(b, n)
    if not ng_a and not ng_b:
        return 1.0
    if not ng_a or not ng_b:
        return 0.0
    return len(ng_a & ng_b) / len(ng_a | ng_b)


def pair_features(name_a, addr_a, country_a, name_b, addr_b, country_b):
    """Compute pairwise similarity features for a candidate pair.

    Returns a dict with feature names as keys and float values.
    """
    na, nb = normalize_name(name_a), normalize_name(name_b)
    aa, ab = normalize_address(addr_a), normalize_address(addr_b)

    na_tokens = _tokens(na)
    nb_tokens = _tokens(nb)
    aa_tokens = _tokens(aa)
    ab_tokens = _tokens(ab)

    name_lev = fuzz.ratio(na, nb) / 100.0
    addr_lev = fuzz.ratio(aa, ab) / 100.0

    return {
        # Name features
        "name_levenshtein_ratio": name_lev,
        "name_token_sort_ratio": fuzz.token_sort_ratio(na, nb) / 100.0,
        "name_token_set_ratio": fuzz.token_set_ratio(na, nb) / 100.0,
        "name_jaro_winkler": JaroWinkler.normalized_similarity(na, nb),
        "name_jaccard": _jaccard(na_tokens, nb_tokens),
        "name_partial_ratio": fuzz.partial_ratio(na, nb) / 100.0,
        "name_ngram_overlap": _ngram_overlap(na, nb, 3),
        # Address features
        "addr_levenshtein_ratio": addr_lev,
        "addr_token_sort_ratio": fuzz.token_sort_ratio(aa, ab) / 100.0,
        "addr_token_set_ratio": fuzz.token_set_ratio(aa, ab) / 100.0,
        "addr_jaro_winkler": JaroWinkler.normalized_similarity(aa, ab),
        "addr_jaccard": _jaccard(aa_tokens, ab_tokens),
        "addr_partial_ratio": fuzz.partial_ratio(aa, ab) / 100.0,
        # Country feature
        "country_match": float(
            normalize_country(country_a) == normalize_country(country_b)
        ),
        # Length features
        "name_len_diff": abs(len(na) - len(nb)),
        "addr_len_diff": abs(len(aa) - len(ab)),
        "name_len_ratio": _len_ratio(na, nb),
        "addr_len_ratio": _len_ratio(aa, ab),
        # Token overlap features
        "name_token_overlap_count": _overlap_count(na_tokens, nb_tokens),
        "name_token_overlap_ratio": _overlap_ratio(na_tokens, nb_tokens),
        "addr_token_overlap_count": _overlap_count(aa_tokens, ab_tokens),
        "addr_token_overlap_ratio": _overlap_ratio(aa_tokens, ab_tokens),
        # Combined score
        "name_addr_combined_score": 0.6 * name_lev + 0.4 * addr_lev,
    }

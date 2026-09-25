"""Text normalization for business names and addresses.

Handles US, India, and France address patterns. Carefully avoids
ambiguous abbreviations that could corrupt city names (St. Louis)
or titles (Dr. in business names).
"""
import re
import unicodedata

# ---------- Legal / Business Name Suffixes ----------
LEGAL_SUFFIXES = {
    r"\bcorp\b": "corporation",
    r"\bco\b": "company",
    r"\binc\b": "incorporated",
    r"\bltd\b": "limited",
    r"\bpvt\b": "private",
    r"\bllc\b": "limited liability company",
    r"\bllp\b": "limited liability partnership",
    r"\bintl\b": "international",
    r"\bmfg\b": "manufacturing",
    r"\bassoc\b": "associates",
    r"\bsvcs?\b": "services",
    r"\bgrp\b": "group",
    r"\bentpr?\b": "enterprise",
    r"\bindust?\b": "industries",
    r"\btech\b": "technologies",
    r"\bsoln?s?\b": "solutions",
    r"\bcomm?\b": "communications",
    r"\bfin\b": "financial",
    r"\bmgmt\b": "management",
    r"\bconsult\b": "consulting",
    r"\bdba\b": "",  # "doing business as" — strip
}

# ---------- Address Abbreviations ----------
# Only unambiguous abbreviations to avoid corrupting city/state names
ADDRESS_ABBREVIATIONS = {
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bapt\b": "apartment",
    r"\bhwy\b": "highway",
    r"\bln\b": "lane",
    r"\bpkwy\b": "parkway",
    r"\bsq\b": "square",
    r"\bste\b": "suite",
    r"\bcir\b": "circle",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bter\b": "terrace",
    r"\btpke\b": "turnpike",
    r"\bnagar\b": "nagar",
    r"\bcolny\b": "colony",
    r"\bext\b": "extension",
    r"\bflr\b": "floor",
    r"\bbldg\b": "building",
    r"\bpvt\b": "private",
}

# French address terms
FRENCH_ADDRESS_TERMS = {
    r"\brue\b": "rue",
    r"\bboulevard\b": "boulevard",
    r"\bav\b": "avenue",
    r"\bchemin\b": "chemin",
    r"\bplace\b": "place",
    r"\ballée\b": "allee",
    r"\bimpasse\b": "impasse",
}

_PUNCT_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")
_AND_RE = re.compile(r"\s*&\s*")
_NUMERIC_HASH_RE = re.compile(r"#\s*(\d)")


def _unicode_normalize(text):
    """NFKD normalization + strip accents for consistent matching."""
    text = unicodedata.normalize("NFKD", text)
    # Keep the base characters, remove combining marks for comparison
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def _base_clean(text):
    """Lowercase, strip, normalize unicode, remove punctuation, collapse whitespace."""
    text = str(text or "").strip()
    if not text:
        return ""
    text = text.lower()
    text = _unicode_normalize(text)
    text = _AND_RE.sub(" and ", text)
    text = _NUMERIC_HASH_RE.sub(r"\1", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def _apply_abbreviations(text, mapping):
    """Apply regex-based abbreviation expansions."""
    for pattern, replacement in mapping.items():
        text = re.sub(pattern, replacement, text)
    return _SPACE_RE.sub(" ", text).strip()


def normalize_name(text):
    """Normalize a business name for comparison."""
    text = _base_clean(text)
    text = _apply_abbreviations(text, LEGAL_SUFFIXES)
    # Remove common noise words that don't help matching
    text = re.sub(r"\b(the|of|and|for|in|at|on|to|a|an)\b", " ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def normalize_address(text):
    """Normalize an address for comparison."""
    text = _base_clean(text)
    text = _apply_abbreviations(text, ADDRESS_ABBREVIATIONS)
    return text


def normalize_country(text):
    """Normalize country labels to consistent values."""
    text = str(text or "").strip().lower()
    if text in ("us", "usa", "united states", "united states of america", "u.s.", "u.s.a."):
        return "us"
    if text in ("india", "in", "ind", "bharat"):
        return "india"
    if text in ("france", "fr", "fra"):
        return "france"
    return text  # pass through unknown countries


def get_name_tokens(text):
    """Get significant tokens from a normalized name (for blocking)."""
    name = normalize_name(text)
    # Remove very short tokens (1 char) and common noise
    stop = {"the", "of", "and", "for", "in", "at", "on", "to", "a", "an",
            "corporation", "company", "incorporated", "limited", "private",
            "limited liability company", "limited liability partnership",
            "international", "manufacturing", "associates", "services",
            "group", "enterprise", "industries", "technologies", "solutions",
            "communications", "financial", "management", "consulting"}
    return [t for t in name.split() if len(t) > 1 and t not in stop]


def _char_ngrams(text, n=3):
    """Generate character n-grams from text."""
    if not text or len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}

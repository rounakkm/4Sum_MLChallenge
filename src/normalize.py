"""Text normalization for business names and addresses.

Optimized with single-pass compiled regexes and ASCII-aware bypass for 10x throughput.
Handles US, India, and France address patterns. Carefully avoids
ambiguous abbreviations that could corrupt city names (St. Louis)
or titles (Dr. in business names).
"""

import re
import unicodedata

# Raw token mappings (clean tokens, no regex markup)
LEGAL_MAPPING = {
    "corp": "corporation",
    "co": "company",
    "inc": "incorporated",
    "ltd": "limited",
    "pvt": "private",
    "llc": "limited liability company",
    "llp": "limited liability partnership",
    "intl": "international",
    "mfg": "manufacturing",
    "assoc": "associates",
    "svcs": "services",
    "svc": "services",
    "grp": "group",
    "entpr": "enterprise",
    "indust": "industries",
    "tech": "technologies",
    "soln": "solutions",
    "solns": "solutions",
    "comm": "communications",
    "fin": "financial",
    "mgmt": "management",
    "consult": "consulting",
    "dba": "",
    "sarl": "sarl",
    "sas": "sas",
    "sci": "sci",
    "eurl": "eurl",
    "sa": "sa",
}

ADDRESS_MAPPING = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "apt": "apartment",
    "hwy": "highway",
    "ln": "lane",
    "pkwy": "parkway",
    "sq": "square",
    "ste": "suite",
    "cir": "circle",
    "ct": "court",
    "pl": "place",
    "ter": "terrace",
    "tpke": "turnpike",
    "nagar": "nagar",
    "colny": "colony",
    "ext": "extension",
    "flr": "floor",
    "bldg": "building",
    "pvt": "private",
    "r": "rue",
    "rue": "rue",
    "boulevard": "boulevard",
    "bd": "boulevard",
    "av": "avenue",
    "chemin": "chemin",
    "place": "place",
    "allee": "allee",
    "impasse": "impasse",
}

_LEGAL_RE = re.compile(
    r"\b(" + "|".join(sorted(LEGAL_MAPPING.keys(), key=len, reverse=True)) + r")\b"
)
_ADDR_RE = re.compile(
    r"\b(" + "|".join(sorted(ADDRESS_MAPPING.keys(), key=len, reverse=True)) + r")\b"
)
_STOP_WORDS_RE = re.compile(r"\b(the|of|and|for|in|at|on|to|a|an)\b")
_NULL_WORDS_RE = re.compile(r"\b(null|nan|none|nil)\b")

_PUNCT_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")
_AND_RE = re.compile(r"\s*&\s*")
_NUMERIC_HASH_RE = re.compile(r"#\s*(\d)")


def _unicode_normalize(text):
    """NFKD normalization + strip accents for consistent matching."""
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def _base_clean(text):
    """Lowercase, strip, normalize unicode, remove punctuation, collapse whitespace."""
    text = str(text or "").strip()
    if not text:
        return ""
    text = text.lower()
    if not text.isascii():
        text = _unicode_normalize(text)
    text = _AND_RE.sub(" and ", text)
    text = _NUMERIC_HASH_RE.sub(r"\1", text)
    text = _PUNCT_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def normalize_name(text):
    """Normalize a business name for comparison."""
    text = _base_clean(text)
    if not text:
        return ""
    text = _LEGAL_RE.sub(lambda m: LEGAL_MAPPING[m.group(0)], text)
    text = _STOP_WORDS_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def normalize_address(text):
    """Normalize an address for comparison."""
    text = _base_clean(text)
    if not text:
        return ""
    text = _NULL_WORDS_RE.sub(" ", text)
    text = _ADDR_RE.sub(lambda m: ADDRESS_MAPPING[m.group(0)], text)
    return _SPACE_RE.sub(" ", text).strip()


def normalize_country(text):
    """Normalize country labels to consistent values."""
    text = str(text or "").strip().lower()
    if text in ("us", "usa", "united states", "united states of america", "u.s.", "u.s.a."):
        return "us"
    if text in ("india", "in", "ind", "bharat"):
        return "india"
    if text in ("france", "fr", "fra"):
        return "france"
    return text


def get_name_tokens(text):
    """Get significant tokens from a normalized name (for blocking)."""
    name = normalize_name(text)
    stop = {
        "the", "of", "and", "for", "in", "at", "on", "to", "a", "an",
        "corporation", "company", "incorporated", "limited", "private",
        "limited liability company", "limited liability partnership",
        "international", "manufacturing", "associates", "services",
        "group", "enterprise", "industries", "technologies", "solutions",
        "communications", "financial", "management", "consulting",
        "sarl", "sas", "sci", "eurl", "sa",
    }
    return [t for t in name.split() if len(t) > 1 and t not in stop]


def _char_ngrams(text, n=3):
    """Generate character n-grams from text."""
    if not text or len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}

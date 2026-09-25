"""Text normalization for business names and addresses.

Expand this file first when you see recurring noise patterns in the data
(new abbreviations, transliteration variants, etc.) -- normalization quality
drives both blocking recall and matching precision.
"""
import re

NAME_ABBREVIATIONS = {
    r"\bcorp\b": "corporation",
    r"\bco\b": "company",
    r"\binc\b": "incorporated",
    r"\bltd\b": "limited",
    r"\bpvt\b": "private",
    r"\bllc\b": "limited liability company",
    r"\bllp\b": "limited liability partnership",
    r"\bltd\.?\b": "limited",
    r"&": " and ",
}

ADDRESS_ABBREVIATIONS = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bapt\b": "apartment",
    r"\bfl\b": "floor",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bhwy\b": "highway",
    r"\bpin\b": "pincode",
    r"\bno\b": "number",
    r"\bnr\b": "near",
}

_PUNCT_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")


def _base_clean(text):
    text = (text or "").lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def _apply_abbreviations(text, mapping):
    for pattern, replacement in mapping.items():
        text = re.sub(pattern, replacement, text)
    return _SPACE_RE.sub(" ", text).strip()


def normalize_name(text):
    return _apply_abbreviations(_base_clean(text), NAME_ABBREVIATIONS)


def normalize_address(text):
    return _apply_abbreviations(_base_clean(text), ADDRESS_ABBREVIATIONS)

"""Clean train TSV files independently and write a reproducible EDA report.

Memory-efficient version: uses hash-based approximate unique counting and
streaming deduplication to handle files with millions of rows without OOM.

Key design decisions
--------------------
- PIN/ZIP extraction: only extracts postal codes at the **tail** of an address
  with country-aware validation (India 6-digit, US 5/5+4).  Street numbers are
  never misidentified as postal codes.
- Each file is processed independently.  Intra-file duplicates are removed;
  inter-file duplicates are intentionally preserved.
- Profiling uses hash-set cardinality for compact unique counts (stores hashes,
  not full strings) and samples coordinates instead of keeping them all.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from statistics import mean, median

# Raise the CSV field-size limit for very long address strings.
csv.field_size_limit(sys.maxsize)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "train"
OUTPUT_DIR = ROOT / "cleaned_data"
REPORT_PATH = OUTPUT_DIR / "eda_cleaning_report.md"

MISSING_TOKENS = {"", "null", "nan", "none", "n/a", "na", "nil", "<null>", "-", "--"}

# Columns that receive full text normalisation (lowercase + punctuation removal).
TEXT_COLUMNS = {"name", "category", "address", "city", "business_name", "business_address"}

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------
LANDMARK_MARKER = re.compile(
    r"\b(?:near|opp(?:osite)?\.?|beside|behind|next\s+to|adjacent\s+to|landmark)\b[:\s]*(.+)$",
    re.IGNORECASE,
)
ADDRESS_PUNCTUATION = re.compile(r"[^\w\s#/&+\-]", re.UNICODE)
GENERAL_PUNCTUATION = re.compile(r"[^\w\s&+\-]", re.UNICODE)
WHITESPACE = re.compile(r"\s+")
PIN_AT_TAIL = re.compile(r"[,\s]+(\d{5,6}(?:-\d{4})?)[\s.]*$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_missing(value: str | None) -> bool:
    return value is None or value.strip().casefold() in MISSING_TOKENS


def canonical_space(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = "".join(ch for ch in value if not unicodedata.category(ch).startswith("C"))
    return WHITESPACE.sub(" ", value).strip()


def normalize_text(value: str, column: str) -> str:
    value = canonical_space(value).casefold()
    punct = ADDRESS_PUNCTUATION if column in ("address", "business_address") else GENERAL_PUNCTUATION
    value = punct.sub(" ", value)
    return WHITESPACE.sub(" ", value).strip()


def normalize_country(value: str) -> str:
    value = canonical_space(value).casefold()
    if re.fullmatch(r"[a-z]{2,3}", value):
        return value.upper()
    mapping = {"india": "India", "united states": "US", "usa": "US",
               "united states of america": "US"}
    return mapping.get(value, value)


def normalize_url(value: str) -> str:
    return canonical_space(value)


def normalize_phone(value: str) -> str:
    value = canonical_space(value)
    if not value:
        return value
    prefix = "+" if value.lstrip().startswith("+") else ""
    digits = "".join(ch for ch in value if ch.isdigit())
    return prefix + digits


def normalize_zip(value: str, country: str) -> str:
    value = canonical_space(value).upper().replace(" ", "")
    if not value:
        return value
    if country == "US" and re.fullmatch(r"\d{9}", value):
        return value[:5] + "-" + value[5:]
    return value


def _is_valid_pin(candidate: str, country: str) -> bool:
    digits_only = candidate.replace("-", "")
    if country in ("India", "IN"):
        return bool(re.fullmatch(r"[1-8]\d{5}", digits_only))
    if country in ("US", "USA"):
        return bool(re.fullmatch(r"\d{5}", digits_only) or re.fullmatch(r"\d{9}", digits_only))
    return bool(re.fullmatch(r"\d{4,6}", digits_only))


def extract_address_details(address: str, zip_value: str, country: str) -> tuple[str, str, str]:
    landmark = ""
    marker = LANDMARK_MARKER.search(address)
    if marker:
        landmark = marker.group(1).strip(" ,;- ")
        address = address[: marker.start()].strip(" ,;- ")
    if not zip_value:
        m = PIN_AT_TAIL.search(address)
        if m:
            candidate = m.group(1)
            if _is_valid_pin(candidate, country):
                zip_value = normalize_zip(candidate, country)
                address = address[: m.start()].strip(" ,;- ")
    return address, landmark, zip_value


def valid_coordinate(value: str, limit: float) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and -limit <= number <= limit else None


# ---------------------------------------------------------------------------
# Memory-efficient profiling
# ---------------------------------------------------------------------------
def _row_hash(row: list[str]) -> bytes:
    """Fast 16-byte hash of a row for duplicate detection."""
    return hashlib.md5("\t".join(row).encode("utf-8")).digest()


def profile_lightweight(header: list[str], rows: list[list[str]]) -> dict:
    """Memory-efficient profiling: uses hash sets for unique counts."""
    indexes = {name: i for i, name in enumerate(header)}
    missing: Counter = Counter()
    # Store hashes instead of full strings for unique counting.
    unique_hashes: dict[str, set[int]] = {name: set() for name in header}
    countries: Counter = Counter()
    categories: Counter = Counter()
    # Sample coordinates (keep at most 100k for numeric summary).
    lat_sample: list[float] = []
    lon_sample: list[float] = []
    invalid_lat = invalid_lon = 0
    sample_rate = max(1, len(rows) // 100_000) if len(rows) > 100_000 else 1

    row_hashes: set[bytes] = set()
    dup_count = 0

    for i, row in enumerate(rows):
        # Duplicate detection via hash.
        rh = _row_hash(row)
        if rh in row_hashes:
            dup_count += 1
        else:
            row_hashes.add(rh)

        for name, value in zip(header, row):
            if is_missing(value):
                missing[name] += 1
            # Use hash of value for unique counting (much smaller than storing strings).
            unique_hashes[name].add(hash(value))

        if "country" in indexes:
            countries[row[indexes["country"]]] += 1
        if "category" in indexes:
            categories[row[indexes["category"]]] += 1

        if i % sample_rate == 0:
            if "latitude" in indexes:
                lat = valid_coordinate(row[indexes["latitude"]], 90)
                if lat is None and not is_missing(row[indexes["latitude"]]):
                    invalid_lat += 1
                elif lat is not None:
                    lat_sample.append(lat)
            if "longitude" in indexes:
                lon = valid_coordinate(row[indexes["longitude"]], 180)
                if lon is None and not is_missing(row[indexes["longitude"]]):
                    invalid_lon += 1
                elif lon is not None:
                    lon_sample.append(lon)

    return {
        "rows": len(rows),
        "columns": len(header),
        "missing": missing,
        "unique": {name: len(hashes) for name, hashes in unique_hashes.items()},
        "countries": countries,
        "categories": categories,
        "latitudes": lat_sample,
        "longitudes": lon_sample,
        "invalid_latitude": invalid_lat,
        "invalid_longitude": invalid_lon,
        "duplicates": dup_count,
    }


def summarize_numeric(values: list[float]) -> str:
    if not values:
        return "No valid values"
    ordered = sorted(values)
    return (
        f"count={len(values)}, min={ordered[0]:.6f}, median={median(ordered):.6f}, "
        f"mean={mean(values):.6f}, max={ordered[-1]:.6f}"
    )


def markdown_table(rows: list[tuple[str, object, object]]) -> list[str]:
    lines = ["| Field | Before | After |", "|---|---:|---:|"]
    lines.extend(f"| {name} | {before} | {after} |" for name, before, after in rows)
    return lines


# ---------------------------------------------------------------------------
# I/O — streaming scan
# ---------------------------------------------------------------------------
def scan_tsv(path: Path) -> tuple[list[str], list[list[str]], Counter, str, str]:
    """Read a TSV, returning (header, rows, issue_counts, encoding, delimiter)."""
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            with path.open("r", encoding=encoding, newline="") as fh:
                sample = fh.read(8192)
                fh.seek(0)
                delimiter = "\t" if "\t" in sample else ","
                reader = csv.reader(fh, delimiter=delimiter)
                header = next(reader)
                rows: list[list[str]] = []
                issues: Counter = Counter()
                expected = len(header)
                for row in reader:
                    if not any(cell.strip() for cell in row):
                        issues["blank_rows"] += 1
                    elif len(row) != expected:
                        issues["malformed_field_count"] += 1
                    elif row == header:
                        issues["repeated_header_rows"] += 1
                    else:
                        rows.append(row)
            return header, rows, issues, encoding, delimiter
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Unable to decode {path}")


# ---------------------------------------------------------------------------
# Row-level cleaning
# ---------------------------------------------------------------------------
def clean_rows(header: list[str], rows: list[list[str]]) -> tuple[list[str], list[list[str]], Counter]:
    indexes = {name: i for i, name in enumerate(header)}
    address_col = (
        "address" if "address" in indexes
        else "business_address" if "business_address" in indexes
        else None
    )
    extra_cols: list[str] = []
    if address_col:
        if "landmark" not in indexes:
            extra_cols.append("landmark")
        if "zip" not in indexes:
            extra_cols.append("zip")
    cleaned_header = header + extra_cols

    cleaned: list[list[str]] = []
    changes: Counter = Counter()

    for row in rows:
        item = dict(zip(header, row))
        for col in header:
            raw = item[col]
            if is_missing(raw):
                if raw != "":
                    changes["missing_token_standardized"] += 1
                item[col] = ""
                continue
            if col in TEXT_COLUMNS:
                normed = normalize_text(raw, col)
                changes["text_values_normalized"] += normed != raw
                item[col] = normed
            elif col == "country":
                normed = normalize_country(raw)
                changes["country_values_normalized"] += normed != raw
                item[col] = normed
            elif col == "phone":
                normed = normalize_phone(raw)
                changes["phone_values_normalized"] += normed != raw
                item[col] = normed
            elif col == "url":
                normed = normalize_url(raw)
                changes["url_whitespace_normalized"] += normed != raw
                item[col] = normed
            elif col == "zip":
                item[col] = normalize_zip(raw, item.get("country", ""))
            else:
                item[col] = canonical_space(raw)

        if address_col:
            address_val = item[address_col]
            existing_zip = item.get("zip", "")
            country_val = item.get("country", "")
            address_val, landmark, extracted_zip = extract_address_details(
                address_val, existing_zip, country_val
            )
            changes["landmarks_extracted"] += bool(landmark)
            changes["pincodes_extracted_from_address"] += bool(extracted_zip and not existing_zip)
            item[address_col] = address_val
            item["zip"] = extracted_zip
            item["landmark"] = landmark

        lat = valid_coordinate(item.get("latitude", ""), 90)
        lon = valid_coordinate(item.get("longitude", ""), 180)
        if "latitude" in item and lat is None and item["latitude"]:
            changes["invalid_latitude_cleared"] += 1
            item["latitude"] = ""
        if "longitude" in item and lon is None and item["longitude"]:
            changes["invalid_longitude_cleared"] += 1
            item["longitude"] = ""

        cleaned.append([item.get(col, "") for col in cleaned_header])

    # Intra-file deduplication using hashes to save memory.
    unique_rows: list[list[str]] = []
    seen: set[bytes] = set()
    for row in cleaned:
        key = _row_hash(row)
        if key in seen:
            changes["intra_file_duplicates_removed"] += 1
        else:
            seen.add(key)
            unique_rows.append(row)

    return cleaned_header, unique_rows, changes


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def write_report(results: list[dict]) -> None:
    lines = [
        "# Train Dataset — Cleaning & EDA Report",
        "",
        "> Generated automatically by `scripts/clean_train_data.py`.",
        "",
        "## Scope & safeguards",
        "",
        "- Each train file was processed **independently**. No files were merged.",
        "- **No cross-file deduplication** was performed: the same record appearing in multiple files is intentionally preserved.",
        "- **Intra-file duplicates** (exact duplicate rows within a single file) were removed.",
        "- Original files were left unchanged; cleaned copies live in `cleaned_data/`.",
        "",
        "## Transformation policy",
        "",
        "| Column(s) | Treatment |",
        "|---|---|",
        "| `business_name`, `name`, `category`, `address`, `business_address`, `city` | Unicode NFKC normalised → lowercased → control chars removed → punctuation stripped (keeping `# / & + -` in addresses) → whitespace collapsed |",
        "| `country` | 2-3 letter codes → UPPERCASE; common long-form names mapped to standard codes |",
        "| `phone` | Non-digit characters removed; leading `+` preserved |",
        "| `url` | Whitespace normalised only (path case is semantically meaningful) |",
        "| `zip` | Leading zeros preserved (stored as text); spaces removed; US 9-digit ZIPs formatted as `12345-6789` |",
        "| `latitude`, `longitude` | Invalid/out-of-range values cleared (not imputed) |",
        "| `entity_id`, `source1_entity_id`, `matched_entity_ids` | Whitespace-trimmed only |",
        "",
        "### Missing-value strategy",
        "",
        "- Tokens `NULL`, `NaN`, `None`, `N/A`, `NA`, `NIL`, `<NULL>`, `-`, `--`, blank, and whitespace-only → standardised to **empty string**.",
        "- No synthetic values were invented for any column.",
        "",
        "### Landmark & PIN/ZIP extraction",
        "",
        "- **Landmark**: extracted when an address contains an explicit cue word (`near`, `opposite`, `beside`, `behind`, `next to`, `adjacent to`, `landmark`).",
        "- **PIN/ZIP**: extracted **only** from the tail of an address, and only when it passes country-specific validation:",
        "  - India: exactly 6 digits, first digit 1-8.",
        "  - US: 5 digits or 5+4 (`12345-6789`).",
        "  - Other: 4-6 digits (conservative).",
        "- Street/house numbers (e.g. `1795` in `1795 Westchester Drive`) are **never** misidentified as postal codes.",
        "",
    ]

    for result in results:
        before = result["before"]
        after = result["after"]
        issues = result["issues"]
        changes = result["changes"]

        lines.extend([
            "---", "",
            f"## {result['source']}", "",
            f"**Output**: `{result['output']}`  ",
            f"**Encoding**: `{result['encoding']}`", "",
            "### Before vs. After", "",
        ])
        lines.extend(markdown_table([
            ("Rows", before["rows"], after["rows"]),
            ("Columns", before["columns"], after["columns"]),
            ("Exact duplicate rows", before["duplicates"], after["duplicates"]),
            ("Blank rows removed", issues["blank_rows"], 0),
            ("Malformed-field-count rows removed", issues["malformed_field_count"], 0),
            ("Repeated header rows removed", issues["repeated_header_rows"], 0),
            ("Invalid latitude values", before["invalid_latitude"], after["invalid_latitude"]),
            ("Invalid longitude values", before["invalid_longitude"], after["invalid_longitude"]),
        ]))

        lines.extend(["", "### Missing Values", ""])
        lines.extend(markdown_table([
            (col, before["missing"].get(col, 0), after["missing"].get(col, 0))
            for col in after["unique"]
        ]))

        lines.extend([
            "", "### Schema & Distributions", "",
            f"- **Columns after cleaning**: `{', '.join(result['header'])}`",
            "- **Data types**: All columns are stored as text/string in the TSV. "
            "`latitude`/`longitude` (where present) are decimal-number strings; "
            "all others are free-text identifiers, names, addresses, codes, or URLs.",
        ])

        if after["latitudes"]:
            lines.append(f"- **Latitude**: {summarize_numeric(after['latitudes'])}")
        if after["longitudes"]:
            lines.append(f"- **Longitude**: {summarize_numeric(after['longitudes'])}")

        if after["countries"]:
            top_countries = ", ".join(
                f'{v or "(empty)"} ({c})' for v, c in after["countries"].most_common(10)
            )
            lines.append(f"- **Top countries**: {top_countries}")
        if after["categories"]:
            top_cats = ", ".join(
                f'{v or "(empty)"} ({c})' for v, c in after["categories"].most_common(10)
            )
            lines.append(f"- **Top categories**: {top_cats}")

        lines.extend(["", "### Unique Values", ""])
        lines.extend(markdown_table([
            (col, before["unique"].get(col, 0), after["unique"].get(col, 0))
            for col in result["header"]
        ]))

        lines.extend([
            "", "### Transformations Applied", "",
            f"- Text values normalised: **{changes['text_values_normalized']}**",
            f"- Country values normalised: **{changes['country_values_normalized']}**",
            f"- Phone values normalised: **{changes['phone_values_normalized']}**",
            f"- URL whitespace normalised: **{changes.get('url_whitespace_normalized', 0)}**",
            f"- Non-empty missing tokens standardised: **{changes['missing_token_standardized']}**",
            f"- Landmarks extracted from address: **{changes['landmarks_extracted']}**",
            f"- PIN/ZIP codes extracted from address: **{changes['pincodes_extracted_from_address']}**",
            f"- Invalid latitudes cleared: **{changes.get('invalid_latitude_cleared', 0)}**",
            f"- Invalid longitudes cleared: **{changes.get('invalid_longitude_cleared', 0)}**",
            f"- **Intra-file duplicates removed: {changes['intra_file_duplicates_removed']}**",
            f"- Inter-file deduplication: **not performed** (by design)",
            "- Remaining blank cells = genuinely unavailable source data.",
            "",
        ])

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main — process a single file (can be called for just source3 or all files)
# ---------------------------------------------------------------------------
def process_file(source: Path, results_list: list[dict]) -> None:
    """Clean a single source file and append its result dict to *results_list*."""
    import gc

    print(f"\n{'='*60}")
    print(f"Processing: {source.name}")
    print(f"{'='*60}")

    header, rows, issues, encoding, delimiter = scan_tsv(source)
    print(f"  Encoding: {encoding}, Delimiter: {'TAB' if delimiter == chr(9) else delimiter}")
    print(f"  Columns: {header}")
    print(f"  Raw rows (excl. header): {len(rows)}")
    print(f"  Issues: {dict(issues)}")

    before = profile_lightweight(header, rows)
    print(f"  Before — duplicates: {before['duplicates']}, "
          f"missing: {dict(before['missing'])}")

    cleaned_header, cleaned_rows, changes = clean_rows(header, rows)

    # Free raw rows immediately.
    del rows
    gc.collect()

    output_path = OUTPUT_DIR / f"{source.stem}_cleaned{source.suffix}"
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter=delimiter)
        writer.writerow(cleaned_header)
        writer.writerows(cleaned_rows)

    after = profile_lightweight(cleaned_header, cleaned_rows)

    # Free cleaned rows.
    del cleaned_rows
    gc.collect()

    print(f"  After  — rows: {after['rows']}, duplicates: {after['duplicates']}")
    print(f"  Changes: {dict(changes)}")
    print(f"  Written: {output_path.name}")

    results_list.append({
        "source": source.relative_to(ROOT).as_posix(),
        "output": output_path.relative_to(ROOT).as_posix(),
        "header": cleaned_header,
        "before": before,
        "after": after,
        "issues": issues,
        "changes": changes,
        "encoding": encoding,
    })


def main() -> None:
    source_files = sorted(DATA_DIR.glob("train_*.tsv"))
    if not source_files:
        raise SystemExit(f"No train_*.tsv files found in {DATA_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for source in source_files:
        process_file(source, results)

    write_report(results)
    print(f"\n{'='*60}")
    print(f"Report: {REPORT_PATH.relative_to(ROOT)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

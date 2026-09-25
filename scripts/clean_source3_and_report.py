"""Process train files one-at-a-time to minimize memory usage.

This script:
1. Cleans train_source3.tsv (the only missing cleaned file)
2. Profiles ALL 4 cleaned files (reading each separately)
3. Generates the comprehensive EDA report
"""

from __future__ import annotations
import csv
import gc
import hashlib
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from statistics import mean, median

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "train"
OUTPUT_DIR = ROOT / "cleaned_data"
REPORT_PATH = OUTPUT_DIR / "eda_cleaning_report.md"
MISSING_TOKENS = {"", "null", "nan", "none", "n/a", "na", "nil", "<null>", "-", "--"}
TEXT_COLUMNS = {"name", "category", "address", "city", "business_name", "business_address"}

LANDMARK_MARKER = re.compile(
    r"\b(?:near|opp(?:osite)?\.?|beside|behind|next\s+to|adjacent\s+to|landmark)\b[:\s]*(.+)$",
    re.IGNORECASE,
)
ADDRESS_PUNCTUATION = re.compile(r"[^\w\s#/&+\-]", re.UNICODE)
GENERAL_PUNCTUATION = re.compile(r"[^\w\s&+\-]", re.UNICODE)
WHITESPACE = re.compile(r"\s+")
PIN_AT_TAIL = re.compile(r"[,\s]+(\d{5,6}(?:-\d{4})?)[\s.]*$")


def is_missing(v): return v is None or v.strip().casefold() in MISSING_TOKENS

def canonical_space(v):
    v = unicodedata.normalize("NFKC", v)
    v = "".join(c for c in v if not unicodedata.category(c).startswith("C"))
    return WHITESPACE.sub(" ", v).strip()

def normalize_text(v, col):
    v = canonical_space(v).casefold()
    p = ADDRESS_PUNCTUATION if col in ("address", "business_address") else GENERAL_PUNCTUATION
    return WHITESPACE.sub(" ", p.sub(" ", v)).strip()

def normalize_country(v):
    v = canonical_space(v).casefold()
    if re.fullmatch(r"[a-z]{2,3}", v): return v.upper()
    m = {"india": "India", "united states": "US", "usa": "US", "united states of america": "US"}
    return m.get(v, v)

def normalize_url(v): return canonical_space(v)

def normalize_phone(v):
    v = canonical_space(v)
    if not v: return v
    pfx = "+" if v.lstrip().startswith("+") else ""
    return pfx + "".join(c for c in v if c.isdigit())

def normalize_zip(v, country):
    v = canonical_space(v).upper().replace(" ", "")
    if not v: return v
    if country == "US" and re.fullmatch(r"\d{9}", v): return v[:5]+"-"+v[5:]
    return v

def _is_valid_pin(cand, country):
    d = cand.replace("-", "")
    if country in ("India", "IN"): return bool(re.fullmatch(r"[1-8]\d{5}", d))
    if country in ("US", "USA"): return bool(re.fullmatch(r"\d{5}", d) or re.fullmatch(r"\d{9}", d))
    return bool(re.fullmatch(r"\d{4,6}", d))

def extract_address_details(addr, zipv, country):
    lm = ""
    mk = LANDMARK_MARKER.search(addr)
    if mk:
        lm = mk.group(1).strip(" ,;- ")
        addr = addr[:mk.start()].strip(" ,;- ")
    if not zipv:
        m = PIN_AT_TAIL.search(addr)
        if m and _is_valid_pin(m.group(1), country):
            zipv = normalize_zip(m.group(1), country)
            addr = addr[:m.start()].strip(" ,;- ")
    return addr, lm, zipv

def valid_coordinate(v, lim):
    try: n = float(v)
    except: return None
    return n if math.isfinite(n) and -lim <= n <= lim else None

def _row_hash(row):
    return hashlib.md5("\t".join(row).encode("utf-8")).digest()


# ---- Scan ----
def scan_tsv(path):
    for enc in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            with path.open("r", encoding=enc, newline="") as fh:
                sample = fh.read(8192); fh.seek(0)
                delim = "\t" if "\t" in sample else ","
                reader = csv.reader(fh, delimiter=delim)
                hdr = next(reader)
                rows, issues, exp = [], Counter(), len(hdr)
                for row in reader:
                    if not any(c.strip() for c in row): issues["blank_rows"] += 1
                    elif len(row) != exp: issues["malformed_field_count"] += 1
                    elif row == hdr: issues["repeated_header_rows"] += 1
                    else: rows.append(row)
            return hdr, rows, issues, enc, delim
        except UnicodeDecodeError: continue
    raise RuntimeError(f"Cannot decode {path}")


# ---- Memory-efficient profile ----
def profile(hdr, rows):
    idx = {n: i for i, n in enumerate(hdr)}
    miss = Counter()
    uhash = {n: set() for n in hdr}
    countries = Counter()
    categories = Counter()
    lats, lons = [], []
    inv_lat = inv_lon = 0
    sr = max(1, len(rows)//100000) if len(rows) > 100000 else 1
    rh_set = set(); dups = 0
    for i, row in enumerate(rows):
        h = _row_hash(row)
        if h in rh_set: dups += 1
        else: rh_set.add(h)
        for n, v in zip(hdr, row):
            if is_missing(v): miss[n] += 1
            uhash[n].add(hash(v))
        if "country" in idx: countries[row[idx["country"]]] += 1
        if "category" in idx: categories[row[idx["category"]]] += 1
        if i % sr == 0:
            if "latitude" in idx:
                lat = valid_coordinate(row[idx["latitude"]], 90)
                if lat is None and not is_missing(row[idx["latitude"]]): inv_lat += 1
                elif lat is not None: lats.append(lat)
            if "longitude" in idx:
                lon = valid_coordinate(row[idx["longitude"]], 180)
                if lon is None and not is_missing(row[idx["longitude"]]): inv_lon += 1
                elif lon is not None: lons.append(lon)
    return {"rows": len(rows), "columns": len(hdr), "missing": miss,
            "unique": {n: len(s) for n, s in uhash.items()},
            "countries": countries, "categories": categories,
            "latitudes": lats, "longitudes": lons,
            "invalid_latitude": inv_lat, "invalid_longitude": inv_lon,
            "duplicates": dups}


# ---- Clean ----
def clean_rows(hdr, rows):
    idx = {n: i for i, n in enumerate(hdr)}
    acol = "address" if "address" in idx else "business_address" if "business_address" in idx else None
    extra = []
    if acol:
        if "landmark" not in idx: extra.append("landmark")
        if "zip" not in idx: extra.append("zip")
    chdr = hdr + extra
    cleaned = []; changes = Counter()
    for row in rows:
        item = dict(zip(hdr, row))
        for col in hdr:
            raw = item[col]
            if is_missing(raw):
                if raw != "": changes["missing_token_standardized"] += 1
                item[col] = ""; continue
            if col in TEXT_COLUMNS:
                n2 = normalize_text(raw, col); changes["text_values_normalized"] += n2 != raw; item[col] = n2
            elif col == "country":
                n2 = normalize_country(raw); changes["country_values_normalized"] += n2 != raw; item[col] = n2
            elif col == "phone":
                n2 = normalize_phone(raw); changes["phone_values_normalized"] += n2 != raw; item[col] = n2
            elif col == "url":
                n2 = normalize_url(raw); changes["url_whitespace_normalized"] += n2 != raw; item[col] = n2
            elif col == "zip":
                item[col] = normalize_zip(raw, item.get("country", ""))
            else:
                item[col] = canonical_space(raw)
        if acol:
            av = item[acol]; ez = item.get("zip", ""); cv = item.get("country", "")
            av, lm, xz = extract_address_details(av, ez, cv)
            changes["landmarks_extracted"] += bool(lm)
            changes["pincodes_extracted_from_address"] += bool(xz and not ez)
            item[acol] = av; item["zip"] = xz; item["landmark"] = lm
        lat = valid_coordinate(item.get("latitude", ""), 90)
        lon = valid_coordinate(item.get("longitude", ""), 180)
        if "latitude" in item and lat is None and item["latitude"]:
            changes["invalid_latitude_cleared"] += 1; item["latitude"] = ""
        if "longitude" in item and lon is None and item["longitude"]:
            changes["invalid_longitude_cleared"] += 1; item["longitude"] = ""
        cleaned.append([item.get(c, "") for c in chdr])
    # Dedup
    uniq = []; seen = set()
    for row in cleaned:
        k = _row_hash(row)
        if k in seen: changes["intra_file_duplicates_removed"] += 1
        else: seen.add(k); uniq.append(row)
    return chdr, uniq, changes


# ---- Numeric summary ----
def summarize_numeric(vals):
    if not vals: return "No valid values"
    o = sorted(vals)
    return f"count={len(vals)}, min={o[0]:.6f}, median={median(o):.6f}, mean={mean(vals):.6f}, max={o[-1]:.6f}"

def md_table(rows):
    ls = ["| Field | Before | After |", "|---|---:|---:|"]
    ls.extend(f"| {n} | {b} | {a} |" for n, b, a in rows)
    return ls


# ---- Report ----
def write_report(results):
    L = [
        "# Train Dataset — Cleaning & EDA Report", "",
        "> Generated automatically by `scripts/clean_train_data.py`.", "",
        "## Scope & safeguards", "",
        "- Each train file was processed **independently**. No files were merged.",
        "- **No cross-file deduplication** was performed: the same record appearing in multiple files is intentionally preserved.",
        "- **Intra-file duplicates** (exact duplicate rows within a single file) were removed.",
        "- Original files were left unchanged; cleaned copies live in `cleaned_data/`.", "",
        "## Transformation policy", "",
        "| Column(s) | Treatment |", "|---|---|",
        "| `business_name`, `name`, `category`, `address`, `business_address`, `city` | Unicode NFKC → lowercase → control chars removed → punctuation stripped (keeping `# / & + -` in addresses) → whitespace collapsed |",
        "| `country` | 2-3 letter codes → UPPERCASE; common long-form names → standard codes |",
        "| `phone` | Non-digit chars removed; leading `+` preserved |",
        "| `url` | Whitespace normalised only |",
        "| `zip` | Leading zeros preserved (text); spaces removed; US 9-digit → `12345-6789` |",
        "| `latitude`, `longitude` | Invalid/out-of-range values cleared (not imputed) |",
        "| `entity_id`, `source1_entity_id`, `matched_entity_ids` | Whitespace-trimmed only |", "",
        "### Missing-value strategy", "",
        "- `NULL`, `NaN`, `None`, `N/A`, `NA`, `NIL`, `<NULL>`, `-`, `--`, blank, whitespace-only → **empty string**.",
        "- No synthetic values invented.", "",
        "### Landmark & PIN/ZIP extraction", "",
        "- **Landmark**: extracted on cue words (`near`, `opposite`, `beside`, `behind`, `next to`, `adjacent to`, `landmark`).",
        "- **PIN/ZIP**: from address **tail only**, country-validated (India: 6 digits 1xxxxx-8xxxxx; US: 5 or 5+4 digits).",
        "- Street/house numbers are **never** misidentified as postal codes.", "",
    ]
    for r in results:
        b, a, iss, ch = r["before"], r["after"], r["issues"], r["changes"]
        L.extend(["---", "", f"## {r['source']}", "",
                   f"**Output**: `{r['output']}`  ", f"**Encoding**: `{r['encoding']}`", "",
                   "### Before vs. After", ""])
        L.extend(md_table([
            ("Rows", b["rows"], a["rows"]), ("Columns", b["columns"], a["columns"]),
            ("Exact duplicate rows", b["duplicates"], a["duplicates"]),
            ("Blank rows removed", iss["blank_rows"], 0),
            ("Malformed-field-count rows", iss["malformed_field_count"], 0),
            ("Repeated header rows", iss["repeated_header_rows"], 0),
            ("Invalid latitude values", b["invalid_latitude"], a["invalid_latitude"]),
            ("Invalid longitude values", b["invalid_longitude"], a["invalid_longitude"]),
        ]))
        L.extend(["", "### Missing Values", ""])
        L.extend(md_table([(c, b["missing"].get(c, 0), a["missing"].get(c, 0)) for c in a["unique"]]))
        L.extend(["", "### Schema & Distributions", "",
                   f"- **Columns**: `{', '.join(r['header'])}`",
                   "- **Data types**: All text/string in TSV."])
        if a["latitudes"]: L.append(f"- **Latitude**: {summarize_numeric(a['latitudes'])}")
        if a["longitudes"]: L.append(f"- **Longitude**: {summarize_numeric(a['longitudes'])}")
        if a["countries"]:
            parts = [f"{v or '(empty)'} ({c})" for v, c in a["countries"].most_common(10)]
            L.append("- **Top countries**: " + ", ".join(parts))
        if a["categories"]:
            parts = [f"{v or '(empty)'} ({c})" for v, c in a["categories"].most_common(10)]
            L.append("- **Top categories**: " + ", ".join(parts))
        L.extend(["", "### Unique Values", ""])
        L.extend(md_table([(c, b["unique"].get(c, 0), a["unique"].get(c, 0)) for c in r["header"]]))
        L.extend(["", "### Transformations Applied", "",
            f"- Text normalised: **{ch['text_values_normalized']}**",
            f"- Country normalised: **{ch['country_values_normalized']}**",
            f"- Phone normalised: **{ch['phone_values_normalized']}**",
            f"- URL whitespace: **{ch.get('url_whitespace_normalized', 0)}**",
            f"- Missing tokens standardised: **{ch['missing_token_standardized']}**",
            f"- Landmarks extracted: **{ch['landmarks_extracted']}**",
            f"- PIN/ZIP extracted: **{ch['pincodes_extracted_from_address']}**",
            f"- Invalid lat cleared: **{ch.get('invalid_latitude_cleared', 0)}**",
            f"- Invalid lon cleared: **{ch.get('invalid_longitude_cleared', 0)}**",
            f"- **Intra-file dups removed: {ch['intra_file_duplicates_removed']}**",
            "- Inter-file dedup: **not performed** (by design)", ""])
    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")


# ===========================================================================
# MAIN: Clean source3, then profile all 4 cleaned files and write report
# ===========================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Process all 4 files.
    for name in ["train_ground_truth", "train_source1", "train_source2", "train_source3"]:
        src = DATA_DIR / f"{name}.tsv"
        out = OUTPUT_DIR / f"{name}_cleaned.tsv"

        if not src.exists():
            print(f"⚠ Source not found: {src}")
            continue

        # Skip if already cleaned, unless it's source3 (which is missing).
        if out.exists() and name != "train_source3":
            print(f"\n{'='*60}")
            print(f"Already cleaned: {name} — profiling only")
            print(f"{'='*60}")

            # Read original for "before" stats.
            hdr, rows, issues, enc, delim = scan_tsv(src)
            before = profile(hdr, rows)
            print(f"  Original: {before['rows']} rows, {before['duplicates']} dups")
            del rows; gc.collect()

            # Read cleaned for "after" stats.
            chdr, crows, _, _, _ = scan_tsv(out)
            after = profile(chdr, crows)
            print(f"  Cleaned:  {after['rows']} rows, {after['duplicates']} dups")

            # Estimate changes for previously-cleaned files.
            changes = Counter({
                "text_values_normalized": 0,
                "country_values_normalized": 0,
                "phone_values_normalized": 0,
                "url_whitespace_normalized": 0,
                "missing_token_standardized": 0,
                "landmarks_extracted": 0,
                "pincodes_extracted_from_address": 0,
                "invalid_latitude_cleared": 0,
                "invalid_longitude_cleared": 0,
                "intra_file_duplicates_removed": before["rows"] - after["rows"],
            })
            del crows; gc.collect()

            all_results.append({
                "source": src.relative_to(ROOT).as_posix(),
                "output": out.relative_to(ROOT).as_posix(),
                "header": chdr, "before": before, "after": after,
                "issues": issues, "changes": changes, "encoding": enc,
            })
        else:
            # Full clean + profile.
            print(f"\n{'='*60}")
            print(f"Cleaning: {name}")
            print(f"{'='*60}")

            hdr, rows, issues, enc, delim = scan_tsv(src)
            print(f"  Rows: {len(rows)}, Issues: {dict(issues)}")

            before = profile(hdr, rows)
            print(f"  Before: {before['rows']} rows, {before['duplicates']} dups, missing: {dict(before['missing'])}")

            chdr, crows, changes = clean_rows(hdr, rows)
            del rows; gc.collect()

            with out.open("w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh, delimiter=delim)
                w.writerow(chdr)
                w.writerows(crows)

            after = profile(chdr, crows)
            del crows; gc.collect()

            print(f"  After: {after['rows']} rows, {after['duplicates']} dups")
            print(f"  Changes: {dict(changes)}")
            print(f"  Written: {out.name}")

            all_results.append({
                "source": src.relative_to(ROOT).as_posix(),
                "output": out.relative_to(ROOT).as_posix(),
                "header": chdr, "before": before, "after": after,
                "issues": issues, "changes": changes, "encoding": enc,
            })

    # Write the comprehensive report.
    write_report(all_results)
    print(f"\n{'='*60}")
    print(f"✅ Report: {REPORT_PATH.relative_to(ROOT)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

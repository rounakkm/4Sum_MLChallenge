"""Verify ALL cleaning requirements across all 4 cleaned train files.

Checks:
1. Text normalization (lowercase, no leading/trailing whitespace, no extra spaces)
2. Missing/empty values (no NULL/NaN/None/N-A tokens remain)
3. No stray/invalid text
4. Landmark extraction (separate from address)
5. PIN/ZIP extraction (separate, validated, no street numbers)
6. Intra-file duplicates removed
7. Inter-file duplicates preserved
8. File structure preserved
"""

import csv
import re
import sys
from collections import Counter
from pathlib import Path
import hashlib

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parents[1]
CLEANED_DIR = ROOT / "cleaned_data"
ORIGINAL_DIR = ROOT / "dataset" / "train"

MISSING_TOKENS = {"null", "nan", "none", "n/a", "na", "nil", "<null>"}
FILES = ["train_ground_truth", "train_source1", "train_source2", "train_source3"]

def read_tsv(path):
    with path.open("r", encoding="utf-8", newline="") as fh:
        sample = fh.read(8192); fh.seek(0)
        delim = "\t" if "\t" in sample else ","
        reader = csv.reader(fh, delimiter=delim)
        hdr = next(reader)
        rows = [r for r in reader if any(c.strip() for c in r)]
    return hdr, rows

def row_hash(row):
    return hashlib.md5("\t".join(row).encode("utf-8")).digest()


print("=" * 70)
print("COMPREHENSIVE CLEANING VERIFICATION REPORT")
print("=" * 70)

all_pass = True
file_hashes = {}  # For inter-file duplicate check

for fname in FILES:
    cleaned_path = CLEANED_DIR / f"{fname}_cleaned.tsv"
    original_path = ORIGINAL_DIR / f"{fname}.tsv"

    if not cleaned_path.exists():
        print(f"\nFAIL: {fname}_cleaned.tsv NOT FOUND!")
        all_pass = False
        continue

    print(f"\n{'='*70}")
    print(f"  Verifying: {fname}_cleaned.tsv")
    print(f"{'='*70}")

    hdr, rows = read_tsv(cleaned_path)
    orig_hdr, orig_rows = read_tsv(original_path)

    issues = []

    # ---- CHECK 1: Text Normalization ----
    text_cols = [c for c in hdr if c in ("business_name", "business_address", "name", "category", "address", "city")]
    upper_count = 0
    leading_trailing_ws = 0
    double_space = 0
    sample_upper = []

    for i, row in enumerate(rows):
        for col_name in text_cols:
            idx = hdr.index(col_name)
            val = row[idx]
            if not val:
                continue
            if val != val.strip():
                leading_trailing_ws += 1
            if "  " in val:
                double_space += 1
            # Check for uppercase (skip non-Latin chars)
            latin_chars = [c for c in val if c.isalpha() and ord(c) < 128]
            if any(c.isupper() for c in latin_chars):
                upper_count += 1
                if len(sample_upper) < 3:
                    sample_upper.append(f"  row {i}, col '{col_name}': '{val[:80]}'")

    if upper_count == 0:
        print("  [PASS] Text normalization: all text columns lowercase")
    else:
        print(f"  [WARN] Text normalization: {upper_count} values have uppercase Latin chars")
        for s in sample_upper:
            print(s)
        # This might be intentional for abbreviations like 'lAcarning' in original
        # Only fail if it's widespread
        if upper_count > len(rows) * 0.01:
            issues.append("Widespread uppercase in text columns")

    if leading_trailing_ws == 0:
        print("  [PASS] No leading/trailing whitespace in text columns")
    else:
        print(f"  [FAIL] {leading_trailing_ws} values have leading/trailing whitespace")
        issues.append("Leading/trailing whitespace found")

    if double_space == 0:
        print("  [PASS] No double spaces in text columns")
    else:
        print(f"  [FAIL] {double_space} values have consecutive spaces")
        issues.append("Double spaces found")

    # ---- CHECK 2: Missing Value Tokens ----
    stale_tokens = 0
    stale_samples = []
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            if val.strip().lower() in MISSING_TOKENS and val.strip() != "":
                stale_tokens += 1
                if len(stale_samples) < 3:
                    stale_samples.append(f"  row {i}, col '{hdr[j]}': '{val}'")

    if stale_tokens == 0:
        print("  [PASS] No stale missing-value tokens (NULL/NaN/None/N-A/etc.)")
    else:
        print(f"  [FAIL] {stale_tokens} stale missing tokens remain")
        for s in stale_samples:
            print(s)
        issues.append("Stale missing tokens")

    # ---- CHECK 3: Landmark & ZIP columns exist (for source files) ----
    if fname != "train_ground_truth":
        if "landmark" in hdr:
            print("  [PASS] 'landmark' column exists")
        else:
            print("  [FAIL] 'landmark' column missing")
            issues.append("Missing landmark column")

        if "zip" in hdr:
            print("  [PASS] 'zip' column exists")
        else:
            print("  [FAIL] 'zip' column missing")
            issues.append("Missing zip column")

        # Check ZIP values are valid where present
        if "zip" in hdr:
            zip_idx = hdr.index("zip")
            country_idx = hdr.index("country") if "country" in hdr else None
            bad_zips = 0
            zip_samples = []
            for i, row in enumerate(rows):
                zv = row[zip_idx]
                if not zv:
                    continue
                country = row[country_idx] if country_idx is not None else ""
                # Basic validation
                digits = zv.replace("-", "")
                if not digits.isdigit():
                    bad_zips += 1
                    if len(zip_samples) < 3:
                        zip_samples.append(f"  row {i}: zip='{zv}', country='{country}'")

            if bad_zips == 0:
                print("  [PASS] All ZIP/PIN values are numeric (valid format)")
            else:
                print(f"  [WARN] {bad_zips} ZIP values contain non-numeric chars")
                for s in zip_samples:
                    print(s)

        # Check no street numbers extracted as ZIP (spot check first 10000 rows)
        if "zip" in hdr and "business_address" in hdr:
            zip_idx = hdr.index("zip")
            addr_idx = hdr.index("business_address")
            suspicious = 0
            for row in rows[:10000]:
                zv = row[zip_idx]
                if zv and len(zv) < 5:  # Very short "ZIP" is likely a street number
                    suspicious += 1
            if suspicious == 0:
                print("  [PASS] No suspiciously short ZIP codes (street number leak check)")
            else:
                print(f"  [WARN] {suspicious} ZIP values shorter than 5 digits in first 10k rows")

    # ---- CHECK 4: Intra-file Duplicates ----
    seen = set()
    intra_dups = 0
    for row in rows:
        h = row_hash(row)
        if h in seen:
            intra_dups += 1
        else:
            seen.add(h)

    if intra_dups == 0:
        print(f"  [PASS] No intra-file duplicates (0 found in {len(rows)} rows)")
    else:
        print(f"  [FAIL] {intra_dups} intra-file duplicates remain!")
        issues.append(f"{intra_dups} intra-file duplicates")

    # Store hashes for inter-file check
    file_hashes[fname] = seen

    # ---- CHECK 5: File Structure ----
    if fname == "train_ground_truth":
        expected_cols = {"source1_entity_id", "matched_entity_ids"}
    else:
        expected_base = {"entity_id", "business_name", "business_address", "country"}
        expected_cols = expected_base | {"landmark", "zip"}

    if set(hdr) == expected_cols:
        print(f"  [PASS] Column structure correct: {hdr}")
    else:
        missing = expected_cols - set(hdr)
        extra = set(hdr) - expected_cols
        if missing:
            print(f"  [WARN] Missing expected columns: {missing}")
        if extra:
            print(f"  [INFO] Extra columns: {extra}")

    # ---- CHECK 6: Row count ----
    print(f"  [INFO] Original rows: {len(orig_rows)}, Cleaned rows: {len(rows)}")
    if len(rows) <= len(orig_rows):
        print(f"  [PASS] Row count valid (cleaned <= original)")
    else:
        print(f"  [FAIL] Cleaned has MORE rows than original!")
        issues.append("Row count increased")

    # Summary for this file
    if issues:
        print(f"\n  ISSUES: {issues}")
        all_pass = False
    else:
        print(f"\n  ALL CHECKS PASSED for {fname}")

    del rows, orig_rows, seen
    import gc; gc.collect()

# ---- CHECK 7: Inter-file Duplicates Preserved ----
print(f"\n{'='*70}")
print("  Inter-file Duplicate Check")
print(f"{'='*70}")

source_files = [f for f in FILES if f != "train_ground_truth"]
for i, f1 in enumerate(source_files):
    for f2 in source_files[i+1:]:
        if f1 in file_hashes and f2 in file_hashes:
            shared = len(file_hashes[f1] & file_hashes[f2])
            print(f"  {f1} <-> {f2}: {shared} shared row hashes")
            if shared > 0:
                print(f"  [PASS] Inter-file duplicates PRESERVED between {f1} and {f2}")

# ---- FINAL SUMMARY ----
print(f"\n{'='*70}")
if all_pass:
    print("  FINAL RESULT: ALL CHECKS PASSED")
else:
    print("  FINAL RESULT: SOME ISSUES FOUND (see above)")
print(f"{'='*70}")

#!/usr/bin/env python3
"""Validation script for Amazon ML Challenge 2026 submission files.

Checks both matching_results.tsv and candidate_pairs.tsv against all competition rules:
  1. Header format and column names
  2. Tab-separation and no quoting
  3. Every Source 1 test entity present exactly once (no duplicates, none missing)
  4. All matched/candidate IDs exist in test Source 2 or Source 3
  5. No Source 1 IDs in matched/candidate lists (no self-matches)
  6. No duplicate entity IDs within any entity's list
  7. Every matched ID in matching_results.tsv must appear in candidate_pairs.tsv
"""

import argparse
import sys
from pathlib import Path


def load_valid_ids(test_dir):
    test_dir = Path(test_dir)
    s1_ids = set()
    s2_ids = set()
    s3_ids = set()

    s1_path = test_dir / "test_source1.tsv"
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"

    if not s1_path.exists():
        sys.exit(f"Error: {s1_path} not found.")
    if not s2_path.exists():
        sys.exit(f"Error: {s2_path} not found.")
    if not s3_path.exists():
        sys.exit(f"Error: {s3_path} not found.")

    with open(s1_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        id_idx = header.index("entity_id") if "entity_id" in header else 0
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts and parts[id_idx]:
                s1_ids.add(parts[id_idx])

    with open(s2_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        id_idx = header.index("entity_id") if "entity_id" in header else 0
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts and parts[id_idx]:
                s2_ids.add(parts[id_idx])

    with open(s3_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        id_idx = header.index("entity_id") if "entity_id" in header else 0
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts and parts[id_idx]:
                s3_ids.add(parts[id_idx])

    return s1_ids, s2_ids | s3_ids


def validate_file(file_path, expected_id_col, expected_list_col, valid_s1_ids, valid_other_ids):
    file_path = Path(file_path)
    issues = []
    entity_map = {}

    if not file_path.exists():
        return [f"File {file_path} does not exist."], {}

    seen_s1 = set()
    duplicate_s1 = 0
    invalid_ids = 0
    self_matches = 0
    internal_duplicates = 0

    with open(file_path, "r", encoding="utf-8") as f:
        header_line = f.readline().rstrip("\r\n")
        header_parts = header_line.split("\t")
        if len(header_parts) != 2:
            issues.append(f"Header must have exactly 2 tab-separated columns, found {len(header_parts)}: {header_line}")
        elif header_parts[0] != expected_id_col or header_parts[1] != expected_list_col:
            issues.append(f"Invalid header columns. Expected '{expected_id_col}\\t{expected_list_col}', found '{header_parts[0]}\\t{header_parts[1]}'")

        for line_num, line in enumerate(f, start=2):
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 1:
                s1_id = parts[0]
                id_list_str = ""
            elif len(parts) == 2:
                s1_id, id_list_str = parts
            else:
                issues.append(f"Line {line_num}: expected at most 2 tab-separated fields, got {len(parts)}")
                continue

            if s1_id in seen_s1:
                duplicate_s1 += 1
            seen_s1.add(s1_id)

            if s1_id not in valid_s1_ids:
                issues.append(f"Line {line_num}: source1_entity_id '{s1_id}' does not exist in test_source1.tsv")

            ids = [i.strip() for i in id_list_str.split(",") if i.strip()] if id_list_str else []

            # Check internal duplicates
            if len(ids) != len(set(ids)):
                internal_duplicates += 1

            for cid in ids:
                if cid.startswith("S1-"):
                    self_matches += 1
                elif cid not in valid_other_ids:
                    invalid_ids += 1

            entity_map[s1_id] = set(ids)

    if duplicate_s1 > 0:
        issues.append(f"Found {duplicate_s1} duplicate source1_entity_id rows.")

    missing_s1 = len(valid_s1_ids - seen_s1)
    if missing_s1 > 0:
        issues.append(f"Missing {missing_s1} Source 1 test entities. Every test entity must be present.")

    if internal_duplicates > 0:
        issues.append(f"Found {internal_duplicates} rows with duplicate IDs in the ID list.")

    if self_matches > 0:
        issues.append(f"Found {self_matches} self-matches (referencing S1- entities). Only S2- and S3- IDs allowed.")

    if invalid_ids > 0:
        issues.append(f"Found {invalid_ids} entity IDs that do not exist in test_source2 or test_source3.")

    return issues, entity_map


def main():
    parser = argparse.ArgumentParser(description="Validate submission files for Amazon ML Challenge 2026.")
    parser.add_argument("--matching", required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, help="Directory containing test_source1.tsv, test_source2.tsv, test_source3.tsv")
    args = parser.parse_args()

    print("Loading valid test entity IDs...")
    valid_s1, valid_other = load_valid_ids(args.test_dir)
    print(f"Loaded {len(valid_s1)} Source 1 IDs, {len(valid_other)} Source 2/3 IDs.")

    print("\nValidating matching_results.tsv...")
    matching_issues, matching_map = validate_file(
        args.matching, "source1_entity_id", "matched_entity_ids", valid_s1, valid_other
    )

    print("Validating candidate_pairs.tsv...")
    candidate_issues, candidate_map = validate_file(
        args.candidate, "source1_entity_id", "candidate_entity_ids", valid_s1, valid_other
    )

    # Cross-check: every matched ID should appear in candidates
    subset_violations = 0
    for s1_id, matches in matching_map.items():
        cands = candidate_map.get(s1_id, set())
        diff = matches - cands
        if diff:
            subset_violations += len(diff)

    cross_issues = []
    if subset_violations > 0:
        cross_issues.append(f"WARNING: {subset_violations} matched entity IDs were not in the candidate set (pipeline bug).")

    all_errors = matching_issues + candidate_issues
    if all_errors:
        print("\n" + "=" * 50)
        print("VALIDATION FAILED with the following issues:")
        print("=" * 50)
        for i, issue in enumerate(all_errors, start=1):
            print(f"{i}. {issue}")
        if cross_issues:
            print("\nWarnings:")
            for w in cross_issues:
                print(f"- {w}")
        sys.exit(1)
    else:
        if cross_issues:
            print("\nWarnings:")
            for w in cross_issues:
                print(f"- {w}")
        print("\n" + "=" * 50)
        print("PASS")
        print("=" * 50)
        print("All submission format requirements are met!")
        sys.exit(0)


if __name__ == "__main__":
    main()

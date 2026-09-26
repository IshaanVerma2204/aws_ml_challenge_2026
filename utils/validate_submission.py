import argparse
import os
import sys
import pandas as pd

def validate(matching_file, candidate_file, test_dir):
    print("=" * 60)
    print("VALIDATING SUBMISSION FILES")
    print("=" * 60)

    issues = []

    # 1. Check file existence
    if not os.path.exists(matching_file):
        issues.append(f"Matching results file not found: {matching_file}")
    if not os.path.exists(candidate_file):
        issues.append(f"Candidate pairs file not found: {candidate_file}")
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")
    for p in [s1_path, s2_path, s3_path]:
        if not os.path.exists(p):
            issues.append(f"Test file not found: {p}")

    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"[{i}] ERROR: {issue}")
        sys.exit(1)

    # Load test sets
    print("Loading test entity IDs...")
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str)
    df_s2 = pd.read_csv(s2_path, sep="\t", dtype=str)
    df_s3 = pd.read_csv(s3_path, sep="\t", dtype=str)

    expected_s1_ids = list(df_s1['entity_id'].values)
    expected_s1_set = set(expected_s1_ids)
    valid_target_set = set(df_s2['entity_id'].values).union(set(df_s3['entity_id'].values))

    print(f"Expected Source 1 entities: {len(expected_s1_set)}")
    print(f"Valid target entities (S2 + S3): {len(valid_target_set)}")

    # Load submission files
    print("\nReading submission files...")
    matching_df = pd.read_csv(matching_file, sep="\t", dtype=str, keep_default_na=False)
    candidate_df = pd.read_csv(candidate_file, sep="\t", dtype=str, keep_default_na=False)

    # 2. Check headers
    if list(matching_df.columns) != ["source1_entity_id", "matched_entity_ids"]:
        issues.append(f"matching_results.tsv headers must be ['source1_entity_id', 'matched_entity_ids'], got {list(matching_df.columns)}")
    if list(candidate_df.columns) != ["source1_entity_id", "candidate_entity_ids"]:
        issues.append(f"candidate_pairs.tsv headers must be ['source1_entity_id', 'candidate_entity_ids'], got {list(candidate_df.columns)}")

    # 3. Check row counts and completeness
    matching_s1 = list(matching_df["source1_entity_id"].values)
    candidate_s1 = list(candidate_df["source1_entity_id"].values)

    if len(matching_s1) != len(expected_s1_ids):
        issues.append(f"matching_results.tsv has {len(matching_s1)} rows, expected {len(expected_s1_ids)}")
    if len(candidate_s1) != len(expected_s1_ids):
        issues.append(f"candidate_pairs.tsv has {len(candidate_s1)} rows, expected {len(expected_s1_ids)}")

    if set(matching_s1) != expected_s1_set:
        issues.append("matching_results.tsv contains missing or extra source1_entity_ids")
    if set(candidate_s1) != expected_s1_set:
        issues.append("candidate_pairs.tsv contains missing or extra source1_entity_ids")

    if len(matching_s1) != len(set(matching_s1)):
        issues.append("matching_results.tsv contains duplicate source1_entity_ids")
    if len(candidate_s1) != len(set(candidate_s1)):
        issues.append("candidate_pairs.tsv contains duplicate source1_entity_ids")

    # 4. Check entity IDs and subset consistency
    matching_dict = dict(zip(matching_df["source1_entity_id"], matching_df["matched_entity_ids"]))
    candidate_dict = dict(zip(candidate_df["source1_entity_id"], candidate_df["candidate_entity_ids"]))

    invalid_target_ids = set()
    dup_ids_in_row = 0
    non_subset_matches = 0

    for s1_id in expected_s1_ids:
        matches = [x.strip() for x in matching_dict.get(s1_id, "").split(",") if x.strip()]
        candidates = [x.strip() for x in candidate_dict.get(s1_id, "").split(",") if x.strip()]

        match_set = set(matches)
        cand_set = set(candidates)

        if len(matches) != len(match_set):
            dup_ids_in_row += 1
        if len(candidates) != len(cand_set):
            dup_ids_in_row += 1

        for m in match_set:
            if m not in valid_target_set:
                invalid_target_ids.add(m)
        for c in cand_set:
            if c not in valid_target_set:
                invalid_target_ids.add(c)

        if not match_set.issubset(cand_set):
            non_subset_matches += 1

    if invalid_target_ids:
        issues.append(f"Found {len(invalid_target_ids)} invalid target IDs that do not exist in test S2 or S3")
    if dup_ids_in_row > 0:
        issues.append(f"Found {dup_ids_in_row} rows with duplicate IDs in matched or candidate lists")
    if non_subset_matches > 0:
        issues.append(f"Found {non_subset_matches} rows where matched_entity_ids is not a subset of candidate_entity_ids")

    if issues:
        print("\n" + "=" * 60)
        print("VALIDATION FAILED WITH ISSUES:")
        for i, issue in enumerate(issues, 1):
            print(f"[{i}] {issue}")
        print("=" * 60)
        sys.exit(1)
    else:
        print("\nPASS: All validation checks passed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate ER Submission Files")
    parser.add_argument("--matching", type=str, required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", type=str, required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", type=str, required=True, help="Path to dataset/test directory")
    args = parser.parse_args()

    validate(args.matching, args.candidate, args.test_dir)

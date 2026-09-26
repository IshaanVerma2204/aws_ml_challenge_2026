import pandas as pd
import os

def load_data(data_dir, split="train"):
    """
    Loads source 1, 2, 3 and optionally ground truth for a given split.
    Reads TSV files with explicit tab separator as required by the challenge.
    """
    s1_path = os.path.join(data_dir, split, f"{split}_source1.tsv")
    s2_path = os.path.join(data_dir, split, f"{split}_source2.tsv")
    s3_path = os.path.join(data_dir, split, f"{split}_source3.tsv")
    
    print(f"Loading {split} Source 1: {s1_path}...", flush=True)
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str).fillna("")
    print(f"Loading {split} Source 2: {s2_path}...", flush=True)
    df_s2 = pd.read_csv(s2_path, sep="\t", dtype=str).fillna("")
    print(f"Loading {split} Source 3: {s3_path}...", flush=True)
    df_s3 = pd.read_csv(s3_path, sep="\t", dtype=str).fillna("")
    
    if split == "train":
        gt_path = os.path.join(data_dir, split, "train_ground_truth.tsv")
        print(f"Loading ground truth: {gt_path}...", flush=True)
        df_gt = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
        return df_s1, df_s2, df_s3, df_gt
    
    return df_s1, df_s2, df_s3

def save_outputs(matching_results, candidate_pairs, output_dir):
    """
    Saves matching_results.tsv and candidate_pairs.tsv in tab-separated format.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    matching_results.to_csv(matching_path, sep="\t", index=False)
    print(f"Saved matching results to {matching_path}", flush=True)
    
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    candidate_pairs.to_csv(candidate_path, sep="\t", index=False)
    print(f"Saved candidate pairs to {candidate_path}", flush=True)

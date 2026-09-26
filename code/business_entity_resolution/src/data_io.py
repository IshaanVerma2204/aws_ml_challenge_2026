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
    df_s1 = normalize_dataframe(df_s1)
    
    print(f"Loading {split} Source 2: {s2_path}...", flush=True)
    df_s2 = pd.read_csv(s2_path, sep="\t", dtype=str).fillna("")
    df_s2 = normalize_dataframe(df_s2)
    
    print(f"Loading {split} Source 3: {s3_path}...", flush=True)
    df_s3 = pd.read_csv(s3_path, sep="\t", dtype=str).fillna("")
    df_s3 = normalize_dataframe(df_s3)
    
    if split == "train":
        gt_path = os.path.join(data_dir, split, "train_ground_truth.tsv")
        print(f"Loading ground truth: {gt_path}...", flush=True)
        df_gt = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
        return df_s1, df_s2, df_s3, df_gt
    
    return df_s1, df_s2, df_s3

def normalize_text(series):
    """
    Expands common abbreviations in business names and addresses using a highly optimized single-pass regex.
    """
    replacements = {
        'corp': 'corporation',
        'pvt': 'private',
        'ltd': 'limited',
        'inc': 'incorporated',
        'co': 'company',
        'llc': 'limited liability company',
        'rd': 'road',
        'st': 'street',
        'ave': 'avenue',
        'blvd': 'boulevard',
        'dr': 'drive',
        'ln': 'lane',
        'ct': 'court',
        'bldg': 'building',
        'fl': 'floor',
        'apt': 'apartment',
        'ste': 'suite',
        'dept': 'department',
        'univ': 'university',
        'intl': 'international',
        'ctr': 'center',
        'mt': 'mount',
        'mfg': 'manufacturing'
    }
    
    s = series.str.lower().str.replace(r'[^\w\s]', ' ', regex=True)
    pattern = r'\b(' + '|'.join(replacements.keys()) + r')\b'
    s = s.str.replace(pattern, lambda m: replacements.get(m.group(1), m.group(1)), regex=True)
    return s.str.replace(r'\s+', ' ', regex=True).str.strip()

def normalize_dataframe(df):
    """Normalizes the business_name and business_address columns."""
    if 'business_name' in df.columns:
        df['business_name'] = normalize_text(df['business_name'])
    if 'business_address' in df.columns:
        df['business_address'] = normalize_text(df['business_address'])
    return df


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

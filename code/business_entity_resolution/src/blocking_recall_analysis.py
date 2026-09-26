import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Analyze recall of blocking stage")
    parser.add_argument("--gt_path", type=str, default="../dataset/train/train_ground_truth.tsv")
    parser.add_argument("--cand_path", type=str, default="../output/candidate_pairs.tsv")
    args = parser.parse_args()
    
    if not os.path.exists(args.gt_path) or not os.path.exists(args.cand_path):
        print("Required files not found.")
        return
        
    print(f"Loading Ground Truth: {args.gt_path}")
    df_gt = pd.read_csv(args.gt_path, sep='\t')
    
    print(f"Loading Candidates: {args.cand_path}")
    df_cand = pd.read_csv(args.cand_path, sep='\t')
    
    # Parse GT
    gt_pairs = set()
    for _, row in df_gt.iterrows():
        s1 = row['source1_entity_id']
        matches = str(row['matched_entity_ids']).split(',')
        for m in matches:
            m = m.strip()
            if m:
                gt_pairs.add((s1, m))
                
    # Parse Candidates
    cand_pairs = set()
    for _, row in df_cand.iterrows():
        s1 = row['source1_entity_id']
        matches = str(row['candidate_entity_ids']).split(',')
        for m in matches:
            m = m.strip()
            if m:
                cand_pairs.add((s1, m))
                
    # Calculate Recall
    if len(gt_pairs) == 0:
        print("No ground truth pairs found.")
        return
        
    hits = len(gt_pairs.intersection(cand_pairs))
    total = len(gt_pairs)
    recall = hits / total
    
    print("-" * 50)
    print(f"BLOCKING RECALL ANALYSIS")
    print("-" * 50)
    print(f"Total Ground Truth Pairs: {total:,}")
    print(f"Pairs Found in Candidates: {hits:,}")
    print(f"Missed Pairs: {total - hits:,}")
    print(f"Recall: {recall * 100:.2f}%")
    print("-" * 50)

if __name__ == "__main__":
    main()

import argparse
import os
import pandas as pd
from data_io import load_data, save_outputs
from blocking import generate_candidates, format_candidate_pairs
from features import compute_features
from model import prepare_training_data, train_model, predict_matches, format_matching_results

def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    args = parser.parse_args()
    
    # 1. Load Training Data
    print("Loading training data...")
    df_s1_train, df_s2_train, df_s3_train, df_gt = load_data(args.data_dir, split="train")
    
    # 2. Generate candidates for training
    df_cand_train = generate_candidates(df_s1_train, df_s2_train, df_s3_train, top_k=20)
    
    # 3. Compute features for training
    df_feat_train = compute_features(df_cand_train, df_s1_train, df_s2_train, df_s3_train)
    
    # 4. Prepare labels and train model
    df_train = prepare_training_data(df_feat_train, df_gt)
    model, threshold = train_model(df_train)
    
    # 5. Load Test Data
    print("Loading test data...")
    df_s1_test, df_s2_test, df_s3_test = load_data(args.data_dir, split="test")
    
    # 6. Generate candidates for test
    df_cand_test = generate_candidates(df_s1_test, df_s2_test, df_s3_test, top_k=20)
    candidate_pairs_df = format_candidate_pairs(df_s1_test, df_cand_test)
    
    # 7. Compute features for test
    df_feat_test = compute_features(df_cand_test, df_s1_test, df_s2_test, df_s3_test)
    
    # 8. Predict matches
    print("Predicting matches for test set...")
    df_matches = predict_matches(model, df_feat_test, threshold)
    matching_results_df = format_matching_results(df_s1_test, df_matches)
    
    # 9. Save Outputs
    save_outputs(matching_results_df, candidate_pairs_df, args.output_dir)

if __name__ == "__main__":
    main()

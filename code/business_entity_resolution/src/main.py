import argparse
import os
import sys
import time
import pickle
import json
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_io import load_data, save_outputs
from blocking import generate_candidates, format_candidate_pairs
from features import compute_features, predict_matches_streamed
from model import prepare_training_data, train_model, predict_matches, format_matching_results


CHECKPOINT_FILE = "output/checkpoint_trained_model.pkl"


def save_checkpoint(model, threshold, path=CHECKPOINT_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({'model': model, 'threshold': threshold}, f)
    print(f"  [Checkpoint] Model saved to {path}", flush=True)


def load_checkpoint(path=CHECKPOINT_FILE):
    if not os.path.exists(path):
        return None, None
    with open(path, 'rb') as f:
        ck = pickle.load(f)
    print(f"  [Checkpoint] Loaded model from {path}  "
          f"(threshold={ck['threshold']:.4f})", flush=True)
    return ck['model'], ck['threshold']


def main():
    parser = argparse.ArgumentParser(description="End-to-End Business Entity Resolution Pipeline")
    parser.add_argument("--data_dir",          type=str,   default="dataset")
    parser.add_argument("--output_dir",        type=str,   default="output")
    parser.add_argument("--top_k",             type=int,   default=None, help="Overrides config.yaml if set")
    parser.add_argument("--min_sim",           type=float, default=None, help="Overrides config.yaml if set")
    parser.add_argument("--train_sample_size", type=int,   default=None, help="Overrides config.yaml if set")
    parser.add_argument("--semantic",          action="store_true",
                        help="Use SentenceTransformers instead of TF-IDF for blocking")
    parser.add_argument("--skip_training",     action="store_true",
                        help="Skip Stages 1-4 and load saved model checkpoint instead")
    parser.add_argument("--force_retrain",     action="store_true",
                        help="Force re-training even if checkpoint exists")
    args = parser.parse_args()
    
    # Load Config YAML
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = {}
    try:
        import yaml
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Loaded configuration from {config_path}")
    except ImportError:
        print("PyYAML not installed. Using default hyperparams.", flush=True)
        
    top_k = args.top_k if args.top_k is not None else config.get('blocking', {}).get('top_k', 20)
    min_sim = args.min_sim if args.min_sim is not None else config.get('blocking', {}).get('min_sim', 0.15)
    train_sample_size = args.train_sample_size if args.train_sample_size is not None else config.get('pipeline', {}).get('train_sample_size', 50000)
    use_semantic = args.semantic or config.get('blocking', {}).get('use_semantic', False)
    
    run_summary = {
        "timestamp": time.time(),
        "args": vars(args),
        "config": config,
        "metrics": {}
    }

    t_start = time.time()
    print("=" * 70, flush=True)
    print("BUSINESS ENTITY RESOLUTION PIPELINE (AWS ML CHALLENGE 2026)", flush=True)
    print("=" * 70, flush=True)

    # ------------------------------------------------------------------ #
    # Decide whether to train or load checkpoint                         #
    # ------------------------------------------------------------------ #
    checkpoint_exists = os.path.exists(CHECKPOINT_FILE)
    do_train = (not args.skip_training) and (args.force_retrain or not checkpoint_exists)

    if not do_train and checkpoint_exists:
        print(f"\n[*] Checkpoint found at '{CHECKPOINT_FILE}'.", flush=True)
        print("    Skipping training (Stages 1-4). Loading saved model...", flush=True)
        model, threshold = load_checkpoint()
        if model is None:
            print("    ERROR: Could not load checkpoint. Re-training...", flush=True)
            do_train = True

    # ------------------------------------------------------------------ #
    # STAGES 1-4: Training (skipped if checkpoint loaded)                #
    # ------------------------------------------------------------------ #
    if do_train:
        # 1. Load Training Data
        print("\n--- STAGE 1: LOAD TRAINING DATA ---", flush=True)
        df_s1_train, df_s2_train, df_s3_train, df_gt = load_data(args.data_dir, split="train")

        if train_sample_size is not None and train_sample_size > 0 and len(df_s1_train) > train_sample_size:
            print(f"Sampling {train_sample_size} Source 1 entities for training...", flush=True)
            df_s1_train = df_s1_train.sample(n=train_sample_size, random_state=42).reset_index(drop=True)

        # 2. Blocking (Train)
        print("\n--- STAGE 2: BLOCKING / CANDIDATE GENERATION (TRAIN) ---", flush=True)
        cand_train_cache_path = os.path.join(args.output_dir, "cache_candidates_train.feather")
        if os.path.exists(cand_train_cache_path):
            print(f"  [Cache] Loading cached train candidates from {cand_train_cache_path}...", flush=True)
            df_cand_train = pd.read_feather(cand_train_cache_path)
        else:
            df_cand_train = generate_candidates(
                df_s1_train, df_s2_train, df_s3_train,
                top_k=top_k, min_sim=min_sim,
                use_semantic=use_semantic
            )
            try:
                os.makedirs(args.output_dir, exist_ok=True)
                df_cand_train.to_feather(cand_train_cache_path)
                print(f"  [Cache] Saved train candidate pairs to {cand_train_cache_path}", flush=True)
            except Exception as e:
                print(f"  [Cache warning] Could not save train cache: {e}", flush=True)

        print("    Injecting missed ground truth positive pairs into training candidates...", flush=True)
        valid_s1_train = set(df_s1_train['entity_id'].values)
        gt_pairs_list = []
        for _, row in df_gt.iterrows():
            s1 = row['source1_entity_id']
            if s1 in valid_s1_train:
                matches = str(row['matched_entity_ids']).split(',')
                for m in matches:
                    m = m.strip()
                    if m:
                        gt_pairs_list.append({'source1_entity_id': s1, 'candidate_entity_id': m, 'tfidf_sim': 1.0})
        
        df_gt_flat = pd.DataFrame(gt_pairs_list)
        if not df_gt_flat.empty:
            df_cand_train = pd.concat([df_cand_train, df_gt_flat], ignore_index=True)
            df_cand_train = df_cand_train.drop_duplicates(subset=['source1_entity_id', 'candidate_entity_id'])
        print(f"    Total candidates after GT injection: {len(df_cand_train):,}", flush=True)

        # 3. Features (Train)
        print("\n--- STAGE 3: FEATURE ENGINEERING (TRAIN) ---", flush=True)
        df_feat_train = compute_features(df_cand_train, df_s1_train, df_s2_train, df_s3_train)

        # 4. Train Model
        print("\n--- STAGE 4: MODEL TRAINING & MACRO F0.5 THRESHOLD TUNING ---", flush=True)
        df_train = prepare_training_data(df_feat_train, df_gt)
        model, threshold = train_model(df_train, df_gt)

        # Save checkpoint so next run skips training
        save_checkpoint(model, threshold)

    # ------------------------------------------------------------------ #
    # STAGES 5-9: Test inference                                         #
    # ------------------------------------------------------------------ #

    # 5. Load Test Data
    print("\n--- STAGE 5: LOAD TEST DATA ---", flush=True)
    df_s1_test, df_s2_test, df_s3_test = load_data(args.data_dir, split="test")

    # 6. Blocking (Test)  <-- GPU-accelerated
    print("\n--- STAGE 6: BLOCKING / CANDIDATE GENERATION (TEST) ---", flush=True)
    cand_cache_path = os.path.join(args.output_dir, "cache_candidates_test.feather")
    if os.path.exists(cand_cache_path):
        print(f"  [Cache] Loading cached test candidates from {cand_cache_path}...", flush=True)
        df_cand_test = pd.read_feather(cand_cache_path)
    else:
        df_cand_test = generate_candidates(
            df_s1_test, df_s2_test, df_s3_test,
            top_k=top_k, min_sim=min_sim,
            use_semantic=use_semantic
        )
        try:
            df_cand_test.to_feather(cand_cache_path)
            print(f"  [Cache] Saved candidate pairs cache to {cand_cache_path}", flush=True)
        except Exception as e:
            print(f"  [Cache warning] Could not save cache: {e}", flush=True)

    print("Formatting candidate_pairs.tsv...", flush=True)
    candidate_pairs_df = format_candidate_pairs(df_s1_test, df_cand_test)
    
    # Save candidate_pairs.tsv immediately
    os.makedirs(args.output_dir, exist_ok=True)
    cand_out_path = os.path.join(args.output_dir, "candidate_pairs.tsv")
    candidate_pairs_df.to_csv(cand_out_path, sep='\t', index=False)
    print(f"  Saved {cand_out_path} ({len(candidate_pairs_df):,} rows)", flush=True)

    # 7 & 8. Streamed Features & Inference (Memory-safe, <500MB RAM)
    print("\n--- STAGE 7 & 8: STREAMED FEATURE ENGINEERING & INFERENCE (TEST) ---", flush=True)
    df_matches = predict_matches_streamed(model, df_cand_test, df_s1_test, df_s2_test, df_s3_test, threshold)
    matching_results_df = format_matching_results(df_s1_test, df_matches)

    # 9. Save matching_results.tsv
    print("\n--- STAGE 9: SAVE FINAL MATCHING RESULTS ---", flush=True)
    match_out_path = os.path.join(args.output_dir, "matching_results.tsv")
    matching_results_df.to_csv(match_out_path, sep='\t', index=False)
    print(f"  Saved {match_out_path} ({len(matching_results_df):,} rows)", flush=True)

    print("\n" + "=" * 70, flush=True)
    total_time = time.time() - t_start
    print(f"PIPELINE DONE IN {total_time:.2f}s!", flush=True)
    print("=" * 70, flush=True)
    
    # Structured logging
    run_summary["metrics"]["total_time_seconds"] = total_time
    run_summary["metrics"]["test_matches_predicted"] = len(matching_results_df)
    if 'threshold' in locals() and threshold is not None:
        run_summary["metrics"]["optimal_threshold"] = float(threshold)
        
    summary_path = os.path.join(args.output_dir, "run_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(run_summary, f, indent=4)
    print(f"Run summary saved to {summary_path}")

if __name__ == "__main__":
    main()

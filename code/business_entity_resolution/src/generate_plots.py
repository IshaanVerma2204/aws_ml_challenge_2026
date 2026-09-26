import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix, classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, fbeta_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_io import load_data
from features import compute_features
from model import prepare_training_data

# Set global aesthetic parameters
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 150})

def compute_f05(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=0.5, average='macro', zero_division=0)

def main():
    print("Generating Model Evaluation Charts...")
    output_dir = "c:/Users/aashi/Desktop/AWSchallenge/output"
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Load Model Checkpoint
    ck_path = os.path.join(output_dir, "checkpoint_trained_model.pkl")
    if not os.path.exists(ck_path):
        print("Model checkpoint not found. Run training first.")
        return
        
    with open(ck_path, 'rb') as f:
        ck = pickle.load(f)
    
    ensemble = ck['model']
    opt_thresh = ck['threshold']
    xgb = ensemble.xgb_model
    lr = ensemble.lr_model
    
    print("Models loaded successfully.")
    
    # 2. Load Train Candidates (Cached)
    cache_path = os.path.join(output_dir, "cache_candidates_train.feather")
    print(f"Loading candidates from {cache_path}...")
    df_cand = pd.read_feather(cache_path)
    
    # Load raw data for feature computation
    data_dir = "c:/Users/aashi/Desktop/AWSchallenge/dataset"
    df_s1, df_s2, df_s3, df_gt = load_data(data_dir, split="train")
    
    # Extract GT positive pairs (Vectorized)
    valid_s1 = set(df_s1['entity_id'].values)
    df_gt_filtered = df_gt[df_gt['source1_entity_id'].isin(valid_s1)].copy()
    df_gt_filtered['matched_entity_ids'] = df_gt_filtered['matched_entity_ids'].astype(str).str.split(',')
    df_gt_exploded = df_gt_filtered.explode('matched_entity_ids')
    df_gt_exploded['matched_entity_ids'] = df_gt_exploded['matched_entity_ids'].str.strip()
    df_gt_exploded = df_gt_exploded[df_gt_exploded['matched_entity_ids'] != '']
    gt_pairs = set(zip(df_gt_exploded['source1_entity_id'], df_gt_exploded['matched_entity_ids']))
                    
    # Label candidates (Vectorized)
    print("Labeling candidates...")
    keys = pd.Series(zip(df_cand['source1_entity_id'], df_cand['candidate_entity_id']))
    df_cand['label'] = keys.isin(gt_pairs).astype(int)
    
    # Sub-sample to speed up feature extraction (Take all positives, sample negatives)
    positives = df_cand[df_cand['label'] == 1]
    negatives = df_cand[df_cand['label'] == 0].sample(n=min(150000, len(df_cand[df_cand['label'] == 0])), random_state=42)
    df_eval = pd.concat([positives, negatives]).reset_index(drop=True)
    
    print(f"Evaluating on {len(df_eval)} pairs ({len(positives)} pos, {len(negatives)} neg)...")
    
    # Compute features
    print("Computing features...")
    df_feat = compute_features(df_eval, df_s1, df_s2, df_s3)
    df_feat['label'] = df_eval['label'].values
    
    X = df_feat.drop(columns=['label', 'source1_entity_id', 'candidate_entity_id'])
    y = df_feat['label'].values
    
    # Get probabilities
    print("Generating predictions...")
    dmatrix = ensemble.xgb_model.get_booster().feature_names
    # Predict XGB
    preds_xgb = xgb.predict_proba(X)[:, 1]
    # Predict LR
    # Scale features for LR
    X_scaled = ensemble.scaler.transform(X)
    preds_lr = lr.predict_proba(X_scaled)[:, 1]
    # Predict Ensemble
    preds_ens = ensemble.predict_proba(X)[:, 1]
    
    # ---------------------------------------------------------
    # PLOT 1: F0.5 vs Threshold for Single vs Combined
    # ---------------------------------------------------------
    print("Plotting F0.5 vs Threshold...")
    thresholds = np.linspace(0.1, 0.95, 40)
    scores_xgb, scores_lr, scores_ens = [], [], []
    
    for t in thresholds:
        scores_xgb.append(compute_f05(y, (preds_xgb >= t).astype(int)))
        scores_lr.append(compute_f05(y, (preds_lr >= t).astype(int)))
        scores_ens.append(compute_f05(y, (preds_ens >= t).astype(int)))
        
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, scores_xgb, label='XGBoost (Single)', color='#1f77b4', linewidth=2.5)
    plt.plot(thresholds, scores_lr, label='Logistic Regression (Single)', color='#ff7f0e', linewidth=2.5, linestyle='--')
    plt.plot(thresholds, scores_ens, label='Ensemble (Combined)', color='#2ca02c', linewidth=3)
    plt.axvline(x=opt_thresh, color='red', linestyle=':', label=f'Optimal Threshold ({opt_thresh:.2f})')
    
    plt.title('Macro F0.5 Score vs. Decision Threshold', fontsize=16, fontweight='bold')
    plt.xlabel('Probability Threshold', fontsize=14)
    plt.ylabel('Macro F0.5 Score', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'f05_vs_threshold.png'))
    plt.close()
    
    # ---------------------------------------------------------
    # PLOT 2: Precision-Recall Curve Comparison
    # ---------------------------------------------------------
    print("Plotting Precision-Recall Curve...")
    plt.figure(figsize=(9, 7))
    
    for name, preds, color, style in [('XGBoost', preds_xgb, '#1f77b4', '-'), 
                                      ('Logistic Regression', preds_lr, '#ff7f0e', '--'), 
                                      ('Ensemble (Combined)', preds_ens, '#2ca02c', '-')]:
        precision, recall, _ = precision_recall_curve(y, preds)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.3f})', color=color, linestyle=style, linewidth=2.5)
        
    plt.title('Precision-Recall Curve Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.legend(loc='lower left', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'precision_recall_comparison.png'))
    plt.close()
    
    # ---------------------------------------------------------
    # PLOT 3: Class-wise Metrics Bar Chart (at Optimal Threshold)
    # ---------------------------------------------------------
    print("Plotting Class-wise Metrics...")
    pred_binary = (preds_ens >= opt_thresh).astype(int)
    
    metrics = {
        'Precision': [precision_score(y, pred_binary, pos_label=0), precision_score(y, pred_binary, pos_label=1)],
        'Recall': [recall_score(y, pred_binary, pos_label=0), recall_score(y, pred_binary, pos_label=1)],
        'F1-Score': [f1_score(y, pred_binary, pos_label=0), f1_score(y, pred_binary, pos_label=1)]
    }
    
    x = np.arange(2)
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, metrics['Precision'], width, label='Precision', color='#4c72b0')
    ax.bar(x, metrics['Recall'], width, label='Recall', color='#dd8452')
    ax.bar(x + width, metrics['F1-Score'], width, label='F1-Score', color='#55a868')
    
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Class-wise Performance Metrics (Ensemble)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Negative Class (No Match)', 'Positive Class (Match)'], fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.1)
    
    # Add labels on top of bars
    for i, v in enumerate(metrics['Precision']):
        ax.text(i - width, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    for i, v in enumerate(metrics['Recall']):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    for i, v in enumerate(metrics['F1-Score']):
        ax.text(i + width, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'class_wise_metrics.png'))
    plt.close()
    
    # ---------------------------------------------------------
    # PLOT 4: Confusion Matrix (Ensemble)
    # ---------------------------------------------------------
    print("Plotting Confusion Matrix...")
    cm = confusion_matrix(y, pred_binary)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 14, "weight": "bold"},
                xticklabels=['Predict Neg', 'Predict Pos'], 
                yticklabels=['Actual Neg', 'Actual Pos'])
    plt.title('Ensemble Confusion Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))
    plt.close()
    
    # ---------------------------------------------------------
    # PLOT 5: XGBoost Feature Importance
    # ---------------------------------------------------------
    print("Plotting Feature Importances...")
    importances = xgb.feature_importances_
    features = X.columns
    
    feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feat_df = feat_df.sort_values(by='Importance', ascending=False).head(15)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis')
    plt.title('Top 15 Feature Importances (XGBoost Backbone)', fontsize=16, fontweight='bold')
    plt.xlabel('Relative Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_importance.png'))
    plt.close()
    
    print(f"\nAll plots successfully generated and saved to: {plots_dir}")

if __name__ == "__main__":
    main()

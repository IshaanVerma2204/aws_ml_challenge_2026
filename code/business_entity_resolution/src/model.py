import pandas as pd
import numpy as np
import xgboost as xgb
import time

def compute_macro_f05(gt_val_map, val_pred_map, all_val_s1_ids):
    """
    Computes exact competition Macro F0.5 score across all validation Source 1 entities.
    Formula: F0.5 = (1.25 * P * R) / (0.25 * P + R)
    Macro-averaged across all S1 entities including singletons.
    """
    f05_scores = []
    
    for s1_id in all_val_s1_ids:
        true_set = gt_val_map.get(s1_id, set())
        pred_set = val_pred_map.get(s1_id, set())
        
        if not true_set and not pred_set:
            f05_scores.append(1.0)
            continue
        if not true_set and pred_set:
            f05_scores.append(0.0)
            continue
        if true_set and not pred_set:
            f05_scores.append(0.0)
            continue
            
        intersection = len(true_set.intersection(pred_set))
        precision = intersection / len(pred_set)
        recall = intersection / len(true_set)
        
        if precision + recall == 0:
            f05_scores.append(0.0)
        else:
            score = (1.25 * precision * recall) / (0.25 * precision + recall)
            f05_scores.append(score)
            
    return float(np.mean(f05_scores))

def prepare_training_data(df_features, df_gt):
    """
    Creates binary ground truth labels for feature dataframe.
    """
    print("Parsing ground truth matching pairs...", flush=True)
    gt_pairs = set()
    for _, row in df_gt.iterrows():
        s1_id = row['source1_entity_id']
        matches = str(row['matched_entity_ids']).split(',')
        for match in matches:
            match_clean = match.strip()
            if match_clean:
                gt_pairs.add((s1_id, match_clean))
                
    labels = [1 if (s1_id, cand_id) in gt_pairs else 0 
              for s1_id, cand_id in zip(df_features['source1_entity_id'], df_features['candidate_entity_id'])]
              
    df_features['label'] = labels
    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    print(f"Dataset prepared: {len(labels)} candidate pairs ({pos_count} positive, {neg_count} negative).", flush=True)
    return df_features

def train_model(df_features, df_gt):
    """
    Trains XGBoost classifier using grouped validation split (by source1_entity_id)
    and sweeps probability threshold to directly maximize Macro F0.5 score.
    """
    feature_cols = [c for c in df_features.columns if c not in ['source1_entity_id', 'candidate_entity_id', 'label']]
    
    unique_s1 = df_features['source1_entity_id'].unique()
    np.random.seed(42)
    val_s1_set = set(np.random.choice(unique_s1, size=int(len(unique_s1) * 0.2), replace=False))
    
    val_mask = df_features['source1_entity_id'].isin(val_s1_set)
    train_mask = ~val_mask
    
    df_train = df_features[train_mask]
    df_val = df_features[val_mask]
    
    X_train, y_train = df_train[feature_cols], df_train['label']
    X_val, y_val = df_val[feature_cols], df_val['label']
    
    print(f"Training XGBoost classifier (Train pairs: {len(X_train)}, Val pairs: {len(X_val)})...", flush=True)
    
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42,
        tree_method='hist'
    )
    
    t0 = time.time()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"Model fit completed in {time.time() - t0:.2f}s.", flush=True)
    
    gt_val_map = {}
    for _, row in df_gt[df_gt['source1_entity_id'].isin(val_s1_set)].iterrows():
        matches = set(x.strip() for x in str(row['matched_entity_ids']).split(',') if x.strip())
        gt_val_map[row['source1_entity_id']] = matches
        
    val_probs = model.predict_proba(X_val)[:, 1]
    df_val = df_val.copy()
    df_val['prob'] = val_probs
    
    print("Sweeping probability threshold for Macro F0.5 optimization...", flush=True)
    best_threshold = 0.50
    best_macro_f05 = 0.0
    
    for th in np.arange(0.30, 0.90, 0.02):
        th = round(th, 2)
        val_matches = df_val[df_val['prob'] >= th]
        val_pred_map = val_matches.groupby('source1_entity_id')['candidate_entity_id'].apply(set).to_dict()
        
        macro_f05 = compute_macro_f05(gt_val_map, val_pred_map, val_s1_set)
        if macro_f05 > best_macro_f05:
            best_macro_f05 = macro_f05
            best_threshold = th
            
    print(f"Optimal Threshold for Macro F0.5: {best_threshold:.2f} | Validation Macro F0.5 Score: {best_macro_f05:.4f}", flush=True)
    return model, best_threshold

def predict_matches(model, df_features, threshold):
    """
    Predicts matches for test set candidate pairs using the trained model and optimal threshold.
    """
    if df_features.empty:
        return pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_id'])
        
    feature_cols = [c for c in df_features.columns if c not in ['source1_entity_id', 'candidate_entity_id', 'label']]
    probs = model.predict_proba(df_features[feature_cols])[:, 1]
    
    df_features['prob'] = probs
    df_matches = df_features[df_features['prob'] >= threshold].copy()
    print(f"Inference complete: {len(df_matches)} matches predicted at threshold >= {threshold:.2f}", flush=True)
    return df_matches

def format_matching_results(df_s1, df_matches):
    """
    Formats matches into matching_results.tsv structure:
    source1_entity_id \t matched_entity_ids (comma-separated).
    Guarantees exactly one row per S1 entity in df_s1.
    """
    if df_matches.empty:
        match_dict = {}
    else:
        grouped = df_matches.groupby('source1_entity_id')['candidate_entity_id'].apply(
            lambda x: ','.join(list(dict.fromkeys(x)))
        )
        match_dict = grouped.to_dict()
        
    res = []
    for s1_id in df_s1['entity_id']:
        res.append({
            'source1_entity_id': s1_id,
            'matched_entity_ids': match_dict.get(s1_id, "")
        })
        
    return pd.DataFrame(res)

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score

def prepare_training_data(df_features, df_gt):
    """
    Creates labels based on ground truth.
    df_gt has source1_entity_id, matched_entity_ids
    """
    # Parse ground truth into pairs
    gt_pairs = set()
    for _, row in df_gt.iterrows():
        s1_id = row['source1_entity_id']
        matches = str(row['matched_entity_ids']).split(',')
        for match in matches:
            if match.strip():
                gt_pairs.add((s1_id, match.strip()))
                
    # Create label column
    labels = []
    for _, row in df_features.iterrows():
        s1_id = row['source1_entity_id']
        cand_id = row['candidate_entity_id']
        labels.append(1 if (s1_id, cand_id) in gt_pairs else 0)
        
    df_features['label'] = labels
    return df_features

def train_model(df_features):
    """
    Trains XGBoost on the feature set.
    """
    feature_cols = [c for c in df_features.columns if c not in ['source1_entity_id', 'candidate_entity_id', 'label']]
    
    X = df_features[feature_cols]
    y = df_features['label']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    
    print("Training model...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=30, verbose=False)
    
    # Predict probabilities for validation
    y_pred_prob = model.predict_proba(X_val)[:, 1]
    
    # Optimize threshold for F_0.5 (beta=0.5)
    best_threshold = 0.5
    best_f05 = 0
    for threshold in np.arange(0.3, 0.9, 0.05):
        y_pred = (y_pred_prob >= threshold).astype(int)
        f05 = fbeta_score(y_val, y_pred, beta=0.5)
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = threshold
            
    print(f"Best Threshold for F0.5: {best_threshold:.2f}, F0.5 Score: {best_f05:.4f}")
    
    return model, best_threshold

def predict_matches(model, df_features, threshold):
    feature_cols = [c for c in df_features.columns if c not in ['source1_entity_id', 'candidate_entity_id', 'label']]
    
    X_test = df_features[feature_cols]
    preds = model.predict_proba(X_test)[:, 1]
    
    df_features['match_prob'] = preds
    df_matches = df_features[df_features['match_prob'] >= threshold]
    
    return df_matches

def format_matching_results(df_s1, df_matches):
    """
    Formats the matches into the required TSV structure.
    """
    if df_matches.empty:
        match_dict = {}
    else:
        grouped = df_matches.groupby('source1_entity_id')['candidate_entity_id'].apply(lambda x: ','.join(list(dict.fromkeys(x))))
        match_dict = grouped.to_dict()
    
    res = []
    for s1_id in df_s1['entity_id']:
        res.append({
            'source1_entity_id': s1_id,
            'matched_entity_ids': match_dict.get(s1_id, "")
        })
    
    return pd.DataFrame(res)

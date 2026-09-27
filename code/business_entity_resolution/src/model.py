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

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

class EnsembleModel:
    def __init__(self, xgb_model, lr_model, rf_model, hgb_model, lgb_model, cb_model, scaler):
        self.xgb_model = xgb_model
        self.lr_model = lr_model
        self.rf_model = rf_model
        self.hgb_model = hgb_model
        self.lgb_model = lgb_model
        self.cb_model = cb_model
        self.scaler = scaler
        
    def predict_proba(self, X):
        X_clean = X.fillna(0)
        xgb_p = self.xgb_model.predict_proba(X_clean)[:, 1]
        rf_p = self.rf_model.predict_proba(X_clean)[:, 1]
        hgb_p = self.hgb_model.predict_proba(X_clean)[:, 1]
        lgb_p = self.lgb_model.predict_proba(X_clean)[:, 1]
        cb_p = self.cb_model.predict_proba(X_clean)[:, 1]
        lr_p = self.lr_model.predict_proba(self.scaler.transform(X_clean))[:, 1]
        ens = 0.3 * xgb_p + 0.2 * lgb_p + 0.2 * cb_p + 0.1 * hgb_p + 0.1 * rf_p + 0.1 * lr_p
        res = np.zeros((len(X), 2))
        res[:, 1] = ens
        return res

def train_model(df_features, df_gt):
    """
    Trains an Ensemble (XGBoost + Logistic Regression) using GroupKFold CV.
    Sweeps OOF predictions to find optimal Macro F0.5 threshold.
    Trains final model on FULL dataset with Hard Negative Mining weights.
    """
    feature_cols = [c for c in df_features.columns if c not in ['source1_entity_id', 'candidate_entity_id', 'label']]
    
    X = df_features[feature_cols].fillna(0)
    y = df_features['label'].values
    groups = df_features['source1_entity_id'].values
    
    pos_weight = (len(y) - sum(y)) / max(1, sum(y))
    
    print("Starting 3-Fold Cross Validation for Ensemble & Hard Negative Mining...", flush=True)
    
    gkf = GroupKFold(n_splits=3)
    oof_preds = np.zeros(len(X))
    
    gt_val_map = {}
    for _, row in df_gt.iterrows():
        matches = set(x.strip() for x in str(row['matched_entity_ids']).split(',') if x.strip())
        gt_val_map[row['source1_entity_id']] = matches
        
    t_cv = time.time()
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"  Fold {fold+1}/3...", flush=True)
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        xgb_m = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_weight,
            eval_metric='logloss', tree_method='hist', random_state=42
        )
        xgb_m.fit(X_train, y_train)
        xgb_p = xgb_m.predict_proba(X_val)[:, 1]
        
        rf_m = RandomForestClassifier(n_estimators=150, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
        rf_m.fit(X_train, y_train)
        rf_p = rf_m.predict_proba(X_val)[:, 1]
        
        hgb_m = HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.05, random_state=42)
        hgb_m.fit(X_train, y_train)
        hgb_p = hgb_m.predict_proba(X_val)[:, 1]
        
        lgb_m = lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
        lgb_m.fit(X_train, y_train)
        lgb_p = lgb_m.predict_proba(X_val)[:, 1]
        
        cb_m = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, random_state=42, verbose=0)
        cb_m.fit(X_train, y_train)
        cb_p = cb_m.predict_proba(X_val)[:, 1]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        
        lr_m = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        lr_m.fit(X_train_s, y_train)
        lr_p = lr_m.predict_proba(X_val_s)[:, 1]
        
        oof_preds[val_idx] = 0.3 * xgb_p + 0.2 * lgb_p + 0.2 * cb_p + 0.1 * hgb_p + 0.1 * rf_p + 0.1 * lr_p
        
    print(f"CV completed in {time.time()-t_cv:.1f}s. Sweeping threshold for Macro F0.5...", flush=True)
    
    df_features_copy = df_features.copy()
    df_features_copy['oof_prob'] = oof_preds
    
    best_threshold = 0.50
    best_macro_f05 = 0.0
    unique_s1 = set(df_gt['source1_entity_id'].values)
    
    for th in np.arange(0.30, 0.90, 0.02):
        th = round(th, 2)
        val_matches = df_features_copy[df_features_copy['oof_prob'] >= th]
        val_pred_map = val_matches.groupby('source1_entity_id')['candidate_entity_id'].apply(set).to_dict()
        
        macro_f05 = compute_macro_f05(gt_val_map, val_pred_map, unique_s1)
        if macro_f05 > best_macro_f05:
            best_macro_f05 = macro_f05
            best_threshold = th
            
    print(f"Optimal Threshold (CV OOF): {best_threshold:.2f} | OOF Macro F0.5 Score: {best_macro_f05:.4f}", flush=True)
    
    print("Training final Ensemble on FULL dataset with Hard Negative Mining weights...", flush=True)
    hard_neg_mask = (oof_preds >= best_threshold) & (y == 0)
    sample_weights = np.ones(len(y))
    sample_weights[hard_neg_mask] = 2.0  # Double weight for hard negatives
    print(f"  Identified {sum(hard_neg_mask):,} hard negatives from OOF.", flush=True)
    
    final_xgb = xgb.XGBClassifier(
        n_estimators=600, max_depth=7, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_weight,
        eval_metric='logloss', tree_method='hist', random_state=42
    )
    final_xgb.fit(X, y, sample_weight=sample_weights)
    
    final_rf = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1)
    final_rf.fit(X, y, sample_weight=sample_weights)

    final_hgb = HistGradientBoostingClassifier(max_iter=400, max_depth=7, learning_rate=0.03, random_state=42)
    final_hgb.fit(X, y, sample_weight=sample_weights)
    
    final_lgb = lgb.LGBMClassifier(n_estimators=600, max_depth=7, learning_rate=0.03, random_state=42, n_jobs=-1)
    final_lgb.fit(X, y, sample_weight=sample_weights)
    
    final_cb = CatBoostClassifier(iterations=600, depth=7, learning_rate=0.03, random_state=42, verbose=0)
    final_cb.fit(X, y, sample_weight=sample_weights)
    
    final_scaler = StandardScaler()
    X_s = final_scaler.fit_transform(X)
    final_lr = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
    final_lr.fit(X_s, y, sample_weight=sample_weights)
    
    final_model = EnsembleModel(final_xgb, final_lr, final_rf, final_hgb, final_lgb, final_cb, final_scaler)
    return final_model, best_threshold

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

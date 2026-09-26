import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import rapidfuzz
from rapidfuzz import fuzz

print("Loading sample training data (nrows)...")
t0 = time.time()
s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", dtype=str, nrows=50000).fillna("")
s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t", dtype=str, nrows=100000).fillna("")
s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t", dtype=str, nrows=100000).fillna("")
gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")

print(f"Data loaded in {time.time() - t0:.2f}s. S1={len(s1)}, S2={len(s2)}, S3={len(s3)}")

# Ground truth map for evaluation
gt_s1_ids = set(s1['entity_id'])
gt_filtered = gt[gt['source1_entity_id'].isin(gt_s1_ids)]
gt_map = {}
for _, row in gt_filtered.iterrows():
    matches = set(x.strip() for x in str(row['matched_entity_ids']).split(',') if x.strip())
    gt_map[row['source1_entity_id']] = matches

# Pool S2 + S3
pool = pd.concat([s2, s3], ignore_index=True)

# Filter pool to valid GT targets for this sample to accurately check recall
valid_target_set = set(pool['entity_id'])
eval_gt_map = {}
for s1_id, matches in gt_map.items():
    valid_m = matches.intersection(valid_target_set)
    if valid_m:
        eval_gt_map[s1_id] = valid_m

# Preprocess text
s1['text'] = (s1['business_name'] + " " + s1['business_address']).str.lower()
pool['text'] = (pool['business_name'] + " " + pool['business_address']).str.lower()

t0 = time.time()
print("Fitting TF-IDF Vectorizer...")
vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=50000, sublinear_tf=True)
tfidf_pool = vec.fit_transform(pool['text'])
tfidf_s1 = vec.transform(s1['text'])

print(f"TF-IDF matrix built in {time.time() - t0:.2f}s. Pool shape: {tfidf_pool.shape}, S1 shape: {tfidf_s1.shape}")

t0 = time.time()
# Fast batch sparse matrix multiplication
batch_size = 10000
top_k = 20
pool_ids = pool['entity_id'].values
s1_ids = s1['entity_id'].values

candidates = []
n_s1 = tfidf_s1.shape[0]

for start in range(0, n_s1, batch_size):
    end = min(start + batch_size, n_s1)
    # Cosine similarity matrix: batch x pool
    sim_matrix = tfidf_s1[start:end] @ tfidf_pool.T
    
    # For each row in batch, get top_k indices
    for i in range(end - start):
        row_s1_id = s1_ids[start + i]
        row_sim = sim_matrix[i].toarray().ravel()
        if len(row_sim) <= top_k:
            best_idx = np.argsort(-row_sim)
        else:
            best_idx = np.argpartition(-row_sim, top_k)[:top_k]
            best_idx = best_idx[np.argsort(-row_sim[best_idx])]
            
        for idx in best_idx:
            sim = row_sim[idx]
            if sim >= 0.15:
                candidates.append((row_s1_id, pool_ids[idx], float(sim)))

t_block = time.time() - t0
print(f"Blocking completed in {t_block:.2f}s. Total candidate pairs: {len(candidates)}")

# Calculate blocking recall on GT
found_matches = 0
total_matches = sum(len(m) for m in eval_gt_map.values())

cand_dict = {}
for s1_id, cand_id, sim in candidates:
    if s1_id not in cand_dict:
        cand_dict[s1_id] = set()
    cand_dict[s1_id].add(cand_id)

for s1_id, true_matches in eval_gt_map.items():
    if s1_id in cand_dict:
        found_matches += len(true_matches.intersection(cand_dict[s1_id]))

print(f"Blocking recall on GT present in pool: {found_matches} / {total_matches} ({100*found_matches/max(1, total_matches):.2f}%)")

# Test RapidFuzz feature extraction speed
print("\nTesting RapidFuzz feature extraction speed...")
t0 = time.time()
s1_dict = s1.set_index('entity_id').to_dict('index')
pool_dict = pool.set_index('entity_id').to_dict('index')

features = []
cand_sub = candidates[:50000]

for s1_id, cand_id, sim in cand_sub:
    rec1 = s1_dict[s1_id]
    rec2 = pool_dict[cand_id]
    n1, n2 = rec1['business_name'], rec2['business_name']
    a1, a2 = rec1['business_address'], rec2['business_address']
    
    r_name = fuzz.ratio(n1, n2)
    r_ts_name = fuzz.token_sort_ratio(n1, n2)
    r_addr = fuzz.ratio(a1, a2)
    r_ts_addr = fuzz.token_sort_ratio(a1, a2)
    features.append((r_name, r_ts_name, r_addr, r_ts_addr, sim))

t_feat = time.time() - t0
print(f"Computed features for 50,000 candidate pairs in {t_feat:.2f}s ({len(cand_sub)/t_feat:.0f} pairs/sec)!")

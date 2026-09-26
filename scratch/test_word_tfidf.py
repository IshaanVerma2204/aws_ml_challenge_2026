import pandas as pd
import numpy as np
import scipy.sparse as sp
import time
from sklearn.feature_extraction.text import TfidfVectorizer

print("1. Loading dataset sample...")
s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", dtype=str, nrows=50000).fillna("")
s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t", dtype=str, nrows=200000).fillna("")
s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t", dtype=str, nrows=200000).fillna("")
gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")

pool = pd.concat([s2, s3], ignore_index=True)

s1['clean_text'] = (s1['business_name'] + " " + s1['business_address']).str.lower()
pool['clean_text'] = (pool['business_name'] + " " + pool['business_address']).str.lower()

gt_s1_ids = set(s1['entity_id'])
gt_filtered = gt[gt['source1_entity_id'].isin(gt_s1_ids)]
eval_gt_map = {}
valid_target_set = set(pool['entity_id'])
for _, row in gt_filtered.iterrows():
    m = set(x.strip() for x in str(row['matched_entity_ids']).split(',') if x.strip()).intersection(valid_target_set)
    if m:
        eval_gt_map[row['source1_entity_id']] = m

print(f"Data ready: S1={len(s1)}, Pool={len(pool)}")

print("2. Testing Word (1,2) + Char_wb (4,5) TF-IDF Vectorizer...")
t0 = time.time()
vec_word = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, max_df=0.20, max_features=40000, sublinear_tf=True, dtype=np.float32)
vec_char = TfidfVectorizer(analyzer='char_wb', ngram_range=(4, 5), min_df=5, max_df=0.15, max_features=40000, sublinear_tf=True, dtype=np.float32)

sample_pool = pool['clean_text'].sample(n=min(100000, len(pool)), random_state=42)
vec_word.fit(sample_pool)
vec_char.fit(sample_pool)

tfidf_w_pool = vec_word.transform(pool['clean_text'])
tfidf_w_s1 = vec_word.transform(s1['clean_text'])

tfidf_c_pool = vec_char.transform(pool['clean_text'])
tfidf_c_s1 = vec_char.transform(s1['clean_text'])

# Combine word and char features
tfidf_pool = sp.hstack([tfidf_w_pool, tfidf_c_pool], format='csr')
tfidf_s1 = sp.hstack([tfidf_w_s1, tfidf_c_s1], format='csr')

print(f"Combined TF-IDF constructed in {time.time() - t0:.2f}s. Pool shape: {tfidf_pool.shape}, S1 shape: {tfidf_s1.shape}")

t0 = time.time()
batch_size = 5000
top_k = 20
pool_ids = pool['entity_id'].values
s1_ids = s1['entity_id'].values

candidates = []
n_s1 = tfidf_s1.shape[0]

tfidf_pool_T = tfidf_pool.T.tocsc()

for start in range(0, n_s1, batch_size):
    tb = time.time()
    end = min(start + batch_size, n_s1)
    sim_batch = (tfidf_s1[start:end] @ tfidf_pool_T).tocsr()
    
    indptr = sim_batch.indptr
    indices = sim_batch.indices
    data = sim_batch.data
    
    for i in range(end - start):
        s1_id = s1_ids[start + i]
        r_s, r_e = indptr[i], indptr[i+1]
        if r_s == r_e:
            continue
            
        d = data[r_s:r_e]
        idx = indices[r_s:r_e]
        
        if len(d) <= top_k:
            best = np.argsort(-d)
        else:
            best = np.argpartition(-d, top_k)[:top_k]
            best = best[np.argsort(-d[best])]
            
        for b in best:
            if d[b] >= 0.15:
                candidates.append((s1_id, pool_ids[idx[b]], float(d[b])))
                
    print(f"  Batch [{start}:{end}] completed in {time.time() - tb:.2f}s. Candidate count: {len(candidates)}", flush=True)

t_block = time.time() - t0
print(f"\nWord+Char Blocking completed in {t_block:.2f}s! Total candidate pairs: {len(candidates)}")

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

print(f"Blocking Recall: {found_matches} / {total_matches} ({100*found_matches/max(1, total_matches):.2f}%)")

import pandas as pd
import numpy as np
import time
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import rapidfuzz
from rapidfuzz import fuzz

print("1. Loading sample dataset...", flush=True)
t0 = time.time()
s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", dtype=str, nrows=20000).fillna("")
s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t", dtype=str, nrows=50000).fillna("")
s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t", dtype=str, nrows=50000).fillna("")
gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")

pool = pd.concat([s2, s3], ignore_index=True)
print(f"Data loaded in {time.time()-t0:.2f}s. S1={len(s1)}, Pool={len(pool)}", flush=True)

gt_s1_ids = set(s1['entity_id'])
gt_filtered = gt[gt['source1_entity_id'].isin(gt_s1_ids)]
eval_gt_map = {}
valid_target_set = set(pool['entity_id'])
for _, row in gt_filtered.iterrows():
    m = set(x.strip() for x in str(row['matched_entity_ids']).split(',') if x.strip()).intersection(valid_target_set)
    if m:
        eval_gt_map[row['source1_entity_id']] = m

s1['text'] = (s1['business_name'] + " " + s1['business_address']).str.lower()
pool['text'] = (pool['business_name'] + " " + pool['business_address']).str.lower()

print("2. Vectorizing text with TF-IDF (char 3-5 n-grams, min_df=2)...", flush=True)
t0 = time.time()
vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, max_features=50000, sublinear_tf=True, dtype=np.float32)
tfidf_pool = vec.fit_transform(pool['text'])
tfidf_s1 = vec.transform(s1['text'])
print(f"TF-IDF matrices built in {time.time()-t0:.2f}s. Pool: {tfidf_pool.shape}, S1: {tfidf_s1.shape}", flush=True)

print("3. Performing Fast Batch Matrix Multiplication Blocking...", flush=True)
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
        row_s1_id = s1_ids[start + i]
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
                candidates.append((row_s1_id, pool_ids[idx[b]], float(d[b])))
                
    print(f"  Processed batch [{start}:{end}] in {time.time()-tb:.2f}s. Candidates count: {len(candidates)}", flush=True)

t_block = time.time() - t0
print(f"Blocking completed in {t_block:.2f}s! Total candidate pairs: {len(candidates)}", flush=True)

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

print(f"Blocking Recall: {found_matches} / {total_matches} ({100*found_matches/max(1, total_matches):.2f}%)", flush=True)

print("4. Testing RapidFuzz Feature Engineering...", flush=True)
t0 = time.time()
s1_dict = s1.set_index('entity_id').to_dict('index')
pool_dict = pool.set_index('entity_id').to_dict('index')

names1 = [s1_dict[c[0]]['business_name'] for c in candidates]
names2 = [pool_dict[c[1]]['business_name'] for c in candidates]
addrs1 = [s1_dict[c[0]]['business_address'] for c in candidates]
addrs2 = [pool_dict[c[1]]['business_address'] for c in candidates]

feat_names = [fuzz.ratio(n1, n2) for n1, n2 in zip(names1, names2)]
feat_ts_names = [fuzz.token_sort_ratio(n1, n2) for n1, n2 in zip(names1, names2)]
feat_addrs = [fuzz.ratio(a1, a2) for a1, a2 in zip(addrs1, addrs2)]

t_feat = time.time() - t0
print(f"Feature engineering for {len(candidates)} pairs completed in {t_feat:.2f}s ({len(candidates)/t_feat:.0f} pairs/sec)!", flush=True)

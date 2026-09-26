import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer

print("Loading data slice...")
s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", dtype=str, nrows=30000).fillna("")
s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t", dtype=str, nrows=500000).fillna("")
s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t", dtype=str, nrows=500000).fillna("")
gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")

pool = pd.concat([s2, s3], ignore_index=True)

s1['clean_text'] = (s1['business_name'] + " " + s1['business_address']).str.lower()
pool['clean_text'] = (pool['business_name'] + " " + pool['business_address']).str.lower()

print(f"Data ready: S1={len(s1)}, Pool={len(pool)}")

print("Testing TfidfVectorizer with max_df=0.25, min_df=5, ngram_range=(3,5)...")
t0 = time.time()
sample_text = pool['clean_text'].sample(n=min(300000, len(pool)), random_state=42)

vec = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(3, 5),
    min_df=5,
    max_df=0.25,  # Prunes stop n-grams appearing in >25% of documents!
    max_features=50000,
    sublinear_tf=True,
    dtype=np.float32
)

vec.fit(sample_text)
tfidf_pool = vec.transform(pool['clean_text'])
tfidf_s1 = vec.transform(s1['clean_text'])
tfidf_pool_T = tfidf_pool.T.tocsc()

print(f"TF-IDF built in {time.time() - t0:.2f}s. Pool shape: {tfidf_pool.shape}, S1 shape: {tfidf_s1.shape}")

t0 = time.time()
batch_size = 1000
top_k = 20
pool_ids = pool['entity_id'].values
s1_ids = s1['entity_id'].values

candidates = []
n_s1 = tfidf_s1.shape[0]

for start in range(0, n_s1, batch_size):
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
                
    if (end // batch_size) % 5 == 0 or end == n_s1:
        print(f"  Progress: {end}/{n_s1} entities processed in {time.time() - t0:.2f}s. Candidates count: {len(candidates)}", flush=True)

print(f"Test completed in {time.time() - t0:.2f}s. Total candidate pairs: {len(candidates)}")

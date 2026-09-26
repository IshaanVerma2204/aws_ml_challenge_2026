import pandas as pd
import numpy as np
import time
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import rapidfuzz
from rapidfuzz import fuzz

print("Loading sample data...")
t0 = time.time()
s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", dtype=str, nrows=50000).fillna("")
s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t", dtype=str, nrows=100000).fillna("")
s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t", dtype=str, nrows=100000).fillna("")
gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")

pool = pd.concat([s2, s3], ignore_index=True)

gt_s1_ids = set(s1['entity_id'])
gt_filtered = gt[gt['source1_entity_id'].isin(gt_s1_ids)]
gt_map = {}
for _, row in gt_filtered.iterrows():
    matches = set(x.strip() for x in str(row['matched_entity_ids']).split(',') if x.strip())
    gt_map[row['source1_entity_id']] = matches

valid_target_set = set(pool['entity_id'])
eval_gt_map = {s1_id: matches.intersection(valid_target_set) for s1_id, matches in gt_map.items() if matches.intersection(valid_target_set)}

s1['text'] = (s1['business_name'] + " " + s1['business_address']).str.lower()
pool['text'] = (pool['business_name'] + " " + pool['business_address']).str.lower()

t0 = time.time()
print("Fitting TF-IDF Vectorizer...")
vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=25000, sublinear_tf=True, dtype=np.float32)
tfidf_pool = vec.fit_transform(pool['text'])
tfidf_s1 = vec.transform(s1['text'])
print(f"TF-IDF matrix built in {time.time() - t0:.2f}s.")

t0 = time.time()
top_k = 20
pool_ids = pool['entity_id'].values
s1_ids = s1['entity_id'].values

# Convert to PyTorch CSR / Sparse Tensors
# Sparse matrix multiplication in PyTorch
pool_sparse = torch.sparse_csr_tensor(
    torch.from_numpy(tfidf_pool.indptr.astype(np.int64)),
    torch.from_numpy(tfidf_pool.indices.astype(np.int64)),
    torch.from_numpy(tfidf_pool.data),
    size=tfidf_pool.shape
).to_sparse_csc() # Transpose pool

s1_csr = torch.sparse_csr_tensor(
    torch.from_numpy(tfidf_s1.indptr.astype(np.int64)),
    torch.from_numpy(tfidf_s1.indices.astype(np.int64)),
    torch.from_numpy(tfidf_s1.data),
    size=tfidf_s1.shape
)

print(f"Converted to PyTorch Tensors in {time.time() - t0:.2f}s.")

t0 = time.time()
batch_size = 2000
candidates = []
n_s1 = tfidf_s1.shape[0]

# Pre-convert pool to dense float32 tensor (200k x 25k float32 = 20MB)
pool_dense_t = torch.from_numpy(tfidf_pool.toarray()).t() # 25000 x 200000

for start in range(0, n_s1, batch_size):
    end = min(start + batch_size, n_s1)
    s1_batch_dense = torch.from_numpy(tfidf_s1[start:end].toarray()) # batch x 25000
    
    # Blazing fast PyTorch CPU matrix multiplication
    sims = torch.mm(s1_batch_dense, pool_dense_t) # batch x 200000
    
    # PyTorch C++ topk
    val_topk, idx_topk = torch.topk(sims, k=top_k, dim=1)
    
    val_np = val_topk.numpy()
    idx_np = idx_topk.numpy()
    
    for i in range(end - start):
        row_s1_id = s1_ids[start + i]
        row_vals = val_np[i]
        row_idxs = idx_np[i]
        for v, idx in zip(row_vals, row_idxs):
            if v >= 0.15:
                candidates.append((row_s1_id, pool_ids[idx], float(v)))

t_block = time.time() - t0
print(f"PyTorch topk blocking completed in {t_block:.2f}s! Total candidate pairs: {len(candidates)}")

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

print(f"Blocking recall: {found_matches} / {total_matches} ({100*found_matches/max(1, total_matches):.2f}%)")

print("\nTesting RapidFuzz feature extraction speed...")
t0 = time.time()
s1_dict = s1.set_index('entity_id').to_dict('index')
pool_dict = pool.set_index('entity_id').to_dict('index')

cands_s1 = [s1_dict[c[0]] for c in candidates]
cands_pool = [pool_dict[c[1]] for c in candidates]

names1 = [r['business_name'] for r in cands_s1]
names2 = [r['business_name'] for r in cands_pool]
addrs1 = [r['business_address'] for r in cands_s1]
addrs2 = [r['business_address'] for r in cands_pool]

r_names = [fuzz.ratio(n1, n2) for n1, n2 in zip(names1, names2)]
r_ts_names = [fuzz.token_sort_ratio(n1, n2) for n1, n2 in zip(names1, names2)]
r_addrs = [fuzz.ratio(a1, a2) for a1, a2 in zip(addrs1, addrs2)]
r_ts_addrs = [fuzz.token_sort_ratio(a1, a2) for a1, a2 in zip(addrs1, addrs2)]

t_feat = time.time() - t0
print(f"Computed features for {len(candidates)} candidate pairs in {t_feat:.2f}s ({len(candidates)/t_feat:.0f} pairs/sec)!")

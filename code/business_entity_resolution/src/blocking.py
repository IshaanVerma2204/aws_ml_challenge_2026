"""
blocking.py  --  GPU-accelerated candidate generation via TF-IDF + LSA + CUDA matmul.

STRATEGY
--------
1. TF-IDF (char n-gram) + TruncatedSVD -> 128-dim LSA vectors
2. Similarity search via batched matrix multiply:
   - GPU path (CUDA): load pool_lsa to GPU, process S1 in batches -> torch.topk
   - CPU fallback:    small S1 batches x small pool chunks -> numpy matmul

GPU SPEEDUP (RTX 4050, 4GB VRAM)
---------------------------------
Test US: 663K S1, 3.8M pool
  CPU: 382 chunks x 663K S1 x 128 dim = ~4h stuck
  GPU: pool_lsa (1.95GB) + S1 batches (5K x 128 = 2.6MB each)
       -> (5K, 128) @ (128, 3.8M) = (5K, 3.8M) = 76MB per batch
       -> torch.topk on GPU directly
       -> 133 batches x ~0.5s = ~1 min total

Memory per GPU batch (s1_gpu_batch=5000, pool=3.8M):
  pool_lsa on GPU:  3.8M x 128 x 4B = 1.95 GB  [resident]
  s1_batch on GPU:  5K   x 128 x 4B = 2.6  MB
  sim on GPU:       5K   x 3.8M x 4B = 76   MB  [freed each batch]
  Total peak:                        ~ 2.1  GB  (fits in 4GB VRAM)
"""

import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# ---------------------------------------------------------------------------#
#  GPU / device setup                                                        #
# ---------------------------------------------------------------------------#

def _get_device():
    """Return torch device: CUDA if available, else CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  [GPU] Using {gpu}  ({vram:.1f} GB VRAM)", flush=True)
            return torch.device('cuda'), torch
        else:
            print("  [CPU] CUDA not available, using CPU numpy path.", flush=True)
            return None, torch
    except ImportError:
        print("  [CPU] PyTorch not installed, using CPU numpy path.", flush=True)
        return None, None


# ---------------------------------------------------------------------------#
#  Pool LSA builder                                                          #
# ---------------------------------------------------------------------------#

def _build_pool_lsa(vectorizer, svd, texts, chunk_size=200_000, prefix=''):
    """
    Stream pool texts through TF-IDF -> SVD in chunks.
    Never stores the full sparse TF-IDF matrix.
    Returns normalised float32 numpy array (n_pool, n_components).
    """
    import scipy.sparse as sp
    n     = len(texts)
    parts = []
    for i in range(0, n, chunk_size):
        chunk = texts.iloc[i:i + chunk_size] if hasattr(texts, 'iloc') \
                else texts[i:i + chunk_size]
        tfidf = vectorizer.transform(chunk).astype(np.float32)
        lsa   = svd.transform(tfidf).astype(np.float32)
        parts.append(lsa)
        if prefix and (i // chunk_size + 1) % 5 == 0:
            pct = min(100, 100 * (i + chunk_size) / n)
            print(f"  {prefix} LSA {pct:.0f}%...", flush=True)
    mat = np.vstack(parts)
    return normalize(mat, norm='l2').astype(np.float32)


# ---------------------------------------------------------------------------#
#  Top-k search  (GPU-first, CPU fallback)                                   #
# ---------------------------------------------------------------------------#

def _topk_search_gpu(s1_lsa, pool_lsa, top_k, min_sim, s1_gpu_batch,
                     device, torch, country, pool_ids, s1_ids):
    """
    GPU path (FP16): load pool_lsa to GPU in float16, stream S1 in safe batches.
    Each batch: (batch, 128) matmul (128, n_pool) in FP16 -> topk on GPU.
    Uses ~2.4 GB peak VRAM on RTX 4050 (safely within 6.0 GB total).
    """
    n_s1   = len(s1_ids)
    n_pool = len(pool_ids)

    # 1. Load pool and S1 in FP16 onto GPU
    pool_gpu = torch.from_numpy(pool_lsa).to(device=device, dtype=torch.float16)  # (n_pool, 128)
    s1_gpu   = torch.from_numpy(s1_lsa).to(device=device, dtype=torch.float16)    # (n_s1, 128)

    pool_vram_gb = pool_gpu.numel() * 2 / 1e9

    # Calculate safe batch size so sim buffer does not exceed 800 MB
    safe_sim_bytes = 800 * 1024 * 1024
    batch_size = max(16, min(128, int(safe_sim_bytes / (n_pool * 2))))
    sim_batch_mb = batch_size * n_pool * 2 / 1e6
    n_batches = (n_s1 + batch_size - 1) // batch_size

    print(f"  [{country}] GPU search (FP16): {n_batches:,} S1 batches (size {batch_size}) x {n_pool:,} pool "
          f"(pool: {pool_vram_gb:.2f} GB VRAM, sim buffer: {sim_batch_mb:.0f} MB)", flush=True)

    cand_s1_list = []
    cand_pool_list = []
    cand_sims_list = []
    t0 = time.time()
    last_print = t0
    actual_k = min(top_k, n_pool)

    for b_idx, s1_start in enumerate(range(0, n_s1, batch_size)):
        s1_end   = min(s1_start + batch_size, n_s1)
        s1_batch = s1_gpu[s1_start:s1_end]             # (batch, 128)

        # FP16 similarity matrix multiplication on GPU tensor cores
        sim = s1_batch @ pool_gpu.T                     # (batch, n_pool) float16

        # Top-k directly on GPU
        top_vals, top_idx = torch.topk(sim, k=actual_k, dim=1, largest=True, sorted=False)
        del sim

        # Move only top-k results (batch, top_k) to CPU numpy
        top_vals_np = top_vals.float().cpu().numpy()
        top_idx_np  = top_idx.cpu().numpy()
        del top_vals, top_idx

        # Vectorized candidate filtering
        valid_rows, valid_cols = np.where(top_vals_np >= min_sim)
        if len(valid_rows) > 0:
            batch_s1_ids = s1_ids[s1_start:s1_end]
            cand_s1   = batch_s1_ids[valid_rows]
            cand_pool = pool_ids[top_idx_np[valid_rows, valid_cols]]
            cand_sims = top_vals_np[valid_rows, valid_cols]

            cand_s1_list.extend(cand_s1.tolist())
            cand_pool_list.extend(cand_pool.tolist())
            cand_sims_list.extend(cand_sims.tolist())

        now = time.time()
        if (now - last_print >= 10.0) or (s1_end == n_s1):
            elapsed = now - t0
            rate    = s1_end / max(elapsed, 1e-3)
            eta_sec = (n_s1 - s1_end) / max(rate, 1e-3)
            print(f"  [{country}] S1 {s1_end:,}/{n_s1:,} ({100*s1_end/n_s1:.1f}%) "
                  f"| {rate:.0f} S1/s | ETA: {eta_sec:.0f}s | candidates: {len(cand_s1_list):,}", flush=True)
            last_print = now

    del pool_gpu, s1_gpu
    torch.cuda.empty_cache()
    
    return pd.DataFrame({
        'source1_entity_id': cand_s1_list,
        'candidate_entity_id': cand_pool_list,
        'tfidf_sim': cand_sims_list
    })

def _topk_search_faiss(s1_embs, pool_embs, top_k, min_sim, country, pool_ids, s1_ids):
    """
    FAISS approximate nearest-neighbor search.
    Requires faiss-gpu or faiss-cpu to be installed.
    """
    print(f"  [{country}] FAISS search: Indexing {len(pool_embs):,} items...", flush=True)
    d = pool_embs.shape[1]
    
    # Use inner product (cosine similarity since vectors are normalized)
    index = faiss.IndexFlatIP(d)
    
    if faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        
    index.add(pool_embs)
    print(f"  [{country}] FAISS search: Querying {len(s1_embs):,} items...", flush=True)
    D, I = index.search(s1_embs, top_k)
    
    candidates = []
    for i in range(len(s1_embs)):
        for j in range(top_k):
            if D[i, j] >= min_sim and I[i, j] >= 0:
                candidates.append({
                    'source1_entity_id':   s1_ids[i],
                    'candidate_entity_id': pool_ids[I[i, j]],
                    'tfidf_sim':           float(D[i, j])
                })
    return candidates


def _topk_search_cpu(s1_lsa, pool_lsa, top_k, min_sim,
                     s1_cpu_batch, pool_chunk_size, country, pool_ids, s1_ids):
    """
    CPU fallback: nested batches keeping peak allocation ~200-400 MB.
    sim_chunk = (s1_cpu_batch, pool_chunk_size) x 4B
    """
    n_s1   = len(s1_ids)
    n_pool = len(pool_ids)
    n_pool_chunks = (n_pool + pool_chunk_size - 1) // pool_chunk_size
    n_s1_batches  = (n_s1  + s1_cpu_batch  - 1) // s1_cpu_batch
    sim_mb = s1_cpu_batch * pool_chunk_size * 4 / 1e6
    print(f"  [{country}] CPU search: {n_s1_batches} S1 batches x "
          f"{n_pool_chunks} pool chunks (sim/iter~{sim_mb:.0f} MB)", flush=True)

    # top-k accumulators
    top_sims = np.full((n_s1, top_k), -1.0, dtype=np.float32)
    top_idxs = np.full((n_s1, top_k), -1,   dtype=np.int32)

    t0 = time.time()
    for s1_start in range(0, n_s1, s1_cpu_batch):
        s1_end  = min(s1_start + s1_cpu_batch, n_s1)
        s1_mat  = s1_lsa[s1_start:s1_end]
        n_s1_b  = s1_end - s1_start
        b_sims  = np.full((n_s1_b, top_k), -1.0, dtype=np.float32)
        b_idxs  = np.full((n_s1_b, top_k), -1,   dtype=np.int32)

        for p_start in range(0, n_pool, pool_chunk_size):
            p_end      = min(p_start + pool_chunk_size, n_pool)
            chunk_size = p_end - p_start

            pool_T = pool_lsa[p_start:p_end].T          # (128, chunk)
            sim    = (s1_mat @ pool_T).astype(np.float32)  # (batch, chunk)
            del pool_T

            local_k = min(top_k, chunk_size)
            if chunk_size <= top_k:
                local_idx  = np.tile(np.arange(chunk_size, dtype=np.int32), (n_s1_b, 1))
                local_sims = sim
            else:
                local_idx  = np.argpartition(-sim, local_k, axis=1)[:, :local_k].astype(np.int32)
                local_sims = np.take_along_axis(sim, local_idx, axis=1)
            del sim

            local_idx_g = local_idx + p_start
            merged_s = np.concatenate([b_sims, local_sims.astype(np.float32)], axis=1)
            merged_i = np.concatenate([b_idxs, local_idx_g],                   axis=1)
            best     = np.argpartition(-merged_s, top_k, axis=1)[:, :top_k]
            b_sims   = np.take_along_axis(merged_s, best, axis=1)
            b_idxs   = np.take_along_axis(merged_i, best, axis=1)

        top_sims[s1_start:s1_end] = b_sims
        top_idxs[s1_start:s1_end] = b_idxs

        elapsed = time.time() - t0
        rate    = s1_end / max(elapsed, 1e-3)
        print(f"  [{country}] S1 {s1_end:,}/{n_s1:,} ({100*s1_end/n_s1:.0f}%)  "
              f"{rate:.0f} S1/s", flush=True)

    candidates = []
    for i in range(n_s1):
        s1_id = s1_ids[i]
        for j in range(top_k):
            sv = float(top_sims[i, j])
            pi = int(top_idxs[i, j])
            if sv >= min_sim and pi >= 0:
                candidates.append({
                    'source1_entity_id':   s1_id,
                    'candidate_entity_id': pool_ids[pi],
                    'tfidf_sim':           sv
                })
    return candidates


# ---------------------------------------------------------------------------#
#  Main blocking function                                                    #
# ---------------------------------------------------------------------------#

def _run_blocking_pass(s1_c, pool_c, country, pass_name, top_k, min_sim, n_components, 
                       use_gpu, device, torch_mod, s1_gpu_batch, s1_cpu_batch, pool_chunk_size,
                       use_semantic=False):
    import time
    t0 = time.time()
    
    if use_semantic and SENTENCE_TRANSFORMERS_AVAILABLE:
        print(f"    [{pass_name}] Embedding {len(pool_c):,} pool items with all-MiniLM-L6-v2...", flush=True)
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if use_gpu else 'cpu')
        pool_lsa = model.encode(pool_c['clean_text'].tolist(), batch_size=256, show_progress_bar=False, normalize_embeddings=True)
        print(f"    [{pass_name}] Embedding {len(s1_c):,} S1 items...", flush=True)
        s1_lsa = model.encode(s1_c['clean_text'].tolist(), batch_size=256, show_progress_bar=False, normalize_embeddings=True)
        pool_lsa = pool_lsa.astype(np.float32)
        s1_lsa = s1_lsa.astype(np.float32)
    else:
        sample_size = min(300_000, len(pool_c))
        sample_text = pool_c['clean_text'].sample(n=sample_size, random_state=42)
        vectorizer  = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(3, 4), min_df=10, max_df=0.3,
            max_features=10_000, sublinear_tf=True, dtype=np.float32
        )
        vectorizer.fit(sample_text)
        print(f"    [{pass_name}] TF-IDF vocab: {len(vectorizer.vocabulary_)}", flush=True)

        t_svd = time.time()
        tfidf_sample = vectorizer.transform(sample_text).astype(np.float32)
        svd = TruncatedSVD(n_components=n_components, random_state=42, n_iter=5)
        svd.fit(tfidf_sample)
        del tfidf_sample
        print(f"    [{pass_name}] SVD done in {time.time()-t_svd:.1f}s  "
              f"(var: {svd.explained_variance_ratio_.sum()*100:.1f}%)", flush=True)

        t_pool  = time.time()
        pool_lsa = _build_pool_lsa(vectorizer, svd, pool_c['clean_text'],
                                   prefix=f'[{country}-{pass_name}]')
        print(f"    [{pass_name}] Pool LSA done in {time.time()-t_pool:.1f}s.", flush=True)

        tfidf_s1 = vectorizer.transform(s1_c['clean_text']).astype(np.float32)
        s1_lsa   = normalize(svd.transform(tfidf_s1).astype(np.float32), norm='l2')
        del tfidf_s1

    pool_ids = pool_c['entity_id'].values
    s1_ids   = s1_c['entity_id'].values

    if FAISS_AVAILABLE:
        cands = _topk_search_faiss(s1_lsa, pool_lsa, top_k, min_sim, country, pool_ids, s1_ids)
    elif use_gpu:
        pool_gb   = pool_lsa.nbytes / 1e9
        vram_free = torch_mod.cuda.get_device_properties(0).total_memory / 1e9 - 0.5
        if pool_gb < vram_free * 0.8:
            cands = _topk_search_gpu(
                s1_lsa, pool_lsa, top_k, min_sim,
                s1_gpu_batch, device, torch_mod,
                country, pool_ids, s1_ids
            )
        else:
            pool_chunk_gpu = max(500_000, int(vram_free * 0.4 * 1e9 / (n_components * 4)))
            cands_list = []
            n_pool = len(pool_ids)
            for p_start in range(0, n_pool, pool_chunk_gpu):
                p_end   = min(p_start + pool_chunk_gpu, n_pool)
                p_chunk = pool_lsa[p_start:p_end]
                chunk_cands = _topk_search_gpu(
                    s1_lsa, p_chunk, top_k, min_sim,
                    s1_gpu_batch, device, torch_mod,
                    country, pool_ids[p_start:p_end], s1_ids
                )
                cands_list.append(chunk_cands)
            cands = pd.concat(cands_list, ignore_index=True) if cands_list else pd.DataFrame()
    else:
        cands = _topk_search_cpu(
            s1_lsa, pool_lsa, top_k, min_sim,
            s1_cpu_batch, pool_chunk_size,
            country, pool_ids, s1_ids
        )

    del pool_lsa, s1_lsa
    print(f"    [{pass_name}] Pass done in {time.time()-t0:.1f}s.  "
          f"Candidates: {len(cands):,}", flush=True)
    return cands

def generate_candidates(df_s1, df_s2, df_s3, top_k=20, min_sim=0.15,
                        n_components=128,
                        s1_gpu_batch=5_000,
                        s1_cpu_batch=3_000,
                        pool_chunk_size=2_000,
                        use_semantic=False):
    df_pool = pd.concat([df_s2, df_s3], ignore_index=True)
    df_s1   = df_s1.copy()
    df_pool = df_pool.copy()

    device, torch_mod = _get_device()
    use_gpu = device is not None

    countries      = df_s1['country'].unique()
    all_candidates = []

    print(f"Starting multi-pass blocking across {len(countries)} countries: {list(countries)}...", flush=True)
    t0_all = time.time()

    for country in countries:
        s1_c   = df_s1[df_s1['country']   == country].copy()
        pool_c = df_pool[df_pool['country'] == country].copy()

        if s1_c.empty or pool_c.empty:
            print(f"  Skipping '{country}'", flush=True)
            continue

        print(f"\nBlocking for country '{country}' "
              f"(S1: {len(s1_c):,}, Pool: {len(pool_c):,})...", flush=True)
        t0 = time.time()
        
        country_cands = []
        
        passes = [
            ("name_only", s1_c['business_name'].str.lower(), pool_c['business_name'].str.lower()),
            ("address_only", s1_c['business_address'].str.lower(), pool_c['business_address'].str.lower()),
            ("combined", (s1_c['business_name'] + ' ' + s1_c['business_address']).str.lower(), 
                         (pool_c['business_name'] + ' ' + pool_c['business_address']).str.lower())
        ]
        
        for pass_name, s1_text, pool_text in passes:
            print(f"  --- Pass: {pass_name} ---", flush=True)
            s1_c['clean_text'] = s1_text
            pool_c['clean_text'] = pool_text
            
            # Use smaller top_k for individual passes to control candidate explosion
            pass_top_k = top_k if pass_name == "combined" else max(1, top_k // 2)
            
            pass_cands = _run_blocking_pass(
                s1_c, pool_c, country, pass_name, pass_top_k, min_sim, n_components,
                use_gpu, device, torch_mod, s1_gpu_batch, s1_cpu_batch, pool_chunk_size,
                use_semantic=use_semantic
            )
            country_cands.append(pass_cands)
            
        df_cands = pd.concat(country_cands, ignore_index=True) if country_cands else pd.DataFrame()
        if not df_cands.empty:
            df_cands = df_cands.sort_values('tfidf_sim', ascending=False)
            df_cands = df_cands.drop_duplicates(subset=['source1_entity_id', 'candidate_entity_id'], keep='first')
            all_candidates.append(df_cands)
            print(f"  Country '{country}' total deduplicated candidates: {len(df_cands):,}", flush=True)
        
        print(f"  Country '{country}' done in {time.time()-t0:.1f}s.", flush=True)

    print(f"\nTotal blocking done in {time.time()-t0_all:.1f}s.", flush=True)
    final_df = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()
    print(f"Total: {len(final_df):,}", flush=True)
    return final_df


# ---------------------------------------------------------------------------#
#  Output formatting                                                         #
# ---------------------------------------------------------------------------#

def format_candidate_pairs(df_s1, df_candidates_flat):
    """
    Format flat candidates into TSV structure.
    Guarantees one row per S1 entity.
    """
    if df_candidates_flat.empty:
        cand_dict = {}
    else:
        grouped = df_candidates_flat.groupby('source1_entity_id')['candidate_entity_id'].apply(
            lambda x: ','.join(list(dict.fromkeys(x)))
        )
        cand_dict = grouped.to_dict()

    res = []
    for s1_id in df_s1['entity_id']:
        res.append({
            'source1_entity_id':    s1_id,
            'candidate_entity_ids': cand_dict.get(s1_id, '')
        })
    return pd.DataFrame(res)

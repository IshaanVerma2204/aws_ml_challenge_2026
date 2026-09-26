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

    candidates = []
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

            for sid, pid, sm in zip(cand_s1, cand_pool, cand_sims):
                candidates.append({
                    'source1_entity_id':   sid,
                    'candidate_entity_id': pid,
                    'tfidf_sim':           float(sm)
                })

        now = time.time()
        if (now - last_print >= 10.0) or (s1_end == n_s1):
            elapsed = now - t0
            rate    = s1_end / max(elapsed, 1e-3)
            eta_sec = (n_s1 - s1_end) / max(rate, 1e-3)
            print(f"  [{country}] S1 {s1_end:,}/{n_s1:,} ({100*s1_end/n_s1:.1f}%) "
                  f"| {rate:.0f} S1/s | ETA: {eta_sec:.0f}s | candidates: {len(candidates):,}", flush=True)
            last_print = now

    del pool_gpu, s1_gpu
    torch.cuda.empty_cache()
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

def generate_candidates(df_s1, df_s2, df_s3, top_k=20, min_sim=0.15,
                        n_components=128,
                        s1_gpu_batch=5_000,
                        s1_cpu_batch=3_000,
                        pool_chunk_size=2_000):
    """
    Generate top-k cosine-LSA candidates per country.

    GPU path  (CUDA available):
      pool_lsa loaded to GPU once. S1 processed in batches of s1_gpu_batch.
      sim = (batch, 128) @ (128, n_pool) -> topk on GPU.
      RTX 4050: ~1-2 min per country for 663K S1 x 3.8M pool.

    CPU path  (no CUDA):
      Nested S1-batch x pool-chunk matmul, ~200 MB per iteration.
      Slower but memory-safe for any system.
    """
    import scipy.sparse as sp

    df_pool = pd.concat([df_s2, df_s3], ignore_index=True)
    df_s1   = df_s1.copy()
    df_pool = df_pool.copy()

    df_s1['clean_text']   = (df_s1['business_name']   + ' ' + df_s1['business_address']).str.lower()
    df_pool['clean_text'] = (df_pool['business_name'] + ' ' + df_pool['business_address']).str.lower()

    device, torch_mod = _get_device()
    use_gpu = device is not None

    countries      = df_s1['country'].unique()
    all_candidates = []

    print(f"Starting blocking across {len(countries)} countries: {list(countries)}...", flush=True)
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

        # ------------------------------------------------------------------ #
        # 1. TF-IDF                                                          #
        # ------------------------------------------------------------------ #
        sample_size = min(300_000, len(pool_c))
        sample_text = pool_c['clean_text'].sample(n=sample_size, random_state=42)
        vectorizer  = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(3, 4), min_df=10, max_df=0.3,
            max_features=10_000, sublinear_tf=True, dtype=np.float32
        )
        vectorizer.fit(sample_text)
        print(f"  TF-IDF vocab: {len(vectorizer.vocabulary_)}", flush=True)

        # ------------------------------------------------------------------ #
        # 2. SVD                                                             #
        # ------------------------------------------------------------------ #
        t_svd = time.time()
        tfidf_sample = vectorizer.transform(sample_text).astype(np.float32)
        svd = TruncatedSVD(n_components=n_components, random_state=42, n_iter=5)
        svd.fit(tfidf_sample)
        del tfidf_sample
        print(f"  SVD done in {time.time()-t_svd:.1f}s  "
              f"(var: {svd.explained_variance_ratio_.sum()*100:.1f}%)", flush=True)

        # ------------------------------------------------------------------ #
        # 3. Pool LSA  (chunked, never stores full sparse TF-IDF)            #
        # ------------------------------------------------------------------ #
        print(f"  Building pool LSA ({len(pool_c):,} rows)...", flush=True)
        t_pool  = time.time()
        pool_lsa = _build_pool_lsa(vectorizer, svd, pool_c['clean_text'],
                                   prefix=f'[{country}]')
        print(f"  Pool LSA done in {time.time()-t_pool:.1f}s.  "
              f"{pool_lsa.shape}  ({pool_lsa.nbytes/1e9:.2f} GB)", flush=True)

        # ------------------------------------------------------------------ #
        # 4. S1 LSA                                                          #
        # ------------------------------------------------------------------ #
        tfidf_s1 = vectorizer.transform(s1_c['clean_text']).astype(np.float32)
        s1_lsa   = normalize(svd.transform(tfidf_s1).astype(np.float32), norm='l2')
        del tfidf_s1
        print(f"  S1 LSA done.  {s1_lsa.shape}", flush=True)

        pool_ids = pool_c['entity_id'].values
        s1_ids   = s1_c['entity_id'].values

        # ------------------------------------------------------------------ #
        # 5. Top-k search                                                    #
        # ------------------------------------------------------------------ #
        if use_gpu:
            # Check VRAM: need pool_lsa + s1_batch + sim
            pool_gb   = pool_lsa.nbytes / 1e9
            vram_free = torch_mod.cuda.get_device_properties(0).total_memory / 1e9 - 0.5
            if pool_gb < vram_free * 0.8:
                # Pool fits in VRAM
                cands = _topk_search_gpu(
                    s1_lsa, pool_lsa, top_k, min_sim,
                    s1_gpu_batch, device, torch_mod,
                    country, pool_ids, s1_ids
                )
            else:
                # Pool too large for VRAM — chunk pool on GPU
                print(f"  Pool ({pool_gb:.2f} GB) > VRAM budget ({vram_free*0.8:.2f} GB). "
                      f"Chunking pool across GPU calls...", flush=True)
                # Increase s1_gpu_batch to scan pool in chunks
                pool_chunk_gpu = max(500_000, int(vram_free * 0.4 * 1e9 / (n_components * 4)))
                cands = []
                n_pool = len(pool_ids)
                t_gpu  = time.time()
                for p_start in range(0, n_pool, pool_chunk_gpu):
                    p_end   = min(p_start + pool_chunk_gpu, n_pool)
                    p_chunk = pool_lsa[p_start:p_end]
                    chunk_cands = _topk_search_gpu(
                        s1_lsa, p_chunk, top_k, min_sim,
                        s1_gpu_batch, device, torch_mod,
                        country, pool_ids[p_start:p_end], s1_ids
                    )
                    cands.extend(chunk_cands)
                    print(f"  Pool chunk {p_end:,}/{n_pool:,} done "
                          f"in {time.time()-t_gpu:.0f}s", flush=True)
        else:
            cands = _topk_search_cpu(
                s1_lsa, pool_lsa, top_k, min_sim,
                s1_cpu_batch, pool_chunk_size,
                country, pool_ids, s1_ids
            )

        del pool_lsa, s1_lsa

        print(f"  Country '{country}' done in {time.time()-t0:.1f}s.  "
              f"Candidates: {len(cands):,}", flush=True)
        all_candidates.extend(cands)

    print(f"\nTotal blocking done in {time.time()-t0_all:.1f}s.  "
          f"Total: {len(all_candidates):,}", flush=True)
    return pd.DataFrame(all_candidates)


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

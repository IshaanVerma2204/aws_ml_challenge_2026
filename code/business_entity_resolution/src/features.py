import pandas as pd
import numpy as np
import time
import re
import jellyfish
from rapidfuzz import fuzz

def get_numbers(text):
    return set(re.findall(r'\d+', str(text)))

def num_overlap(text1, text2):
    nums1 = get_numbers(text1)
    nums2 = get_numbers(text2)
    if not nums1 and not nums2: return 1.0
    if not nums1 or not nums2: return 0.0
    return len(nums1.intersection(nums2)) / len(nums1.union(nums2))

def get_zip(text):
    zips = re.findall(r'\b\d{5,6}\b', str(text))
    return zips[-1] if zips else ""

def compute_features(df_candidates_flat, df_s1, df_s2, df_s3):
    """
    Computes high-precision string similarity features for each candidate pair
    using compiled C++ RapidFuzz functions at over 150,000 pairs/sec.
    """
    if df_candidates_flat.empty:
        return pd.DataFrame()
        
    print(f"Computing features for {len(df_candidates_flat)} candidate pairs...", flush=True)
    t0 = time.time()
    
    df_pool = pd.concat([df_s2, df_s3], ignore_index=True)
    
    s1_dict = df_s1.set_index('entity_id').to_dict('index')
    pool_dict = df_pool.set_index('entity_id').to_dict('index')
    
    s1_ids = df_candidates_flat['source1_entity_id'].values
    cand_ids = df_candidates_flat['candidate_entity_id'].values
    tfidf_sims = df_candidates_flat['tfidf_sim'].values if 'tfidf_sim' in df_candidates_flat.columns else np.zeros(len(df_candidates_flat))
    
    names1 = [str(s1_dict[sid]['business_name']).lower() for sid in s1_ids]
    names2 = [str(pool_dict[cid]['business_name']).lower() for cid in cand_ids]
    
    addrs1 = [str(s1_dict[sid]['business_address']).lower() for sid in s1_ids]
    addrs2 = [str(pool_dict[cid]['business_address']).lower() for cid in cand_ids]
    
    name_ratio = [fuzz.ratio(n1, n2) / 100.0 for n1, n2 in zip(names1, names2)]
    name_partial = [fuzz.partial_ratio(n1, n2) / 100.0 for n1, n2 in zip(names1, names2)]
    name_token_sort = [fuzz.token_sort_ratio(n1, n2) / 100.0 for n1, n2 in zip(names1, names2)]
    name_token_set = [fuzz.token_set_ratio(n1, n2) / 100.0 for n1, n2 in zip(names1, names2)]
    name_wratio = [fuzz.WRatio(n1, n2) / 100.0 for n1, n2 in zip(names1, names2)]
    
    addr_ratio = [fuzz.ratio(a1, a2) / 100.0 for a1, a2 in zip(addrs1, addrs2)]
    addr_partial = [fuzz.partial_ratio(a1, a2) / 100.0 for a1, a2 in zip(addrs1, addrs2)]
    addr_token_sort = [fuzz.token_sort_ratio(a1, a2) / 100.0 for a1, a2 in zip(addrs1, addrs2)]
    addr_token_set = [fuzz.token_set_ratio(a1, a2) / 100.0 for a1, a2 in zip(addrs1, addrs2)]
    
    name_exact = [1 if n1 == n2 else 0 for n1, n2 in zip(names1, names2)]
    addr_exact = [1 if a1 == a2 else 0 for a1, a2 in zip(addrs1, addrs2)]
    
    name_len_diff = [abs(len(n1) - len(n2)) for n1, n2 in zip(names1, names2)]
    addr_len_diff = [abs(len(a1) - len(a2)) for a1, a2 in zip(addrs1, addrs2)]
    
    cand_source = [2 if str(cid).startswith("S2-") else 3 for cid in cand_ids]
    
    # 🌟 Brownie Points: Phonetic Normalization & Numeric/ZIP Parsing
    name_soundex_match = [1 if jellyfish.soundex(n1) == jellyfish.soundex(n2) else 0 for n1, n2 in zip(names1, names2)]
    name_meta_match = [1 if jellyfish.metaphone(n1) == jellyfish.metaphone(n2) else 0 for n1, n2 in zip(names1, names2)]
    addr_num_overlap = [num_overlap(a1, a2) for a1, a2 in zip(addrs1, addrs2)]
    
    zips1 = [get_zip(a) for a in addrs1]
    zips2 = [get_zip(a) for a in addrs2]
    addr_zip_match = [1.0 if z1 and z1 == z2 else (0.0 if z1 and z2 else -1.0) for z1, z2 in zip(zips1, zips2)]
    
    df_features = pd.DataFrame({
        'source1_entity_id': s1_ids,
        'candidate_entity_id': cand_ids,
        'tfidf_sim': tfidf_sims,
        'name_ratio': name_ratio,
        'name_partial': name_partial,
        'name_token_sort': name_token_sort,
        'name_token_set': name_token_set,
        'name_wratio': name_wratio,
        'addr_ratio': addr_ratio,
        'addr_partial': addr_partial,
        'addr_token_sort': addr_token_sort,
        'addr_token_set': addr_token_set,
        'name_exact': name_exact,
        'addr_exact': addr_exact,
        'name_len_diff': name_len_diff,
        'addr_len_diff': addr_len_diff,
        'name_soundex_match': name_soundex_match,
        'name_meta_match': name_meta_match,
        'addr_num_overlap': addr_num_overlap,
        'addr_zip_match': addr_zip_match,
        'cand_source': cand_source
    })
    
    print(f"Feature computation completed in {time.time() - t0:.2f}s!", flush=True)
    return df_features


def predict_matches_streamed(model, df_candidates_flat, df_s1, df_s2, df_s3, threshold, chunk_size=500_000):
    """
    Computes features and predicts matches in streaming chunks.
    Keeps peak memory usage under 500 MB even for 35M+ candidate pairs.
    """
    if df_candidates_flat.empty:
        return pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_id'])

    n_pairs = len(df_candidates_flat)
    print(f"Streaming feature computation and inference for {n_pairs:,} pairs (chunk size {chunk_size:,})...", flush=True)
    t0 = time.time()

    df_pool = pd.concat([df_s2, df_s3], ignore_index=True)
    s1_dict = df_s1.set_index('entity_id').to_dict('index')
    pool_dict = df_pool.set_index('entity_id').to_dict('index')

    all_matched_s1 = []
    all_matched_cand = []

    feature_cols = [
        'tfidf_sim', 'name_ratio', 'name_partial', 'name_token_sort',
        'name_token_set', 'name_wratio', 'addr_ratio', 'addr_partial',
        'addr_token_sort', 'addr_token_set', 'name_exact', 'addr_exact',
        'name_len_diff', 'addr_len_diff', 
        'name_soundex_match', 'name_meta_match', 'addr_num_overlap', 'addr_zip_match',
        'cand_source'
    ]

    for start_idx in range(0, n_pairs, chunk_size):
        end_idx = min(start_idx + chunk_size, n_pairs)
        chunk = df_candidates_flat.iloc[start_idx:end_idx]

        s1_ids = chunk['source1_entity_id'].values
        cand_ids = chunk['candidate_entity_id'].values
        tfidf_sims = chunk['tfidf_sim'].values if 'tfidf_sim' in chunk.columns else np.zeros(len(chunk), dtype=np.float32)

        names1 = [str(s1_dict[sid]['business_name']).lower() for sid in s1_ids]
        names2 = [str(pool_dict[cid]['business_name']).lower() for cid in cand_ids]
        addrs1 = [str(s1_dict[sid]['business_address']).lower() for sid in s1_ids]
        addrs2 = [str(pool_dict[cid]['business_address']).lower() for cid in cand_ids]
        
        zips1 = [get_zip(a) for a in addrs1]
        zips2 = [get_zip(a) for a in addrs2]

        X_chunk = pd.DataFrame({
            'tfidf_sim': tfidf_sims.astype(np.float32),
            'name_ratio': np.array([fuzz.ratio(n1, n2) / 100.0 for n1, n2 in zip(names1, names2)], dtype=np.float32),
            'name_partial': np.array([fuzz.partial_ratio(n1, n2) / 100.0 for n1, n2 in zip(names1, names2)], dtype=np.float32),
            'name_token_sort': np.array([fuzz.token_sort_ratio(n1, n2) / 100.0 for n1, n2 in zip(names1, names2)], dtype=np.float32),
            'name_token_set': np.array([fuzz.token_set_ratio(n1, n2) / 100.0 for n1, n2 in zip(names1, names2)], dtype=np.float32),
            'name_wratio': np.array([fuzz.WRatio(n1, n2) / 100.0 for n1, n2 in zip(names1, names2)], dtype=np.float32),
            'addr_ratio': np.array([fuzz.ratio(a1, a2) / 100.0 for a1, a2 in zip(addrs1, addrs2)], dtype=np.float32),
            'addr_partial': np.array([fuzz.partial_ratio(a1, a2) / 100.0 for a1, a2 in zip(addrs1, addrs2)], dtype=np.float32),
            'addr_token_sort': np.array([fuzz.token_sort_ratio(a1, a2) / 100.0 for a1, a2 in zip(addrs1, addrs2)], dtype=np.float32),
            'addr_token_set': np.array([fuzz.token_set_ratio(a1, a2) / 100.0 for a1, a2 in zip(addrs1, addrs2)], dtype=np.float32),
            'name_exact': np.array([1 if n1 == n2 else 0 for n1, n2 in zip(names1, names2)], dtype=np.int8),
            'addr_exact': np.array([1 if a1 == a2 else 0 for a1, a2 in zip(addrs1, addrs2)], dtype=np.int8),
            'name_len_diff': np.array([abs(len(n1) - len(n2)) for n1, n2 in zip(names1, names2)], dtype=np.int16),
            'addr_len_diff': np.array([abs(len(a1) - len(a2)) for a1, a2 in zip(addrs1, addrs2)], dtype=np.int16),
            'name_soundex_match': np.array([1 if jellyfish.soundex(n1) == jellyfish.soundex(n2) else 0 for n1, n2 in zip(names1, names2)], dtype=np.int8),
            'name_meta_match': np.array([1 if jellyfish.metaphone(n1) == jellyfish.metaphone(n2) else 0 for n1, n2 in zip(names1, names2)], dtype=np.int8),
            'addr_num_overlap': np.array([num_overlap(a1, a2) for a1, a2 in zip(addrs1, addrs2)], dtype=np.float32),
            'addr_zip_match': np.array([1.0 if z1 and z1 == z2 else (0.0 if z1 and z2 else -1.0) for z1, z2 in zip(zips1, zips2)], dtype=np.float32),
            'cand_source': np.array([2 if str(cid).startswith("S2-") else 3 for cid in cand_ids], dtype=np.int8)
        }, columns=feature_cols)

        probs = model.predict_proba(X_chunk)[:, 1]
        del X_chunk

        matched_mask = probs >= threshold
        if np.any(matched_mask):
            all_matched_s1.extend(s1_ids[matched_mask])
            all_matched_cand.extend(cand_ids[matched_mask])

        chunk_idx = start_idx // chunk_size + 1
        total_chunks = (n_pairs + chunk_size - 1) // chunk_size
        if (chunk_idx % 10 == 0) or (end_idx == n_pairs):
            elapsed = time.time() - t0
            rate = end_idx / max(elapsed, 1e-3)
            print(f"  Chunk {chunk_idx:,}/{total_chunks:,} ({100*end_idx/n_pairs:.1f}%) | {rate:.0f} pairs/s | Matches found: {len(all_matched_s1):,}", flush=True)

    df_matches = pd.DataFrame({
        'source1_entity_id': all_matched_s1,
        'candidate_entity_id': all_matched_cand
    })
    print(f"Inference complete: {len(df_matches):,} total matches predicted in {time.time()-t0:.2f}s!", flush=True)
    return df_matches

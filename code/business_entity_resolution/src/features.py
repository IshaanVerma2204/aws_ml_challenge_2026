import pandas as pd
import numpy as np
import jellyfish
from tqdm import tqdm

def safe_jaccard(s1, s2):
    s1_set = set(s1.split())
    s2_set = set(s2.split())
    if not s1_set or not s2_set:
        return 0.0
    return len(s1_set.intersection(s2_set)) / len(s1_set.union(s2_set))

def compute_features(df_candidates_flat, df_s1, df_s2, df_s3):
    """
    Computes string similarity features for each candidate pair.
    """
    print("Computing features...")
    df_pool = pd.concat([df_s2, df_s3], ignore_index=True)
    
    s1_dict = df_s1.set_index('entity_id').to_dict('index')
    pool_dict = df_pool.set_index('entity_id').to_dict('index')
    
    features = []
    
    for _, row in tqdm(df_candidates_flat.iterrows(), total=len(df_candidates_flat), desc="Feature engineering"):
        s1_id = row['source1_entity_id']
        cand_id = row['candidate_entity_id']
        
        s1_rec = s1_dict[s1_id]
        cand_rec = pool_dict[cand_id]
        
        name1 = str(s1_rec['business_name']).lower()
        name2 = str(cand_rec['business_name']).lower()
        
        addr1 = str(s1_rec['business_address']).lower()
        addr2 = str(cand_rec['business_address']).lower()
        
        # Name features
        name_lev = jellyfish.levenshtein_distance(name1, name2)
        name_jaro = jellyfish.jaro_winkler_similarity(name1, name2)
        name_jaccard = safe_jaccard(name1, name2)
        name_exact = int(name1 == name2)
        
        # Address features
        addr_lev = jellyfish.levenshtein_distance(addr1, addr2)
        addr_jaro = jellyfish.jaro_winkler_similarity(addr1, addr2)
        addr_jaccard = safe_jaccard(addr1, addr2)
        addr_exact = int(addr1 == addr2)
        
        # Length features
        name_len_diff = abs(len(name1) - len(name2))
        addr_len_diff = abs(len(addr1) - len(addr2))
        
        features.append({
            'source1_entity_id': s1_id,
            'candidate_entity_id': cand_id,
            'name_lev': name_lev,
            'name_jaro': name_jaro,
            'name_jaccard': name_jaccard,
            'name_exact': name_exact,
            'addr_lev': addr_lev,
            'addr_jaro': addr_jaro,
            'addr_jaccard': addr_jaccard,
            'addr_exact': addr_exact,
            'name_len_diff': name_len_diff,
            'addr_len_diff': addr_len_diff
        })
        
    df_features = pd.DataFrame(features)
    return df_features

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

def generate_candidates(df_s1, df_s2, df_s3, top_k=20):
    """
    Generates candidate pairs using TF-IDF + k-NN blocking per country.
    df_s1: Source 1 dataframe
    df_s2: Source 2 dataframe
    df_s3: Source 3 dataframe
    top_k: Number of neighbors to retrieve per entity per source combination
    
    Returns: DataFrame with columns [source1_entity_id, candidate_entity_id]
    """
    # Combine S2 and S3 as the target search pool
    df_pool = pd.concat([df_s2, df_s3], ignore_index=True)
    
    # Text for indexing
    df_s1['text'] = (df_s1['business_name'] + " " + df_s1['business_address']).str.lower()
    df_pool['text'] = (df_pool['business_name'] + " " + df_pool['business_address']).str.lower()
    
    countries = df_s1['country'].unique()
    
    all_candidates = []
    
    print("Running blocking/candidate generation...")
    for country in tqdm(countries, desc="Blocking by country"):
        s1_country = df_s1[df_s1['country'] == country].copy()
        pool_country = df_pool[df_pool['country'] == country].copy()
        
        if pool_country.empty or s1_country.empty:
            continue
            
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=100000)
        
        # Fit on combined to have same vocab, but this might be memory heavy. 
        # Fit on pool is usually fine, then transform both.
        vectorizer.fit(pool_country['text'])
        
        tfidf_pool = vectorizer.transform(pool_country['text'])
        tfidf_s1 = vectorizer.transform(s1_country['text'])
        
        # Determine effective k
        n_neighbors = min(top_k, tfidf_pool.shape[0])
        
        # Use cosine similarity (NearestNeighbors with cosine metric)
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', n_jobs=-1)
        nn.fit(tfidf_pool)
        
        distances, indices = nn.kneighbors(tfidf_s1)
        
        pool_ids = pool_country['entity_id'].values
        s1_ids = s1_country['entity_id'].values
        
        # Construct candidate pairs
        for i, s1_id in enumerate(s1_ids):
            for j in range(n_neighbors):
                # distance 0 means identical, 1 means completely different
                if distances[i, j] < 0.8: # threshold to prune very bad candidates
                    all_candidates.append({
                        'source1_entity_id': s1_id,
                        'candidate_entity_id': pool_ids[indices[i, j]]
                    })
                    
    # Also handle entities that might have no valid candidates to ensure every S1 entity is tracked?
    # Actually, candidate_pairs.tsv needs one row per S1 entity. 
    # If a candidate list is empty, it's just an empty string.
    
    df_candidates_flat = pd.DataFrame(all_candidates)
    return df_candidates_flat

def format_candidate_pairs(df_s1, df_candidates_flat):
    """
    Formats the flat candidate pairs into the required TSV structure.
    """
    if df_candidates_flat.empty:
        candidate_dict = {}
    else:
        grouped = df_candidates_flat.groupby('source1_entity_id')['candidate_entity_id'].apply(lambda x: ','.join(list(dict.fromkeys(x))))
        candidate_dict = grouped.to_dict()
    
    res = []
    for s1_id in df_s1['entity_id']:
        res.append({
            'source1_entity_id': s1_id,
            'candidate_entity_ids': candidate_dict.get(s1_id, "")
        })
    
    return pd.DataFrame(res)

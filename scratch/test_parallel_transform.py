import pandas as pd
import numpy as np
import scipy.sparse as sp
import time
from concurrent.futures import ProcessPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer

def _transform_chunk(args):
    vectorizer, chunk_texts = args
    return vectorizer.transform(chunk_texts)

def transform_parallel(vectorizer, texts, n_jobs=8, chunk_size=200000):
    print(f"Parallel transforming {len(texts)} documents using {n_jobs} CPU processes...", flush=True)
    t0 = time.time()
    
    n = len(texts)
    chunks = [texts.iloc[i:i+chunk_size] if hasattr(texts, 'iloc') else texts[i:i+chunk_size] 
              for i in range(0, n, chunk_size)]
              
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        matrices = list(executor.map(_transform_chunk, [(vectorizer, c) for c in chunks]))
        
    res = sp.vstack(matrices, format='csr')
    print(f"Parallel transform completed in {time.time() - t0:.2f}s! Matrix shape: {res.shape}", flush=True)
    return res

if __name__ == "__main__":
    print("Testing parallel transform on 1,000,000 text strings...")
    texts = pd.Series(["business corporation enterprise logistics trading " + str(i) for i in range(1000000)])
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=5, max_df=0.25, max_features=50000, dtype=np.float32)
    vec.fit(texts.sample(100000))
    
    m = transform_parallel(vec, texts, n_jobs=8)

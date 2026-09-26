import pandas as pd
import numpy as np
import scipy.sparse as sp
import time
from sklearn.feature_extraction.text import TfidfVectorizer

print("Testing chunked batch transform on 1,000,000 text strings...")
texts = pd.Series(["business corporation enterprise logistics trading " + str(i) for i in range(1000000)])

vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=5, max_df=0.25, max_features=50000, dtype=np.float32)
vec.fit(texts.sample(100000))

def transform_in_batches(vectorizer, texts, batch_size=200000):
    print(f"Transforming {len(texts)} documents in batches of {batch_size}...", flush=True)
    t0 = time.time()
    matrices = []
    n = len(texts)
    for i in range(0, n, batch_size):
        chunk = texts.iloc[i:i+batch_size]
        matrices.append(vectorizer.transform(chunk))
    res = sp.vstack(matrices, format='csr')
    print(f"Chunked transform completed in {time.time() - t0:.2f}s! Matrix shape: {res.shape}", flush=True)
    return res

m = transform_in_batches(vec, texts)

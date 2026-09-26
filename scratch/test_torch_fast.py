import torch
import numpy as np
import scipy.sparse as sp
import time

print("Testing PyTorch Multi-threaded BLAS Matrix Multiplication...")
print(f"PyTorch version: {torch.__version__}, Threads: {torch.get_num_threads()}")

# Create realistic sparse matrices
N_s1 = 1000
N_pool = 1000000
D = 25000

print(f"Creating test sparse matrices: S1={N_s1}, Pool={N_pool}, Vocab={D}...")
s1_sparse = sp.random(N_s1, D, density=0.01, format='csr', dtype=np.float32)
pool_sparse = sp.random(N_pool, D, density=0.01, format='csr', dtype=np.float32)

t0 = time.time()
print("Running SciPy sparse dot product...")
sim_scipy = s1_sparse @ pool_sparse.T
print(f"SciPy completed in {time.time() - t0:.2f}s!")

t0 = time.time()
print("Running PyTorch BLAS multi-threaded matrix multiplication...")
s1_t = torch.from_numpy(s1_sparse.toarray())
pool_t = torch.from_numpy(pool_sparse.toarray()).t()

sim_torch = torch.mm(s1_t, pool_t)
val_topk, idx_topk = torch.topk(sim_torch, k=20, dim=1)
print(f"PyTorch BLAS + topk completed in {time.time() - t0:.2f}s!")

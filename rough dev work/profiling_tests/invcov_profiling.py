"""
Python file prepared to run using the scalene profiler. As such, routines do not contain a docstring
"""

import numpy as np
import cupy as cp
import seaborn as sns
import matplotlib.pyplot as plt
import corrcal
from cupyx.profiler import benchmark
# from scalene import scalene_profiler

# scalene_profiler.start()

def zeropad(array, edges, xp):
    largest_block = xp.diff(edges).max()
    n_blocks = edges.size - 1
    if array.ndim == 1:   #should only be the case for the noise matrix
        out = xp.zeros((n_blocks, int(largest_block)))
    else:
        out  = xp.zeros((n_blocks, int(largest_block), int(array.shape[1])))
    for block, (start, stop) in enumerate(zip(edges, edges[1:])):
        start, stop = int(start), int(stop)
        out[block, :stop - start] = array[start:stop]
    return out


def undo_zeropad(array, edges, xp):
    if array.ndim == 2:   #once again only the case for the noise matrix
        out = xp.zeros((int(edges[-1])))
    else:
        out = xp.zeros((int(edges[-1]), int(array.shape[2])))
    for block, (start, stop) in enumerate(zip(edges, edges[1:])):
        start, stop = int(start), int(stop)
        out[start:stop] = array[block, :stop - start]
    return out

def inverse_covariance(N, Del, Sig, edges, xp):

    Del = zeropad(Del, edges, xp = xp)
    Sig = zeropad(Sig, edges, xp = xp)
    N_inv = 1/N     
    N_inv = zeropad(N_inv, edges, xp = xp)

    temp = N_inv[..., None] * Del    
    temp2 = xp.transpose(Del, [0, 2, 1]) @ temp
    L_del = xp.linalg.cholesky(xp.eye(Del.shape[2])[None, ...] + temp2)   
    Del_prime = temp @ xp.transpose(xp.linalg.inv(L_del).conj(), [0, 2, 1]) 
          
    A = N_inv[..., None] * Sig
    B = xp.transpose(Sig.conj(), [0, 2, 1]) @ Del_prime
    W = A - Del_prime @ xp.transpose(B.conj(), [0, 2, 1])
    L_sig = xp.linalg.cholesky(
        xp.eye(Sig.shape[2]) + xp.sum(xp.transpose(A.conj(), [0, 2, 1]) @ Sig, axis = 0) - xp.sum(B @ xp.transpose(B.conj(), [0, 2, 1]), axis = 0)
    )
    Sig_prime = W @ xp.linalg.inv(L_sig).T.conj()[None, ...]

    N_inv = undo_zeropad(N_inv, edges, xp = xp)
    Del_prime = undo_zeropad(Del_prime, edges, xp = xp)
    Sig_prime = undo_zeropad(Sig_prime, edges, xp = xp)   

    return N_inv, Del_prime, Sig_prime


#the main parametes describing our problem.
n_bl = 1000
n_eig = 3
n_src = 6
xp = cp  #run things on the gpu using cupy

#random array of edges for the diffuse matrix
edges = xp.unique(xp.random.randint(1, n_bl-1, size = 10))
edges = xp.concatenate((xp.array([0]), edges, xp.array([n_bl])))
print(f"The edges of the redundant blocks have indices{edges}")

#some random noise, diffuse, and source covariance matrices
sim_noise_mat = xp.random.rand(n_bl)**2   #in principle this is squared since is a variance
sim_diff_mat = xp.random.rand(n_bl, n_eig)
sim_src_mat = xp.random.rand(n_bl, n_src)

#actually go and find inverse components
N_inv, Del_p, Sig_p = inverse_covariance(sim_noise_mat, sim_diff_mat, sim_src_mat, edges=edges, xp = xp)

# scalene_profiler.end()


"""
For utilizing Cupy benchmark profiler. Ie. print the cpu and gpu times along with the rest of the code
"""
# test_results = str(benchmark(inverse_covariance, (sim_noise_mat, sim_diff_mat, sim_src_mat, edges, xp), n_repeat=100))
# test_results = test_results.split()
# cpu_t = float(test_results[3])/1e6
# gpu_t = float(test_results[14])/1e6
# print(f"Time on cpu: {cpu_t:.6f}s")
# print(f"Time on gpu: {gpu_t:.6f}s")
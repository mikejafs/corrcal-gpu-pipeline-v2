#simply here to test the timings for the pure numpy 
#version of the zeropad function

import timeit
import numpy as np
import cupy as cp
from cupyx.profiler import benchmark

def zeropad(array, edges, xp):
    """
    Pads an input noise, diffuse, or source matrix with zeros according to
    the largest redundant block in the diffuse matrix. Note that although the input arrays
    will have a 1D or 2D shape for noise or diff/sourc matrices respectively, the routine converts 
    to either 2D or 3D shapes so we can easily perform block mutliplication in the inverse covariance
    function.

    Parameters
    ----------
    array: Input noise, diffuse, or source matrix. Should be of shape (n_bl,), (n_bl, n_eig), or (n_bl, n_src) respectively
    edges: Array containing indices corresponding to the edges of redundant blocks in the diffuse matrix
            Note that the "edges" index the beginning row (or "edge") of each redundant block
    
    Returns
    -------
    out: The output zero-padded noise, diffuse, or source matrix where each matrix has also been reshaped 
        to be easily used in the inverse covariance function that performs mutliplication over blocks. The 
        output matrices have shapes of either (n_blocks, largest_red_block), (n_blocks, largest_red_block, n_eig), 
        or (n_blocks, largest_red_block, n_src) respectively.
    """

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


#let's benchmark the zeropad function

#the main parametes describing our problem.
n_bl = 120000
n_eig = 10
n_src = 500
xp = np  #run things on the gpu using cupy

#random array of edges for the diffuse matrix
edges = xp.unique(xp.random.randint(1, n_bl-1, size = 500))
edges = xp.concatenate((xp.array([0]), edges, xp.array([n_bl])))
# print(f"The edges of the redundant blocks have indices{edges}")

#some random noise, diffuse, and source covariance matrices
sim_noise_mat = xp.random.rand(n_bl)**2   #in theory this is squared since is a variance
sim_diff_mat = xp.random.rand(n_bl, n_eig)
sim_src_mat = xp.random.rand(n_bl, n_src)

# result = str(benchmark(zeropad, (sim_diff_mat, edges, xp), n_repeat=1000))
# result = result.split()
# cpu_t = float(result[3]) / 1e6
# gpu_t = float(result[14]) / 1e6
# print("Times measured using Cupy:")
# print(f"Time on CPU: {cpu_t:.6f}s")
# print(f"Time on GPU: {gpu_t:.6f}s")

#Note that the gpu benchmark function from cupy may not be the 
#best way to time things here for a pure cpu code. Let's try with 
#timeit below

# start = timeit.default_timer()
# zeropad(sim_diff_mat, edges, xp)
# stop = timeit.default_timer()
# print()
# print("Time measured in seconds using timeit is:")
# print(stop - start)

print()

num_iter = 1000
print(f"CPU times using the timeit function with {num_iter} iterations")
time_taken = timeit.timeit('zeropad(sim_diff_mat, edges, xp)', 
                           setup='from __main__ import zeropad, sim_diff_mat, edges, xp', 
                           number=num_iter)

print(f"Execution time: {time_taken/num_iter:.6f} seconds for {num_iter} iterations")


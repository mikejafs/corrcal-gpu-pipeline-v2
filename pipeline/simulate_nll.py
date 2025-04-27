"""
Module designed for simulating the NLL and comparing output
to corrcal's.
"""

import cupy as cp
import numpy as np
from corrcal import SparseCov
from corrcal.optimize import *
from optimize import *
from simulate_params import *

def simulate_nll(n_ant, n_eig, n_src):

    spms = SimCorrcalParams(n_ant, n_eig, n_src, xp=cp)
    ant_1_array, ant_2_array = spms.ant_arrays()
    # n_bl = spms.n_bl()
    edges = spms.edges()

    sim_data = spms.sim_data()
    noise = sim_data[0]
    diff = sim_data[1]
    src = sim_data[2]
    gains = sim_data[3]
    data = sim_data[4]

    """ SIMULATE GPU VERSION """
    nll_gpu = gpu_nll(gains,
                  noise,
                  diff,
                  src,
                  data,
                  edges,
                  ant_1_array,
                  ant_2_array,
                  scale=1,
                  phs_norm_fac=cp.inf
                  )
    
    # print(f"gpu nll dtype {(nll_gpu.shape)}")
    print(nll_gpu)

    """ SIMULATE CPU VERSION """
    noise = cp.asnumpy(noise)
    diff = cp.asnumpy(diff)
    src = cp.asnumpy(src)
    gains = cp.asnumpy(gains)
    data = cp.asnumpy(data)
    edges = cp.asnumpy(edges)
    ant_1_array = cp.asnumpy(ant_1_array)
    ant_2_array = cp.asnumpy(ant_2_array)


    cov = SparseCov(noise, src, diff, edges, spms.n_eig, isinv=False)
    cinv, logd = cov.inv(return_det=True)
    # print(f"cpu log det is {logd}")

    nll_cpu = nll(gains, cov, data, ant_1_array, ant_2_array, scale=1, phs_norm_fac=np.inf)
    # print(f"cpu dtype {(nll_cpu).shape}")
    print(nll_cpu)

    """return comparison results"""
    return np.allclose(nll_gpu, nll_cpu)


if __name__ == "__main__":
    n_ant = 100
    n_eig = 3
    n_src = 5

    #TODO: add tests for benchmarking and ability to run off
    #      many realizations to determine if the two algorithms
    #      ever mismatch. As was done in simulating grad_nll.

    # simulate_nll(n_ant, n_eig, n_src)
    truth = simulate_nll(n_ant, n_eig, n_src)
    print(truth)

"""
Module testing the inverse covariance functionalities against
the analagous corrcal versions
"""

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import cupy as cp
import numpy as np
from corrcal import SparseCov
from simulate_params import *
from invcov import *
from zp_puregpu_funcs_py import *

def simulate(n_ant, n_eig, n_src):
    
    spms = SimCorrcalParams(n_ant, n_eig, n_src, xp=cp)
    edges = spms.edges()

    sim_data = spms.sim_data()
    noise = sim_data[0]
    diff = sim_data[1]
    src = sim_data[2]

    zp_noise, nb, lb = zeroPad(noise, edges, return_inv=True)
    zp_diff, nb, lb = zeroPad(diff, edges, return_inv=False)
    zp_src, nb, lb = zeroPad(src, edges, return_inv=False)

    """GPU Calc"""
    logdet, inv_noise, inv_diff, inv_src = inverse_covariance(zp_noise, 
                                                              zp_diff, 
                                                              zp_src,
                                                              xp=cp,
                                                              ret_det=True,
                                                              N_is_inv=True
                                                              )
    
    inv_noise = undo_zeroPad(inv_noise, edges, ReImsplit=True)
    inv_diff = undo_zeroPad(inv_diff, edges, ReImsplit=True)
    inv_src = undo_zeroPad(inv_src, edges, ReImsplit=True)

    # print(inv_noise)
    
    """CPU calc"""
    noise = cp.asnumpy(noise)
    diff = cp.asnumpy(diff)
    src = cp.asnumpy(src)
    edges = cp.asnumpy(edges)
    
    cov = SparseCov(noise, src, diff, edges, spms.n_eig, isinv=False)
    cinv, logd = cov.inv(return_det=True)
    cpu_n = cinv.noise
    cpu_d = cinv.diff_mat
    cpu_s = cinv.src_mat

    print(np.allclose(logdet, logd))
    print(np.allclose(inv_noise, cpu_n))
    print(np.allclose(inv_diff, cpu_d))
    print(np.allclose(inv_src, cpu_s))

    
if __name__ == "__main__":
    n_ant = 10
    n_eig = 3
    n_src = 5
    simulate(n_ant, n_eig, n_src)
"""
Module containing the correct way to simulate each of the input parameters and matrices.

Pulled from the most recent (as of 2025-04-26) updates to simulating the grad_nll function.
Can look at the simulate_grad.py file to see a more detailed view of use motivation for these
parameters.
"""

import numpy as np
import cupy as cp

class SimCorrcalParams():
    def __init__(self, n_ant, n_eig, n_src, xp):
        self.n_ant = n_ant
        self.n_eig = n_eig
        self.n_src = n_src
        self.n_gains = 2*n_ant
        self.xp = xp

    def ant_arrays(self):
        ant_1_array, ant_2_array = cp.tril_indices(self.n_ant, k=-1)
        return ant_1_array, ant_2_array
    
    def n_bl(self):
        ant_1_array = self.ant_arrays()[0]
        n_bl = 2*len(ant_1_array)
        return n_bl

    def edges(self):
        edges = (self.xp.unique(self.xp.random.randint(1, int(self.n_bl() / 2) - 1, size=(self.n_ant,))* 2))
        edges = self.xp.concatenate((self.xp.array([0]), edges, self.xp.array([self.n_bl()])))
        return edges
    
    def sim_data(self):
        noise_mat = self.xp.random.rand(self.n_bl(), dtype='float64')
        diff_mat = self.xp.random.rand(self.n_bl(), self.n_eig, dtype='float64')
        src_mat = self.xp.random.rand(self.n_bl(), self.n_src, dtype='float64')
        gains = self.xp.random.rand(self.n_gains, dtype='float64')
        data = self.xp.random.rand(self.n_bl(), dtype='float64')
        return noise_mat, diff_mat, src_mat, gains, data

    # def return_cpu_data(self):
    #     noise_mat = cp.asnumpy(self.sim_data()[0])
    #     diff_mat = cp.asnumpy(self.sim_data()[1])
    #     src_mat = cp.asnumpy(self.sim_data()[2])
    #     gains_mat = cp.asnumpy(self.sim_data()[3])
    #     data_vec = cp.asnumpy(self.sim_data()[4])
    #     edges_mat = cp.asnumpy(self.edges())
    #     ant_1_data = cp.asnumpy(self.ant_arrays()[0])
    #     ant_2_data = cp.asnumpy(self.ant_arrays()[1])
    #     return noise_mat, diff_mat, src_mat, gains_mat, data_vec, edges_mat, ant_1_data, ant_2_data



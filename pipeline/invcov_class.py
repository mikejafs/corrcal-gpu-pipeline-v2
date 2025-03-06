"""Add working routines from mat inverse v8 and create class around them"""

#For benchmarking, should probably create a class that uses the benchmark code or maybe just a function for now
#that is in its own folder called tests or something. The file name can be something like benchmark.py... We can expand
#upon this later...

"""
Module containing the sparse covariance class, with the methods corresponding to the functions in the orgiginal
invcov folder
"""

"""
NOTE: The plan is to create each method so they don't require any individual params. The user lterally creates an 
    instance of the sparse covariance class (providing all necessary attributes in the process) and can simply run each
    of the methods quickly after....
    
    This means that each of the arguments in the functions need to become attributes.
"""

import numpy as np
import cupy as cp
import seaborn as sns
import matplotlib.pyplot as plt
import corrcal
from cupyx.profiler import benchmark
from zp_puregpu_funcs_py import *

class SparseCovarianceGPU:
    """
    Class representing the 2-level sparse covariance object.

    Attributes:
        array:
        edges:
        xp
    Methods:

    """
    def __init__(self, noise_mat, diff_mat, src_mat, edges, xp, is_zeropaded=True):
        self.noise_mat = noise_mat
        self.diff_mat = diff_mat
        self.src_mat = src_mat
        self.edges = edges
        self.xp = xp
        self.is_zeropadded = is_zeropaded
        if self.is_zeropadded:
            self.noise_mat_zp = zeroPad(self.noise_mat, self.edges, True)
            self.diff_mat_zp = zeroPad(self.diff_mat, self.edges, False)
            self.src_mat_zp = zeroPad(self.src_mat, self.edges, False)
        else:
            raise NotImplementedError("Cannot initialize a non-zeropadded sparse"
                                      "cov object yet")

    def zeropad(self, array, edges, return_inv=True):
        """Zeropads an array according to the largest block in the diffuse covariance matrix"""

        """
        SHOULD ACTUALLY PUT THE ZEROPADDING HERE AND JUST UPDATE SELF.DIFF_MAT ETC IF METHOD
        IS CALLED (REFER TO BOBBY'S GITHUB SPARSE COV CLASS)
        """

        pass

    def apply_gains(self):
        pass

    def inverse_covariance(N, Del, Sig, edges, xp, ret_det=False, N_is_inv=True):
        """
        Given the components of the 2-level sparse covariance object, computes
        the components of the inverse covariance object. Currectly does not
        support the option to return the determinant of the covariance.
        """

        if N_is_inv:
            N_inv = N
        else:
            N_inv = 1 / N

        temp = N_inv[..., None] * Del
        temp2 = xp.transpose(Del, [0, 2, 1]) @ temp
        L_del = xp.linalg.cholesky(xp.eye(Del.shape[2])[None, ...] + temp2)
        Del_prime = temp @ xp.transpose(xp.linalg.inv(L_del).conj(), [0, 2, 1])

        A = N_inv[..., None] * Sig
        B = xp.transpose(Sig.conj(), [0, 2, 1]) @ Del_prime
        W = A - Del_prime @ xp.transpose(B.conj(), [0, 2, 1])
        L_sig = xp.linalg.cholesky(
            xp.eye(Sig.shape[2]) + xp.sum(
                xp.transpose(A.conj(), [0, 2, 1]) @ Sig, axis=0
            ) - xp.sum(
                B @ xp.transpose(B.conj(), [0, 2, 1]), axis=0
            )
        )
        Sig_prime = W @ xp.linalg.inv(L_sig).T.conj()[None, ...]

        if ret_det:
            logdet = (xp.sum(xp.diagonal(xp.log(L_del), axis2=1, axis1=2)) + xp.sum(xp.diagonal(xp.log(L_sig))))
            # cp.cuda.Stream.null.synchronize()
            return logdet, N_inv, Del_prime, Sig_prime
        else:
            pass
        # cp.cuda.Stream.null.synchronize()
        return N_inv, Del_prime, Sig_prime


    def sparden_convert(Array, largest_block, n_blocks, n_bl, n_eig, edges, xp, zeroPad=True):
        """
        Converts either the dense diffuse matrix to sparse, or the sparse diffuse matrix to dense.
        The array (either dense or sparse) should be simply handed to the function and the desired operation
        (sparse-to-dense or dense-to-sparse) will be performed automatically

        Parameters
        ----------
        Array: Either dense or sparse diffuse matrix
        n_bls: Number of baselines used in the calculation of the sparse diffuse matrix
        n_eig: Number of eigenmodes being used to construct the sparse diffuse matrix
        edges: An array controlling the edges of the redundant group blocks in the sparse diffuse matrix
        xp: The choice to either run the computation on the gpu (xp = cp) or cpu (xp = np)

        Returns
        -------
        out: If the dense form was provided, the sparse form with shape (n_bls x n_eig) will be returned.
            If the sparse form was provided, the dense form with shape (n_bls x n_eig*n_grps) with
            n_grps = # redundant groups will be returned.
        """

        if Array.shape[1] == n_eig:
            n_grp = edges.size - 1

            if zeroPad:
                out = xp.zeros((n_blocks*largest_block, n_eig*n_grp))
                for i in range(n_blocks):
                    out[i*largest_block:(i+1)*largest_block, i*n_eig:(i+1)*n_eig
                        ] = Array[i*largest_block:(i+1)*largest_block]
            else:
                out = xp.zeros((n_bl, n_eig*n_grp))
                for i, (start, stop) in enumerate(zip(edges, edges[1:])):
                    out[start:stop, i*n_eig : (i+1)*n_eig] = Array[start:stop]
        else:
            if zeroPad:
                raise NotImplementedError("Dense to sparse has not been implimented with zeropadded arrays yet")
                # out = xp.zeros((n_blocks * largest_block))
            else:
                out = xp.zeros((n_bl, n_eig))
                for i, (start, stop) in enumerate(zip(edges, edges[1:])):
                    out[start:stop] = Array[start:stop, i*n_eig : (i+1)*n_eig]

        return out





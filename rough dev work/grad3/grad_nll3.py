
"""
MODULE CONTAINIG THE GRADIENT OF THE NEGATIVE LOG LIKELIHOOD FUNCTION

This version used the 'chain rule' version of the gradient

-------------------------------------------------------------------
NOTE: (UPDATES) 
Same as the grad2 updates (ie. uses all grad2 functions - will later 
change to just grad stuff), but now that things are (almost) working 
this just everything, but with cleaned up comments and functions in
preparation to add things in a more or less finalized version to
the gpu_pipeline directory.
--------------------------------------------------------------------
"""
import ctypes
import os
import sys

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
target_dir = os.path.join(parent_dir, 'corrcal_gpu_pipeline', 'pipeline')
sys.path.insert(0, target_dir)

import cupy as cp
from zp_puregpu_funcs_py import *
from utils import *
from invcov import *
from populate_grad_py import *


#full grad function
def gpu_grad_nll(n_ant, 
                 gains, 
                 data, 
                 scale, 
                 phs_norm_fac, 
                 noise, 
                 diff_mat, 
                 src_mat, 
                 edges, 
                 ant_1_array, 
                 ant_2_array, 
                 xp):
    """
    Compute the GPU-accelerated gradient of the negative log-likelihood.

    This function zero-pads covariance and gain matrices, applies gains to sky and diffuse models,
    inverts the total covariance, and assembles the gradient with phase normalization.

    Parameters
    ----------
    n_ant : int
        Number of antennas.
    gains : xp.ndarray
        Interleaved real/imag gain vector of length 2*n_ant.
    data : xp.ndarray
        Measured visibility data (non zero-padded).
    scale : float
        Factor by which to scale the final gradient.
    phs_norm_fac : float
        Normalization factor applied to phase regularization.
    noise : xp.ndarray
        Noise covariance matrix.
    diff_mat : xp.ndarray
        Diffuse-sky covariance matrix.
    src_mat : xp.ndarray
        Point-source covariance matrix.
    edges : array-like
        Zero-padding specification for matrix dimensions.
    ant_1_array, ant_2_array : array-like
        Baseline antenna index pairs for gain mapping.
    xp : module
        Array library (e.g., `cupy` or `numpy`) for execution.

    Returns
    -------
    xp.ndarray
        Interleaved real/imag gradient vector of length 2*n_ant, scaled by `scale`.
    """
    #zeropad noise, diffuse, source matrices, and gain matrices
    zp_noise_inv, lb, nb = zeroPad(noise, edges, return_inv=True)
    zp_noise, _, _ = zeroPad(noise, edges, return_inv=False)  #need the non-inverse for constructing regular sparce cov
    zp_diff_mat, _, _ = zeroPad(diff_mat, edges, return_inv=False)
    zp_src_mat, _, _ = zeroPad(src_mat, edges, return_inv=False)
    zp_data, _, _ = zeroPad(data, edges, return_inv=False)
    zp_cplex_gain_mat = zeropad_gains(gains, edges, ant_1_array, ant_2_array, xp = cp, return_inv=False)

    #apply gains to the source and diffuse matrices (ie. constructing the 'true' convariance)
    gain_diff_mat = apply_gains(zp_cplex_gain_mat, zp_diff_mat, xp=cp)
    gain_src_mat = apply_gains(zp_cplex_gain_mat, zp_src_mat, xp=cp)

    inv_noise, inv_diff, inv_src = inverse_covariance(zp_noise_inv, gain_diff_mat, gain_src_mat, cp, ret_det=False, N_is_inv=True)

    #Now compute p = C^-1 @ data => Might want to construct my own __matmul__ function for this
    p = sparse_cov_times_vec(inv_noise, inv_diff, inv_src, zp_data, isinv=True)

    #compute q = (C - N) @ G.T @ p
    q = p.copy()
    q[:, ::2] = zp_cplex_gain_mat.real*p[:, ::2] + zp_cplex_gain_mat.imag*p[:, 1::2]
    q[:, 1::2] = -zp_cplex_gain_mat.imag*p[:, ::2] + zp_cplex_gain_mat.real*p[:, 1::2]

    #in computing q, we just make noise = 0 and run the C \times d function
    zp_noise = zp_noise.reshape(nb, lb, 1) #1D mats are left as 2D and not 2D + 1 col so that invcov runs so need to reshape here
    zp_noise = cp.zeros_like(zp_noise)
    q = sparse_cov_times_vec(zp_noise, zp_diff_mat, zp_src_mat, q, isinv=False)

    #compute s and t => Note this bring the shape of s & t to 1/2len(p or q)
    zp_s = p[:, ::2]*q[: ,::2] + p[:, 1::2]*q[:, 1::2]
    zp_t = p[:, 1::2]*q[:, ::2] - p[:, ::2]*q[:, 1::2]
    
    #compute the inverse power
    inv_power = cp.sum(
        inv_diff[:, ::2]**2 + inv_diff[:, 1::2]**2, axis=2
    ) + cp.sum(
        inv_src[:, ::2]**2 + inv_src[:, 1::2]**2, axis=2
    )

    #reshape inverse power anticipating reversing the zeropadding
    inv_power = inv_power.reshape(nb, int(lb/2), 1)

    #accumulate gradient
    #~~ need to first undo the zeropadding of s, t, and P
    # (note that s and t are impicitely p and q) 
    s = undo_zeroPad(zp_s, edges, ReImsplit=False)
    t = undo_zeroPad(zp_t, edges, ReImsplit=False)
    P = undo_zeroPad(inv_power, edges, ReImsplit=False)

    #fill out the dLdG gradient (n_ant x n_ant) matrix
    gradr, gradi = populate_gradient(
        n_ant, gains, s, t, P, noise, ant_1_array, ant_2_array
    )

    #calculate dLdg
    A = gradr + gradr.T
    B = gradi.T - gradi
    dLdgr = A@gains[::2] + B@gains[1::2]
    dLdgi = A@gains[1::2] - B@gains[::2]

    #initialize and populate alternating Re and Im full gradient vector
    gradient = cp.zeros((2*len(dLdgr)))
    gradient[::2] = dLdgr
    gradient[1::2] = dLdgi

    #phase normalization
    amps = cp.sqrt(gains[::2]**2 + gains[1::2]**2)
    phases = cp.arctan2(gains[1::2], gains[::2])
    n_ants = gains.size/2
    grad_phs_prefac = 2 * cp.sum(phases) / (amps * n_ants**2 * phs_norm_fac**2)
    gradient[::2] -= grad_phs_prefac * cp.sin(phases)
    gradient[1::2] += grad_phs_prefac * cp.cos(phases)

    return gradient/scale



def populate_gradient(n_ant, gains, s, t, P, noise, ant_1_inds, ant_2_inds):
    """
    Compute the real and imaginary parts of the negative log-likelihood gradient matrix.

    Thin Python wrapper around the CUDA `populate_gradient` kernel, which fills
    the dL/dG matrices for each antenna pair in parallel.

    Parameters
    ----------
    n_ant : int
        Number of antennas (matrix dimension).
    gains : cupy.ndarray
        Interleaved real/imag gain vector of length 2*n_ant.
    s : cupy.ndarray
        Real part of the p·q product for each baseline (length = n_bl).
    t : cupy.ndarray
        Imaginary part of the p·q product for each baseline (length = n_bl).
    P : cupy.ndarray
        Inverse-power term P[k] = (∑ |inv_cov|²)_k for each baseline (length = n_bl).
    noise : cupy.ndarray
        Noise weights for each baseline (length = 2*n_bl).
    ant_1_inds, ant_2_inds : cupy.ndarray of int64
        Arrays of length n_bl giving the antenna indices for each baseline.

    Returns
    -------
    dLdGr : cupy.ndarray, shape (n_ant, n_ant)
        Real part of the gradient matrix ∂L/∂G.
    dLdGi : cupy.ndarray, shape (n_ant, n_ant)
        Imaginary part of the gradient matrix ∂L/∂G.
    """
    dLdGr = cp.zeros((n_ant, n_ant))
    dLdGi = cp.zeros((n_ant, n_ant))
    
    n_bls = ant_1_inds.size
    # print(n_bls)

    # n_ant = len(gains)
    pop_grad_lib.populate_gradient(
        ctypes.cast(gains.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(s.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(t.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(P.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(noise.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(dLdGr.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(dLdGi.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(ant_1_inds.data.ptr, ctypes.POINTER(ctypes.c_long)),
        ctypes.cast(ant_2_inds.data.ptr, ctypes.POINTER(ctypes.c_long)),
        n_ant,
        n_bls,
    )
    return dLdGr, dLdGi





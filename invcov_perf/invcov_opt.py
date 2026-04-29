# import numpy as np
import ctypes
import cupy as cp
import seaborn as sns
import matplotlib.pyplot as plt
import corrcal
from cupyx.profiler import benchmark
import cupyx.scipy.linalg as cpx_linalg



# Load fused temp/temp2 library (adjust path if needed)
temp_lib = ctypes.cdll.LoadLibrary(
    "/home/mike/corrcal_gpu_pipeline/invcov_perf/fused_temp_temp2.so"
)

temp_lib.fused_temp_temp2.argtypes = [
    ctypes.c_void_p,  # Ninv
    ctypes.c_void_p,  # Del
    ctypes.c_void_p,  # Temp
    ctypes.c_void_p,  # Temp2
    ctypes.c_int,     # B
    ctypes.c_int,     # L
    ctypes.c_int      # r
]

def fused_temp_temp2(Ninv, Del, Temp, Temp2):
    """
    Ninv: (B, L)      float32, CuPy
    Del : (B, L, r)   float32, CuPy
    Temp: (B, L, r)   float32, CuPy (preallocated)
    Temp2: (B, r, r)  float32, CuPy (preallocated, must be zeroed before call)
    """
    B, L = Ninv.shape
    r = Del.shape[2]

    assert Ninv.dtype == cp.float32
    assert Del.dtype == cp.float32
    assert Temp.dtype == cp.float32
    assert Temp2.dtype == cp.float32

    temp_lib.fused_temp_temp2(
        ctypes.c_void_p(Ninv.data.ptr),
        ctypes.c_void_p(Del.data.ptr),
        ctypes.c_void_p(Temp.data.ptr),
        ctypes.c_void_p(Temp2.data.ptr),
        ctypes.c_int(B),
        ctypes.c_int(L),
        ctypes.c_int(r),
    )





def tri_inv_3x3(L, xp):
    """
    Analytic inverse of a 3x3 lower-triangular matrix (or batch of them).

    Parameters
    ----------
    L : array_like, shape (..., 3, 3)
        Lower-triangular factors (e.g. Cholesky factors).
    xp : module
        numpy or cupy.

    Returns
    -------
    Linv : array_like, shape (..., 3, 3)
        Inverses of L, lower-triangular, same dtype/device as L.
    """
    orig_shape = L.shape
    assert orig_shape[-2:] == (3, 3), "tri_inv_3x3 expects (..., 3, 3) input"

    # Flatten batch dims into one
    Lr = L.reshape(-1, 3, 3)  # (B_flat, 3, 3)
    # Extract the non-zero entries of the lower-triangular
    L00 = Lr[:, 0, 0]
    L10 = Lr[:, 1, 0]
    L20 = Lr[:, 2, 0]
    L11 = Lr[:, 1, 1]
    L21 = Lr[:, 2, 1]
    L22 = Lr[:, 2, 2]

    # Diagonal elements of the inverse
    i00 = 1.0 / L00
    i11 = 1.0 / L11
    i22 = 1.0 / L22

    # Off-diagonal elements
    i10 = -L10 * i00 * i11
    # (l10*l21 - l11*l20)/(l00*l11*l22)
    i20 = (L10 * L21 - L11 * L20) / (L00 * L11 * L22)
    i21 = -L21 * i11 * i22

    # Assemble Linv, still lower-triangular
    Linv = xp.zeros_like(Lr)
    Linv[:, 0, 0] = i00
    Linv[:, 1, 0] = i10
    Linv[:, 2, 0] = i20
    Linv[:, 1, 1] = i11
    Linv[:, 2, 1] = i21
    Linv[:, 2, 2] = i22

    return Linv.reshape(orig_shape)



def inverse_covariance_v2(N, Del, Sig, xp, ret_det=False, N_is_inv=True,
                          I_del=None, I_sig=None):
    """
    Same math as inverse_covariance, but:
      * avoid xp.linalg.inv on 3x3 triangular matrices
      * optionally reuse identity matrices if provided
    """
    B, L, r_eig = Del.shape
    r_src = Sig.shape[2]

    # --- handle noise inverse ---
    if N_is_inv:
        N_inv = N
    else:
        N_inv = 1.0 / N   # (B, L)

    Ninv_col = N_inv[..., None]   # (B, L, 1)

    # --- identity caches (so we don't reallocate every call) ---
    if I_del is None:
        I_del = xp.eye(r_eig, dtype=Del.dtype)[None, ...]  # (1, 3, 3)
    if I_sig is None:
        I_sig = xp.eye(r_src, dtype=Sig.dtype)             # (3, 3)

    # ====================
    # 1. Diffuse block
    # ====================
    temp = Ninv_col * Del                                   # (B, L, r_eig)
    temp2 = xp.matmul(Del.transpose(0, 2, 1), temp)         # (B, r_eig, r_eig)

    L_del = xp.linalg.cholesky(I_del + temp2)               # (B, 3, 3)

    # Avoid xp.linalg.inv on (B,3,3)
    L_del_inv = tri_inv_3x3(L_del, xp)                      # (B, 3, 3)
    L_del_inv_H = L_del_inv.transpose(0, 2, 1).conj()       # (B, 3, 3)

    Del_prime = xp.matmul(temp, L_del_inv_H)                # (B, L, r_eig)

    # ====================
    # 2. Source block
    # ====================
    A = Ninv_col * Sig                                      # (B, L, r_src)
    Bmat = xp.matmul(Sig.transpose(0, 2, 1), Del_prime)     # (B, r_src, r_eig)

    W = A - xp.matmul(Del_prime, Bmat.transpose(0, 2, 1).conj())  # (B, L, r_src)

    term1 = xp.matmul(A.transpose(0, 2, 1).conj(), Sig)          # (B, r_src, r_src)
    term2 = xp.matmul(Bmat, Bmat.transpose(0, 2, 1).conj())      # (B, r_src, r_src)

    K_sig = I_sig + xp.sum(term1, axis=0) - xp.sum(term2, axis=0)  # (r_src, r_src)

    L_sig = xp.linalg.cholesky(K_sig)                           # (3, 3)

    # Instead of xp.linalg.inv(L_sig), use tri_inv_3x3 on a length-1 batch
    L_sig_inv = tri_inv_3x3(L_sig[None, ...], xp)[0]            # (3, 3)
    L_sig_inv_H = L_sig_inv.T.conj()                            # (3, 3)

    Sig_prime = xp.matmul(W, L_sig_inv_H[None, ...])            # (B, L, r_src)

    # ====================
    # 3. logdet if requested
    # ====================
    if ret_det:
        logdet = 2 * (
            xp.sum(xp.diagonal(xp.log(L_del), axis1=1, axis2=2)) +
            xp.sum(xp.diagonal(xp.log(L_sig)))
        )
        return logdet, N_inv, Del_prime, Sig_prime

    return N_inv, Del_prime, Sig_prime





"""SCRATCH BASED PREALLOC"""
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

import cupy as cp

import cupy as cp

class InverseCovScratch:
    def __init__(self, B, L, r_eig, r_src, xp=cp):
        self.xp = xp

        # Main workspaces, all fp32
        self.temp      = xp.empty((B, L, r_eig), dtype=xp.float32)
        self.temp2     = xp.empty((B, r_eig, r_eig), dtype=xp.float32)
        self.L_del     = xp.empty((B, r_eig, r_eig), dtype=xp.float32)
        self.Del_prime = xp.empty((B, L, r_eig), dtype=xp.float32)

        self.A  = xp.empty((B, L, r_src), dtype=xp.float32)
        self.B  = xp.empty((B, r_src, r_eig), dtype=xp.float32)
        self.W  = xp.empty((B, L, r_src), dtype=xp.float32)
        self.L_sig = xp.empty((r_src, r_src), dtype=xp.float32)

        # Precomputed *contiguous* transposes (to be filled once)
        self.Del_T = xp.empty((B, r_eig, L), dtype=xp.float32)  # Del^T
        self.Sig_T = xp.empty((B, r_src, L), dtype=xp.float32)  # Sig^H

        # Identity matrices
        self.I_del = xp.eye(r_eig, dtype=xp.float32)[None, :, :]
        self.I_sig = xp.eye(r_src, dtype=xp.float32)


def inverse_covariance_prealloc(N, Del, Sig, scratch, ret_det=False, N_is_inv=True):
    xp = scratch.xp
    B, L, r_eig = Del.shape
    r_src = Sig.shape[2]

    # Ensure fp32
    N   = N.astype(xp.float32, copy=False)
    Del = Del.astype(xp.float32, copy=False)
    Sig = Sig.astype(xp.float32, copy=False)

    # 1. Noise inverse
    if N_is_inv:
        N_inv = N
    else:
        N_inv = 1.0 / N

    Ninv_col = N_inv[..., None]  # (B, L, 1)

    # 2. temp = N_inv * Del
    scratch.temp[...] = Ninv_col * Del  # (B, L, r_eig)

    # 3. temp2 = Del^T @ temp  (use precomputed Del_T = (B, r_eig, L))
    xp.matmul(
        scratch.Del_T,          # (B, r_eig, L)
        scratch.temp,           # (B, L, r_eig)
        out=scratch.temp2       # (B, r_eig, r_eig)
    )

    # 4. L_del = chol(I + temp2)
    scratch.L_del[...] = xp.linalg.cholesky(scratch.I_del + scratch.temp2)

    # 5. Del_prime = temp @ inv(L_del)^H
    L_del_inv_H = xp.linalg.inv(scratch.L_del).transpose(0, 2, 1).conj()
    xp.matmul(scratch.temp, L_del_inv_H, out=scratch.Del_prime)  # (B, L, r_eig)

    # 6. A = N_inv * Sig
    scratch.A[...] = Ninv_col * Sig  # (B, L, r_src)

    # 7. B = Sig^H @ Del_prime (use precomputed Sig_T = (B, r_src, L))
    xp.matmul(
        scratch.Sig_T.conj(),   # (B, r_src, L)
        scratch.Del_prime,      # (B, L, r_eig)
        out=scratch.B           # (B, r_src, r_eig)
    )

    # 8. W = A - Del_prime @ B^H
    xp.matmul(
        scratch.Del_prime,                      # (B, L, r_eig)
        scratch.B.transpose(0, 2, 1).conj(),    # (B, r_eig, r_src)
        out=scratch.W                           # (B, L, r_src)
    )
    scratch.W[...] = scratch.A - scratch.W

    # 9. L_sig = chol(I + sum(A^H Sig) - sum(B B^H))
    term1 = xp.matmul(
        scratch.A.transpose(0, 2, 1).conj(),    # (B, r_src, L)
        Sig                                     # (B, L, r_src)
    )                                           # -> (B, r_src, r_src)

    term2 = xp.matmul(
        scratch.B,                              # (B, r_src, r_eig)
        scratch.B.transpose(0, 2, 1).conj()     # (B, r_eig, r_src)
    )                                           # -> (B, r_src, r_src)

    K_sig = scratch.I_sig + xp.sum(term1, axis=0) - xp.sum(term2, axis=0)
    scratch.L_sig[...] = xp.linalg.cholesky(K_sig)  # (r_src, r_src)

    # 10. Sig_prime = W @ inv(L_sig)^H
    L_sig_inv_H = xp.linalg.inv(scratch.L_sig).T.conj()  # (r_src, r_src)
    Sig_prime = scratch.W @ L_sig_inv_H                 # (B, L, r_src)

    # 11. log(det)
    if ret_det:
        logdet = 2 * (
            xp.sum(xp.diagonal(xp.log(scratch.L_del), axis1=1, axis2=2)) +
            xp.sum(xp.diagonal(xp.log(scratch.L_sig)))
        )
        return logdet, N_inv, scratch.Del_prime, Sig_prime

    return N_inv, scratch.Del_prime, Sig_prime

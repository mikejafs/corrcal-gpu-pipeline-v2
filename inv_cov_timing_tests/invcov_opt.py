# import numpy as np
import ctypes
import cupy as cp
import seaborn as sns
import matplotlib.pyplot as plt
import corrcal
from cupyx.profiler import benchmark
import cupyx.scipy.linalg as cpx_linalg


chol_lib = ctypes.cdll.LoadLibrary("/home/mike/corrcal_gpu_pipeline/inv_cov_timing_tests/batched3_chol.so")
siglib = ctypes.cdll.LoadLibrary("/home/mike/corrcal_gpu_pipeline/inv_cov_timing_tests/sigprime_fused.so")

chol_lib.batched_cholesky_3x3.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int
]
siglib.fused_sigprime.argtypes = [
    ctypes.c_void_p,  # W
    ctypes.c_void_p,  # Lsig
    ctypes.c_void_p,  # SigPrime
    ctypes.c_int,     # B
    ctypes.c_int      # L
]

def fused_sigprime(W, Lsig):
    B, L, r = W.shape
    assert r == 3

    out = cp.empty_like(W)

    siglib.fused_sigprime(
        ctypes.c_void_p(W.data.ptr),
        ctypes.c_void_p(Lsig.data.ptr),
        ctypes.c_void_p(out.data.ptr),
        ctypes.c_int(B),
        ctypes.c_int(L)
    )

    return out


del_lib = ctypes.cdll.LoadLibrary(
    "/home/mike/corrcal_gpu_pipeline/inv_cov_timing_tests/corrected_fused_delprime.so"
)

del_lib.corrected_fused_delprime.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int
]

def fused_delprime(temp, Ldel):
    B, L, r = temp.shape
    out = cp.empty_like(temp)
    del_lib.corrected_fused_delprime(
        temp.data.ptr,
        Ldel.data.ptr,
        out.data.ptr,
        ctypes.c_int(B),
        ctypes.c_int(L)
    )
    return out


def batched_cholesky_3x3(arr):
    """
    arr must be shape (B, 3, 3), double, contiguous
    """
    B = arr.shape[0]
    ptr = arr.data.ptr

    chol_lib.batched_cholesky_3x3(
        ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double)),
        int(B)
    )
    # cp.cuda.Stream.null.synchronize()

# assert arr.flags.c_contiguous




class InverseCovScratch:
    def __init__(self, B, L, r_eig, r_src, xp):
        self.xp = xp

        # Temporary buffers for Del-side
        self.temp = xp.empty((B, L, r_eig))
        self.temp2 = xp.empty((B, r_eig, r_eig))
        self.L_del = xp.empty((B, r_eig, r_eig))
        self.Del_prime = xp.empty((B, L, r_eig))

        # Temporary buffers for Sig-side
        self.A = xp.empty((B, L, r_src))
        self.B = xp.empty((B, r_src, r_eig))
        self.W = xp.empty((B, L, r_src))
        self.L_sig = xp.empty((r_src, r_src))   # ONLY ONE — matches your original code
                                                # (note: this is NOT batched)

        # Identity matrices
        self.I_del = xp.eye(r_eig)[None, :, :]
        self.I_sig = xp.eye(r_src)



def inverse_covariance_prealloc(N, Del, Sig, scratch, ret_det=False, N_is_inv=True):
    xp = scratch.xp
    B, L, r_eig = Del.shape
    r_src = Sig.shape[2]

    # ----------------------------
    # 1. Noise inverse
    # ----------------------------
    if N_is_inv:
        N_inv = N
    else:
        N_inv = 1.0 / N

    Ninv_col = N_inv[..., None]   # (B, L, 1)

    # ----------------------------
    # 2. temp = N^{-1} * Delta
    # ----------------------------
    scratch.temp[...] = Ninv_col * Del

    # ----------------------------
    # 3. temp2 = Δᵀ temp
    # ----------------------------
    xp.matmul(
        Del.transpose(0, 2, 1),
        scratch.temp,
        out=scratch.temp2
    )

    # ----------------------------
    # 4. L_del = chol(I + temp2)
    # ----------------------------
    # scratch.L_del[...] = xp.linalg.cholesky(scratch.I_del + scratch.temp2)

    # Ensure contiguous memory (broadcast makes it non-contiguous)
    tmp = scratch.I_del + scratch.temp2
    scratch.temp2 = cp.ascontiguousarray(tmp)

    batched_cholesky_3x3(scratch.temp2)

    scratch.L_del[...] = scratch.temp2

    # ----------------------------
    # 5. Del_prime = temp @ inv(L_del)ᵀ*
    # ----------------------------
    # L_del_inv_H = xp.linalg.inv(scratch.L_del).transpose(0, 2, 1).conj()
    # xp.matmul(scratch.temp, L_del_inv_H, out=scratch.Del_prime)

    scratch.Del_prime[...] = fused_delprime(scratch.temp, scratch.L_del)


    # ----------------------------
    # 6. A = N^{-1} * Sig
    # ----------------------------
    scratch.A[...] = Ninv_col * Sig

    # ----------------------------
    # 7. B = Sigᵀ Δ'
    # ----------------------------
    xp.matmul(
        Sig.transpose(0, 2, 1),
        scratch.Del_prime,
        out=scratch.B
    )

    # ----------------------------
    # 8. W = A - Δ' Bᵀ
    # ----------------------------
    xp.matmul(
        scratch.Del_prime,
        scratch.B.transpose(0, 2, 1).conj(),
        out=scratch.W
    )
    scratch.W[...] = scratch.A - scratch.W

    # ----------------------------
    # 9. L_sig = chol(I + sum(Aᵀ Sig) - sum(B Bᵀ))
    # IMPORTANT: exactly matching your ORIGINAL math
    # ----------------------------
    # term1 = xp.matmul(
    #     scratch.A.transpose(0, 2, 1).conj(),
    #     Sig
    # )
    # term2 = xp.matmul(
    #     scratch.B,
    #     scratch.B.transpose(0, 2, 1).conj()
    # )

    # K_sig = scratch.I_sig + xp.sum(term1, axis=0) - xp.sum(term2, axis=0)

    # scratch.L_sig[...] = xp.linalg.cholesky(K_sig)


    # termAHS = sum_i A_i^H @ Sig_i
    AHS = xp.matmul(
        scratch.A.transpose(0, 2, 1).conj(),   # (B, r_src, L)
        Sig                                     # (B, L, r_src)
    )
    termAHS = xp.sum(AHS, axis=0)               # (r_src, r_src)

    # termBB = sum_i B_i @ B_i^H
    BBH = xp.matmul(
        scratch.B,                              # (B, r_src, r_eig)
        scratch.B.transpose(0, 2, 1).conj()     # (B, r_eig, r_src)
    )
    termBB = xp.sum(BBH, axis=0)                # (r_src, r_src)

    # Final K_sig
    K_sig = scratch.I_sig + termAHS - termBB

    # Cholesky
    scratch.L_sig[...] = xp.linalg.cholesky(K_sig)


    # ----------------------------
    # 10. Sig_prime = W @ inv(L_sig)ᵀ*
    # ----------------------------
    # L_sig_inv_H = xp.linalg.inv(scratch.L_sig).T.conj()
    # Sig_prime = scratch.W @ L_sig_inv_H

    Sig_prime = fused_sigprime(scratch.W, scratch.L_sig)


    # ----------------------------
    # 11. logdet (matching your exact expression)
    # ----------------------------
    if ret_det:
        logdet = 2 * (
            xp.sum(xp.diagonal(xp.log(scratch.L_del), axis1=1, axis2=2)) +
            xp.sum(xp.diagonal(xp.log(scratch.L_sig)))
        )
        # cp.cuda.Stream.null.synchronize()

        return logdet, N_inv, scratch.Del_prime, Sig_prime

    return N_inv, scratch.Del_prime, Sig_prime

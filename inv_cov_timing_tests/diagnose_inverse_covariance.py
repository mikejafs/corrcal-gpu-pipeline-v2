import cupy as cp
import numpy as np
import ctypes

from invcov_opt import (
    inverse_covariance_prealloc,
    InverseCovScratch,
    batched_cholesky_3x3,
    fused_delprime,
)
from zp_puregpu_funcs_py import zeroPad
from simulate_params import SimCorrcalParams
from corrcal import SparseCov

# =========================
# Setup small test instance
# =========================
n_ant = 40
n_eig = 3
n_src = 3

sp = SimCorrcalParams(n_ant, n_eig, n_src, precision="float64", xp=cp)
edges = sp.edges()

noise, diff, src, *_ = sp.sim_data()

zp_noise, _, _ = zeroPad(noise, edges, return_inv=True)
zp_diff,  _, _ = zeroPad(diff,  edges, return_inv=False)
zp_src,   _, _ = zeroPad(src,   edges, return_inv=False)

B, L, _ = zp_diff.shape

print("\n=== SHAPES ===")
print("noise:", zp_noise.shape)
print("diff :", zp_diff.shape)
print("src  :", zp_src.shape)

scratch = InverseCovScratch(B, L, n_eig, n_src, xp=cp)

print("\n=== MEMORY LAYOUT ===")
for name, arr in [
    ("temp", scratch.temp),
    ("temp2", scratch.temp2),
    ("L_del", scratch.L_del),
    ("W", scratch.W),
]:
    print(
        f"{name}: C-contiguous? {arr.flags.c_contiguous}, "
        f"shape={arr.shape}, ptr={hex(arr.data.ptr)}"
    )


# ========================================================
# 1. Compute temp and L_del EXACTLY like real pipeline does
# ========================================================

# temp = N^{-1} * Del
scratch.temp[...] = zp_noise[..., None] * zp_diff

# temp2 = Δᵀ temp
cp.matmul(zp_diff.transpose(0, 2, 1), scratch.temp, out=scratch.temp2)

# K = I + temp2
K = scratch.temp2 + scratch.I_del

# Make contiguous BEFORE Cholesky
K = cp.ascontiguousarray(K)

# Apply GPU batched Cholesky
batched_cholesky_3x3(K)

# Store L_del
scratch.L_del[...] = K

# Show user that L_del is actually triangular
print("\n=== L_del STRUCTURE CHECK ===")
L0 = cp.asnumpy(scratch.L_del[0])
print("L_del[0]:\n", L0)
print("Is lower triangular? ", np.allclose(np.tril(L0), L0))
print("Is upper triangular? ", np.allclose(np.triu(L0), L0))
print("L @ L.T (should reproduce K[0]):")
print(L0 @ L0.T)


# ====================================================
# 2. Δ′ sanity check comparing block 0 CPU vs GPU ONLY
# ====================================================
print("\n=== Δ′ DIAGNOSTIC ===")

# CPU reference (block 0)
temp_cpu = cp.asnumpy(scratch.temp[0])
L_cpu   = cp.asnumpy(scratch.L_del[0])
L_inv   = np.linalg.inv(L_cpu)
cpu_dp0 = temp_cpu @ L_inv.T

# GPU fused kernel
gpu_dp = fused_delprime(scratch.temp, scratch.L_del)
gpu_dp0 = cp.asnumpy(gpu_dp[0])

print("CPU first row:", cpu_dp0[0])
print("GPU first row:", gpu_dp0[0])
print("Difference norm:", np.linalg.norm(cpu_dp0 - gpu_dp0))


# =====================================
# 3. FULL PIPELINE CPU/GPU MATCH TEST
# =====================================
print("\n=== FULL CORRCAL MATCH TEST ===")

logdet_gpu, Ninvgpu, Dpgpu, Spgpu = inverse_covariance_prealloc(
    zp_noise, zp_diff, zp_src, scratch, ret_det=True, N_is_inv=True
)

# CPU version
noise_cpu = cp.asnumpy(noise)
diff_cpu  = cp.asnumpy(diff)
src_cpu   = cp.asnumpy(src)
edges_cpu = cp.asnumpy(edges)

cov = SparseCov(noise_cpu, src_cpu, diff_cpu, edges_cpu, n_eig, isinv=False)
cinv, logd_cpu = cov.inv(return_det=True)

# Compare
print("logdet match?     ", np.allclose(logd_cpu, cp.asnumpy(logdet_gpu)))

print("diff_mat match?   ",
      np.allclose(cinv.diff_mat, cp.asnumpy(Dpgpu)))

print("src_mat match?    ",
      np.allclose(cinv.src_mat, cp.asnumpy(Spgpu)))

import ctypes
import os

chol_lib = ctypes.cdll.LoadLibrary("/home/mike/corrcal_gpu_pipeline/inv_cov_timing_tests/batched3_chol.so")

print("Loaded .so at:", chol_lib._name)
print("Functions:", dir(chol_lib))
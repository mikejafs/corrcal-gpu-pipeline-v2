import ctypes
import time
import numpy as np
import cupy as cp
import seaborn as sns
import matplotlib.pyplot as plt
from cupyx.profiler import benchmark

full_path = "/home/mike/corrcal_gpu_pipeline/rough dev work/zp_pure_gpu/zp_puregpu_1d.so"

zp_cuda_lib = ctypes.cdll.LoadLibrary(full_path)

zp_cuda_lib.zeroPad.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_long),
    ctypes.c_int,
    ctypes.c_int
]

def zeroPad(array, edges):
    array = cp.array(array, dtype=cp.double)
    edges = cp.array(edges, dtype=cp.int64)
    largest_block = cp.array(cp.diff(edges).max(), dtype = cp.int32)
    n_blocks = cp.array(edges.size - 1, dtype = cp.int32)
    largest_block = int(largest_block.get())
    n_blocks = int(n_blocks.get())

    out_array = cp.zeros((n_blocks*largest_block), dtype = cp.double)

    zp_cuda_lib.zeroPad(
        ctypes.cast(array.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(out_array.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(edges.data.ptr, ctypes.POINTER(ctypes.c_long)),
        n_blocks,
        largest_block
    )
    cp.cuda.Stream.null.synchronize()
    return out_array, largest_block, n_blocks

def run(benchmark_zp, return_zp, return_plot):
    n_bl = 120000
    n_ant = 500

    #for more easy to verify case:
    # array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    # edges = np.array([0, 2, 9, 14])

    #or:
    # array = np.random.randint(1, 9, size=n_bl)

    #can use this array along with the seaborn heatmap
    #to even more easily check things are working
    # array = cp.full(n_bl, 1, dtype=cp.double)

    #Set up random array & some simulated edges array
    array = cp.random.rand(n_bl)
    edges = cp.unique(cp.random.randint(1, n_bl - 1, size = n_ant))
    edges = cp.concatenate((cp.array([0]), edges, cp.array([n_bl], dtype = cp.int64)))

    if return_zp:
        zp_array, largest_block, n_blocks = zeroPad(array, edges)
        zp_array = zp_array.reshape(n_blocks, largest_block)
        print(zp_array)

    if return_plot:
        try:
            zp_array
        except NameError:
            raise NameError("You need to return the zp_array if you want to make a plot of it")
        else:
            fig, ax = plt.subplots(figsize = (14, 10))        
            sns.heatmap(zp_array, ax = ax)
            plt.pause(5)
            plt.close(fig)


    if benchmark_zp:
        test_results = str(benchmark(zeroPad, (array, edges), n_repeat=1000))
        test_results = test_results.split()
        cpu_t = float(test_results[3])/1e6
        gpu_t = float(test_results[14])/1e6
        print(f"Time on cpu: {cpu_t:.6f}s")
        print(f"Time on gpu: {gpu_t:.6f}s")

if __name__ == "__main__":
    run(benchmark_zp=True, return_zp=False, return_plot=False)


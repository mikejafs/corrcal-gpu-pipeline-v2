import ctypes
import cupy as cp
import seaborn as sns
import matplotlib.pyplot as plt

full_path = "/home/mike/corrcal_gpu_pipeline/rough dev work/zp_pure_gpu/zp_final/zp_puregpu_funcs.so"

zp_cuda_lib = ctypes.cdll.LoadLibrary(full_path)

zp_cuda_lib.zeroPad1d.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_long),
    ctypes.c_int,
    ctypes.c_int
]
zp_cuda_lib.zeroPad2d.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_long),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]

def zeroPad1d(array, edges):
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
    # cp.cuda.Stream.null.synchronize()
    return out_array, largest_block, n_blocks

def zeroPad2d(array, edges):
    array = cp.array(array, dtype=cp.double)
    edges = cp.array(edges, dtype=cp.int64)
    array_cols = array.shape[1]
    largest_block = cp.array(cp.diff(edges).max(), dtype = cp.int32)
    n_blocks = cp.array(edges.size - 1, dtype = cp.int32)
    largest_block = int(largest_block.get())
    n_blocks = int(n_blocks.get())

    out_array = cp.zeros((n_blocks*largest_block*array_cols), dtype = cp.double)

    zp_cuda_lib.zeroPad(
        ctypes.cast(array.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(out_array.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(edges.data.ptr, ctypes.POINTER(ctypes.c_long)),
        array_cols,
        n_blocks,
        largest_block
    )
    # cp.cuda.Stream.null.synchronize()
    return out_array, largest_block, n_blocks

def zeroPad(array, edges):
    array = cp.array(array, dtype=cp.double)
    edges = cp.array(edges, dtype=cp.int64)
    largest_block = cp.array(cp.diff(edges).max(), dtype = cp.int32)
    n_blocks = cp.array(edges.size - 1, dtype = cp.int32)
    largest_block = int(largest_block.get())
    n_blocks = int(n_blocks.get())

    if array.ndim == 1: 
        out_array = cp.zeros((n_blocks*largest_block), dtype = cp.double)
        zp_cuda_lib.zeroPad1d(
            ctypes.cast(array.data.ptr, ctypes.POINTER(ctypes.c_double)),
            ctypes.cast(out_array.data.ptr, ctypes.POINTER(ctypes.c_double)),
            ctypes.cast(edges.data.ptr, ctypes.POINTER(ctypes.c_long)),
            n_blocks,
            largest_block
        )
        # cp.cuda.Stream.null.synchronize()
    else:
        array_cols = array.shape[1]
        out_array = cp.zeros((n_blocks*largest_block*array_cols), dtype = cp.double)
        zp_cuda_lib.zeroPad2d(
            ctypes.cast(array.data.ptr, ctypes.POINTER(ctypes.c_double)),
            ctypes.cast(out_array.data.ptr, ctypes.POINTER(ctypes.c_double)),
            ctypes.cast(edges.data.ptr, ctypes.POINTER(ctypes.c_long)),
            array_cols,
            n_blocks,
            largest_block
        )
        # cp.cuda.Stream.null.synchronize()
    return out_array, largest_block, n_blocks

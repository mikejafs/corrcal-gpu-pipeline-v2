import ctypes
import cupy as cp
import seaborn as sns
import matplotlib.pyplot as plt

full_path = "/home/mike/corrcal_gpu_pipeline/pipeline/zp_puregpu_funcs.so"

zp_cuda_lib = ctypes.cdll.LoadLibrary(full_path)

zp_cuda_lib.zeroPad1d.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_long),
    ctypes.c_int,
    ctypes.c_int
]
zp_cuda_lib.undo_zeroPad1d.argtypes = [
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
zp_cuda_lib.undo_zeroPad2d.argtypes = [
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

def zeroPad(array, edges, return_inv):
    """
    Could be a good idea to reshape everything to 2d and 3d before returning
    so that the user doesn't have to reshape manually. Need to check if this 
    makes sense given the current inv covariance routine...
    """
    array = cp.array(array, dtype=cp.double)
    edges = cp.array(edges, dtype=cp.int64)
    largest_block = cp.array(cp.diff(edges).max(), dtype = cp.int32)
    n_blocks = cp.array(edges.size - 1, dtype = cp.int32)
    largest_block = int(largest_block.get())
    n_blocks = int(n_blocks.get())

    if return_inv:
        array = 1/array
    else:
        pass

    if array.ndim == 1: 
        out_array = cp.zeros((n_blocks*largest_block), dtype = cp.double)
        zp_cuda_lib.zeroPad1d(
            ctypes.cast(array.data.ptr, ctypes.POINTER(ctypes.c_double)),
            ctypes.cast(out_array.data.ptr, ctypes.POINTER(ctypes.c_double)),
            ctypes.cast(edges.data.ptr, ctypes.POINTER(ctypes.c_long)),
            n_blocks,
            largest_block
        )
        cp.cuda.Stream.null.synchronize()
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
        out_array = out_array.reshape(n_blocks*largest_block, array_cols)
        cp.cuda.Stream.null.synchronize()

    return out_array, largest_block, n_blocks

    
def undo_zeroPad(array, edges):
    array = cp.array(array, dtype=cp.double)
    edges = cp.array(edges, dtype=cp.int64)
    largest_block = cp.array(cp.diff(edges).max(), dtype = cp.int32)
    n_blocks = cp.array(edges.size - 1, dtype = cp.int32)
    largest_block = int(largest_block.get())
    n_blocks = int(n_blocks.get())

    if array.shape[2] == 1:
        array = array.reshape(n_blocks*largest_block)
        out_array = cp.zeros((int(edges[-1])), dtype = cp.double)
        zp_cuda_lib.undo_zeroPad1d(
            ctypes.cast(array.data.ptr, ctypes.POINTER(ctypes.c_double)),
            ctypes.cast(out_array.data.ptr, ctypes.POINTER(ctypes.c_double)),
            ctypes.cast(edges.data.ptr, ctypes.POINTER(ctypes.c_long)),
            n_blocks,
            largest_block
        )
        cp.cuda.Stream.null.synchronize()
    else:
        array_cols = array.shape[2]
        array = array.reshape(n_blocks*largest_block*array_cols)
        # print(array)
        out_array = cp.zeros((int(edges[-1]), array_cols), dtype = cp.double)
        zp_cuda_lib.undo_zeroPad2d(
            ctypes.cast(array.data.ptr, ctypes.POINTER(ctypes.c_double)),
            ctypes.cast(out_array.data.ptr, ctypes.POINTER(ctypes.c_double)),
            ctypes.cast(edges.data.ptr, ctypes.POINTER(ctypes.c_long)),
            array_cols,
            n_blocks,
            largest_block
        )
        cp.cuda.Stream.null.synchronize()

    return out_array

    
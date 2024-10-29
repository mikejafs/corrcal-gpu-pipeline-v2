import numpy as np
import ctypes
from . import gpu_funcs

def zeroPad1d(array, edges):
    array = np.array(array, dtype=np.double)
    edges = np.array(edges, dtype=np.int64)
    array_size = array.shape[0] 
    largest_block = np.array(np.diff(edges).max(), dtype = np.int32)
    n_blocks = np.array(edges.size - 1, dtype = np.int32)

    out_array = np.zeros((n_blocks*largest_block), dtype = np.double)

    gpu_funcs.zeroPad1d(
        array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        edges.ctypes.data_as(ctypes.POINTER(ctypes.c_long)),
        array_size,
        n_blocks,
        largest_block
    )
    return out_array, largest_block, n_blocks


def zeroPad2d(array, edges):
    array = np.array(array, dtype=np.double)
    edges = np.array(edges, dtype=np.int64)
    array_rows = array.shape[0]
    array_cols = array.shape[1]
    largest_block = np.array(np.diff(edges).max(), dtype = np.int32)
    n_blocks = np.array(edges.size - 1, dtype = np.int32)
    
    out_array = np.zeros((n_blocks*largest_block*array_cols), dtype = np.double)

    gpu_funcs.zeroPad2d(
        array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        edges.ctypes.data_as(ctypes.POINTER(ctypes.c_long)),
        array_rows,
        array_cols,
        n_blocks,
        largest_block
    )
    return out_array, largest_block, n_blocks


import ctypes
import cupy as cp
import seaborn as sns  # if you actually use these elsewhere
import matplotlib.pyplot as plt

# Update this path to wherever you put the new .so
full_path = "/home/mike/corrcal_gpu_pipeline/invcov_perf/zp_puregpu_funcs_fp32.so"

zp_cuda_lib = ctypes.cdll.LoadLibrary(full_path)

# --- function signatures (float32) ---
zp_cuda_lib.zeroPad1d_f32.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_long),
    ctypes.c_int,
    ctypes.c_int
]
zp_cuda_lib.undo_zeroPad1d_f32.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_long),
    ctypes.c_int,
    ctypes.c_int
]
zp_cuda_lib.zeroPad2d_f32.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_long),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]
zp_cuda_lib.undo_zeroPad2d_f32.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_long),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]

"""
Standalone 1d and 2d versions (float32).
Mostly for testing; main entrypoint is zeroPad_f32 / undo_zeroPad_f32.
"""
def zeroPad1d_f32(array, edges):
    array = cp.array(array, dtype=cp.float32)
    edges = cp.array(edges, dtype=cp.int64)

    largest_block = cp.array(cp.diff(edges).max(), dtype=cp.int64)
    n_blocks = cp.array(edges.size - 1, dtype=cp.int64)
    largest_block = int(largest_block.get())
    n_blocks = int(n_blocks.get())

    out_array = cp.zeros((n_blocks * largest_block), dtype=cp.float32)

    zp_cuda_lib.zeroPad1d_f32(
        ctypes.cast(array.data.ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(out_array.data.ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(edges.data.ptr, ctypes.POINTER(ctypes.c_long)),
        n_blocks,
        largest_block
    )
    return out_array, largest_block, n_blocks


def zeroPad2d_f32(array, edges):
    array = cp.array(array, dtype=cp.float32)
    edges = cp.array(edges, dtype=cp.int64)

    array_cols = array.shape[1]
    largest_block = cp.array(cp.diff(edges).max(), dtype=cp.int64)
    n_blocks = cp.array(edges.size - 1, dtype=cp.int64)
    largest_block = int(largest_block.get())
    n_blocks = int(n_blocks.get())

    out_array = cp.zeros((n_blocks * largest_block * array_cols), dtype=cp.float32)

    zp_cuda_lib.zeroPad2d_f32(
        ctypes.cast(array.data.ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(out_array.data.ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(edges.data.ptr, ctypes.POINTER(ctypes.c_long)),
        array_cols,
        n_blocks,
        largest_block
    )
    return out_array, largest_block, n_blocks


def zeroPad_f32(array, edges, return_inv):
    """
    Zeropads an input matrix according to the largest block in the diffuse
    sky covariance matrix, in float32.

    Params
    ------
    array : 1D or 2D-like
    edges : block boundary indices
    return_inv : bool
        If True, replace array with 1/array before padding (for noise).

    Returns
    -------
    out_array : float32
        For 1D input: (n_blocks, largest_block)
        For 2D input: (n_blocks, largest_block, array_cols)
    largest_block : int
    n_blocks : int
    """
    array = cp.array(array, dtype=cp.float32)
    edges = cp.array(edges, dtype=cp.int64)

    largest_block = cp.array(cp.diff(edges).max(), dtype=cp.int64)
    n_blocks = cp.array(edges.size - 1, dtype=cp.int64)
    largest_block = int(largest_block.get())
    n_blocks = int(n_blocks.get())

    if return_inv:
        array = 1.0 / array
    else:
        pass

    if array.ndim == 1:
        out_array = cp.zeros((n_blocks * largest_block), dtype=cp.float32)
        zp_cuda_lib.zeroPad1d_f32(
            ctypes.cast(array.data.ptr, ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(out_array.data.ptr, ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(edges.data.ptr, ctypes.POINTER(ctypes.c_long)),
            n_blocks,
            largest_block
        )
        out_array = out_array.reshape(n_blocks, largest_block)
        cp.cuda.Stream.null.synchronize()
    else:
        array_cols = array.shape[1]
        out_array = cp.zeros((n_blocks * largest_block * array_cols),
                             dtype=cp.float32)
        zp_cuda_lib.zeroPad2d_f32(
            ctypes.cast(array.data.ptr, ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(out_array.data.ptr, ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(edges.data.ptr, ctypes.POINTER(ctypes.c_long)),
            array_cols,
            n_blocks,
            largest_block
        )
        out_array = out_array.reshape(n_blocks, largest_block, array_cols)
        cp.cuda.Stream.null.synchronize()

    return out_array, largest_block, n_blocks


def undo_zeroPad_f32(array, edges, ReImsplit=False):
    """
    Inverse of zeroPad_f32, returning the original (float32) matrix.

    Parameters
    ----------
    array : result of zeroPad_f32
        2D or 3D CuPy array.
    edges : block edges
    ReImsplit : bool
        Same semantics as your original function.
    """
    array = cp.array(array, dtype=cp.float32)
    edges = cp.array(edges, dtype=cp.int64)

    largest_block = cp.array(cp.diff(edges).max(), dtype=cp.int32)
    n_blocks = cp.array(edges.size - 1, dtype=cp.int32)
    largest_block = int(largest_block.get())
    n_blocks = int(n_blocks.get())

    if ReImsplit:
        # full baseline count
        n_bl = int(edges[-1])
    else:
        largest_block = int(largest_block / 2)
        n_bl = int(edges[-1] / 2)
        edges = edges // 2

    if array.ndim == 2:
        array = array.reshape(n_blocks * largest_block)
        out_array = cp.zeros(n_bl, dtype=cp.float32)
        zp_cuda_lib.undo_zeroPad1d_f32(
            ctypes.cast(array.data.ptr, ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(out_array.data.ptr, ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(edges.data.ptr, ctypes.POINTER(ctypes.c_long)),
            n_blocks,
            largest_block
        )
        cp.cuda.Stream.null.synchronize()
    else:
        array_cols = array.shape[2]
        array = array.reshape(n_blocks * largest_block * array_cols)
        out_array = cp.zeros((int(edges[-1]), array_cols), dtype=cp.float32)
        zp_cuda_lib.undo_zeroPad2d_f32(
            ctypes.cast(array.data.ptr, ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(out_array.data.ptr, ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(edges.data.ptr, ctypes.POINTER(ctypes.c_long)),
            array_cols,
            n_blocks,
            largest_block
        )
        cp.cuda.Stream.null.synchronize()

    return out_array


# Optional convenience aliases so your existing code that imports
# zeroPad / undo_zeroPad still works, but now in float32.
zeroPad = zeroPad_f32
undo_zeroPad = undo_zeroPad_f32

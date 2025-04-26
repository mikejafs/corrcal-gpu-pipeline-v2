"""
Module to house the importing of the shared library for accumulate gradient cuda function
"""

import ctypes
import cupy as cp

full_path = "/home/mike/Thesis_rough_work/gradient_programming/grad2/populate_grad2.so"

pop_grad_lib = ctypes.cdll.LoadLibrary(full_path)

pop_grad_lib.populate_gradient.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_long),
    ctypes.POINTER(ctypes.c_long),
    ctypes.c_int,
    ctypes.c_int
]
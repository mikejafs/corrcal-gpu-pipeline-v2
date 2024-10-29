#TODO: Combine the functions from run_zp 1 and 2 D into one 
#function with some logic to determine which cuda cp function
#should be used depending on the shape of the input array

import ctypes as ct
from pathlib import Path
import subprocess

#Compile CUDA library
subprocess.run(["python3", "build_cuda.py"])

#locate and load .so file
lib_dir = Path(__file__).parent / "src"
lib_path = lib_dir / "gpu_funcs.so"   #NEED TO SET THIS TO WHATEVER THE .SO FILE IS

if not lib_path.exists():
    raise FileNotFoundError(f"{lib_path} does not exist. Compilation may have failed.")

zp_cuda_lib = ct.cdll.LoadLibrary(str(lib_path))
print("Shared lib load successful")

zp_cuda_lib.zeroPad1d.argtypes = [
    ct.POINTER(ct.c_double),
    ct.POINTER(ct.c_double),
    ct.POINTER(ct.c_long),
    ct.c_int,
    ct.c_int,
    ct.c_int
]

zp_cuda_lib.zeroPad2d.argtypes = [
    ct.POINTER(ct.c_double),
    ct.POINTER(ct.c_double),
    ct.POINTER(ct.c_long),
    ct.c_int,
    ct.c_int,
    ct.c_int,
    ct.c_int
]


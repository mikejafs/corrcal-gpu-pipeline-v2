import subprocess
from pathlib import Path

def compile_cuda():
    lib_dir = Path(__file__).parent / "src"
    lib_path = lib_dir / "gpu_funcs.so"
    cuda_src = lib_dir / "gpu_funcs.cu"

    lib_dir.mkdir(parents=True, exist_ok=True)

    if not lib_path.exists():
        print(f"Compiling {cuda_src} to {lib_path}...")
        result = subprocess.run([
            "nvcc", "-shared", "-o", str(lib_path), str(cuda_src)
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print("Error during compilation:")
            print(result.stderr)
            raise RuntimeError("CUDA compilation failed")
        else:
            print("Compilation successful!")
    else:
        print(f"{lib_path} already exists. Skippign compilation")

if __name__ == "__main__":
    compile_cuda()

    
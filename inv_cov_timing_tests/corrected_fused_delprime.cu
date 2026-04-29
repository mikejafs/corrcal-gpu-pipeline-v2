// nvcc -O3 -arch=sm_86 -shared -Xcompiler -fPIC corrected_fused_delprime.cu -o corrected_fused_delprime.so

#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {

//
// Analytic inverse of a *row-major* 3×3 LOWER TRIANGULAR matrix:
//
//   L = [ L00   L01(=0)  L02(=0)
//         L10   L11      L12(=0)
//         L20   L21      L22      ]
//
// Stored in row-major memory as:
//   [ L00 L01 L02  L10 L11 L12  L20 L21 L22 ]
//
// NumPy/CuPy use this semantics when interpreting a row-major array that
// has upper entries = 0.
//
// This produces an inverse L_inv that ALSO obeys row-major lower-triangular layout.
//
__device__ __forceinline__
void invert_lower3_rowmajor(const double* L, double* Linv)
{
    // Load lower-triangular entries
    double L00 = L[0];
    double L01 = L[1];   // = 0
    double L02 = L[2];   // = 0

    double L10 = L[3];
    double L11 = L[4];
    double L12 = L[5];   // = 0

    double L20 = L[6];
    double L21 = L[7];
    double L22 = L[8];

    // Compute inverse of lower-triangular matrix (standard forward substitution)
    double i00 = 1.0 / L00;

    double i10 = -L10 * i00 / L11;
    double i20 = -(L20 * i00 + L21 * i10) / L22;

    double i11 = 1.0 / L11;
    double i21 = -L21 * i11 / L22;

    double i22 = 1.0 / L22;

    // Store back in row-major lower-triangular form
    Linv[0] = i00;  Linv[1] = 0.0;  Linv[2] = 0.0;
    Linv[3] = i10;  Linv[4] = i11;  Linv[5] = 0.0;
    Linv[6] = i20;  Linv[7] = i21;  Linv[8] = i22;
}


//
// Fused Δ′ kernel
// Computes:  Δ′[b,l,:] = temp[b,l,:] @ (L^{-1})ᵀ
//
// Row-major shapes:
//   temp:     (B, L, 3)
//   Ldel:     (B, 3, 3)
//   DelPrime: (B, L, 3)
//
// All arrays are packed row-major.
//
__global__
void corrected_fused_delprime_kernel(
    const double* __restrict__ temp,     // (B*L*3)
    const double* __restrict__ Ldel,     // (B*9)
    double* __restrict__ DelPrime,       // (B*L*3)
    int B,
    int L)
{
    int b = blockIdx.x;
    int l = blockIdx.y * blockDim.x + threadIdx.x;

    if (b >= B || l >= L) return;

    // Shared memory for one inverse per block (transpose included)
    __shared__ double LinvT[9];

    // First thread computes inverse of Ldel[b]
    if (threadIdx.x == 0)
    {
        const double* Lptr = &Ldel[b * 9];

        double Linv[9];
        invert_lower3_rowmajor(Lptr, Linv);

        // Store transpose (L_inv^T) in shared space
        LinvT[0] = Linv[0];
        LinvT[1] = Linv[3];
        LinvT[2] = Linv[6];

        LinvT[3] = Linv[1];
        LinvT[4] = Linv[4];
        LinvT[5] = Linv[7];

        LinvT[6] = Linv[2];
        LinvT[7] = Linv[5];
        LinvT[8] = Linv[8];
    }

    __syncthreads();

    // Flattened index into row-major temp[b,l,:]
    int idx = (b * L + l) * 3;

    double t0 = temp[idx + 0];
    double t1 = temp[idx + 1];
    double t2 = temp[idx + 2];

    // Multiply temp[b,l,:] by (L^{-1})ᵀ
    double o0 = t0 * LinvT[0] + t1 * LinvT[1] + t2 * LinvT[2];
    double o1 = t0 * LinvT[3] + t1 * LinvT[4] + t2 * LinvT[5];
    double o2 = t0 * LinvT[6] + t1 * LinvT[7] + t2 * LinvT[8];

    DelPrime[idx + 0] = o0;
    DelPrime[idx + 1] = o1;
    DelPrime[idx + 2] = o2;
}


//
// C interface
//
void corrected_fused_delprime(
    const double* temp,
    const double* Ldel,
    double* DelPrime,
    int B,
    int L)
{
    dim3 block(256);
    dim3 grid(B, (L + block.x - 1) / block.x);

    corrected_fused_delprime_kernel<<<grid, block>>>(
        temp, Ldel, DelPrime, B, L
    );

    cudaDeviceSynchronize();
}

} // extern "C"

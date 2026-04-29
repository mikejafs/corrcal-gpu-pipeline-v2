//nvcc -O3 -arch=sm_86 -shared -Xcompiler -fPIC sigprime_fused.cu -o sigprime_fused.so

#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {

//
// Analytically invert a 3×3 lower triangular matrix
// L = [ l00  0    0
//       l10  l11  0
//       l20  l21  l22 ]
// Compute L_inv (also lower triangular)
//
__device__ __forceinline__
void invert_lower3(const double* L, double* Linv)
{
    double l00 = L[0];
    double l10 = L[1];
    double l20 = L[2];
    double l11 = L[4];
    double l21 = L[5];
    double l22 = L[8];

    double i00 = 1.0 / l00;
    double i11 = 1.0 / l11;
    double i22 = 1.0 / l22;

    double i10 = -l10 * i00 * i11;
    double i20 = -(l20 * i00 + l21 * i10) * i22;
    double i21 = -l21 * i11 * i22;

    Linv[0] = i00;  Linv[1] = 0.0;  Linv[2] = 0.0;
    Linv[3] = i10;  Linv[4] = i11;  Linv[5] = 0.0;
    Linv[6] = i20;  Linv[7] = i21;  Linv[8] = i22;
}


//
// Fused kernel: computes Sig_prime = W @ L_sig_inv^T
// Input shapes:
//   W:        (B, L, r_src)  = (B, L, 3)
//   L_sig:    (3, 3)
// Output:
//   Sig_prime: (B, L, 3)
//
// All in row-major
//
__global__
void fused_sigprime_kernel(
    const double* __restrict__ W,        // (B*L*3)
    const double* __restrict__ Lsig,     // (3*3)
    double* __restrict__ SigPrime,       // (B*L*3)
    int B, int L)
{
    int b = blockIdx.x;
    int l = blockIdx.y * blockDim.x + threadIdx.x;

    if (b >= B || l >= L) return;

    // Shared memory: store L_sig_inv^T once per block
    __shared__ double LinvT[9];

    if (threadIdx.x == 0) {
        double Linv[9];
        invert_lower3(Lsig, Linv);

        // store transpose of Linv
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

    int idx = (b * L + l) * 3;

    double w0 = W[idx + 0];
    double w1 = W[idx + 1];
    double w2 = W[idx + 2];

    double o0 =
        w0 * LinvT[0] +
        w1 * LinvT[1] +
        w2 * LinvT[2];

    double o1 =
        w0 * LinvT[3] +
        w1 * LinvT[4] +
        w2 * LinvT[5];

    double o2 =
        w0 * LinvT[6] +
        w1 * LinvT[7] +
        w2 * LinvT[8];

    SigPrime[idx + 0] = o0;
    SigPrime[idx + 1] = o1;
    SigPrime[idx + 2] = o2;
}


// C-callable launch function
void fused_sigprime(
    const double* W,
    const double* Lsig,
    double* SigPrime,
    int B,
    int L)
{
    dim3 grid(B);
    dim3 block(256);
    dim3 grid2(B, (L + block.x - 1) / block.x);

    fused_sigprime_kernel<<<grid2, block>>>(W, Lsig, SigPrime, B, L);
    cudaDeviceSynchronize();
}

} // extern "C"

//nvcc -O3 -shared -o batched3_chol.so batched_cholesky_3x3.cu -Xcompiler -fPIC


extern "C" {

__global__ void chol3x3_kernel(double* __restrict__ A, int B)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B) return;

    // Row-major pointer to this 3x3 block
    double* M = A + idx * 9;

    // Load matrix entries in row-major order
    double a00 = M[0];
    double a01 = M[1];
    double a02 = M[2];
    double a11 = M[4];
    double a12 = M[5];
    double a22 = M[8];

    // --- Cholesky decomposition (lower tri) ---
    double L00 = sqrt(a00);

    double L10 = a01 / L00;
    double L20 = a02 / L00;

    double L11 = sqrt(a11 - L10 * L10);

    double L21 = (a12 - L20 * L10) / L11;

    double L22 = sqrt(a22 - L20 * L20 - L21 * L21);

    // --- Store back in row-major ---
    M[0] = L00;  M[1] = 0.0;  M[2] = 0.0;
    M[3] = L10;  M[4] = L11;  M[5] = 0.0;
    M[6] = L20;  M[7] = L21;  M[8] = L22;
}

void batched_cholesky_3x3(double* A, int B)
{
    int threads = 128;
    int blocks = (B + threads - 1) / threads;

    chol3x3_kernel<<<blocks, threads>>>(A, B);
}

} // extern "C"

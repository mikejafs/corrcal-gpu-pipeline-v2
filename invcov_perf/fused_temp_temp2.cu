// nvcc -O3 -arch=sm_86 -shared -Xcompiler -fPIC fused_temp_temp2.cu -o fused_temp_temp2.so

#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {

__global__
void fused_temp_temp2_kernel(
    const float* __restrict__ Ninv,   // (B, L)
    const float* __restrict__ Del,    // (B, L, r)
    float* __restrict__ Temp,         // (B, L, r)
    float* __restrict__ Temp2,        // (B, r, r)  (accumulator)
    int B,
    int L,
    int r
){
    int b = blockIdx.x;
    int l = blockIdx.y * blockDim.x + threadIdx.x;

    if (b >= B || l >= L) return;

    // Flat index for (b,l,:)
    int base_bl = (b * L + l) * r;

    // Load N_inv[b,l]
    float ninv = Ninv[b * L + l];

    // local storage for one row of Del and Temp
    // r is small (e.g. 3), so this is fine
    float d_local[16];
    float t_local[16];

    // compute Temp(b,l,:) = N_inv(b,l) * Del(b,l,:)
    for (int k = 0; k < r; ++k) {
        float d = Del[base_bl + k];
        float t = ninv * d;
        d_local[k] = d;
        t_local[k] = t;
        Temp[base_bl + k] = t;
    }

    // accumulate Temp2(b,i,j) += Del(b,l,i) * Temp(b,l,j)
    int base_b = b * r * r;
    for (int i = 0; i < r; ++i) {
        float di = d_local[i];
        int row_base = base_b + i * r;
        for (int j = 0; j < r; ++j) {
            float contrib = di * t_local[j];
            atomicAdd(&Temp2[row_base + j], contrib);
        }
    }
}

void fused_temp_temp2(
    const float* Ninv,
    const float* Del,
    float* Temp,
    float* Temp2,
    int B,
    int L,
    int r
){
    dim3 block(256);
    dim3 grid(B, (L + block.x - 1) / block.x);

    fused_temp_temp2_kernel<<<grid, block>>>(
        Ninv, Del, Temp, Temp2, B, L, r
    );
    cudaDeviceSynchronize();
}

} // extern "C"

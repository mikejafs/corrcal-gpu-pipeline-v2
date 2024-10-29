//nvcc -shared -o zp_cuda_1d.so zp_puregpu_1d.cu -Xcompiler -fPIC

#include <stdio.h>

extern "C"
{
    __global__ void zeroPad_kernel(
        double* in_array,
        double* out_array,
        long* edges,
        int n_blocks,
        int largest_block
    ){
        int blockidx = blockIdx.x*blockDim.x + threadIdx.x;
        int idx = blockIdx.y*blockDim.y + threadIdx.y;
        if (blockidx < n_blocks){   
            long start = edges[blockidx];
            long stop = edges[blockidx + 1];
            long block_size = stop - start;
            if (idx < block_size){
                out_array[blockidx*largest_block + idx] = in_array[start + idx];
            }
        }
    }

    void zeroPad(
        double* in_array,
        double* out_array,
        long* edges,
        // int in_array_size,
        int n_blocks,
        int largest_block
    ){
        //define thread and threadblock sizes & launch kernel
        //Note the prblm is in 2D, so we need a grid of threads
        dim3 threadsPerBlock(32, 32);
        dim3 numBlocks((n_blocks + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                        (largest_block + threadsPerBlock.y - 1) / threadsPerBlock.y);

        zeroPad_kernel<<<numBlocks, threadsPerBlock>>>(
            in_array,
            out_array,
            edges,
            n_blocks,
            largest_block
            );
    }
}
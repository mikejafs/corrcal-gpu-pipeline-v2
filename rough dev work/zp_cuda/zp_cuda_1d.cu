//nvcc -shared -o zp_cuda_1d.so zp_cuda_1d.cu -Xcompiler -fPIC

#include <stdio.h>

extern "C"
{
    __global__ void zeroPad_kernel(
        double* in_array,
        double* out_array,
        long* edges,
        int in_array_size,
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
        int in_array_size,
        int n_blocks,
        int largest_block
    ){
        //define device variables
        double *d_in_array, *d_out_array;
        long* d_edges;

        //allocate memory on the device
        size_t in_array_bytes = in_array_size * sizeof(double);
        size_t out_array_bytes = n_blocks * largest_block * sizeof(double);
        size_t edges_bytes = (n_blocks + 1) * sizeof(long);

        cudaMalloc(&d_edges, edges_bytes);
        cudaMalloc(&d_in_array, in_array_bytes);
        cudaMalloc(&d_out_array, out_array_bytes);

        //copy data from host to device
        cudaMemcpy(d_in_array, in_array, in_array_bytes, cudaMemcpyHostToDevice);
        // cudaMemcpy(d_out_array, out_array, out_array_bytes, cudaMemcpyHostToDevice); // don't need this since cudaMalloc seems to initialize to 0 
        cudaMemcpy(d_edges, edges, edges_bytes, cudaMemcpyHostToDevice);

        //define thread and threadblock sizes & launch kernel
        //Note the prblm is in 2D, so we need a grid of threads
        dim3 threadsPerBlock(32, 32);
        dim3 numBlocks((n_blocks + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                        (largest_block + threadsPerBlock.y - 1) / threadsPerBlock.y);

        zeroPad_kernel<<<numBlocks, threadsPerBlock>>>(
            d_in_array,
            d_out_array,
            d_edges,
            in_array_size,
            n_blocks,
            largest_block 
            );
        cudaDeviceSynchronize();

        //copy data back to the host
        cudaMemcpy(out_array, d_out_array, out_array_bytes, cudaMemcpyDeviceToHost);

        //free memory on the device
        cudaFree(d_in_array);
        cudaFree(d_out_array);
    }
}
//nvcc -shared -o zp_cuda_2d.so zp_cuda_2d.cu -Xcompiler -fPIC

#include <stdio.h>

//nvcc -shared -o zp_cuda_1d.so zp_cuda_1d.cu -Xcompiler -fPIC

#include <stdio.h>

extern "C"
{
    __global__ void zeroPad1d_kernel(
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

    __global__ void zeroPad2d_kernel(
        double* in_array,
        double* out_array,
        long* edges,
        int in_array_cols,
        int n_blocks,
        int largest_block
    ){
        int blockidx = blockIdx.x*blockDim.x + threadIdx.x;
        int row_idx = blockIdx.y*blockDim.y + threadIdx.y;
        int col_idx = blockIdx.z*blockDim.z + threadIdx.z;

        // printf("test1");
        if (blockidx < n_blocks){
            long start = edges[blockidx];
            long stop = edges[blockidx + 1];
            long block_size = stop - start;
            if (row_idx < block_size){
                // printf("test");
                if (col_idx < in_array_cols){
                // for (int col_idx = 0; col_idx < in_array_cols; col_idx ++){
                    out_array[in_array_cols*(blockidx*largest_block + row_idx) + col_idx] 
                    = in_array[in_array_cols*(start + row_idx) + col_idx];

                }
            } 
        }
    }


    void zeroPad1d(
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

        zeroPad1d_kernel<<<numBlocks, threadsPerBlock>>>(
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


    void zeroPad2d(
        double* in_array,
        double* out_array,
        long* edges,
        int in_array_rows,
        int in_array_cols,
        int n_blocks,
        int largest_block
    ){
        //define device variables
        double *d_in_array, *d_out_array;
        long* d_edges;

        //allocate memory on the device
        size_t in_array_bytes = in_array_rows * in_array_cols * sizeof(double);
        size_t out_array_bytes = n_blocks * largest_block * in_array_cols * sizeof(double);
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

        //TODO: Understand my gpu architecture and per considerations well enough
        //to be able to choose the correct values for threadsperblock in each
        //dim & to know if this layout of numBlocks is the best way to do things
        dim3 threadsPerBlock(8, 8, in_array_cols);
        dim3 numBlocks((n_blocks + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                        (largest_block + threadsPerBlock.y - 1) / threadsPerBlock.y,
                        (in_array_cols + threadsPerBlock.z - 1) / threadsPerBlock.z);
         
        zeroPad2d_kernel<<<numBlocks, threadsPerBlock>>>(
            d_in_array,
            d_out_array,
            d_edges,
            in_array_cols,
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

// nvcc -shared -o zp_puregpu_funcs_fp32.so zp_puregpu_funcs_fp32.cu -Xcompiler -fPIC
#include <stdio.h>

extern "C"
{
    __global__ void zeroPad1d_kernel_f32(
        float* in_array,
        float* out_array,
        long* edges,
        int n_blocks,
        int largest_block
    ){
        int blockidx = blockIdx.x * blockDim.x + threadIdx.x;
        int idx      = blockIdx.y * blockDim.y + threadIdx.y;

        if (blockidx < n_blocks){
            long start = edges[blockidx];
            long stop  = edges[blockidx + 1];
            long block_size = stop - start;

            if (idx < block_size){
                out_array[blockidx * largest_block + idx] =
                    in_array[start + idx];
            }
        }
    }

    __global__ void undo_zeroPad1d_kernel_f32(
        float* in_array,
        float* out_array,
        long* edges,
        int n_blocks,
        int largest_block
    ){
        int blockidx = blockIdx.x * blockDim.x + threadIdx.x;
        int idx      = blockIdx.y * blockDim.y + threadIdx.y;

        if (blockidx < n_blocks){
            long start = edges[blockidx];
            long stop  = edges[blockidx + 1];
            long block_size = stop - start;

            if (idx < block_size){
                out_array[start + idx] =
                    in_array[blockidx * largest_block + idx];
            }
        }
    }

    __global__ void zeroPad2d_kernel_f32(
        float* in_array,
        float* out_array,
        long* edges,
        int in_array_cols,
        int n_blocks,
        int largest_block
    ){
        int blockidx = blockIdx.x * blockDim.x + threadIdx.x;
        int row_idx  = blockIdx.y * blockDim.y + threadIdx.y;
        int col_idx  = blockIdx.z * blockDim.z + threadIdx.z;

        if (blockidx < n_blocks){
            long start = edges[blockidx];
            long stop  = edges[blockidx + 1];
            long block_size = stop - start;

            if (row_idx < block_size && col_idx < in_array_cols){
                out_array[in_array_cols * (blockidx * largest_block + row_idx) + col_idx] =
                    in_array[in_array_cols * (start + row_idx) + col_idx];
            }
        }
    }

    __global__ void undo_zeroPad2d_kernel_f32(
        float* in_array,
        float* out_array,
        long* edges,
        int in_array_cols,
        int n_blocks,
        int largest_block
    ){
        int blockidx = blockIdx.x * blockDim.x + threadIdx.x;
        int row_idx  = blockIdx.y * blockDim.y + threadIdx.y;
        int col_idx  = blockIdx.z * blockDim.z + threadIdx.z;

        if (blockidx < n_blocks){
            long start = edges[blockidx];
            long stop  = edges[blockidx + 1];
            long block_size = stop - start;

            if (row_idx < block_size && col_idx < in_array_cols){
                out_array[in_array_cols * (start + row_idx) + col_idx] =
                    in_array[in_array_cols * (blockidx * largest_block + row_idx) + col_idx];
            }
        }
    }

    // -------------------------
    // Host wrappers (float32)
    // -------------------------
    void zeroPad1d_f32(
        float* in_array,
        float* out_array,
        long* edges,
        int n_blocks,
        int largest_block
    ){
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks(
            (n_blocks      + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (largest_block + threadsPerBlock.y - 1) / threadsPerBlock.y
        );

        zeroPad1d_kernel_f32<<<numBlocks, threadsPerBlock>>>(
            in_array,
            out_array,
            edges,
            n_blocks,
            largest_block
        );
    }

    void undo_zeroPad1d_f32(
        float* in_array,
        float* out_array,
        long* edges,
        int n_blocks,
        int largest_block
    ){
        dim3 threadsPerBlock(8, 8);
        dim3 numBlocks(
            (n_blocks      + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (largest_block + threadsPerBlock.y - 1) / threadsPerBlock.y
        );

        undo_zeroPad1d_kernel_f32<<<numBlocks, threadsPerBlock>>>(
            in_array,
            out_array,
            edges,
            n_blocks,
            largest_block
        );
    }

    void zeroPad2d_f32(
        float* in_array,
        float* out_array,
        long* edges,
        int in_array_cols,
        int n_blocks,
        int largest_block
    ){
        // Same launch shape as your FP64 version
        dim3 threadsPerBlock(8, 8, in_array_cols);
        dim3 numBlocks(
            (n_blocks      + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (largest_block + threadsPerBlock.y - 1) / threadsPerBlock.y,
            (in_array_cols + threadsPerBlock.z - 1) / threadsPerBlock.z
        );

        zeroPad2d_kernel_f32<<<numBlocks, threadsPerBlock>>>(
            in_array,
            out_array,
            edges,
            in_array_cols,
            n_blocks,
            largest_block
        );
        cudaDeviceSynchronize();
    }

    void undo_zeroPad2d_f32(
        float* in_array,
        float* out_array,
        long* edges,
        int in_array_cols,
        int n_blocks,
        int largest_block
    ){
        dim3 threadsPerBlock(8, 8, in_array_cols);
        dim3 numBlocks(
            (n_blocks      + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (largest_block + threadsPerBlock.y - 1) / threadsPerBlock.y,
            (in_array_cols + threadsPerBlock.z - 1) / threadsPerBlock.z
        );

        undo_zeroPad2d_kernel_f32<<<numBlocks, threadsPerBlock>>>(
            in_array,
            out_array,
            edges,
            in_array_cols,
            n_blocks,
            largest_block
        );
        cudaDeviceSynchronize();
    }
}

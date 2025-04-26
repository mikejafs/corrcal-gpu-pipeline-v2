//nvcc -shared -o populate_grad2.so populate_grad2.cu -Xcompiler -fPIC

#include <stdio.h>

extern "C"
{
    __global__ void populate_gradient_kernel(
        double *gains,
        double *s,
        double *t,
        double *P,
        double *noise,
        double *outr,
        double *outi,
        long *ant_1_inds,
        long *ant_2_inds,
        int n_ant,
        int n_bl
    ){
        int k = blockIdx.x*blockDim.x + threadIdx.x;
        if (k < n_bl){
            int k1 = ant_1_inds[k];
            int k2 = ant_2_inds[k];
    
            // Compute the product of complex gains.
            double G_kr = gains[2*k1]*gains[2*k2] + gains[2*k1+1]*gains[2*k2+1];
            double G_ki = gains[2*k1+1]*gains[2*k2] - gains[2*k1]*gains[2*k2+1];
    
            // Compute the prefactor for the trace contribution.
            double prefac = noise[2*k] * P[k] / (G_kr*G_kr + G_ki*G_ki);

            // Fill out the dLdG matrices
            outr[k1*n_ant + k2] = prefac*G_kr - s[k];
            outi[k1*n_ant + k2] = prefac*G_ki - t[k]; 
            outr[k2*n_ant + k1] = prefac*G_kr - s[k]; 
            outi[k2*n_ant + k1] =  - (prefac*G_ki - t[k]); 
            
        }
    }

    void populate_gradient(
        double *gains,
        double *s,
        double *t,
        double *P,
        double *noise,
        double *outr,
        double *outi,
        long *ant_1_inds,
        long *ant_2_inds,
        int n_ant,
        int n_bl
    ){
        int NUM_THREADS = 256;
        int NUM_BLOCKS = (n_bl + NUM_THREADS - 1)/NUM_THREADS;

        populate_gradient_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(
            gains,
            s,
            t,
            P,
            noise,
            outr,
            outi,
            ant_1_inds,
            ant_2_inds,
            n_ant,
            n_bl
        );
    }
}
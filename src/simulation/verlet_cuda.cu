/**
 * verlet_cuda.cu
 * 
 * Implementation of Verlet integration using CUDA
 **/

#include <stdio.h>
#include <cuda.h>

#include <helper_cuda.h>
#include "verlet_cuda.cuh"
#include "physics.h"
#include "../utils/array-utils.h"

__global__ integrate_position(x, v, h, n_bodies) {
    unsigned int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    
    // per avere n_bodies arbitrariamente grande e riusare i threads
    while (tid < n_bodies) {
        x[tid][0] += h * v[tid][0];
        x[tid][1] += h * v[tid][1];
        
        tid += blockDim.x * gridDim.x;
    }
}

__global__ compute_v(v, a, h, n_bodies) {
    unsigned int tid = (blockDim.x * blockIdx.x) + threadIcd.x;
    
    // per avere n_bodies arbitrariamente grande e riusare i threads
    while (tid < n_bodies) {
        v[tid][0] += 0.5 * h * a[tid][0];
        v[tid][1] += 0.5 * h * a[tid][1];
        
        tid += blockDim.x * gridDim.x;
    }
}

__global__ compute_a(M, x, a, n_bodies, eps) {
    unsigned int tid = (blockDim.x * blockIdx.x) + threadIcd.x;
    unsigned int j;
    
    // per avere n_bodies arbitrariamente grande e riusare i threads
    while (tid < n_bodies) {
        a[tid][0] = 0.0;
        a[tid][1] = 0.0;
        
        
        // calcolo accelerazione sul corpo i-esimo, data da tutti gli altri corpi
        for (j = 0; j < n_bodies; j++) {
            if (tid == j) {
                continue;
            }

            double dx = x[j][0] - x[tid][0];
            double dy = x[j][1] - x[tid][1];

            double distance = sqrt(dx * dx + dy * dy);
            double d3 = distance * distance * distance;
            double denom;

            if (d3 < 1e-15) {
                denom = d3 + eps;
            } else {
                denom = d3;
            }

            double a_i_val = (M[j] / denom);
            a[tid][0] += a_i_val * dx;
            a[tid][1] += a_i_val * dy;
        }
        
        tid += blockDim.x * gridDim.x;
    }
}

void verlet_cuda(double2 x[], double2 v[], double M[], const int n_bodies, const double h,
                 const int t_max, const double EPS, const int n_blocks, const int threads_per_block) {
    printf("\nAllocating GPU data...");
    double *dev_M, *dev_x, *dev_v, *dev_a;
    
    size_t M_size = n_bodies * sizeof(double);
    size_t f2_size = n_bodies * sizeof(double2);
    
    // alloco i vettori sulla GPU
    checkCudaErrors(cudaMalloc((void **)&dev_M, M_size));
    checkCudaErrors(cudaMalloc((void **)&dev_x, f2_size));
    checkCudaErrors(cudaMalloc((void **)&dev_v, f2_size));
    checkCudaErrors(cudaMalloc((void **)&dev_a, f2_size));
    printf("done");
    
    printf("\nCopying M, x and v vectors from RAM to GPU global memory...");
    checkCudaErrors(cudaMemcpy(dev_M, M, M_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_x, x, f2_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_v, v, f2_size, cudaMemcpyHostToDevice));
    printf("done");
    
    printf("\n\nBegin integration...compute initial acceleration...");
    // calcolo accelerazioni iniziali
    compute_a<<<n_blocks, threads_per_block>>>(dev_M, dev_x, dev_a, n_bodies, eps);
    // FIXME: AWAIT
    printf("done\n");
    
    // integra nel tempo
    printf("\nGo, go, go!")
    for (int t = 1; t <= t_max; t++) {
        if ((t % 100) == 0) {
            printf("\nCUDA execution at timestep %d...", t);
        }
        
        // compute velocities and new positions
        // v_t+1/2 = v_t + (1/2) * h * a_t
        // lancio kernel per calcolo velocità v_t+1/2
        compute_v<<<n_blocks, threads_per_block>>>(dev_v, dev_a, h, n_bodies);  // dopo questo passaggio, dev_v contiene le velocità aggiornate al tempo t + 1/2
        // FIXME: AWAIT
        
        // lancio kernel per calcolo posizione x_t+1
        integrate_position<<<n_blocks, threads_per_block>>>(dev_x, dev_v, h, n_bodies);  // ora x := x_t+1
        // FIXME: AWAIT
                
        // ora per calcolare la v_t+1 mi servono le nuove accelerazioni
        // lancio kernel per calcolo accelerazioni in posizioni x_t+1
        compute_a<<<n_blocks, threads_per_block>>>(dev_M, dev_x, dev_a, n_bodies, eps);
        // FIXME: AWAIT
                
        // lancio kernel per calcolo velocità v_t+1
        compute_v<<<>>>(dev_v, dev_a, h, n_bodies);
        // FIXME: AWAIT
        
        if ((t % 100) == 0) {
            printf("done timestep %d", t);
        }
    }

    // copia i risultati dal device GPU all'host CPU
    checkCudaErrors(cudaMemcpy(x, dev_x, f1_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(v, dev_v, f1_size, cudaMemcpyDeviceToHost));
    
    // libera tutta la memoria sul device GPU
    checkCudaErrors(cudaFree(dev_M));
    checkCudaErrors(cudaFree(dev_x));
    checkCudaErrors(cudaFree(dev_v));
    checkCudaErrors(cudaFree(dev_a));
}
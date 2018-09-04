/**
 * verlet_cuda.cu
 * 
 * Implementation of Verlet integration using CUDA
 **/

#include <stdio.h>
#include <cuda.h>
#include <helper_cuda.h>

#include "verlet_cuda.h"

__global__ void integrate_position(double2 x[], double2 v[], const double h, const int n_bodies) {
    unsigned int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    
    // to reuse threads in case n_bodies > n_threads_total
    while (tid < n_bodies) {
        x[tid].x += h * v[tid].x;
        x[tid].y += h * v[tid].y;
        
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void compute_v(double2 v[], double2 a[], const double h, const int n_bodies) {
    unsigned int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    
    // to reuse threads in case n_bodies > n_threads_total
    while (tid < n_bodies) {
        v[tid].x += 0.5 * h * a[tid].x;
        v[tid].y += 0.5 * h * a[tid].y;
        
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void compute_a(double M[], double2 x[], double2 a[], const int n_bodies, const double eps) {
    unsigned int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    unsigned int j;
    
    while (tid < n_bodies) {
        a[tid].x = 0.0;
        a[tid].y = 0.0;
        double2 r;
        
        // compute acceleration on the i-th body
        for (j = 0; j < n_bodies; j++) {
            if (tid == j) {
                continue;
            }

            r.x = x[j].x - x[tid].x;
            r.y = x[j].y - x[tid].y;
            
            double distance = sqrt(r.x * r.x + r.y * r.y + eps);
            double d3 = distance * distance * distance;
            
            d3 = M[j] / d3;

            a[tid].x += d3 * r.x;
            a[tid].y += d3 * r.y;
        }
        
        tid += blockDim.x * gridDim.x;
    }
}


float verlet_cuda(double2 x[], double2 v[], double M[], const int n_bodies, const double h,
                 const int t_max, const double EPS, const int n_blocks, const int threads_per_block) {
    printf("\nAllocating GPU data...");
    double *dev_M;
    double2 *dev_x, *dev_v, *dev_a;
    
    size_t M_size = n_bodies * sizeof(double);
    size_t d2_size = n_bodies * sizeof(double2);
    
    // alloco i vettori sulla GPU
    checkCudaErrors(cudaMalloc((void **)&dev_M, M_size));
    checkCudaErrors(cudaMalloc((void **)&dev_x, d2_size));
    checkCudaErrors(cudaMalloc((void **)&dev_v, d2_size));
    checkCudaErrors(cudaMalloc((void **)&dev_a, d2_size));
    printf("done");
    
    printf("\nCopying M, x and v vectors from RAM to GPU global memory...");
    checkCudaErrors(cudaMemcpy(dev_M, M, M_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_x, x, d2_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_v, v, d2_size, cudaMemcpyHostToDevice));
    printf("done");
    
    // create the events for timing the kernel
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    
    cudaEventRecord(start);
    
    
    printf("\n\nBegin integration...compute initial acceleration...");
    // calcolo accelerazioni iniziali
    compute_a<<<n_blocks, threads_per_block>>>(dev_M, dev_x, dev_a, n_bodies, EPS);
    printf("done\n");
    
    // integra nel tempo
    for (int t = 1; t <= t_max; t++) {
        if ((t % 100) == 0) {
            printf("\nCUDA execution at timestep %d...", t);
        }
        
        // compute velocities and new positions
        // v_t+1/2 = v_t + (1/2) * h * a_t
        // lancio kernel per calcolo velocità v_t+1/2
        compute_v<<<n_blocks, threads_per_block>>>(dev_v, dev_a, h, n_bodies);  // dopo questo passaggio, dev_v contiene le velocità aggiornate al tempo t + 1/2
        
        // lancio kernel per calcolo posizione x_t+1
        integrate_position<<<n_blocks, threads_per_block>>>(dev_x, dev_v, h, n_bodies);  // ora x := x_t+1
                
        // ora per calcolare la v_t+1 mi servono le nuove accelerazioni
        // lancio kernel per calcolo accelerazioni in posizioni x_t+1
        compute_a<<<n_blocks, threads_per_block>>>(dev_M, dev_x, dev_a, n_bodies, EPS);
                
        // lancio kernel per calcolo velocità v_t+1
        compute_v<<<n_blocks, threads_per_block>>>(dev_v, dev_a, h, n_bodies);
        // cudaDeviceSynchronize();  // così lancio sempre e solo 4 kernels in fila e non riempio la coda del device
        
        if ((t % 100) == 0) {
            printf("done timestep %d", t);
            cudaDeviceSynchronize();  // così lancio sempre e solo 4 kernels in fila e non riempio la coda del device
        }
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // compute elapsed time
    float elapsed_millisecs;
    cudaEventElapsedTime(&elapsed_millisecs, start, stop);
    
    // destroy the events - clean up behind me
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    // copia i risultati dal device GPU all'host CPU
    checkCudaErrors(cudaMemcpy(x, dev_x, d2_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(v, dev_v, d2_size, cudaMemcpyDeviceToHost));
    
    // libera tutta la memoria sul device GPU
    checkCudaErrors(cudaFree(dev_M));
    checkCudaErrors(cudaFree(dev_x));
    checkCudaErrors(cudaFree(dev_v));
    checkCudaErrors(cudaFree(dev_a));
    
    return elapsed_millisecs / 1000.0;
}
/**
 * verlet_cuda.cu
 * 
 * Implementation of Verlet integration using CUDA
 **/

#include <cuda.h>

#ifndef VERLET_CUDA_H
#define VERLET_CUDA_H

void verlet_cuda(double2 x[], double2 v[], double M[], const int n_bodies, const double h,
                 const int t_max, const double EPS, const int n_blocks, const int threads_per_block);

#endif
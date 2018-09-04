/**
 * main.cu
 * 
 * N-body gravitational simulation using CUDA
 **/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "omp.h"

#include "cli.h"
#include "txt_reader.h"
#include "verlet_cuda.h"


#define EPS 1e-15

int main(int argc, char const *argv[])
{
    // read all data and setup simulation
    int n_bodies, n_blocks, threads_per_block, t_max;
    double h;
    parse_cli_args(argc, argv, &n_bodies, &n_blocks, &threads_per_block, &t_max, &h);

    printf("\n\n----------------------");
    printf("\nReading input files...\n");
    printf("\nReading x coordinates...");
    double *x_coord = read_txt("x.txt", n_bodies);
    printf("done");

    printf("\nReading y coordinates...");
    double *y_coord = read_txt("y.txt", n_bodies);
    printf("done");

    printf("\nConverting x and y into position array double2 X...");
    double2 x[n_bodies];
    for (int i = 0; i < n_bodies; i++) {
        x[i].x = x_coord[i];
        x[i].y = y_coord[i];
    }
    free(x_coord);
    free(y_coord);
    printf("done");

    printf("\nReading v_x...");
    double *vx = read_txt("vx.txt", n_bodies);
    printf("done");

    printf("\nReading v_y...");
    double *vy = read_txt("vy.txt", n_bodies);
    printf("done");

    printf("\nConverting vx and vy into velocity array V...");
    double2 v[n_bodies];
    for (int i = 0; i < n_bodies; i++) {
        v[i].x = vx[i];
        v[i].y = vy[i];
    }
    free(vx);
    free(vy);
    printf("done");

    printf("\nReading mass file...");
    double *mass = read_txt("mass.txt", n_bodies);
    printf("done");

    printf("\nInit of mass array M...");
    double M[n_bodies];
    for (int i = 0; i < n_bodies; i++)
    {
        M[i] = mass[i];
    }
    free(mass);
    printf("done");


    // now the data is in HOST memory, I want to put it in DEVICE memory
    printf("\n\n--------------------------------------------------------------------");
    printf("\nStarting Verlet integrator using %d blocks of %d threads each...\n", n_blocks, threads_per_block);
    
    // create the events for timing the kernel
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float computing_seconds;
    
    cudaEventRecord(start);
    
    computing_seconds = verlet_cuda(x, v, M, n_bodies, h, t_max, EPS, n_blocks, threads_per_block);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // compute elapsed time
    float elapsed_millisecs;
    cudaEventElapsedTime(&elapsed_millisecs, start, stop);
    
    // destroy the events - clean up behind me
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    
    printf("\nVerlet integration finished!\n\n");
    printf("\nTotal time spent is: %f seconds\n\n", elapsed_millisecs / 1000.0);
    printf("\nComputing only time is %f seconds\n\n", computing_seconds);
    
    printf("\n\nWriting output file...");
    FILE *output_file;
    output_file = fopen("results_cuda_naive_with_overheads.csv", "a");

    if (output_file == NULL) {
        perror("Cannot open output file 'results_cuda_naive_with_overheads.csv'");
        fclose(output_file);
        return EXIT_FAILURE;
    }

    fprintf(output_file, "%f,%f,%d,%d,%d,%d\n", elapsed_millisecs / 1000.0, computing_seconds, n_bodies, t_max, n_blocks, threads_per_block);
    fclose(output_file);
    printf("done\n\n");
    
    return EXIT_SUCCESS;
}

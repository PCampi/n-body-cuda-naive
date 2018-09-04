/**
 * cli.c
 * 
 * Command line utilities
 **/
#include <stdio.h>
#include <stdlib.h>

void parse_cli_args(int argc, char const *argv[], int *n_bodies, int *n_blocks, int *threads_per_block, int *t_max, double *h) {
    if (argc != 6) {
        printf("\nWrong number of input arguments, should be 6, got %d\n", argc);
        exit(EXIT_FAILURE);
    }

    const char *str_num_bodies = argv[1];
    *n_bodies = atoi(str_num_bodies);
    if (*n_bodies < 1) {
        printf("\nCannot work with %d bodies. Use a number > 0.", *n_bodies);
        exit(EXIT_FAILURE);
    }

    const char *str_n_blocks = argv[2];
    *n_blocks = atoi(str_n_blocks);
    if (*n_blocks < 1) {
        printf("\nCannot work with %d blocks. Use a number >= 1.", *n_blocks);
        exit(EXIT_FAILURE);
    }
    
    const char *str_threads_per_blocks = argv[3];
    *threads_per_block = atoi(str_threads_per_blocks);
    if (*threads_per_block < 1 || *threads_per_block > 1024) {
        printf("\nCannot work with %d threads per block. Use a number 1 <= tpb <= 1024.", *threads_per_block);
        exit(EXIT_FAILURE);
    }

    const char *str_t_max = argv[4];
    *t_max = atoi(str_t_max);
    if (*t_max <= 0) {
        printf("\nCannot work with %d time points. Use a number >= 1.", *t_max);
        exit(EXIT_FAILURE);
    }

    const char *str_h = argv[5];
    *h = strtod(str_h, NULL);
    if (*h <= 0.0) {
        printf("\nCannot work with a step of %f, choose a number > 0.0", *h);
        exit(EXIT_FAILURE);
    }

    printf("\n\nSimulation parameters:\n-----------------------");
    printf("\nBody number: %d", *n_bodies);
    printf("\nCUDA blocks: %d", *n_blocks);
    printf("\nCUDA threads per block: %d", *threads_per_block);
    printf("\nTime steps: %d", *t_max);
    printf("\nIntegration h: %f", *h);
}

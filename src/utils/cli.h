/**
 * cli.h
 * 
 * Command line utilities
 **/
#ifndef CLI_UTILS_H_
#define CLI_UTILS_H_

void parse_cli_args(int argc, char const *argv[], int *n_bodies, int *n_blocks, int *threads_per_block, int *t_max, double *h);

#endif

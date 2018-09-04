/**
 * txt_reader.c
 * 
 * Implements txt reading facilities.
 **/

#include <stdio.h>
#include <stdlib.h>

#include "txt_reader.h"

double *read_txt(const char *filename, const int n_bodies) {
    int max_chars = 100;

    // open the file with checks
    FILE *fp;
    fp = fopen(filename, "r");

    if (fp == NULL) {
        perror("\nError opening file");
        exit(EXIT_FAILURE);
    }

    char line[max_chars];
    char *buf;
    double *result = (double *)malloc(n_bodies * sizeof(double));

    int i = 0;
    while (i < n_bodies && (buf = fgets(line, max_chars, fp)) != NULL) {
        result[i] = strtod(buf, NULL);
        i++;
    }

    fclose(fp);

    if (i < n_bodies) {
        printf("\nError: wanted n_bodies = %d but only got %d lines\n", n_bodies, i);
        exit(EXIT_FAILURE);
    }

    return result;
}

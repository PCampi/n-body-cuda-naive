/**
 * csv.c
 * 
 * Read a CSV file of doubles
 **/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "csv.h"

// FIXME: usare STRTOK di stdlib.h per splittare ricorsivamente su \n e su ,
double **read_csv(const char filename[]) {
    FILE *fp;
    fp = fopen(filename, "r");

    if (fp == NULL) {
        perror("\nError opening file...\n");
        exit(EXIT_FAILURE);
    }

    fclose(fp);
}

// char *line_to_tokens(const char *line, const char delimiter) {
//     int n_fields = 1;
//     int i = 0;
//     char ch;

//     while ((ch = line[i]) != NULL) {
//         if (ch == delimiter) {
//             n_fields++;
//         }
//         i++;
//     }
       
//     char *token;
//     char delimiters[1] = {delimiter};

//     if (n_fields == 1) {

//     }

//     while ((token = strtok(line, delimiters)) != NULL) {
        
//     }
// }

double* read_line(const char* line, const int n, const char sep) {
    // allocate space for the new row
    double *row = (double*) malloc(n * sizeof(double));
    // array of separators positions
    int sep_positions[n];
    // array of elements length
    int elements_len[n];

    // current char
    char current_char;
    int i = 0;
    int sep_index = 0;

    do {
        current_char = line[i];
        printf("\nLine[%d] = %c", i, current_char);

        if (current_char == sep || current_char == '\n') {
            if (sep_index < n) {
                printf("\nFound separator '%c' at index %d", current_char, i);
                sep_positions[sep_index] = i;
                printf("\nsep_positions[%d] = %d", sep_index, i);
                sep_index++;
            } // else it is error, should manage it...
        }
        
        i++;
    } while (current_char != '\0');


    for (int j = 0; j < n; j++) {
        if (j == 0) {
            elements_len[j] = sep_positions[j];
            printf("\nElement[%d] len is %d", j, elements_len[j]);
        } else {
            elements_len[j] = sep_positions[j] - sep_positions[j - 1];
            printf("\nElement[%d] len is %d", j, elements_len[j]);
        }
    }

    // finally read the elements
    int start = 0;
    int stop = 0;

    for (int j = 0; j < n; j++) {
        if (j == 0) {
            stop = elements_len[j] - 1;
            printf("\nCycle %d, start = %d, stop = %d", j, start, stop);
        } else {
            start = sep_positions[j - 1] + 1;
            stop = start + elements_len[j];
            printf("\nCycle %d, start = %d, stop = %d", j, start, stop);
        }

        char str[elements_len[j]];
        int l = 0;
        for (int k = start; k <= stop; k++) {
            str[l] = line[k];
            l++;
        }
        
        printf("\nElement[%d] = %s", j, str);

        row[j] = strtod(str, NULL);
    }

    return row;
}

// int main(void) {
//     char line[] = "2.41e+02,7.52\n";
//     int n = 2;
//     char sep = ',';

//     printf("\nConverting line\n");
//     double *row = read_line(line, n, sep);
//     printf("\n\nRow is: [");
//     for (int i = 0; i < n; i++) {
//         printf("%f, ", row[i]);
//     }
//     printf("]\n");
//     free(row);
// }

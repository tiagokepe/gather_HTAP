#include <stdlib.h>
#include <stdio.h>
#include "utils/defines.h"

void sum_two_cols(double *sum, const double *col1, const double *col2);

int main(int argc, char **argv) {
    double **table = (double**)malloc(sizeof(double *) * NUM_COLS);
    for(size_t i=0; i < NUM_COLS; ++i) {
        table[i] = (double *)malloc(sizeof(double) * NUM_TUPLES);
    }
 
    double *sum_col = (double *)malloc(sizeof(double) * NUM_TUPLES);
    sum_two_cols(sum_col, table[0], table[1]);

    for(size_t i=0; i < NUM_COLS; ++i) {
        free(table[i]);
    }
    free(table);
    return 0;
}

void sum_two_cols(double *sum, const double *col1, const double *col2) {
    for(size_t i=0; i < NUM_TUPLES; ++i) {
        sum[i] = col1[i] + col2[i];
    }
}


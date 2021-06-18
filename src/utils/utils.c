#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void print_sum_col(const double *col) {
    printf("COL:[");
    for(size_t i=0; i < NUM_TUPLES; ++i) {
        printf("%f", col[i]);
        if((i+1) != NUM_TUPLES) {
            printf(", ");
        }
    }
    printf("]\n");
}

void populate_table(double *table) {
    size_t col_idx;
    for(size_t i=0; i < NUM_TUPLES; ++i) {
        for(size_t j=0; j < NUM_COLS; ++j) {
            col_idx = i*NUM_COLS + j;
            table[col_idx] = (double)col_idx;
        }
    }
}

void print_table(const double *table) {
    size_t col_idx;
    for(size_t i=0; i < NUM_TUPLES; ++i) {
        printf("ROW[%ld]: [", i);
        for(size_t j=0; j < NUM_COLS; ++j) {
            col_idx = i*NUM_COLS + j;
            printf("%f", table[col_idx]);
            if((j+1) != NUM_COLS ) {
                printf(", ");
            }
        }
        printf("]\n");
    }
}

void print_m512d(const __m512d col_vec) {
    double col_val[NUM_TUPLES];
    memcpy(col_val, &col_vec, sizeof(col_val));

    printf("COL[ ");
    for(int i=7; i >= 0; --i) {
        printf("%f", col_val[i]);
        if(i > 0) {
            printf(", ");
        }
    }
    printf("\n");
}


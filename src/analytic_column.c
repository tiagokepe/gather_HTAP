#include <stdlib.h>
#include <stdio.h>
#include "utils/defines.h"

void sum_two_cols(double *sum, const double *col1, const double *col2);

#if DEBUG
void populate_table(double **table);
void print_table(double **table);
void print_column(const double *col);
#endif

int main(int argc, char **argv) {
    if(argc != 3) {
        printf("Usage: %s idx_col1 idx_col2\n", argv[0]);
        exit(1);
    }

    size_t idx_col1 = atoi(argv[1]);
    size_t idx_col2 = atoi(argv[2]);
    if(idx_col1 >= NUM_COLS || idx_col2 >= NUM_COLS) {
        printf("ERROR: Column indexes out of bound\n");
        exit(1);
    }

    double **table = (double**)malloc(sizeof(double *) * NUM_COLS);
    for(size_t i=0; i < NUM_COLS; ++i) {
        table[i] = (double *)malloc(sizeof(double) * NUM_TUPLES);
    }

    #if DEBUG
    populate_table(table);
    print_table(table);
    #endif

    double *sum_col = (double *)malloc(sizeof(double) * NUM_TUPLES);
    sum_two_cols(sum_col, table[idx_col1], table[idx_col2]);

    #if DEBUG
    print_column(sum_col);
    #endif

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

#if DEBUG
void populate_table(double **table) {
    for(size_t i=0; i < NUM_COLS; ++i) {
        for(size_t j=0; j < NUM_TUPLES; ++j) {
            table[i][j] = (double) i;
        }
    }
}

void print_table(double **table) {
    for(size_t i=0; i < NUM_COLS; ++i) {
        printf("COL[%li]: ", i);
        for(size_t j=0; j < NUM_TUPLES; ++j) {
            printf("%0.1f", table[i][j]);
            if((j+1) < NUM_TUPLES)
                printf(", ");
        }
        printf("\n");
    }
}

void print_column(const double *col) {
    printf("COL[");
    for(size_t i=0; i < NUM_TUPLES; ++i) {
        printf("%0.1f", col[i]);
        if((i+1) < NUM_TUPLES) {
            printf(", ");
        }
    }
    printf("]\n");
}
#endif

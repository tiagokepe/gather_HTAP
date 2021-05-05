#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include "utils/defines.h"

#if DEBUG
#include "utils/utils.h"
#endif

void sum_two_cols(const double *table, double *sum, size_t idx_col1, size_t idx_col2);

int main() {
    double *table = (double*)aligned_alloc(TUPLE_SIZE, sizeof(double) * NUM_TUPLES * NUM_COLS);
    #if DEBUG
    populate_table(table);
    print_table(table);
    #endif

    double *sum_cols = (double*)aligned_alloc(TUPLE_SIZE, sizeof(double) * NUM_TUPLES);
    sum_two_cols(table, sum_cols, 0, 1);
    #if DEBUG
    print_sum_col(sum_cols);
    #endif
 
    free(table);
    return 0;
}

void sum_two_cols(const double *table, double *sum, size_t idx_col1, size_t idx_col2) {
    if(idx_col1 >= NUM_COLS || idx_col2 >= NUM_COLS) {
        printf("sum_two_rand_cols(): Column indexes out of bound\n");
        exit(1);
    }
    #if DEBUG
    printf("\n----SUM columns %li and %li-----\n", idx_col1, idx_col2);
    #endif

    __m512i vindex = _mm512_set_epi64(0,SZ,SZ*2,SZ*3,SZ*4,SZ*5,SZ*6,SZ*7);
    size_t stride_addr;
    size_t stride_col1, stride_col2;
    for(size_t stride_idx=0; stride_idx < NUM_TUPLES; stride_idx+=8) {
        stride_addr = stride_idx * NUM_COLS;
        #if DEBUG
        printf("stride_idx: %li, stride_addr: %li\n", stride_idx, stride_addr);
        #endif

        stride_col1 = stride_addr + idx_col1;
        __m512d col_vec1 = _mm512_i64gather_pd(vindex, &table[stride_col1], 1); //do the gather col1
        #if DEBUG
        print_m512d(col_vec1);
        #endif

        stride_col2 = stride_addr + idx_col2;
        __m512d col_vec2 = _mm512_i64gather_pd(vindex, &table[stride_col2], 1); //do the gather col2
        #if DEBUG
        print_m512d(col_vec2);
        #endif

        __m512d sum_vec = _mm512_add_pd(col_vec1, col_vec2); //sum_vec = col1 + col2
        #if DEBUG
        print_m512d(sum_vec);
        #endif

        _mm512_stream_pd(&sum[stride_idx], sum_vec);
        //memcpy(&sum[stride_addr], &sum_vec, sizeof(double) * 8);
    }
}


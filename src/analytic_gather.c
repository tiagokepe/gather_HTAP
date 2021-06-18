#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include "utils/defines.h"
#include "utils/global_vars.h" //TUPLE_SIZE and NUM_COLS
#include <stdint.h>
#include <string.h>
//#include <cstddef>
//#include <typeinfo>
#include "operators.h"

//#if DEBUG
#include "utils/utils.h"
//#endif

void sum_two_cols(const double *table, double *sum, size_t idx_col1, size_t idx_col2);


void Q1( const double *table, uint64_t proj_map, uint64_t aggfunc, size_t idx_f10, uint32_t x, int op); //SELECT f4,f5 FROM table WHERE f10 > x
void Q2( const double *table, uint64_t proj_map, uint64_t aggfunc, size_t idx_f10, uint32_t x, int op); //SELECT * FROM table-a WHERE f10 > x (most fields do not match)
void Q3( const double *table, uint64_t proj_map, uint64_t aggfunc, size_t idx_f10, uint32_t x, int op); //SELECT * FROM table-b WHERE f10 > x (most fields do match)
void Q4( const double *table, uint64_t proj_map, uint64_t aggfunc, size_t idx_f10, uint32_t x, int op); //SELECT SUM(f9) FROM table-a WHERE f10 > x (few fields match, f9 and f10 in same cacheline)
void Q5( const double *table, uint64_t proj_map, uint64_t aggfunc, size_t idx_f10, uint32_t x, int op); //SELECT SUM(f9) FROM table-b WHERE f10 > x (most fields match, f9 and f10 in same cacheline)
void Q6( const double *table, uint64_t proj_map, uint64_t aggfunc, size_t idx_f10, uint32_t x, int op); //SELECT AVG(f1) FROM table-a WHERE f10 > x (few fields match)
void Q7( const double *table, uint64_t proj_map, uint64_t aggfunc, size_t idx_f10, uint32_t x, int op); //SELECT AVG(f1) FROM table-b WHERE f10 > x (most fields match)

////
void Q8( const double *table1, const double *table2, uint64_t proj_map1, uint64_t proj_map2, uint64_t aggfunc, size_t idx_a1f1, size_t idx_a2f1, int op1, size_t idx_a1f9, size_t idx_a2f9, int op2); //SELECT table-a1.f3, table-a2.f4 FROM table-a1, table-a2 WHERE table-a1.f1 > table-a2.f1 AND table-a1.f9 = table-a2.f9 (JOIN example)
void Q9( const double *table1, const double *table2, uint64_t proj_map1, uint64_t proj_map2, uint64_t aggfunc, size_t idx_a1f9, size_t idx_a2f9, int op); //SELECT table-a1.f3, table-a2.f4 FROM table-a1, table-a2 WHERE table-a1.f9 = table-a2.f9 (JOIN example)
void Q10( const double *table, uint64_t proj_map, uint64_t aggfunc, size_t idx_f1, uint32_t x1, int op1, size_t idx_f9, uint32_t x2, int op2); //SELECT f3,f4 FROM table-a WHERE f1 > x1 AND f9 > x2 (f1 and f9 in different cachelines)
void Q11( const double *table, uint64_t proj_map, uint64_t aggfunc, size_t idx_f1, uint32_t x1, int op1, size_t idx_f2, uint32_t x2, int op2); //SELECT f3,f4 FROM table-a WHERE f1 > x1 AND f2 > x2 (f1 and f2 in same cacheline)
///
void Q12( const double *table, size_t idx_f3, uint64_t targetval1, size_t idx_f4, uint64_t  targetval2, size_t idx_f10, uint32_t z, int op); //UPDATE table-a SET f3 = x, f4 = WHERE f10 = z
void Q13( const double *table, size_t idx_f9, uint64_t targetval, size_t idx_f10, uint32_t y, int op); //UPDATE table-a SET f9 = x WHERE f10 = z
void Q14( const double *table, uint64_t proj_map, uint64_t aggfunc, size_t idx_f10, uint32_t x, int op); //SELECT SUM(f2-wide) FROM table-a 
void Q15( const double *table, uint64_t proj_map, uint64_t aggfunc, size_t idx_f10, uint32_t x, int op); //SELECT f3,f6,f10 FROM table-a 
 
int main(int argc, char **argv) {
    if(argc != 5) {
        printf("Usage: %s TUPLE_SIZE NUM_COLS idx_col1 idx_col2\n", argv[0]);
        exit(1);
    }

	// global vars
    TUPLE_SIZE = atoi(argv[1]);
    NUM_COLS = atoi(argv[2]);

    size_t idx_col1 = atoi(argv[3]);
    size_t idx_col2 = atoi(argv[4]);
    if(idx_col1 >= NUM_COLS || idx_col2 >= NUM_COLS) {
        printf("ERROR: Column indexes out of bound\n");
        exit(1);
    }

    double *table = (double*)aligned_alloc(TUPLE_SIZE, sizeof(double) * NUM_TUPLES * NUM_COLS);
    //#if DEBUG
    populate_table(table);
    //print_table(table);
    //#endif

    double *sum_cols = (double*)aligned_alloc(TUPLE_SIZE, sizeof(double) * NUM_TUPLES);
    sum_two_cols(table, sum_cols, idx_col1, idx_col2);
    #if DEBUG
    print_sum_col(sum_cols);
    #endif

	//__mmask8 * bitmap = (__mmask8*)aligned_alloc(TUPLE_SIZE, sizeof(__mmask8)*(NUM_TUPLES << 3)); //8b per position, representing bitmap
	//memset(bitmap, 0, sizeof(__mmask8) * NUM_TUPLES << 3);	
	//uint64_t totalmatches = filter(table, 3, 50.0, bitmap);
	//printf("%u\n", bitmap[0]); 

	uint64_t proj_size = 2;
	uint64_t * projfields = (uint64_t *)aligned_alloc(TUPLE_SIZE, sizeof(uint64_t)*proj_size);
	projfields[0] = 3;
	projfields[1] = 5;
	
	//double ** projected = (double **)aligned_alloc(TUPLE_SIZE, sizeof(double*)*proj_size);
	//for (int i = 0; i < proj_size; i++){
	//	projected[i] = (double *)aligned_alloc(TUPLE_SIZE, sizeof(double)*totalmatches);
	//	memset(projected[i], 0, sizeof(double) * totalmatches);
	//}
	//project(table, bitmap, projfields, proj_size, projected);
	aggfunction aggfunc = NOTHING;

	double ** projected = (double **)aligned_alloc(TUPLE_SIZE, sizeof(double*)*proj_size);
	uint64_t totalmatches = Q1_7(table, projfields, proj_size, projected, aggfunc, 7, 50.0 , 1);
	printf("projected fields:\n");
	for (int i = 0; i < proj_size; i++){
		for (int j = 0; j < totalmatches; j++){
		
			printf("%lf ", projected[i][j]);
		}
			printf("\n");
	}
	
 
    free(table);
	free(projfields);
	for (int i = 0; i < proj_size; i++){
		free(projected[i]);
	}
	free(projected);
    return 0;
}

void sum_two_cols(const double *table, double *sum, size_t idx_col1, size_t idx_col2) {
    #if DEBUG
    printf("\n----SUM columns %li and %li-----\n", idx_col1, idx_col2);
    #endif

    __m512i vindex = _mm512_set_epi64(0, TUPLE_SIZE, TUPLE_SIZE*2, TUPLE_SIZE*3, TUPLE_SIZE*4, TUPLE_SIZE*5, TUPLE_SIZE*6, TUPLE_SIZE*7);
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


#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include "utils/defines.h"
#include "utils/global_vars.h" //TUPLE_SIZE and NUM_COLS
#include <stdint.h>
#include <string.h>
//#include <math.h>
//#include <time.h>
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
    if(argc != 7) {
        printf("Usage: %s TUPLE_SIZE NUM_COLS idx_col1 idx_col2 tablevaluefile compop\n", argv[0]);
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
    double *tableb = (double*)aligned_alloc(TUPLE_SIZE, sizeof(double) * NUM_TUPLES * NUM_COLS);
	
	//#if DEBUG
    populate_table(table);
	//srand(time(NULL));
	//double range = (double)(NUM_TUPLES*NUM_COLS*0.2);
	//printf("range is %lf\n", range); 
	FILE * inputfile = fopen(argv[5], "r");
	//double input = 0;
	//for (uint32_t i = 0; i < NUM_TUPLES*NUM_COLS; i++)
	//{
	 //	tableb[i] = (double)fmod((double)rand(),range);
	//		printf("%lf\n", tableb[i]);
			//fscanf(inputfile,"%lf ", &input);
	//		tableb[i] = input;
	//}
	fread(tableb, 8, NUM_TUPLES*NUM_COLS, inputfile);

	fclose(inputfile);

    //print_table(table);
    //#endif

//    double *sum_cols = (double*)aligned_alloc(TUPLE_SIZE, sizeof(double) * NUM_TUPLES);
  //  sum_two_cols(table, sum_cols, idx_col1, idx_col2);
   // #if DEBUG
   // print_sum_col(sum_cols);
   // #endif


	uint64_t a = sizeof(__mmask8);
	printf("a = %lu\n", a);

	uint64_t proj_size = 2;

	uint64_t * projfields = (uint64_t *)aligned_alloc(TUPLE_SIZE, sizeof(uint64_t)*proj_size);
	projfields[0] = 3;
	projfields[1] = 4;

//Q1-7
	aggfunction aggfunc = NOTHING;

	double ** projected = (double **)aligned_alloc(TUPLE_SIZE, sizeof(double*)*proj_size);
//	double * sum = (double*)Q1_7(table, projfields, proj_size, projected, aggfunc, 10, (NUM_COLS << 10), 1);
	size_t fields[2] = {1,9};
	double targets[2] = {(double)(NUM_COLS << 20),(double) (NUM_COLS << 22)};
	uint64_t totalmatches =	Q10_11(table, projfields, proj_size, projected, aggfunc, &fields[0], &targets[0], 2);
	//double sum = res[0];
	//printf("aggregate result: %lf\n", *sum);
	printf("totalmatches = %lu\n", totalmatches);


//############################
	
//Q8-9
/*	size_t list1[1] = { 9 };
	size_t list2[1] = { 9 };
	uint32_t list3[2] = {0};

	uint64_t outlen = 0;

	address_pair * res2 = (address_pair*)hash_join(table, tableb, &list1[0], &list2[0], &list3[0], 1, NUM_TUPLES, NUM_TUPLES, &outlen);
	printf("got outlen %lu\n", outlen);

	double ** projected = (double **)aligned_alloc(TUPLE_SIZE, sizeof(double*)*outlen);
	for (int i = 0; i < proj_size; i++){
		projected[i] = (double *)aligned_alloc(TUPLE_SIZE, sizeof(double)*outlen);
		memset(projected[i], 0, sizeof(double) * outlen);
	}
	for (size_t i = 0; i < outlen; i++)
	{
		projected[0][i] = res2->addr1[3];
		projected[1][i] = res2->addr2[4];
	}

	printf("projected last %lf\n", projected[0][outlen-1]);	
*/
//########################

//	uint64_t totalmatches = outlen;
	printf("projected fields:\n");
	for (int j = 0; j < totalmatches; j++){
		for (int i = 0; i < proj_size; i++){
			if (totalmatches > 0)		
				printf("%lf ", projected[i][j]);
		}
		printf("\n");
	}

/*	if (totalmatches > 0)
	{
		printf("%lf \n", projected[0][totalmatches-1]);
    }*/
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


#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "utils/defines.h"
#include "utils/global_vars.h"
#include "col_operators.h"

#include "utils/utils.h"

void sum_two_cols(double *sum, const double *col1, const double *col2);
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

#if DEBUG
void populate_table(double **table);
void print_table(double **table);
void print_column(const double *col);
#endif

///############################################################################
///
int main(int argc, char **argv) {
    if(argc < 8) {
        printf("Usage: %s TUPLE_SIZE NUM_COLS proj_size idx_col2 tableavaluefile tablebvaluefile proj1\n", argv[0]);
        exit(1);
    }


    TUPLE_SIZE = atoi(argv[1]);
    NUM_COLS = atoi(argv[2]);

    size_t idx_col1 = atoi(argv[3]);
	//if (argc != (7 + idx_col1)) { 
//		printf("wrong number of projections passed as args\n");
//		exit(1);
//	}
    size_t idx_col2 = atoi(argv[4]);
    if(idx_col1 >= NUM_COLS || idx_col2 >= NUM_COLS) {
        printf("ERROR: Column indexes out of bound\n");
        exit(1);
    }

///############################################################################
///table initialization

    double *table = (double*)aligned_alloc(TUPLE_SIZE, sizeof(double) * NUM_TUPLES * NUM_COLS);
    double *tableb = (double*)aligned_alloc(TUPLE_SIZE, sizeof(double) * NUM_TUPLES * NUM_COLS);


	FILE * tableafile = fopen(argv[5], "r");
	if (tableafile == NULL)
	{	
		printf("failed to open tablea file\n");
		exit(1);
	}
	size_t filestatus = fread(table, 8, NUM_TUPLES*NUM_COLS, tableafile);
	if (filestatus != NUM_TUPLES*NUM_COLS)
	{
		printf("fread for tablea failed to read some items\n");
	} 
	fclose(tableafile);
//	srand(time(NULL));
//	double range = (double)(NUM_TUPLES*NUM_COLS*0.2);
	FILE * tablebfile = fopen(argv[6], "r");
	if (tablebfile == NULL)
	{	
		printf("failed to open tableb file\n");
		exit(1);
	}
	filestatus = fread(tableb, 8, NUM_TUPLES*NUM_COLS, tablebfile);
	if (filestatus != NUM_TUPLES*NUM_COLS)
	{
		printf("fread for tableb failed to read some items\n");
	} 
	fclose(tablebfile);

///

///############################################################################
	///projection initialization and definition
	uint64_t proj_size = idx_col1;
//	uint64_t proj_size = NUM_COLS;

	uint64_t * projfields = (uint64_t *)malloc(sizeof(uint64_t)*proj_size);
	for (int i = 0; i < proj_size; i++)
	{
		projfields[i] = atoi(argv[i+7]);
//		projfields[i] = i;
	}


///############################################################################
//Q1-7, Q10-11, Q14-Q15
	aggfunction aggfunc = NOTHING;

	double ** projected = (double **)aligned_alloc(TUPLE_SIZE,sizeof(double*)*proj_size);
//	uint64_t * totalmatches = (uint64_t *)Q1_7(table, projfields, proj_size, projected, aggfunc, 10, (double)(TUPLE_SIZE << 10), 1);
//	double * sum = (double *)Q1_7(table, projfields, proj_size, projected, aggfunc, 10, (double)(TUPLE_SIZE << 10), 1);
	size_t fields[2] = {1,9};
	double targets[2] = {(double)NUM_COLS, (double)TUPLE_SIZE};
	uint64_t * totalmatches = (uint64_t *)Q10_11(table, projfields, proj_size, projected, aggfunc, &fields[0], &targets[0], 0); //Q10,Q11, Q15 (0 comparisons)

//    printf("aggregate result: %lf\n", *sum);
//	free(sum);
	printf("totalmatches = %lu\n", *totalmatches);
	if ( (*totalmatches) > 0)
	{
		printf("%lf \n", projected[0][(*totalmatches)-1]);
   }
	free(totalmatches);
///############################################################################
//Q8-9

/*	size_t list1[1] = { 9};
	size_t list2[1] = { 9};
	uint32_t list3[1] = {0};

	uint64_t outlen = 0;

	address_pair * res2 = (address_pair*)hash_join(table, tableb, &list1[0], &list2[0], &list3[0], 1, NUM_TUPLES, NUM_TUPLES, &outlen);


	double ** projected = (double **)aligned_alloc(TUPLE_SIZE, sizeof(double*)*outlen);
	for (int i = 0; i < proj_size; i++){
		projected[i] = (double *)aligned_alloc(TUPLE_SIZE, sizeof(double)*outlen);
		memset(projected[i], 0, sizeof(double) * outlen);
	}
	for (size_t i = 0; i < outlen; i++)
	{
		projected[0][i] = res2->addr1[projfields[0]*NUM_TUPLES];
		projected[1][i] = res2->addr2[projfields[1]*NUM_TUPLES];
	}

	printf("projected last %lf\n", projected[0][outlen-1]);	
*/

///############################################################################
///projection printf for sanity checking

/*
//	uint64_t totalmatches = outlen;
	printf("projected fields:\n");
	for (int j = 0; j < totalmatches; j++){
		for (int i = 0; i < proj_size; i++){
			if (totalmatches > 0)		
				printf("%lf ", projected[i][j]);
		}
		printf("\n");
	}*/



///############################################################################
//Q12-Q13
/*
//	size_t idx_upds[2] = {3,4};
	size_t idx_upds[1] = {9};
//	double updatevals[2] = {(double)NUM_COLS, (double)TUPLE_SIZE};
	double updatevals[1] = {(double)NUM_COLS};
	size_t checkfield = atoi(argv[7]);
//	uint64_t upds = 2;
	uint64_t upds = 1;
	uint64_t totalmatches = UPDATEfilter(table, checkfield, 41.0, &updatevals[0], &idx_upds[0], upds); 
	printf("totmatches = %lu\n", totalmatches);
*/


///############################################################################
///finalize, free memory	
    free(table);
	free(tableb);
	free(projfields);
	for (int i = 0; i < proj_size; i++){
		free(projected[i]);
	}
	free(projected);
	return 0;
}
/*
void sum_two_cols(double *sum, const double *col1, const double *col2) {
    for(size_t i=0; i < NUM_TUPLES; ++i) {
        sum[i] = col1[i] + col2[i];
    }
}
*/


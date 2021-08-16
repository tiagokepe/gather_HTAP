#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include "utils/defines.h"
#include "utils/global_vars.h" //TUPLE_SIZE and NUM_COLS
#include <stdint.h>
#include <string.h>
#include "utils/utils.h"


typedef enum {
	NOTHING,
	SUM,
	AVG,
	COUNT
} aggfunction;

typedef struct address_pair {
	double * addr1;
	double * addr2;
} address_pair;

uint64_t filter(double *table,  size_t idx_col1, double val, __mmask8 * bitmap);
uint64_t UPDATEfilter(double *table, size_t idx_col1, double val, double * updateval, size_t * idx_update, uint64_t updates);
uint64_t filterAND(double *table, size_t idx_col1, double val, __mmask8 * bitmap);
void project(double *table, __mmask8 *bitmap, uint64_t * proj_vec, uint64_t proj_size, double ** projected);
void * Q1_7( double *table, uint64_t * projfields, uint64_t projections, double ** projected, aggfunction aggop, size_t idx_f10, uint32_t x, int op);
void * aggregate(double *table, double * proj_col, uint64_t * len, aggfunction aggop);
address_pair * hash_join(double *a, double *b, size_t * list1, size_t * list2, uint32_t * list3, uint32_t lencomps, size_t lena, size_t lenb,size_t * outlen);
void * Q10_11( double *table, uint64_t * projfields, uint64_t projections, double ** projected, aggfunction aggop, size_t * fields, double * targets, size_t comps);
void * Q12_Q13( double *table, uint64_t * updatefields, double * updatevalues, uint64_t updates, size_t idx_f10, double z, int op);

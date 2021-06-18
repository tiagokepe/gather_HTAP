
#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include "utils/defines.h"
#include "utils/global_vars.h" //TUPLE_SIZE and NUM_COLS
#include <stdint.h>
#include <string.h>

#include "operators.h"
#include "utils/utils.h"

uint64_t filter(const double *table, size_t idx_col1, double val, __mmask8 * bitmap)
{
	//gather stride
    __m512i vindex = _mm512_set_epi64(0, TUPLE_SIZE, TUPLE_SIZE*2, TUPLE_SIZE*3, TUPLE_SIZE*4, TUPLE_SIZE*5, TUPLE_SIZE*6, TUPLE_SIZE*7);
	
    size_t stride_addr;
    size_t stride_col1, stride_col2;

	uint64_t base_address = 0;
	
	const int op1 = 1;
	__m512d col_vec2 = _mm512_set1_pd(val);//sets 2nd avx reg with val in all positions 
	printf("val is %lf\n", val);
    for(size_t stride_idx=0; stride_idx < NUM_TUPLES; stride_idx+=8) {
        stride_addr = stride_idx * NUM_COLS;
        stride_col1 = stride_addr + idx_col1;
        __m512d col_vec1 = _mm512_i64gather_pd(vindex, &table[stride_col1], 1); //do the gather col1
		//#if DEBUG
        //print_m512d(col_vec1);
        //#endif

        __mmask8 res_vec_mask = _mm512_cmp_pd_mask(col_vec1, col_vec2, op1); //mask = col1 OP col2

		//approach 2: bitmap
		_store_mask8(&bitmap[(stride_idx << 3)], res_vec_mask);	

		//convert mask bits to integer and count bits		
		uint32_t maskbits_gpr = _cvtmask8_u32(res_vec_mask);
		base_address += _mm_popcnt_u32(maskbits_gpr);
	}
	return base_address;
}

void project(const double *table, __mmask8 *bitmap, uint64_t * proj_vec, uint64_t proj_size, double ** projected)
{
	//gather stride
    __m512i vindex = _mm512_set_epi64(0, TUPLE_SIZE, TUPLE_SIZE*2, TUPLE_SIZE*3, TUPLE_SIZE*4, TUPLE_SIZE*5, TUPLE_SIZE*6, TUPLE_SIZE*7);
	__m512d zerodef = _mm512_setzero_pd();	
	
    size_t stride_addr, stride_col;

	uint64_t base_address = 0;

    for(size_t stride_idx = 0; stride_idx < NUM_TUPLES; stride_idx+=8) {

		for (uint64_t k = 0; k < proj_size; k++){// loop around intended projections
        	stride_addr = stride_idx * NUM_COLS;
        	stride_col = stride_addr + proj_vec[k];//obtain index for this column
        	__m512d col_vec = _mm512_mask_i64gather_pd(zerodef, bitmap[stride_idx << 3],vindex, &table[stride_col], 1); //do the gather col using mask
			//store col_vec using mask
			_mm512_mask_compressstoreu_pd(&projected[k][base_address], bitmap[stride_idx << 3], col_vec); //projection matrix contains a vector for each projected field	
		}
		//convert mask bits to integer		
		uint32_t maskbits_gpr = _cvtmask8_u32(bitmap[stride_idx << 3]);
		
		//count number of tuples added to selection, add to base address
		base_address += _mm_popcnt_u32(maskbits_gpr);
	}
}

void * aggregate(const double *table, double *proj_col, size_t len, aggfunction aggop){

	double * retval = malloc(sizeof(double));
	switch(aggop){
		case SUM:
		{
			double s = 0;
			for(uint32_t i = 0; i < len; i++)
				s += proj_col[i];
			*retval = s;
			return retval; 
			break;
		}
		case AVG:
		{
			double s = 0;
			for(uint32_t i = 0; i < len; i++)
				s += proj_col[i];
			s = s / len;
			*retval = s;
			return retval; 
			break;
		}
		case COUNT:{
			return &len;
			break;
		}	
		default:
			return NULL;

	}
	return NULL;
}

uint64_t Q1_7( const double *table, uint64_t * projfields, uint64_t projections, double ** projected, aggfunction aggop, size_t idx_f10, uint32_t x, int op)
{
	uint64_t totalmatches;

	__mmask8 * bitmap = (__mmask8*)aligned_alloc(TUPLE_SIZE, sizeof(__mmask8)*(NUM_TUPLES << 3)); //8b per position, representing bitmap
	memset(bitmap, 0, sizeof(__mmask8) * NUM_TUPLES << 3);	
	totalmatches = filter(table, idx_f10, x, bitmap);
	
	for (int i = 0; i < projections; i++){
		projected[i] = (double *)aligned_alloc(TUPLE_SIZE, sizeof(double)*totalmatches);
		memset(projected[i], 0, sizeof(double) * totalmatches);
	}
	project(table, bitmap, projfields, projections, projected);		

	if (aggop != NOTHING){
		aggregate(table, projected[0], totalmatches, aggop);
	}	

	free(bitmap);
	return totalmatches;
}

#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include "utils/defines.h"
#include "utils/global_vars.h" //TUPLE_SIZE and NUM_COLS
#include <stdint.h>
#include <string.h>

#include "operators.h"
#include "utils/utils.h"
#include "vecmurmur.h"
#include "hash_table.h"


int ceil_log2(unsigned long long x)
{
  static const unsigned long long t[6] = {
    0xFFFFFFFF00000000ull,
    0x00000000FFFF0000ull,
    0x000000000000FF00ull,
    0x00000000000000F0ull,
    0x000000000000000Cull,
    0x0000000000000002ull
  };

  int y = (((x & (x - 1)) == 0) ? 0 : 1);
  int j = 32;
  int i;

  for (i = 0; i < 6; i++) {
    int k = (((x & t[i]) == 0) ? 0 : j);
    y += k;
    x >>= k;
    j >>= 1;
  }

  return y;
}

uint64_t filter(double *table, size_t idx_col1, double val, __mmask8 * bitmap)
{
	//gather stride regs
    __m512i vindex1 = _mm512_set_epi64(TUPLE_SIZE*7, TUPLE_SIZE*6, TUPLE_SIZE*5, TUPLE_SIZE*4, TUPLE_SIZE*3, TUPLE_SIZE*2, TUPLE_SIZE, 0);
    __m512i vindex2 = _mm512_set_epi64(TUPLE_SIZE*7, TUPLE_SIZE*6, TUPLE_SIZE*5, TUPLE_SIZE*4, TUPLE_SIZE*3, TUPLE_SIZE*2, TUPLE_SIZE, 0);
    __m512i vindex3 = _mm512_set_epi64(TUPLE_SIZE*7, TUPLE_SIZE*6, TUPLE_SIZE*5, TUPLE_SIZE*4, TUPLE_SIZE*3, TUPLE_SIZE*2, TUPLE_SIZE, 0);
    __m512i vindex4 = _mm512_set_epi64(TUPLE_SIZE*7, TUPLE_SIZE*6, TUPLE_SIZE*5, TUPLE_SIZE*4, TUPLE_SIZE*3, TUPLE_SIZE*2, TUPLE_SIZE, 0);
    
	//__m512i vindex4 = _mm512_set_epi64(0, TUPLE_SIZE, TUPLE_SIZE*2, TUPLE_SIZE*3, TUPLE_SIZE*4, TUPLE_SIZE*5, TUPLE_SIZE*6, TUPLE_SIZE*7);
	
    size_t stride_addr1;
    size_t stride_addr2;
    size_t stride_addr3;
    size_t stride_addr4;
    size_t stride_col1;
    size_t stride_col2;
    size_t stride_col3;
    size_t stride_col4;

	uint64_t popcnt = 0;
	
	const int op1 = 30;
	__m512d add_vec1 = _mm512_set1_pd(val);//sets 2nd avx reg with val in all positions 
	__m512d add_vec2 = _mm512_set1_pd(val);//sets 2nd avx reg with val in all positions 
	__m512d add_vec3 = _mm512_set1_pd(val);//sets 2nd avx reg with val in all positions 
	__m512d add_vec4 = _mm512_set1_pd(val);//sets 2nd avx reg with val in all positions 
	
    int log2cols =   ceil_log2(NUM_COLS);	
	size_t upperbound = NUM_TUPLES >> 2;
	size_t stride_j = upperbound;
	size_t stride_k = upperbound << 1;
	size_t stride_l = stride_k + upperbound;

    for(size_t stride_idx=0; stride_idx < upperbound; stride_idx+=8) {
	
//        stride_addr1 = stride_idx * NUM_COLS;
 //  		stride_addr2 = stride_j * NUM_COLS;
//		stride_addr3 = stride_k * NUM_COLS;
//		stride_addr4 = stride_l * NUM_COLS;
		stride_addr1 = stride_idx << log2cols;
		stride_addr2 = stride_j << log2cols;
		stride_addr3 = stride_k << log2cols;
		stride_addr4 = stride_l << log2cols;
	

	    stride_col1 = stride_addr1 + idx_col1;
	    stride_col2 = stride_addr2 + idx_col1;
	    stride_col3 = stride_addr3 + idx_col1;
	    stride_col4 = stride_addr4 + idx_col1;
        
		__m512d col_vec1 = _mm512_i64gather_pd(vindex1, &table[stride_col1], 1); //do the gather col1
		__m512d col_vec2 = _mm512_i64gather_pd(vindex2, &table[stride_col2], 1); //do the gather col1
		__m512d col_vec3 = _mm512_i64gather_pd(vindex3, &table[stride_col3], 1); //do the gather col1
		__m512d col_vec4 = _mm512_i64gather_pd(vindex4, &table[stride_col4], 1); //do the gather col1
		//#if DEBUG
        //print_m512d(col_vec1);
        //#endif

        __mmask8 res_vec_mask1 = _mm512_cmp_pd_mask(col_vec1, add_vec1, op1); //mask = col1 OP col2
        __mmask8 res_vec_mask2 = _mm512_cmp_pd_mask(col_vec2, add_vec2, op1); //mask = col1 OP col2
        __mmask8 res_vec_mask3 = _mm512_cmp_pd_mask(col_vec3, add_vec3, op1); //mask = col1 OP col2
        __mmask8 res_vec_mask4 = _mm512_cmp_pd_mask(col_vec4, add_vec4, op1); //mask = col1 OP col2

		//approach 2: bitmap
		_store_mask8(&bitmap[(stride_idx >> 3)], res_vec_mask1);	
		_store_mask8(&bitmap[(stride_j >> 3)], res_vec_mask2);	
		_store_mask8(&bitmap[(stride_k >> 3)], res_vec_mask3);	
		_store_mask8(&bitmap[(stride_l >> 3)], res_vec_mask4);	
		//manual indices
		stride_j += 8;
		stride_k += 8;
		stride_l += 8;
		//convert mask bits to integer and count bits		
		uint32_t maskbits_gpr;
		maskbits_gpr = _cvtmask8_u32(res_vec_mask1);
		popcnt += _mm_popcnt_u32(maskbits_gpr);
		maskbits_gpr = _cvtmask8_u32(res_vec_mask2);
		popcnt += _mm_popcnt_u32(maskbits_gpr);
		maskbits_gpr = _cvtmask8_u32(res_vec_mask3);
		popcnt += _mm_popcnt_u32(maskbits_gpr);
		maskbits_gpr = _cvtmask8_u32(res_vec_mask4);
		popcnt += _mm_popcnt_u32(maskbits_gpr);
	}
	return popcnt;
}

uint64_t UPDATEfilter(double *table, size_t idx_col1, double val, double * updateval, size_t * idx_update, uint64_t updates)
{
	//gather stride
    __m512i vindex1 = _mm512_set_epi64(TUPLE_SIZE*7, TUPLE_SIZE*6, TUPLE_SIZE*5, TUPLE_SIZE*4, TUPLE_SIZE*3, TUPLE_SIZE*2, TUPLE_SIZE, 0);
    __m512i vindex2 = _mm512_set_epi64(TUPLE_SIZE*7, TUPLE_SIZE*6, TUPLE_SIZE*5, TUPLE_SIZE*4, TUPLE_SIZE*3, TUPLE_SIZE*2, TUPLE_SIZE, 0);
    __m512i vindex3 = _mm512_set_epi64(TUPLE_SIZE*7, TUPLE_SIZE*6, TUPLE_SIZE*5, TUPLE_SIZE*4, TUPLE_SIZE*3, TUPLE_SIZE*2, TUPLE_SIZE, 0);
    __m512i vindex4 = _mm512_set_epi64(TUPLE_SIZE*7, TUPLE_SIZE*6, TUPLE_SIZE*5, TUPLE_SIZE*4, TUPLE_SIZE*3, TUPLE_SIZE*2, TUPLE_SIZE, 0);
	
    size_t stride_addr1;
    size_t stride_addr2;
    size_t stride_addr3;
    size_t stride_addr4;
    size_t stride_col1;
    size_t stride_col2;
    size_t stride_col3;
    size_t stride_col4;

	uint64_t popcnt = 0;
	
	const int op1 = 0;
	__m512d add_vec1 = _mm512_set1_pd(val);//sets 2nd avx reg with val in all positions 
	__m512d add_vec2 = _mm512_set1_pd(val);//sets 2nd avx reg with val in all positions 
	__m512d add_vec3 = _mm512_set1_pd(val);//sets 2nd avx reg with val in all positions 
	__m512d add_vec4 = _mm512_set1_pd(val);//sets 2nd avx reg with val in all positions 
	size_t upperbound = NUM_TUPLES >> 2;
	size_t stride_j = upperbound;
	size_t stride_k = upperbound << 1;
	size_t stride_l = stride_k + upperbound;
    
	for(size_t stride_idx=0; stride_idx < upperbound; stride_idx+=8) {
        stride_addr1 = stride_idx * NUM_COLS;
   		stride_addr2 = stride_j * NUM_COLS;
		stride_addr3 = stride_k * NUM_COLS;
		stride_addr4 = stride_l * NUM_COLS;

	    stride_col1 = stride_addr1 + idx_col1;
	    stride_col2 = stride_addr2 + idx_col1;
	    stride_col3 = stride_addr3 + idx_col1;
	    stride_col4 = stride_addr4 + idx_col1;
		__m512d col_vec1 = _mm512_i64gather_pd(vindex1, &table[stride_col1], 1); //do the gather col1
		__m512d col_vec2 = _mm512_i64gather_pd(vindex2, &table[stride_col2], 1); //do the gather col1
		__m512d col_vec3 = _mm512_i64gather_pd(vindex3, &table[stride_col3], 1); //do the gather col1
		__m512d col_vec4 = _mm512_i64gather_pd(vindex4, &table[stride_col4], 1); //do the gather col1

        __mmask8 res_vec_mask1 = _mm512_cmp_pd_mask(col_vec1, add_vec1, op1); //mask = col1 OP col2
        __mmask8 res_vec_mask2 = _mm512_cmp_pd_mask(col_vec2, add_vec2, op1); //mask = col1 OP col2
        __mmask8 res_vec_mask3 = _mm512_cmp_pd_mask(col_vec3, add_vec3, op1); //mask = col1 OP col2
        __mmask8 res_vec_mask4 = _mm512_cmp_pd_mask(col_vec4, add_vec4, op1); //mask = col1 OP col2

		//now using the mask we scatter a vector for each update value and column
		for (int i = 0; i < updates; i++)
		{	
			__m512d update_col1 = _mm512_set1_pd(updateval[i]);
			__m512d update_col2 = _mm512_set1_pd(updateval[i]);
			__m512d update_col3 = _mm512_set1_pd(updateval[i]);
			__m512d update_col4 = _mm512_set1_pd(updateval[i]);
			size_t stride_updcol1 = stride_addr1 + idx_update[i];
			size_t stride_updcol2 = stride_addr2 + idx_update[i];
			size_t stride_updcol3 = stride_addr3 + idx_update[i];
			size_t stride_updcol4 = stride_addr4 + idx_update[i];
			
			_mm512_mask_i64scatter_pd(&table[stride_updcol1], res_vec_mask1, vindex1, update_col1, 1); //write update value on matching fields
			_mm512_mask_i64scatter_pd(&table[stride_updcol2], res_vec_mask2, vindex2, update_col2, 1); //write update value on matching fields
			_mm512_mask_i64scatter_pd(&table[stride_updcol3], res_vec_mask3, vindex3, update_col3, 1); //write update value on matching fields
			_mm512_mask_i64scatter_pd(&table[stride_updcol4], res_vec_mask4, vindex4, update_col4, 1); //write update value on matching fields
		}
		//convert mask bits to integer and count bits		
		uint32_t maskbits_gpr = _cvtmask8_u32(res_vec_mask1);
		popcnt += _mm_popcnt_u32(maskbits_gpr);
		maskbits_gpr = _cvtmask8_u32(res_vec_mask2);
		popcnt += _mm_popcnt_u32(maskbits_gpr);
		maskbits_gpr = _cvtmask8_u32(res_vec_mask3);
		popcnt += _mm_popcnt_u32(maskbits_gpr);
		maskbits_gpr = _cvtmask8_u32(res_vec_mask4);
		popcnt += _mm_popcnt_u32(maskbits_gpr);
	}
	return popcnt;
}

uint64_t filterAND(double *table, size_t idx_col1, double val, __mmask8 * bitmap)
{
	//gather stride
    __m512i vindex1 = _mm512_set_epi64(TUPLE_SIZE*7, TUPLE_SIZE*6, TUPLE_SIZE*5, TUPLE_SIZE*4, TUPLE_SIZE*3, TUPLE_SIZE*2, TUPLE_SIZE, 0);
    __m512i vindex2 = _mm512_set_epi64(TUPLE_SIZE*7, TUPLE_SIZE*6, TUPLE_SIZE*5, TUPLE_SIZE*4, TUPLE_SIZE*3, TUPLE_SIZE*2, TUPLE_SIZE, 0);
    __m512i vindex3 = _mm512_set_epi64(TUPLE_SIZE*7, TUPLE_SIZE*6, TUPLE_SIZE*5, TUPLE_SIZE*4, TUPLE_SIZE*3, TUPLE_SIZE*2, TUPLE_SIZE, 0);
    __m512i vindex4 = _mm512_set_epi64(TUPLE_SIZE*7, TUPLE_SIZE*6, TUPLE_SIZE*5, TUPLE_SIZE*4, TUPLE_SIZE*3, TUPLE_SIZE*2, TUPLE_SIZE, 0);
	
    size_t stride_addr1;
    size_t stride_addr2;
    size_t stride_addr3;
    size_t stride_addr4;
    size_t stride_col1;
    size_t stride_col2;
    size_t stride_col3;
    size_t stride_col4;

	uint64_t popcnt = 0;
	
	const int op1 = 30;
	__m512d add_vec1 = _mm512_set1_pd(val);//sets 2nd avx reg with val in all positions 
	__m512d add_vec2 = _mm512_set1_pd(val);//sets 2nd avx reg with val in all positions 
	__m512d add_vec3 = _mm512_set1_pd(val);//sets 2nd avx reg with val in all positions 
	__m512d add_vec4 = _mm512_set1_pd(val);//sets 2nd avx reg with val in all positions 
	
	size_t upperbound = NUM_TUPLES >> 2;
	size_t stride_j = upperbound;
	size_t stride_k = upperbound << 1;
	size_t stride_l = stride_k + upperbound;

    for(size_t stride_idx=0; stride_idx < upperbound; stride_idx+=8) {
        stride_addr1 = stride_idx * NUM_COLS;
   		stride_addr2 = stride_j * NUM_COLS;
		stride_addr3 = stride_k * NUM_COLS;
		stride_addr4 = stride_l * NUM_COLS;

	    stride_col1 = stride_addr1 + idx_col1;
	    stride_col2 = stride_addr2 + idx_col1;
	    stride_col3 = stride_addr3 + idx_col1;
	    stride_col4 = stride_addr4 + idx_col1;
        
		__m512d col_vec1 = _mm512_i64gather_pd(vindex1, &table[stride_col1], 1); //do the gather col1
		__m512d col_vec2 = _mm512_i64gather_pd(vindex2, &table[stride_col2], 1); //do the gather col1
		__m512d col_vec3 = _mm512_i64gather_pd(vindex3, &table[stride_col3], 1); //do the gather col1
		__m512d col_vec4 = _mm512_i64gather_pd(vindex4, &table[stride_col4], 1); //do the gather col1

        __mmask8 res_vec_mask1 = _mm512_cmp_pd_mask(col_vec1, add_vec1, op1); //mask = col1 OP col2
        __mmask8 res_vec_mask2 = _mm512_cmp_pd_mask(col_vec2, add_vec2, op1); //mask = col1 OP col2
        __mmask8 res_vec_mask3 = _mm512_cmp_pd_mask(col_vec3, add_vec3, op1); //mask = col1 OP col2
        __mmask8 res_vec_mask4 = _mm512_cmp_pd_mask(col_vec4, add_vec4, op1); //mask = col1 OP col2

		//AND approach: load bitmap, AND, store bitmap.
		__mmask8 current_val1 = _load_mask8(&bitmap[(stride_idx >> 3)]);
		__mmask8 current_val2 = _load_mask8(&bitmap[(stride_j >> 3)]);
		__mmask8 current_val3 = _load_mask8(&bitmap[(stride_k >> 3)]);
		__mmask8 current_val4 = _load_mask8(&bitmap[(stride_l >> 3)]);
		res_vec_mask1 = _kand_mask8(res_vec_mask1, current_val1);
		res_vec_mask2 = _kand_mask8(res_vec_mask2, current_val2);
		res_vec_mask3 = _kand_mask8(res_vec_mask3, current_val3);
		res_vec_mask4 = _kand_mask8(res_vec_mask4, current_val4);

		_store_mask8(&bitmap[(stride_idx >> 3)], res_vec_mask1);	
		_store_mask8(&bitmap[(stride_j >> 3)], res_vec_mask2);	
		_store_mask8(&bitmap[(stride_k >> 3)], res_vec_mask3);	
		_store_mask8(&bitmap[(stride_l >> 3)], res_vec_mask4);	

		//convert mask bits to integer and count bits		
		uint32_t maskbits_gpr;
		maskbits_gpr = _cvtmask8_u32(res_vec_mask1);
		popcnt += _mm_popcnt_u32(maskbits_gpr);
		maskbits_gpr = _cvtmask8_u32(res_vec_mask2);
		popcnt += _mm_popcnt_u32(maskbits_gpr);
		maskbits_gpr = _cvtmask8_u32(res_vec_mask3);
		popcnt += _mm_popcnt_u32(maskbits_gpr);
		maskbits_gpr = _cvtmask8_u32(res_vec_mask4);
		popcnt += _mm_popcnt_u32(maskbits_gpr);
	}
	return popcnt;
}

void project(double *table, __mmask8 *bitmap, uint64_t * proj_vec, uint64_t proj_size, double ** projected)
{
	//gather stride
    __m512i vindex1 = _mm512_set_epi64(TUPLE_SIZE*7, TUPLE_SIZE*6, TUPLE_SIZE*5, TUPLE_SIZE*4, TUPLE_SIZE*3, TUPLE_SIZE*2, TUPLE_SIZE, 0);
    __m512i vindex2 = _mm512_set_epi64(TUPLE_SIZE*7, TUPLE_SIZE*6, TUPLE_SIZE*5, TUPLE_SIZE*4, TUPLE_SIZE*3, TUPLE_SIZE*2, TUPLE_SIZE, 0);
    __m512i vindex3 = _mm512_set_epi64(TUPLE_SIZE*7, TUPLE_SIZE*6, TUPLE_SIZE*5, TUPLE_SIZE*4, TUPLE_SIZE*3, TUPLE_SIZE*2, TUPLE_SIZE, 0);
    __m512i vindex4 = _mm512_set_epi64(TUPLE_SIZE*7, TUPLE_SIZE*6, TUPLE_SIZE*5, TUPLE_SIZE*4, TUPLE_SIZE*3, TUPLE_SIZE*2, TUPLE_SIZE, 0);
	__m512d zerodef1 = _mm512_setzero_pd();	
	__m512d zerodef2 = _mm512_setzero_pd();	
	__m512d zerodef3 = _mm512_setzero_pd();	
	__m512d zerodef4 = _mm512_setzero_pd();	
	
	size_t stride_addr1;
    size_t stride_addr2;
    size_t stride_addr3;
    size_t stride_addr4;
	size_t stride_col1;
    size_t stride_col2;
    size_t stride_col3;
    size_t stride_col4;
    int log2cols =   ceil_log2(NUM_COLS);	

	for (uint64_t k = 0; k < proj_size; k++)
	{// loop around intended projections
		uint64_t cur_staddr = 0;
		for(size_t stride_idx = 0; stride_idx < NUM_TUPLES; stride_idx+=32) 
		{
				//compute 4 indices
			stride_addr1 = stride_idx << log2cols;
			stride_addr2 = (stride_idx+8) << log2cols;
			stride_addr3 = (stride_idx+16) << log2cols;
			stride_addr4 = (stride_idx+24) << log2cols;
	//		stride_addr2 = (stride_idx+8) * NUM_COLS;
	//		stride_addr3 = (stride_idx+16) * NUM_COLS;
//			stride_addr4 = (stride_idx+24) * NUM_COLS;


			__mmask8 mask = bitmap[stride_idx >> 3];
			size_t bmap2 = (stride_idx+8) >> 3;
			size_t bmap3 = (stride_idx+16) >> 3;
			size_t bmap4 = (stride_idx+24) >> 3;
		//pre-compute store addresses.
			size_t staddr1 = cur_staddr;
			cur_staddr+= _mm_popcnt_u32(_cvtmask8_u32(mask));		
			size_t staddr2 = cur_staddr; 
			cur_staddr+= _mm_popcnt_u32(_cvtmask8_u32(bitmap[bmap2]));		
			size_t staddr3 = cur_staddr; 
			cur_staddr+= _mm_popcnt_u32(_cvtmask8_u32(bitmap[bmap3]));		
			size_t staddr4 = cur_staddr;		
			cur_staddr+= _mm_popcnt_u32(_cvtmask8_u32(bitmap[bmap4]));		

				stride_col1 = stride_addr1 + proj_vec[k];//obtain index for this column
				stride_col2 = stride_addr2 + proj_vec[k];//obtain index for this column
				stride_col3 = stride_addr3 + proj_vec[k];//obtain index for this column
				stride_col4 = stride_addr4 + proj_vec[k];//obtain index for this column
				__m512d col_vec1 = _mm512_mask_i64gather_pd(zerodef1, mask, vindex1, &table[stride_col1], 1); //do the gather col using bitmap mask. Division by 8 as a single position of bitmap contains 1 byte, so 8 bits for the 8 values in stride_idx step
				__m512d col_vec2 = _mm512_mask_i64gather_pd(zerodef2, bitmap[bmap2],vindex2, &table[stride_col2], 1); //do the gather col using bitmap mask. Division by 8 as a single position of bitmap contains 1 byte, so 8 bits for the 8 values in stride_idx step
				__m512d col_vec3 = _mm512_mask_i64gather_pd(zerodef3, bitmap[bmap3],vindex3, &table[stride_col3], 1); //do the gather col using bitmap mask. Division by 8 as a single position of bitmap contains 1 byte, so 8 bits for the 8 values in stride_idx step
				__m512d col_vec4 = _mm512_mask_i64gather_pd(zerodef4, bitmap[bmap4],vindex4, &table[stride_col4], 1); //do the gather col using bitmap mask. Division by 8 as a single position of bitmap contains 1 byte, so 8 bits for the 8 values in stride_idx step
			//store sequentially only valid values of col_vec using bitmap mask
				_mm512_mask_compressstoreu_pd(&projected[k][staddr1], mask, col_vec1); //projection matrix contains a vector for each projected field	
				_mm512_mask_compressstoreu_pd(&projected[k][staddr2], bitmap[bmap2], col_vec2); //projection matrix contains a vector for each projected field	
				_mm512_mask_compressstoreu_pd(&projected[k][staddr3], bitmap[bmap3], col_vec3); //projection matrix contains a vector for each projected field	
				_mm512_mask_compressstoreu_pd(&projected[k][staddr4], bitmap[bmap4], col_vec4); //projection matrix contains a vector for each projected field	
		}
	}
}

void * aggregate(double *table, double *proj_col, uint64_t * len, aggfunction aggop){

	double * retval = malloc(sizeof(double));
	switch(aggop){
		case SUM:
		{
			double s = 0.0;
			for(uint32_t i = 0; i < (*len); i++){
				s += proj_col[i];
			}
			*retval = s;
			free(len);
			return retval; 
			break;
		}
		case AVG:
		{
			double s = 0.0;
			for(uint32_t i = 0; i < (*len); i++)
				s += proj_col[i];
			if ((*len) == 0)
				s = 0;
			else
				s = s / (*len);
			*retval = s;
			free(len);
			return retval; 
			break;
		}
		case COUNT:{
			return (uint64_t*)len;
			break;
		}	
		default:
			return NULL;

	}
	return NULL;
}

address_pair * hash_join(double *a, double *b, size_t * listfieldsa, size_t * listfieldsb,  uint32_t * listcomps, uint32_t lencomps, size_t lena, size_t lenb, size_t * outlen)
{

    size_t stride_addr;
    size_t stride_col;

	htype *table;
	ctype counter;
	table = malloc(lena * sizeof(htype));
	if (!table){
		printf("error allocating ht\n");
		exit(1);
	}
	init_table(table,lena);
	init_counter(&counter);
		

	//gather stride
    __m512i vindex = _mm512_set_epi64(TUPLE_SIZE*7, TUPLE_SIZE*6, TUPLE_SIZE*5, TUPLE_SIZE*4, TUPLE_SIZE*3, TUPLE_SIZE*2, TUPLE_SIZE, 0);
	// BUILD PHASE - HASH TABLE CONSTRUCTION USING THE SMALLEST TABLE
    for(size_t stride_idx=0; stride_idx < lena; stride_idx+=8) {
        stride_addr = stride_idx * NUM_COLS;
        stride_col = stride_addr + listfieldsa[0]; //replace 1 by field var
		 __m512d keyvec = _mm512_i64gather_pd(vindex, &a[stride_col], 1); //do the gather col1
        __m512i keyvals = _mm512_cvtpd_epi64(keyvec); //explicitly convert to long, otherwise we run into serious trouble.
		__m512i index = _mm512_fnv1a_epi64(keyvals);		 // "index" receives an index created using fnv1a hash

		//now for each index insert into ht. Inefficient, but can't really vectorize unless we change the hashtable to some dynamic reallocation strategy
		for (int i = 0; i < 8; i++)
		{
			keyval data;
			data.key = keyvec[i];
			uint64_t offset = (i*TUPLE_SIZE) >> 3; //dereferencing a double in next line, thus we divide by 8 (>> 3)
			data.address = (double*)&a[stride_col + offset];
			uint32_t hashkey = index[i] % lena; //find index within range of hash table
			store(data, hashkey, table, &counter);
		}
	}

	uint32_t outsize = (lena/2);//use half of the smallest table as conservative base
	address_pair * output = malloc(sizeof(address_pair)*outsize); 
	uint32_t matches = 0;


	// PROBE PHASE - run through table b and find matches
	for (size_t stride_idx = 0; stride_idx < lenb; stride_idx+=8)
	{	
        stride_addr = stride_idx * NUM_COLS;
        stride_col = stride_addr + listfieldsb[0]; //2nd field var
		//can't vectorize cause of tuple indirection
        __m512d keyvec = _mm512_i64gather_pd(vindex, &b[stride_col], 1); //do the gather col1, implicit double -> long int cast here
        __m512i keyvals = _mm512_cvtpd_epi64(keyvec); //explicitly convert to long, otherwise we run into serious trouble.
		__m512i index = _mm512_fnv1a_epi64(keyvals);		 // "index" receives an index created using fnv1a hash
		for (int i = 0; i < 8; i++)
		{
			keyval data;
			data.key = keyvec[i];
			uint64_t offset = (i*NUM_COLS);
			data.address = (double*)&b[stride_col + offset];
			uint32_t hashindex = index[i] % lena;
			htype *aux;
			aux = &table[hashindex];

			if (table[hashindex].data.key != 0) 
			{

				for (int i = 0; i < table[hashindex].hits; i++)
				{
					if (aux->data.key == data.key) //first field comparison
					{
						//check if all conditions are respected
						int stillmatching = 1;
						double * tuple1 = (double*)aux->data.address - (listfieldsa[0]); //removing offset
						double * tuple2 = (double*)data.address - (listfieldsa[0]);	//removing offset

						for (int comps = 1; comps < lencomps; comps++) //first comparison was equal op used for hash table, now we run through every comparison operator in listcomps passed for fields listed in listfieldsa and listfieldsb
						{
							switch(listcomps[comps]){
							case 0: stillmatching = stillmatching && ( tuple1[listfieldsa[comps]] == tuple2[listfieldsb[comps]]);
								break;
							case 1: stillmatching = stillmatching && tuple1[listfieldsa[comps]] != tuple2[listfieldsb[comps]];
								break;
							case 2: stillmatching = stillmatching && tuple1[listfieldsa[comps]] > tuple2[listfieldsb[comps]];
								break;
							case 3: stillmatching = stillmatching && tuple1[listfieldsa[comps]] >= tuple2[listfieldsb[comps]];
								break;
							case 4: stillmatching = stillmatching && tuple1[listfieldsa[comps]] < tuple2[listfieldsb[comps]];
									break;
							case 5: stillmatching = stillmatching && tuple1[listfieldsa[comps]] <= tuple2[listfieldsb[comps]];
								break;
							default:
								printf("invalid comparison operator\n");	
							}

						}

						if (stillmatching)
						{
						// a match was found, add to output
							address_pair input;
							input.addr1 = tuple1;
							input.addr2 = tuple2;
							output[matches] = input;
							matches++;
							if (matches >= outsize)
							{//we have to resize! we cannot fit any more data in here
								outsize *= 2; //double the size
								output = realloc(output,sizeof(address_pair)*outsize);
							}
						}
					}			
					aux = aux->next;
				}
			}
		}
	}	
	*outlen = matches; //additional parameter to return the size of row matches
	return output;
}



void * Q1_7( double *table, uint64_t * projfields, uint64_t projections, double ** projected, aggfunction aggop, size_t idx_f10, uint32_t x, int op)
{
	uint64_t * totalmatches = malloc(sizeof(uint64_t));

	__mmask8 * bitmap = (__mmask8*)aligned_alloc(64, sizeof(__mmask8)*(NUM_TUPLES >> 3)); //8b per position, representing bitmap
	memset(bitmap, 0, sizeof(__mmask8) * NUM_TUPLES >> 3);	
	*totalmatches = filter(table, idx_f10, x, bitmap);

	for (int i = 0; i < projections; i++){
		projected[i] = (double *)aligned_alloc(TUPLE_SIZE, sizeof(double)*(*totalmatches));
		memset(projected[i], 0, sizeof(double) * (*totalmatches));
	}
	project(table, bitmap, projfields, projections, projected);		
	free(bitmap);

	if (aggop != NOTHING){
		return (double*)aggregate(table, projected[0], totalmatches, aggop);
	}	
	
	return totalmatches;
}


/*void * Q12_Q13( const double *table, uint64_t * updatefields, double * updatevalues, uint64_t updates, size_t idx_f10, double z, int op)
{

	uint64_t totalmatches = 0;
	printf("entered here somehow\n");
	totalmatches = UPDATEfilter(table, idx_f10, z, updatevalues, updatefields, updates);
	return totalmatches;
}
*/

void * Q10_11( double *table, uint64_t * projfields, uint64_t projections, double ** projected, aggfunction aggop, size_t * fields, double * targets, size_t comps)
{
	uint64_t * totalmatches = malloc(sizeof(uint64_t));
	*totalmatches = NUM_TUPLES;
	__mmask8 * bitmap = (__mmask8*)aligned_alloc(TUPLE_SIZE, sizeof(__mmask8)*(NUM_TUPLES >> 3)); //8b per position, representing bitmap
	memset(bitmap, 0xff, sizeof(__mmask8) * NUM_TUPLES >> 3);	//initialize bitmap with 1 in every field
	for (int i  = 0; i < comps; i++)
	{
		*totalmatches = filterAND(table, fields[i], targets[i], bitmap);
	}	

	for (int i = 0; i < projections; i++){
		projected[i] = (double *)aligned_alloc(TUPLE_SIZE, sizeof(double)* (*totalmatches));
		memset(projected[i], 0, sizeof(double) * (* totalmatches));
	}
	project(table, bitmap, projfields, projections, projected);		

	if (aggop != NOTHING){
		return (double*)aggregate(table, projected[0], totalmatches, aggop);
	}	

	free(bitmap);
	return (uint64_t *)totalmatches;
}

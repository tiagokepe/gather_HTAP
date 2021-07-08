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

uint64_t filter(const double *table, size_t idx_col1, double val, __mmask8 * bitmap)
{
	//gather stride
    __m512i vindex = _mm512_set_epi64(0, TUPLE_SIZE, TUPLE_SIZE*2, TUPLE_SIZE*3, TUPLE_SIZE*4, TUPLE_SIZE*5, TUPLE_SIZE*6, TUPLE_SIZE*7);
	
    size_t stride_addr;
    size_t stride_col;

	uint64_t base_address = 0;
	
	const int op1 = 17;
	__m512d col_vec2 = _mm512_set1_pd(val);//sets 2nd avx reg with val in all positions 
    for(size_t stride_idx=0; stride_idx < NUM_TUPLES; stride_idx+=8) {
        stride_addr = stride_idx * NUM_COLS;
        stride_col = stride_addr + idx_col1;
        __m512d col_vec1 = _mm512_i64gather_pd(vindex, &table[stride_col], 1); //do the gather col1
		//#if DEBUG
        //print_m512d(col_vec1);
        //#endif

        __mmask8 res_vec_mask = _mm512_cmp_pd_mask(col_vec1, col_vec2, op1); //mask = col1 OP col2

		//approach 2: bitmap
		_store_mask8(&bitmap[(stride_idx >> 3)], res_vec_mask);	

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
        	__m512d col_vec = _mm512_mask_i64gather_pd(zerodef, bitmap[stride_idx >> 3],vindex, &table[stride_col], 1); //do the gather col using mask
			//store col_vec using mask
			_mm512_mask_compressstoreu_pd(&projected[k][base_address], bitmap[stride_idx >> 3], col_vec); //projection matrix contains a vector for each projected field	
		}
		//convert mask bits to integer		
		uint32_t maskbits_gpr = _cvtmask8_u32(bitmap[stride_idx >> 3]);
		
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
			for(uint32_t i = 0; i < len; i++){
				s += proj_col[i];
			}
			*retval = s;
			return retval; 
			break;
		}
		case AVG:
		{
			double s = 0;
			for(uint32_t i = 0; i < len; i++)
				s += proj_col[i];
			if (len == 0)
				s = 0;
			else
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

address_pair * hash_join(const double *a, const double *b, size_t * list1, size_t * list2,  uint32_t * list3, uint32_t lencomps, size_t lena, size_t lenb, size_t * outlen)
{

    size_t stride_addr;
    size_t stride_col;

	uint64_t base_address = 0;
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
    __m512i vindex = _mm512_set_epi64(0, TUPLE_SIZE, TUPLE_SIZE*2, TUPLE_SIZE*3, TUPLE_SIZE*4, TUPLE_SIZE*5, TUPLE_SIZE*6, TUPLE_SIZE*7);
	// BUILD PHASE - HASH TABLE CONSTRUCTION USING THE SMALLEST TABLE
    for(size_t stride_idx=0; stride_idx < lena; stride_idx+=8) {
        stride_addr = stride_idx * NUM_COLS;
        stride_col = stride_addr + list1[0]; //replace 1 by field var
        __m512i keyvec = _mm512_i64gather_epi64(vindex, &a[stride_col], 1); //do the gather col1.. hopefully it works?! implicit double -> long int cast here
		__m512i index = _mm512_fnv1a_epi64(keyvec);		 // "index" receives an index created using fnv1a hash

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
        stride_col = stride_addr + list2[0]; //2nd field var
		//can't vectorize cause of tuple indirection
        __m512i keyvec = _mm512_i64gather_epi64(vindex, &b[stride_col], 1); //do the gather col1, implicit double -> long int cast here
		__m512i index = _mm512_fnv1a_epi64(keyvec);		 // "index" receives an index created using fnv1a hash
		for (int i = 0; i < 8; i++)
		{
			keyval data;
			data.key = keyvec[i];
			uint64_t offset = (i*TUPLE_SIZE) >> 3;
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
						double * tuple1 = (double*)aux->data.address;
						double * tuple2 = (double*)data.address;	

						for (int comps = 1; comps < lencomps; comps++) //first comparison was equal op used for hash table, now we run through every comparison operator in list3 passed for fields listed in list1 and list2
						{
							switch(list3[comps]){
							case 0: stillmatching = stillmatching && ( tuple1[list1[comps]] == tuple2[list2[comps]]);
								break;
							case 1: stillmatching = stillmatching && tuple1[list1[comps]] != tuple2[list2[comps]];
								break;
							case 2: stillmatching = stillmatching && tuple1[list1[comps]] > tuple2[list2[comps]];
								break;
							case 3: stillmatching = stillmatching && tuple1[list1[comps]] >= tuple2[list2[comps]];
								break;
							case 4: {
									stillmatching = stillmatching && tuple1[list1[comps]] < tuple2[list2[comps]];
									break;
									}
							case 5: stillmatching = stillmatching && tuple1[list1[comps]] <= tuple2[list2[comps]];
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



void * Q1_7( const double *table, uint64_t * projfields, uint64_t projections, double ** projected, aggfunction aggop, size_t idx_f10, uint32_t x, int op)
{
	uint64_t totalmatches;

	__mmask8 * bitmap = (__mmask8*)aligned_alloc(TUPLE_SIZE, sizeof(__mmask8)*(NUM_TUPLES >> 3)); //8b per position, representing bitmap
	memset(bitmap, 0, sizeof(__mmask8) * NUM_TUPLES >> 3);	
	totalmatches = filter(table, idx_f10, x, bitmap);
	printf("got totalmatches %lu\n", totalmatches);

	for (int i = 0; i < projections; i++){
		projected[i] = (double *)aligned_alloc(TUPLE_SIZE, sizeof(double)*totalmatches);
		memset(projected[i], 0, sizeof(double) * totalmatches);
	}
	project(table, bitmap, projfields, projections, projected);		

	if (aggop != NOTHING){
		return aggregate(table, projected[0], totalmatches, aggop);
	}	

	free(bitmap);
	return totalmatches;
}

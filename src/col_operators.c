#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include "utils/defines.h"
#include "utils/global_vars.h" //TUPLE_SIZE and NUM_COLS
#include <stdint.h>
#include <string.h>

#include "col_operators.h"
#include "utils/utils.h"
#include "vecmurmur.h"
#include "hash_table.h"

uint64_t filter(double *table, size_t idx_col1, double val, __mmask8 * bitmap)
{
	uint64_t popcnt = 0;
	
	const int op1 = 17;
	__m512d add_vec1 = _mm512_set1_pd(val);//sets 2nd avx reg with val in all positions 
//	__m512d add_vec2 = _mm512_set1_pd(val);//sets 2nd avx reg with val in all positions 
//	__m512d add_vec3 = _mm512_set1_pd(val);//sets 2nd avx reg with val in all positions 
//	__m512d add_vec4 = _mm512_set1_pd(val);//sets 2nd avx reg with val in all positions 
	
	size_t offset = NUM_TUPLES * idx_col1; //desired column
	size_t stride_col1 = offset; //initial position (0/4)
//	size_t stride_col2 = offset + upperbound; //  1/4 pos
//	size_t stride_col3 = offset + (upperbound << 1); // 2/4 pos
//	size_t stride_col4 = stride_col3 + upperbound; // 3/4 pos

    for(size_t stride_idx=0; stride_idx < NUM_TUPLES; stride_idx+=8) {
        
		__m512d col_vec1 = _mm512_load_pd(&table[stride_col1]); 
//		__m512d col_vec2 = _mm512_load_pd(&table[stride_col2]);
//		__m512d col_vec3 = _mm512_load_pd(&table[stride_col3]);
//		__m512d col_vec4 = _mm512_load_pd(&table[stride_col4]);

        __mmask8 res_vec_mask1 = _mm512_cmp_pd_mask(col_vec1, add_vec1, op1); //mask = col1 OP col2
//        __mmask8 res_vec_mask2 = _mm512_cmp_pd_mask(col_vec2, add_vec2, op1); //mask = col1 OP col2
 //       __mmask8 res_vec_mask3 = _mm512_cmp_pd_mask(col_vec3, add_vec3, op1); //mask = col1 OP col2
  //      __mmask8 res_vec_mask4 = _mm512_cmp_pd_mask(col_vec4, add_vec4, op1); //mask = col1 OP col2

		//manual indices
		_store_mask8(&bitmap[((stride_col1 - offset) >> 3)], res_vec_mask1);	
//		_store_mask8(&bitmap[((stride_col2 - offset) >> 3)], res_vec_mask2);	
//		_store_mask8(&bitmap[((stride_col3 - offset) >> 3)], res_vec_mask3);	
//		_store_mask8(&bitmap[((stride_col4 - offset) >> 3)], res_vec_mask4);
		stride_col1 += 8;
//		stride_col2 += 8;
//		stride_col3 += 8;
//		stride_col4 += 8;
		//convert mask bits to integer and count bits		
		uint32_t maskbits_gpr;
		maskbits_gpr = _cvtmask8_u32(res_vec_mask1);
		popcnt += _mm_popcnt_u32(maskbits_gpr);
//		maskbits_gpr = _cvtmask8_u32(res_vec_mask2);
//		popcnt += _mm_popcnt_u32(maskbits_gpr);
//		maskbits_gpr = _cvtmask8_u32(res_vec_mask3);
//		popcnt += _mm_popcnt_u32(maskbits_gpr);
//		maskbits_gpr = _cvtmask8_u32(res_vec_mask4);
//		popcnt += _mm_popcnt_u32(maskbits_gpr);
	}
	return popcnt;
}

uint64_t UPDATEfilter(double *table, size_t idx_col1, double val, double * updateval, size_t * idx_update, uint64_t updates)
{
	uint64_t popcnt = 0;
	
	size_t upperbound = NUM_TUPLES >> 2;
	size_t offset = NUM_TUPLES* idx_col1; //desired column
	size_t stride_col1 = offset; //initial position (0/4)
	size_t stride_col2 = offset + upperbound; //  1/4 pos
	size_t stride_col3 = offset + (upperbound << 1); // 2/4 pos
	size_t stride_col4 = stride_col3 + upperbound; // 3/4 pos
	
	const int op1 = 0;
	__m512d add_vec1 = _mm512_set1_pd(val);//sets 2nd avx reg with val in all positions 
	__m512d add_vec2 = _mm512_set1_pd(val);//sets 2nd avx reg with val in all positions 
	__m512d add_vec3 = _mm512_set1_pd(val);//sets 2nd avx reg with val in all positions 
	__m512d add_vec4 = _mm512_set1_pd(val);//sets 2nd avx reg with val in all positions 
    for(size_t stride_idx=0; stride_idx < upperbound; stride_idx+=8) {
       
		__m512d col_vec1 = _mm512_load_pd(&table[stride_col1]); 
		__m512d col_vec2 = _mm512_load_pd(&table[stride_col2]);
		__m512d col_vec3 = _mm512_load_pd(&table[stride_col3]);
		__m512d col_vec4 = _mm512_load_pd(&table[stride_col4]);

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

		/// stride_col1 contains address of column being checked (offset val). If we substract offset, we get to first column. Then we add the update column offset
		/// so we are at the right position 
			size_t update_offset = NUM_TUPLES * idx_update[i]; //desired column
			size_t stride_updcol1 = stride_col1 + update_offset - offset; 
			size_t stride_updcol2 = stride_col2 + update_offset - offset;
			size_t stride_updcol3 = stride_col3 + update_offset - offset;
			size_t stride_updcol4 = stride_col4 + update_offset - offset;
			
			_mm512_mask_store_pd(&table[stride_updcol1], res_vec_mask1,update_col1); //write update value on matching fields
			_mm512_mask_store_pd(&table[stride_updcol2], res_vec_mask2,update_col2); //write update value on matching fields
			_mm512_mask_store_pd(&table[stride_updcol3], res_vec_mask3,update_col3); //write update value on matching fields
			_mm512_mask_store_pd(&table[stride_updcol4], res_vec_mask4,update_col4); //write update value on matching fields
		}
		//manual indices' update
		stride_col1 += 8;
		stride_col2 += 8;
		stride_col3 += 8;
		stride_col4 += 8;
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
	uint64_t popcnt = 0;
	
	const int op1 = 30;
	__m512d add_vec1 = _mm512_set1_pd(val);//sets 2nd avx reg with val in all positions 
	__m512d add_vec2 = _mm512_set1_pd(val);//sets 2nd avx reg with val in all positions 
	__m512d add_vec3 = _mm512_set1_pd(val);//sets 2nd avx reg with val in all positions 
	__m512d add_vec4 = _mm512_set1_pd(val);//sets 2nd avx reg with val in all positions 
	
	size_t upperbound = NUM_TUPLES >> 2;
	size_t offset = NUM_TUPLES * idx_col1; //desired column
	
	size_t stride_col1 = offset; //initial position (0/4)
	size_t stride_col2 = offset + upperbound; //  1/4 pos
	size_t stride_col3 = offset + (upperbound << 1); // 2/4 pos
	size_t stride_col4 = stride_col3 + upperbound; // 3/4 pos
    
	for(size_t stride_idx=0; stride_idx < upperbound; stride_idx+=8) {
		
		__m512d col_vec1 = _mm512_load_pd(&table[stride_col1]); 
		__m512d col_vec2 = _mm512_load_pd(&table[stride_col2]); 
		__m512d col_vec3 = _mm512_load_pd(&table[stride_col3]); 
		__m512d col_vec4 = _mm512_load_pd(&table[stride_col4]); 

        __mmask8 res_vec_mask1 = _mm512_cmp_pd_mask(col_vec1, add_vec1, op1); //mask = col1 OP col2
        __mmask8 res_vec_mask2 = _mm512_cmp_pd_mask(col_vec2, add_vec2, op1); //mask = col1 OP col2
        __mmask8 res_vec_mask3 = _mm512_cmp_pd_mask(col_vec3, add_vec3, op1); //mask = col1 OP col2
        __mmask8 res_vec_mask4 = _mm512_cmp_pd_mask(col_vec4, add_vec4, op1); //mask = col1 OP col2

		//AND approach: load bitmap, AND, store bitmap.
		__mmask8 current_val1 = _load_mask8(&bitmap[((stride_col1 - offset) >> 3)]);
		__mmask8 current_val2 = _load_mask8(&bitmap[((stride_col2 - offset) >> 3)]);
		__mmask8 current_val3 = _load_mask8(&bitmap[((stride_col3 - offset) >> 3)]);
		__mmask8 current_val4 = _load_mask8(&bitmap[((stride_col4 - offset) >> 3)]);
		res_vec_mask1 = _kand_mask8(res_vec_mask1, current_val1);
		res_vec_mask2 = _kand_mask8(res_vec_mask2, current_val2);
		res_vec_mask3 = _kand_mask8(res_vec_mask3, current_val3);
		res_vec_mask4 = _kand_mask8(res_vec_mask4, current_val4);

		_store_mask8(&bitmap[((stride_col1 - offset) >> 3)], res_vec_mask1);	
		_store_mask8(&bitmap[((stride_col2 - offset) >> 3)], res_vec_mask2);	
		_store_mask8(&bitmap[((stride_col3 - offset) >> 3)], res_vec_mask3);	
		_store_mask8(&bitmap[((stride_col4 - offset) >> 3)], res_vec_mask4);	
		
		//manual indices
		stride_col1 += 8;
		stride_col2 += 8;
		stride_col3 += 8;
		stride_col4 += 8;

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

void project(double *table, __mmask8 *bitmap, uint64_t * proj_vec, uint64_t proj_size, double ** projected)
{
	__m512d zerodef1 = _mm512_setzero_pd();	
	
	for (uint64_t k = 0; k < proj_size; k++){// loop around intended projections
		size_t stride_col1 = proj_vec[k] * NUM_TUPLES; //projection column offset
	
		uint64_t base_address = 0;
		for(size_t stride_idx = 0; stride_idx < NUM_TUPLES; stride_idx+=8) {
			
			uint32_t maskbits_gpr;
			__mmask8 mask = bitmap[stride_idx >> 3];	
			__m512d col_vec1 = _mm512_mask_load_pd(zerodef1, mask, &table[stride_col1 + stride_idx]); //do the gather col using mask
			maskbits_gpr = _cvtmask8_u32(mask);

			_mm512_mask_compressstoreu_pd(&projected[k][base_address], mask, col_vec1); //projection matrix contains a vector for each projected field	
			base_address += _mm_popcnt_u32(maskbits_gpr);
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
		}
		break;

		case AVG:
		{
			double s = 0.0;
			for(uint32_t i = 0; i < (*len); i++)
				s += proj_col[i];
			if ((*len) == 0)
				s = 0.0;
			else
				s = s / ( *len);
			*retval = s;
			free(len);
			return retval; 
		}
			break;
		case COUNT:{
			return (uint64_t*)len;
		}	
		break;
		default:
			return NULL;

	}
	return NULL;
}

address_pair * hash_join(double *a, double *b, size_t * listfieldsa, size_t * listfieldsb,  uint32_t * listcomps, uint32_t lencomps, size_t lena, size_t lenb, size_t * outlen)
{
	htype *table;
	ctype counter;
	table = malloc(lena * sizeof(htype));
	if (!table){
		printf("error allocating ht\n");
		exit(1);
	}
	init_table(table,lena);
	init_counter(&counter);
		
	
	size_t fieldoffset_a = NUM_TUPLES * listfieldsa[0]; //desired column
	size_t stride_col1 = fieldoffset_a; //initial position (0/4)
	// BUILD PHASE - HASH TABLE CONSTRUCTION USING THE SMALLEST TABLE
    for(size_t stride_idx=0; stride_idx < lena; stride_idx+=8) {

        __m512d keyvec1 = _mm512_load_pd(&a[stride_col1]); //do the gather col1.. hopefully it works?! implicit double -> long int cast here
//        __m512i keyvec2 = _mm512_load_epi64(&a[stride_col2]); //do the gather col1.. hopefully it works?! implicit double -> long int cast here
 //      __m512i keyvec3 = _mm512_load_epi64(&a[stride_col3]); //do the gather col1.. hopefully it works?! implicit double -> long int cast here
  //      __m512i keyvec4 = _mm512_load_epi64(&a[stride_col4]); //do the gather col1.. hopefully it works?! implicit double -> long int cast here
        __m512i keyval1 = _mm512_cvtpd_epi64(keyvec1); //do the gather col1.. hopefully it works?! implicit double -> long int cast here

		__m512i index1 = _mm512_fnv1a_epi64(keyval1);		 // "index" receives an index created using fnv1a hash
//		__m512i index2 = _mm512_fnv1a_epi64(keyvec2);		 // "index" receives an index created using fnv1a hash
//		__m512i index3 = _mm512_fnv1a_epi64(keyvec3);		 // "index" receives an index created using fnv1a hash
//		__m512i index4 = _mm512_fnv1a_epi64(keyvec4);		 // "index" receives an index created using fnv1a hash

		//now for each index insert into ht. Inefficient, but can't really vectorize unless we change the hashtable to some dynamic reallocation strategy
		for (int i = 0; i < 8; i++)
		{
			keyval data1;
//			keyval data2;
//			keyval data3;
//			keyval data4;
			data1.key = keyvec1[i];
//			data2.key = keyvec2[i];
//			data3.key = keyvec3[i];
//			data4.key = keyvec4[i];
			data1.address = (double*)&a[stride_col1 + i];
//			data2.address = (double*)&a[stride_col2 + i];
//			data3.address = (double*)&a[stride_col3 + i];
//			data4.address = (double*)&a[stride_col4 + i];
			uint32_t hashkey1 = index1[i] % lena; //find index within range of hash table
//			uint32_t hashkey2 = index2[i] % lena; //find index within range of hash table
//			uint32_t hashkey3 = index3[i] % lena; //find index within range of hash table
//			uint32_t hashkey4 = index4[i] % lena; //find index within range of hash table
			store(data1, hashkey1, table, &counter);
//			store(data2, hashkey2, table, &counter);
//			store(data3, hashkey3, table, &counter);
//			store(data4, hashkey4, table, &counter);
		}
		//manual indice update
		stride_col1 += 8;
//		stride_col2 += 8;
//		stride_col3 += 8;
//		stride_col4 += 8;
	}

	uint32_t outsize = (lena/2);//use half of the smallest table as conservative base
	address_pair * output = malloc(sizeof(address_pair)*outsize); 
	uint32_t matches = 0;


	size_t fieldoffset_b = listfieldsb[0]*NUM_TUPLES;
	stride_col1 = fieldoffset_b; //initial position (0/4)
//	stride_col2 = offset + upperbound; //  1/4 pos
//	stride_col3 = offset + (upperbound << 1); // 2/4 pos
//	stride_col4 = stride_col3 + upperbound; // 3/4 pos

	// PROBE PHASE - run through table b and find matches
	for (size_t stride_idx = 0; stride_idx < lenb; stride_idx+=8)
	{	
		//can't vectorize cause of tuple indirection
        __m512d keyvec1 = _mm512_load_pd(&b[stride_col1]); //
//        __m512d keyvec2 = _mm512_load_pd(&b[stride_col2]); //
 //       __m512d keyvec3 = _mm512_load_pd(&b[stride_col3]); //
  //      __m512d keyvec4 = _mm512_load_pd(&b[stride_col4]); //
        
		__m512i keyval1 = _mm512_cvtpd_epi64(keyvec1);
//		__m512i keyval2 = _mm512_cvtpd_epi64(keyvec2);
//		__m512i keyval3 = _mm512_cvtpd_epi64(keyvec3);
//		__m512i keyval4 = _mm512_cvtpd_epi64(keyvec4);

		__m512i index1 = _mm512_fnv1a_epi64(keyval1);		 // "index" receives an index created using fnv1a hash
//		__m512i index2 = _mm512_fnv1a_epi64(keyval2);		 // "index" receives an index created using fnv1a hash
//		__m512i index3 = _mm512_fnv1a_epi64(keyval3);		 // "index" receives an index created using fnv1a hash
//		__m512i index4 = _mm512_fnv1a_epi64(keyval4);		 // "index" receives an index created using fnv1a hash
		for (int i = 0; i < 8; i++)
		{
			keyval data;
			data.key = keyvec1[i];
//			data[1].key = keyvec2[i];
//			data[2].key = keyvec3[i];
//			data[3].key = keyvec4[i];
			data.address = (double*)&b[stride_col1 + i];
//			data[1].address = (double*)&b[stride_col2 + i];
//			data[2].address = (double*)&b[stride_col3 + i];
//			data[3].address = (double*)&b[stride_col4 + i];

			uint32_t hashindexvec[4];
			hashindexvec[0] = index1[i] % lena;
//			hashindexvec[1] = index2[i] % lena;
//			hashindexvec[2] = index3[i] % lena;
//			hashindexvec[3] = index4[i] % lena;
			
//			for (int unroll = 0; unroll < 4; unroll++)
//			{
				htype *aux;
				uint32_t hashindex = hashindexvec[0];
				aux = &table[hashindex];

				if (table[hashindex].data.key != 0) 
				{

					for (int i = 0; i < table[hashindex].hits; i++)
					{
						if (aux->data.key == data.key) //first field comparison
						{
							//check if all conditions are respected
							int stillmatching = 1;
	
							double * tuple1 = (double*)aux->data.address - (fieldoffset_a); // remove offset to get base address of tuple
							double * tuple2 = (double*)data.address - (fieldoffset_b); //remove offset to get base address of tuple	


							for (int comps = 1; comps < lencomps; comps++) //first comparison was equal op used for hash table, now we run through every comparison operator in listcomps passed for fields listed in listfieldsa and listfieldsb
							{
								double aval = tuple1[listfieldsa[comps]*NUM_TUPLES];
								double bval = tuple2[listfieldsb[comps]*NUM_TUPLES];

								switch(listcomps[comps]){
								case 0: stillmatching = stillmatching && ( aval == bval);
									break;
								case 1: stillmatching = stillmatching && ( aval != bval);
									break;
								case 2: stillmatching = stillmatching && ( aval > bval);
									break;
								case 3: stillmatching = stillmatching && ( aval >= bval);
									break;
								case 4:	stillmatching = stillmatching && ( aval < bval);
									break;
								case 5: stillmatching = stillmatching && ( aval <= bval);
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
			//}
		}
		//manual indice update
		stride_col1 += 8;
//		stride_col2 += 8;
//		stride_col3 += 8;
//		stride_col4 += 8;
	}	
	*outlen = matches; //additional parameter to return the size of row matches
	return output;
}



void * Q1_7( double *table, uint64_t * projfields, uint64_t projections, double ** projected, aggfunction aggop, size_t idx_f10, uint32_t x, int op)
{
	uint64_t * totalmatches = malloc(sizeof(uint64_t));

	__mmask8 * bitmap = (__mmask8*)aligned_alloc(TUPLE_SIZE, sizeof(__mmask8)*(NUM_TUPLES >> 3)); //8b per position, representing bitmap
	memset(bitmap, 0, sizeof(__mmask8) * (NUM_TUPLES >> 3));	
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
	return (uint64_t *)totalmatches;
}



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
		projected[i] = (double *)aligned_alloc(TUPLE_SIZE, sizeof(double)*(*totalmatches));
		memset(projected[i], 0, sizeof(double) * (* totalmatches));
	}
	project(table, bitmap, projfields, projections, projected);		

	if (aggop != NOTHING){
		return (double*)aggregate(table, projected[0], totalmatches, aggop);
	}	

	free(bitmap);
	return (uint64_t*)totalmatches;
}

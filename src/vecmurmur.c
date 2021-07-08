#include "vecmurmur.h"

__m512i _mm512_murmur3_epi64(__m512i keys, const uint32_t seed)
{

  __m512i c1 = _mm512_set1_epi64(0xcc9e2d51);
  __m512i c2 = _mm512_set1_epi64(0x1b873593);
  const int r1 = 15;
  const int r2 = 13;
  const __m512i m = _mm512_set1_epi64(5);
  const __m512i n = _mm512_set1_epi64(0xe6546b64);
  __m512i hash = _mm512_set1_epi64(seed);
  __m512i k1 = _mm512_setzero_si512();
  __m512i k2 = _mm512_setzero_si512();

  //Multiply
  keys = _mm512_mullo_epi64(keys, c1);

  //Rotate left
  k1 = _mm512_slli_epi64(keys, r1);
  k2 = _mm512_srli_epi64(keys, (32-r1));
  keys = _mm512_or_si512(k1, k2);

  //Multiply
  keys = _mm512_mullo_epi64(keys, c2);

  //XOR
  hash = _mm512_xor_si512(hash, keys);

  //Rotate left
  k1 = _mm512_slli_epi64(hash, r2);
  k2 = _mm512_srli_epi64(hash, (32-r2));
  hash = _mm512_or_si512(k1, k2);

  //Multiply Add
  hash = _mm512_mullo_epi64(hash, m);
  hash = _mm512_add_epi64(hash, n);

  //FINAL AVALANCHE!

  c1 = _mm512_set1_epi64(0x85ebca6b);
  c2 = _mm512_set1_epi64(0xc2b2ae35);

  //Shift 16 and xor
  k1 = _mm512_srli_epi64(hash, 16);
  hash = _mm512_xor_si512(hash, k1);

  hash = _mm512_mullo_epi64(hash, c1);

  k1 = _mm512_srli_epi64(hash, 13);
  hash = _mm512_xor_si512(hash, k1);

  hash = _mm512_mullo_epi64(hash, c2);

  k1 = _mm512_srli_epi64(hash, 16);
  hash = _mm512_xor_si512(hash, k1);

  return hash;

}

inline __m512i _mm512_fnv1a_epi64(__m512i data)
{
	__m512i hash = _mm512_set1_epi64(2166136261);
	__m512i prime = _mm512_set1_epi64(16777619);
	__m512i shift;

	//Extract the byte
	shift = _mm512_slli_epi64(data, 24);
	shift = _mm512_srli_epi64(shift, 24);

	//Hash
	hash = _mm512_xor_si512(hash, shift);
	hash = _mm512_mullo_epi64(hash, prime);

	//Extract the byte
	shift = _mm512_slli_epi64(data, 16);
	shift = _mm512_srli_epi64(shift, 24);

	//Hash
	hash = _mm512_xor_si512(hash, shift);
	hash = _mm512_mullo_epi64(hash, prime);

	//Extract the byte
	shift = _mm512_slli_epi64(data, 8);
	shift = _mm512_srli_epi64(shift, 24);

	//Hash
	hash = _mm512_xor_si512(hash, shift);
	hash = _mm512_mullo_epi64(hash, prime);

	//Extract the byte
	shift = _mm512_srli_epi64(data, 24);

	//Hash
	hash = _mm512_xor_si512(hash, shift);
	hash = _mm512_mullo_epi64(hash, prime);

	return hash;
}


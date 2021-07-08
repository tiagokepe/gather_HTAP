#include <stdint.h>
#define MURMURSEED 0x0D50064F7
#include <immintrin.h>

__m512i _mm512_murmur3_epi64(__m512i keys, const uint32_t seed);
__m512i _mm512_fnv1a_epi64(__m512i data);

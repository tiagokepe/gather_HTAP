//#include <zmmintrin.h>

#include <immintrin.h>

int main () {
    __m256i vindex = _mm256_set_epi32(0,2048,4096,6144,8192,10240,12288,14366);

    __m512d Bvec = _mm512_i32gather_pd(vindex, &B[k*SZ + j], 8); //do the gather

    for (int i = 0; i < SZ; i++) {
        __m512d Cres = _mm512_set_pd(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for (int j = 0; j < SZ; j++) {
            for (int k = 0; k < SZ; k+=8) {
                //load A
                __m512d Avec = _mm512_load_pd(&A[i*SZ + k]);
                //load B
                __m512d Bvec = _mm512_i32gather_pd(vindex, &B[k*SZ + j], 8); //do the gather
                //Cres += A * B
                Cres = _mm512_fmadd_pd(Avec, Bvec, Cres);
            }
            //C = store Cres
            C[i*SZ + j] = _mm512_reduce_add_pd(Cres);
        }
    }
}

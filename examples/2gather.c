#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>

#define SZ 64

static void matmul_vgather_ijk(const double * A, const double * B, double * C){
    __m256i vindex = _mm256_set_epi32(0,SZ,SZ*2,SZ*3,SZ*4,SZ*5,SZ*6,SZ*7);
    for (int i = 0; i < SZ; i++){
        __m512d Cres = _mm512_set_pd(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for (int j = 0; j < SZ; j++){
              for (int k = 0; k < SZ; k+=8){
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

int main() {

    double *A = (double*)aligned_alloc(64,sizeof(double)*SZ);
    double *B = (double*)aligned_alloc(64,sizeof(double)*SZ);
    double *C = (double*)aligned_alloc(64,sizeof(double)*SZ);
    
    printf("Pointers: %p, %p, %p\n", A, B, C);
    printf("A[0] = %f\n", A[0]);

    matmul_vgather_ijk(A, B, C);
    printf("OK.");

    free(A);
    free(B);
    free(C);
    return 0;
}

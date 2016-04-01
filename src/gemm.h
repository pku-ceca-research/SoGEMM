#ifndef GEMM_H__
#define GEMM_H__

void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float BETA,
    float *C, int ldc);

#ifdef SDS
void gemm_sds(int TA, int TB, int M, int N, int K, float ALPHA,
    float *A, int lda, 
    float *B, int ldb,
    float BETA,
    float *C, int ldc);
#endif

#endif
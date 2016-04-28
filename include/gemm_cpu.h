#ifndef GEMM_CPU_H__
#define GEMM_CPU_H__

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float BETA,
    float *C, int ldc);

#endif

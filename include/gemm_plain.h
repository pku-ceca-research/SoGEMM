#ifndef GEMM_PLAIN_H__
#define GEMM_PLAIN_H__

void gemm_plain(int TA, int TB, int M, int N, int K, float ALPHA, 
    float *A, int lda,
    float *B, int ldb,
    float BETA,
    float *C, int ldc);

/* timers */
void gemm_plain_init_clock();
double gemm_plain_end_clock();
#endif

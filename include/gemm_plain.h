#ifndef GEMM_PLAIN_H__
#define GEMM_PLAIN_H__

template <typename VectorT, typename ScalarT>
void gemm_plain(int TA, int TB, int M, int N, int K,
    ScalarT ALPHA, 
    VectorT *A, int lda,
    VectorT *B, int ldb,
    ScalarT BETA,
    VectorT *C, int ldc);

/* timers */
void gemm_plain_init_clock();
double gemm_plain_end_clock();
#endif

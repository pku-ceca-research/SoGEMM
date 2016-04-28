#ifndef GEMM_BLOCK_H__
#define GEMM_BLOCK_H__

#include "gemm_types.h"

/* exposed for testing */
void gemm_block_kernel(float ALPHA, float BETA, BlockedMatrix *A_blk, BlockedMatrix *B_blk, BlockedMatrix *C_blk);

void gemm_block(int TA, int TB, int M, int N, int K, float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float BETA, 
    float *C, int ldc);
#endif

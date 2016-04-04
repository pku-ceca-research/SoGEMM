#ifndef GEMM_TRANS_H__
#define GEMM_TRANS_H__

float *trans_to_blocked(int T, float *A, int m, int n, int lda, int blk_m, int blk_n);
void trans_from_blocked(int T, float *A, float *A_block, int M, int N, int lda, int blk_m, int blk_n);
blocked_matrix* flatten_matrix_to_blocked(int T, float *A, int M, int N, int lda, int blk_m, int blk_n);


#endif
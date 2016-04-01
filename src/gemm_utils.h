#ifndef GEMM_UTILS_H__
#define GEMM_UTILS_H__

#ifndef D_TYPE
#define D_TYPE float
#endif

void print_matrix(D_TYPE *A, int M, int N);
void print_blocked_matrix(D_TYPE *A, int M, int N, int blk_m, int blk_n);
D_TYPE *random_matrix(int M, int N);
D_TYPE *random_general_matrix(int M, int N, int lda);
int is_matrix_equal(D_TYPE *A, D_TYPE *B, int M, int N, int lda);

#endif
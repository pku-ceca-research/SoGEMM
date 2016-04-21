#ifndef GEMM_GRID_H__
#define GEMM_GRID_H__

#include "gemm_consts.h"
#include "gemm_types.h"

void gemm_grid(int TA, int TB, int M, int N, int K, float ALPHA,
	float *A, int lda,
	float *B, int ldb,
	float BETA,
	float *C, int ldc);


#endif

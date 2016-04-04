#ifndef GEMM_GRID_H__
#define GEMM_GRID_H__

#define BLK_M 64
#define BLK_N 64
#define BLK_K 64

typedef struct {
  float *mat;     // matrix data
  int T;          // transpose flag
  int H, W;       // height and width
  int ld;         // leading dimension
  int bH, bW;     // block height and width
} blocked_matrix;

void gemm_grid(int TA, int TB, int M, int N, int K, float ALPHA,
	float *A, int lda,
	float *B, int ldb,
	float BETA,
	float *C, int ldc);

#endif

#include <stdio.h>
#include "gemm_block_unit.h"

void mmult_kernel(
    float A[BLK_M][BLK_K],
    float B[BLK_K][BLK_N],
    float ALPHA,
    float T[BLK_M*BLK_N])
{
#pragma HLS INLINE self
// scale for GEMM array partitions
#if GEMM_SCALE == 0
  #pragma HLS array_partition variable=A block factor=16 dim=2
  #pragma HLS array_partition variable=B block factor=16 dim=1
#elif GEMM_SCALE == 1
  #pragma HLS array_partition variable=A block factor=16 dim=2
  #pragma HLS array_partition variable=B block factor=16 dim=1
#elif GEMM_SCALE == 2
  #pragma HLS array_partition variable=A block factor=8 dim=2
  #pragma HLS array_partition variable=B block factor=8 dim=1
#endif

  int i, j, k;
  RowLoop: for (i = 0; i < BLK_M; i ++) {
    ColLoop: for (j = 0; j < BLK_N; j ++) {
    #pragma HLS pipeline II=1
      float sum = 0.0;
      ProductLoop: for (k = 0; k < BLK_K; k ++) {
        float temp = A[i][k] * B[k][j];
        sum += temp;
      }
      T[i*BLK_N+j] = sum;
    }
  }
}

void gemm_block_units_mmult(
        float A[BLK_M*BLK_K],
        float B[BLK_K*BLK_N],
        float ALPHA,
        float T[BLK_M*BLK_N])
{
  float A_buf[BLK_M][BLK_K];
  float B_buf[BLK_K][BLK_N];

  int i, j;
  RowCopy: for (i = 0; i < BLK_M; i ++)
    ColCopy: for (j = 0; j < BLK_K; j ++) {
    #pragma HLS pipeline
      A_buf[i][j] = ALPHA * A[i*BLK_K+j];
      // assume A and B has the same shape
      B_buf[i][j] = B[i*BLK_N+j];
    }

  mmult_kernel(A_buf,B_buf,ALPHA,T);
}

void mplus_kernel(
        float T[BLK_M][BLK_N],
        float C[BLK_M][BLK_N],
        float R[BLK_M*BLK_N])
{
#pragma HLS INLINE self
  int i, j;
  PlusRow: for (i = 0; i < BLK_M; i ++)
    PlusCol: for (j = 0; j < BLK_N; j ++)
    #pragma HLS pipeline II=1
      R[i*BLK_N+j] = C[i][j] + T[i][j];
}

void gemm_block_units_mplus(
        float T[BLK_M*BLK_N],
        float C[BLK_M*BLK_N],
        float R[BLK_M*BLK_N])
{
  float T_buf[BLK_M][BLK_N];
  float C_buf[BLK_M][BLK_N];

  int i, j;
  PlusRowCopy: for (i = 0; i < BLK_M; i ++)
    PlusColCopy: for (j = 0; j < BLK_N; j ++) {
    #pragma HLS pipeline
      T_buf[i][j] = T[i*BLK_N+j];
      C_buf[i][j] = C[i*BLK_N+j];
    }

  mplus_kernel(T_buf,C_buf,R);
}

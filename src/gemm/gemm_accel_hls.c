#include "gemm_accel_hls.h"

void gemm_accel_kernel(
    float A[BLK_M][BLK_K], 
    float B[BLK_N][BLK_K], 
    float C[BLK_M][BLK_N], 
    float ALPHA, 
    float R[BLK_M*BLK_N])
{
#pragma HLS INLINE
#if GEMM_SCALE == 3
#pragma HLS array_partition variable=A block factor=24 dim=2
#pragma HLS array_partition variable=B block factor=24 dim=1
#else
#pragma HLS array_partition variable=A block factor=16 dim=2
#pragma HLS array_partition variable=B block factor=16 dim=1
#endif
  
  int i, j, k;
  Row: for (i = 0; i < BLK_M; i ++) {
    Col: for (j = 0; j < BLK_N; j ++) {
    #pragma HLS pipeline II=1
      float sum = C[i][j];
      for (k = 0; k < BLK_K; k ++) {
        float res = A[i][k] * B[k][j];
        sum += res;
      }
      R[i*BLK_N+j] = sum;
    }
  }
}

void gemm_accel_full(
    float A[BLK_M*BLK_K], 
    float B[BLK_N*BLK_K], 
    float C[BLK_M*BLK_N], 
    float ALPHA, 
    float R[BLK_M*BLK_N])
{
  int i, j;
  float A_buf[BLK_M][BLK_K], B_buf[BLK_K][BLK_N], C_buf[BLK_M][BLK_N];
  RowCopy: for (i = 0; i < BLK_M; i ++) 
    ColCopy: for (j = 0; j < BLK_N; j ++) {
    #pragma HLS pipeline
      A_buf[i][j] = ALPHA * A[i*BLK_N+j];
      B_buf[i][j] = B[i*BLK_N+j];
      C_buf[i][j] = C[i*BLK_N+j];
    }

  gemm_accel_kernel(A_buf,B_buf,C_buf,ALPHA,R);
}

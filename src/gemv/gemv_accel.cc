
#include <cstdio>
#include "gemv_accel.hh"

void gemv_accel_kernel(float A[GEMV_BLK_M][GEMV_BLK_N], float X[GEMV_BLK_N], 
    float Y[GEMV_BLK_M], float R[GEMV_BLK_M], float ALPHA, float BETA)
{
#pragma HLS inline
#pragma HLS array_partition variable=A block factor=16 dim=2 
#pragma HLS array_partition variable=X block factor=16 dim=1 

  int i, j;
  float sum, res, tmp;
#pragma HLS resource variable=tmp core=FAddSub_nodsp
  GEMVRow: for (i = 0; i < GEMV_BLK_M; i ++) {
    #pragma HLS pipeline II=1
    sum = Y[i]; // initial value
    GEMVCol: for (j = 0; j < GEMV_BLK_N; j ++) {
      res = A[i][j] * X[j];
      tmp = sum + res;
      sum = tmp;
    }
    R[i] = sum;
  }
}

void gemv_accel(float A[GEMV_BLK_M*GEMV_BLK_N], float X[GEMV_BLK_N], float Y[GEMV_BLK_M], 
    float R[GEMV_BLK_M], float ALPHA, float BETA)
{
  // store A on board BRAM
  int i, j;
  float A_buf[GEMV_BLK_M][GEMV_BLK_N];
  float X_buf[GEMV_BLK_N];
  float Y_buf[GEMV_BLK_M];
  RowCopy: for (i = 0; i < GEMV_BLK_M; i ++) {
    ColCopy: for (j = 0; j < GEMV_BLK_N; j ++) {
    #pragma HLS PIPELINE II=1 
      A_buf[i][j] = ALPHA * A[i*GEMV_BLK_N+j]; 
      if (i == 0) X_buf[j] = X[j];
      if (j == 0) Y_buf[i] = Y[i];
    }
  }

  gemv_accel_kernel(A_buf,X_buf,Y_buf,R,ALPHA,BETA);
}

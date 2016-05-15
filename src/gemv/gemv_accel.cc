
#include "gemv_accel.hh"

float gemv_accel_kernel(float A[GEMV_BLK_N], float x[GEMV_BLK_N], float y, float ALPHA, float BETA)
{
#pragma HLS INLINE
#pragma HLS array_partition variable=A block factor=16 dim=1 
#pragma HLS array_partition variable=x block factor=16 dim=1 
  int i;
  float sum = 0.0;
  for (i = 0; i < GEMV_BLK_N; i ++) {
  #pragma HLS UNROLL factor=4
    float res = ALPHA * A[i] * x[i];
    sum += res;
  }
  return sum + y;
}

float gemv_accel(float A[GEMV_BLK_N], float x[GEMV_BLK_N], float y, float ALPHA, float BETA)
{
  // store A on board BRAM
  int i;
  float A_buf[GEMV_BLK_N];
  float x_buf[GEMV_BLK_N];
  for (i = 0; i < GEMV_BLK_N; i ++) {
  #pragma HLS PIPELINE II=1 
    A_buf[i] = A[i]; 
    x_buf[i] = x[i];
  }

  return gemv_accel_kernel(A_buf,x_buf,y,ALPHA,BETA);
}

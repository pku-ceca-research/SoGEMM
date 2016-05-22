#include <stdlib.h>
#include <string.h>
#include "gemm_consts.h"
#include "gemm_block_unit.h"
#include "gemm_accel_hls.h"

void gemm_accel(
    float A[BLK_M*BLK_K], 
    float B[BLK_K*BLK_N], 
    float C[BLK_M*BLK_N], 
    float R[BLK_M*BLK_N],
    float ALPHA, 
    float BETA)
{
#ifdef GEMM_HLS
  gemm_accel_full(A,B,C,ALPHA,R);
#else
  
#ifdef GEMM_FULL_MODE
  float T[BLK_M*BLK_N];
  #pragma HLS dataflow
#else
  #ifdef SDS
  #include "sds_lib.h"
  #define malloc(x) (sds_malloc((x)))
  #define free(x) (sds_free((x)))
  #endif
  float *T = (float*) malloc(sizeof(float)*BLK_SIZE_MN);
#endif /* GEMM_FULL_MODE */

  gemm_block_units_mmult(A, B, ALPHA, T);
  gemm_block_units_mplus(T, C, R);
#ifndef GEMM_FULL_MODE
  free(T);
#endif /* GEMM_FULL_MODE */

#endif /* GEMM_HLS */
}

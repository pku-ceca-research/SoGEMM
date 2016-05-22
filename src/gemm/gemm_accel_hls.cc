/**
 * file: src/gemm/gemm_accel_hls.c
 * VectorType and ScalarType are defined in "gemm_consts.h",
 * controlled by GEMM_HALF_FLOAT macro
 */
#include "gemm_consts.h"
#include "gemm_accel_hls.h"

#ifdef GEMM_HALF_FLOAT
/**
 * GEMM_HALF_FLOAT switches the input array data type. 
 * It also controls the behaviour of this whole gemm_sds function.
 * If it's not under the VHLS environment, nothing will happen.
 */
#ifdef __SDSVHLS__
#include "hls_half.h"
typedef half VectorType;
// #else
/* Under other environment, vector type will remain float */
// #error "Must be compiled in VHLS mode."
#endif /* __SDS_VHLS__ */
#else 
typedef float VectorType;
#endif /* GEMM_HALF_FLOAT */
typedef float ScalarType;

#if (defined GEMM_HALF_FLOAT && !defined __SDSVHLS__)
void gemm_accel_full(
    float A[BLK_M*BLK_K], 
    float B[BLK_N*BLK_K], 
    float C[BLK_M*BLK_N], 
    float ALPHA, 
    float R[BLK_M*BLK_N])
{}
#else

void gemm_accel_kernel(
    VectorType A[BLK_M][BLK_K], 
    VectorType B[BLK_K][BLK_N], 
    VectorType C[BLK_M][BLK_N], 
    ScalarType ALPHA, 
    float R[BLK_M*BLK_N])
{
#pragma HLS INLINE
#if GEMM_SCALE == 2
#pragma HLS array_partition variable=A block factor=12 dim=2
#pragma HLS array_partition variable=B block factor=12 dim=1
#elif GEMM_SCALE == 3
#pragma HLS array_partition variable=A block factor=24 dim=2
#pragma HLS array_partition variable=B block factor=24 dim=1
#elif GEMM_SCALE == 4
#pragma HLS array_partition variable=A block factor=28 dim=2
#pragma HLS array_partition variable=B block factor=28 dim=1
#elif GEMM_SCALE == 5
#pragma HLS array_partition variable=A block factor=48 dim=2
#pragma HLS array_partition variable=B block factor=48 dim=1
#elif GEMM_SCALE == 6
#pragma HLS array_partition variable=A block factor=32 dim=2
#pragma HLS array_partition variable=B block factor=32 dim=1
#elif GEMM_SCALE == 7
#pragma HLS array_partition variable=A block factor=30 dim=2
#pragma HLS array_partition variable=B block factor=30 dim=1
#else
#pragma HLS array_partition variable=A block factor=16 dim=2
#pragma HLS array_partition variable=B block factor=16 dim=1
#endif
  
  int i, j, k;
  VectorType tmp, sum, res;
#ifdef GEMM_NO_ADD_DSP
#ifdef GEMM_HALF_FLOAT
// #pragma HLS resource variable=tmp core=AddSub_nodsp
#elif GEMM_RESOURCE_CONSTRAINT
#pragma HLS allocation instances=fmul limit=64 operation
#pragma HLS allocation instances=fadd limit=64 operation
#else
#pragma HLS resource variable=tmp core=FAddSub_nodsp
#endif /* GEMM_HALF_FLOAT */
#endif /* GEMM_NO_ADD_DSP */
  Row: for (i = 0; i < BLK_M; i ++) {
    Col: for (j = 0; j < BLK_N; j ++) {
    #pragma HLS pipeline II=1
      sum = C[i][j];
    #ifndef GEMM_RESOURCE_PARTITION
      for (k = 0; k < BLK_K; k ++) {
        res = A[i][k] * B[k][j];
        tmp = sum + res;
        sum = tmp;
      }
    #else
    /* In this branch, the resource will be manually allocated */
      VectorType dsp;
    #ifndef GEMM_DSP_UPPER
    #define GEMM_DSP_UPPER 16
    #endif
    #ifdef GEMM_HALF_FLOAT
    // #pragma HLS resource variable=dsp core=AddSub_fulldsp
    #else
    #pragma HLS resource variable=dsp core=FAddSub_fulldsp
    #endif /* GEMM_HALF_FLOAT */
      for (k = 0; k < GEMM_DSP_UPPER; k ++) {
        res = A[i][k] * B[k][j];
        dsp = sum + res;
        sum = dsp;
      }
      for (k = GEMM_DSP_UPPER; k < BLK_K; k ++) {
        res = A[i][k] * B[k][j];
        tmp = sum + res;
        sum = tmp;
      }
    #endif
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
  int lda = BLK_N, ldb = BLK_N, ldc = BLK_N;
  VectorType A_buf[BLK_M][BLK_K], B_buf[BLK_K][BLK_N], C_buf[BLK_M][BLK_N];

  RowCopy: for (i = 0; i < BLK_M; i ++) 
    ColCopy: for (j = 0; j < BLK_N; j ++) {
    #pragma HLS pipeline II=1
    /**
     * if the matrix is irregular, then A's 2nd dim will not equal to BLK_N. 
     * however, as BLK_M == BLK_N, then ldb will remain the same
     */
    #ifdef GEMM_IRREGULAR
      lda = BLK_K;
      if (j < BLK_K) 
    #endif
      /* GEMM_WITH_ALPHA is compatible with outer configurations */
    #ifdef GEMM_WITH_ALPHA
      A_buf[i][j] = ALPHA * A[i*lda+j];
    #else 
      A_buf[i][j] = A[i*lda+j];
    #endif
    #ifdef GEMM_IRREGULAR
      if (i < BLK_K) 
    #endif
      B_buf[i][j] = (VectorType) B[i*ldb+j];
      C_buf[i][j] = (VectorType) C[i*ldc+j];
    }

  gemm_accel_kernel(A_buf,B_buf,C_buf,ALPHA,R);
}
#endif

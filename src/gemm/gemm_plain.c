#include <string.h>
#include <assert.h>
#include <time.h>

#include "gemm_consts.h"
#include "gemm_utils.h"
#include "gemm_plain.h"
#include "gemm_accel.h"

/* global buffers */
static float *A_buf,*B_buf,*C_buf,*R_buf;
/* global timers */
static double total_init_A_buf_time;
static double total_init_B_buf_time;

// helpers
void gemm_plain_alloc_buffers();
void gemm_plain_clean_buffers();
void gemm_plain_free_buffers();
// calculation
void gemm_plain_kernel(int TA, int TB, int M, int N, int K, float ALPHA, 
    float *A, int lda,
    float *B, int ldb,
    float BETA,
    float *C, int ldc);

void gemm_plain(int TA, int TB, int M, int N, int K, float ALPHA, 
    float *A, int lda,
    float *B, int ldb,
    float BETA,
    float *C, int ldc)
{
  // TODO: Add more parameters to choose different shape of buffers 
  gemm_plain_alloc_buffers();
  gemm_plain_clean_buffers();
  // main calculation
  gemm_plain_kernel(TA,TB,M,N,K,ALPHA,A,lda,B,ldb,BETA,C,ldc);
  gemm_plain_free_buffers(); 
}

void gemm_plain_init_clock() 
{
  total_init_A_buf_time = 0.0;
  total_init_B_buf_time = 0.0;
}

double gemm_plain_end_clock() 
{
  printf("Timer results:\n");
  printf("InitA: %lfs\n", total_init_A_buf_time);
  printf("InitB: %lfs\n", total_init_B_buf_time);
  return total_init_A_buf_time + total_init_B_buf_time;
}

#ifdef __SDSCC__
#include "sds_lib.h"
#define malloc(x) (sds_alloc(x))
#define free(x) (sds_free(x))
#endif

void gemm_plain_alloc_buffers() 
{
  A_buf = (float*)malloc(sizeof(float)*BLK_SIZE_MK);
  B_buf = (float*)malloc(sizeof(float)*BLK_SIZE_KN);
  C_buf = (float*)malloc(sizeof(float)*BLK_SIZE_MN);
  R_buf = (float*)malloc(sizeof(float)*BLK_SIZE_MN);

  if (!A_buf || !B_buf || !C_buf || !R_buf) {
    fprintf(stderr, "Error buffer allocation\n");
    exit(1);
  }
}

void gemm_plain_clean_buffers() 
{
  memset(A_buf, 0, sizeof(float)*BLK_SIZE_MK);
  memset(B_buf, 0, sizeof(float)*BLK_SIZE_KN);
  memset(C_buf, 0, sizeof(float)*BLK_SIZE_MN);
  memset(R_buf, 0, sizeof(float)*BLK_SIZE_MN);
}

void gemm_plain_free_buffers()
{
  free(A_buf);
  free(B_buf);
  free(C_buf);
  free(R_buf);
}

inline void copy_blk_to_A_buf(int T, int M, int K, int bi, int bk, float *A, int lda, float ALPHA) 
{
  int i, k, _i, _k;
  if (T) {
    for (i = 0; i < BLK_M && i + bi < M; i ++)
      for (k = 0; k < BLK_K && k + bk < K; k ++) {
        _i = i+bi, _k = k+bk;
        #ifdef GEMM_WITH_ALPHA
        A_buf[i*BLK_K+k] = A[_k*lda+_i];
        #else
        A_buf[i*BLK_K+k] = ALPHA * A[_k*lda+_i];
        #endif
      }
  } else {
    for (i = 0; i < BLK_M && i + bi < M; i ++)
      for (k = 0; k < BLK_K && k + bk < K; k ++) {
        _i = i+bi, _k = k+bk;
        #ifdef GEMM_WITH_ALPHA
        A_buf[i*BLK_K+k] = A[_i*lda+_k];
        #else
        A_buf[i*BLK_K+k] = ALPHA * A[_i*lda+_k];
        #endif
      }
  }
}

inline void copy_blk_to_B_buf(int T, int K, int N, int bj, int bk, float *B, int ldb)
{
  int j, k, _j, _k;
  if (T) {
    for (k = 0; k < BLK_K && k + bk < K; k ++)
      for (j = 0; j < BLK_N && j + bj < N; j ++) {
        _k = k+bk, _j = j+bj;
        B_buf[k*BLK_N+j] = B[_j*ldb+_k];
      }
  } else {
    for (k = 0; k < BLK_K && k + bk < K; k ++)
      for (j = 0; j < BLK_N && j + bj < N; j ++) {
        _k = k+bk, _j = j+bj;
        B_buf[k*BLK_N+j] = B[_k*ldb+_j];
      }
  }
}

void gemm_plain_kernel(int TA, int TB, int M, int N, int K, float ALPHA, 
    float *A, int lda,
    float *B, int ldb,
    float BETA,
    float *C, int ldc)
{
  /* block indices */
  int bi, bj, bk;
  /* indices */
  int i, j;
  /* original indices */
  int _i, _j;

  // clock_t start, end;
  for (bi = 0; bi < M; bi += BLK_M) {
    for (bj = 0; bj < N; bj += BLK_N) {
      // initialize C_buf
      for (i = 0; i < BLK_M; i ++)
        for (j = 0; j < BLK_N; j ++) {
          _i = bi+i, _j = bj+j; 
          C_buf[i*BLK_N+j] = 
            (_j < N && _i < M) ? BETA * C[_i*ldc+_j] : 0.0;
        }
                   
      for (bk = 0; bk < K; bk += BLK_K) { 
        copy_blk_to_A_buf(TA, M, K, bi, bk, A, lda, ALPHA);
        copy_blk_to_B_buf(TB, K, N, bj, bk, B, ldb);
        // compute
        gemm_accel(A_buf,B_buf,C_buf,R_buf,ALPHA,BETA);
        // clean up
        memset(A_buf, 0, sizeof(float)*BLK_SIZE_MK);
        memset(B_buf, 0, sizeof(float)*BLK_SIZE_KN);
        // copy back
        memcpy(C_buf, R_buf, sizeof(float)*BLK_SIZE_MN);
      } 

      // write back C_buf
      for (i = 0; i < BLK_M; i ++)
        for (j = 0; j < BLK_N; j ++) {
          _i = bi+i, _j = bj+j; 
          if (_j < N && _i < M)
            C[_i*ldc+_j] = C_buf[i*BLK_N+j];
        }
    } 
  }
}

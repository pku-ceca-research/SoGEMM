
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <time.h>

#include "gemv_accel.hh"
#include "gemv_accel_call.hh"
#include "gemv_sds.hh"

#ifdef __SDSCC__
#include "sds_lib.h"
#define malloc(x) (sds_alloc(x))
#define free(x) (sds_free(x))
#endif

static float *A_buf, *X_buf, *Y_buf, *R_buf;

void gemv_sds_kernel(int T, int M, int N, float ALPHA, 
    const float *A, int lda,
    const float *X, int ldx,
    float BETA, 
    float *Y, int ldy);

void gemv_sds(int TA, int M, int N, float ALPHA, 
    const float *A, int lda,
    const float *X, int ldx,
    float BETA, 
    float *Y, int ldy)
{
  A_buf = (float *) malloc(sizeof(float)*GEMV_BLK_N*GEMV_BLK_M);
  X_buf = (float *) malloc(sizeof(float)*GEMV_BLK_N);
  Y_buf = (float *) malloc(sizeof(float)*GEMV_BLK_M);
  R_buf = (float *) malloc(sizeof(float)*GEMV_BLK_M);

  gemv_sds_kernel(TA, M, N, ALPHA, A, lda, X, ldx, BETA, Y, ldy);

  free(A_buf);
  free(X_buf);
  free(Y_buf);
  free(R_buf);
}

inline void copy_to_A_buf(int T, const float *A, int bi, int bj, int M, int N, int lda)
{
  int i, j, _i, _j;
  if (!T) {
    for (i = 0; i < GEMV_BLK_M; i ++) {
      for (j = 0; j < GEMV_BLK_N; j ++) {
        _i = bi + i, _j = bj + j;
        A_buf[i*GEMV_BLK_N+j] = 
          (_i<M && _j<N) ? A[_i*lda+_j] : 0.0;
      }
    }
  } else {
    for (i = 0; i < GEMV_BLK_M; i ++) {
      for (j = 0; j < GEMV_BLK_N; j ++) {
        _i = bi + i, _j = bj + j;
        A_buf[i*GEMV_BLK_N+j] =
          (_i<N && _j<M) ? A[_j*lda+_i] : 0.0;
      }
    }
  }
}

inline void copy_to_X_buf(const float *X, int bi, int N, int ldx) 
{
  int i, _i;
  for (i = 0; i < GEMV_BLK_N; i ++) {
    _i = bi + i;
    X_buf[i] = (_i<N) ? X[_i*ldx] : 0.0;
  }
}

// normal
void gemv_sds_kernel(int T, int M, int N, float ALPHA, 
    const float *A, int lda,
    const float *X, int ldx,
    float BETA, 
    float *Y, int ldy) 
{
  /* inner-block indices */
  int i, _i;
  /* block indices */
  int bi, bj;
  for (bi = 0; bi < M; bi += GEMV_BLK_M) {
    // initialize Y vector
    for (i = 0; i < GEMV_BLK_M; i ++)
      Y_buf[i] = (i+bi < M) ? (BETA * Y[(i+bi)*ldy]) : 0.0;
    // call GEMV
    for (bj = 0; bj < N; bj += GEMV_BLK_N) {
      copy_to_A_buf(T, A, bi, bj, M, N, lda);
      copy_to_X_buf(X, bj, N, ldx);
      // core
      gemv_accel_call(A_buf,X_buf,Y_buf,R_buf,ALPHA,BETA);
      memcpy(Y_buf, R_buf, sizeof(float)*GEMV_BLK_M);
    }
    // copy back to Y
    for (i = 0; i < GEMV_BLK_M; i ++) {
      _i = i + bi;
      if (_i < M)
        Y[_i*ldy] = Y_buf[i];
    }
  }
}

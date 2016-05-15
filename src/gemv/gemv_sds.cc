
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

static float *A_buf, *x_buf;

void gemv_sds_kernel(int T, int M, int N, float ALPHA, 
    const float *A, int lda,
    const float *X, int ldx,
    float BETA, 
    float *Y, int ldy);

void gemv_sds(int TA, int M, int N, float ALPHA, 
    const float *A, int lda,
    const float *x, int ldx,
    float BETA, 
    float *y, int ldy)
{
  A_buf = (float *) malloc(sizeof(float)*GEMV_BLK_N);
  x_buf = (float *) malloc(sizeof(float)*GEMV_BLK_N);

  gemv_sds_kernel(TA, M, N, ALPHA, A, lda, x, ldx, BETA, y, ldy);

  free(A_buf);
  free(x_buf);
}

// normal
void gemv_sds_kernel(int T, int M, int N, float ALPHA, 
    const float *A, int lda,
    const float *X, int ldx,
    float BETA, 
    float *Y, int ldy) 
{
  int i, j, _x;
  int x_buf_size;
  float y;
  
  for (i = 0; i < M; i ++) {
    y = Y[i*ldy] * BETA;

    for (j = 0; j < N; j += GEMV_BLK_N) {
      x_buf_size = (j+GEMV_BLK_N<N) ? GEMV_BLK_N : (N-j); 
      if (!T) {
        memcpy(A_buf, &A[j], sizeof(float)*x_buf_size);
      } else {
        for (_x = 0; _x < x_buf_size; _x ++)
          A_buf[_x*GEMV_BLK_N] = A[j*lda+(i+_x)];
      }
      if (ldx == 1) {
        memcpy(x_buf, &X[j], sizeof(float)*x_buf_size);
      } else {
        for (_x = 0; _x < GEMV_BLK_N && j+_x < N; _x ++)
          x_buf[_x] = X[(_x+j)*ldx];
      }
      if (x_buf_size < GEMV_BLK_N)
        for (_x = x_buf_size; _x < GEMV_BLK_N; _x ++) 
          x_buf[_x] = A_buf[_x] = 0.0;
      
      y = gemv_accel_call(A_buf, x_buf, y, ALPHA, BETA);
    }

    Y[i*ldy] = y;
  }
}

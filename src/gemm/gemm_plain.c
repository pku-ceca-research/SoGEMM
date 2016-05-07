#include <string.h>
#include <assert.h>
#include <time.h>

#include "gemm_consts.h"
#include "gemm_utils.h"
#include "gemm_plain.h"
#include "gemm_block_unit.h"

/* global buffers */
float *A_buf,*B_buf,*C_buf,*R_buf,*T_buf;
/* global shape */
const int blk_m = BLK_M;
const int blk_n = BLK_N;
const int blk_k = BLK_K;
const int num_depth = NUM_DEPTH;
const int num_pipes = NUM_PIPES;
const int blk_size_mk  = BLK_M * BLK_K;
const int blk_size_kn  = BLK_K * BLK_N;
const int blk_size_mn  = BLK_M * BLK_N;
const int pipe_size_mk = BLK_M * BLK_K * NUM_PIPES;
const int pipe_size_kn = BLK_K * BLK_N * NUM_PIPES;
const int pipe_size_mn = BLK_M * BLK_N * NUM_PIPES;
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

void gemm_plain_alloc_buffers() 
{
  A_buf = (float*)MALLOC(sizeof(float)*num_depth*pipe_size_mk);
  B_buf = (float*)MALLOC(sizeof(float)*num_depth*pipe_size_kn);
  T_buf = (float*)MALLOC(sizeof(float)*num_depth*pipe_size_mn);
  C_buf = (float*)MALLOC(sizeof(float)*num_depth*blk_size_mn);
  R_buf = (float*)MALLOC(sizeof(float)*num_depth*blk_size_mn);

  if (!A_buf || !B_buf || !C_buf || !T_buf || !R_buf) {
    fprintf(stderr, "Error buffer allocation\n");
    exit(1);
  }
}

void gemm_plain_clean_buffers() 
{
  memset(A_buf, 0, sizeof(float)*num_depth*pipe_size_mk);
  memset(B_buf, 0, sizeof(float)*num_depth*pipe_size_kn);
  memset(T_buf, 0, sizeof(float)*num_depth*pipe_size_mn);
  memset(C_buf, 0, sizeof(float)*num_depth*blk_size_mn);
  memset(R_buf, 0, sizeof(float)*num_depth*blk_size_mn);
}

void gemm_plain_free_buffers()
{
  FREE(A_buf);
  FREE(B_buf);
  FREE(T_buf);
  FREE(C_buf);
  FREE(R_buf);
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
  int i, j, k, d, p;
  /* original indices */
  int _i, _j, _k;

  clock_t start, end;
  for (bi = 0; bi < M; bi += blk_m) {
    for (bj = 0; bj < N; bj += blk_n*num_depth) {
      // initialize C_buf
      for (d = 0; d < num_depth; d ++)
        for (i = 0; i < blk_m; i ++)
          for (j = 0; j < blk_n; j ++) {
            _i = bi+i, _j = bj+j+blk_n*d; 
            C_buf[d*blk_size_mn+i*blk_n+j] = 
              (_j < N && _i < M) ? BETA * C[_i*ldc+_j] : 0.0;
          }
                   
      for (bk = 0; bk < K; bk += blk_k*num_pipes) { 
        // initialize a
        start = clock();
        for (d = 0; d < num_depth; d ++)
          for (p = 0; p < num_pipes; p ++)
            for (i = 0; i < blk_m; i ++)
              for (k = 0; k < blk_k; k ++) {
                _i = i+bi, _k = p*blk_k+k+bk;
                A_buf[d*pipe_size_mk+p*blk_size_mk+i*blk_k+k] = 
                  (_k < K && _i < M) ? 
                  (TA ? A[_k*lda+_i]: A[_i*lda+_k]) :
                  0.0;
              }
        end = clock();
        total_init_A_buf_time += (double)(end-start)/CLOCKS_PER_SEC;

        // initialize B
        start = clock();
        for (d = 0; d < num_depth; d ++)
          for (p = 0; p < num_pipes; p ++)
            for (k = 0; k < blk_k; k ++)
              for (j = 0; j < blk_n; j ++) {
                _k = k+p*blk_k+bk, _j = j+d*blk_n+bj;
                B_buf[d*pipe_size_kn+p*blk_size_kn+k*blk_n+j] =
                  (_k < K && _j < N) ?
                  (TB ? B[_j*ldb+_k] : B[_k*ldb+_j]) :
                  0.0;
              }
        end = clock();
        total_init_B_buf_time += (double)(end-start)/CLOCKS_PER_SEC;

        // compute
        gemm_block_units_mmult(A_buf,B_buf,ALPHA,T_buf);
        gemm_block_units_mplus(T_buf,C_buf,R_buf);

        // copy back
        memcpy(C_buf, R_buf, sizeof(float)*num_depth*blk_size_mn);
      } 

      // write back C_buf
      for (d = 0; d < num_depth; d ++)
        for (i = 0; i < blk_m; i ++)
          for (j = 0; j < blk_n; j ++) {
            _i = bi+i, _j = bj+j+blk_n*d; 
            if (_j < N && _i < M)
              C[_i*ldc+_j] = C_buf[d*blk_size_mn+i*blk_n+j];
          }
    } 
  }
}

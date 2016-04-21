
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <getopt.h>
#include <time.h>

#include "gemm_consts.h"
#include "gemm_types.h"
#include "gemm_utils.h"
#include "gemm_block.h"

int test_gemm_block_main(int M, int N, int K, int verbose, int iter);

void gemm_block_main_golden(BlockedMatrix *A_blk, BlockedMatrix *B_blk, BlockedMatrix *C_blk,
    float *C_golden, float ALPHA, float BETA);

int main(int argc, char *argv[]) {
  
  PRINT_TITLE("TEST GEMM BLOCK COMPUTATIONS");
  
  int c;
  int iter = 1;
  int M=BLK_M*4, N=BLK_N*4, K=BLK_K*4;
  int verbose=0;

  while ((c = getopt(argc, argv, "M:N:K:vi:")) != -1) {
    switch (c) {
      case 'M': M = atoi(optarg); break;
      case 'N': N = atoi(optarg); break;
      case 'K': K = atoi(optarg); break;
      case 'v': verbose = 1; break;
      case 'i': iter = atoi(optarg); break;
      default:
        fprintf(stderr, "Can't recognise option: %c\n", (char) c);
        exit(1);
    }
  }

  printf("Iter: %d\n", iter);
  test_gemm_block_main(M, N, K, verbose, iter);
  return 0;
}

void gemm_block_main_golden(BlockedMatrix *A_blk, BlockedMatrix *B_blk, BlockedMatrix *C_blk,
    float *C_golden, float ALPHA, float BETA)
{
  int i, j, k, x, y, t;
  int base;
  int num_blk_M = A_blk->H/BLK_M;
  int num_blk_N = B_blk->W/BLK_N;
  int num_blk_K = A_blk->W/BLK_K;

  for (i = 0; i < num_blk_M; i ++) {
    for (j = 0; j < num_blk_N; j ++) {
      base = (i*num_blk_N+j)*BLK_M*BLK_N;
      // initialize C_golden
      for (x = 0; x < BLK_M; x ++)
        for (y = 0; y < BLK_N; y ++)
          C_golden[base+x*BLK_N+y] = BETA * C_blk->mat[base+x*BLK_N+y];

      for (k = 0; k < num_blk_K; k ++) {
        int A_base = (i*num_blk_K+k)*BLK_M*BLK_K;
        int B_base = (k*num_blk_N+j)*BLK_K*BLK_N;

        for (x = 0; x < BLK_M; x ++) 
          for (y = 0; y < BLK_N; y ++) 
            for (t = 0; t < BLK_K; t ++) {
              float a = A_blk->mat[A_base+x*BLK_K+t];
              float b = B_blk->mat[B_base+t*BLK_N+y];
              C_golden[base+x*BLK_N+y] += a * b * ALPHA;
            } 
      }
    } 
  }
}

int test_gemm_block_main(int M, int N, int K, int verbose, int iter) {
  int i, I;
  srand(time(NULL));

  PRINT_HIGHLIGHT("# TEST gemm_block_nn");
 
  BlockedMatrix *A_blk = random_blocked_matrix(0, M, K, K, BLK_M, BLK_K);
  BlockedMatrix *B_blk = random_blocked_matrix(0, K, N, N, BLK_K, BLK_N);
  BlockedMatrix *C_blk = random_blocked_matrix(0, M, N, N, BLK_M, BLK_N);

  printf("A_blk =\n");
  show_blocked_matrix(A_blk, verbose);
  printf("B_blk =\n");
  show_blocked_matrix(B_blk, verbose);
  printf("C_blk =\n");
  show_blocked_matrix(C_blk, verbose);

  float ALPHA = (float)rand()/RAND_MAX;
  float BETA  = (float)rand()/RAND_MAX;
  printf("ALPHA: %lf\n", ALPHA);
  printf("BETA:  %lf\n", BETA);

  int num_blk_M = A_blk->H/BLK_M;
  int num_blk_N = B_blk->W/BLK_N;
  int num_blk_K = A_blk->W/BLK_K;

  printf("NUM_BLK_M: %d\n", num_blk_M);
  printf("NUM_BLK_N: %d\n", num_blk_N);
  printf("NUM_BLK_K: %d\n", num_blk_K);

  float *C_golden = (float*) malloc(sizeof(float)*C_blk->H*C_blk->W);

  clock_t start, end;
  start = clock();
  for (I = 0; I < iter; I ++) {
    gemm_block_main_golden(A_blk,B_blk,C_blk,C_golden,ALPHA,BETA);
  }
  end = clock();
  printf("FINISHED software: %lfs\n", (double)(end-start)/CLOCKS_PER_SEC/iter);

  // Calculate final result
  start = clock();
  for (I = 0; I < iter; I ++) {
    gemm_block_main(ALPHA, BETA, A_blk, B_blk, C_blk);
  }
  end = clock();
  printf("FINISHED hardware: %lfs\n", (double)(end-start)/CLOCKS_PER_SEC/iter);

  if (iter == 1)
    for (i = 0; i < C_blk->H*C_blk->W; i++) {
      C_blk->mat[i] /= 10;
      C_golden[i] /= 10;
      if (i < 5)
        printf("[%3d] result %lf golden %lf\n", i, C_blk->mat[i], C_golden[i]);
      
      if (fabsf(C_blk->mat[i]-C_golden[i]) >= 1e-5) {
        fprintf(stderr, "ERROR[%3d] result %lf golden %lf\n", 
            i, C_blk->mat[i], C_golden[i]);
        exit(1);
      }
    }
  else
    printf("Skipped test\n");


  PRINT_PASSED("PASSED!");

  free_blocked_matrix(A_blk);
  free_blocked_matrix(B_blk);
  free_blocked_matrix(C_blk);
  free(C_golden);

  return 1;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <getopt.h>

#include "gemm_consts.h"
#include "gemm_utils.h"
#include "gemm_cpu.h"
#include "gemm_sds.h"

int main(int argc, char *argv[]) {
  int c;
  int verbose = 0;
  int iter = 1;
  int TA = 0, TB = 0;
  int M=2, N=2, K=2;
  while ((c = getopt(argc, argv, "vm:n:k:i:ab")) != -1) 
    switch (c) {
      case 'v': 
        verbose = 1;
        break;
      case 'i':
        iter = atoi(optarg);
        break;
      case 'a': TA = 1; break;
      case 'b': TB = 1; break;
      case 'm': M = atoi(optarg); break;
      case 'n': N = atoi(optarg); break;
      case 'k': K = atoi(optarg); break;
      default:
        fprintf(stderr, "Unknown option: %d\n", c);
        exit(1);
    }

  PRINT_TITLE("TEST GEMM SDS version");
  printf("Iteration: %d\n", iter);
  printf("TA:     %d\n", TA);
  printf("TB:     %d\n", TB);
  printf("M:      %d\n", M);
  printf("N:      %d\n", N);
  printf("K:      %d\n", K);
  printf("BLK_M:  %d\n", BLK_M);
  printf("BLK_N:  %d\n", BLK_N);
  printf("BLK_K:  %d\n", BLK_K);
  printf("PIPES:  %d\n", NUM_PIPES);
  printf("DEPTH:  %d\n", NUM_DEPTH);

  float *A = (float *) malloc(sizeof(float)*M*K);
  float *B = (float *) malloc(sizeof(float)*K*N);
  float *C = (float *) malloc(sizeof(float)*M*N);
  float *G = (float *) malloc(sizeof(float)*M*N);

  int lda = (TA) ? M : K;
  int ldb = (TB) ? K : N;
  int ldc = N;

  int i;
  for (i = 0; i < M*K; i++) A[i] = (float) 1;
  for (i = 0; i < K*N; i++) B[i] = (float) 1;
  for (i = 0; i < M*N; i++) C[i] = G[i] = (float) 1;

  if (verbose) {
    print_matrix(A, M, K);
    print_matrix(B, K, N);
    print_matrix(C, M, N);
  }

  float ALPHA = 1.0;
  float BETA  = 1.0;

  clock_t start, end;
  start = clock();
  for (i = 0; i < iter; i++)
    gemm_sds(TA,TB,M,N,K,ALPHA,A,lda,B,ldb,BETA,C,ldc);
  end = clock();
  printf("FINISHED hardware: %lfs\n", (double)(end-start)/CLOCKS_PER_SEC/iter);

  start = clock();
  for (i = 0; i < iter; i++)
    gemm_cpu(TA,TB,M,N,K,ALPHA,A,lda,B,ldb,BETA,G,ldc);
  end = clock();
  printf("FINISHED software: %lfs\n", (double)(end-start)/CLOCKS_PER_SEC/iter);

  for (i = 0; i < M*N; i++)
    if (fabsf(C[i]-G[i]) >= 1e-6) {
      fprintf(stderr, "[%3d] C[i]=%lf G[i]=%lf\n", i, C[i], G[i]);
      exit(1);
    }

  PRINT_PASSED("PASSED!");

  free(A);
  free(B);
  free(C);
  free(G);

  return 0;
}

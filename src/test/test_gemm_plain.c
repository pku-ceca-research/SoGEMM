
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <getopt.h>

#include "gemm_consts.h"
#include "gemm_utils.h"
#include "gemm_plain.h"
#include "gemm_block.h"
#include "gemm_cpu.h"

int c;
int M=BLK_M, N=BLK_N, K=BLK_K, TA=0, TB=0;
char *file_name;
int iter = 1;
int verbose = 0;
float ALPHA = 1.0, BETA = 1.0;

// matrix
float *A, *B, *C, *C_golden;
int lda, ldb, ldc;

// helpers
void show_options();
void init_matrix();
void free_matrix();
int compare();

int main(int argc, char *argv[])
{
  while ((c = getopt(argc, argv, "m:n:k:f:vi:ab")) != -1)
    switch (c) {
      case 'm':
        M = atoi(optarg); break;
      case 'n':
        N = atoi(optarg); break;
      case 'k':
        K = atoi(optarg); break;
      case 'a': TA = 1; break;
      case 'b': TB = 1; break;
      case 'i': iter = atoi(optarg); break;
      case 'v': verbose = 1; break;
      case 'f': file_name = (char *) optarg; break;
      default:
        fprintf(stderr, "Unrecognised option %c\n", c);
        exit(1);
    }

  show_options();
  init_matrix();
  
  int I;
  clock_t start, end;

  PRINT_HIGHLIGHT("TEST origin");
  start = clock();
  gemm_plain_init_clock();
  for (I = 0; I < iter; I ++)
    gemm_plain(TA,TB,M,N,K,ALPHA,A,lda,B,ldb,BETA,C,ldc);
  end = clock();
  double total_time = (double)(end-start)/CLOCKS_PER_SEC;
  double extra_time = gemm_plain_end_clock();
  printf("Finished **origin**: %lfs extra=%lfs compute=%lfs\n", 
      total_time/iter,
      extra_time/iter,
      (total_time-extra_time)/iter);


  PRINT_HIGHLIGHT("TEST golden");
  start = clock();
  for (I = 0; I < iter; I ++)
    gemm_cpu(TA,TB,M,N,K,ALPHA,A,lda,B,ldb,BETA,C_golden,ldc);
  end = clock();
  printf("Finished **golden**: %lfs\n", (double)(end-start)/CLOCKS_PER_SEC/iter);

  if (compare()) {
    PRINT_PASSED("PASSED");
  } else {
    PRINT_FAILED("FAILED");
  }

  PRINT_HIGHLIGHT("TEST blocked (additional)");
  start = clock();
  for (I = 0; I < iter; I ++)
    gemm_block(TA,TB,M,N,K,ALPHA,A,lda,B,ldb,BETA,C,ldc);
  end = clock();
  printf("Finished **blocked**: %lfs\n", (double)(end-start)/CLOCKS_PER_SEC/iter);

  free_matrix();

  return 0;
}

void show_options() 
{
  PRINT_TITLE("TEST gemm_plain");
  printf("Iterations: %d\n", iter);
  printf("A: %3d X %3d T=%d\n", M, K, TA);
  printf("B: %3d X %3d T=%d\n", K, N, TB);
  printf("C: %3d X %3d\n", M, N);
  printf("file_name: %s\n", ((file_name) ? file_name : "NULL"));
}

void init_matrix() 
{
  A = (float *) MALLOC(sizeof(float)*M*K);
  B = (float *) MALLOC(sizeof(float)*K*N);
  C = (float *) MALLOC(sizeof(float)*M*N);
  C_golden = (float *) MALLOC(sizeof(float)*M*N);

  int i;
  for (i = 0; i < M*K; i ++) A[i] = 1.0;
  for (i = 0; i < K*N; i ++) B[i] = 1.0;
  for (i = 0; i < M*N; i ++) C[i] = C_golden[i] = 1.0;

  lda = TA ? M : K;
  ldb = TB ? K : N;
  ldc = N;
}

void free_matrix() 
{
  free(A);
  free(B);
  free(C);
  free(C_golden);
}

int compare() 
{
  int i;
  for (i = 0; i < M*N; i ++)
    if (fabsf(C[i]-C_golden[i]) > 1e-6) {
      fprintf(stderr, "Error[%4d] origin=%lf golden=%lf\n",
          i, C[i], C_golden[i]);
      return 0;
    }
  return 1;
}

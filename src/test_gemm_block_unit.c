
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <getopt.h>

#include "gemm_utils.h"
#include "gemm_consts.h"
#include "gemm_block_unit.h"

int main(int argc, char *argv[]) {
  srand(time(NULL));

  int c;
  int iter = 1;
  while ((c = getopt(argc, argv, "i:")) != -1)
    switch (c) {
      case 'i':
        iter = atoi(optarg);
        break;
      default:
        fprintf(stderr, "Unrecognized option: %c\n", c);
        exit(1);
    }

  float *A = (float *) MALLOC(sizeof(float)*NUM_DEPTH*PIPE_SIZE_MK);
  float *B = (float *) MALLOC(sizeof(float)*NUM_DEPTH*PIPE_SIZE_KN);
  float *T = (float *) MALLOC(sizeof(float)*NUM_DEPTH*PIPE_SIZE_MN);
  float *C = (float *) MALLOC(sizeof(float)*NUM_DEPTH*BLK_SIZE_MN);
  float *R = (float *) MALLOC(sizeof(float)*NUM_DEPTH*BLK_SIZE_MN);
  float *R_golden = (float *) MALLOC(sizeof(float)*NUM_DEPTH*BLK_SIZE_MN);
  if (!A || !B || !T || !C || !R || !R_golden) {
    fprintf(stderr, "Can't initialize memories\n");
    exit(1);
  }

  memset(A, 0, sizeof(float)*NUM_DEPTH*PIPE_SIZE_MK);
  memset(B, 0, sizeof(float)*NUM_DEPTH*PIPE_SIZE_KN);
  memset(T, 0, sizeof(float)*NUM_DEPTH*PIPE_SIZE_MN);
  memset(C, 0, sizeof(float)*NUM_DEPTH*BLK_SIZE_MN);
  memset(R, 0, sizeof(float)*NUM_DEPTH*BLK_SIZE_MN);
  memset(R_golden, 0, sizeof(float)*NUM_DEPTH*BLK_SIZE_MN);

  int d, p, i, j, k;
  int t;
  float ALPHA = 0.1;
  float BETA  = 1.0;

  PRINT_TITLE("TEST gemm_block_unit");
  printf("NUM_ITER:\t %d\n", iter);
  printf("NUM_PIPES:\t %d\n", NUM_PIPES);
  printf("NUM_DEPTH:\t %d\n", NUM_DEPTH);
  printf("BLK_M:\t\t %d\n", BLK_M);
  printf("BLK_N:\t\t %d\n", BLK_N);
  printf("BLK_K:\t\t %d\n", BLK_K);
  printf("ALPHA:\t\t %f\n", ALPHA);
  printf("BETA:\t\t %f\n", BETA);

  for (i = 0; i < NUM_DEPTH*PIPE_SIZE_MK; i ++) A[i] = (float)rand()/RAND_MAX;
  for (i = 0; i < NUM_DEPTH*PIPE_SIZE_KN; i ++) B[i] = (float)rand()/RAND_MAX;
  for (i = 0; i < NUM_DEPTH*PIPE_SIZE_MN; i ++) T[i] = 0.0;
  for (i = 0; i < NUM_DEPTH*BLK_SIZE_MN;  i ++) {
    R[i] = 0.0;
    C[i] = (float)rand()/RAND_MAX;
  }
  printf("Matrix initialised\n");
  
  clock_t start, end;
  start = clock();
  for (t = 0; t < iter; t ++) { 
    gemm_block_units_mmult(A,B,T);
    gemm_block_units_mplus(T,C,R);
  }
  end = clock();
  printf("FINISHED origin: %lfs\n", (double)(end-start)/CLOCKS_PER_SEC);

  start = clock();
  for (t = 0; t < iter; t++) {
    for (i = 0; i < NUM_DEPTH*BLK_SIZE_MN; i ++) 
      R_golden[i] = C[i]*BETA;

    for (d = 0; d < NUM_DEPTH; d ++)
      for (p = 0; p < NUM_PIPES; p ++)
        for (i = 0; i < BLK_M; i ++)
          for (j = 0; j < BLK_N; j ++)
            for (k = 0; k < BLK_K; k ++)
              R_golden[d*BLK_SIZE_MN+i*BLK_N+j] += 
                ALPHA * 
                A[d*PIPE_SIZE_MK+p*BLK_SIZE_MK+i*BLK_K+k] * 
                B[d*PIPE_SIZE_KN+p*BLK_SIZE_KN+k*BLK_N+j];
  }
  end = clock();
    
  printf("FINISHED golden: %lfs\n", (double)(end-start)/CLOCKS_PER_SEC);

  int is_equal = 1;
  for (i = 0; i < NUM_DEPTH*BLK_SIZE_MN; i ++) {
    if (fabsf(R[i]-R_golden[i]) >= 1e-5)
      printf("%lf %lf\n", R[i], R_golden[i]);

    is_equal &= fabsf(R[i]-R_golden[i]) < 1e-5;
  }

  if (is_equal) {
    PRINT_PASSED("PASSED!");
  } else {
    PRINT_FAILED("FAILED!");
  }

  free(A);
  free(B);
  free(T);
  free(C);
  free(R);
  free(R_golden);

  return 0;
}

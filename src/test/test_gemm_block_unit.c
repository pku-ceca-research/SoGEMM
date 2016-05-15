
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <getopt.h>
#include <sys/time.h>

#include "gemm_utils.h"
#include "gemm_consts.h"
#include "gemm_accel.h"

#ifdef __SDSCC__
#include "sds_lib.h"
#define malloc(x) (sds_alloc((x)))
#define free(x) (sds_free((x)))
#endif

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

  float *A = (float *) malloc(sizeof(float)*BLK_SIZE_MK);
  float *B = (float *) malloc(sizeof(float)*BLK_SIZE_KN);
  float *T = (float *) malloc(sizeof(float)*BLK_SIZE_MN);
  float *C = (float *) malloc(sizeof(float)*BLK_SIZE_MN);
  float *R = (float *) malloc(sizeof(float)*BLK_SIZE_MN);
  float *R_golden = (float *) malloc(sizeof(float)*BLK_SIZE_MN);
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

  int i, j, k;
  int t;
  float ALPHA = 0.1;
  float BETA  = 0;

  PRINT_TITLE("TEST gemm_block_unit");
  printf("NUM_ITER:\t %d\n", iter);
  printf("NUM_PIPES:\t %d\n", NUM_PIPES);
  printf("NUM_DEPTH:\t %d\n", NUM_DEPTH);
  printf("BLK_M:\t\t %d\n", BLK_M);
  printf("BLK_N:\t\t %d\n", BLK_N);
  printf("BLK_K:\t\t %d\n", BLK_K);
  printf("ALPHA:\t\t %f\n", ALPHA);
  printf("BETA:\t\t %f\n", BETA);

  for (i = 0; i < BLK_SIZE_MK; i ++) A[i] = 0.1;// (float)rand()/RAND_MAX;
  for (i = 0; i < BLK_SIZE_KN; i ++) B[i] = 0.1;// (float)rand()/RAND_MAX;
  for (i = 0; i < BLK_SIZE_MN; i ++) T[i] = 0.0;
  for (i = 0; i < BLK_SIZE_MN; i ++) {
    R[i] = 0.0;
    C[i] = (float)rand()/RAND_MAX;
  }
  printf("Matrix initialised\n");
  
  struct timeval start_time, end_time;
  double total_time;

  gettimeofday(&start_time, NULL);
  for (t = 0; t < iter; t ++)
    gemm_accel(A,B,C,R,ALPHA,BETA);
  gettimeofday(&end_time, NULL);
  total_time = (double)(end_time.tv_sec-start_time.tv_sec)+(end_time.tv_usec-start_time.tv_usec)*1e-6;
  printf("FINISHED origin: %lfs GFLOPS: %lf\n", total_time/iter, (BLK_M*BLK_N*BLK_K*3)/total_time*iter*1e-9);

  gettimeofday(&start_time, NULL);
  for (t = 0; t < iter; t++) {
    for (i = 0; i < NUM_DEPTH*BLK_SIZE_MN; i ++) 
      R_golden[i] = C[i];

    for (i = 0; i < BLK_M; i ++)
      for (j = 0; j < BLK_N; j ++)
        for (k = 0; k < BLK_K; k ++)
          R_golden[i*BLK_N+j] += 
            ALPHA * A[i*BLK_K+k] * B[k*BLK_N+j];
  }
  gettimeofday(&end_time, NULL);
  total_time = (double)(end_time.tv_sec-start_time.tv_sec)+(end_time.tv_usec-start_time.tv_usec)*1e-6;
  printf("FINISHED golden: %lfs\n", total_time/iter);

  int is_equal = 1;
  for (i = 0; i < BLK_SIZE_MN; i ++) {
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

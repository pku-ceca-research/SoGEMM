
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <time.h>
#include <getopt.h>
#include <math.h>

#include "gemv_sds.hh"

int main(int argc, char *argv[]) {
  int c;
  int M = 32, N = 32, T = 0;
  int iteration = 1;
  while ((c = getopt(argc, argv, "m:n:i:t")) != -1) 
    switch (c) {
      case 'm': M = atoi(optarg); break;
      case 'n': N = atoi(optarg); break;
      case 'i': iteration = atoi(optarg); break;
      case 't': T = 1; break;
      default:
        fprintf(stderr, "usage: %s -m [M] -n [N] -i [iteration]\n",
            argv[0]);
        exit(1);
    }

  float *A = (float*) malloc(sizeof(float)*M*N);
  float *X = (float*) malloc(sizeof(float)*N);
  float *Y = (float*) malloc(sizeof(float)*M);
  float *Y_golden = (float*) malloc(sizeof(float)*M);

  int i, j, k;
  for (i = 0; i < M * N; i ++)
    A[i] = (float)rand()/RAND_MAX;
  for (i = 0; i < N; i++) X[i] = (float)rand()/RAND_MAX;
  for (i = 0; i < M; i++) Y[i] = Y_golden[i] = (float)rand()/RAND_MAX;

  float ALPHA = (float)rand()/RAND_MAX;
  float BETA = (float)rand()/RAND_MAX;
  int lda = (T) ? M : N;

  printf("ALPHA: %12.6lf\n", ALPHA);
  printf("BETA:  %12.6lf\n", BETA);
  printf("T:     %5d\n", T);
  printf("LDA:   %5d\n", lda);
  printf("A:     %5d X %5d\n", M, N);
  printf("X:     %5d X %5d\n", N, 1);
  printf("Y:     %5d X %5d\n", M, 1);

  clock_t start, end;
  start = clock();
  for (i = 0; i < iteration; i ++)
    gemv_sds(T, M, N, ALPHA, (const float*)A, lda, (const float*)X, 1, BETA, Y, 1);
  end = clock();
  double total_time = (double)(end-start)/CLOCKS_PER_SEC/iteration;
  printf("Finished GEMV SDS test: %lf s\n", total_time);
  // printf("compute: %lf s\n", compute_time);
  // printf("memory:  %lf s\n", total_time-compute_time);

  start = clock();
  for (i = 0; i < iteration; i ++) {
    for (j = 0; j < M; j ++)
      Y_golden[j] *= BETA;
    for (j = 0; j < M; j ++)
      for (k = 0; k < N; k ++) {
        float a = (T) ? A[k*lda+j] : A[j*lda+k];
        Y_golden[j] += ALPHA * a * X[k];
      }
  }
  end = clock();
  printf("Finished GEMV golden test: %lf s\n", (double)(end-start)/CLOCKS_PER_SEC/iteration); 

  if (iteration == 1)
    for (i = 0; i < M; i ++)
      if (fabsf(Y_golden[i]-Y[i]) > 1e-6) {
        fprintf(stderr, "ERROR at %3d: golden=%lf origin=%lf\n",
            i, Y_golden[i], Y[i]);
        exit(1);
      } 
 
  printf("PASSED\n");
  return 0;
}

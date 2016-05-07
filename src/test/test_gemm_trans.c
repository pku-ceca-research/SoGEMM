
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <getopt.h>

#include "gemm_consts.h"
#include "test_gemm_trans.h"

void test_flatten_matrix_to_blocked(int M, int N, int verbose) {
  float *A = random_matrix(M, N);
  float *A_T = random_matrix(N, M);
  int blk_m = BLK_M;
  int blk_n = BLK_N;

  int is_valid = 1;

  PRINT_HIGHLIGHT("# Test normal transform ...");

  BlockedMatrix *A_BLK = flatten_matrix_to_blocked(0,A,M,N,N,blk_m,blk_n);
  if (verbose) {
    printf("A =>\n");
    print_matrix(A,M,N);
  }
  show_blocked_matrix(A_BLK, verbose);

  int num_blk_M = A_BLK->H / A_BLK->bH;
  int num_blk_N = A_BLK->W / A_BLK->bW;
  int num_blk = num_blk_M * num_blk_N;
  int blk_id;
  int blk_size = blk_m*blk_n;
  for (blk_id = 0; blk_id < num_blk; blk_id ++) {
    int blk_i = blk_id / num_blk_N;
    int blk_j = blk_id % num_blk_N;
    int src_i = blk_i * A_BLK->bH;
    int src_j = blk_j * A_BLK->bW;
    int x, y;
    for (x = 0; x < A_BLK->bH; x++)
      for (y = 0; y < A_BLK->bW; y++) {
        int src_blk_i = src_i + x;
        int src_blk_j = src_j + y;
        if (src_blk_i >= M || src_blk_j >= N)
          continue;
        float src = A[src_blk_i*N+src_blk_j];
        float dst = A_BLK->mat[blk_id*blk_size+x*A_BLK->bW+y];
        // printf("(%3d, %3d) src=%lf dst=%lf\n", src_blk_i, src_blk_j, src, dst);
        is_valid &= (fabsf(src-dst) < 1e-6);
      }
  }
  if (is_valid) {
    PRINT_PASSED("PASSED");
  } else {
    PRINT_FAILED("FAILED");
  }

  PRINT_HIGHLIGHT("# Test transposed transform ...");

  BlockedMatrix *A_T_BLK = flatten_matrix_to_blocked(1,A_T,M,N,M,blk_m,blk_n);
  if (verbose) {
    printf("A_T =>\n");
    print_matrix(A_T,N,M);
  }
  show_blocked_matrix(A_T_BLK, verbose);

  is_valid = 1;
  int T_M = N, T_N = M;
  for (blk_id = 0; blk_id < num_blk; blk_id ++) {
    int blk_i = blk_id % num_blk_M;
    int blk_j = blk_id / num_blk_M;
    int src_i = blk_i * A_T_BLK->bW;
    int src_j = blk_j * A_T_BLK->bH;
    int x, y;
    for (x = 0; x < A_T_BLK->bW; x++)
      for (y = 0; y < A_T_BLK->bH; y++) {
        int src_blk_i = src_i + x;
        int src_blk_j = src_j + y;
        if (src_blk_i >= T_M || src_blk_j >= T_N)
          continue;
        float src = A_T[src_blk_i*T_N+src_blk_j];
        float dst = A_T_BLK->mat[blk_id*blk_size+y*A_T_BLK->bW+x];
        // printf("(%3d, %3d) src=%lf dst=%lf\n", src_blk_i, src_blk_j, src, dst);
        is_valid &= (fabsf(src-dst) < 1e-6);
      }
  }
  if (is_valid) {
    PRINT_PASSED("PASSED");
  } else { 
    PRINT_FAILED("FAILED");
  }

  free(A);
  free(A_T);
  free_blocked_matrix(A_BLK);
  free_blocked_matrix(A_T_BLK);
}

void test_blocked_matrix_to_flatten(int M, int N, int verbose) {
  
  int blk_m = BLK_M, blk_n = BLK_N;
  BlockedMatrix *blk_mat   = random_blocked_matrix(0,M,N,N,blk_m,blk_n);
  BlockedMatrix *blk_mat_T = random_blocked_matrix(1,M,N,M,blk_m,blk_n);
  
  float *A   = (float *) malloc(sizeof(float)*M*N);
  float *A_T = (float *) malloc(sizeof(float)*N*M);
  memset(A_T, 0, sizeof(float)*N*M);

  // int num_blk_M = blk_mat->H/blk_mat->bH;
  int num_blk_N = blk_mat->W/blk_mat->bW;
  int i, j, x, y;
  int is_valid = 1;

  PRINT_HIGHLIGHT("# TEST normal blocked matrix");
  blocked_matrix_to_flatten(blk_mat, A);
  printf("A =>\n");
  show_blocked_matrix(blk_mat, verbose);
  if (verbose) {
    printf("Transformed => \n");
    print_matrix(A, M, N);
  }
  
  for (i = 0; i < M; i += blk_m)
    for (j = 0; j < N; j += blk_n)
      for (x = 0; x < blk_m; x ++)
        for (y = 0; y < blk_n; y ++) {
          if (x + i >= M || y + j >= N)
            continue;
          int blk_i = i / blk_m, blk_j = j / blk_n;
          int blk_id = blk_i * num_blk_N + blk_j;
          int blk_base = blk_id * blk_m * blk_n;
          int blk_idx = blk_base + x * blk_n + y;
          float src = A[(i+x)*N+(j+y)];
          float dst = blk_mat->mat[blk_idx];
          is_valid &= (fabsf(src-dst) < 1e-6);
          if (!is_valid)
            printf("(%3d %3d %3d %3d) blk_id=%3d src=%lf dst=%lf\n",
                i, j, x, y, blk_id, src, dst);
        }

  if (is_valid) {
    PRINT_PASSED("PASSED!");
  } else {
    PRINT_FAILED("FAILED");
  }

  PRINT_HIGHLIGHT("# TEST transposed blocked matrix");
  blocked_matrix_to_flatten(blk_mat_T, A_T);
  printf("A_T =>\n");
  show_blocked_matrix(blk_mat_T, verbose);
  if (verbose) {
    printf("Transformed =>\n");
    print_matrix(A_T, N, M);
  }

  is_valid = 1;
  for (i = 0; i < N; i += blk_n)
    for (j = 0; j < M; j += blk_m)
      for (x = 0; x < blk_n; x ++)
        for (y = 0; y < blk_m; y ++) {
          if (x + i >= N || y + j >= M)
            continue;
          int blk_i = i / blk_n, blk_j = j / blk_m;
          int blk_id = blk_j * num_blk_N + blk_i;
          int blk_base = blk_id * blk_m * blk_n;
          int blk_idx = blk_base + y * blk_n + x;
          float src = A_T[(i+x)*M+(j+y)];
          float dst = blk_mat_T->mat[blk_idx];
          is_valid &= (fabsf(src-dst) < 1e-6);
          if (!is_valid)
            printf("(%3d %3d %3d %3d) blk_id=%3d src=%lf dst=%lf\n",
                i, j, x, y, blk_id, src, dst);
        }

  if (is_valid) {
    PRINT_PASSED("PASSED!");
  } else {
    PRINT_FAILED("FAILED");
  }
}

int main(int argc, char *argv[]) {
  int c;
  int verbose = 0;
  int m = 1, n = 1;
  while ((c = getopt(argc, argv, "vm:n:")) != -1 )
    switch (c) {
      case 'v':
        verbose = 1; break;
      case 'm':
        m = atoi(optarg); break;
      case 'n':
        n = atoi(optarg); break;
      default:
        fprintf(stderr, "Unknown option %d\n", c);
        exit(1);
    }

  test_flatten_matrix_to_blocked(m, n, verbose);
  test_blocked_matrix_to_flatten(m, n, verbose);

  return 0;
}

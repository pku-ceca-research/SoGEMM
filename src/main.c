
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "gemm.h"

// We assume the gemm software version is always correct
int gemm_test() { return 0; };

void assign_random_matrix(float *A, int M, int N) {
    int i;
    for (i = 0; i < M * N; i++)
        A[i] = (float) rand()/RAND_MAX;
}

void print_matrix(float *A, int M, int N) {
    int i, j;
    for (i = 0; i < M; i++) {
        printf("[ ");
        for (j = 0; j < N; j++) {
            printf("%10.3f", A[i*N+j]);
            if (j != N-1)
                printf(", ");
        }
        printf(" ]\n");
    }
}

void run_soft_gemm_test(int TA, int TB, int M, int N, int K,
    float *A, int lda, 
    float *B, int ldb,
    float *C, int ldc,
    int iter)
{
    int i;
    // software gemm
    clock_t start = clock(), end;
    for (i = 0; i < iter; i++) 
        gemm(TA, TB, M, N, K, 1, A, lda, B, ldb, 1, C, ldc);
    end = clock();
    double flop = ((double)M)*N*(2.*K+2.)*iter; 
    double gflop = flop * 1e-9;
    double seconds = (double)(end-start)/CLOCKS_PER_SEC;
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s, %lf GFLOPS\n",
        M, K, K, N, TA, TB, seconds, gflop/seconds);
}

void test_gemm_calculation(int TA, int TB, int M, int N, int K) {
    int iter = 10;
    float *A, *B, *C;

    // initialize    
    A = (float *) malloc(sizeof(float) * M * K);
    B = (float *) malloc(sizeof(float) * N * K);
    C = (float *) malloc(sizeof(float) * M * N);

    int i;
    assign_random_matrix(A, M, K);
    assign_random_matrix(B, K, N);
    assign_random_matrix(C, M, N);
    
    int lda = (!TA) ? K : M;
    int ldb = (!TB) ? N : K;
    int ldc = N;

    run_soft_gemm_test(TA, TB, M, N, K, A, lda, B, ldb, C, ldc, iter);

    free(A);
    free(B);
    free(C);

    return ;
}

int main(int argc, char *argv[]) {
    int M, N, K, I;

    M = 64;
    N = 128;
    K = 256;
    I = 10;
    
    printf("Test Started\n");

    int i;
    for (i = 0; i < I; i++) {
        printf("Running test pass #%d ...\n", i+1);

        test_gemm_calculation(0, 0, M, N, K);
        test_gemm_calculation(0, 1, M, N, K);
        test_gemm_calculation(1, 0, M, N, K);
        test_gemm_calculation(1, 1, M, N, K);
    }

    printf("Test Ended\n");
    return 0;
}
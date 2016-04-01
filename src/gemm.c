
#include "gemm.h"

// Normal A(M,K) X Normal B(K,N)
void gemm_nn(int M, int N, int K, float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float *C, int ldc) 
{
    int i, j, k;
    for (i = 0; i < M; i++) {
        for (k = 0; k < K; k++) {
            register float A_PART = ALPHA*A[i*lda+k];
            for (j = 0; j < N; j++) {
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

// Normal A(M,K) X Transport B(N,K)
void gemm_nt(int M, int N, int K, float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float *C, int ldc) 
{
    int i, j, k;
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            register float sum = 0; 
            for (k = 0; k < K; k++) {
                sum += ALPHA*A[i*lda+k]*B[j*ldb+k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

// Transport A(K,M) X Normal B(K,N)
void gemm_tn(int M, int N, int K, float ALPHA,
    float *A, int lda, 
    float *B, int ldb,
    float *C, int ldc) 
{
    int i, j, k;
    for (i = 0; i < M; i++) {
        for (k = 0; k < K; k++) {
            register float A_PART = ALPHA*A[k*lda+i];
            for (j = 0; j < N; j++) {
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

// Transport A(K,M) X Transport B(N,K)
void gemm_tt(int M, int N, int K, float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float *C, int ldc)
{
    int i, j, k;
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            register float sum = 0;
            for (k = 0; k < K; k++) {
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
    float *A, int lda, 
    float *B, int ldb,
    float BETA,
    float *C, int ldc)
{
    int i, j;
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            C[i*ldc+j] *= BETA;
        }
    }
    // choose transport flag
    if (!TA && !TB)
        gemm_nn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    else if (TA && !TB)
        gemm_tn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    else if (!TA && TB)
        gemm_nt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    else
        gemm_tt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
}

void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
    float *A, int lda, 
    float *B, int ldb,
    float BETA,
    float *C, int ldc)
{
    gemm_cpu(TA, TB, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
}
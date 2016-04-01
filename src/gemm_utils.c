
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#include "gemm_utils.h"

// copied
void print_matrix(D_TYPE *A, int M, int N) {
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

void print_blocked_matrix(D_TYPE *A, int M, int N, int blk_m, int blk_n) {
    int M_align = (int) ceil((double)M/blk_m)*blk_m;
    int N_align = (int) ceil((double)N/blk_n)*blk_n;

    int i;
    int blk_size = blk_m*blk_n;
    for (i = 0; i < (M_align*N_align); i += blk_size) {
        printf("[%d]:\n", i/blk_size);
        print_matrix(A+i, blk_m, blk_n);
    }
}

D_TYPE *random_matrix(int M, int N) {
    int i;
    D_TYPE *A = (D_TYPE *) malloc(sizeof(D_TYPE)*M*N);

    for (i = 0; i < M*N; i ++)
        A[i] = (D_TYPE) rand()/RAND_MAX;

    return A;
}

D_TYPE *random_general_matrix(int M, int N, int lda) {
    if (lda < N) {
        fprintf(stderr, "LDA(%d) should be leq to N(%d)\n", lda, N);
        exit(1);
    }

    int i, j;
    D_TYPE *A = (D_TYPE *) malloc(sizeof(D_TYPE)*M*lda);

    for (i = 0; i < M; i++)
        for (j = 0; j < N; j++)
            A[i*lda+j] = (D_TYPE) rand()/RAND_MAX;

    return A;
}

int is_matrix_equal(D_TYPE *A, D_TYPE *B, int M, int N, int lda) {
    int is_equal = 1;
    int i, j;
    for (i = 0; i < M; i++)
        for (j = 0; j < N; j++)
            is_equal &= (A[i*lda+j] == B[i*lda+j]);
    return is_equal;
}

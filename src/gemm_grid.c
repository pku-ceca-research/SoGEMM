
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "gemm_utils.h"
#include "gemm_grid.h"

// it seems that sds_alloc will fail in this case...
// #ifdef SDS
// #include "sds_lib.h"
// #define MALLOC(x) sds_alloc((x))
// #define FREE(x) sds_free((x))
// #else
#define MALLOC malloc
#define FREE free
// #endif

int get_blocked_width(int orig, int blk_width) {
	return (int) ceil((double)orig/blk_width)*blk_width;
}

/* This is a very basic version of transformation:
 * lda
 */
float *trans_to_blocked(float *A, int m, int n, int lda, int blk_m, int blk_n) {
	// sanity checks
	if (!A) {
		fprintf(stderr, "A should not be NULL\n");
		exit(1);
	}
	if (lda < n) {
		fprintf(stderr, "LDA(%d) should be larger than N(%d)\n", lda, n);
		exit(1);
	}

	// m_align and n_align decides the matrix size
	int m_align = get_blocked_width(m, blk_m);
	int n_align = get_blocked_width(n, blk_n);

	int blk_size = blk_m*blk_n;
	int blk_per_m = m_align/blk_m;
	int blk_per_n = n_align/blk_n;

	float *A_block = (float *) MALLOC(sizeof(float)*m_align*n_align);
	if (!A_block) {
		fprintf(stderr, "Can't allocate memory for (%d, %d)\n", m_align, n_align);
		exit(1);
	}

	int i, j, x, y;
	for (i = 0; i < m_align; i += blk_m) {
		for (j = 0; j < n_align; j += blk_n) {
			int blk_i = i/blk_m;
			int blk_j = j/blk_n;
			int blk_id = blk_i*blk_per_n+blk_j;
			// inside block
			for (x = 0; x < blk_m && x + i < m; x ++) {
				for (y = 0; y < blk_n && y + j < n; y ++) {
					int dst_idx = blk_id*blk_size+x*blk_n+y;
					int src_idx = (i+x)*n+j+y;
					A_block[dst_idx] = A[src_idx];
				}
			}
		}
	}

	return A_block;
}

void trans_from_blocked(float *A, float *A_block, int M, int N, int lda, int blk_m, int blk_n) {
	int i, j;

	int M_align = get_blocked_width(M, blk_m);
	int N_align = get_blocked_width(N, blk_n);
	
	int blk_size = blk_m*blk_n;
	int num_blk = M_align*N_align/blk_size;
	int num_blk_m = M_align/blk_m;
	int num_blk_n = N_align/blk_n;

	for (i = 0; i < M_align*N_align; i += blk_size) {
		int blk_id = i/blk_size;
		int blk_id_m = blk_id/num_blk_n;
		int blk_id_n = blk_id%num_blk_n;

		for (j = 0; j < blk_size; j++) {
			int x = blk_id_m*blk_m+j/blk_n;
			int y = blk_id_n*blk_n+j%blk_n;
			if (x < M && y < N)
				A[x*lda+y] = A_block[i+j];
		}
	}
}


/* INTERFACE */

#ifdef BLK_ON_THE_FLY
/* ON THE FLY version:
 * There will not be a pre-allocated blocked array. 
 * The time complexity will remain the same.
 * But the space will be greatly reduced (nearly 2x).
 * HOWEVER, this version is much slower. I think it is mainly due to the 
 * cache-miss
 */

void gemm_mmult(float ALPHA,
	float A[BLK_M*BLK_K], 
	float B[BLK_K*BLK_N],
	float C[BLK_M*BLK_N])
{
	int i, j, k;
	for (i = 0; i < BLK_M; i++) 
		for (j = 0; j < BLK_N; j++) {
			float sum = 0.0;
			for (k = 0; k < BLK_K; k++)
				sum += ALPHA*A[i*BLK_K+k]*B[k*BLK_N+j];
			C[i*BLK_N+j] = sum;
		}
}

void gemm_madd(
	float A[BLK_M*BLK_N], 
	float B[BLK_M*BLK_N],
	float C[BLK_M*BLK_N])
{
	int i, j, k;
	for (i = 0; i < BLK_M; i++) 
		for (j = 0; j < BLK_N; j++)
			C[i*BLK_N+j] = A[i*BLK_N+j] + B[i*BLK_N+j];
}

void gemm_grid_nn(int M, int N, int K, float ALPHA,
	float *A, int lda, 
	float *B, int ldb,
	float *C, int ldc)
{
	// buffers
	float* A_BLK = (float *) MALLOC(sizeof(float)*BLK_M*BLK_K);
	float* B_BLK = (float *) MALLOC(sizeof(float)*BLK_K*BLK_N);
	float* C_BLK = (float *) MALLOC(sizeof(float)*BLK_M*BLK_N); 
	float* T_BLK = (float *) MALLOC(sizeof(float)*BLK_M*BLK_N);
	float* R_BLK = (float *) MALLOC(sizeof(float)*BLK_M*BLK_N);

	// main 3-loop
	int i, j, k;
	for (i = 0; i < M; i += BLK_M) {
		for (j = 0; j < N; j += BLK_N) {
			for (k = 0; k < K; k += BLK_K) {
				// inner loops - initialize
				int blk_i, blk_j, blk_k;
				for (blk_i = i; blk_i < BLK_M+i; blk_i ++) {
					for (blk_j = j; blk_j < BLK_N+j; blk_j ++) {
						for (blk_k = k; blk_k < BLK_K+k; blk_k ++) {
							// id list
							int blk_id_a = (blk_i-i)*BLK_K+(blk_k-k);
							int blk_id_b = (blk_k-k)*BLK_N+(blk_j-j);
							int blk_id_c = (blk_i-i)*BLK_N+(blk_j-j);
							// assignment
							A_BLK[blk_id_a] = (blk_i>=M||blk_k>=K) ? 0.0 : A[blk_i*lda+blk_k];
							B_BLK[blk_id_b] = (blk_j>=N||blk_k>=K) ? 0.0 : B[blk_k*ldb+blk_j];
							C_BLK[blk_id_c] = (blk_i>=M||blk_j>=N) ? 0.0 : C[blk_i*ldc+blk_j];
						}
					}
				}
				// computation
				gemm_mmult(ALPHA, A_BLK, B_BLK, T_BLK);
				gemm_madd(T_BLK, C_BLK, R_BLK);
				// inner loops - write back
				for (blk_i = i; blk_i < BLK_M+i && blk_i < M; blk_i ++) {
					for (blk_j = j; blk_j < BLK_N+j && blk_j < N; blk_j ++) {
						C[blk_i*ldc+blk_j] = R_BLK[(blk_i-i)*BLK_N+(blk_j-j)];
					}
				}
			}
		}
	}

	FREE(A_BLK);
	FREE(B_BLK);
	FREE(C_BLK); 
	FREE(T_BLK);
	FREE(R_BLK);
}

void gemm_grid(int TA, int TB, int M, int N, int K, float ALPHA,
	float *A, int lda,
	float *B, int ldb,
	float BETA,
	float *C, int ldc)
{
	// printf("gemm_grid ON_THE_FLY TA=%d TB=%d M=%d N=%d K=%d ALPHA=%f BETA=%f\n", 
	// 	TA, TB, M, N, K, ALPHA, BETA);
	int i;
	for (i = 0; i < M*N; i++)
		C[i] *= BETA;

	if (!TA && !TB)
		gemm_grid_nn(M,N,K,ALPHA,A,lda,B,ldb,C,ldc);
	else 
		; // TODO
}

#else /* BLK_ON_THE_FLY */

/* synthesis on hardware */
#pragma SDS data access_pattern(in_A:SEQUENTIAL, in_B:SEQUENTIAL, out_C:SEQUENTIAL)
void gemm_mmult(float ALPHA, 
	float A_BLK[BLK_M*BLK_K], 
	float B_BLK[BLK_K*BLK_N],
	float C_BLK[BLK_M*BLK_N])
{
	int i, j, k;
	for (i = 0; i < BLK_M; i++) {
		for (j = 0; j < BLK_N; j++) {
			#ifdef USE_PIPELINE
				#pragma HLS PIPELINE II=1
			#endif
			float sum = 0.0;
			for (k = 0; k < BLK_K; k++) {
				sum += ALPHA*A_BLK[i*BLK_K+k]*B_BLK[k*BLK_N+j];
			}
			// here is +=
			C_BLK[i*BLK_N+j] = sum;
		}
	}
}

#pragma SDS data access_pattern(A:SEQUENTIAL, B:SEQUENTIAL, C:SEQUENTIAL)
void gemm_madd(
	float A[BLK_M*BLK_N], 
	float B[BLK_M*BLK_N], 
	float C[BLK_M*BLK_N]) 
{
	int i, j;
	for (i = 0; i < BLK_M; i++)
		for (j = 0; j < BLK_N; j++)
			#ifdef USE_PIPELINE
				#pragma HLS PIPELINE II=1
			#endif
			C[i*BLK_N+j] = A[i*BLK_N+j] + B[i*BLK_N+j];
}

void gemm_grid_nn(int M, int N, int K, float ALPHA, 
	float *A, int lda,
	float *B, int ldb,
	float *C, int ldc)
{
	int blk_i, blk_j, blk_k;
	int i;
	
	int num_blk_m = M/BLK_M;
	int num_blk_n = N/BLK_N;
	int num_blk_k = K/BLK_K;

	int blk_size_A = BLK_M*BLK_K;
	int blk_size_B = BLK_K*BLK_N;
	int blk_size_C = BLK_M*BLK_N;

	float* T_BLK = (float *) MALLOC(sizeof(float)*blk_size_C); // for C
	float* R_BLK = (float *) MALLOC(sizeof(float)*blk_size_C); // for C

	for (blk_i = 0; blk_i < num_blk_m; blk_i++) {
		for (blk_j = 0; blk_j < num_blk_n; blk_j++) {
			int blk_id_C = blk_i*num_blk_n+blk_j;

			float *C_BLK = C+blk_id_C*blk_size_C;
			for (blk_k = 0; blk_k < num_blk_k; blk_k++) {
				int blk_id_A = blk_i*num_blk_k+blk_k;
				int blk_id_B = blk_k*num_blk_n+blk_j;
				
				float *A_BLK = A+blk_id_A*blk_size_A;
				float *B_BLK = B+blk_id_B*blk_size_B;

				// call matrix ops
				gemm_mmult(ALPHA,A_BLK,B_BLK,T_BLK);
				gemm_madd(T_BLK,C_BLK,R_BLK);

				memcpy(C_BLK,R_BLK,sizeof(float)*blk_size_C);
			}
		}
	}

	FREE(T_BLK);
	FREE(R_BLK);
}

void gemm_grid(int TA, int TB, int M, int N, int K, float ALPHA,
	float *A, int lda,
	float *B, int ldb,
	float BETA,
	float *C, int ldc)
{
	// printf("gemm_grid TA=%d TB=%d M=%d N=%d K=%d ALPHA=%f BETA=%f\n", 
	// 	TA, TB, M, N, K, ALPHA, BETA);
	float *A_block = trans_to_blocked(A,M,K,lda,BLK_M,BLK_K);
	float *B_block = trans_to_blocked(B,K,N,ldb,BLK_K,BLK_N);
	float *C_block = trans_to_blocked(C,M,N,ldc,BLK_M,BLK_N);

	// print_blocked_matrix(A_block,M,K,BLK_M,BLK_K);
	// print_blocked_matrix(B_block,K,N,BLK_K,BLK_N);

	int M_align = get_blocked_width(M, BLK_M);
	int N_align = get_blocked_width(N, BLK_N);
	int K_align = get_blocked_width(K, BLK_K);
	// printf("Aligned M=%d N=%d K=%d\n", M_align, N_align, K_align);

	int i;
	for (i = 0; i < (M_align*N_align); i++)
		C_block[i] *= BETA;

	if (!TA && !TB)
		gemm_grid_nn(M_align,N_align,K_align,ALPHA,A_block,lda,B_block,ldb,C_block,ldc);
	else 
		; // TODO

	// print_blocked_matrix(C_block,M,N,BLK_M,BLK_N);
	trans_from_blocked(C,C_block,M,N,ldc,BLK_M,BLK_N);

	FREE(A_block);
	FREE(B_block);
	FREE(C_block);
}
#endif
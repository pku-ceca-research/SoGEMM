
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <getopt.h>
#include <time.h>

#include "gemm_utils.h"
#include "gemm.h"
#include "gemm_grid.h"
#include "gemm_trans.h"

#define NUM_MAX_TESTS 100

int test_valid_blocking(float *A, float *A_block, int M, int N, int blk_m, int blk_n) {
	int is_valid = 1;
	int i, j, k;

	#ifdef VERBOSE
		print_blocked_matrix(A_block, M, N, blk_m, blk_n);
	#endif

	int M_align = (int) ceil((double)M/blk_m)*blk_m;
	int N_align = (int) ceil((double)N/blk_n)*blk_n;
	
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
			float A_block_val = A_block[i+j];
			
			is_valid &= 
				(x >= M || y >= N) ? 
				(A_block_val < 1e-5) : 
				(abs(A_block_val-A[x*N+y]) < 1e-5);
		}
	}

	return is_valid;
}

void test_valid_blocking_all(int M, int N) {
	printf("TEST valid blocking (%d, %d) ...\n", M, N);

	int blk_m, blk_n;
	float *A = (float *) malloc(sizeof(float)*M*N);

	int i;
	for (i = 0; i < M*N; i ++)
		A[i] = (float) rand()/RAND_MAX;

	// printf("A = \n");
	// print_matrix(A, M, N);

	printf("TESTING ...\n");
	int is_valid = 1;
	
	clock_t start = clock(), end;
	for (blk_m = 1; blk_m <= M; blk_m ++) {
		for (blk_n = 1; blk_n <= N; blk_n ++) {
			// printf("Blocked A(%3d, %3d) =\t", blk_m, blk_n);
			float *A_block = trans_to_blocked(0,A,M,N,N,blk_m,blk_n);

			is_valid &= test_valid_blocking(A, A_block, M, N, blk_m, blk_n);

			free(A_block);
		}
	}
	end = clock();
	double sec = (double)(end-start)/CLOCKS_PER_SEC;
	double sec_per_test = sec / (M * N);

	if (is_valid)
		printf("ALL TEST PASSED!\n");
	else 
		printf("ERROR\n");
	printf("TEST valid blocking (%d, %d) in %6lf s %6lfs for each matrix\n", 
		M, N, sec, sec_per_test);
}

void test_trans_to_blocked_bandwidth(int M, int N) {
	printf("TEST trans_to_blocked bandwidth (%d, %d) ...\n", M, N);
	int i, j, iter=100;
	int num_test_cases = 6;

	float *A = (float *) malloc(sizeof(float)*M*N);

	for (i = 0; i < M*N; i ++)
		A[i] = (float) rand()/RAND_MAX;

	for (i = 0; i < num_test_cases; i ++) {
		int blk_width = 2 << i;
		float *A_block;

		double sec = 0.0;
		for (j = 0; j < iter; j++) {
			clock_t start = clock();
			A_block = trans_to_blocked(0,A,M,N,N,blk_width,blk_width);
			clock_t end = clock();
			sec += (double)(end-start)/CLOCKS_PER_SEC;
		}
		sec /= iter;
		printf("Blocking (%d,%d) Time %4lf s\n", blk_width, blk_width, sec);

		free(A_block);
	}
}

void test_trans_and_trans_back(int M, int N) {

}

void test_normal_mmult(int M, int N, int K, int iter) {
	float *A = random_matrix(M, K);
	float *B = random_matrix(K, N);
	float *C = random_matrix(M, N);

	float ALPHA = (float) rand()/RAND_MAX;
	float BETA = (float) rand()/RAND_MAX;
	float *C_golden = (float*) malloc(sizeof(float)*M*N);
	memcpy(C_golden, C, sizeof(float)*M*N);
	
	int i;

	printf("TESTING SW version: iter=%d\n", iter);
	clock_t start = clock(), end;
	for (i = 0; i < iter; i++)
		gemm(0,0,M,N,K,ALPHA,A,K,B,N,BETA,C_golden,N);
	end = clock();
	double sw_sec = (double)(end-start)/CLOCKS_PER_SEC;

	printf("TESTING HW version: iter=%d\n", iter);
	start = clock();
	for (i = 0; i < iter; i++)
		gemm_grid(0,0,M,N,K,ALPHA,A,K,B,N,BETA,C,N);
	end = clock();
	double hw_sec = (double)(end-start)/CLOCKS_PER_SEC;
	
	int is_equal = 1;
	for (i = 0; i < M*N; i++)
		is_equal &= (abs(C_golden[i]-C[i]) < 1e-6);

	double flop = ((double)M)*N*(2.0*K+2.0)*iter;
	double gflop = flop*1e-9;
	printf("GEMM test %dx%d * %dx%d:\n", M, K, K, N);
	printf("SW %lf s, %lf GFlops\n", sw_sec, gflop/sw_sec);
	printf("HW %lf s, %lf GFlops\n", hw_sec, gflop/hw_sec);
	printf("Increase = %lf\n", (sw_sec-hw_sec)/sw_sec);
	if (is_equal)
		printf("PASSED!\n");
	else 
		printf("FAILED!\n");

	free(A);
	free(B);
	free(C);
	free(C_golden);
}

int main(int argc, char *argv[]) {	
	int c;
	int M=16, N=16, K=16;
	int iter=10;
	char *test_list[NUM_MAX_TESTS];
	const char *tokens;

	int i;
	int num_test = 0;

	while ((c = getopt(argc, argv, "m:n:k:t:i:")) != -1) {
		switch (c) {
			case 'm': M = atoi(optarg); break;
			case 'n': N = atoi(optarg); break;
			case 'k': K = atoi(optarg); break;
			case 'i': iter = atoi(optarg); break;
			case 't':
				/* List of test names, seperated by comma */
				tokens = ",";
				char *test_list_str = (char *) optarg;
				char *test_name = (char *) strtok(test_list_str, tokens);
				for (i = 0; i < NUM_MAX_TESTS && test_name != NULL; i ++) {
					test_list[i] = test_name;
					test_name = (char *) strtok(NULL, tokens);
				}
				num_test = i;
				break;
			default:
				fprintf(stderr, "Can't recognize option: %c %d\n", c, (int) c);
				exit(1);
		}
	}

	for (i = 0; i < num_test; i++) {
		char *test_name = test_list[i];
		printf("=== TEST[%d] name=%s\n", i, test_name);

		if (strcmp(test_name, "TRANS") == 0) {
			// test_trans_and_trans_back(M, N);
			float *A = random_matrix(M, N);
			float *A_T = random_matrix(N, M);
			int blk_m = 2;
			int blk_n = 3;

			blocked_matrix *A_BLK = flatten_matrix_to_blocked(0,A,M,N,N,blk_m,blk_n);
			printf("A =>\n");
			print_matrix(A,M,N);
			print_blocked_matrix(A_BLK->mat,M,N,blk_m,blk_n);

			blocked_matrix *A_T_BLK = flatten_matrix_to_blocked(1,A_T,M,N,M,blk_m,blk_n);
			printf("A_T =>\n");
			print_matrix(A_T,N,M);
			print_blocked_matrix(A_T_BLK->mat,M,N,blk_m,blk_n);

		} else if (strcmp(test_name, "NORMAL_MMULT") == 0) {
			test_normal_mmult(M, N, K, 1);
		} else if (strcmp(test_name, "NORMAL_MMULT_PROF") == 0) {
			test_normal_mmult(M, N, K, iter);
		}
	}

	// test_valid_blocking_all(M, N);
	// test_trans_to_blocked_bandwidth(M, N);
	// test_gemm_grid(M, N, K);
	// test_trans_and_trans_back(M, N);
	return 0;
}
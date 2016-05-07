#ifndef TEST_GEMM_TRANS_H__
#define TEST_GEMM_TRANS_H__

#include "gemm_types.h"
#include "gemm_trans.h"
#include "gemm_utils.h"

void test_flatten_matrix_to_blocked(int M, int N, int verbose);
void test_blocked_matrix_to_flatten(int M, int N, int verbose);

#endif
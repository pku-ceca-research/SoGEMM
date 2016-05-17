
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __SDSCC__
#include "sds_lib.h"
#endif

#include "gemm_sds.h"

#include "gemm_utils.h"
#include "gemm_consts.h"
#include "gemm_types.h"
#include "gemm_trans.h"
#ifdef GEMM_BLOCK
#include "gemm_block.h"
#endif
#include "gemm_plain.h"

void gemm_sds(int TA, int TB, int M, int N, int K,
    float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float BETA,
    float *C, int ldc)
{
#ifdef GEMM_BLOCK
  gemm_block(TA,TB,M,N,K,ALPHA,A,lda,B,ldb,BETA,C,ldc);
#else
  gemm_plain<float,float>(TA,TB,M,N,K,ALPHA,A,lda,B,ldb,BETA,C,ldc);
#endif
}

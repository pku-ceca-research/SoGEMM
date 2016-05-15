#ifndef GEMM_BLOCK_UNIT_H__
#define GEMM_BLOCK_UNIT_H__

#ifdef __SDSCC__
#include <stdlib.h>
#include "sds_lib.h"
#endif
/* constants are defined here */
#include "gemm_consts.h"

#ifndef GEMM_FULL_MODE
#pragma SDS data access_pattern(A:SEQUENTIAL, B:SEQUENTIAL, T:SEQUENTIAL)
#ifdef SLOW_SYS_PORT
#pragma SDS data sys_port(A:ACP, B:ACP, T:ACP)
#else
#pragma SDS data sys_port(A:AFI, B:AFI, T:AFI)
#endif
#endif
void gemm_block_units_mmult(
        float A[BLK_M*BLK_K],
        float B[BLK_K*BLK_N],
        float ALPHA,
        float T[BLK_M*BLK_N]);

#ifndef GEMM_FULL_MODE
#pragma SDS data access_pattern(C:SEQUENTIAL, T:SEQUENTIAL, R:SEQUENTIAL)
#ifdef SLOW_SYS_PORT
#pragma SDS data sys_port(T:ACP, C:ACP, R:ACP)
#else
#pragma SDS data sys_port(T:AFI, C:AFI, R:AFI)
#endif
#endif
void gemm_block_units_mplus(
        float T[BLK_M*BLK_N],
        float C[BLK_M*BLK_N],
        float R[BLK_M*BLK_N]);

#endif

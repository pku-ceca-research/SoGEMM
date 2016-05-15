#ifndef GEMM_ACCEL_HLS_H__
#define GEMM_ACCEL_HLS_H__

#include "gemm_consts.h"

#ifndef GEMM_FULL_MODE
#pragma SDS data access_pattern(A:SEQUENTIAL, B:SEQUENTIAL, C:SEQUENTIAL, R:SEQUENTIAL)
#pragma SDS data data_mover(A:AXIDMA_SIMPLE, B:AXIDMA_SIMPLE, C:AXIDMA_SIMPLE, R:AXIDMA_SIMPLE)
#endif
void gemm_accel_full(
    float A[BLK_M*BLK_K], 
    float B[BLK_M*BLK_K], 
    float C[BLK_M*BLK_N], 
    float ALPHA, 
    float R[BLK_M*BLK_N]);

#endif

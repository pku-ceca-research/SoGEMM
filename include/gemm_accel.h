#ifndef GEMM_ACCEL_H__
#define GEMM_ACCEL_H__

#ifdef GEMM_FULL_MODE
  #pragma SDS data access_pattern(A:SEQUENTIAL, B:SEQUENTIAL, C:SEQUENTIAL, R:SEQUENTIAL)
  // #pragma SDS data dim(A[BLK_M][BLK_K], B[BLK_N][BLK_K], C[BLK_M][BLK_N], R[BLK_M][BLK_N])
  // #pragma SDS data data_mover(A:AXIDMA_2D, B:AXIDMA_2D, C:AXIDMA_2D, R:AXIDMA_2D)
  // system port
  #ifdef SLOW_SYS_PORT
  #pragma SDS data sys_port(A:ACP, B:ACP, C:ACP, R:ACP)
  #else
  #pragma SDS data sys_port(A:AFI, B:AFI, C:AFI, R:AFI)
  #endif
// copy method
// #ifdef ZERO_COPY
// #pragma SDS data zero_copy(A[0:BLK_M][0:BLK_K], B[0:BLK_K][0:BLK_N], C[0:BLK_M][0:BLK_N], R[0:BLK_M][0:BLK_N])
// #else
// #pragma SDS data copy(A[0:BLK_M][0:BLK_K], B[0:BLK_K][0:BLK_N], C[0:BLK_M][0:BLK_N], R[0:BLK_M][0:BLK_N])
// #endif
#endif
void gemm_accel(
    float A[BLK_M*BLK_K], 
    float B[BLK_K*BLK_N], 
    float C[BLK_M*BLK_N], 
    float R[BLK_M*BLK_N],
    float ALPHA, float BETA);

#endif

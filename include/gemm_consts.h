#ifndef GEMM_CONSTS_H__
#define GEMM_CONSTS_H__

#if GEMM_SCALE == 0
  #define DIM 32
#elif GEMM_SCALE == 1
  #define DIM 16
#elif GEMM_SCALE == 2
  #define DIM 8
#elif GEMM_SCALE == 3
  #define DIM 48
#endif
#define BLK_M DIM
#define BLK_N BLK_M
#define BLK_K BLK_M

#define NUM_PIPES 1
#define NUM_DEPTH 1

#define BLK_SIZE_MN (BLK_M*BLK_N)
#define BLK_SIZE_KN (BLK_N*BLK_K)
#define BLK_SIZE_MK (BLK_M*BLK_K)

#define PIPE_SIZE_MN (BLK_SIZE_MN*NUM_PIPES)
#define PIPE_SIZE_KN (BLK_SIZE_KN*NUM_PIPES)
#define PIPE_SIZE_MK (BLK_SIZE_MK*NUM_PIPES)

#endif

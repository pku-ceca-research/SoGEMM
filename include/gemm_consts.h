#ifndef GEMM_CONSTS_H__
#define GEMM_CONSTS_H__

#define BLK_M     16
#define BLK_N     16
#define BLK_K     16

#define NUM_PIPES 1
#define NUM_DEPTH 1

#define BLK_SIZE_MN (BLK_M*BLK_N)
#define BLK_SIZE_KN (BLK_N*BLK_K)
#define BLK_SIZE_MK (BLK_M*BLK_K)

#define PIPE_SIZE_MN (BLK_SIZE_MN*NUM_PIPES)
#define PIPE_SIZE_KN (BLK_SIZE_KN*NUM_PIPES)
#define PIPE_SIZE_MK (BLK_SIZE_MK*NUM_PIPES)

#endif

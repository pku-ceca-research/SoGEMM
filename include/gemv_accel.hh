#ifndef GEMV_ACCEL_HH__
#define GEMV_ACCEL_HH__

// declare inner block size
#ifndef GEMV_BLK_M
#define GEMV_BLK_M 32
#endif

#ifndef GEMV_BLK_N
#define GEMV_BLK_N 32
#endif

#pragma SDS data access_pattern(A:SEQUENTIAL, R:SEQUENTIAL)
void gemv_accel(float A[GEMV_BLK_M*GEMV_BLK_N], float X[GEMV_BLK_N], float Y[GEMV_BLK_M],
    float R[GEMV_BLK_M], float ALPHA, float BETA);

#endif

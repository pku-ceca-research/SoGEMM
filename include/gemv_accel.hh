#ifndef GEMV_ACCEL_HH__
#define GEMV_ACCEL_HH__

#ifndef GEMV_BLK_N
#define GEMV_BLK_N 64
#endif

#pragma SDS data access_pattern(A:SEQUENTIAL, x:SEQUENTIAL)
#pragma SDS data sys_port(A:AFI, x:AFI)
float gemv_accel(float A[GEMV_BLK_N], float x[GEMV_BLK_N], float y, float ALPHA, float BETA);

#endif

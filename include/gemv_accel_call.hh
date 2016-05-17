#ifndef GEMV_ACCEL_CALL_HH__
#define GEMV_ACCEL_CALL_HH__

void gemv_accel_call(float A[GEMV_BLK_M*GEMV_BLK_N], float X[GEMV_BLK_N],
    float Y[GEMV_BLK_M], float R[GEMV_BLK_M], float ALPHA, float BETA);

#endif

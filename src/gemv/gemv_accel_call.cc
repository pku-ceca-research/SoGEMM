#include "gemv_accel.hh"
#include "gemv_accel_call.hh"

void gemv_accel_call(float A[GEMV_BLK_M*GEMV_BLK_N], float X[GEMV_BLK_N],
    float Y[GEMV_BLK_M], float R[GEMV_BLK_N], float ALPHA, float BETA)
{
  gemv_accel(A,X,Y,R,ALPHA,BETA); 
}

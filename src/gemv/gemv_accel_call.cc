#include "gemv_accel.hh"
#include "gemv_accel_call.hh"

float gemv_accel_call(float *A, float *x, float y, float ALPHA, float BETA)
{
  return gemv_accel(A,x,y,ALPHA,BETA); 
}

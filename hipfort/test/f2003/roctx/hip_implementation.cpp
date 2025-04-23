#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void empty()
{
}


extern "C"
{
  void launch()
  {
    hipLaunchKernelGGL((empty), dim3(1), dim3(1), 0, 0);
  }
}

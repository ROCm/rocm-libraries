#include <hip/hip_runtime.h>

using T = HIP_PLUGIN_POINTWISE_ADD_TYPE;

extern "C" __global__ void PointwiseAdd(const T* a, const T* b, T* c)
{
    unsigned i = blockIdx.x * HIP_PLUGIN_POINTWISE_ADD_BLOCK_SIZE + threadIdx.x;
    // Keep the 'a[i] + b[i]' expression literal: a multi-root test mutates it by regex.
    c[i] = a[i] + b[i];
}

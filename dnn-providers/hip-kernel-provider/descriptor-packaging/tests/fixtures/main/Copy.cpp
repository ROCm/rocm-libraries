#include <hip/hip_runtime.h>

using T = HIP_PLUGIN_COPY_TYPE;

extern "C" __global__ void Copy(const T* a, T* c)
{
    unsigned i = blockIdx.x * HIP_PLUGIN_COPY_BLOCK_SIZE + threadIdx.x;
    c[i] = a[i];
}

// Example hip UKD source for the production descriptor tree.
//
// Deliberately trivial: this exists to prove the hip producer's compile ->
// pack -> install path end to end, not to be a useful kernel. The build
// defines come from the descriptor's kernel_source.build.defines.

#include <hip/hip_runtime.h>

using T = HKP_EXAMPLE_POINTWISE_ADD_TYPE;

extern "C" __global__ void PointwiseAdd(const T* a, const T* b, T* c, unsigned n)
{
    unsigned i = blockIdx.x * HKP_EXAMPLE_POINTWISE_ADD_BLOCK_SIZE + threadIdx.x;
    if(i < n)
    {
        c[i] = a[i] + b[i];
    }
}

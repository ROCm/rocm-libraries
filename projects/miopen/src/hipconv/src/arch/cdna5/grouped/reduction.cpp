#include "reduction.hpp"
#include <hip/hip_runtime.h>

namespace hipconv::cdna5
{

// Split-K reduction: sum the `num_partitions` per-(q-tile, batch) partial dW
// buffers (each `dW_total` fp32 elements, laid out [partition][dW]) into dW.
__global__ void conv2d_grouped_multi_g_wgrad_reduce_cdna5(float* __restrict__ dW,
                                                          const float* __restrict__ partials,
                                                          int num_partitions,
                                                          unsigned long long dW_total)
{
    const unsigned long long e = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if(e >= dW_total)
        return;
    float sum = 0.0f;
    for(int p = 0; p < num_partitions; p++)
        sum += partials[(unsigned long long)p * dW_total + e];
    dW[e] = sum;
}

} // namespace hipconv::cdna5

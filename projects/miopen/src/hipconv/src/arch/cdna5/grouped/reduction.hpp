#pragma once

namespace hipconv::cdna5
{

// Split-K reduction: sum the `num_partitions` per-(q-tile, batch) partial dW
// buffers (each `dW_total` fp32 elements, laid out [partition][dW]) into dW.
__global__ void conv2d_grouped_multi_g_wgrad_reduce_cdna5(float* __restrict__ dW,
                                                          const float* __restrict__ partials,
                                                          int num_partitions,
                                                          unsigned long long dW_total);

} // namespace hipconv::cdna5

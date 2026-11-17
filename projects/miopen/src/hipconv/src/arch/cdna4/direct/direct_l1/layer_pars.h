#pragma once

#include <hip/hip_runtime.h>

namespace hipconv::cdna4
{

struct InputPars
{
    // Batch dimension
    const int N;

    // Height of the feature map
    const int H;

    // Width of the feature map
    const int W;

    // Number of channels
    const int C;

    __host__ __device__ size_t size() const { return static_cast<size_t>(N) * H * W * C; }
};

// Define the shape of a weights tensor.
template <int Kh_, int Kw_>
struct WeightPars
{
    static constexpr int Kh = Kh_;

    static constexpr int Kw = Kw_;

    // Number of output channels
    const int K;

    // Number of input channels
    const int C;
};

// Define the shape of an output tensor (NHWC, grouped).
struct OutputPars
{
    // Batch dimension
    const int N;

    // Height of the output feature map
    const int Ho;

    // Width of the output feature map
    const int Wo;

    // Number of groups
    const int groups;

    // Output channels per group
    const int K_per_group;

    // Output channels per pixel (groups * K_per_group)
    __host__ __device__ int K_per_pixel() const { return groups * K_per_group; }
};

} // namespace hipconv::cdna4

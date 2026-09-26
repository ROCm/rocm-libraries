#include "algorithm.h"

#include <array>

using hipconv::ConvAlgorithm;
using hipconv::ConvKernelSpan;
using hipconv::ConvParams;

extern const ConvKernelSpan depthwise_1d_toeplitz_cdna5_kernels;

namespace
{

bool is_applicable(const ConvParams& par)
{
    // One channel per group (groups == C == K) identifies depthwise uniquely, a dense conv having
    // channels_per_group() == C != 1, and admits the degenerate C == K == groups == 1.
    if(par.k != par.c || par.channels_per_group() != 1)
        return false;
    return par.dilation_h == 1 && par.dilation_w == 1;
}

constexpr std::array<const ConvKernelSpan*, 1> kernel_groups = {
    &depthwise_1d_toeplitz_cdna5_kernels,
};

} // namespace

// Host-only: the device linker must not see this object or its function pointer.
#ifndef __HIP_DEVICE_COMPILE__
extern const ConvAlgorithm depthwise_cdna5{is_applicable, kernel_groups};
#endif

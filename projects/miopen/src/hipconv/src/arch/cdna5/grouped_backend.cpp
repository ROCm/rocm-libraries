#include "algorithm.h"

#include <array>

using hipconv::Conv2dParams;
using hipconv::ConvAlgorithm;
using hipconv::ConvKernelSpan;

extern const ConvKernelSpan grouped_multi_g_cdna5_kernels;
// Unified wgrad kernel covering group channel counts G in {4,8,16,32}
extern const ConvKernelSpan grouped_multi_g_wgrad_cdna5_kernels;

namespace
{

bool is_applicable(const Conv2dParams& par)
{
    if(par.groups == 1)
        return false;
    return par.dilation_h == 1 && par.dilation_w == 1;
}

constexpr std::array<const ConvKernelSpan*, 2> kernel_groups = {
    &grouped_multi_g_cdna5_kernels,
    &grouped_multi_g_wgrad_cdna5_kernels,
};

} // namespace

// Host-only: the device linker must not see this object or its function pointer.
#ifndef __HIP_DEVICE_COMPILE__
extern const ConvAlgorithm grouped_cdna5{is_applicable, kernel_groups};
#endif

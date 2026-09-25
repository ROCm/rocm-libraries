#include "algorithm.h"

#include <array>

using hipconv::Conv2dParams;
using hipconv::ConvAlgorithm;
using hipconv::ConvKernelSpan;

extern const ConvKernelSpan grouped_4c_cdna4_kernels;
extern const ConvKernelSpan grouped_4c_wgrad_cdna4_kernels;
extern const ConvKernelSpan grouped_4c_wgrad_tf32_cdna4_kernels;
extern const ConvKernelSpan grouped_8c_cdna4_kernels;
extern const ConvKernelSpan grouped_8c_wgrad_cdna4_kernels;
extern const ConvKernelSpan grouped_8c_wgrad_tf32_cdna4_kernels;
extern const ConvKernelSpan grouped_16c_cdna4_kernels;
extern const ConvKernelSpan grouped_16c_wgrad_cdna4_kernels;
extern const ConvKernelSpan grouped_16c_wgrad_tf32_cdna4_kernels;
extern const ConvKernelSpan grouped_32c_cdna4_kernels;
extern const ConvKernelSpan grouped_32c_wgrad_cdna4_kernels;
extern const ConvKernelSpan grouped_32c_wgrad_tf32_cdna4_kernels;

namespace
{

bool is_applicable(const Conv2dParams& par)
{
    if(par.groups == 1)
        return false;
    return par.dilation_h == 1 && par.dilation_w == 1;
}

constexpr std::array<const ConvKernelSpan*, 12> kernel_groups = {
    &grouped_32c_cdna4_kernels,
    &grouped_32c_wgrad_cdna4_kernels,
    &grouped_32c_wgrad_tf32_cdna4_kernels,
    &grouped_16c_cdna4_kernels,
    &grouped_16c_wgrad_cdna4_kernels,
    &grouped_16c_wgrad_tf32_cdna4_kernels,
    &grouped_8c_cdna4_kernels,
    &grouped_8c_wgrad_cdna4_kernels,
    &grouped_8c_wgrad_tf32_cdna4_kernels,
    &grouped_4c_cdna4_kernels,
    &grouped_4c_wgrad_cdna4_kernels,
    &grouped_4c_wgrad_tf32_cdna4_kernels,
};

} // namespace

// Host-only: the device linker must not see this object or its function pointer.
#ifndef __HIP_DEVICE_COMPILE__
extern const ConvAlgorithm grouped_cdna4{is_applicable, kernel_groups};
#endif

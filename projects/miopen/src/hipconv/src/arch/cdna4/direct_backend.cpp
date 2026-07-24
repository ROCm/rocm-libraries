#include "algorithm.h"
#include "direct/direct_fp16bf16.h"

#include <array>

using hipconv::Conv2dParams;
using hipconv::ConvAlgorithm;
using hipconv::ConvKernelSpan;

// Defined in its own TU (direct/direct_l1.cpp) so the large
// direct_l1 kernel compiles in parallel with the other arch kernels.
extern const ConvKernelSpan direct_l1_cdna4_kernels;

namespace
{

bool is_applicable(const Conv2dParams& par)
{
    return par.dilation_h == 1 && par.dilation_w == 1;
}

const ConvKernelSpan direct_cdna4_kernels{hipconv::cdna4::direct::kernel_ptrs};

constexpr std::array<const ConvKernelSpan*, 2> kernel_groups = {
    &direct_l1_cdna4_kernels,
    &direct_cdna4_kernels,
};

} // namespace

// Host-only: the device linker must not see this object or its function pointer.
#ifndef __HIP_DEVICE_COMPILE__
extern const ConvAlgorithm direct_cdna4{is_applicable, kernel_groups};
#endif

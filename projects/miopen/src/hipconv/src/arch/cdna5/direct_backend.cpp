#include "algorithm.h"
#include "direct/direct.h"

#include <array>

using hipconv::Conv2dParams;
using hipconv::ConvAlgorithm;
using hipconv::ConvKernelSpan;

namespace
{

bool is_applicable(const Conv2dParams& par)
{
    return par.dilation_h == 1 && par.dilation_w == 1;
}

const ConvKernelSpan direct_cdna5_kernels{hipconv::cdna5::direct::kernel_ptrs};

constexpr std::array<const ConvKernelSpan*, 1> kernel_groups = {
    &direct_cdna5_kernels,
};

} // namespace

// Host-only: the device linker must not see this object or its function pointer.
#ifndef __HIP_DEVICE_COMPILE__
extern const ConvAlgorithm direct_cdna5{is_applicable, kernel_groups};
#endif

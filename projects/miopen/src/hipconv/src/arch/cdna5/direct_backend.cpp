#include "algorithm.h"

#include <array>

using hipconv::ConvAlgorithm;
using hipconv::ConvKernelSpan;
using hipconv::ConvParams;

// Symbols defined by autoshard
extern const ConvKernelSpan direct_cdna5_kernels;

namespace
{

bool is_applicable(const ConvParams& par)
{
    return par.dilation_h == 1 && par.dilation_w == 1;
}

constexpr std::array<const ConvKernelSpan*, 1> kernel_groups = {
    &direct_cdna5_kernels,
};

} // namespace

// Host-only: the device linker must not see this object or its function pointer.
#ifndef __HIP_DEVICE_COMPILE__
extern const ConvAlgorithm direct_cdna5{is_applicable, kernel_groups};
#endif

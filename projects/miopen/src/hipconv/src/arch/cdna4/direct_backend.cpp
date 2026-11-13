#include "algorithm.h"

#include <array>

using hipconv::ConvAlgorithm;
using hipconv::ConvKernelGroup;
using hipconv::ConvParams;

// Defined by autoshard
extern const ConvKernelGroup direct_cdna4_kernels;
extern const ConvKernelGroup direct_l1_cdna4_kernels;
extern const ConvKernelGroup direct_wgrad_cdna4_kernels;
extern const ConvKernelGroup patch_embed_cdna4_kernels;
extern const ConvKernelGroup direct_flex_cdna4_kernels;

namespace
{

// Each family decides dilation itself: direct_flex supports any, the others none.
bool is_applicable(const ConvParams& /*par*/)
{
    return true;
}

constexpr std::array<const ConvKernelGroup*, 5> kernel_groups = {
    &direct_l1_cdna4_kernels,
    &direct_cdna4_kernels,
    &direct_wgrad_cdna4_kernels,
    &patch_embed_cdna4_kernels,
    &direct_flex_cdna4_kernels,
};

} // namespace

// Host-only: the device linker must not see this object or its function pointer.
#ifndef __HIP_DEVICE_COMPILE__
extern const ConvAlgorithm direct_cdna4{is_applicable, kernel_groups};
#endif

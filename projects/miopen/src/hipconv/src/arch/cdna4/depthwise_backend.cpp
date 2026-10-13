#include "algorithm.h"

#include <array>

using hipconv::ConvAlgorithm;
using hipconv::ConvKernelSpan;
using hipconv::ConvParams;

extern const ConvKernelSpan depthwise_1d_toeplitz_cdna4_kernels;
extern const ConvKernelSpan depthwise_wgrad_hankel_cdna4_kernels;
#ifdef HIPCONV_ENABLE_CDNA4_DW_2D_TOEPLITZ
extern const ConvKernelSpan depthwise_2d_toeplitz_cdna4_kernels;
#endif

namespace
{

bool is_applicable(const ConvParams& par)
{
    // Depthwise: one channel per group (groups == C == K).
    //
    // Distinguishes depthwise from dense (channels_per_group() != 1) and admits the
    // degenerate single-channel case (C == K == groups == 1).
    if(par.k != par.c || par.channels_per_group() != 1)
        return false;
    return par.dilation_h == 1 && par.dilation_w == 1;
}

// Groups in auto-pick order: 1D column-Toeplitz (fprop, dgrad), then Hankel wgrad.
//
// The 2D patch-Toeplitz variant never beats 1D, so it needs an opt-in macro and is last.
#ifdef HIPCONV_ENABLE_CDNA4_DW_2D_TOEPLITZ
constexpr std::array<const ConvKernelSpan*, 3> kernel_groups = {
    &depthwise_1d_toeplitz_cdna4_kernels,
    &depthwise_wgrad_hankel_cdna4_kernels,
    &depthwise_2d_toeplitz_cdna4_kernels,
};
#else
constexpr std::array<const ConvKernelSpan*, 2> kernel_groups = {
    &depthwise_1d_toeplitz_cdna4_kernels,
    &depthwise_wgrad_hankel_cdna4_kernels,
};
#endif

} // namespace

// Host-only: the device linker must not see this object or its function pointer.
#ifndef __HIP_DEVICE_COMPILE__
extern const ConvAlgorithm depthwise_cdna4{is_applicable, kernel_groups};
#endif

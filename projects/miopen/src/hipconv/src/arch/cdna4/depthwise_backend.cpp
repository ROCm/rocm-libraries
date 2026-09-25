#include "algorithm.h"

#include <array>

using hipconv::Conv2dParams;
using hipconv::ConvAlgorithm;
using hipconv::ConvKernelSpan;

extern const ConvKernelSpan depthwise_1d_toeplitz_cdna4_kernels;
#ifdef HIPCONV_ENABLE_CDNA4_DW_2D_TOEPLITZ
extern const ConvKernelSpan depthwise_2d_toeplitz_cdna4_kernels;
#endif

namespace
{

bool is_applicable(const Conv2dParams& par)
{
    // Depthwise: one channel per group (groups == C == K).
    //
    // Distinguishes depthwise from dense (channels_per_group() != 1) and admits the
    // degenerate single-channel case (C == K == groups == 1).
    if(par.k != par.c || par.channels_per_group() != 1)
        return false;
    return par.dilation_h == 1 && par.dilation_w == 1;
}

// 1D column-Toeplitz + row-streaming kernels: 3x3/5x5/7x7, stride 1 and 2, fprop and dgrad.
//
// dgrad is the rot180 correlation of dY; stride-2 dgrad runs it over a 2x-upsampled dY.
//
// The 2D patch-Toeplitz kernel is an experimental variant excluded from the build by
// default (it never beats the 1D kernel). Define HIPCONV_ENABLE_CDNA4_DW_2D_TOEPLITZ to
// append it here; kept after 1D so it stays out of the auto-pick front (valid_configs[0])
// but remains reachable via --config / the external tuner.
#ifdef HIPCONV_ENABLE_CDNA4_DW_2D_TOEPLITZ
constexpr std::array<const ConvKernelSpan*, 2> kernel_groups = {
    &depthwise_1d_toeplitz_cdna4_kernels,
    &depthwise_2d_toeplitz_cdna4_kernels,
};
#else
constexpr std::array<const ConvKernelSpan*, 1> kernel_groups = {
    &depthwise_1d_toeplitz_cdna4_kernels,
};
#endif

} // namespace

// Host-only: the device linker must not see this object or its function pointer.
#ifndef __HIP_DEVICE_COMPILE__
extern const ConvAlgorithm depthwise_cdna4{is_applicable, kernel_groups};
#endif

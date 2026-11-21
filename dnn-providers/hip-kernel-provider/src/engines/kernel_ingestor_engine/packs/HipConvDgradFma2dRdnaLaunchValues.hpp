// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <string_view>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>

#include "engines/kernel_ingestor_engine/packs/ConvRdnaCommon.hpp"

/**
 * @file HipConvDgradFma2dRdnaLaunchValues.hpp
 * @brief The launch state of `hipkernel:hip_conv_dgrad_fma_2d_rdna` and its saved form: a
 *        versioned dispatch name and the named values the launch is computed from.
 *
 * Use a new version for a change to the values, to the kernel arguments, or to how the grid
 * is derived; keep the handler of the old version registered while plans saved under it must
 * load.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// The tensor uids a matched conv_dgrad graph binds, in kernel argument order.
struct HipConvDgradFma2dRdnaBinding
{
    int64_t dy = 0;
    int64_t w = 0;
    int64_t dx = 0;
};

/// Everything a hip_conv_dgrad_fma_2d_rdna launch is computed from. The problem is in the
/// forward view: `in` is dx, `out` is dy; the height and width axes (slots 1 and 2) are used,
/// depth is 1.
struct HipConvDgradFma2dRdnaLaunchInputs
{
    HipConvDgradFma2dRdnaBinding binding;
    conv_rdna::ConvProblem problem;
};

inline constexpr std::string_view HIP_CONV_DGRAD_FMA_2D_RDNA_DISPATCH_SYMBOL_V1
    = "hipkernel.hip_conv_dgrad_fma_2d_rdna.dispatch.v1";

/// dx channels per workgroup (the kernel's TILE_N) for @p problem: the narrowest of 32 and 64
/// that holds one group's dx channels, 128 above that.
int64_t hipConvDgradFma2dRdnaTileN(const conv_rdna::ConvProblem& problem);

/// The values of the `hipkernel.hip_conv_dgrad_fma_2d_rdna.dispatch.v1` contract: `dy.uid`,
/// `w.uid`, `dx.uid`, `n`, `c`, `k`, `groups`, and per spatial axis (`_h`, `_w`) `in_`,
/// `filter_`, `out_`, `pad_`, `stride_`, `dilation_` (in the forward view: `in_` is dx, `out_`
/// is dy).
hipdnn_plugin_sdk::ingestor::MetadataValues
    hipConvDgradFma2dRdnaLaunchValues(const HipConvDgradFma2dRdnaBinding& binding,
                                      const conv_rdna::ConvProblem& problem);

/// Reads the values of the `hipkernel.hip_conv_dgrad_fma_2d_rdna.dispatch.v1` contract. Every
/// extent, stride, dilation and the group count must be in [1, INT_MAX], every padding in
/// [0, INT_MAX], C and K must split into the groups, the grid must fit the launch limits, and
/// the kernel's tap and K-iteration counts must fit in int.
/// Dtype, layout and the channel tile are not saved: they are compiled into the saved kernel.
///
/// @throws HipdnnPluginException with a load refusal when the dispatch name is not that
///         contract, when a value is missing, extra, of another type or out of range, or when
///         the groups do not divide C and K, the grid exceeds the launch limits or the kernel's
///         counts would not fit in int.
HipConvDgradFma2dRdnaLaunchInputs readHipConvDgradFma2dRdnaLaunchInputs(
    const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs);

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

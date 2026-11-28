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
 * @file HipConvDgradFma1dRdnaLaunchValues.hpp
 * @brief The variant geometry and launch state of `hipkernel:hip_conv_dgrad_fma_1d_rdna` and
 *        its saved form: a versioned dispatch name and the named values the launch is computed
 *        from.
 *
 * Use a new version for a change to the values, to the kernel arguments, or to how the grid
 * is derived; keep the handler of the old version registered while plans saved under it must
 * load.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// How a 1-D backward-data variant tiles dx; the grid depends on it. Dtype, layout, the
/// specialized filter width and stride are compiled into the variant and only matter for
/// picking it.
struct HipConvDgradFma1dRdnaGeometry
{
    /// The runtime-parameter kernel (filter, padding, stride, dilation and groups as launch
    /// arguments).
    bool generic = false;
    /// Output phases per lane of the specialized kernel (CT_QR); 1 for the generic kernel.
    int qPerLane = 1;
    /// dx channels per wave of the specialized kernel (CT_CO_W); 1 for the generic kernel.
    int channelsPerWave = 1;
};

/// The tensor uids a matched conv_dgrad graph binds, in kernel argument order.
struct HipConvDgradFma1dRdnaBinding
{
    int64_t dy = 0;
    int64_t w = 0;
    int64_t dx = 0;
};

/// Everything a hip_conv_dgrad_fma_1d_rdna launch is computed from. The problem is in the
/// forward view: `in` is dx, `out` is dy; only the width axis (slot 2) is used.
struct HipConvDgradFma1dRdnaLaunchInputs
{
    HipConvDgradFma1dRdnaBinding binding;
    conv_rdna::ConvProblem problem;
    HipConvDgradFma1dRdnaGeometry geometry;
};

inline constexpr std::string_view HIP_CONV_DGRAD_FMA_1D_RDNA_DISPATCH_SYMBOL_V1
    = "hipkernel.hip_conv_dgrad_fma_1d_rdna.dispatch.v1";

/// The variant @p problem takes: the specialized one when it is NCL with groups == 1,
/// dilation 1 and a filter width and stride among the built specializations (with the channel
/// tile its dx channel count gives), the generic one otherwise.
HipConvDgradFma1dRdnaGeometry hipConvDgradFma1dRdnaGeometry(const conv_rdna::ConvProblem& problem);

/// The values of the `hipkernel.hip_conv_dgrad_fma_1d_rdna.dispatch.v1` contract: `dy.uid`,
/// `w.uid`, `dx.uid`, `n`, `c`, `k`, `groups`, the width axis `in_w`, `filter_w`, `out_w`,
/// `pad_w`, `stride_w`, `dilation_w` (in the forward view: `in_w` is the dx length, `out_w` the
/// dy length), and the geometry `generic`, `q_per_lane`, `channels_per_wave`.
hipdnn_plugin_sdk::ingestor::MetadataValues
    hipConvDgradFma1dRdnaLaunchValues(const HipConvDgradFma1dRdnaBinding& binding,
                                      const conv_rdna::ConvProblem& problem,
                                      const HipConvDgradFma1dRdnaGeometry& geometry);

/// Reads the values of the `hipkernel.hip_conv_dgrad_fma_1d_rdna.dispatch.v1` contract. Every
/// extent, stride, dilation and the group count must be in [1, INT_MAX], the padding in
/// [0, INT_MAX], C and K multiples of the group count, and the geometry one the engine builds
/// within the launch limits.
///
/// @throws HipdnnPluginException with a load refusal when the dispatch name is not that
///         contract, or when a value is missing, extra, of another type or out of range.
HipConvDgradFma1dRdnaLaunchInputs readHipConvDgradFma1dRdnaLaunchInputs(
    const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs);

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

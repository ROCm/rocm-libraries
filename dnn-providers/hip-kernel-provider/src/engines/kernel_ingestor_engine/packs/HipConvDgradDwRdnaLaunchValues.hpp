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
 * @file HipConvDgradDwRdnaLaunchValues.hpp
 * @brief The variant geometry and launch state of `hipkernel:hip_conv_dgrad_dw_rdna` and its
 *        saved form: a versioned dispatch name and the named values the launch is computed
 *        from.
 *
 * Use a new version for a change to the values, to the kernel arguments, or to how the grid
 * is derived; keep the handler of the old version registered while plans saved under it must
 * load.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// How a depthwise backward-data variant walks dx; the grid depends on it. Dtype, layout, the
/// specialized filter width and stride are compiled into the variant and only matter for
/// picking it.
struct HipConvDgradDwRdnaGeometry
{
    /// The runtime-parameter kernel (filter, padding, stride and dilation as launch arguments).
    bool generic = false;
    /// Output phases per lane of the specialized kernel (DW_TW); 1 for the generic kernel.
    int tileQ = 1;
};

/// The tensor uids a matched depthwise conv_dgrad graph binds, in kernel argument order.
struct HipConvDgradDwRdnaBinding
{
    int64_t dy = 0;
    int64_t w = 0;
    int64_t dx = 0;
};

/// Everything a hip_conv_dgrad_dw_rdna launch is computed from. The problem is in the forward
/// view: `in` is dx, `out` is dy.
struct HipConvDgradDwRdnaLaunchInputs
{
    HipConvDgradDwRdnaBinding binding;
    conv_rdna::ConvProblem problem;
    HipConvDgradDwRdnaGeometry geometry;
};

inline constexpr std::string_view HIP_CONV_DGRAD_DW_RDNA_DISPATCH_SYMBOL_V1
    = "hipkernel.hip_conv_dgrad_dw_rdna.dispatch.v1";

/// The variant @p problem takes: the specialized one when it is a 1-D NCL problem with K == C,
/// dilation 1 and a filter width and stride among the built specializations, the generic one
/// otherwise.
HipConvDgradDwRdnaGeometry hipConvDgradDwRdnaGeometry(const conv_rdna::ConvProblem& problem);

/// The values of the `hipkernel.hip_conv_dgrad_dw_rdna.dispatch.v1` contract: `dy.uid`,
/// `w.uid`, `dx.uid`, `n`, `c`, `k`, per spatial axis (`_d`, `_h`, `_w`) `in_`, `filter_`,
/// `out_`, `pad_`, `stride_`, `dilation_` (in the forward view: `in_` is dx, `out_` is dy), and
/// the geometry `generic`, `tile_q`.
hipdnn_plugin_sdk::ingestor::MetadataValues
    hipConvDgradDwRdnaLaunchValues(const HipConvDgradDwRdnaBinding& binding,
                                   const conv_rdna::ConvProblem& problem,
                                   const HipConvDgradDwRdnaGeometry& geometry);

/// Reads the values of the `hipkernel.hip_conv_dgrad_dw_rdna.dispatch.v1` contract. Every
/// extent, stride and dilation must be in [1, INT_MAX], every padding in [0, INT_MAX], K a
/// multiple of C, and the geometry one the engine builds within the launch limits.
///
/// @throws HipdnnPluginException with a load refusal when the dispatch name is not that
///         contract, or when a value is missing, extra, of another type or out of range.
HipConvDgradDwRdnaLaunchInputs readHipConvDgradDwRdnaLaunchInputs(
    const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs);

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

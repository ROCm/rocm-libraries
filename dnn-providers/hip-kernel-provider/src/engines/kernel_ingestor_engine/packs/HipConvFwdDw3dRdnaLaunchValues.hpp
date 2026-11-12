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
 * @file HipConvFwdDw3dRdnaLaunchValues.hpp
 * @brief The variant geometry and launch state of `hipkernel:hip_conv_fwd_dw3d_rdna` and its
 *        saved form: a versioned dispatch name and the named values the launch is computed
 *        from.
 *
 * Use a new version for a change to the values, to the kernel arguments, or to how the grid
 * is derived; keep the handler of the old version registered while plans saved under it must
 * load.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// How a depthwise 3D variant tiles the output; the grid depends on it. Dtype, layout, the
/// specialized filter, H / W stride and dilation are compiled into the variant and only matter
/// for picking it.
struct HipConvFwdDw3dRdnaGeometry
{
    /// The runtime-parameter kernel (filter, stride and dilation as launch arguments).
    bool generic = false;
    /// Output depths and columns per thread (the specialized kernel; the generic one is 1 x 4).
    int tileD = 1;
    int tileW = 4;
    /// NCDHW grid with one (n, k) output volume per blockIdx.y and 64-thread blocks.
    bool uniformK = false;
};

/// The tensor uids a matched depthwise conv graph binds, in kernel argument order.
struct HipConvFwdDw3dRdnaBinding
{
    int64_t x = 0;
    int64_t w = 0;
    int64_t y = 0;
};

/// Everything a hip_conv_fwd_dw3d_rdna launch is computed from.
struct HipConvFwdDw3dRdnaLaunchInputs
{
    HipConvFwdDw3dRdnaBinding binding;
    conv_rdna::ConvProblem problem;
    HipConvFwdDw3dRdnaGeometry geometry;
};

inline constexpr std::string_view HIP_CONV_FWD_DW3D_RDNA_DISPATCH_SYMBOL_V1
    = "hipkernel.hip_conv_fwd_dw3d_rdna.dispatch.v1";

/// The variant @p problem takes: the specialized one when its filter, H / W stride and dilation
/// are among the built specializations and its grid fits, the generic one otherwise.
HipConvFwdDw3dRdnaGeometry hipConvFwdDw3dRdnaGeometry(const conv_rdna::ConvProblem& problem);

/// The values of the `hipkernel.hip_conv_fwd_dw3d_rdna.dispatch.v1` contract: `x.uid`, `w.uid`,
/// `y.uid`, `n`, `c`, `k`, per spatial axis (`_d`, `_h`, `_w`) `in_`, `filter_`, `out_`, `pad_`,
/// `stride_`, `dilation_`, and the geometry `generic`, `tile_d`, `tile_w`, `uniform_k`.
hipdnn_plugin_sdk::ingestor::MetadataValues
    hipConvFwdDw3dRdnaLaunchValues(const HipConvFwdDw3dRdnaBinding& binding,
                                   const conv_rdna::ConvProblem& problem,
                                   const HipConvFwdDw3dRdnaGeometry& geometry);

/// Reads the values of the `hipkernel.hip_conv_fwd_dw3d_rdna.dispatch.v1` contract. Every
/// extent, stride and dilation must be in [1, INT_MAX], every padding in [0, INT_MAX], K a
/// multiple of C, the geometry one the engine builds, and the grid within the launch limits.
///
/// @throws HipdnnPluginException with a load refusal when the dispatch name is not that
///         contract, or when a value is missing, extra, of another type or out of range.
HipConvFwdDw3dRdnaLaunchInputs readHipConvFwdDw3dRdnaLaunchInputs(
    const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs);

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

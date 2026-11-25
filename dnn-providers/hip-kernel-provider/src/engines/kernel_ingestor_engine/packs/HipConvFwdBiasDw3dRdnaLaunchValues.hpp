// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <string_view>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>

#include "engines/kernel_ingestor_engine/packs/ConvRdnaCommon.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvRdnaDw3d.hpp"

/**
 * @file HipConvFwdBiasDw3dRdnaLaunchValues.hpp
 * @brief The launch state of `hipkernel:hip_conv_fwd_bias_dw3d_rdna` and its saved form: a
 *        versioned dispatch name and the named values the launch is computed from.
 *
 * Use a new version for a change to the values, to the kernel arguments, or to how the grid
 * is derived; keep the handler of the old version registered while plans saved under it must
 * load.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// The tensor uids a matched depthwise conv + bias graph binds: the convolution input and
/// weights, the bias, and the output of the ADD (the convolution output is virtual and never
/// stored).
struct HipConvFwdBiasDw3dRdnaBinding
{
    int64_t x = 0;
    int64_t w = 0;
    int64_t b = 0;
    int64_t y = 0;
};

/// Everything a hip_conv_fwd_bias_dw3d_rdna launch is computed from.
struct HipConvFwdBiasDw3dRdnaLaunchInputs
{
    HipConvFwdBiasDw3dRdnaBinding binding;
    conv_rdna::ConvProblem problem;
    conv_rdna::dw3d::Geometry geometry;
};

inline constexpr std::string_view HIP_CONV_FWD_BIAS_DW3D_RDNA_DISPATCH_SYMBOL_V1
    = "hipkernel.hip_conv_fwd_bias_dw3d_rdna.dispatch.v1";

/// The values of the `hipkernel.hip_conv_fwd_bias_dw3d_rdna.dispatch.v1` contract: `x.uid`,
/// `w.uid`, `b.uid`, `y.uid` and the problem and geometry values of
/// conv_rdna::dw3d::addProblemLaunchValues.
hipdnn_plugin_sdk::ingestor::MetadataValues
    hipConvFwdBiasDw3dRdnaLaunchValues(const HipConvFwdBiasDw3dRdnaBinding& binding,
                                       const conv_rdna::ConvProblem& problem,
                                       const conv_rdna::dw3d::Geometry& geometry);

/// Reads the values of the `hipkernel.hip_conv_fwd_bias_dw3d_rdna.dispatch.v1` contract, with
/// the checks of conv_rdna::dw3d::readProblemLaunchValues.
///
/// @throws HipdnnPluginException with a load refusal when the dispatch name is not that
///         contract, or when a value is missing, extra, of another type or out of range.
HipConvFwdBiasDw3dRdnaLaunchInputs readHipConvFwdBiasDw3dRdnaLaunchInputs(
    const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs);

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

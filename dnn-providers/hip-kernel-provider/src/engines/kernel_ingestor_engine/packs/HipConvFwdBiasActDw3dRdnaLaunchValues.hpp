// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <optional>
#include <string_view>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>

#include "engines/kernel_ingestor_engine/packs/ConvRdnaCommon.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvRdnaDw3d.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvRdnaFusedEpilogue.hpp"

/**
 * @file HipConvFwdBiasActDw3dRdnaLaunchValues.hpp
 * @brief The launch state of `hipkernel:hip_conv_fwd_bias_act_dw3d_rdna` and its saved form: a
 *        versioned dispatch name and the named values the launch is computed from.
 *
 * Use a new version for a change to the values, to the kernel arguments, or to how the grid
 * is derived; keep the handler of the old version registered while plans saved under it must
 * load.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// The tensor uids a matched depthwise conv + (bias +) activation graph binds: the convolution
/// input and weights, the bias when the graph has one, and the output of the activation (the
/// convolution and ADD outputs are virtual and never stored).
struct HipConvFwdBiasActDw3dRdnaBinding
{
    int64_t x = 0;
    int64_t w = 0;
    std::optional<int64_t> b;
    int64_t y = 0;
};

/// Everything a hip_conv_fwd_bias_act_dw3d_rdna launch is computed from.
struct HipConvFwdBiasActDw3dRdnaLaunchInputs
{
    HipConvFwdBiasActDw3dRdnaBinding binding;
    conv_rdna::ConvProblem problem;
    conv_rdna::dw3d::Geometry geometry;
    conv_rdna::Activation activation;
};

inline constexpr std::string_view HIP_CONV_FWD_BIAS_ACT_DW3D_RDNA_DISPATCH_SYMBOL_V1
    = "hipkernel.hip_conv_fwd_bias_act_dw3d_rdna.dispatch.v1";

/// The values of the `hipkernel.hip_conv_fwd_bias_act_dw3d_rdna.dispatch.v1` contract: `x.uid`,
/// `w.uid`, `b.uid` (only when the graph has a bias), `y.uid`, the problem and geometry values
/// of conv_rdna::dw3d::addProblemLaunchValues, `activation` (the hipDNN pointwise mode name,
/// e.g. `RELU_FWD`) and `activation.p0` to `activation.p2` (conv_rdna::Activation).
hipdnn_plugin_sdk::ingestor::MetadataValues
    hipConvFwdBiasActDw3dRdnaLaunchValues(const HipConvFwdBiasActDw3dRdnaBinding& binding,
                                          const conv_rdna::ConvProblem& problem,
                                          const conv_rdna::dw3d::Geometry& geometry,
                                          const conv_rdna::Activation& activation);

/// Reads the values of the `hipkernel.hip_conv_fwd_bias_act_dw3d_rdna.dispatch.v1` contract,
/// with the checks of conv_rdna::dw3d::readProblemLaunchValues.
///
/// @throws HipdnnPluginException with a load refusal when the dispatch name is not that
///         contract, or when a value is missing, extra, of another type or out of range,
///         including an activation the engine does not serve and a parameter that is not a
///         finite float.
HipConvFwdBiasActDw3dRdnaLaunchInputs readHipConvFwdBiasActDw3dRdnaLaunchInputs(
    const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs);

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

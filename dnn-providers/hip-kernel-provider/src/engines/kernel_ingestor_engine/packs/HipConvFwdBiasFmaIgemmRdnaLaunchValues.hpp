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
 * @file HipConvFwdBiasFmaIgemmRdnaLaunchValues.hpp
 * @brief The launch state of `hipkernel:hip_conv_fwd_bias_fma_igemm_rdna` and its saved form:
 *        a versioned dispatch name and the named values the launch is computed from.
 *
 * Use a new version for a change to the values, to the kernel arguments, or to how the grid
 * is derived; keep the handler of the old version registered while plans saved under it must
 * load.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// The tensor uids a matched conv + bias graph binds: the convolution input and weights, the
/// bias, and the output of the ADD (the convolution output is virtual and never stored).
struct HipConvFwdBiasFmaIgemmRdnaBinding
{
    int64_t x = 0;
    int64_t w = 0;
    int64_t b = 0;
    int64_t y = 0;
};

/// Everything a hip_conv_fwd_bias_fma_igemm_rdna launch is computed from.
struct HipConvFwdBiasFmaIgemmRdnaLaunchInputs
{
    HipConvFwdBiasFmaIgemmRdnaBinding binding;
    conv_rdna::ConvProblem problem;
};

inline constexpr std::string_view HIP_CONV_FWD_BIAS_FMA_IGEMM_RDNA_DISPATCH_SYMBOL_V1
    = "hipkernel.hip_conv_fwd_bias_fma_igemm_rdna.dispatch.v1";

/// The values of the `hipkernel.hip_conv_fwd_bias_fma_igemm_rdna.dispatch.v1` contract:
/// `x.uid`, `w.uid`, `b.uid`, `y.uid`, then the problem values of
/// conv_rdna::fma_igemm::addProblemLaunchValues.
hipdnn_plugin_sdk::ingestor::MetadataValues
    hipConvFwdBiasFmaIgemmRdnaLaunchValues(const HipConvFwdBiasFmaIgemmRdnaBinding& binding,
                                           const conv_rdna::ConvProblem& problem);

/// Reads the values of the `hipkernel.hip_conv_fwd_bias_fma_igemm_rdna.dispatch.v1` contract,
/// with the checks of conv_rdna::fma_igemm::readProblemLaunchValues.
///
/// @throws HipdnnPluginException with a load refusal when the dispatch name is not that
///         contract, or when a value is missing, extra, of another type or out of range.
HipConvFwdBiasFmaIgemmRdnaLaunchInputs readHipConvFwdBiasFmaIgemmRdnaLaunchInputs(
    const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs);

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

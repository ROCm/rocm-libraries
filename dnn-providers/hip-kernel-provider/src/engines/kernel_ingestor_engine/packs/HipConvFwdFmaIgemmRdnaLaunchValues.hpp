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
 * @file HipConvFwdFmaIgemmRdnaLaunchValues.hpp
 * @brief The launch state of `hipkernel:hip_conv_fwd_fma_igemm_rdna` and its saved form: a versioned
 *        dispatch name and the named values the launch is computed from.
 *
 * Use a new version for a change to the values, to the kernel arguments, or to how the grid
 * is derived; keep the handler of the old version registered while plans saved under it must
 * load.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// The tensor uids a matched conv graph binds, in kernel argument order.
struct HipConvFwdFmaIgemmRdnaBinding
{
    int64_t x = 0;
    int64_t w = 0;
    int64_t y = 0;
};

/// Everything a hip_conv_fwd_fma_igemm_rdna launch is computed from.
struct HipConvFwdFmaIgemmRdnaLaunchInputs
{
    HipConvFwdFmaIgemmRdnaBinding binding;
    conv_rdna::ConvProblem problem;
};

inline constexpr std::string_view HIP_CONV_FWD_FMA_IGEMM_RDNA_DISPATCH_SYMBOL_V1
    = "hipkernel.hip_conv_fwd_fma_igemm_rdna.dispatch.v1";

/// The values of the `hipkernel.hip_conv_fwd_fma_igemm_rdna.dispatch.v1` contract: `x.uid`, `w.uid`,
/// `y.uid`, `n`, `c`, `k`, `groups`, and per spatial axis (`_d`, `_h`, `_w`; a 2-D plan saves
/// depth 1) `in_`, `filter_`, `out_`, `pad_`, `stride_`, `dilation_`.
hipdnn_plugin_sdk::ingestor::MetadataValues
    hipConvFwdFmaIgemmRdnaLaunchValues(const HipConvFwdFmaIgemmRdnaBinding& binding,
                                       const conv_rdna::ConvProblem& problem);

/// Reads the values of the `hipkernel.hip_conv_fwd_fma_igemm_rdna.dispatch.v1` contract. Every extent,
/// stride and dilation must be in [1, INT_MAX], every padding in [0, INT_MAX], C and K must
/// split into the groups, and the grid must fit the launch limits. Dtype and layout are not
/// saved: they are compiled into the saved kernel.
///
/// @throws HipdnnPluginException with a load refusal when the dispatch name is not that
///         contract, or when a value is missing, extra, of another type or out of range.
HipConvFwdFmaIgemmRdnaLaunchInputs readHipConvFwdFmaIgemmRdnaLaunchInputs(
    const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs);

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

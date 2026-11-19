// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipblaslt_plugin
{

/// Opts in to HIPBLASLT_CONV_LIBRARY_ACTIVATION_ENGINE when set to 1.
inline constexpr const char* CONV_LIBRARY_ACTIVATIONS_ENV_NAME
    = "HIPBLASLT_PROVIDER_CONV_LIBRARY_ACTIVATIONS";

/**
 * @brief Whether HIPBLASLT_PROVIDER_CONV_LIBRARY_ACTIVATIONS is 1: the provider then offers
 *        HIPBLASLT_CONV_LIBRARY_ACTIVATION_ENGINE, the explicit-GEMM convolution whose fused
 *        relu, clamp, swish (beta 1), tanh GELU and fp32 sigmoid run in hipBLASLt's epilogue
 *        (GemmEpilogueSupport::libraryActivations).
 *
 * Opt-in because those epilogues do not give the result every other engine gives for special
 * values: relu maps NaN to 0, clamp maps NaN to its upper bound, swish and tanh GELU map -inf to
 * NaN. No numerical note says so, so a caller could not otherwise exclude the engine.
 */
inline bool convLibraryActivationsEnabled()
{
    using namespace hipdnn_data_sdk::utilities;
    return trim(getEnv(CONV_LIBRARY_ACTIVATIONS_ENV_NAME)) == "1";
}

} // namespace hipblaslt_plugin

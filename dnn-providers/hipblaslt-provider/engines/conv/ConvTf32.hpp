// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipblaslt_plugin
{

/// The temporary switch for the TF32 explicit-GEMM convolution engines.
inline constexpr const char* CONV_TF32_ENV_NAME = "HIPBLASLT_PROVIDER_CONV_TF32";

/**
 * @brief Whether the provider registers its TF32 explicit-GEMM convolution engines
 *        (HIPBLASLT_CONV_TF32_ENGINE, ROCBLAS_CONV_TF32_ENGINE): HIPBLASLT_PROVIDER_CONV_TF32 set
 *        to 1, true, on or yes.
 *
 * Temporary. hipDNN has no way yet for a graph to permit TF32, or to exclude engines that use
 * it, so registering the engines unconditionally would let autotuning pick TF32 for any fp32
 * convolution. Behind the switch, autotuning times them beside the fp32 engines. Once hipDNN
 * carries numerical notes, the engines report DOWN_CONVERT_INPUTS instead and the switch goes.
 */
inline bool convTf32EnginesEnabled()
{
    using namespace hipdnn_data_sdk::utilities;
    const std::string value = toLower(trim(getEnv(CONV_TF32_ENV_NAME)));
    return value == "1" || value == "true" || value == "on" || value == "yes";
}

} // namespace hipblaslt_plugin

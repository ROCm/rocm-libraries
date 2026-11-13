// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipblaslt_plugin
{

/// The temporary switch for the reduced-precision-reduction convolution engines, shared by every
/// provider that has one.
inline constexpr const char* CONV_RPR_ENV_NAME = "HIPDNN_CONV_ALLOW_REDUCED_PRECISION_REDUCTION";

/**
 * @brief Whether reduced-precision-reduction convolution engines (ROCBLAS_CONV_RPR_ENGINE,
 *        HIPBLASLT_CONV_RPR_ENGINE) are registered and applicable:
 *        HIPDNN_CONV_ALLOW_REDUCED_PRECISION_REDUCTION set to 1, true, on or yes.
 *
 * Temporary. These engines round partial sums to fp16 (cuDNN's REDUCED_PRECISION_REDUCTION), and
 * hipDNN has no per-engine numerical notes yet to report it or let a caller exclude them, so
 * registering them unconditionally would let autotuning pick one for any fp16 convolution. Once
 * notes exist, the engines report REDUCED_PRECISION_REDUCTION, are offered under FLOAT or HALF
 * compute as cuDNN offers them, and the switch goes.
 */
inline bool convReducedPrecisionEnginesEnabled()
{
    using namespace hipdnn_data_sdk::utilities;
    const std::string value = toLower(trim(getEnv(CONV_RPR_ENV_NAME)));
    return value == "1" || value == "true" || value == "on" || value == "yes";
}

} // namespace hipblaslt_plugin

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
 * Temporary. hipDNN reports per-engine numerical notes (RFC 0025), but a caller cannot yet
 * exclude an engine by a precision note: the native and Python frontends have no
 * select_numeric_notes / deselect_numeric_notes, and the cuDNN v9 compatibility layer filters
 * on NONDETERMINISTIC only (deselecting DOWN_CONVERT_INPUTS fails with GRAPH_NOT_SUPPORTED). So
 * a graph has no way to permit TF32 or to keep it out, and registering the engines
 * unconditionally would let autotuning pick TF32 for any fp32 convolution. Behind the switch,
 * autotuning times them beside the fp32 engines. The engines report no numerical notes, so
 * hipDNN reports them NONDETERMINISTIC. Follow-up: once a caller can exclude engines by a
 * precision note, the engines report it (hip-kernel-provider/conv/README.md, "Lower-precision
 * plans") and the switch goes.
 */
inline bool convTf32EnginesEnabled()
{
    using namespace hipdnn_data_sdk::utilities;
    const std::string value = toLower(trim(getEnv(CONV_TF32_ENV_NAME)));
    return value == "1" || value == "true" || value == "on" || value == "yes";
}

} // namespace hipblaslt_plugin

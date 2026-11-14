// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <string>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

#include "common/ConvRepackFormulation.hpp"

namespace hipblaslt_plugin
{

/// Overrides the explicit-GEMM convolution engines' repack workspace cap, in bytes.
inline constexpr const char* CONV_REPACK_WORKSPACE_CAP_ENV_NAME
    = "HIPBLASLT_PROVIDER_CONV_REPACK_WORKSPACE_CAP";

/// The smallest override honored: below it, nearly every im2col plan would be banded.
inline constexpr size_t MIN_REPACK_WORKSPACE_CAP_BYTES = size_t{1} << 20;

/**
 * @brief The repack workspace cap the explicit-GEMM convolution engines register with:
 *        HIPBLASLT_PROVIDER_CONV_REPACK_WORKSPACE_CAP (decimal bytes, at least 1 MiB) when set,
 *        otherwise MAX_REPACK_WORKSPACE_BYTES. Read once, when the engines register.
 *
 * For measuring what the cap costs: a larger cap slices fewer images per stage.
 */
inline size_t repackWorkspaceCapBytes()
{
    using namespace hipdnn_data_sdk::utilities;
    const std::string value = trim(getEnv(CONV_REPACK_WORKSPACE_CAP_ENV_NAME));
    if(value.empty())
    {
        return MAX_REPACK_WORKSPACE_BYTES;
    }
    char* end = nullptr;
    errno = 0;
    const unsigned long long bytes = std::strtoull(value.c_str(), &end, 10);
    if(errno != 0 || end == value.c_str() || *end != '\0' || bytes < MIN_REPACK_WORKSPACE_CAP_BYTES)
    {
        return MAX_REPACK_WORKSPACE_BYTES;
    }
    return static_cast<size_t>(bytes);
}

} // namespace hipblaslt_plugin

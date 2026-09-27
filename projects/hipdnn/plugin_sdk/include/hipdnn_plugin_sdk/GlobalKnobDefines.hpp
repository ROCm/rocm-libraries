// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <optional>
#include <string>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace hipdnn_plugin_sdk
{
static constexpr const char* BENCHMARKING_KNOB_NAME = "global.benchmarking";
static constexpr const char* WORKSPACE_SIZE_LIMIT_KNOB_NAME = "global.workspace_size_limit";

/// Process-wide override for `global.benchmarking`, independent of any engine's knob.
/// Not gated on HIPDNN_ENABLE_KERNEL_INGESTOR: MIOpen consumes it in flag-off builds too.
static constexpr const char* FORCE_BENCHMARKING_ENV_NAME = "HIPDNN_FORCE_BENCHMARKING";

/// Reads and parses HIPDNN_FORCE_BENCHMARKING per call, never cached. Normalizes with
/// toLower(trim(...)) and matches a literal set, following
/// hipdnn_data_sdk::logging::detail::stringToSeverity(). Unset, empty, whitespace-only,
/// or any unrecognized value returns std::nullopt.
///
/// Unset and empty are indistinguishable -- getEnv() yields "" for both -- and are
/// silent. Any other value that fails to parse logs a WARN naming the variable and the
/// value.
inline std::optional<bool> benchmarkingOverrideFromEnv()
{
    const auto raw = hipdnn_data_sdk::utilities::getEnv(FORCE_BENCHMARKING_ENV_NAME);
    const std::string normalized
        = hipdnn_data_sdk::utilities::toLower(hipdnn_data_sdk::utilities::trim(raw));

    if(raw.empty())
    {
        return std::nullopt;
    }
    if(normalized == "1" || normalized == "true" || normalized == "on" || normalized == "yes"
       || normalized == "enable" || normalized == "enabled")
    {
        return true;
    }
    if(normalized == "0" || normalized == "false" || normalized == "off" || normalized == "no"
       || normalized == "disable" || normalized == "disabled")
    {
        return false;
    }

    HIPDNN_PLUGIN_LOG_WARN("ignoring unrecognized value for " << FORCE_BENCHMARKING_ENV_NAME
                                                              << ": '" << raw << "'");
    return std::nullopt;
}
}

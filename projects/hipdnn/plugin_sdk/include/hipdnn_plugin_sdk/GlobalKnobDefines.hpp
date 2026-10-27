// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <optional>
#include <string>

#include <hipdnn_data_sdk/utilities/EnvBool.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace hipdnn_plugin_sdk
{
static constexpr const char* BENCHMARKING_KNOB_NAME = "global.benchmarking";
static constexpr const char* WORKSPACE_SIZE_LIMIT_KNOB_NAME = "global.workspace_size_limit";

/// Process-wide override for `global.benchmarking`, independent of any engine's knob.
/// Not gated on HIPDNN_ENABLE_KERNEL_INGESTOR: MIOpen consumes it in flag-off builds too.
static constexpr const char* FORCE_BENCHMARKING_ENV_NAME = "HIPDNN_FORCE_BENCHMARKING";

/// Reads and parses HIPDNN_FORCE_BENCHMARKING per call, never cached, with the token ladder
/// of hipdnn_data_sdk::utilities::parseBoolToken(). Unset, empty, whitespace-only, or any
/// unrecognized value returns std::nullopt.
///
/// Unset and empty are indistinguishable -- getEnv() yields "" for both -- and are
/// silent. Any other value that fails to parse logs a WARN naming the variable and the
/// value.
inline std::optional<bool> benchmarkingOverrideFromEnv()
{
    const auto parsed = hipdnn_data_sdk::utilities::readBoolEnv(FORCE_BENCHMARKING_ENV_NAME);
    if(parsed.unrecognized)
    {
        HIPDNN_PLUGIN_LOG_WARN("ignoring unrecognized value for " << FORCE_BENCHMARKING_ENV_NAME
                                                                  << ": '" << parsed.raw << "'");
    }
    return parsed.value;
}

/// Opt-in switch for validating each benchmark candidate's output before a winner is
/// accepted. Not gated on HIPDNN_ENABLE_KERNEL_INGESTOR, like the override above.
static constexpr const char* VALIDATE_BENCHMARKING_ENV_NAME = "HIPDNN_VALIDATE_BENCHMARKING";

/// Reads HIPDNN_VALIDATE_BENCHMARKING per call, never cached, with the token ladder of
/// hipdnn_data_sdk::utilities::parseBoolToken(). Unset and empty return false silently; an
/// unrecognized value logs a WARN naming the variable and the value and returns false.
inline bool validateBenchmarkingFromEnv()
{
    const auto parsed = hipdnn_data_sdk::utilities::readBoolEnv(VALIDATE_BENCHMARKING_ENV_NAME);
    if(parsed.unrecognized)
    {
        HIPDNN_PLUGIN_LOG_WARN("ignoring unrecognized value for " << VALIDATE_BENCHMARKING_ENV_NAME
                                                                  << ": '" << parsed.raw << "'");
    }
    return parsed.value.value_or(false);
}
}

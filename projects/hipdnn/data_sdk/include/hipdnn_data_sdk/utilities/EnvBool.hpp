// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// EnvBool is the one token ladder every hipDNN boolean environment variable shares
// (HIPDNN_FORCE_BENCHMARKING, HIPDNN_VALIDATE_BENCHMARKING, HIPDNN_DISABLE_CACHE).

#include <optional>
#include <string>
#include <string_view>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

namespace hipdnn_data_sdk::utilities
{

/// The outcome of reading one boolean token.
struct EnvBool
{
    /// The parsed value; nullopt when the token is empty or unrecognized.
    std::optional<bool> value;
    /// True only for a non-empty token outside both accepted sets.
    bool unrecognized = false;
    /// The untrimmed token, for the caller's own diagnostics.
    std::string raw;
};

/// Trims and lower-cases @p raw, then matches it against a literal set: true for
/// {1, true, on, yes, enable, enabled}, false for {0, false, off, no, disable, disabled}.
/// Empty and whitespace-only tokens are neither value nor unrecognized.
inline EnvBool parseBoolToken(std::string_view raw)
{
    EnvBool result;
    result.raw = std::string(raw);
    const std::string normalized = toLower(trim(result.raw));
    if(normalized.empty())
    {
        return result;
    }
    if(normalized == "1" || normalized == "true" || normalized == "on" || normalized == "yes"
       || normalized == "enable" || normalized == "enabled")
    {
        result.value = true;
    }
    else if(normalized == "0" || normalized == "false" || normalized == "off" || normalized == "no"
            || normalized == "disable" || normalized == "disabled")
    {
        result.value = false;
    }
    else
    {
        result.unrecognized = true;
    }
    return result;
}

/// parseBoolToken() over the environment variable @p name, read per call. Unset and empty
/// are indistinguishable (getEnv() yields "" for both).
inline EnvBool readBoolEnv(const char* name)
{
    return parseBoolToken(getEnv(name));
}

} // namespace hipdnn_data_sdk::utilities

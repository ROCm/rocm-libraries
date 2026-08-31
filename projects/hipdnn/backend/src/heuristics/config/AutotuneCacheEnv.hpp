// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

#include <string>

namespace hipdnn_backend::heuristics::config
{

/// Reads HIPDNN_DISABLE_EXACT_ENGINE_CACHE and returns whether the exact-match autotune
/// cache is disabled. Explicit falsy values (`0`, `false`, `off`) leave the cache enabled;
/// only a recognized truthy value disables it.
inline bool exactCacheDisabled()
{
    static constexpr const char* ENV_VAR = "HIPDNN_DISABLE_EXACT_ENGINE_CACHE";
    const std::string normalized = hipdnn_data_sdk::utilities::toLower(
        hipdnn_data_sdk::utilities::trim(hipdnn_data_sdk::utilities::getEnv(ENV_VAR, "")));

    return normalized == "1" || normalized == "true" || normalized == "on" || normalized == "yes"
           || normalized == "enable" || normalized == "enabled";
}

} // namespace hipdnn_backend::heuristics::config

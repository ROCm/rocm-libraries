// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_data_sdk/utilities/EnvBool.hpp>

namespace hipdnn_backend::heuristics::config
{

/// Reads HIPDNN_DISABLE_EXACT_ENGINE_CACHE and returns whether the exact-match autotune
/// cache is disabled. Explicit falsy values (`0`, `false`, `off`) leave the cache enabled;
/// only a recognized truthy value disables it.
inline bool exactCacheDisabled()
{
    return hipdnn_data_sdk::utilities::readBoolEnv("HIPDNN_DISABLE_EXACT_ENGINE_CACHE")
        .value.value_or(false);
}

} // namespace hipdnn_backend::heuristics::config

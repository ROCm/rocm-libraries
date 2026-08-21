// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// CacheRoot resolves the single root directory hipDNN's on-disk caches (winner cache,
// autotune cache, and future consumers) share, one subdirectory per consumer beneath it.

#include <filesystem>
#include <hipdnn_data_sdk/utilities/CacheRootDefaults.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <string>
#include <system_error>

namespace hipdnn_data_sdk::utilities
{

/// Resolves and ensures the existence of hipDNN's shared on-disk cache root directory.
///
/// Resolution order: `HIPDNN_CACHE_DIR` if set to a non-empty value, otherwise a
/// CMake-baked per-platform default (`~/.cache/hipdnn/` on Linux,
/// `%USERPROFILE%\.hipdnn\cache\` on Windows), passed through expandUser() so a leading
/// `~`/`%USERPROFILE%` follows `$HOME` at run time. The directory is created if absent.
///
/// Never throws. If the resolved path cannot be created or is unusable (e.g. it exists as
/// a file), returns an empty/invalid path -- callers must fall back to in-memory-only
/// behavior.
///
/// @return The cache root directory, guaranteed to exist, on success; an empty/invalid
///     path on any resolution or filesystem failure.
inline std::filesystem::path cacheRoot()
{
    std::string rawPath = getEnv("HIPDNN_CACHE_DIR");
    if(rawPath.empty())
    {
        rawPath = HIPDNN_DATA_SDK_CACHE_ROOT_DEFAULT;
    }

    const std::string expanded = expandUser(rawPath);
    if(expanded.empty())
    {
        return {};
    }

    std::filesystem::path resolved(expanded);

    std::error_code failed;
    std::filesystem::create_directories(resolved, failed);
    if(failed || !std::filesystem::is_directory(resolved))
    {
        // Creation failed, or the path already existed as something other than a
        // directory -- either way, no on-disk cache is available.
        return {};
    }

    return resolved;
}

} // namespace hipdnn_data_sdk::utilities

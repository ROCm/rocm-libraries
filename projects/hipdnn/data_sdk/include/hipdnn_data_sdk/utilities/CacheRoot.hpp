// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// CacheRoot resolves the single root directory hipDNN's on-disk caches (winner cache,
// autotune cache, and future consumers) share, one subdirectory per consumer beneath it.

#if defined(_WIN32)
#include <algorithm>
#include <cwctype>
#endif
#include <filesystem>
#include <hipdnn_data_sdk/utilities/CacheRootDefaults.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <string>
#include <string_view>
#include <system_error>

namespace hipdnn_data_sdk::utilities
{

/// True if `HIPDNN_DISABLE_CACHE` is set to a recognized truthy value, matching the token
/// set `HIPDNN_FORCE_BENCHMARKING` accepts (case-insensitive, whitespace-trimmed). Unset,
/// empty, or any other value is treated as not-disabled, so a typo fails open rather than
/// silently disabling the cache.
inline bool cacheDisabledByEnv()
{
    const std::string normalized = toLower(trim(getEnv("HIPDNN_DISABLE_CACHE")));
    return normalized == "1" || normalized == "true" || normalized == "on" || normalized == "yes"
           || normalized == "enable" || normalized == "enabled";
}

#if defined(_WIN32)
namespace detail
{
/// True if @p path still begins with the literal token `%userprofile%`, matched
/// case-insensitively exactly as expandUserW() matches it, and followed by `/`, `\`, or
/// end-of-string. expandUserW() replaces a qualifying leading token with USERPROFILE's
/// value, so a path that still starts with the token means USERPROFILE was unset -- not
/// that the caller asked for a directory literally named "%userprofile%".
inline bool startsWithUserProfileToken(const std::wstring& path)
{
    static const std::wstring kUserProfileToken = L"%userprofile%";
    std::wstring lowerPath = path;
    std::transform(lowerPath.begin(), lowerPath.end(), lowerPath.begin(), ::towlower);
    return lowerPath.size() >= kUserProfileToken.size()
           && lowerPath.compare(0, kUserProfileToken.size(), kUserProfileToken) == 0
           && (lowerPath.size() == kUserProfileToken.size()
               || path[kUserProfileToken.size()] == L'/'
               || path[kUserProfileToken.size()] == L'\\');
}
} // namespace detail
#endif

/// Resolves and ensures the existence of hipDNN's shared on-disk cache root directory.
///
/// `HIPDNN_DISABLE_CACHE` is checked first: if set to a truthy value, this returns an
/// empty path unconditionally -- see cacheDisabledByEnv(). Otherwise, resolution order is:
/// `HIPDNN_CACHE_DIR` if set to a non-empty value, otherwise a CMake-baked per-platform
/// default (`~/.cache/hipdnn/` on Linux, `%USERPROFILE%\.hipdnn\cache\` on Windows),
/// passed through expandUser() so a leading `~`/`%USERPROFILE%` follows $HOME at run
/// time. The directory is created if absent.
///
/// On Windows this is composed entirely in UTF-16 (getEnvW()/expandUserW()), never
/// through the narrow getEnv()/expandUser(): GetEnvironmentVariableA transcodes the
/// natively-UTF-16 environment through the process ANSI code page, while
/// std::filesystem::path's narrow constructor is interpreted as UTF-8 by MSVC's
/// implementation. Those two encodings disagree for any non-ASCII byte -- e.g. a
/// non-ASCII account name in `%USERPROFILE%` -- which would otherwise corrupt the path
/// and fail `create_directories` silently, defeating the CreateFileW the design
/// (LineStore.hpp) chose specifically to support such paths.
///
/// Never throws. If the resolved path cannot be created or is unusable (e.g. it exists as
/// a file), returns an empty/invalid path -- callers must fall back to in-memory-only
/// behavior. This is also the caching-disabled return value, so every caller already
/// treats an empty root as "no disk cache" without a separate disabled check.
///
/// @return The cache root directory, guaranteed to exist, on success; an empty/invalid
///     path if disabled via `HIPDNN_DISABLE_CACHE`, or on any resolution or filesystem
///     failure.
inline std::filesystem::path cacheRoot()
{
    if(cacheDisabledByEnv())
    {
        return {};
    }

#if defined(_WIN32)
    // HIPDNN_DATA_SDK_CACHE_ROOT_DEFAULT is a CMake-baked literal composed only of ASCII
    // characters (see data_sdk/CMakeLists.txt), so widening it byte-for-byte is exact --
    // no transcoding step is involved, unlike the environment reads below.
    std::wstring rawPath = getEnvW(L"HIPDNN_CACHE_DIR");
    if(rawPath.empty())
    {
        const std::string_view defaultNarrow = HIPDNN_DATA_SDK_CACHE_ROOT_DEFAULT;
        rawPath.assign(defaultNarrow.begin(), defaultNarrow.end());
    }

    const std::wstring expanded = expandUserW(rawPath);
    // expandUserW() returns its input unchanged when USERPROFILE is unset, so a surviving
    // leading '~' or '%userprofile%' means the home directory is unknown, not that the
    // user asked for a directory with that literal name. Creating it would silently make
    // the cache per-working-directory. Empty is what every caller already treats as "no
    // disk cache".
    if(expanded.empty() || expanded.front() == L'~' || detail::startsWithUserProfileToken(expanded))
    {
        return {};
    }

    std::filesystem::path resolved(expanded);
#else
    std::string rawPath = getEnv("HIPDNN_CACHE_DIR");
    if(rawPath.empty())
    {
        rawPath = HIPDNN_DATA_SDK_CACHE_ROOT_DEFAULT;
    }

    const std::string expanded = expandUser(rawPath);
    // expandUser() returns its input unchanged when HOME is unset, so a surviving leading '~' means
    // the home directory is unknown, not that the user asked for a directory named "~". Creating it
    // would silently make the cache per-working-directory. Empty is what every caller already
    // treats as "no disk cache".
    if(expanded.empty() || expanded.front() == '~')
    {
        return {};
    }

    std::filesystem::path resolved(expanded);
#endif

    std::error_code failed;
    std::filesystem::create_directories(resolved, failed);
    // The error_code overload of is_directory(): the throwing one would break this
    // function's never-throws contract if the path became unreadable after creation.
    std::error_code queryFailed;
    if(failed || !std::filesystem::is_directory(resolved, queryFailed))
    {
        // Creation failed, or the path already existed as something other than a
        // directory -- either way, no on-disk cache is available.
        return {};
    }

    return resolved;
}

} // namespace hipdnn_data_sdk::utilities

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>
#include <string_view>

#ifdef _WIN32

#ifndef NOMINMAX
#define NOMINMAX
#endif

// clang-format off
#include <windows.h>
#include <shlwapi.h>
// clang-format on

#elif defined(__linux__)

#include <fnmatch.h>

#else

#error "Unsupported platform"

#endif

namespace hipdnn_integration_tests
{

/// Returns the lowercase platform name the test binary was built for
/// ("windows" or "linux"). Used to match against the optional
/// `platforms` field in [[test_skips]] TOML entries.
inline std::string currentPlatform()
{
#ifdef _WIN32
    return "windows";
#elif defined(__linux__)
    return "linux";
#endif
}

/// Cross-platform glob match supporting '*' and '?' wildcards.
/// Returns true if @p text matches @p pattern.
inline bool globMatch(std::string_view pattern, std::string_view text)
{
    const std::string patternStr(pattern);
    const std::string textStr(text);

#ifdef _WIN32
    return PathMatchSpecA(textStr.c_str(), patternStr.c_str()) != FALSE;
#elif defined(__linux__)
    return fnmatch(patternStr.c_str(), textStr.c_str(), 0) == 0;
#endif
}

} // namespace hipdnn_integration_tests

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

// MSVC ships no <unistd.h>; it spells getpid() as _getpid() in <process.h>.
#include <process.h>

#elif defined(__linux__)

#include <fnmatch.h>
#include <unistd.h>

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

/// This process's id. Used to stamp a name no sibling process can draw.
///
/// tests/ScratchDirectory.hpp carries its own copy under `scratch::`. Five lines
/// of #ifdef are cheaper to repeat than to make that test-only header include
/// this one, which would pull <windows.h> into every test translation unit that
/// wants a scratch directory.
inline int currentProcessId()
{
#ifdef _WIN32
    return _getpid();
#elif defined(__linux__)
    return ::getpid();
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

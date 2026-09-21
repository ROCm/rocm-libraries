// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/PluginApi.h>

#include <array>
#include <cstdio>

namespace hipdnn_backend::heuristics::detail
{

constexpr size_t BUILT_IN_LOG_BUFFER_SIZE = 1024;

} // namespace hipdnn_backend::heuristics::detail

// Shared logging macro for built-in heuristic policies. Each built-in declares
// its own file-scope g_loggingCallback / g_logLevel globals (set via the C-ABI
// SetLoggingCallback / SetLogLevel pseudo-plugin entrypoints) and uses this
// macro to format a prefixed message and dispatch through whatever sink the
// host wired up.
//
// Usage:
//   #define MY_LOG(severity, ...) \
//       HIPDNN_BUILTIN_HEURISTIC_LOG(                                            \
//           g_loggingCallback, g_logLevel, severity, "[MyBuiltIn] ", __VA_ARGS__)
#define HIPDNN_BUILTIN_HEURISTIC_LOG(callback, threshold, severity, prefix, ...)             \
    do                                                                                       \
    {                                                                                        \
        if((callback) != nullptr && (severity) >= (threshold))                               \
        {                                                                                    \
            std::array<char, ::hipdnn_backend::heuristics::detail::BUILT_IN_LOG_BUFFER_SIZE> \
                _buf{};                                                                      \
            std::snprintf(_buf.data(), _buf.size(), prefix __VA_ARGS__);                     \
            (callback)((severity), _buf.data());                                             \
        }                                                                                    \
    } while(0)

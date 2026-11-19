// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/PluginApi.h>

#include <array>
#include <atomic>
#include <cstdio>

namespace hipdnn_backend::heuristics::detail
{

constexpr size_t BUILT_IN_LOG_BUFFER_SIZE = 1024;

} // namespace hipdnn_backend::heuristics::detail

// Shared logging macro for built-in heuristic policies. Each built-in declares
// its own file-scope std::atomic g_loggingCallback / g_logLevel globals (set via the C-ABI
// SetLoggingCallback / SetLogLevel pseudo-plugin entrypoints, which every handle creation
// calls while other threads may be logging) and uses this macro to format a prefixed
// message and dispatch through whatever sink the host wired up. The macro loads each
// global once per use.
//
// Usage:
//   #define MY_LOG(severity, ...) \
//       HIPDNN_BUILTIN_HEURISTIC_LOG(                                            \
//           g_loggingCallback, g_logLevel, severity, "[MyBuiltIn] ", __VA_ARGS__)
#define HIPDNN_BUILTIN_HEURISTIC_LOG(callback, threshold, severity, prefix, ...)              \
    do                                                                                        \
    {                                                                                         \
        const hipdnnCallback_t _callback = (callback).load(std::memory_order_relaxed);        \
        if(_callback != nullptr && (severity) >= (threshold).load(std::memory_order_relaxed)) \
        {                                                                                     \
            std::array<char, ::hipdnn_backend::heuristics::detail::BUILT_IN_LOG_BUFFER_SIZE>  \
                _buf{};                                                                       \
            std::snprintf(_buf.data(), _buf.size(), prefix __VA_ARGS__);                      \
            _callback((severity), _buf.data());                                               \
        }                                                                                     \
    } while(0)

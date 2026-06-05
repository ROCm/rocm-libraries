// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include <string>

namespace example_provider
{

// NOLINTBEGIN(cppcoreguidelines-macro-usage)

/// Check a HIP runtime API call and throw on failure.
#define HIP_CHECK(call)                                                        \
    do                                                                         \
    {                                                                          \
        const hipError_t status = (call);                                      \
        if(status != hipSuccess)                                               \
        {                                                                      \
            throw hipdnn_plugin_sdk::HipdnnPluginException(                    \
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,                           \
                std::string(#call) + " failed: " + hipGetErrorString(status)); \
        }                                                                      \
    } while(0)

/// Check an HIPRTC API call and throw on failure.
#define HIPRTC_CHECK(call)                                                        \
    do                                                                            \
    {                                                                             \
        const hiprtcResult status = (call);                                       \
        if(status != HIPRTC_SUCCESS)                                              \
        {                                                                         \
            throw hipdnn_plugin_sdk::HipdnnPluginException(                       \
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,                              \
                std::string(#call) + " failed: " + hiprtcGetErrorString(status)); \
        }                                                                         \
    } while(0)

// NOLINTEND(cppcoreguidelines-macro-usage)

} // namespace example_provider

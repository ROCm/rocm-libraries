// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once
#include <hip/hiprtc.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace hip_kernel_provider::compilation
{

// NOLINTBEGIN(cppcoreguidelines-macro-usage)

// For HIP runtime API calls
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

// For hipRTC API calls
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

} // namespace hip_kernel_provider::compilation

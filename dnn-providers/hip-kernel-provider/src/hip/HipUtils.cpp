// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipUtils.hpp"

#include <hipdnn_plugin_sdk/PluginBaseTypes.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace hip_kernel_provider
{

std::string getDeviceString(hipStream_t stream)
{
    int deviceId = -1;
    auto status = hipStreamGetDevice(stream, &deviceId);
    if(status != hipSuccess)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            hipdnnPluginStatus_t::HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "hipStreamGetDevice failed with error code: " + std::to_string(status));
    }

    hipDeviceProp_t props;
    status = hipGetDeviceProperties(&props, deviceId);
    if(status != hipSuccess)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            hipdnnPluginStatus_t::HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "hipGetDeviceProperties failed with error code: " + std::to_string(status));
    }
    std::string archStr(props.gcnArchName);

    return archStr.substr(0, archStr.find(':'));
}

}

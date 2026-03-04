// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipKernelUtils.hpp"

#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace hip_kernel_provider::hip_kernel_utils
{

hipdnnPluginDeviceBuffer_t findDeviceBuffer(int64_t uid,
                                            const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                            uint32_t numDeviceBuffers)
{
    for(uint32_t i = 0; i < numDeviceBuffers; i++)
    {
        if(uid == deviceBuffers[i].uid)
        {
            return deviceBuffers[i];
        }
    }

    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
        "Device buffer with the uid: " + std::to_string(uid)
            + " not found in the provided device buffers.");
}

const hipdnn_data_sdk::data_objects::TensorAttributes& findTensorAttributes(
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    int64_t uid)
{
    if(auto tensorAttr = tensorMap.find(uid); tensorAttr != tensorMap.end())
    {
        return *tensorAttr->second;
    }

    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                   "Failed to find tensor with UID in tensorMap: "
                                                       + std::to_string(uid));
}

} // namespace hip_kernel_provider::hip_kernel_utils

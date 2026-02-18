// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipblasltUtils.hpp"

namespace hipblaslt_plugin::hipblaslt_utils
{

hipDataType tensorDataTypeToHipDataType(const hipdnn_data_sdk::data_objects::DataType& dataType)
{
    switch(dataType)
    {
    case hipdnn_data_sdk::data_objects::DataType::FLOAT:
        return HIP_R_32F;
    case hipdnn_data_sdk::data_objects::DataType::INT32:
        return HIP_R_32I;
    case hipdnn_data_sdk::data_objects::DataType::HALF:
        return HIP_R_16F;
    case hipdnn_data_sdk::data_objects::DataType::BFLOAT16:
        return HIP_R_16BF;
    case hipdnn_data_sdk::data_objects::DataType::INT8:
        return HIP_R_8I;
    default:
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported data type for hipBLASLt: "
                + std::string(hipdnn_data_sdk::data_objects::toString(dataType)));
    }
}

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

hipdnn_plugin_sdk::TensorAttributesWrapper findTensorAttributes(
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    int64_t uid)
{
    if(auto tensorAttr = tensorMap.find(uid); tensorAttr != tensorMap.end())
    {
        return hipdnn_plugin_sdk::TensorAttributesWrapper(tensorAttr->second);
    }

    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                   "Failed to find tensor with UID in tensorMap: "
                                                       + std::to_string(uid));
}

} // namespace hipblaslt_plugin::hipblaslt_utils

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "MiopenUtils.hpp"

namespace miopen_legacy_plugin
{

namespace miopen_utils
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

    throw hipdnn_plugin::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                               "Device buffer with the uid: " + std::to_string(uid)
                                                   + " not found in the provided device buffers.");
}

miopenDataType_t tensorDataTypeToMiopenDataType(const hipdnn_sdk::data_objects::DataType& dataType)
{
    switch(dataType)
    {
    case hipdnn_sdk::data_objects::DataType::FLOAT:
        return miopenFloat;
    case hipdnn_sdk::data_objects::DataType::HALF:
        return miopenHalf;
    case hipdnn_sdk::data_objects::DataType::BFLOAT16:
        return miopenBFloat16;
    default:
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported data type for MIOpen: "
                + std::string(hipdnn_sdk::data_objects::toString(dataType)));
    }
}

const hipdnn_sdk::data_objects::TensorAttributes& findTensorAttributes(
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap,
    int64_t uid)
{
    if(auto tensorAttr = tensorMap.find(uid); tensorAttr != tensorMap.end())
    {
        return *tensorAttr->second;
    }

    throw hipdnn_plugin::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                               "Failed to find tensor with UID in tensorMap: "
                                                   + std::to_string(uid));
}

MiopenTensor createTensor(
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap,
    int64_t uid)
{
    const auto& tensorAttr = findTensorAttributes(tensorMap, uid);
    return {tensorAttr};
}

size_t getSpatialDimCount(const hipdnn_sdk::data_objects::TensorAttributes& attr)
{
    if(attr.dims()->size() < 3)
    {
        throw hipdnn_plugin::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Tensor must have at least 3 dimensions, but got: "
                + std::to_string(attr.dims()->size()));
    }

    return attr.dims()->size() - 2;
}

}

}

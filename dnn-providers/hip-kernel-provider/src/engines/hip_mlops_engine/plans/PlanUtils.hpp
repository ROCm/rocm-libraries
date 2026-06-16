// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h"
#include "hipdnn_plugin_sdk/PluginException.hpp"
#include <cstddef>

namespace hip_kernel_provider
{

/**
* Given a data SDK type, returns a string of the HIP type that can be used to pass
* it as a kernel argument. Intended to be used when kernel source code supports
* multiple tensors type which are instantiated via a preprocessor definition.
*/
inline const char* getKernelParamTypeString(hipdnn_flatbuffers_sdk::data_objects::DataType type)
{
    switch(type)
    {
    case hipdnn_flatbuffers_sdk::data_objects::DataType::HALF:
        return "half";
    case hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16:
        return "ushort";
    case hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT:
        return "float";
    default:
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            std::string("Unsupported data type: ")
                + hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(type));
    }
}

namespace batchnorm
{

/**
 * Calculates the vector size for batchnorm forward inference plans based on the layout and channel information.
 */
inline size_t computeVectorSize(bool isLayoutNHWC, int channels, unsigned int inCstride)
{
    if(isLayoutNHWC)
    {
        if(channels % 4 == 0)
        {
            return 4;
        }
        return channels % 2 == 0 ? 2 : 1;
    }

    if(inCstride % 4 == 0)
    {
        return 4;
    }
    return inCstride % 2 == 0 ? 2 : 1;
}

} // namespace batchnorm

} // namespace hip_kernel_provider

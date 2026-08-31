// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h"
#include "hipdnn_plugin_sdk/PluginException.hpp"
#include <cstddef>
#include <limits>
#include <string>

namespace hip_kernel_provider
{

/**
 * Safely narrows an int64_t value to int, throwing HipdnnPluginException
 * if the value is out of range.
 */
inline int checkedNarrowToInt(int64_t value, const char* name)
{
    if(value < static_cast<int64_t>(std::numeric_limits<int>::min())
       || value > static_cast<int64_t>(std::numeric_limits<int>::max()))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            std::string(name) + " value " + std::to_string(value) + " exceeds int range");
    }
    return static_cast<int>(value);
}

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
        return "__bf16";
    case hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT:
        return "float";
    default:
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            std::string("Unsupported data type: ")
                + hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(type));
    }
}

/**
* Given a value and a step, returns the rounded up division of value over step (i.e. ceil).
*/
template <typename T>
inline T align(T value, T step)
{
    return (value + step - 1) / step;
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

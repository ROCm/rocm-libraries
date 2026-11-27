// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h"
#include "hipdnn_plugin_sdk/PluginException.hpp"
#include <cstddef>
#include <cstdint>
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

/**
 * One kernel launch, with a one-dimensional grid and block.
 */
struct KernelLaunchShape
{
    const char* kernelName;
    int64_t blocks; // grid size along x
    int64_t threadsPerBlock; // block size along x
};

/**
 * HIP rejects a launch whose total thread count reaches 2^32.
 * Returns true iff blocks >= 1, threadsPerBlock >= 1 and blocks * threadsPerBlock <= UINT32_MAX,
 * evaluated without overflow.
 */
inline bool launchFitsThreadLimit(int64_t blocks, int64_t threadsPerBlock)
{
    return blocks >= 1 && threadsPerBlock >= 1
           && blocks
                  <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) / threadsPerBlock;
}

/**
 * Throws HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM) naming the kernel, blocks and
 * threads when launchFitsThreadLimit is false.
 */
inline void checkLaunchFitsThreadLimit(const KernelLaunchShape& shape)
{
    if(!launchFitsThreadLimit(shape.blocks, shape.threadsPerBlock))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            std::string("Unsupported launch of kernel ") + shape.kernelName + ": "
                + std::to_string(shape.blocks) + " blocks of "
                + std::to_string(shape.threadsPerBlock)
                + " threads reach the 2^32 thread limit or are empty");
    }
}

namespace batchnorm
{

/**
 * Calculates the vector size for batchnorm forward inference kernels based on the layout and channel information.
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

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <unordered_map>

#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

/**
 * @brief Macro that returns and prints info log on passing provided condition.
 * Arguments after first are passed into std::format().
 * Requires a HIP_KERNEL_LOG_PREFIX in scope which will prefix the log
 */
#define HIP_KERNEL_RETURN_FALSE_IF(condition, message)                            \
    do                                                                            \
    {                                                                             \
        if(condition)                                                             \
        {                                                                         \
            HIPDNN_PLUGIN_LOG_INFO(std::string{HIP_KERNEL_LOG_PREFIX} + message); \
            return false;                                                         \
        }                                                                         \
    } while(0)

namespace hip_kernel_provider::hip_kernel_utils
{

enum class ActivationMode : int
{
    PASTHRU = 0,
    LOGISTIC = 1, // sigmoid
    TANH = 2,
    RELU = 3,
    SOFTRELU = 4, // softplus
    ABS = 5,
    POWER = 6,
    CLIPPED_RELU = 7,
    LEAKY_RELU = 8,
    ELU = 9,
    CLAMP = 10
};

struct ActivationParams
{
    ActivationMode mode;
    double alpha;
    double beta;
    double gamma;
};

ActivationParams
    parseActivation(const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& attrs);

hipdnnPluginDeviceBuffer_t findDeviceBuffer(int64_t uid,
                                            const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                            uint32_t numDeviceBuffers);

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& findTensorAttributes(
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    int64_t uid);

bool isChannelLastLayout(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* tensor);

}

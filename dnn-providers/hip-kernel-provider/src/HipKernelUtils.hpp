// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <unordered_map>

#include <hipdnn_data_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

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

ActivationParams parseActivation(const hipdnn_data_sdk::data_objects::PointwiseAttributes& attrs);

hipdnnPluginDeviceBuffer_t findDeviceBuffer(int64_t uid,
                                            const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                            uint32_t numDeviceBuffers);

const hipdnn_data_sdk::data_objects::TensorAttributes& findTensorAttributes(
    const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    int64_t uid);

bool isChannelLastLayout(const hipdnn_data_sdk::data_objects::TensorAttributes* tensor);

}

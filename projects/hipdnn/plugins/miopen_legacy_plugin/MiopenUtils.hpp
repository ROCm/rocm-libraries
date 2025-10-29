// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <optional>
#include <unordered_map>

#include <hipdnn_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_sdk/plugin/PluginException.hpp>
#include <hipdnn_sdk/plugin/PluginFlatbufferTypeHelpers.hpp>
#include <miopen/miopen.h>

#include "MiopenTensor.hpp"

#define LOG_ON_MIOPEN_FAILURE(status)                                                    \
    do                                                                                   \
    {                                                                                    \
        if(status != miopenStatusSuccess)                                                \
        {                                                                                \
            HIPDNN_LOG_ERROR("MIOpen error occurred: {}", miopenGetErrorString(status)); \
        }                                                                                \
    } while(0)

#define THROW_ON_MIOPEN_FAILURE(status)                                                 \
    do                                                                                  \
    {                                                                                   \
        if(status != miopenStatusSuccess)                                               \
        {                                                                               \
            throw hipdnn_plugin::HipdnnPluginException(                                 \
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,                                    \
                "MIOpen error occurred: " + std::string(miopenGetErrorString(status))); \
        }                                                                               \
    } while(0)

namespace miopen_legacy_plugin
{

namespace miopen_utils
{

struct ActivationParams
{
    miopenActivationMode_t mode;
    double alpha;
    double beta;
    double gamma;
};

std::optional<ActivationParams>
    mapPointwiseModeToMiopenActivation(const hipdnn_sdk::data_objects::PointwiseAttributes& attrs);

hipdnnPluginDeviceBuffer_t findDeviceBuffer(int64_t uid,
                                            const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                            uint32_t numDeviceBuffers);

miopenDataType_t tensorDataTypeToMiopenDataType(const hipdnn_sdk::data_objects::DataType& dataType);

const hipdnn_sdk::data_objects::TensorAttributes& findTensorAttributes(
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap,
    int64_t uid);

MiopenTensor createTensor(
    const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>& tensorMap,
    int64_t uid);

size_t getSpatialDimCount(const hipdnn_sdk::data_objects::TensorAttributes& attr);

}

}

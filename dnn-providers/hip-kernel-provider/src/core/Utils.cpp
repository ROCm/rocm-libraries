// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "Utils.hpp"

#include <cstring>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace hip_kernel_provider::core::utils
{

ActivationParams
    parseActivation(const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& attrs)
{
    using PM = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

    switch(attrs.operation())
    {
    case PM::RELU_FWD:
    case PM::RELU_BWD:
    {
        if(attrs.relu_lower_clip() && attrs.relu_upper_clip())
        {
            return ActivationParams{ActivationMode::CLAMP,
                                    static_cast<double>(*attrs.relu_lower_clip()),
                                    static_cast<double>(*attrs.relu_upper_clip()),
                                    0.0};
        }
        if(attrs.relu_upper_clip())
        {
            return ActivationParams{ActivationMode::CLIPPED_RELU,
                                    static_cast<double>(*attrs.relu_upper_clip()),
                                    0.0,
                                    0.0};
        }
        if(attrs.relu_lower_clip_slope())
        {
            return ActivationParams{ActivationMode::LEAKY_RELU,
                                    static_cast<double>(*attrs.relu_lower_clip_slope()),
                                    0.0,
                                    0.0};
        }
        if(attrs.relu_lower_clip().has_value() && attrs.relu_lower_clip().value() != 0.f)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "Standard relu with a non-zero lower_clip is not supported");
        }
        return ActivationParams{ActivationMode::RELU, 0.0, 0.0, 0.0};
    }
    case PM::SIGMOID_FWD:
    case PM::SIGMOID_BWD:
        return ActivationParams{ActivationMode::LOGISTIC, 0.0, 0.0, 0.0};
    case PM::TANH_FWD:
    case PM::TANH_BWD:
        return ActivationParams{ActivationMode::TANH, 1.0, 1.0, 0.0};
    case PM::ELU_FWD:
    case PM::ELU_BWD:
    {
        const double alpha = attrs.elu_alpha() ? static_cast<double>(*attrs.elu_alpha()) : 1.0;
        return ActivationParams{ActivationMode::ELU, alpha, 0.0, 0.0};
    }
    case PM::SOFTPLUS_FWD:
    case PM::SOFTPLUS_BWD:
        if(attrs.softplus_beta())
        {
            if(static_cast<double>(*attrs.softplus_beta()) != 1.0)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                               "Softplus only supports beta = 1.0");
            }
        }
        return ActivationParams{ActivationMode::SOFTRELU, 0.0, 0.0, 0.0};
    case PM::ABS:
        return ActivationParams{ActivationMode::ABS, 0.0, 0.0, 0.0};
    case PM::IDENTITY:
        return ActivationParams{ActivationMode::PASTHRU, 0.0, 0.0, 0.0};
    default:
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Unsupported activation operation");
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

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& findTensorAttributes(
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
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

bool isChannelLastLayout(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* tensor)
{
    const auto* strides = tensor->strides();
    const auto* dims = tensor->dims();
    const size_t numDims = dims->size();

    // Extract stride order from strides
    const std::vector<int64_t> stridesVec(strides->begin(), strides->end());
    const std::vector<int64_t> strideOrder
        = hipdnn_data_sdk::utilities::extractStrideOrder(stridesVec);

    // Compare against known layouts
    if(numDims == 4)
    {
        const auto layoutNchw = hipdnn_data_sdk::utilities::TensorLayout::NCHW;
        const auto layoutNhwc = hipdnn_data_sdk::utilities::TensorLayout::NHWC;

        if(strideOrder == layoutNhwc.strideOrder)
        {
            return true;
        }
        if(strideOrder == layoutNchw.strideOrder)
        {
            return false;
        }
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported tensor layout for 4D tensor. Only NCHW and NHWC are supported.");
    }

    if(numDims == 5)
    {
        const auto layoutNcdhw = hipdnn_data_sdk::utilities::TensorLayout::NCDHW;
        const auto layoutNdhwc = hipdnn_data_sdk::utilities::TensorLayout::NDHWC;

        if(strideOrder == layoutNdhwc.strideOrder)
        {
            return true;
        }
        if(strideOrder == layoutNcdhw.strideOrder)
        {
            return false;
        }
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Unsupported tensor layout for 5D tensor. Only NCDHW and NDHWC are supported.");
    }

    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_BAD_PARAM,
        "Tensor must be 4D or 5D for layout detection. Got " + std::to_string(numDims)
            + "D tensor.");
}

ScalarOperand makeScalarOperand(
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    int64_t uid,
    const char* paramName)
{
    const auto* attr = tensorMap.at(uid);
    if(attr->is_runtime_pass_by_value()
       && attr->value_type() == hipdnn_flatbuffers_sdk::data_objects::TensorValue::NONE)
    {
        return ScalarOperand{uid, attr->data_type(), true, 0.0};
    }
    return ScalarOperand{
        uid,
        attr->data_type(),
        false,
        hipdnn_flatbuffers_sdk::utilities::extractDoubleFromTensorValue(attr, paramName)};
}

namespace
{

template <typename T>
double readHostScalar(const void* ptr)
{
    T value;
    std::memcpy(&value, ptr, sizeof(T));
    return static_cast<double>(value);
}

} // namespace

double resolveScalarOperand(const ScalarOperand& op,
                            const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                            uint32_t numDeviceBuffers)
{
    if(!op.isRuntimeUserSupplied)
    {
        return op.bakedDefault;
    }

    const hipdnnPluginDeviceBuffer_t buffer
        = findDeviceBuffer(op.uid, deviceBuffers, numDeviceBuffers);
    const void* ptr = buffer.ptr;

    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    switch(op.dataType)
    {
    case DataType::DOUBLE:
        return readHostScalar<double>(ptr);
    case DataType::FLOAT:
        return readHostScalar<float>(ptr);
    case DataType::HALF:
        return static_cast<double>(
            static_cast<float>(readHostScalar<hipdnn_data_sdk::types::half>(ptr)));
    case DataType::BFLOAT16:
        return static_cast<double>(
            static_cast<float>(readHostScalar<hipdnn_data_sdk::types::bfloat16>(ptr)));
    case DataType::INT32:
        return readHostScalar<int32_t>(ptr);
    case DataType::INT64:
        return readHostScalar<int64_t>(ptr);
    case DataType::BOOLEAN:
        return readHostScalar<bool>(ptr);
    case DataType::UNSET:
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Scalar operand has UNSET data type");
    default:
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "Scalar operand has unsupported data type");
    }
}

} // namespace hip_kernel_provider::core::utils

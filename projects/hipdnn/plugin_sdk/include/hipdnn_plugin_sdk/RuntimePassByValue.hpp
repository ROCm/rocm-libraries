// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

/**
 * @file RuntimePassByValue.hpp
 * @brief Shared plugin-side helpers for RFC 0016 runtime pass-by-value scalar tensors.
 *
 * A pass-by-value scalar tensor (epsilon, momentum, ...) can be in one of three states:
 *  - Compile-time constant (`is_runtime_pass_by_value()==false`, value present): the plugin
 *    API version floor stays at the baseline; the value is baked at plan-build time.
 *  - Runtime-with-default (`is_runtime_pass_by_value()==true`, value present): the graph
 *    floors the host at `K_PASS_BY_VALUE_MIN_API_VERSION`, but the plugin still reads the
 *    baked default at plan-build time. Any `device_buffers` slot for that uid must be
 *    ignored -- the frontend never delivers an override for a tensor that already has a
 *    default, and `Graph::execute()` forwards the caller's entire variant pack unfiltered.
 *  - Pure runtime user-supplied (`is_runtime_pass_by_value()==true`, no value): the plugin
 *    cannot resolve a value until execute, where it must read a host-supplied scalar from
 *    the `device_buffers` slot matching the tensor's uid.
 *
 * `ScalarOperand` captures this classification once at plan-build time (`makeScalarOperand`)
 * and defers the actual value lookup to execute (`resolveScalarOperand`), which is the only
 * point at which `device_buffers` may safely be consulted.
 */

#include <cstdint>
#include <cstring>
#include <unordered_map>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace hipdnn_plugin_sdk
{

/// @brief Linear-scans `deviceBuffers` for the entry matching `uid`.
/// @throws HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE) if absent.
inline hipdnnPluginDeviceBuffer_t findDeviceBuffer(int64_t uid,
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

    throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                "Device buffer with the uid: " + std::to_string(uid)
                                    + " not found in the provided device buffers.");
}

/// @brief A scalar tensor operand (epsilon/momentum) resolved either at plan-build
/// (compile-time constant or runtime-with-default) or at execute (pure runtime
/// user-supplied, i.e. is_runtime_pass_by_value() && value_type() == NONE).
struct ScalarOperand
{
    int64_t uid = 0;
    hipdnn_flatbuffers_sdk::data_objects::DataType dataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;
    bool isRuntimeUserSupplied = false;
    double bakedDefault = 0.0;
};

/// @brief Builds a ScalarOperand from the op-graph tensor at plan-build time.
/// Pure user-supplied tensors (is_runtime_pass_by_value() && value_type()==NONE)
/// record uid+dtype only, deferring the read to execute. Every other state
/// (compile-time constant, or runtime-with-default) extracts the baked value now.
inline ScalarOperand makeScalarOperand(
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

namespace detail
{

template <typename T>
double readHostScalar(const void* ptr)
{
    T value;
    std::memcpy(&value, ptr, sizeof(T));
    return static_cast<double>(value);
}

} // namespace detail

/// @brief Resolves a ScalarOperand at execute time. Pure user-supplied operands
/// read the host scalar from the matching device_buffers slot (throws
/// HIPDNN_PLUGIN_STATUS_INVALID_VALUE if absent); all other operands return the
/// baked default and ignore any device_buffers slot for that uid.
inline double resolveScalarOperand(const ScalarOperand& op,
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
        return detail::readHostScalar<double>(ptr);
    case DataType::FLOAT:
        return detail::readHostScalar<float>(ptr);
    case DataType::HALF:
        return static_cast<double>(
            static_cast<float>(detail::readHostScalar<hipdnn_data_sdk::types::half>(ptr)));
    case DataType::BFLOAT16:
        return static_cast<double>(
            static_cast<float>(detail::readHostScalar<hipdnn_data_sdk::types::bfloat16>(ptr)));
    case DataType::INT32:
        return detail::readHostScalar<int32_t>(ptr);
    case DataType::INT64:
        return detail::readHostScalar<int64_t>(ptr);
    case DataType::BOOLEAN:
        return detail::readHostScalar<bool>(ptr);
    case DataType::UNSET:
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                    "Scalar operand has UNSET data type");
    default:
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                    "Scalar operand has unsupported data type");
    }
}

} // namespace hipdnn_plugin_sdk

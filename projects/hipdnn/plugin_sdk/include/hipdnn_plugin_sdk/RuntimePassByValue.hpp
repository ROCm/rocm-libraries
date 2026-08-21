// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

/**
 * @file RuntimePassByValue.hpp
 * @brief Shared plugin-side helpers for RFC 0016 runtime pass-by-value scalar tensors.
 *
 * A pass-by-value scalar (epsilon, momentum, ...) is one of three states:
 *  - Compile-time constant (`is_runtime_pass_by_value()==false`, value present): baked
 *    at plan-build time; no API version floor.
 *  - Runtime-with-default (`is_runtime_pass_by_value()==true`, value present): floors
 *    the host at `K_PASS_BY_VALUE_MIN_API_VERSION`, but the plugin still bakes the
 *    default at plan-build time. Any `device_buffers` slot for that uid is ignored --
 *    the frontend never overrides a tensor that already has a default, even though
 *    `Graph::execute()` forwards the caller's whole variant pack unfiltered.
 *  - Pure runtime user-supplied (`is_runtime_pass_by_value()==true`, no value): resolved
 *    only at execute, by reading the host scalar from the `device_buffers` slot matching
 *    the tensor's uid.
 *
 * `ScalarOperand` captures this classification at plan-build time (`makeScalarOperand`)
 * and defers value lookup to execute (`resolveScalarOperand`), the only point where
 * `device_buffers` may safely be consulted.
 */

#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <variant>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace hipdnn_plugin_sdk
{

/// A pass-by-value scalar carried without widening: one arm per supported DataType.
/// `double` is first so a default-constructed ScalarValue holds 0.0. Narrower than the
/// frontend/flatbuffer TensorValue union (no UINT8/INT8/FP8/FP6/FP4); makeScalarOperand()
/// throws HIPDNN_PLUGIN_STATUS_BAD_PARAM at plan-build time for any dtype outside this set.
using ScalarValue = std::variant<double, // DataType::DOUBLE
                                 float, // DataType::FLOAT
                                 hipdnn_data_sdk::types::half, // DataType::HALF
                                 hipdnn_data_sdk::types::bfloat16, // DataType::BFLOAT16
                                 int32_t, // DataType::INT32
                                 int64_t, // DataType::INT64
                                 bool>; // DataType::BOOLEAN

/// Collapses a ScalarValue to double for floating-point consumers (epsilon/momentum).
/// Widening is explicit here rather than hidden in the SDK. Integer arms are checked
/// for exact round-trip (double loses precision above 2^53) and throw rather than
/// silently rounding, though no shipped consumer is integer today.
inline double toDouble(const ScalarValue& value)
{
    return std::visit(
        [](auto v) -> double {
            using T = decltype(v);
            if constexpr(std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>)
            {
                const auto widened = static_cast<double>(v);
                if(static_cast<T>(widened) != v)
                {
                    throw HipdnnPluginException(
                        HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                        "toDouble: integer scalar value is not exactly representable as double");
                }
                return widened;
            }
            else
            {
                return static_cast<double>(v);
            }
        },
        value);
}

/// @brief A scalar tensor operand (epsilon/momentum): resolved at plan-build for
/// compile-time-constant/runtime-with-default, or deferred to execute for pure
/// runtime user-supplied (is_runtime_pass_by_value() && value_type()==NONE).

struct ScalarOperand
{
    int64_t uid = 0;
    hipdnn_flatbuffers_sdk::data_objects::DataType dataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;
    bool isRuntimeUserSupplied = false;
    ScalarValue bakedDefault;
};

/// @brief Builds a ScalarOperand from the op-graph tensor at plan-build time. Pure
/// user-supplied tensors record uid+dtype only, deferring the read to execute; every
/// other state extracts the baked value now.
inline ScalarOperand makeScalarOperand(
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    int64_t uid,
    const char* paramName)
{
    const auto it = tensorMap.find(uid);
    if(it == tensorMap.end())
    {
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                    std::string(paramName) + " tensor uid not found in graph");
    }
    const auto* attr = it->second;
    if(attr->is_runtime_pass_by_value()
       && attr->value_type() == hipdnn_flatbuffers_sdk::data_objects::TensorValue::NONE)
    {
        return ScalarOperand{uid, attr->data_type(), true, 0.0};
    }

    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    using hipdnn_flatbuffers_sdk::utilities::extractValueFromTensorValue;
    switch(attr->data_type())
    {
    case DataType::DOUBLE:
        return ScalarOperand{uid,
                             attr->data_type(),
                             false,
                             ScalarValue{extractValueFromTensorValue<double>(attr, paramName)}};
    case DataType::FLOAT:
        return ScalarOperand{uid,
                             attr->data_type(),
                             false,
                             ScalarValue{extractValueFromTensorValue<float>(attr, paramName)}};
    case DataType::HALF:
        return ScalarOperand{uid,
                             attr->data_type(),
                             false,
                             ScalarValue{extractValueFromTensorValue<hipdnn_data_sdk::types::half>(
                                 attr, paramName)}};
    case DataType::BFLOAT16:
        return ScalarOperand{
            uid,
            attr->data_type(),
            false,
            ScalarValue{
                extractValueFromTensorValue<hipdnn_data_sdk::types::bfloat16>(attr, paramName)}};
    case DataType::INT32:
        return ScalarOperand{uid,
                             attr->data_type(),
                             false,
                             ScalarValue{extractValueFromTensorValue<int32_t>(attr, paramName)}};
    case DataType::INT64:
        return ScalarOperand{uid,
                             attr->data_type(),
                             false,
                             ScalarValue{extractValueFromTensorValue<int64_t>(attr, paramName)}};
    case DataType::BOOLEAN:
        return ScalarOperand{uid,
                             attr->data_type(),
                             false,
                             ScalarValue{extractValueFromTensorValue<bool>(attr, paramName)}};
    case DataType::UNSET:
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                    "Scalar operand has UNSET data type");
    default:
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                    "Scalar operand has unsupported data type");
    }
}

namespace detail
{

template <typename T>
T readHostScalar(const void* ptr)
{
    T value;
    std::memcpy(&value, ptr, sizeof(T));
    return value;
}

} // namespace detail

/// @brief Resolves a ScalarOperand at execute time: pure user-supplied operands read
/// the host scalar from the matching device_buffers slot (throws
/// HIPDNN_PLUGIN_STATUS_INVALID_VALUE if absent); all other operands return the baked
/// default, ignoring any device_buffers slot for that uid.
inline ScalarValue resolveScalarOperand(const ScalarOperand& op,
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
    if(ptr == nullptr)
    {
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                    "Device buffer for uid " + std::to_string(op.uid)
                                        + " has a null pointer");
    }

    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    switch(op.dataType)
    {
    case DataType::DOUBLE:
        return ScalarValue{detail::readHostScalar<double>(ptr)};
    case DataType::FLOAT:
        return ScalarValue{detail::readHostScalar<float>(ptr)};
    case DataType::HALF:
        return ScalarValue{detail::readHostScalar<hipdnn_data_sdk::types::half>(ptr)};
    case DataType::BFLOAT16:
        return ScalarValue{detail::readHostScalar<hipdnn_data_sdk::types::bfloat16>(ptr)};
    case DataType::INT32:
        return ScalarValue{detail::readHostScalar<int32_t>(ptr)};
    case DataType::INT64:
        return ScalarValue{detail::readHostScalar<int64_t>(ptr)};
    case DataType::BOOLEAN:
        return ScalarValue{detail::readHostScalar<bool>(ptr)};
    case DataType::UNSET:
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                    "Scalar operand has UNSET data type");
    default:
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                    "Scalar operand has unsupported data type");
    }
}

} // namespace hipdnn_plugin_sdk

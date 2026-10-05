// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstring>
#include <flatbuffers/flatbuffers.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace hipdnn_flatbuffers_sdk::utilities
{

/// Convert std::optional<T> to flatbuffers::Optional<T>.
template <typename T>
flatbuffers::Optional<T> toFlatbufferOptional(const std::optional<T>& opt)
{
    return opt.has_value() ? flatbuffers::Optional<T>(*opt) : flatbuffers::nullopt;
}

/// Convert flatbuffers::Optional<T> to std::optional<T>.
template <typename T>
std::optional<T> toStdOptional(const flatbuffers::Optional<T>& opt)
{
    return opt.has_value() ? std::optional<T>(opt.value()) : std::nullopt;
}

template <typename T>
inline std::vector<T> convertFlatBufferVectorToStdVector(const flatbuffers::Vector<T>* in)
{
    std::vector<T> out;

    if(in)
    {
        out.resize(in->size());
        for(::flatbuffers::uoffset_t i = 0; i < in->size(); i++)
        {
            out[i] = in->Get(i);
        }
    }

    return out;
}

namespace detail
{

// The object and table APIs share the same TensorValue structs.
template <typename Value>
auto readTensorValue(data_objects::TensorValue valueType, const void* value, const char* paramName)
{
    if(valueType != data_objects::TensorValueTraits<Value>::enum_value)
    {
        throw std::runtime_error(std::string(paramName) + " has a mismatched tensor value type");
    }
    return static_cast<const Value*>(value)->value();
}

template <typename Convert>
auto convertTensorValue(data_objects::DataType dataType,
                        data_objects::TensorValue valueType,
                        const void* value,
                        const char* paramName,
                        Convert convert)
{
    if(value == nullptr)
    {
        throw std::runtime_error(std::string(paramName) + " must be a pass-by-value tensor");
    }

    switch(dataType)
    {
    case data_objects::DataType::DOUBLE:
        return convert(readTensorValue<data_objects::Float64Value>(valueType, value, paramName));
    case data_objects::DataType::FLOAT:
        return convert(readTensorValue<data_objects::Float32Value>(valueType, value, paramName));
    case data_objects::DataType::HALF:
        return convert(readTensorValue<data_objects::Float16Value>(valueType, value, paramName));
    case data_objects::DataType::BFLOAT16:
        return convert(readTensorValue<data_objects::BFloat16Value>(valueType, value, paramName));
    case data_objects::DataType::INT32:
        return convert(readTensorValue<data_objects::Int32Value>(valueType, value, paramName));
    case data_objects::DataType::INT64:
        return convert(readTensorValue<data_objects::Int64Value>(valueType, value, paramName));
    case data_objects::DataType::BOOLEAN:
        return convert(readTensorValue<data_objects::BoolValue>(valueType, value, paramName));
    case data_objects::DataType::UINT8:
        return convert(readTensorValue<data_objects::Float8Value>(valueType, value, paramName));
    case data_objects::DataType::INT8:
    {
        const auto bits = readTensorValue<data_objects::Float8Value>(valueType, value, paramName);
        int8_t signedValue;
        std::memcpy(&signedValue, &bits, sizeof(signedValue));
        return convert(signedValue);
    }
    case data_objects::DataType::FP8_E4M3:
        return convert(static_cast<float>(hipdnn_data_sdk::types::fp8_e4m3::from_bits(
            readTensorValue<data_objects::Float8Value>(valueType, value, paramName))));
    case data_objects::DataType::FP8_E5M2:
        return convert(static_cast<float>(hipdnn_data_sdk::types::fp8_e5m2::from_bits(
            readTensorValue<data_objects::Float8Value>(valueType, value, paramName))));
    case data_objects::DataType::FP8_E4M3_FNUZ:
        return convert(static_cast<float>(hipdnn_data_sdk::types::fp8_e4m3_fnuz::from_bits(
            readTensorValue<data_objects::Float8Value>(valueType, value, paramName))));
    case data_objects::DataType::FP8_E5M2_FNUZ:
        return convert(static_cast<float>(hipdnn_data_sdk::types::fp8_e5m2_fnuz::from_bits(
            readTensorValue<data_objects::Float8Value>(valueType, value, paramName))));
    case data_objects::DataType::UNSET:
        throw std::runtime_error(std::string(paramName) + " tensor has UNSET data type");
    default:
        throw std::runtime_error(std::string(paramName) + " has unsupported data type");
    }
}

// Integer arithmetic preserves ties-to-even rounding and subnormals regardless
// of the host's floating-point settings. Shifts must be in [1, 53].
inline uint64_t roundRightToEven(uint64_t value, unsigned int shift)
{
    const auto result = value >> shift;
    const auto remainder = value & ((uint64_t{1} << shift) - 1);
    const auto midpoint = uint64_t{1} << (shift - 1);
    const bool roundUp = remainder > midpoint || (remainder == midpoint && (result & 1) != 0);
    return result + static_cast<uint64_t>(roundUp);
}

inline float floatFromBits(uint32_t bits)
{
    static_assert(sizeof(float) == sizeof(bits) && std::numeric_limits<float>::is_iec559);
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline float finiteFloat(float value, const char* paramName)
{
    if(!std::isfinite(value))
    {
        throw std::runtime_error(std::string(paramName) + " has a nonfinite tensor value");
    }
    return value;
}

inline float finiteFloat(double value, const char* paramName)
{
    if(!std::isfinite(value)
       || std::abs(value) > static_cast<double>(std::numeric_limits<float>::max()))
    {
        throw std::runtime_error(std::string(paramName)
                                 + " tensor value is outside finite float range");
    }

    static_assert(sizeof(double) == sizeof(uint64_t) && std::numeric_limits<double>::is_iec559);
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const auto sign = static_cast<uint32_t>(bits >> 32) & 0x80000000U;
    const auto exponent = static_cast<int>((bits >> 52) & 0x7ffU) - 1023;
    if(exponent < -150)
    {
        return floatFromBits(sign);
    }
    const auto significand = (bits & 0x000fffffffffffffULL) | (uint64_t{1} << 52);
    if(exponent < -126)
    {
        // Round in units of the minimum binary32 subnormal, 2^-149.
        return floatFromBits(sign
                             | static_cast<uint32_t>(roundRightToEven(
                                 significand, static_cast<unsigned int>(-exponent - 97))));
    }
    // The leading significand bit and any rounding carry add to the exponent.
    // The range check above keeps the result finite.
    return floatFromBits(sign
                         | ((static_cast<uint32_t>(exponent + 126) << 23)
                            + static_cast<uint32_t>(roundRightToEven(significand, 29))));
}

template <typename Integer, std::enable_if_t<std::is_integral_v<Integer>, int> = 0>
float finiteFloat(Integer value, const char* /*paramName*/)
{
    // Int8 tensor values are signed, so a signed char sign-extends by design.
    const auto signedValue = static_cast<int64_t>(value); // NOLINT(bugprone-signed-char-misuse)
    const bool negative = signedValue < 0;
    const auto magnitude = negative ? uint64_t{0} - static_cast<uint64_t>(signedValue)
                                    : static_cast<uint64_t>(signedValue);
    if(magnitude <= (uint64_t{1} << 24))
    {
        return static_cast<float>(signedValue); // Exact, including bool and 8-bit integers.
    }

    // Find the highest bit without converting the integer through double.
    auto leading = magnitude;
    unsigned int exponent = 0;
    for(const unsigned int shift : {32U, 16U, 8U, 4U, 2U, 1U})
    {
        if((leading >> shift) != 0)
        {
            leading >>= shift;
            exponent += shift;
        }
    }
    const auto significand = static_cast<uint32_t>(roundRightToEven(magnitude, exponent - 23));
    const auto sign = negative ? 0x80000000U : 0U;
    return floatFromBits(sign | (((exponent + 126) << 23) + significand));
}

} // namespace detail

template <typename TargetType>
TargetType extractValueFromTensorValue(const data_objects::TensorAttributesT& tensorAttr,
                                       const char* paramName)
{
    return detail::convertTensorValue(tensorAttr.data_type,
                                      tensorAttr.value.type,
                                      tensorAttr.value.value,
                                      paramName,
                                      [](auto value) { return static_cast<TargetType>(value); });
}

template <typename TargetType>
TargetType extractValueFromTensorValue(const data_objects::TensorAttributes* tensorAttr,
                                       const char* paramName)
{
    if(tensorAttr == nullptr)
    {
        throw std::runtime_error(std::string(paramName) + " tensor attribute is null");
    }
    return detail::convertTensorValue(tensorAttr->data_type(),
                                      tensorAttr->value_type(),
                                      tensorAttr->value(),
                                      paramName,
                                      [](auto value) { return static_cast<TargetType>(value); });
}

/// Convert a stored scalar to finite float32 using round-to-nearest, ties-to-even.
/// Check the dtype/union pair and double range before conversion.
/// Stored runtime defaults are read too; callers decide whether to publish them.
inline float extractFiniteFloatFromTensorValue(const data_objects::TensorAttributesT& tensorAttr,
                                               const char* paramName)
{
    return detail::convertTensorValue(
        tensorAttr.data_type,
        tensorAttr.value.type,
        tensorAttr.value.value,
        paramName,
        [paramName](auto value) { return detail::finiteFloat(value, paramName); });
}

inline float extractFiniteFloatFromTensorValue(const data_objects::TensorAttributes* tensorAttr,
                                               const char* paramName)
{
    if(tensorAttr == nullptr)
    {
        throw std::runtime_error(std::string(paramName) + " tensor attribute is null");
    }
    return detail::convertTensorValue(
        tensorAttr->data_type(),
        tensorAttr->value_type(),
        tensorAttr->value(),
        paramName,
        [paramName](auto value) { return detail::finiteFloat(value, paramName); });
}

inline double extractDoubleFromTensorValue(const data_objects::TensorAttributesT& tensorAttr,
                                           const char* paramName)
{
    return extractValueFromTensorValue<double>(tensorAttr, paramName);
}

inline double extractDoubleFromTensorValue(const data_objects::TensorAttributes* tensorAttr,
                                           const char* paramName)
{
    return extractValueFromTensorValue<double>(tensorAttr, paramName);
}

/// @brief Reads a host scalar of `dataType` from `hostPtr` and returns it as
/// TargetType. Mirrors resolveScalarOperand's dtype switch
/// (DOUBLE/FLOAT/HALF/BFLOAT16/INT32/INT64/BOOLEAN). Throws std::runtime_error
/// on UNSET/unsupported dtype or a null pointer.
template <typename TargetType>
TargetType
    readHostScalarAs(const void* hostPtr, data_objects::DataType dataType, const char* paramName)
{
    if(hostPtr == nullptr)
    {
        throw std::runtime_error(std::string(paramName) + " host scalar pointer is null");
    }

    auto readAs = [hostPtr](auto typeTag) -> TargetType {
        using SourceType = decltype(typeTag);
        SourceType value;
        std::memcpy(&value, hostPtr, sizeof(SourceType));
        return static_cast<TargetType>(value);
    };

    switch(dataType)
    {
    case data_objects::DataType::DOUBLE:
        return readAs(double{});
    case data_objects::DataType::FLOAT:
        return readAs(float{});
    case data_objects::DataType::HALF:
        return readAs(hipdnn_data_sdk::types::half{});
    case data_objects::DataType::BFLOAT16:
        return readAs(hipdnn_data_sdk::types::bfloat16{});
    case data_objects::DataType::INT32:
        return readAs(int32_t{});
    case data_objects::DataType::INT64:
        return readAs(int64_t{});
    case data_objects::DataType::BOOLEAN:
        return readAs(bool{});
    case data_objects::DataType::UNSET:
        throw std::runtime_error(std::string(paramName) + " tensor has UNSET data type");
    default:
        throw std::runtime_error(std::string(paramName) + " has unsupported data type");
    }
}

/// @brief Resolution seam mirroring resolveScalarOperand: a pure runtime
/// pass-by-value scalar (is_runtime_pass_by_value with no baked value) reads its
/// host value from the variant-pack slot for its uid; every other tensor (baked
/// compile-time constant or runtime-with-default) returns the baked graph value
/// and ignores the pack. Throws std::runtime_error if a pure-runtime scalar's
/// pack slot is missing or null.
template <typename TargetType>
TargetType resolveScalarFromVariantPack(const data_objects::TensorAttributesT& tensorAttr,
                                        const std::unordered_map<int64_t, void*>& variantPack,
                                        const char* paramName)
{
    if(tensorAttr.is_runtime_pass_by_value && tensorAttr.value.value == nullptr)
    {
        auto it = variantPack.find(tensorAttr.uid);
        if(it == variantPack.end() || it->second == nullptr)
        {
            throw std::runtime_error(
                std::string(paramName)
                + " runtime pass-by-value tensor missing host value in variant pack");
        }
        return readHostScalarAs<TargetType>(it->second, tensorAttr.data_type, paramName);
    }
    return extractValueFromTensorValue<TargetType>(tensorAttr, paramName);
}

inline double
    resolveDoubleScalarFromVariantPack(const data_objects::TensorAttributesT& tensorAttr,
                                       const std::unordered_map<int64_t, void*>& variantPack,
                                       const char* paramName)
{
    return resolveScalarFromVariantPack<double>(tensorAttr, variantPack, paramName);
}

/// @brief Reads the runtime pass-by-value flag off a serialized tensor table.
inline bool isTensorRuntimePassByValue(const data_objects::TensorAttributes* tensor)
{
    return tensor != nullptr && tensor->is_runtime_pass_by_value();
}

/// True if `tensor` is a pass-by-value scalar in ANY state (compile-time
/// constant, runtime-with-default, or pure runtime user-supplied)
inline bool isPassByValueTensor(const data_objects::TensorAttributes* tensor)
{
    return tensor != nullptr
           && (tensor->is_runtime_pass_by_value()
               || tensor->value_type() != data_objects::TensorValue::NONE);
}

/// @brief Reads the runtime pass-by-value flag off a mutable tensor object.
inline bool isTensorRuntimePassByValue(const data_objects::TensorAttributesT* tensor)
{
    return tensor != nullptr && tensor->is_runtime_pass_by_value;
}

/// @brief True if any tensor obtained by applying `project` to an element of
/// `range` is a runtime pass-by-value scalar. `project` maps an element to a
/// tensor pointer accepted by isTensorRuntimePassByValue, so the same flag
/// semantics are shared across the mutable-object graph (GraphDescriptor) and
/// the serialized-table graph (EnginePluginResourceManager).
template <typename Range, typename Project>
bool anyTensorIsRuntimePassByValue(const Range& range, Project project)
{
    for(const auto& element : range)
    {
        if(isTensorRuntimePassByValue(project(element)))
        {
            return true;
        }
    }
    return false;
}

}

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstring>
#include <flatbuffers/flatbuffers.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <optional>
#include <stdexcept>
#include <string>
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

template <typename TargetType>
TargetType extractValueFromTensorValue(const data_objects::TensorAttributesT& tensorAttr,
                                       const char* paramName)
{
    if(tensorAttr.value.value == nullptr)
    {
        throw std::runtime_error(std::string(paramName) + " must be a pass-by-value tensor");
    }

    switch(tensorAttr.data_type)
    {
    case data_objects::DataType::DOUBLE:
        if(auto val = tensorAttr.value.AsFloat64Value())
        {
            return static_cast<TargetType>(val->value());
        }
        break;
    case data_objects::DataType::FLOAT:
        if(auto val = tensorAttr.value.AsFloat32Value())
        {
            return static_cast<TargetType>(val->value());
        }
        break;
    case data_objects::DataType::HALF:
        if(auto val = tensorAttr.value.AsFloat16Value())
        {
            return static_cast<TargetType>(val->value());
        }
        break;
    case data_objects::DataType::BFLOAT16:
        if(auto val = tensorAttr.value.AsBFloat16Value())
        {
            return static_cast<TargetType>(val->value());
        }
        break;
    case data_objects::DataType::INT32:
        if(auto val = tensorAttr.value.AsInt32Value())
        {
            return static_cast<TargetType>(val->value());
        }
        break;
    case data_objects::DataType::INT64:
        if(auto val = tensorAttr.value.AsInt64Value())
        {
            return static_cast<TargetType>(val->value());
        }
        break;
    case data_objects::DataType::BOOLEAN:
        if(auto val = tensorAttr.value.AsBoolValue())
        {
            return static_cast<TargetType>(val->value());
        }
        break;
    case data_objects::DataType::UINT8:
        if(auto val = tensorAttr.value.AsFloat8Value())
        {
            return static_cast<TargetType>(val->value());
        }
        break;
    case data_objects::DataType::INT8:
        if(auto val = tensorAttr.value.AsFloat8Value())
        {
            return static_cast<TargetType>(val->value());
        }
        break;
    case data_objects::DataType::FP8_E4M3:
        if(auto val = tensorAttr.value.AsFloat8Value())
        {
            auto fp8 = hipdnn_data_sdk::types::fp8_e4m3::from_bits(val->value());
            return static_cast<TargetType>(static_cast<float>(fp8));
        }
        break;
    case data_objects::DataType::FP8_E5M2:
        if(auto val = tensorAttr.value.AsFloat8Value())
        {
            auto bfp8 = hipdnn_data_sdk::types::fp8_e5m2::from_bits(val->value());
            return static_cast<TargetType>(static_cast<float>(bfp8));
        }
        break;
    case data_objects::DataType::FP8_E4M3_FNUZ:
        if(auto val = tensorAttr.value.AsFloat8Value())
        {
            auto fp8 = hipdnn_data_sdk::types::fp8_e4m3_fnuz::from_bits(val->value());
            return static_cast<TargetType>(static_cast<float>(fp8));
        }
        break;
    case data_objects::DataType::FP8_E5M2_FNUZ:
        if(auto val = tensorAttr.value.AsFloat8Value())
        {
            auto bfp8 = hipdnn_data_sdk::types::fp8_e5m2_fnuz::from_bits(val->value());
            return static_cast<TargetType>(static_cast<float>(bfp8));
        }
        break;
    case data_objects::DataType::UNSET:
        throw std::runtime_error(std::string(paramName) + " tensor has UNSET data type");
    default:
        throw std::runtime_error(std::string(paramName) + " has unsupported data type");
    }

    throw std::runtime_error(std::string(paramName) + " must be a pass-by-value tensor");
}

template <typename TargetType>
TargetType extractValueFromTensorValue(const data_objects::TensorAttributes* tensorAttr,
                                       const char* paramName)
{
    if(tensorAttr == nullptr)
    {
        throw std::runtime_error(std::string(paramName) + " tensor attribute is null");
    }

    data_objects::TensorAttributesT unpacked;
    tensorAttr->UnPackTo(&unpacked);

    return extractValueFromTensorValue<TargetType>(unpacked, paramName);
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

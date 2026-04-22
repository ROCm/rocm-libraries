// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <optional>
#include <stdexcept>
#include <string>
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

}

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace hipdnn_sdk::utilities
{

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

// Extract value from TensorAttributes Flatbuffer pointer and cast to target type
template <typename TargetType>
TargetType extractValueFromTensorValue(const hipdnn_sdk::data_objects::TensorAttributes* tensorAttr,
                                       const char* paramName)
{
    if(tensorAttr == nullptr)
    {
        throw std::runtime_error(std::string(paramName) + " tensor attribute is null");
    }

    switch(tensorAttr->data_type())
    {
    case hipdnn_sdk::data_objects::DataType::DOUBLE:
        if(auto val = tensorAttr->value_as_Float64Value())
        {
            return static_cast<TargetType>(val->value());
        }
        break;
    case hipdnn_sdk::data_objects::DataType::FLOAT:
        if(auto val = tensorAttr->value_as_Float32Value())
        {
            return static_cast<TargetType>(val->value());
        }
        break;
    case hipdnn_sdk::data_objects::DataType::HALF:
        if(auto val = tensorAttr->value_as_Float16Value())
        {
            return static_cast<TargetType>(val->value());
        }
        break;
    case hipdnn_sdk::data_objects::DataType::BFLOAT16:
        if(auto val = tensorAttr->value_as_BFloat16Value())
        {
            return static_cast<TargetType>(val->value());
        }
        break;
    case hipdnn_sdk::data_objects::DataType::INT32:
        if(auto val = tensorAttr->value_as_Int32Value())
        {
            return static_cast<TargetType>(val->value());
        }
        break;
    case hipdnn_sdk::data_objects::DataType::UINT8:
        if(auto val = tensorAttr->value_as_Float8Value())
        {
            return static_cast<TargetType>(val->value());
        }
        break;
    case hipdnn_sdk::data_objects::DataType::UNSET:
        throw std::runtime_error(std::string(paramName) + " tensor has UNSET data type");
    default:
        throw std::runtime_error(std::string(paramName) + " has unsupported data type");
    }

    throw std::runtime_error(std::string(paramName) + " must be a pass-by-value tensor");
}

// Extract value from TensorAttributesT unpacked reference and cast to target type
template <typename TargetType>
TargetType
    extractValueFromTensorValue(const hipdnn_sdk::data_objects::TensorAttributesT& tensorAttr,
                                const char* paramName)
{
    if(tensorAttr.value.value == nullptr)
    {
        throw std::runtime_error(std::string(paramName) + " must be a pass-by-value tensor");
    }

    switch(tensorAttr.data_type)
    {
    case hipdnn_sdk::data_objects::DataType::DOUBLE:
        if(auto val = tensorAttr.value.AsFloat64Value())
        {
            return static_cast<TargetType>(val->value());
        }
        break;
    case hipdnn_sdk::data_objects::DataType::FLOAT:
        if(auto val = tensorAttr.value.AsFloat32Value())
        {
            return static_cast<TargetType>(val->value());
        }
        break;
    case hipdnn_sdk::data_objects::DataType::HALF:
        if(auto val = tensorAttr.value.AsFloat16Value())
        {
            return static_cast<TargetType>(val->value());
        }
        break;
    case hipdnn_sdk::data_objects::DataType::BFLOAT16:
        if(auto val = tensorAttr.value.AsBFloat16Value())
        {
            return static_cast<TargetType>(val->value());
        }
        break;
    case hipdnn_sdk::data_objects::DataType::INT32:
        if(auto val = tensorAttr.value.AsInt32Value())
        {
            return static_cast<TargetType>(val->value());
        }
        break;
    case hipdnn_sdk::data_objects::DataType::UINT8:
        if(auto val = tensorAttr.value.AsFloat8Value())
        {
            return static_cast<TargetType>(val->value());
        }
        break;
    case hipdnn_sdk::data_objects::DataType::UNSET:
        throw std::runtime_error(std::string(paramName) + " tensor has UNSET data type");
    default:
        throw std::runtime_error(std::string(paramName) + " has unsupported data type");
    }

    throw std::runtime_error(std::string(paramName) + " must be a pass-by-value tensor");
}

// Convenience function for extracting double values from Flatbuffer pointer
inline double
    extractDoubleFromTensorValue(const hipdnn_sdk::data_objects::TensorAttributes* tensorAttr,
                                 const char* paramName)
{
    return extractValueFromTensorValue<double>(tensorAttr, paramName);
}

// Convenience function for extracting double values from unpacked reference
inline double
    extractDoubleFromTensorValue(const hipdnn_sdk::data_objects::TensorAttributesT& tensorAttr,
                                 const char* paramName)
{
    return extractValueFromTensorValue<double>(tensorAttr, paramName);
}

}

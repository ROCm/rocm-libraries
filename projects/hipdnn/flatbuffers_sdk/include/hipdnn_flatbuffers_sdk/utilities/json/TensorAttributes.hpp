// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/json/Common.hpp>

namespace hipdnn_flatbuffers_sdk::data_objects
{

// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& tensorAttrJson,
                    const data_objects::TensorAttributes& tensorAttr)
{
    tensorAttrJson["uid"] = tensorAttr.uid();
    tensorAttrJson["data_type"] = tensorAttr.data_type();
    tensorAttrJson["dims"] = tensorAttr.dims();
    tensorAttrJson["strides"] = tensorAttr.strides();
    tensorAttrJson["name"] = flatbuffers::safeStr(tensorAttr.name());
    tensorAttrJson["virtual"] = tensorAttr.virtual_();

    // Serialize TensorValue union if present
    auto valueType = tensorAttr.value_type();
    if(valueType != data_objects::TensorValue::NONE)
    {
        tensorAttrJson["value_type"] = valueType;
        switch(valueType)
        {
        case data_objects::TensorValue::Float32Value:
            tensorAttrJson["value"] = tensorAttr.value_as_Float32Value()->value();
            break;
        case data_objects::TensorValue::Float16Value:
            tensorAttrJson["value"] = tensorAttr.value_as_Float16Value()->value();
            break;
        case data_objects::TensorValue::BFloat16Value:
            tensorAttrJson["value"] = tensorAttr.value_as_BFloat16Value()->value();
            break;
        case data_objects::TensorValue::Float8Value:
            tensorAttrJson["value"] = tensorAttr.value_as_Float8Value()->value();
            break;
        case data_objects::TensorValue::Int32Value:
            tensorAttrJson["value"] = tensorAttr.value_as_Int32Value()->value();
            break;
        case data_objects::TensorValue::Int64Value:
            tensorAttrJson["value"] = tensorAttr.value_as_Int64Value()->value();
            break;
        case data_objects::TensorValue::Float64Value:
            tensorAttrJson["value"] = tensorAttr.value_as_Float64Value()->value();
            break;
        default:
            break;
        }
    }
}

}

namespace hipdnn_flatbuffers_sdk::json
{
template <>
inline auto to<data_objects::TensorAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                               const nlohmann::json& entry)
{
    auto uid = entry.at("uid").get<int64_t>();
    auto name = entry.at("name").get<std::string>();
    auto dataType = entry.at("data_type").get<data_objects::DataType>();
    auto dims = entry.at("dims").get<std::vector<int64_t>>();
    auto strides = entry.at("strides").get<std::vector<int64_t>>();
    const bool isVirtual = entry.at("virtual").get<bool>();

    // Check if TensorValue union is present
    if(entry.contains("value_type"))
    {
        auto valueType = entry.at("value_type").get<data_objects::TensorValue>();

        // Deserialize value based on type
        flatbuffers::Offset<void> valueOffset;
        switch(valueType)
        {
        case data_objects::TensorValue::Float32Value:
        {
            auto val = entry.at("value").get<float>();
            const data_objects::Float32Value floatVal(val);
            valueOffset = builder.CreateStruct(floatVal).Union();
            break;
        }
        case data_objects::TensorValue::Float16Value:
        {
            auto val = entry.at("value").get<float>();
            const data_objects::Float16Value halfVal(val);
            valueOffset = builder.CreateStruct(halfVal).Union();
            break;
        }
        case data_objects::TensorValue::BFloat16Value:
        {
            auto val = entry.at("value").get<float>();
            const data_objects::BFloat16Value bfloatVal(val);
            valueOffset = builder.CreateStruct(bfloatVal).Union();
            break;
        }
        case data_objects::TensorValue::Float8Value:
        {
            auto val = entry.at("value").get<uint8_t>();
            const data_objects::Float8Value float8Val(val);
            valueOffset = builder.CreateStruct(float8Val).Union();
            break;
        }
        case data_objects::TensorValue::Int32Value:
        {
            auto val = entry.at("value").get<int32_t>();
            const data_objects::Int32Value intVal(val);
            valueOffset = builder.CreateStruct(intVal).Union();
            break;
        }
        case data_objects::TensorValue::Int64Value:
        {
            auto val = entry.at("value").get<int64_t>();
            const data_objects::Int64Value int64Val(val);
            valueOffset = builder.CreateStruct(int64Val).Union();
            break;
        }
        case data_objects::TensorValue::Float64Value:
        {
            auto val = entry.at("value").get<double>();
            const data_objects::Float64Value doubleVal(val);
            valueOffset = builder.CreateStruct(doubleVal).Union();
            break;
        }
        default:
            throw std::runtime_error("hipdnn_flatbuffers_sdk::json::to<TensorAttributes>(): "
                                     "Unsupported TensorValue type");
        }

        return data_objects::CreateTensorAttributes(builder,
                                                    uid,
                                                    builder.CreateString(name),
                                                    dataType,
                                                    builder.CreateVector(strides),
                                                    builder.CreateVector(dims),
                                                    isVirtual,
                                                    valueType,
                                                    valueOffset);
    }

    // No TensorValue, use the Direct version
    return data_objects::CreateTensorAttributesDirect(
        builder, uid, name.c_str(), dataType, &strides, &dims, isVirtual);
}
}

#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

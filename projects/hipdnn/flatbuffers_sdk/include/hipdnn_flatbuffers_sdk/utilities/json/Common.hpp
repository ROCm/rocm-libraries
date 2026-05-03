// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

#include <flatbuffers/flatbuffer_builder.h>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <nlohmann/detail/macro_scope.hpp>
#include <nlohmann/json.hpp>

// NOLINTBEGIN(readability-identifier-naming)

// When implicit conversions are defined, the explicit conversion function is disabled in the nlohmann json library.
// This may be unintended behavior
#ifdef JSON_USE_IMPLICIT_CONVERSIONS
NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{
template <typename BasicJsonType, typename T>
void from_json(const BasicJsonType& j, std::optional<T>& opt)
{
    if(j.is_null())
    {
        opt = std::nullopt;
    }
    else
    {
        opt.emplace(j.template get<T>());
    }
}
}
NLOHMANN_JSON_NAMESPACE_END
#endif

namespace flatbuffers
{

inline void to_json(nlohmann::json& json, const String* str)
{
    if(str == nullptr)
    {
        return;
    }

    json = str->str();
}

template <class T>
// NOLINTNEXTLINE(readability-identifier-naming)
void to_json(nlohmann::json& vectorList, const Vector<T>* vec)
{
    vectorList = nlohmann::json::array();
    if(vec == nullptr)
    {
        return;
    }

    for(auto v : *vec)
    {
        vectorList.push_back(v);
    }
}

template <class T>
// NOLINTNEXTLINE(readability-identifier-naming)
void to_json(nlohmann::json& vectorList, const Vector<Offset<T>>* vec)
{
    vectorList = nlohmann::json::array();
    if(vec == nullptr)
    {
        return;
    }
    for(auto v : *vec)
    {
        vectorList.push_back(*v);
    }
}

inline const char* safeStr(const String* s)
{
    return s != nullptr ? s->c_str() : "";
}

}

// NOLINTEND(readability-identifier-naming)

namespace hipdnn_flatbuffers_sdk::data_objects
{
NLOHMANN_JSON_SERIALIZE_ENUM(DataType,
                             {
                                 {DataType::UNSET, "unset"},
                                 {DataType::FLOAT, "float"},
                                 {DataType::HALF, "half"},
                                 {DataType::BFLOAT16, "bfloat16"},
                                 {DataType::DOUBLE, "double"},
                                 {DataType::UINT8, "uint8"},
                                 {DataType::INT32, "int32"},
                                 {DataType::INT8, "int8"},
                                 {DataType::FP8_E4M3, "fp8_e4m3"},
                                 {DataType::FP8_E5M2, "fp8_e5m2"},
                                 {DataType::FP8_E8M0, "fp8_e8m0"},
                                 {DataType::FP4_E2M1, "fp4_e2m1"},
                                 {DataType::INT4, "int4"},
                                 {DataType::FP6_E2M3, "fp6_e2m3"},
                                 {DataType::FP6_E3M2, "fp6_e3m2"},
                                 {DataType::INT64, "int64"},
                             }

)

NLOHMANN_JSON_SERIALIZE_ENUM(TensorValue,
                             {
                                 {TensorValue::NONE, "NONE"},
                                 {TensorValue::Float32Value, "Float32Value"},
                                 {TensorValue::Float16Value, "Float16Value"},
                                 {TensorValue::BFloat16Value, "BFloat16Value"},
                                 {TensorValue::Float8Value, "Float8Value"},
                                 {TensorValue::Int32Value, "Int32Value"},
                                 {TensorValue::Int64Value, "Int64Value"},
                                 {TensorValue::Float64Value, "Float64Value"},
                             }

)
}

namespace hipdnn_flatbuffers_sdk::json
{

template <class T>
inline auto to(flatbuffers::FlatBufferBuilder& builder, const nlohmann::json& entry);

template <class T>
inline auto toVector(flatbuffers::FlatBufferBuilder& builder, const nlohmann::json& entry)
{
    if(!entry.is_array())
    {
        throw std::runtime_error(
            "hipdnn_flatbuffers_sdk::json::to<vector<T>>(): field is not an array");
    }
    std::vector<flatbuffers::Offset<T>> ret;
    for(const auto& v : entry)
    {
        ret.push_back(to<T>(builder, v));
    }

    return ret;
}

}

#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <flatbuffers/flatbuffer_builder.h>
#include <hipdnn_sdk/data_objects/data_types_generated.h>
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

}

// NOLINTEND(readability-identifier-naming)

namespace hipdnn_sdk::data_objects
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
                             }

)
}

namespace hipdnn_sdk::json
{

template <class T>
inline auto to(flatbuffers::FlatBufferBuilder& builder, const nlohmann::json& entry);

template <class T>
inline auto toVector(flatbuffers::FlatBufferBuilder& builder, const nlohmann::json& entry)
{
    if(!entry.is_array())
    {
        throw std::runtime_error("hipdnn_sdk::json::to<vector<T>>(): field is not an array");
    }
    std::vector<flatbuffers::Offset<T>> ret;
    for(const auto& v : entry)
    {
        ret.push_back(to<T>(builder, v));
    }

    return ret;
}

}

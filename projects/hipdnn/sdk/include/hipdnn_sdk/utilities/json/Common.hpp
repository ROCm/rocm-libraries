// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "data_types_generated.h"
#include <flatbuffers/flatbuffer_builder.h>
#include <nlohmann/detail/macro_scope.hpp>
#include <nlohmann/json.hpp>

namespace std
{
template <class T>
// NOLINTNEXTLINE(readability-identifier-naming)
void to_json(nlohmann::json& vectorList, vector<T> const& vec)
{
    vectorList = nlohmann::json::array();
    for(auto v : vec)
    {
        vectorList.push_back(v);
    }
}

template <class T>
// NOLINTNEXTLINE(readability-identifier-naming)
void from_json(const nlohmann::json& vecJson, vector<T>& vec)
{
    if(!vecJson.is_array())
    {
        throw std::runtime_error("from_json: Attempting to deserialize non-array into vector");
    }
    vec.reserve(vecJson.size());
    for(const auto& v : vecJson)
    {
        vec.push_back(v.get<T>());
    }
}

template <class T>
// NOLINTNEXTLINE(readability-identifier-naming)
void from_json(const nlohmann::json& entry, optional<T>& opt)
{
    opt = (entry.is_null()) ? std::nullopt : std::optional<T>{entry.get<T>()};
}

}

namespace flatbuffers
{
template <class T>
// NOLINTNEXTLINE(readability-identifier-naming)
void to_json(nlohmann::json& vectorList, Vector<Offset<T>> const& vec)
{
    vectorList = nlohmann::json::array();
    for(auto v : vec)
    {
        vectorList.push_back(*v);
    }
}

}

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

template <class T, class Key>
std::optional<T> optionalValue(nlohmann::json obj, Key&& key)
{
    auto it = obj.find(std::forward<Key>(key));
    return (it != obj.end()) ? std::optional<T>(it->template get<T>()) : std::nullopt;
}

template <class T>
auto to(flatbuffers::FlatBufferBuilder& builder, nlohmann::json const& entry);

template <class T>
auto toVector(flatbuffers::FlatBufferBuilder& builder, const nlohmann::json& entry)
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

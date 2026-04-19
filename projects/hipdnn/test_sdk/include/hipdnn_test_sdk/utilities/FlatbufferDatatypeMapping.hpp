// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>

namespace hipdnn_test_sdk::utilities
{
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::half;
}

namespace hipdnn_test_sdk::utilities
{

template <hipdnn_flatbuffers_sdk::data_objects::DataType DT>
constexpr auto datatypeToNative()
{
    using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;

    if constexpr(DT == DataType::FLOAT)
    {
        return float{};
    }
    else if constexpr(DT == DataType::HALF)
    {
        return half{};
    }
    else if constexpr(DT == DataType::DOUBLE)
    {
        return double{};
    }
    else if constexpr(DT == DataType::INT32)
    {
        return int32_t{};
    }
    else if constexpr(DT == DataType::BFLOAT16)
    {
        return bfloat16{};
    }
    else
    {
        // NOLINTNEXTLINE(misc-redundant-expression) Intentional: DT != DT is a dependent false for constexpr-if
        static_assert(DT != DT, "Unsupported DataType");
    }
}

inline std::variant<float, half, double, int32_t, bfloat16>
    datatypeToNativeVariant(hipdnn_flatbuffers_sdk::data_objects::DataType type)
{
    using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;

    switch(type)
    {
    case DataType::FLOAT:
        return float{};
        break;
    case DataType::HALF:
        return half{};
        break;
    case DataType::DOUBLE:
        return double{};
        break;
    case DataType::INT32:
        return int32_t{};
        break;
    case DataType::BFLOAT16:
        return bfloat16{};
        break;
    default:
        throw std::runtime_error("Error: Invalid type");
    }
}

template <typename T>
constexpr hipdnn_flatbuffers_sdk::data_objects::DataType nativeTypeToDataType()
{
    if constexpr(std::is_same_v<T, float>)
    {
        return hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
    }
    else if constexpr(std::is_same_v<T, half>)
    {
        return hipdnn_flatbuffers_sdk::data_objects::DataType::HALF;
    }
    else if constexpr(std::is_same_v<T, double>)
    {
        return hipdnn_flatbuffers_sdk::data_objects::DataType::DOUBLE;
    }
    else if constexpr(std::is_same_v<T, int32_t>)
    {
        return hipdnn_flatbuffers_sdk::data_objects::DataType::INT32;
    }
    else if constexpr(std::is_same_v<T, bfloat16>)
    {
        return hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16;
    }
    else
    {
        static_assert(sizeof(T) == 0, "Unsupported native type");
    }
}

template <hipdnn_flatbuffers_sdk::data_objects::DataType DT>
using DataTypeToNative = decltype(datatypeToNative<DT>());

template <typename T>
using NativeToDataType = decltype(nativeTypeToDataType<T>());

}

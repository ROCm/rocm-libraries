// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/data_objects/data_types_generated.h>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>

namespace hipdnn_sdk::test_utilities
{

template <hipdnn_sdk::data_objects::DataType DT>
constexpr auto datatypeToNative()
{
    using DataType = hipdnn_sdk::data_objects::DataType;

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
        return hip_bfloat16{};
    }
    else
    {
        static_assert(DT != DT, "Unsupported DataType");
    }
}

inline std::variant<float, half, double, int32_t, hip_bfloat16>
    datatypeToNativeVariant(hipdnn_sdk::data_objects::DataType type)
{
    using DataType = hipdnn_sdk::data_objects::DataType;

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
        return hip_bfloat16{};
        break;
    default:
        throw std::runtime_error("Error: Invalid type");
    }
}

template <typename T>
constexpr hipdnn_sdk::data_objects::DataType nativeTypeToDataType()
{
    if constexpr(std::is_same_v<T, float>)
    {
        return hipdnn_sdk::data_objects::DataType::FLOAT;
    }
    else if constexpr(std::is_same_v<T, half>)
    {
        return hipdnn_sdk::data_objects::DataType::HALF;
    }
    else if constexpr(std::is_same_v<T, double>)
    {
        return hipdnn_sdk::data_objects::DataType::DOUBLE;
    }
    else if constexpr(std::is_same_v<T, int32_t>)
    {
        return hipdnn_sdk::data_objects::DataType::INT32;
    }
    else if constexpr(std::is_same_v<T, hip_bfloat16>)
    {
        return hipdnn_sdk::data_objects::DataType::BFLOAT16;
    }
    else
    {
        static_assert(sizeof(T) == 0, "Unsupported native type");
    }
}

template <hipdnn_sdk::data_objects::DataType DT>
using DataTypeToNative = decltype(datatypeToNative<DT>());

template <typename T>
using NativeToDataType = decltype(nativeTypeToDataType<T>());

}

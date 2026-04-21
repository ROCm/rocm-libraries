// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>

namespace hipdnn_test_sdk::utilities
{
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::fp4_e2m1;
using hipdnn_data_sdk::types::fp6_e2m3;
using hipdnn_data_sdk::types::fp6_e3m2;
using hipdnn_data_sdk::types::fp8_e4m3;
using hipdnn_data_sdk::types::fp8_e5m2;
using hipdnn_data_sdk::types::fp8_e8m0;
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
    else if constexpr(DT == DataType::FP8_E4M3)
    {
        return fp8_e4m3{};
    }
    else if constexpr(DT == DataType::FP8_E5M2)
    {
        return fp8_e5m2{};
    }
    else if constexpr(DT == DataType::FP8_E8M0)
    {
        return fp8_e8m0{};
    }
    else if constexpr(DT == DataType::FP4_E2M1)
    {
        return fp4_e2m1{};
    }
    else if constexpr(DT == DataType::FP6_E2M3)
    {
        return fp6_e2m3{};
    }
    else if constexpr(DT == DataType::FP6_E3M2)
    {
        return fp6_e3m2{};
    }
    else
    {
        // NOLINTNEXTLINE(misc-redundant-expression) Intentional: DT != DT is a dependent false for constexpr-if
        static_assert(DT != DT, "Unsupported DataType");
    }
}

inline std::variant<float,
                    half,
                    double,
                    int32_t,
                    bfloat16,
                    fp8_e4m3,
                    fp8_e5m2,
                    fp8_e8m0,
                    fp4_e2m1,
                    fp6_e2m3,
                    fp6_e3m2>
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
    case DataType::FP8_E4M3:
        return fp8_e4m3{};
        break;
    case DataType::FP8_E5M2:
        return fp8_e5m2{};
        break;
    case DataType::FP8_E8M0:
        return fp8_e8m0{};
        break;
    case DataType::FP4_E2M1:
        return fp4_e2m1{};
        break;
    case DataType::FP6_E2M3:
        return fp6_e2m3{};
        break;
    case DataType::FP6_E3M2:
        return fp6_e3m2{};
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
    else if constexpr(std::is_same_v<T, fp8_e4m3>)
    {
        return hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3;
    }
    else if constexpr(std::is_same_v<T, fp8_e5m2>)
    {
        return hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E5M2;
    }
    else if constexpr(std::is_same_v<T, fp8_e8m0>)
    {
        return hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E8M0;
    }
    else if constexpr(std::is_same_v<T, fp4_e2m1>)
    {
        return hipdnn_flatbuffers_sdk::data_objects::DataType::FP4_E2M1;
    }
    else if constexpr(std::is_same_v<T, fp6_e2m3>)
    {
        return hipdnn_flatbuffers_sdk::data_objects::DataType::FP6_E2M3;
    }
    else if constexpr(std::is_same_v<T, fp6_e3m2>)
    {
        return hipdnn_flatbuffers_sdk::data_objects::DataType::FP6_E3M2;
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

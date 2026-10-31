// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/types/Fp4E2M1.hpp>
#include <hipdnn_data_sdk/types/Fp6E2M3.hpp>
#include <hipdnn_data_sdk/types/Fp6E3M2.hpp>
#include <hipdnn_data_sdk/types/Fp8E4M3.hpp>
#include <hipdnn_data_sdk/types/Fp8E5M2.hpp>
#include <hipdnn_data_sdk/types/Fp8E8M0.hpp>
#include <hipdnn_data_sdk/types/Half.hpp>

#include <cstdint>

namespace hipdnn_gpu_ref::detail
{

template <typename T>
struct HipRtcTypeName;

template <>
struct HipRtcTypeName<float>
{
    static constexpr const char* VALUE = "float";
};

template <>
struct HipRtcTypeName<hipdnn_data_sdk::types::half>
{
    static constexpr const char* VALUE = "_Float16";
};

template <>
struct HipRtcTypeName<hipdnn_data_sdk::types::bfloat16>
{
    static constexpr const char* VALUE = "__bf16";
};

template <>
struct HipRtcTypeName<double>
{
    static constexpr const char* VALUE = "double";
};

template <>
struct HipRtcTypeName<int8_t>
{
    static constexpr const char* VALUE = "signed char";
};

template <>
struct HipRtcTypeName<uint8_t>
{
    static constexpr const char* VALUE = "unsigned char";
};

template <>
struct HipRtcTypeName<int32_t>
{
    static constexpr const char* VALUE = "int";
};

template <>
struct HipRtcTypeName<hipdnn_data_sdk::types::fp8_e4m3>
{
    static constexpr const char* VALUE = "__hip_fp8_e4m3";
};

template <>
struct HipRtcTypeName<hipdnn_data_sdk::types::fp4_e2m1>
{
    static constexpr const char* VALUE = "__hip_fp4_e2m1";
};

template <>
struct HipRtcTypeName<hipdnn_data_sdk::types::fp6_e2m3>
{
    static constexpr const char* VALUE = "__hip_fp6_e2m3";
};

template <>
struct HipRtcTypeName<hipdnn_data_sdk::types::fp6_e3m2>
{
    static constexpr const char* VALUE = "__hip_fp6_e3m2";
};

template <>
struct HipRtcTypeName<hipdnn_data_sdk::types::fp8_e5m2>
{
    static constexpr const char* VALUE = "__hip_fp8_e5m2";
};

template <>
struct HipRtcTypeName<hipdnn_data_sdk::types::fp8_e8m0>
{
    static constexpr const char* VALUE = "__hip_fp8_e8m0";
};

} // namespace hipdnn_gpu_ref::detail

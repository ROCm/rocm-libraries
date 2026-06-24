// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/types/Fp8E4M3.hpp>
#include <hipdnn_data_sdk/types/Fp8E4M3Fnuz.hpp>
#include <hipdnn_data_sdk/types/Fp8E5M2.hpp>
#include <hipdnn_data_sdk/types/Fp8E5M2Fnuz.hpp>
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

// fp8 inputs decode to float in software via the matching gpu_ref:: storage structs
// defined in the self-contained kernel header GpuRefTypes.h.
template <>
struct HipRtcTypeName<hipdnn_data_sdk::types::fp8_e4m3>
{
    static constexpr const char* VALUE = "gpu_ref::fp8_e4m3";
};

template <>
struct HipRtcTypeName<hipdnn_data_sdk::types::fp8_e5m2>
{
    static constexpr const char* VALUE = "gpu_ref::fp8_e5m2";
};

template <>
struct HipRtcTypeName<hipdnn_data_sdk::types::fp8_e4m3_fnuz>
{
    static constexpr const char* VALUE = "gpu_ref::fp8_e4m3_fnuz";
};

template <>
struct HipRtcTypeName<hipdnn_data_sdk::types::fp8_e5m2_fnuz>
{
    static constexpr const char* VALUE = "gpu_ref::fp8_e5m2_fnuz";
};

} // namespace hipdnn_gpu_ref::detail

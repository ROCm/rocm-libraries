// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/types/Fp8E4M3.hpp>
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

// fp8 E4M3 (OCP) maps to the self-contained device decode type defined in GpuRefTypes.h
// (gpu_ref::GpuRefFp8E4M3); the SDPA kernel's `using namespace gpu_ref` brings it into scope.
template <>
struct HipRtcTypeName<hipdnn_data_sdk::types::fp8_e4m3>
{
    static constexpr const char* VALUE = "GpuRefFp8E4M3";
};

} // namespace hipdnn_gpu_ref::detail

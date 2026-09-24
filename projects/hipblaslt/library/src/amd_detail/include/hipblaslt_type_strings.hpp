// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Type names as hipblaslt-bench and the tuning file spell them. Kept apart from
// auxiliary.hpp, whose FP8 helpers need HIP compilation, so that host-only code
// can parse them too.

#include <hipblaslt/hipblaslt.h>

#include <string>

// clang-format off
HIPBLASLT_EXPORT
constexpr hipDataType string_to_hip_datatype(const std::string& value)
{
    if (value == "f8_fnuz_r")
    {
        return HIP_R_8F_E4M3_FNUZ;
    }
    else if (value == "bf8_fnuz_r")
    {
        return HIP_R_8F_E5M2_FNUZ;
    }

    if (value == "f8_r")
    {
        return HIP_R_8F_E4M3;
    }
    else if (value == "bf8_r")
    {
        return HIP_R_8F_E5M2;
    }

    return
        value == "f32_r" || value == "s" ? HIP_R_32F :
        value == "f64_r" || value == "d" ? HIP_R_64F :
        value == "f16_r" || value == "h" ? HIP_R_16F :
        value == "bf16_r"                ? HIP_R_16BF :
        value == "i8_r" || value == "i8" ? HIP_R_8I :
        value == "f32_c" || value == "c" ? HIP_C_32F  :
        value == "f64_c" || value == "z" ? HIP_C_64F  :
        value == "f6_r"                  ? static_cast<hipDataType>(HIP_R_6F_E2M3) :
        value == "bf6_r"                 ? static_cast<hipDataType>(HIP_R_6F_E3M2) :
        value == "f4_r"                  ? static_cast<hipDataType>(HIP_R_4F_E2M1) :
        value == "i32_r" || value == "i" ? HIP_R_32I :
        value == "f8_fnuz_r"             ? HIP_R_8F_E4M3_FNUZ :
        value == "bf8_fnuz_r"            ? HIP_R_8F_E5M2_FNUZ :
        value == "f8_r"                  ? HIP_R_8F_E4M3 :
        value == "bf8_r"                 ? HIP_R_8F_E5M2 :
        value == "e8_r"                  ? HIP_R_8F_UE8M0 :
        value == "e5m3_r"                ? static_cast<hipDataType>(HIP_R_8F_E5M3_EXT) :
        HIPBLASLT_DATATYPE_INVALID;
}

HIPBLASLT_EXPORT
constexpr hipblasComputeType_t string_to_hipblas_computetype(const std::string& value)
{
    return
        value == "f32_r" || value == "s" || value == "c" || value == "f32_c" ?  HIPBLAS_COMPUTE_32F  :
        value == "xf32_r" || value == "x" ? HIPBLAS_COMPUTE_32F_FAST_TF32 :
        value == "f64_r" || value == "d" || value == "z" || value == "f64_c" ? HIPBLAS_COMPUTE_64F :
        value == "i32_r" || value == "i" ? HIPBLAS_COMPUTE_32I :
        value == "f32_f16_r" ? HIPBLAS_COMPUTE_32F_FAST_16F :
        value == "f32_bf16_r" ? HIPBLAS_COMPUTE_32F_FAST_16BF :
        HIPBLASLT_COMPUTE_TYPE_INVALID;
}
// clang-format on

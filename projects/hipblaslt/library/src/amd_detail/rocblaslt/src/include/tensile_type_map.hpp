// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocblaslt-types.h"

#include <rocisa/include/enum.hpp>

#include <cassert>

/******************************************************
 * Map a hipblaslt data type to a corresponding Tensile type *
 ******************************************************/
inline rocisa::DataType hipDataType_to_tensile_type(hipDataType type)
{
    switch(type)
    {
    case HIP_R_16F:
        return rocisa::DataType::Half;
    case HIP_R_32F:
        return rocisa::DataType::Float;
    case HIP_R_64F:
        return rocisa::DataType::Double;
    case HIP_R_16BF:
        return rocisa::DataType::BFloat16;
    case HIP_R_8F_E4M3_FNUZ:
        return rocisa::DataType::Float8_fnuz;
    case HIP_R_8F_E5M2_FNUZ:
        return rocisa::DataType::BFloat8_fnuz;
    case HIP_R_8F_E4M3:
        return rocisa::DataType::Float8;
    case HIP_R_8F_E5M2:
        return rocisa::DataType::BFloat8;
    case HIP_R_8I:
        return rocisa::DataType::Int8;
    case HIP_R_32I:
        return rocisa::DataType::Int32;
    case HIP_C_32F:
        return rocisa::DataType::ComplexFloat;
    case HIP_C_64F:
        return rocisa::DataType::ComplexDouble;
    // MX 6/4 data types
    case HIP_R_6F_E2M3:
        return rocisa::DataType::Float6;
    case HIP_R_6F_E3M2:
        return rocisa::DataType::BFloat6;
    case HIP_R_4F_E2M1:
        return rocisa::DataType::Float4;
    default:
        assert(!"hipDataType_to_tensile_type: non-supported type");
        return rocisa::DataType::None;
    }
}

inline rocisa::DataType rocComputeType_to_tensile_type(rocblaslt_compute_type type)
{
    switch(type)
    {
    case rocblaslt_compute_f32_fast_xf32:
        return rocisa::DataType::XFloat32;
    case rocblaslt_compute_f16:
    case rocblaslt_compute_f32:
    case rocblaslt_compute_f32_fast_f16:
    case rocblaslt_compute_f32_fast_bf16:
    case rocblaslt_compute_f32_fast_f8_fnuz:
    case rocblaslt_compute_f32_fast_bf8_fnuz:
    case rocblaslt_compute_f32_fast_f8bf8_fnuz:
    case rocblaslt_compute_f32_fast_bf8f8_fnuz:
    case rocblaslt_compute_f32_fast_f8:
    case rocblaslt_compute_f32_fast_bf8:
    case rocblaslt_compute_f32_fast_f8bf8:
    case rocblaslt_compute_f32_fast_bf8f8:
        return rocisa::DataType::Float;
    case rocblaslt_compute_f64:
        return rocisa::DataType::Double;
    case rocblaslt_compute_i32:
        return rocisa::DataType::Int32;
    default:
        assert(!"rocDataType_to_tensile_type: non-supported type");
        return rocisa::DataType::None;
    }
}

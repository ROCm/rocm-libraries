// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*********************************************************
 * Helper for describing a kernel's operands to Origami. *
 *********************************************************/

#pragma once

#include <hipblaslt/hipblaslt.h>

#include "origami/types.hpp"

/**
 * @brief Convert hipDataType to analytical::DataType
 */
inline origami::data_type_t hipDataType_to_analytical_type(hipDataType type)
{
    switch(type)
    {
    case HIP_R_16F:
        return origami::data_type_t::Half;
    case HIP_R_32F:
        return origami::data_type_t::Float;
    case HIP_R_16BF:
        return origami::data_type_t::BFloat16;
    case HIP_R_8F_E4M3:
        return origami::data_type_t::Float8;
    case HIP_R_8F_E5M2:
        return origami::data_type_t::BFloat8;
    case HIP_R_6F_E2M3:
        return origami::data_type_t::Float6;
    case HIP_R_6F_E3M2:
        return origami::data_type_t::BFloat6;
    case HIP_R_4F_E2M1:
        return origami::data_type_t::Float4;
    default:
        return origami::data_type_t::None;
    }
}

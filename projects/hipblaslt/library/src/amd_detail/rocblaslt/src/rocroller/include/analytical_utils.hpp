// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*********************************************************
 * Helper for describing a kernel's operands to Origami. *
 *********************************************************/

#pragma once

#include "kernel_data_types.hpp"

#include "origami/types.hpp"

/**
 * @brief Convert hipblaslt::DataType to analytical::DataType
 */
inline origami::data_type_t kernel_data_type_to_analytical_type(hipblaslt::DataType type)
{
    switch(type)
    {
        case hipblaslt::DataType::Half:
            return origami::data_type_t::Half;
        case hipblaslt::DataType::Float:
            return origami::data_type_t::Float;
        case hipblaslt::DataType::BFloat16:
            return origami::data_type_t::BFloat16;
        case hipblaslt::DataType::FP8:
            return origami::data_type_t::Float8;
        case hipblaslt::DataType::BF8:
            return origami::data_type_t::BFloat8;
        case hipblaslt::DataType::FP6:
            return origami::data_type_t::Float6;
        case hipblaslt::DataType::BF6:
            return origami::data_type_t::BFloat6;
        case hipblaslt::DataType::FP4:
            return origami::data_type_t::Float4;
        default:
            return origami::data_type_t::None;
    }
}

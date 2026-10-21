// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*********************************************************
 * Helper for using Origami with rocRoller. *
 *********************************************************/

#pragma once

// See kernel_type.hpp for why this is a compile definition and not
// __has_include.
#ifdef HIPBLASLT_HAVE_ROCROLLER_HEADERS
#include <rocRoller/DataTypes/DataTypes.hpp>
#else
#include "rocroller_enums.hpp"
#endif

#include "origami/types.hpp"

/**
 * @brief Convert rocRoller::Datatype to analytical::DataType
 */
inline origami::data_type_t rocroller_type_to_analytical_type(rocRoller::DataType type)
{
    switch(type)
    {
        case rocRoller::DataType::Half:
            return origami::data_type_t::Half;
        case rocRoller::DataType::Float:
            return origami::data_type_t::Float;
        case rocRoller::DataType::BFloat16:
            return origami::data_type_t::BFloat16;
        case rocRoller::DataType::FP8:
            return origami::data_type_t::Float8;
        case rocRoller::DataType::BF8:
            return origami::data_type_t::BFloat8;
        case rocRoller::DataType::FP6:
            return origami::data_type_t::Float6;
        case rocRoller::DataType::BF6:
            return origami::data_type_t::BFloat6;
        case rocRoller::DataType::FP4:
            return origami::data_type_t::Float4;
        default:
            return origami::data_type_t::None;
    }
}

/*! \file */
/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

/*********************************************************
 * Helper for using Origami with rocRoller. *
 *********************************************************/

#pragma once

#include <rocRoller/DataTypes/DataTypes.hpp>

#include <origami/utils.hpp>

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

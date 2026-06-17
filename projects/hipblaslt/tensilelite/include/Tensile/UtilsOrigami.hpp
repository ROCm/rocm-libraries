/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once
#include <stdexcept>
#include <origami/origami.hpp>
#include <rocisa/include/enum.hpp>
#include <origami/simulator/tensilelite/formocast_simulator.hpp>

#include <Tensile/Macros.hpp>

TENSILE_HIDDEN_BEGIN

namespace TensileLite
{

    inline origami::data_type_t datatypeToAnalyticalDatatype(rocisa::DataType type)
    {
        switch(type)
        {
        case rocisa::DataType::Float:
            return origami::data_type_t::Float;
        case rocisa::DataType::Double:
            return origami::data_type_t::Double;
        case rocisa::DataType::ComplexFloat:
            return origami::data_type_t::ComplexFloat;
        case rocisa::DataType::ComplexDouble:
            return origami::data_type_t::ComplexDouble;
        case rocisa::DataType::Half:
            return origami::data_type_t::Half;
        case rocisa::DataType::Int8x4:
            return origami::data_type_t::Int8x4;
        case rocisa::DataType::Int32:
            return origami::data_type_t::Int32;
        case rocisa::DataType::BFloat16:
            return origami::data_type_t::BFloat16;
        case rocisa::DataType::Int8:
            return origami::data_type_t::Int8;
        case rocisa::DataType::Int64:
            return origami::data_type_t::Int64;
        case rocisa::DataType::XFloat32:
            return origami::data_type_t::XFloat32;
        case rocisa::DataType::Float8_fnuz:
            return origami::data_type_t::Float8_fnuz;
        case rocisa::DataType::BFloat8_fnuz:
            return origami::data_type_t::BFloat8_fnuz;
        case rocisa::DataType::Float8BFloat8_fnuz:
            return origami::data_type_t::Float8BFloat8_fnuz;
        case rocisa::DataType::BFloat8Float8_fnuz:
            return origami::data_type_t::BFloat8Float8_fnuz;
        case rocisa::DataType::Float8:
            return origami::data_type_t::Float8;
        case rocisa::DataType::BFloat8:
            return origami::data_type_t::BFloat8;
        case rocisa::DataType::Float8BFloat8:
            return origami::data_type_t::Float8BFloat8;
        case rocisa::DataType::BFloat8Float8:
            return origami::data_type_t::BFloat8Float8;
        case rocisa::DataType::Float6:
            return origami::data_type_t::Float6;
        case rocisa::DataType::BFloat6:
            return origami::data_type_t::BFloat6;   
        case rocisa::DataType::Float4:
            return origami::data_type_t::Float4;

        default:
            throw std::runtime_error("Unsupported data type: " + std::to_string(static_cast<int>(type)));
        }
    }
} // namespace TensileLite

TENSILE_HIDDEN_END

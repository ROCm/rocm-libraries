/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
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
#include "instruction/mfma.hpp"

namespace rocisa
{
    DataType instTypeToDataType(InstType instType)
    {
        switch(instType)
        {
        case InstType::INST_F16:
            return DataType::Half;
        case InstType::INST_F32:
            return DataType::Float;
        case InstType::INST_F64:
            return DataType::Double;
        case InstType::INST_BF16:
            return DataType::BFloat16;
        case InstType::INST_I8:
            return DataType::Int8;
        case InstType::INST_U8:
            return DataType::Int8;
        case InstType::INST_I32:
            return DataType::Int32;
        case InstType::INST_XF32:
            return DataType::XFloat32;
        case InstType::INST_F8:
            return DataType::Float8;
        case InstType::INST_BF8:
            return DataType::BFloat8;
        case InstType::INST_F8_BF8:
            return DataType::Float8BFloat8;
        case InstType::INST_BF8_F8:
            return DataType::BFloat8Float8;
        case InstType::INST_F6:
            return DataType::Float6;
        case InstType::INST_BF6:
            return DataType::BFloat6;
        // TODO: check IssueLatency for mixmode.
        case InstType::INST_F6_B6:
            return DataType::Float6;
        case InstType::INST_B6_F6:
            return DataType::BFloat6;
        case InstType::INST_F4:
            return DataType::Float4;
        case InstType::INST_F8_F4:
        case InstType::INST_F4_F8:
        case InstType::INST_F6_F4:
        case InstType::INST_F4_F6:
        case InstType::INST_F8_F6:
        case InstType::INST_F6_F8:
        case InstType::INST_F8_B6:
        case InstType::INST_B6_F8:
            return DataType::Float8;
        case InstType::INST_B8_F4:
        case InstType::INST_F4_B8:
        case InstType::INST_B6_F4:
        case InstType::INST_F4_B6:
        case InstType::INST_B8_F6:
        case InstType::INST_F6_B8:
        case InstType::INST_B8_B6:
        case InstType::INST_B6_B8:
            return DataType::BFloat8;
        default:
            throw std::runtime_error("Unknown instruction type");
        }
    }

    bool is8bitFloat(DataType value)
    {
        switch(value)
        {
        case DataType::Float8:
        case DataType::BFloat8:
        case DataType::Float8BFloat8:
        case DataType::BFloat8Float8:
        case DataType::Float8_fnuz:
        case DataType::BFloat8_fnuz:
        case DataType::Float8BFloat8_fnuz:
        case DataType::BFloat8Float8_fnuz:
            return true;
        default:
            return false;
        }
    }
} // namespace rocisa

/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#include <rocRoller/Operations/T_Mul.hpp>

#include <fmt/core.h>

namespace rocRoller
{
    namespace Operations
    {
        inline T_Mul::T_Mul(OperationTag a, OperationTag b)
            : BaseOperation()
            , a(a)
            , b(b)
        {
        }

        inline std::unordered_set<OperationTag> T_Mul::getInputs() const
        {
            return {a, b};
        }

        inline std::string T_Mul::toString() const
        {
            return fmt::format("T_Mul {} {}", a.value, b.value);
        }

        inline bool T_Mul::operator==(T_Mul const& rhs) const
        {
            return m_tag == rhs.m_tag && a == rhs.a && b == rhs.b;
        }
    }
}

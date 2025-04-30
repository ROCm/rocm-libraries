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

#include "LabelAllocator.hpp"

#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    inline LabelAllocator::LabelAllocator(std::string prefix)
        : m_prefix(std::move(prefix))
    {
    }

    inline Register::ValuePtr LabelAllocator::label(std::string name)
    {
        name    = escapeSymbolName(std::move(name));
        auto rv = fmt::format("{}_{}", m_prefix, name);

        while(m_generatedLabels.contains(rv))
        {
            rv = fmt::format("{}_{}_{}", m_prefix, m_count, name);
            m_count++;
        }
        m_generatedLabels.insert(rv);
        return Register::Value::Label(rv);
    }
}

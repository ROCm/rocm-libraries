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

/**
 * T_Mul (tensor/matrix multiply) command.
 */

#pragma once

#include <memory>
#include <unordered_set>

#include <rocRoller/Operations/Operation.hpp>
#include <rocRoller/Serialization/Base_fwd.hpp>

namespace rocRoller
{
    namespace Operations
    {
        class T_Mul : public BaseOperation
        {
        public:
            T_Mul() = delete;
            T_Mul(OperationTag a, OperationTag b);

            std::unordered_set<OperationTag> getInputs() const;
            std::string                      toString() const;

            OperationTag a, b;

            bool operator==(T_Mul const&) const;

            template <typename T1, typename T2, typename T3>
            friend struct rocRoller::Serialization::MappingTraits;
        };

    }
}

#include <rocRoller/Operations/T_Mul_impl.hpp>

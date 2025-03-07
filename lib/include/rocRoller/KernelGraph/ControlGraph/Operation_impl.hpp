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

#include <rocRoller/KernelGraph/ControlGraph/Operation.hpp>

namespace rocRoller
{
    namespace KernelGraph::ControlGraph
    {
        template <CConcreteOperation Op>
        inline std::string name(const Op& x)
        {
            return x.name();
        }

        /*
         * Helpers
         */
        inline std::string name(const Operation& x)
        {
            return std::visit([](const auto& a) { return a.name(); }, x);
        }

        struct OperationToStringVisitor
        {
            template <CHasToStringMember Op>
            std::string operator()(Op const& op)
            {
                return op.toString();
            }

            template <typename Op>
            requires(!CHasToStringMember<Op>) std::string operator()(Op const& op)
            {
                return op.name();
            }
        };

        inline std::string toString(const Operation& x)
        {
            return std::visit(OperationToStringVisitor{}, x);
        }

        template <typename T>
        concept CHasDataTypeMember = requires(T const& op)
        {
            {
                op.dataType
                } -> std::convertible_to<DataType>;
        };

        template <typename T>
        concept CHasVarTypeMember = requires(T const& op)
        {
            {
                op.varType
                } -> std::convertible_to<VariableType>;
        };

        struct OperationDataTypeVisitor
        {
            template <CHasDataTypeMember Op>
            DataType operator()(Op const& op)
            {
                return op.dataType;
            }

            template <CHasVarTypeMember Op>
            DataType operator()(Op const& op)
            {
                return op.varType.dataType;
            }

            template <typename Op>
            requires(!CHasDataTypeMember<Op> && !CHasVarTypeMember<Op>) DataType
                operator()(Op const& op)
            {
                return DataType::None;
            }
        };

        inline DataType getDataType(const Operation& x)
        {
            return std::visit(OperationDataTypeVisitor(), x);
        }
    }
}

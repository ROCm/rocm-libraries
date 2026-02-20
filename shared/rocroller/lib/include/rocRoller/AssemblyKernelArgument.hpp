/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2026 AMD ROCm(TM) Software
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

#include <string>

#include <rocRoller/Expression.hpp>

namespace rocRoller
{

    class AssemblyKernelArgument
    {
    public:
        AssemblyKernelArgument() = default;

        AssemblyKernelArgument(std::string               name,
                               VariableType              variableType,
                               DataDirection             dataDir = DataDirection::ReadOnly,
                               Expression::ExpressionPtr expr    = nullptr,
                               int                       offset  = -1,
                               int                       size    = -1);

        bool operator==(AssemblyKernelArgument const&) const;

        std::string toString() const;

        std::string getName() const
        {
            return m_name;
        }
        void setName(std::string name)
        {
            m_name = name;
        }

        VariableType getVariableType() const
        {
            return m_variableType;
        }
        void setVariableType(VariableType variableType)
        {
            m_variableType = variableType;
        }

        DataDirection getDataDirection() const
        {
            return m_dataDirection;
        }
        void setDataDirection(DataDirection dataDir)
        {
            m_dataDirection = dataDir;
        }

        int getOffset() const
        {
            return m_offset;
        }
        void setOffset(int offset)
        {
            m_offset = offset;
        }

        int getSize() const
        {
            return m_size;
        }
        void setSize(int size)
        {
            m_size = size;
        }

        Expression::ExpressionPtr const& getExpression() const
        {
            return m_expression;
        }
        void setExpression(Expression::ExpressionPtr const& expr);

        Expression::ExpressionPtr const& getSimplifiedExpr() const
        {
            return m_simplifiedExpr;
        }
        Expression::ExpressionPtr const& getRestoredExpr() const
        {
            return m_restoredExpr;
        }
        Expression::ExpressionPtr const& getSimplifiedRestoredExpr() const
        {
            return m_simplifiedRestoredExpr;
        }

    private:
        template <typename T1, typename T2, typename T3>
        friend struct rocRoller::Serialization::MappingTraits;

        std::string   m_name;
        VariableType  m_variableType;
        DataDirection m_dataDirection = DataDirection::ReadOnly;

        Expression::ExpressionPtr m_expression = nullptr;

        Expression::ExpressionPtr m_simplifiedExpr         = nullptr;
        Expression::ExpressionPtr m_restoredExpr           = nullptr;
        Expression::ExpressionPtr m_simplifiedRestoredExpr = nullptr;

        int m_offset = -1;
        int m_size   = -1;
    };

    std::ostream& operator<<(std::ostream& stream, AssemblyKernelArgument const& arg);
}

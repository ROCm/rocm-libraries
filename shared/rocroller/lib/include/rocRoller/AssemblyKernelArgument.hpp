// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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

        bool getPreloaded() const
        {
            return m_preloaded;
        }
        void setPreloaded(bool value)
        {
            m_preloaded = value;
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

        bool m_preloaded = false;
    };

    std::ostream& operator<<(std::ostream& stream, AssemblyKernelArgument const& arg);
}

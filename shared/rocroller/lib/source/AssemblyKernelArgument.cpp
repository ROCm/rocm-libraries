// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <string>

#include <rocRoller/AssemblyKernelArgument.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    AssemblyKernelArgument::AssemblyKernelArgument(std::string               name,
                                                   VariableType              variableType,
                                                   DataDirection             dataDir,
                                                   Expression::ExpressionPtr expr,
                                                   int                       offset,
                                                   int                       size)
        : m_name(name)
        , m_variableType(variableType)
        , m_dataDirection(dataDir)
        , m_offset(offset)
        , m_size(size)
    {
        setExpression(expr);
    }

    void AssemblyKernelArgument::setExpression(Expression::ExpressionPtr const& expr)
    {
        m_expression     = expr;
        m_simplifiedExpr = simplify(m_expression);
        if(identical(m_simplifiedExpr, m_expression))
            m_simplifiedExpr = nullptr;

        m_restoredExpr = Expression::restoreCommandArguments(m_expression);
        if(identical(m_restoredExpr, m_expression))
            m_restoredExpr = nullptr;

        if(m_restoredExpr)
        {
            m_simplifiedRestoredExpr = simplify(m_restoredExpr);
            if(identical(m_simplifiedRestoredExpr, m_restoredExpr))
                m_simplifiedRestoredExpr = nullptr;
        }
    }

    bool AssemblyKernelArgument::operator==(AssemblyKernelArgument const& rhs) const
    {
        return m_name == rhs.m_name //
               && m_variableType == rhs.m_variableType //
               && m_dataDirection == rhs.m_dataDirection //
               && equivalent(m_expression, rhs.m_expression) //
               && m_offset == rhs.m_offset //
               && m_size == rhs.m_size && m_preloaded == rhs.m_preloaded;
    }

    std::string AssemblyKernelArgument::toString() const
    {
        auto rv = concatenate("KernelArg{", m_name, ", ", m_variableType);

        if(m_dataDirection != DataDirection::ReadOnly)
            rv += concatenate(", ", m_dataDirection);

        rv += concatenate(", ", m_expression);
        if(m_expression)
            rv += concatenate("(c ", complexity(m_expression), ")");

        if(m_offset != -1)
            rv += concatenate(", o:", m_offset);

        if(m_size != -1)
            rv += concatenate(", s:", m_size);

        if(m_preloaded)
            rv += concatenate(", preloaded");

        return rv + "}";
    }

    std::ostream& operator<<(std::ostream& stream, AssemblyKernelArgument const& arg)
    {
        return stream << arg.toString();
    }
}

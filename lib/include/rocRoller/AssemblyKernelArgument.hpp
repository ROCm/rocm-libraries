/**
 */

#pragma once

#include <string>

#include <rocRoller/Expression.hpp>

namespace rocRoller
{

    struct AssemblyKernelArgument
    {
        std::string   name;
        VariableType  variableType;
        DataDirection dataDirection = DataDirection::ReadOnly;

        Expression::ExpressionPtr expression = nullptr;

        int offset = -1;
        int size   = -1;

        bool operator==(AssemblyKernelArgument const&) const;

        std::string toString() const;
    };

    std::ostream& operator<<(std::ostream& stream, AssemblyKernelArgument const& arg);
}

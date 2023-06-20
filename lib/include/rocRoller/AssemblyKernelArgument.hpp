/**
 */

#pragma once

#include <string>

#include "Expression.hpp"

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

        bool operator==(AssemblyKernelArgument const&) const = default;
    };
}

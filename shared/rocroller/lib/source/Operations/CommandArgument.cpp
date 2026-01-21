// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Expression.hpp>
#include <rocRoller/Operations/CommandArgument.hpp>

namespace rocRoller
{
    Expression::ExpressionPtr CommandArgument::expression()
    {
        return std::make_shared<Expression::Expression>(shared_from_this());
    }
}

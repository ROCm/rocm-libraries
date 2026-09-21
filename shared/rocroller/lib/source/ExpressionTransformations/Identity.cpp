// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Expression_fwd.hpp>

namespace rocRoller
{
    namespace Expression
    {
        ExpressionPtr identity(ExpressionPtr expr)
        {
            return expr;
        }
    }
}

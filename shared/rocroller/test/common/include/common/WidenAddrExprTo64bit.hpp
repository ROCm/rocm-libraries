// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Expression.hpp>
namespace rocRollerTest
{
    /**
        * @brief Widen an expression from (u)int32 to (u)int64.
        *
        * The implementation is not comprehensive (yet and intentionally).
        *
        * This is a customized expression transformation that is intended
        * to be used only for expressions of calculating addresses in order to test
        * the correctness of address calculation.
        * Top-level expr input given to this function should be 64bit since it is
        * representing an address. Any contained expressions are promoted to 64bit
        * as leaf expressions (Kernel or Command Argument, Register::ValuePtr)
        * are promoted to 64bit.
        *
        * @param expr Input expression. Its resultType should be 64bit.
        * @return ExpressionPtr Transformed expression
        */
    rocRoller::Expression::ExpressionPtr
        widenAddrExprTo64bit(rocRoller::Expression::ExpressionPtr expr);
}

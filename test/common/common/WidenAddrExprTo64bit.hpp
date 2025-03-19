/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

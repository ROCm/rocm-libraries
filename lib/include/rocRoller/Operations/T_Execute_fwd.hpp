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

#include <string>
#include <variant>

namespace rocRoller
{
    namespace Operations
    {
        struct T_Execute;
        struct E_Unary;
        struct E_Binary;
        struct E_Ternary;
        struct E_Neg;
        struct E_Abs;
        struct E_RandomNumber;
        struct E_Not;
        struct E_Cvt;
        struct E_StochasticRoundingCvt;
        struct E_Add;
        struct E_Sub;
        struct E_Mul;
        struct E_Div;
        struct E_And;
        struct E_Or;
        struct E_GreaterThan;
        struct E_Conditional;

        using XOp = std::variant<E_Neg,
                                 E_Abs,
                                 E_Not,
                                 E_RandomNumber,
                                 E_Cvt,
                                 E_StochasticRoundingCvt,
                                 E_Add,
                                 E_Sub,
                                 E_Mul,
                                 E_Div,
                                 E_And,
                                 E_Or,
                                 E_GreaterThan,
                                 E_Conditional>;

        template <typename T>
        concept CXOp = requires()
        {
            requires std::constructible_from<XOp, T>;
            requires !std::same_as<XOp, T>;
        };

        std::string name(XOp const&);

        template <CXOp T>
        std::string name();
    }
}

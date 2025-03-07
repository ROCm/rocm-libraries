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

#include <variant>

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/InstructionValues/Register_fwd.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/Utilities/Timer.hpp>

namespace rocRoller
{
    namespace Expression
    {
        /*
         * to string
         */

        struct ExpressionToStringVisitor
        {
            template <CTernary Expr>
            std::string operator()(Expr const& expr) const
            {
                return concatenate(ExpressionInfo<Expr>::name(),
                                   "(",
                                   call(expr.lhs),
                                   ", ",
                                   call(expr.r1hs),
                                   ", ",
                                   call(expr.r2hs),
                                   ")");
            }

            template <CBinary Expr>
            std::string operator()(Expr const& expr) const
            {
                return concatenate(
                    ExpressionInfo<Expr>::name(), "(", call(expr.lhs), ", ", call(expr.rhs), ")");
            }

            template <CUnary Expr>
            std::string operator()(Expr const& expr) const
            {
                return concatenate(ExpressionInfo<Expr>::name(), "(", call(expr.arg), ")");
            }

            std::string operator()(BitFieldExtract const& expr) const
            {
                return concatenate(ExpressionInfo<BitFieldExtract>::name(),
                                   "(",
                                   call(expr.arg),
                                   ", width:",
                                   expr.width,
                                   ", offset:",
                                   expr.offset,
                                   ")");
            }

            std::string operator()(Register::ValuePtr const& expr) const
            {
                // This allows an unallocated register value to be rendered into a string which
                // improves debugging by allowing the string representation of that expression
                // to be put into the source file as a comment.
                // Trying to generate the code for the expression will throw an exception.

                std::string tostr = "UNALLOCATED";
                if(expr->canUseAsOperand())
                    tostr = expr->toString();

                // The call() function appends the result type, so add ":" to separate the
                // value from the type.
                return tostr + ":";
            }

            std::string operator()(ScaledMatrixMultiply const& expr) const
            {
                return concatenate("ScaledMatrixMultiply(",
                                   call(expr.matA),
                                   ", ",
                                   call(expr.matB),
                                   ", ",
                                   call(expr.matC),
                                   ", ",
                                   call(expr.scaleA),
                                   ", ",
                                   call(expr.scaleB),
                                   ")");
            }

            std::string operator()(CommandArgumentPtr const& expr) const
            {
                if(expr)
                    return concatenate("CommandArgument(", expr->name(), ")");
                else
                    return "CommandArgument(nullptr)";
            }

            std::string operator()(CommandArgumentValue const& expr) const
            {
                return std::visit([](auto const& val) { return concatenate(val, ":"); }, expr);
            }

            std::string operator()(AssemblyKernelArgumentPtr const& expr) const
            {
                // The call() function appends the result type, so add ":" to separate the
                // value from the type.
                return expr->name + ":";
            }

            std::string operator()(WaveTilePtr const& expr) const
            {
                return "WaveTile";
            }

            std::string operator()(DataFlowTag const& expr) const
            {
                return concatenate("DataFlowTag(", expr.tag, ")");
            }

            std::string call(Expression const& expr) const
            {
                auto functionalPart = std::visit(*this, expr);
                auto vt             = resultVariableType(expr);
                functionalPart += TypeAbbrev(vt);

                std::string comment = getComment(expr);
                if(comment.length() > 0)
                {
                    return concatenate("{", comment, ": ", functionalPart, "}");
                }

                return functionalPart;
            }

            std::string call(ExpressionPtr expr) const
            {
                if(!expr)
                    return "nullptr";

                return call(*expr);
            }
        };

        std::string toString(Expression const& expr)
        {
            auto visitor = ExpressionToStringVisitor();
            return visitor.call(expr);
        }

        std::string toString(ExpressionPtr const& expr)
        {
            auto visitor = ExpressionToStringVisitor();
            return visitor.call(expr);
        }
    }
}

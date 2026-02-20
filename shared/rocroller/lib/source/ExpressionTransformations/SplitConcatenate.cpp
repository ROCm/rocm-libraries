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

#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>

namespace rocRoller
{
    namespace Expression
    {
        struct SplitConcatenateVisitor
        {
            std::vector<ExpressionPtr> operator()(CommandArgumentValue const& value) const
            {
                return std::visit(
                    [](auto const& val) -> std::vector<ExpressionPtr> {
                        using T = std::decay_t<decltype(val)>;
                        if constexpr(std::is_same_v<T, uint64_t>)
                        {
                            // Split uint64_t into two Raw32 values (low 32 bits, then high 32 bits)
                            uint32_t low  = static_cast<uint32_t>(val);
                            uint32_t high = static_cast<uint32_t>((val >> 32));
                            return {literal(Raw32(low)), literal(Raw32(high))};
                        }
                        else
                        {
                            return {literal(val)};
                        }
                    },
                    value);
            }

            template <typename Expr>
            std::vector<ExpressionPtr> operator()(Expr const& expr) const
            {
                return {std::make_shared<Expression>(expr)};
            }

            std::vector<ExpressionPtr> call(ExpressionPtr expr) const
            {
                if(!expr)
                    return {expr};

                return std::visit(*this, *expr);
            }
        };

        /**
         * Splits uint64_t literal operands in a Concatenate expression into two Raw32 operands.
         */
        Concatenate splitConcatenate(Concatenate const& expr)
        {
            Concatenate                cpy = expr;
            std::vector<ExpressionPtr> operands;
            auto const                 visitor = SplitConcatenateVisitor();

            for(auto const& operand : expr.operands)
            {
                auto split = visitor.call(operand);
                operands.insert(operands.end(), split.begin(), split.end());
            }

            cpy.operands = std::move(operands);
            return cpy;
        }
    }
}

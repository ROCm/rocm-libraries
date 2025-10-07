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

namespace rocRoller
{
    namespace Expression
    {
        std::vector<ExpressionPtr> splitBitfield(BitfieldCombine const& expr, const size_t dstSize)
        {
            constexpr uint32_t         DWORD = 32;
            std::vector<ExpressionPtr> fields;
            uint32_t                   dstStartBit = expr.dstOffset;
            uint32_t                   dstEndBit   = expr.dstOffset + expr.width - 1;
            uint32_t                   numDwords   = (dstSize + DWORD - 1) / DWORD;

            for(int i = 0; i < numDwords; ++i)
            {
                int dwordStartBit = i * DWORD;
                int dwordEndBit   = dwordStartBit + DWORD - 1;

                // No overlap with this dword
                if(dstStartBit > dwordEndBit || dstEndBit < dwordStartBit)
                {
                    std::cout << "No overlap with dword " << i
                              << ": dwordStartBit=" << dwordStartBit
                              << ", dwordEndBit=" << dwordEndBit << std::endl;

                    auto dstDWord = std::make_shared<Expression>(
                        BitFieldExtract{expr.rhs, "", DataType::UInt32, dwordStartBit, DWORD});

                    // If we can evaluate at translation time, do it now to simplify the expression
                    auto srcEval = tryEvaluate(dstDWord);
                    if(srcEval.has_value())
                    {
                        std::cout << "Extracted field evaluated to " << srcEval.value()
                                  << std::endl;
                        fields.push_back(std::make_shared<Expression>(srcEval.value()));
                    }
                    else
                    {
                        std::cout << "Extracted field: " << dstDWord << std::endl;
                        fields.push_back(dstDWord);
                    }
                    continue;
                }

                std::cout << "Bitfield overlaps dword " << i << ": dwordStartBit=" << dwordStartBit
                          << ", dwordEndBit=" << dwordEndBit << std::endl;
            }

            return fields;
        }

        struct SplitBitfieldCombineExpressionVisitor
        {
            template <CUnary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;

                cpy.arg = call(expr.arg);
                return std::make_shared<Expression>(cpy);
            }

            template <CBinary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;

                cpy.lhs = call(expr.lhs);
                cpy.rhs = call(expr.rhs);
                return std::make_shared<Expression>(cpy);
            }

            template <CTernary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;

                cpy.lhs  = call(expr.lhs);
                cpy.r1hs = call(expr.r1hs);
                cpy.r2hs = call(expr.r2hs);

                return std::make_shared<Expression>(cpy);
            }

            template <CNary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                auto cpy = expr;
                std::ranges::for_each(cpy.operands, [this](auto& op) { op = call(op); });
                return std::make_shared<Expression>(std::move(cpy));
            }

            ExpressionPtr operator()(ScaledMatrixMultiply const& expr) const
            {
                auto cpy = expr;

                cpy.matA   = call(expr.matA);
                cpy.matB   = call(expr.matB);
                cpy.matC   = call(expr.matC);
                cpy.scaleA = call(expr.scaleA);
                cpy.scaleB = call(expr.scaleB);

                return std::make_shared<Expression>(cpy);
            }

            ExpressionPtr operator()(BitfieldCombine const& expr) const
            {
                // Print debug info
                std::cout << "Visiting BitfieldCombine: " << expr << std::endl;

                constexpr uint32_t DWORD = 32;
                auto               cpy   = expr;

                cpy.lhs = call(expr.lhs);
                cpy.rhs = call(expr.rhs);

                auto dstSize = resultVariableType(expr.rhs).getElementSize() * 8;
                AssertFatal(expr.dstOffset + expr.width <= dstSize,
                            "BitfieldCombine out of bounds: dstOffset={} + width={} > dstSize={}",
                            expr.dstOffset,
                            expr.width,
                            dstSize);

                auto srcSize = resultVariableType(expr.lhs).getElementSize() * 8;
                AssertFatal(expr.srcOffset + expr.width <= srcSize,
                            "BitfieldCombine out of bounds: srcOffset={} + width={} > srcSize={}",
                            expr.srcOffset,
                            expr.width,
                            srcSize);

                // No need to split if destination size is less than or equal to 32 bits
                if(dstSize <= DWORD)
                    return std::make_shared<Expression>(cpy);

                std::vector<ExpressionPtr> fields = splitBitfield(expr, dstSize);
                auto concatenateExpr = std::make_shared<Expression>(Concatenate{fields});
                return concatenateExpr;
            }

            template <CValue Value>
            ExpressionPtr operator()(Value const& expr) const
            {
                return std::make_shared<Expression>(expr);
            }

            ExpressionPtr call(ExpressionPtr expr) const
            {
                if(!expr)
                    return expr;

                return std::visit(*this, *expr);
            }
        };

        /**
         * TODO: add description
         */
        ExpressionPtr splitBitfieldCombine(ExpressionPtr expr)
        {
            auto visitor = SplitBitfieldCombineExpressionVisitor();
            return visitor.call(expr);
        }

    }
}

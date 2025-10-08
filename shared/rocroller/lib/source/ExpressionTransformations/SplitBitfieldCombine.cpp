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
            uint32_t                   combineStartBit = expr.dstOffset;
            uint32_t                   combineEndBit   = expr.dstOffset + expr.width - 1;
            uint32_t                   numDwords       = (dstSize + DWORD - 1) / DWORD;

            for(int i = 0; i < numDwords; ++i)
            {
                uint32_t dwordStartBit = i * DWORD;
                uint32_t dwordEndBit   = dwordStartBit + DWORD - 1;

                // Get new destination dword
                ExpressionPtr dstDWord     = bfe(DataType::UInt32, expr.rhs, dwordStartBit, DWORD);
                auto          dstDWordEval = tryEvaluate(dstDWord);
                if(dstDWordEval.has_value())
                    dstDWord = literal(dstDWordEval.value());

                // No overlap with this dword
                if(combineStartBit > dwordEndBit || combineEndBit < dwordStartBit)
                {
                    fields.push_back(dstDWord);
                }
                else
                {
                    uint32_t      overlapStart = std::max(combineStartBit, dwordStartBit);
                    uint32_t      overlapEnd   = std::min(combineEndBit, dwordEndBit);
                    uint32_t      overlapWidth = overlapEnd - overlapStart + 1;
                    ExpressionPtr subBitfieldCombine
                        = bfc(expr.lhs,
                              dstDWord,
                              expr.srcOffset + (overlapStart - combineStartBit),
                              overlapStart - dwordStartBit,
                              overlapWidth);
                    // auto subBitfieldCombineEval = tryEvaluate(subBitfieldCombine);
                    // if(subBitfieldCombineEval.has_value())
                    //     subBitfieldCombine
                    //         = literal(subBitfieldCombineEval.value());

                    fields.push_back(subBitfieldCombine);
                }
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

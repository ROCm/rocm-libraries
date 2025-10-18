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

        struct DeepBitfieldExtractVisitor
        {

            DeepBitfieldExtractVisitor(uint32_t offset, uint32_t width)
                : m_offset(offset)
                , m_width(width)
            {
            }

            ExpressionPtr operator()(BitfieldCombine const& expr) const
            {
                uint32_t combineStartBit = expr.dstOffset;
                uint32_t combineEndBit   = expr.dstOffset + expr.width - 1;
                uint32_t endBit          = m_offset + m_width - 1;
                // No overlap with this dword
                if(combineStartBit > endBit || combineEndBit < m_offset)
                {
                    return call(expr.rhs);
                }
                return std::make_shared<Expression>(expr);
            }

            ExpressionPtr operator()(Concatenate const& expr) const
            {
                uint32_t operandStartBit = 0;
                uint32_t operandEndBit   = 0;
                uint32_t endBit          = m_offset + m_width - 1;
                for(int i = 0; i < expr.operands.size(); ++i)
                {
                    operandStartBit = operandEndBit;
                    operandEndBit += resultVariableType(expr.operands[i]).getElementSize() * 8;
                    // bitField is fully contained within this operand
                    if(operandStartBit <= m_offset && endBit <= operandEndBit)
                    {
                        this->m_offset -= operandStartBit;
                        return call(expr.operands[i]);
                    }
                }

                return std::make_shared<Expression>(expr);
            }

            template <typename Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                return std::make_shared<Expression>(expr);
            }

            ExpressionPtr call(ExpressionPtr expr) const
            {
                if(!expr)
                    return expr;

                return std::visit(*this, *expr);
            }

            uint32_t get_offset() const
            {
                return m_offset;
            }

        private:
            mutable uint32_t m_offset;
            uint32_t         m_width;
        };

        /**
         * Returns a BitFieldExtract expression that extracts the specified bitfield from the given expression.
         * Looks through Concatenate expressions to find the corresponding operand to extract.
         * Looks through BitfieldCombine expressions to extract from its destination operand if the BitfieldCombine and BitFieldExtract do not overlap.
         */
        ExpressionPtr
            deepExtract(ExpressionPtr expr, DataType type, uint32_t offset, uint32_t width)
        {
            auto visitor   = DeepBitfieldExtractVisitor(offset, width);
            auto extracted = visitor.call(expr);
            offset         = visitor.get_offset();

            if(offset == 0 && width == resultVariableType(extracted).getElementSize() * 8)
            {
                return extracted;
            }

            ExpressionPtr extract = bfe(type, extracted, width, offset);
            return extract;
        }

        std::vector<ExpressionPtr>
            splitBitfield(BitfieldCombine const& expr, const size_t dstSize, const size_t dwordSize)
        {
            std::vector<ExpressionPtr> fields;
            uint32_t                   combineStartBit = expr.dstOffset;
            uint32_t                   combineEndBit   = expr.dstOffset + expr.width - 1;
            uint32_t                   numDwords       = (dstSize + dwordSize - 1) / dwordSize;

            for(int i = 0; i < numDwords; ++i)
            {
                uint32_t dwordStartBit = i * dwordSize;
                uint32_t dwordEndBit   = dwordStartBit + dwordSize - 1;

                // Get new destination dword
                ExpressionPtr dstDWord
                    = deepExtract(expr.rhs, DataType::UInt32, dwordStartBit, dwordSize);

                // No overlap with this dword
                if(combineStartBit > dwordEndBit || combineEndBit < dwordStartBit)
                {
                    fields.push_back(dstDWord);
                }
                else
                {
                    uint32_t overlapStart = std::max(combineStartBit, dwordStartBit);
                    uint32_t overlapEnd   = std::min(combineEndBit, dwordEndBit);
                    uint32_t overlapWidth = overlapEnd - overlapStart + 1;
                    uint32_t srcOffset    = expr.srcOffset + (overlapStart - combineStartBit);
                    uint32_t dstOffset    = overlapStart - dwordStartBit;

                    ExpressionPtr srcDWord = expr.lhs;
                    if(resultVariableType(expr.lhs).getElementSize() * 8 > dwordSize)
                    {
                        srcDWord  = bfe(DataType::UInt32, expr.lhs, srcOffset, overlapWidth);
                        srcOffset = 0;
                    }

                    ExpressionPtr subBitfieldCombine
                        = bfc(srcDWord, dstDWord, srcOffset, dstOffset, overlapWidth);

                    fields.push_back(subBitfieldCombine);
                }
            }

            return fields;
        }

        struct SplitBitfieldCombineExpressionVisitor
        {
            const uint32_t dwordSize = 32;

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
                auto cpy = expr;

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
                if(dstSize <= dwordSize && srcSize <= dwordSize)
                    return std::make_shared<Expression>(cpy);

                std::vector<ExpressionPtr> fields = splitBitfield(cpy, dstSize, dwordSize);
                auto                       concatenateExpr
                    = std::make_shared<Expression>(Concatenate{{fields}, resultVariableType(expr)});

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
         * Splits BitfieldCombine expressions that target more than 32 bits into a Concatenate of 32 bit sub-expressions.
         */
        ExpressionPtr splitBitfieldCombine(ExpressionPtr expr)
        {
            auto visitor = SplitBitfieldCombineExpressionVisitor();
            return simplify(visitor.call(expr));
        }

    }
}

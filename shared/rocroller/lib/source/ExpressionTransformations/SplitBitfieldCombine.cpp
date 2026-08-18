// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>

namespace rocRoller
{
    namespace Expression
    {
        /**
         * Splits a BitFieldExtract that crosses dword boundaries into two separate extracts
         * combined with OR. This is necessary because bfe code generation does not support
         * extracting from a source that has two registers (e.g. 64-bit source) into a destination
         * that has one register (e.g. 32-bit destination).
         */
        ExpressionPtr splitBitfieldExtract(ExpressionPtr src,
                                           uint32_t      srcOffset,
                                           uint32_t      width,
                                           uint32_t      dwordSize,
                                           DataType      dataType)
        {
            AssertFatal(width <= dwordSize,
                        "Expected width to be less than or equal to dword size");
            AssertFatal(dataType == DataType::Raw32, "Expected data type to be Raw32");

            uint32_t srcStartDword = srcOffset / dwordSize;
            uint32_t srcEndDword   = (srcOffset + width - 1) / dwordSize;

            // If extraction is within a single dword, just return simple bfe
            // Even if src is 64 bits, CSE simplifies this to extracting from
            // a single register
            if(srcStartDword == srcEndDword)
                return bfe(dataType, src, srcOffset, width);

            // Extraction crosses dword boundary - need to split
            uint32_t firstDwordEndBit = (srcStartDword + 1) * dwordSize - 1;
            uint32_t lowerWidth       = firstDwordEndBit - srcOffset + 1;
            uint32_t upperWidth       = width - lowerWidth;
            uint32_t upperOffset      = (srcStartDword + 1) * dwordSize;

            ExpressionPtr lower = bfe(dataType, src, srcOffset, lowerWidth);
            ExpressionPtr upper = bfe(dataType, src, upperOffset, upperWidth);

            // Combine
            ExpressionPtr shiftedUpper = upper << literal(lowerWidth);
            return lower | shiftedUpper;
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
                    = simplify(bfe(DataType::Raw32, expr.rhs, dwordStartBit, dwordSize));

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
                        srcDWord = splitBitfieldExtract(
                            expr.lhs, srcOffset, overlapWidth, dwordSize, DataType::Raw32);
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
                AssertFatal(
                    expr.dstOffset + expr.width <= dstSize,
                    fmt::format(
                        "BitfieldCombine out of bounds: dstOffset={} + width={} > dstSize={}",
                        expr.dstOffset,
                        expr.width,
                        dstSize));

                auto srcSize = resultVariableType(expr.lhs).getElementSize() * 8;
                AssertFatal(
                    expr.srcOffset + expr.width <= srcSize,
                    fmt::format(
                        "BitfieldCombine out of bounds: srcOffset={} + width={} > srcSize={}",
                        expr.srcOffset,
                        expr.width,
                        srcSize));

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

// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Expression.hpp>

namespace rocRoller
{
    namespace Expression
    {
        struct LowerBitfieldCombineExpressionVisitor
        {
            template <CUnary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                cpy.arg  = call(expr.arg);
                return std::make_shared<Expression>(cpy);
            }

            template <CBinary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                cpy.lhs  = call(expr.lhs);
                cpy.rhs  = call(expr.rhs);
                return std::make_shared<Expression>(cpy);
            }

            ExpressionPtr operator()(ScaledMatrixMultiply const& expr) const
            {
                ScaledMatrixMultiply cpy = expr;
                cpy.matA                 = call(expr.matA);
                cpy.matB                 = call(expr.matB);
                cpy.matC                 = call(expr.matC);
                cpy.scaleA               = call(expr.scaleA);
                cpy.scaleB               = call(expr.scaleB);
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

            ExpressionPtr operator()(BitfieldCombine const& expr) const
            {
                auto lhs = expr.lhs;
                lhs      = call(expr.lhs);
                if(lhs)
                {
                    AssertFatal(resultVariableType(lhs).getElementSize() <= 4u,
                                "Currently BitfieldCombine only supports: src size <= 1 dword");
                    AssertFatal(resultVariableType(lhs).getElementSize() * 8u
                                    >= expr.srcOffset + expr.width,
                                "Bitfield exceeds the number of bits of source, source size "
                                "(bytes), offset, width = ",
                                ShowValue(resultVariableType(lhs).getElementSize()),
                                ShowValue(expr.srcOffset),
                                ShowValue(expr.width));
                }

                auto rhs = expr.rhs;
                rhs      = call(expr.rhs);
                if(rhs)
                {
                    AssertFatal(resultVariableType(rhs).getElementSize() == 4u,
                                "Currently BitfieldCombine only supports: dst size = 1 dword");
                    AssertFatal(resultVariableType(rhs).getElementSize() * 8u
                                    >= expr.dstOffset + expr.width,
                                "Bitfield exceeds the number of bits of destination, destination "
                                "size (bytes), offset, width = ",
                                ShowValue(resultVariableType(rhs).getElementSize()),
                                ShowValue(expr.dstOffset),
                                ShowValue(expr.width));
                }

                AssertFatal(expr.width > 0,
                            "BitfieldCombine width must be greater than 0, width 0 should have "
                            "been optimized away by simplify");
                AssertFatal(expr.width <= 32,
                            "BitfieldCombine width must be less than or equal to 32");

                // Calculate width mask, handling width=32 case to avoid UB
                uint32_t widthMask;
                if(expr.width == 32)
                {
                    widthMask = std::numeric_limits<uint32_t>::max();
                }
                else
                {
                    widthMask = (1u << expr.width) - 1u;
                }

                auto const srcIsZero = expr.srcIsZero && expr.srcIsZero.value();
                if(not srcIsZero)
                {
                    rocRoller::Raw32 srcMask(widthMask << expr.srcOffset);
                    lhs = (literal(srcMask) & lhs); // Extract bits
                }

                auto const dstIsZero = expr.dstIsZero && expr.dstIsZero.value();
                if(not dstIsZero)
                {
                    rocRoller::Raw32 dstMask(~(widthMask << expr.dstOffset));
                    rhs = (literal(dstMask) & rhs); // Clear bits
                }

                if(expr.dstOffset > expr.srcOffset)
                    lhs = lhs << literal(expr.dstOffset - expr.srcOffset);
                else if(expr.dstOffset < expr.srcOffset)
                    lhs = logicalShiftR(lhs, literal(expr.srcOffset - expr.dstOffset));

                ExpressionPtr ret = lhs | rhs;
                setComment(ret, expr.comment);

                // Keep lowered expression type consistent with the original type
                // Otherwise, a Raw32 expr may become a UInt32 when lowered
                auto exprType = resultVariableType(expr);
                auto retType  = resultVariableType(ret);
                if(retType != exprType)
                {
                    AssertFatal(exprType.getElementSize() == retType.getElementSize(),
                                "Expression type size mismatch after lowering BitfieldCombine");
                    ret = reinterpret(exprType.dataType, ret);
                }

                return ret;
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
         * Replace a BitfieldCombine expression with:
         *
         *   srcMask =   ((1 << width) - 1) << srcOffset
         *   dstMask = ~(((1 << width) - 1) << dstOffset)
         *   dst = shift((srcMask & src), abs(srcOffset-dstOffset)) | (dstMask & dst)
         *
         *   Note: src=lhs, dst=rhs
         */
        ExpressionPtr lowerBitfieldCombine(ExpressionPtr expr)
        {
            auto visitor = LowerBitfieldCombineExpressionVisitor();
            return visitor.call(expr);
        }

    }
}

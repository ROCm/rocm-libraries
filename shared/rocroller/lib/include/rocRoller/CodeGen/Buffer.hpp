// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/Expression.hpp>

namespace rocRoller
{
    namespace BufferDescriptor
    {
        Expression::ExpressionPtr SetDefaults(Expression::ExpressionPtr bufferExpr, ContextPtr ctx);
        Expression::ExpressionPtr GetDefaultOptions(ContextPtr ctx);
        Expression::ExpressionPtr SetBasePointer(Expression::ExpressionPtr bufferExpr,
                                                 Expression::ExpressionPtr ptrExpr,
                                                 ContextPtr                ctx);
        Expression::ExpressionPtr GetBasePointer(Expression::ExpressionPtr bufferExpr,
                                                 ContextPtr                ctx);
        Expression::ExpressionPtr IncrementBasePointer(Expression::ExpressionPtr bufferExpr,
                                                       Expression::ExpressionPtr offsetExpr,
                                                       ContextPtr                ctx);
        Expression::ExpressionPtr SetSize(Expression::ExpressionPtr bufferExpr,
                                          Expression::ExpressionPtr sizeExpr,
                                          ContextPtr                ctx);
        Expression::ExpressionPtr GetSize(Expression::ExpressionPtr bufferExpr);
        Expression::ExpressionPtr SetOptions(Expression::ExpressionPtr bufferExpr,
                                             Expression::ExpressionPtr optsExpr);
        Expression::ExpressionPtr GetOptions(Expression::ExpressionPtr bufferExpr);
    }
}

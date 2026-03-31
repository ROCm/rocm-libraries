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
                                                 Expression::ExpressionPtr ptrExpr);
        Expression::ExpressionPtr GetBasePointer(Expression::ExpressionPtr bufferExpr);
        Expression::ExpressionPtr IncrementBasePointer(Expression::ExpressionPtr bufferExpr,
                                                       Expression::ExpressionPtr offsetExpr);
        Expression::ExpressionPtr SetSize(Expression::ExpressionPtr bufferExpr,
                                          Expression::ExpressionPtr sizeExpr);
        Expression::ExpressionPtr GetSize(Expression::ExpressionPtr bufferExpr);
        Expression::ExpressionPtr SetOptions(Expression::ExpressionPtr bufferExpr,
                                             Expression::ExpressionPtr optsExpr);
        Expression::ExpressionPtr GetOptions(Expression::ExpressionPtr bufferExpr);
    }
}

// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Context_fwd.hpp>

namespace rocRoller
{
    namespace TDMDescriptor
    {
        enum class DataSize : int
        {
            OneByte = 0,
            TwoBytes,
            FourBytes,
            EightBytes,
            Count
        };

        Expression::ExpressionPtr DataSizeToExpression(DataSize option);

        Expression::ExpressionPtr SetDataSize(Expression::ExpressionPtr tdmExpr, DataSize dataSize);

        Expression::ExpressionPtr SetLDSAddress(Expression::ExpressionPtr tdmExpr,
                                                Expression::ExpressionPtr ldsAddrExpr);

        Expression::ExpressionPtr SetGlobalAddress(Expression::ExpressionPtr tdmExpr,
                                                   Expression::ExpressionPtr globalAddrExpr);

        Expression::ExpressionPtr SetTileDims(Expression::ExpressionPtr tdmExpr,
                                              Expression::ExpressionPtr tileDim0Expr,
                                              Expression::ExpressionPtr tileDim1Expr,
                                              Expression::ExpressionPtr tileDim2Expr = nullptr);

        Expression::ExpressionPtr SetTensorDims(Expression::ExpressionPtr tdmExpr,
                                                Expression::ExpressionPtr tensorDim0Expr,
                                                Expression::ExpressionPtr tensorDim1Expr);

        Expression::ExpressionPtr SetTensorStrides(Expression::ExpressionPtr tdmExpr,
                                                   Expression::ExpressionPtr tensorDim0StrideExpr,
                                                   Expression::ExpressionPtr tensorDim1StrideExpr);

        Expression::ExpressionPtr SetDefaults(Expression::ExpressionPtr tdmExpr, ContextPtr ctx);
    }
}

// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/CodeGen/Annotate.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/TensorDataMover.hpp>
#include <rocRoller/Utilities/Generator.hpp>

namespace rocRoller
{
    namespace TDMDescriptor
    {
        constexpr size_t MAX_PAD_AMOUNT_DWORDS   = 128;
        constexpr size_t MAX_PAD_INTERVAL_DWORDS = 256;

        namespace TDMInfo
        {
            struct BitfieldInfo
            {
                uint32_t bitoffset;
                uint32_t bitwidth;
            };
            // TDM Group 0
            constexpr BitfieldInfo Reserved0{0, 30};
            constexpr BitfieldInfo GatherIndexSize{30, 1};
            constexpr BitfieldInfo GatherMode{31, 1};
            constexpr BitfieldInfo LDSAddress{32, 32};
            constexpr BitfieldInfo GlobalAddress{64, 57};
            constexpr BitfieldInfo Reserved1{121, 5};
            constexpr BitfieldInfo Type{126, 2};

            // TDM Group 1
            constexpr BitfieldInfo WorkgroupMask{128, 16};
            constexpr BitfieldInfo DataSize{144, 2};
            constexpr BitfieldInfo SendAtomicBarrierOption{146, 1};
            constexpr BitfieldInfo TensorIterateMode{147, 1};
            constexpr BitfieldInfo PaddingMode{148, 1};
            constexpr BitfieldInfo EarlyTimeoutOption{149, 1};
            constexpr BitfieldInfo PadInterval{150, 3};
            constexpr BitfieldInfo PadAmount{153, 7};
            constexpr BitfieldInfo AtomicBarrierAddress{160, 16};
            constexpr BitfieldInfo TensorDim0{176, 32};
            constexpr BitfieldInfo TensorDim1{208, 32};
            constexpr BitfieldInfo TileDim0{240, 16};
            constexpr BitfieldInfo TileDim1{256, 16};
            constexpr BitfieldInfo TileDim2{272, 16};
            constexpr BitfieldInfo TensorDim0Stride{288, 48};
            constexpr BitfieldInfo TensorDim1Stride{336, 48};
        }

        Expression::ExpressionPtr DataSizeToExpression(DataSize option)
        {
            switch(option)
            {
            case DataSize::OneByte:
                return Expression::literal(0);
            case DataSize::TwoBytes:
                return Expression::literal(1);
            case DataSize::FourBytes:
                return Expression::literal(2);
            case DataSize::EightBytes:
                return Expression::literal(3);
            default:
                Throw<FatalError>(fmt::format("Invalid DataSize {}", static_cast<int>(option)));
            }
            return nullptr;
        }

        Expression::ExpressionPtr SetDataSize(Expression::ExpressionPtr tdmExpr, DataSize dataSize)
        {
            AssertFatal(tdmExpr, "TDM expression cannot be null.");
            auto exprVarType = resultVariableType(tdmExpr);
            AssertFatal(exprVarType.pointerType == PointerType::TDM,
                        "TDM expression must be of TDM pointer type. ",
                        ShowValue(exprVarType));

            auto dataSizeExpr = DataSizeToExpression(dataSize);

            const auto [bitoffset, bitwidth] = TDMInfo::DataSize;
            tdmExpr                          = bfc(dataSizeExpr, tdmExpr, 0, bitoffset, bitwidth);
            return tdmExpr;
        }

        Expression::ExpressionPtr SetLDSAddress(Expression::ExpressionPtr tdmExpr,
                                                Expression::ExpressionPtr ldsAddrExpr)
        {
            AssertFatal(tdmExpr && ldsAddrExpr,
                        "TDM expression and LDS address expression cannot be null.");
            auto exprVarType = resultVariableType(tdmExpr);
            AssertFatal(exprVarType.pointerType == PointerType::TDM,
                        "TDM expression must be of TDM pointer type. ",
                        ShowValue(exprVarType));

            // Ensure type is valid
            auto ldsAddrExprType = resultVariableType(ldsAddrExpr);
            AssertFatal(DataTypeInfo::Get(ldsAddrExprType).elementBits == 32,
                        "LDS address must be of type UInt32, got ",
                        ldsAddrExprType);

            const auto [bitoffset, bitwidth] = TDMInfo::LDSAddress;
            tdmExpr                          = bfc(ldsAddrExpr, tdmExpr, 0, bitoffset, bitwidth);
            return tdmExpr;
        }

        Expression::ExpressionPtr SetGlobalAddress(Expression::ExpressionPtr tdmExpr,
                                                   Expression::ExpressionPtr globalAddrExpr)
        {
            AssertFatal(tdmExpr && globalAddrExpr,
                        "TDM expression and Global address expression cannot be null.");
            auto exprVarType = resultVariableType(tdmExpr);
            AssertFatal(exprVarType.pointerType == PointerType::TDM,
                        "TDM expression must be of TDM pointer type. ",
                        ShowValue(exprVarType));

            // Ensure type is valid
            auto globalAddrExprType = resultVariableType(globalAddrExpr);
            AssertFatal(DataTypeInfo::Get(globalAddrExprType).elementBits == 64,
                        "Global address must be of type UInt64, got ",
                        globalAddrExprType);

            const auto [bitoffset, bitwidth] = TDMInfo::GlobalAddress;
            tdmExpr                          = bfc(globalAddrExpr, tdmExpr, 0, bitoffset, bitwidth);
            return tdmExpr;
        }

        Expression::ExpressionPtr SetTileDims(Expression::ExpressionPtr tdmExpr,
                                              Expression::ExpressionPtr tileDim0Expr,
                                              Expression::ExpressionPtr tileDim1Expr,
                                              Expression::ExpressionPtr tileDim2Expr)
        {
            AssertFatal(tdmExpr && tileDim0Expr && tileDim1Expr,
                        "TDM expression and tile dimension expressions cannot be null.");
            auto exprVarType = resultVariableType(tdmExpr);
            AssertFatal(exprVarType.pointerType == PointerType::TDM,
                        "TDM expression must be of TDM pointer type. ",
                        ShowValue(exprVarType));

            const auto zero = Expression::literal(0);

            {
                const auto [bitoffset, bitwidth] = TDMInfo::TileDim0;
                tdmExpr = bfc(tileDim0Expr, tdmExpr, 0, bitoffset, bitwidth);
            }
            {
                const auto [bitoffset, bitwidth] = TDMInfo::TileDim1;
                tdmExpr = bfc(tileDim1Expr, tdmExpr, 0, bitoffset, bitwidth);
            }

            if(not tileDim2Expr)
            {
                tileDim2Expr = zero;
            }

            {
                const auto [bitoffset, bitwidth] = TDMInfo::TileDim2;
                tdmExpr = bfc(tileDim2Expr, tdmExpr, 0, bitoffset, bitwidth);
            }
            return tdmExpr;
        }

        Expression::ExpressionPtr SetTensorDims(Expression::ExpressionPtr tdmExpr,
                                                Expression::ExpressionPtr tensorDim0Expr,
                                                Expression::ExpressionPtr tensorDim1Expr)
        {
            AssertFatal(tdmExpr && tensorDim0Expr && tensorDim1Expr,
                        "TDM expression and tensor dimension expressions cannot be null.");
            auto exprVarType = resultVariableType(tdmExpr);
            AssertFatal(exprVarType.pointerType == PointerType::TDM,
                        "TDM expression must be of TDM pointer type. ",
                        ShowValue(exprVarType));

            // Ensure type is valid
            auto tensorDim0ExprType = resultVariableType(tensorDim0Expr);
            AssertFatal(DataTypeInfo::Get(tensorDim0ExprType).elementBits == 32,
                        "Tensor Dim0 must be of type UInt32, got ",
                        tensorDim0ExprType);
            auto tensorDim1ExprType = resultVariableType(tensorDim1Expr);
            AssertFatal(DataTypeInfo::Get(tensorDim1ExprType).elementBits == 32,
                        "Tensor Dim1 must be of type UInt32, got ",
                        tensorDim1ExprType);

            {
                const auto [bitoffset, bitwidth] = TDMInfo::TensorDim0;
                tdmExpr = bfc(tensorDim0Expr, tdmExpr, 0, bitoffset, bitwidth);
            }

            {
                const auto [bitoffset, bitwidth] = TDMInfo::TensorDim1;
                tdmExpr = bfc(tensorDim1Expr, tdmExpr, 0, bitoffset, bitwidth);
            }
            return tdmExpr;
        }

        Expression::ExpressionPtr SetTensorStrides(Expression::ExpressionPtr tdmExpr,
                                                   Expression::ExpressionPtr tensorDim0StrideExpr,
                                                   Expression::ExpressionPtr tensorDim1StrideExpr)
        {
            AssertFatal(tdmExpr && tensorDim0StrideExpr && tensorDim1StrideExpr,
                        "TDM expression and tensor dimension stride expressions cannot be null.");
            auto exprVarType = resultVariableType(tdmExpr);
            AssertFatal(exprVarType.pointerType == PointerType::TDM,
                        "TDM expression must be of TDM pointer type. ",
                        ShowValue(exprVarType));

            // Ensure type is valid
            auto tensorDim0StrideExprType = resultVariableType(tensorDim0StrideExpr);
            AssertFatal(DataTypeInfo::Get(tensorDim0StrideExprType).elementBits == 64,
                        "Tensor Dim0 Stride must be of type UInt64, got ",
                        tensorDim0StrideExprType);
            auto tensorDim1StrideExprType = resultVariableType(tensorDim1StrideExpr);
            AssertFatal(DataTypeInfo::Get(tensorDim1StrideExprType).elementBits == 64,
                        "Tensor Dim1 Stride must be of type UInt64, got ",
                        tensorDim1StrideExprType);

            {
                const auto [bitoffset, bitwidth] = TDMInfo::TensorDim0Stride;
                tdmExpr = bfc(tensorDim0StrideExpr, tdmExpr, 0, bitoffset, bitwidth);
            }

            {
                const auto [bitoffset, bitwidth] = TDMInfo::TensorDim1Stride;
                tdmExpr = bfc(tensorDim1StrideExpr, tdmExpr, 0, bitoffset, bitwidth);
            }
            return tdmExpr;
        }

        Expression::ExpressionPtr SetPadInterval(Expression::ExpressionPtr tdmExpr,
                                                 uint32_t                  numDwords)
        {
            AssertFatal(tdmExpr, "TDM expression expression cannot be null.");
            auto exprVarType = resultVariableType(tdmExpr);
            AssertFatal(exprVarType.pointerType == PointerType::TDM,
                        "TDM expression must be of TDM pointer type. ",
                        ShowValue(exprVarType));

            AssertFatal((numDwords & (numDwords - 1)) == 0,
                        "Padding interval must a power of two.",
                        ShowValue(numDwords));
            AssertFatal(numDwords <= MAX_PAD_INTERVAL_DWORDS,
                        "Padding interval must be at most 256 dwords",
                        ShowValue(numDwords),
                        ShowValue(MAX_PAD_INTERVAL_DWORDS));

            const auto [bitoffset, bitwidth] = TDMInfo::PadInterval;
            const uint32_t padIntervalValue  = numDwords > 1 ? log(numDwords) - 1u : 0;
            tdmExpr = bfc(Expression::literal(padIntervalValue), tdmExpr, 0, bitoffset, bitwidth);
            return tdmExpr;
        }

        Expression::ExpressionPtr SetPadAmount(Expression::ExpressionPtr tdmExpr,
                                               uint32_t                  numDwords)
        {
            AssertFatal(tdmExpr, "TDM expression expression cannot be null.");
            auto exprVarType = resultVariableType(tdmExpr);
            AssertFatal(exprVarType.pointerType == PointerType::TDM,
                        "TDM expression must be of TDM pointer type. ",
                        ShowValue(exprVarType));

            AssertFatal(numDwords <= MAX_PAD_AMOUNT_DWORDS,
                        "Padding amount must be at most 128 dwords",
                        ShowValue(numDwords),
                        ShowValue(MAX_PAD_AMOUNT_DWORDS));

            const auto amount                = numDwords > 0 ? numDwords - 1 : 0;
            const auto [bitoffset, bitwidth] = TDMInfo::PadAmount;
            tdmExpr = bfc(Expression::literal(amount), tdmExpr, 0, bitoffset, bitwidth);
            return tdmExpr;
        }

        Expression::ExpressionPtr SetDefaults(Expression::ExpressionPtr tdmExpr, ContextPtr ctx)
        {
            AssertFatal(tdmExpr && ctx, "TDM expression and context cannot be null.");
            auto exprVarType = resultVariableType(tdmExpr);
            AssertFatal(exprVarType.pointerType == PointerType::TDM,
                        "TDM expression must be of TDM pointer type. ",
                        ShowValue(exprVarType));

            const auto zero       = Expression::literal(0, DataType::UInt32);
            const auto zero64bits = Expression::literal(0, DataType::UInt64);
            const auto one        = Expression::literal(1, DataType::UInt32);
            const auto two        = Expression::literal(2, DataType::UInt32);

            const auto reserved0 = TDMInfo::Reserved0;
            tdmExpr              = bfc(one, tdmExpr, 0, reserved0.bitoffset, reserved0.bitwidth);

            const auto gatherIndexSize = TDMInfo::GatherIndexSize;
            tdmExpr = bfc(zero, tdmExpr, 0, gatherIndexSize.bitoffset, gatherIndexSize.bitwidth);

            const auto gatherMode = TDMInfo::GatherMode;
            tdmExpr = bfc(zero, tdmExpr, 0, gatherMode.bitoffset, gatherMode.bitwidth);

            const auto ldsAddress = TDMInfo::LDSAddress;
            tdmExpr = bfc(zero, tdmExpr, 0, ldsAddress.bitoffset, ldsAddress.bitwidth);

            const auto globalAddress = TDMInfo::GlobalAddress;
            tdmExpr = bfc(zero64bits, tdmExpr, 0, globalAddress.bitoffset, globalAddress.bitwidth);

            const auto reserved1 = TDMInfo::Reserved1;
            tdmExpr              = bfc(zero, tdmExpr, 0, reserved1.bitoffset, reserved1.bitwidth);

            const auto type = TDMInfo::Type;
            tdmExpr         = bfc(two, tdmExpr, 0, type.bitoffset, type.bitwidth);

            const auto workgroupMask = TDMInfo::WorkgroupMask;
            tdmExpr = bfc(zero, tdmExpr, 0, workgroupMask.bitoffset, workgroupMask.bitwidth);

            tdmExpr = SetDataSize(tdmExpr, DataSize::OneByte);

            const auto sendAtomicBarrierOption = TDMInfo::SendAtomicBarrierOption;
            tdmExpr                            = bfc(zero,
                          tdmExpr,
                          0,
                          sendAtomicBarrierOption.bitoffset,
                          sendAtomicBarrierOption.bitwidth);

            const auto tensorIterateMode = TDMInfo::TensorIterateMode;
            tdmExpr
                = bfc(zero, tdmExpr, 0, tensorIterateMode.bitoffset, tensorIterateMode.bitwidth);

            const auto paddingMode = TDMInfo::PaddingMode;
            tdmExpr = bfc(zero, tdmExpr, 0, paddingMode.bitoffset, paddingMode.bitwidth);

            const auto earlyTimeoutOption = TDMInfo::EarlyTimeoutOption;
            tdmExpr
                = bfc(zero, tdmExpr, 0, earlyTimeoutOption.bitoffset, earlyTimeoutOption.bitwidth);

            tdmExpr = SetPadInterval(tdmExpr, 0);

            tdmExpr = SetPadAmount(tdmExpr, 0);

            const auto atomicBarrierAddress = TDMInfo::AtomicBarrierAddress;
            tdmExpr                         = bfc(
                zero, tdmExpr, 0, atomicBarrierAddress.bitoffset, atomicBarrierAddress.bitwidth);

            tdmExpr = SetTensorDims(tdmExpr, zero, zero);

            tdmExpr = SetTileDims(tdmExpr, zero, zero);

            tdmExpr = SetTensorStrides(tdmExpr, zero64bits, zero64bits);

            return tdmExpr;
        }
    }
}

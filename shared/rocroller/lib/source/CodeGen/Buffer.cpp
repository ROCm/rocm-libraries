// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Buffer.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Expression.hpp>

namespace rocRoller
{
    namespace BufferDescriptor
    {
        using Expression::ExpressionPtr;

        ExpressionPtr GetDefaultOptions(ContextPtr ctx)
        {
            AssertFatal(ctx, "Context cannot be null.");

            if(ctx->targetArchitecture().HasCapability(
                   GPUCapability::HasBufferOutOfBoundsCheckOption))
            {
                // Bits 29:28 are for Out-of-Bounds check.
                //   0 - index >= NumRecords || offset + payload > stride, used for structured buffers.
                //   1 - index >= NumRecords, used for raw buffers (RR default)
                //   2 - NumRecords == 0, empty buffers
                //
                // Bits 17:12 are for data format.
                //   5 - 8_UINT. Currently, everything is buffer-loaded in terms of bytes.
                // TODO: Add GFX12 buffer descriptor when other formats and/or features are needed.
                uint32_t outOfBoundsCheck = (1u << 28);
                uint32_t dataFormat       = (5u << 12);
                if(ctx->targetArchitecture().HasCapability(
                       GPUCapability::HasBufferFormatSpecInSOffsetField))
                {
                    // 0 - index >= NumRecords, used for raw buffers (RR default)
                    // 1 - index >= NumRecords || offset + payload > stride, used for structured buffers.
                    // 2 - NumRecords == 0, empty buffers
                    outOfBoundsCheck = 0;
                    dataFormat       = 0;
                }
                return Expression::literal(outOfBoundsCheck | dataFormat, DataType::UInt32);
            }
            // 0x00020000
            return Expression::literal((4u << 15), DataType::UInt32);
        }

        ExpressionPtr SetDefaults(ExpressionPtr bufferExpr, ContextPtr ctx)
        {
            AssertFatal(bufferExpr && ctx, "Buffer and context cannot be null.");
            auto exprVarType = resultVariableType(bufferExpr);
            AssertFatal(exprVarType.pointerType == PointerType::Buffer,
                        "Buffer expression must be of buffer pointer type. ",
                        ShowValue(exprVarType));

            bufferExpr = BufferDescriptor::SetSize(
                bufferExpr, Expression::literal(2147483548ull, DataType::UInt64), ctx);
            bufferExpr = BufferDescriptor::SetOptions(bufferExpr, GetDefaultOptions(ctx));
            return bufferExpr;
        }

        ExpressionPtr
            SetBasePointer(ExpressionPtr bufferExpr, ExpressionPtr addressExpr, ContextPtr ctx)
        {
            AssertFatal(ctx, "Context cannot be null.");
            AssertFatal(bufferExpr && addressExpr,
                        "Buffer and address expressions cannot be null.");
            auto exprVarType = resultVariableType(bufferExpr);
            AssertFatal(exprVarType.pointerType == PointerType::Buffer,
                        "Buffer expression must be of buffer pointer type. ",
                        ShowValue(exprVarType));

            // Ensure type is valid
            auto addressExprType = resultVariableType(addressExpr);
            AssertFatal(DataTypeInfo::Get(addressExprType).elementBits == 64,
                        "Base pointer must be of type UInt64, got ",
                        addressExprType);

            if(ctx->targetArchitecture().HasCapability(
                   GPUCapability::HasBufferFormatSpecInSOffsetField))
            {
                // s1[24:0] s0[31:0] 57-bit Base byte address.
                return bfc(addressExpr, bufferExpr, 0, 0, 57);
            }
            return bfc(addressExpr, bufferExpr, 0, 0, 64);
        }

        ExpressionPtr
            GetBasePointer(ExpressionPtr bufferExpr, ExpressionPtr valueExpr, ContextPtr ctx)
        {
            AssertFatal(bufferExpr, "Buffer expression cannot be null.");
            auto exprVarType = resultVariableType(bufferExpr);
            AssertFatal(exprVarType.pointerType == PointerType::Buffer,
                        "Buffer expression must be of buffer pointer type. ",
                        ShowValue(exprVarType));

            if(ctx->targetArchitecture().HasCapability(
                   GPUCapability::HasBufferFormatSpecInSOffsetField))
            {
                // s1[24:0] s0[31:0] 57-bit Base byte address.
                auto basePointer = bfe(DataType::UInt64, bufferExpr, 0, 57);
                return bfc(basePointer + valueExpr, bufferExpr, 0, 0, 57);
            }
            return bfe(DataType::UInt64, bufferExpr, 0, 64);
        }

        ExpressionPtr
            IncrementBasePointer(ExpressionPtr bufferExpr, ExpressionPtr valueExpr, ContextPtr ctx)
        {
            AssertFatal(ctx, "Context cannot be null.");
            AssertFatal(bufferExpr && valueExpr, "Buffer and value expressions cannot be null.");
            auto exprVarType = resultVariableType(bufferExpr);
            AssertFatal(exprVarType.pointerType == PointerType::Buffer,
                        "Buffer expression must be of buffer pointer type. ",
                        ShowValue(exprVarType));

            if(ctx->targetArchitecture().HasCapability(
                   GPUCapability::HasBufferFormatSpecInSOffsetField))
            {
                // s1[24:0] s0[31:0] 57-bit Base byte address.
                auto basePointer = bfe(DataType::UInt64, bufferExpr, 0, 57);
                return bfc(basePointer + valueExpr, bufferExpr, 0, 0, 57);
            }
            auto basePointer = bfe(DataType::UInt64, bufferExpr, 0, 64);
            return bfc(basePointer + valueExpr, bufferExpr, 0, 0, 64);
        }

        ExpressionPtr SetSize(ExpressionPtr bufferExpr, ExpressionPtr sizeExpr, ContextPtr ctx)
        {
            AssertFatal(bufferExpr && sizeExpr, "Buffer and size expressions cannot be null.");
            auto exprVarType = resultVariableType(bufferExpr);
            AssertFatal(exprVarType.pointerType == PointerType::Buffer,
                        "Buffer expression must be of buffer pointer type. ",
                        ShowValue(exprVarType));

            if(ctx->targetArchitecture().HasCapability(
                   GPUCapability::HasBufferFormatSpecInSOffsetField))
            {
                // Ensure type is valid
                auto sizeExprType = resultVariableType(sizeExpr);
                if(DataTypeInfo::Get(sizeExprType).elementBits < 64)
                    sizeExpr = Expression::convert(DataType::UInt64, sizeExpr);

                // s3[5:0] s2[31:0] s1[31:25] 45-bit numRecords
                return bfc(sizeExpr, bufferExpr, 0, 57, 45);
            }
            return bfc(sizeExpr, bufferExpr, 0, 64, 32);
        }

        ExpressionPtr GetSize(ExpressionPtr bufferExpr)
        {
            AssertFatal(bufferExpr, "Buffer expression cannot be null.");
            auto exprVarType = resultVariableType(bufferExpr);
            AssertFatal(exprVarType.pointerType == PointerType::Buffer,
                        "Buffer expression must be of buffer pointer type. ",
                        ShowValue(exprVarType));

            return bfe(DataType::UInt32, bufferExpr, 64, 32);
        }

        ExpressionPtr SetOptions(ExpressionPtr bufferExpr, ExpressionPtr optsExpr)
        {
            AssertFatal(bufferExpr && optsExpr, "Buffer and options expressions cannot be null.");
            auto exprVarType = resultVariableType(bufferExpr);
            AssertFatal(exprVarType.pointerType == PointerType::Buffer,
                        "Buffer expression must be of buffer pointer type. ",
                        ShowValue(exprVarType));

            return bfc(optsExpr, bufferExpr, 0, 96, 32);
        }

        ExpressionPtr GetOptions(ExpressionPtr bufferExpr)
        {
            AssertFatal(bufferExpr, "Buffer expression cannot be null.");
            auto exprVarType = resultVariableType(bufferExpr);
            AssertFatal(exprVarType.pointerType == PointerType::Buffer,
                        "Buffer expression must be of buffer pointer type. ",
                        ShowValue(exprVarType));

            return bfe(DataType::UInt32, bufferExpr, 96, 32);
        }
    }
}

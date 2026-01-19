/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2022-2025 AMD ROCm(TM) Software
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

#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Buffer.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Context.hpp>

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
                return Expression::literal((1u << 28) | (5u << 12), DataType::UInt32);
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
                bufferExpr, Expression::literal(2147483548u, DataType::UInt32));
            bufferExpr = BufferDescriptor::SetOptions(bufferExpr, GetDefaultOptions(ctx));
            return bufferExpr;
        }

        ExpressionPtr SetBasePointer(ExpressionPtr bufferExpr, ExpressionPtr addressExpr)
        {
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

            return bfc(addressExpr, bufferExpr, 0, 0, 64);
        }

        ExpressionPtr GetBasePointer(ExpressionPtr bufferExpr)
        {
            AssertFatal(bufferExpr, "Buffer expression cannot be null.");
            auto exprVarType = resultVariableType(bufferExpr);
            AssertFatal(exprVarType.pointerType == PointerType::Buffer,
                        "Buffer expression must be of buffer pointer type. ",
                        ShowValue(exprVarType));

            return bfe(DataType::UInt64, bufferExpr, 0, 64);
        }

        ExpressionPtr IncrementBasePointer(ExpressionPtr bufferExpr, ExpressionPtr valueExpr)
        {
            AssertFatal(bufferExpr && valueExpr, "Buffer and value expressions cannot be null.");
            auto exprVarType = resultVariableType(bufferExpr);
            AssertFatal(exprVarType.pointerType == PointerType::Buffer,
                        "Buffer expression must be of buffer pointer type. ",
                        ShowValue(exprVarType));

            auto basePointer = bfe(DataType::UInt64, bufferExpr, 0, 64);
            return bfc(basePointer + valueExpr, bufferExpr, 0, 0, 64);
        }

        ExpressionPtr SetSize(ExpressionPtr bufferExpr, ExpressionPtr sizeExpr)
        {
            AssertFatal(bufferExpr && sizeExpr, "Buffer and size expressions cannot be null.");
            auto exprVarType = resultVariableType(bufferExpr);
            AssertFatal(exprVarType.pointerType == PointerType::Buffer,
                        "Buffer expression must be of buffer pointer type. ",
                        ShowValue(exprVarType));

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

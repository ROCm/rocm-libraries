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
    /*
     * Creates buffer descriptor object from existing SGPRs
     */

    BufferDescriptor::BufferDescriptor(Register::ValuePtr srd, ContextPtr context)
    {
        m_bufferResourceDescriptor = srd;
        m_context                  = context;
    }

    /*
     * Creates buffer descriptor object from context, no existing SGPRs
     * Requires the use of the BufferDescriptor::setup()
     */
    BufferDescriptor::BufferDescriptor(ContextPtr context)
    {
        VariableType bufferPointer{DataType::None, PointerType::Buffer};
        m_bufferResourceDescriptor
            = std::make_shared<Register::Value>(context, Register::Type::Scalar, bufferPointer, 1);
        m_context = context;
    }

    Generator<Instruction> BufferDescriptor::setup()
    {
        co_yield m_context->copier()->copy(
            m_bufferResourceDescriptor->subset({2}), Register::Value::Literal(2147483548), "");
        co_yield setDefaultOpts();
    }

    uint32_t BufferDescriptor::getDefaultOptionsValue(ContextPtr ctx)
    {
        if(ctx->targetArchitecture().HasCapability(GPUCapability::HasBufferOutOfBoundsCheckOption))
        {
            // Bits 29:28 are for Out-of-Bounds check.
            //   0 - index >= NumRecords || offset + payload > stride, used for structured buffers.
            //   1 - index >= NumRecords, used for raw buffers (RR default)
            //   2 - NumRecords == 0, empty buffers
            //
            // Bits 17:12 are for data format.
            //   5 - 8_UINT. Currently, everything is buffer-loaded in terms of bytes.
            // TODO: Add GFX12 buffer descriptor when other formats and/or features are needed.
            return (1u << 28) | (5u << 12);
        }
        // 0x00020000
        return (4u << 15);
    }

    Generator<Instruction> BufferDescriptor::setDefaultOpts()
    {
        uint32_t opts = getDefaultOptionsValue(m_context);
        co_yield m_context->copier()->copy(m_bufferResourceDescriptor->subset({3}),
                                           Register::Value::Literal(opts),
                                           "default options");
    }

    Generator<Instruction> BufferDescriptor::incrementBasePointer(Register::ValuePtr value)
    {
        co_yield generateOp<Expression::Add>(m_bufferResourceDescriptor->subset({0, 1}),
                                             m_bufferResourceDescriptor->subset({0, 1}),
                                             value);
    }

    Generator<Instruction> BufferDescriptor::setBasePointer(Register::ValuePtr value)
    {
        co_yield m_context->copier()->copy(m_bufferResourceDescriptor->subset({0, 1}), value, "");
    }

    Generator<Instruction> BufferDescriptor::setSize(Register::ValuePtr value)
    {
        co_yield m_context->copier()->copy(m_bufferResourceDescriptor->subset({2}), value, "");
    }

    Generator<Instruction> BufferDescriptor::setOptions(Register::ValuePtr value)
    {
        co_yield m_context->copier()->copy(m_bufferResourceDescriptor->subset({3}), value, "");
    }

    Register::ValuePtr BufferDescriptor::allRegisters() const
    {
        return m_bufferResourceDescriptor;
    }

    Register::ValuePtr BufferDescriptor::descriptorOptions() const
    {
        return m_bufferResourceDescriptor->subset({3});
    }
}

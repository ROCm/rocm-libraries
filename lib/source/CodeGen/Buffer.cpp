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
    std::string toString(GFX9BufferDescriptorOptions::DataFormatValue val)
    {
        switch(val)
        {
        case GFX9BufferDescriptorOptions::DFInvalid:
            return "DFInvalid";
        case GFX9BufferDescriptorOptions::DF8:
            return "DF8";
        case GFX9BufferDescriptorOptions::DF16:
            return "DF16";
        case GFX9BufferDescriptorOptions::DF8_8:
            return "DF8_8";
        case GFX9BufferDescriptorOptions::DF32:
            return "DF32";
        case GFX9BufferDescriptorOptions::DF16_16:
            return "DF16_16";
        case GFX9BufferDescriptorOptions::DF10_11_11:
            return "DF10_11_11";
        case GFX9BufferDescriptorOptions::DF11_11_10:
            return "DF11_11_10";
        case GFX9BufferDescriptorOptions::DF10_10_10_2:
            return "DF10_10_10_2";
        case GFX9BufferDescriptorOptions::DF2_10_10_10:
            return "DF2_10_10_10";
        case GFX9BufferDescriptorOptions::DF8_8_8_8:
            return "DF8_8_8_8";
        case GFX9BufferDescriptorOptions::DF32_32:
            return "DF32_32";
        case GFX9BufferDescriptorOptions::DF16_16_16_16:
            return "DF16_16_16_16";
        case GFX9BufferDescriptorOptions::DF32_32_32:
            return "DF32_32_32";
        case GFX9BufferDescriptorOptions::DF32_32_32_32:
            return "DF32_32_32_32";
        case GFX9BufferDescriptorOptions::DFReserved:
            return "DFReserved";
        };

        Throw<FatalError>("Invalid DataFormatValue: " + ShowValue((int)(val)));
    }

    std::ostream& operator<<(std::ostream& stream, GFX9BufferDescriptorOptions::DataFormatValue val)
    {
        return stream << toString(val);
    }

    GFX9BufferDescriptorOptions::GFX9BufferDescriptorOptions()
    {
        setRawValue(0);
        data_format = DF32;
    }

    GFX9BufferDescriptorOptions::GFX9BufferDescriptorOptions(uint32_t raw)
    {
        setRawValue(raw);
    }

    void GFX9BufferDescriptorOptions::setRawValue(uint32_t val)
    {
        static_assert(sizeof(val) == sizeof(*this));

        memcpy(this, &val, sizeof(val));

        validate();
    }

    void GFX9BufferDescriptorOptions::validate() const
    {
        AssertFatal(_unusedA == 0 && _unusedB == 0, "Reserved bits must be set to 0\n", toString());
        AssertFatal(type == 0, "Resource type must be set to 0 for buffers\n", toString());
    }

    uint32_t GFX9BufferDescriptorOptions::rawValue() const
    {
        uint32_t rv;

        static_assert(sizeof(rv) == sizeof(*this));

        memcpy(&rv, this, sizeof(rv));
        return rv;
    }

    Register::ValuePtr GFX9BufferDescriptorOptions::literal() const
    {
        return Register::Value::Literal(rawValue());
    }

    std::string GFX9BufferDescriptorOptions::toString() const
    {
        std::ostringstream msg;

        auto flags = msg.flags();
        msg << "GFX9BufferDescriptorOptions: " << std::showbase << std::hex << std::internal
            << std::setfill('0') << std::setw(2 + 32 / 4) << rawValue() << std::endl;
        msg.flags(flags);

        msg << "    dst_sel_x: " << dst_sel_x << std::endl;
        msg << "    dst_sel_y: " << dst_sel_y << std::endl;
        msg << "    dst_sel_z: " << dst_sel_z << std::endl;
        msg << "    dst_sel_w: " << dst_sel_w << std::endl;
        msg << "    num_format: " << num_format << std::endl;
        msg << "    data_format: " << data_format << std::endl;
        msg << "    user_vm_enable: " << user_vm_enable << std::endl;
        msg << "    user_vm_mode: " << user_vm_mode << std::endl;
        msg << "    index_stride: " << index_stride << std::endl;
        msg << "    add_tid_enable: " << add_tid_enable << std::endl;
        msg << "    _unusedA: " << _unusedA << std::endl;
        msg << "    nv: " << nv << std::endl;
        msg << "    _unusedB: " << _unusedB << std::endl;
        msg << "    type: " << type << std::endl;

        return msg.str();
    }

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

    Generator<Instruction> BufferDescriptor::setDefaultOpts()
    {
        if(m_context->targetArchitecture().HasCapability(
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
            uint32_t opts = (1u << 28) | (5u << 12);
            co_yield m_context->copier()->copy(m_bufferResourceDescriptor->subset({3}),
                                               Register::Value::Literal(opts),
                                               "default options");
        }
        else
        {
            co_yield m_context->copier()->copy(m_bufferResourceDescriptor->subset({3}),
                                               Register::Value::Literal(0x00020000),
                                               "default options");
        }
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

    Register::ValuePtr BufferDescriptor::basePointerAndStride() const
    {
        return m_bufferResourceDescriptor->subset({0, 1});
    }

    Register::ValuePtr BufferDescriptor::size() const
    {
        return m_bufferResourceDescriptor->subset({2});
    }

    Register::ValuePtr BufferDescriptor::descriptorOptions() const
    {
        return m_bufferResourceDescriptor->subset({3});
    }
}

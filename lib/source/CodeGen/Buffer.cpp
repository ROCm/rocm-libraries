/**
 * @copyright Copyright 2022 Advanced Micro Devices, Inc.
 */

#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Buffer.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Context.hpp>

namespace rocRoller
{
    BufferDescriptor::BufferDescriptor(std::shared_ptr<Context> context)
    {
        m_bufferResourceDescriptor = std::make_shared<Register::Value>(
            context, Register::Type::Scalar, DataType::Raw32, 4);
        m_context = context;
    }

    Generator<Instruction> BufferDescriptor::setup()
    {
        co_yield m_bufferResourceDescriptor->allocate();
    }

    Generator<Instruction>
        BufferDescriptor::incrementBasePointer(std::shared_ptr<Register::Value> value)
    {
        co_yield generateOp<Expression::Add>(m_bufferResourceDescriptor->subset({0, 1}),
                                             m_bufferResourceDescriptor->subset({0, 1}),
                                             value);
    }

    Generator<Instruction> BufferDescriptor::setBasePointer(std::shared_ptr<Register::Value> value)
    {
        co_yield m_context->copier()->copy(m_bufferResourceDescriptor->subset({0, 1}), value, "");
    }

    Generator<Instruction> BufferDescriptor::setSize(std::shared_ptr<Register::Value> value)
    {
        co_yield m_context->copier()->copy(m_bufferResourceDescriptor->subset({2}), value, "");
    }

    Generator<Instruction> BufferDescriptor::setOptions(std::shared_ptr<Register::Value> value)
    {
        co_yield m_context->copier()->copy(m_bufferResourceDescriptor->subset({3}), value, "");
    }

    std::shared_ptr<Register::Value> BufferDescriptor::allRegisters() const
    {
        return m_bufferResourceDescriptor;
    }

    std::shared_ptr<Register::Value> BufferDescriptor::basePointerAndStride() const
    {
        return m_bufferResourceDescriptor->subset({0, 1});
    }

    std::shared_ptr<Register::Value> BufferDescriptor::size() const
    {
        return m_bufferResourceDescriptor->subset({2});
    }

    std::shared_ptr<Register::Value> BufferDescriptor::descriptorOptions() const
    {
        return m_bufferResourceDescriptor->subset({3});
    }
}

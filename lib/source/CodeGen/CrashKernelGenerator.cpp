/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/CrashKernelGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/InstructionValues/LabelAllocator.hpp>

namespace rocRoller
{
    CrashKernelGenerator::CrashKernelGenerator(ContextPtr context)
        : m_context(context)
    {
    }

    CrashKernelGenerator::~CrashKernelGenerator() = default;

    Generator<Instruction> CrashKernelGenerator::generateCrashSequence(AssertOpKind assertOpKind)
    {
        switch(assertOpKind)
        {
        case AssertOpKind::MemoryViolation:
        {
            auto context = m_context.lock();
            co_yield writeToNullPtr();
            co_yield Instruction::Wait(
                WaitCount::Zero(context->targetArchitecture(), "DEBUG: Wait after memory write"));
        }
        break;
        case AssertOpKind::STrap:
            co_yield sTrap();
            break;
        case AssertOpKind::NoOp:
            Throw<FatalError>("Unexpected AssertOpKind::NoOp");
        default:
            Throw<FatalError>("Unknown AssertOpKind");
        }
    }

    Generator<Instruction> rocRoller::CrashKernelGenerator::writeToNullPtr()
    {
        auto context     = m_context.lock();
        auto invalidAddr = std::make_shared<Register::Value>(
            context, Register::Type::Vector, DataType::Int64, 1);
        co_yield context->copier()->copy(invalidAddr, Register::Value::Literal(0L));

        auto dummyData = std::make_shared<Register::Value>(
            context, Register::Type::Vector, DataType::Int32, 1);
        co_yield context->copier()->copy(dummyData, Register::Value::Literal(42));

        co_yield context->mem()->storeGlobal(invalidAddr, dummyData, /*offset*/ 0, /*numBytes*/ 4);
    }

    Generator<Instruction> CrashKernelGenerator::sTrap()
    {
        auto context = m_context.lock();
        auto trapID  = Register::Value::Literal(0x02);
        co_yield_(Instruction("s_trap", {}, {trapID}, {}, ""));
    }
}

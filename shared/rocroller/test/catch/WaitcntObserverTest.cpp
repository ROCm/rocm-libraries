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

#include "TestContext.hpp"

#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/CodeGen/WaitCount.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/InstructionValues/Register.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

using namespace rocRoller;
using Catch::Matchers::ContainsSubstring;

TEST_CASE("EmptyQueue adds s_waitcnt only when queue is not empty", "[observer][waitcnt]")
{
    auto context = TestContext::ForDefaultTarget();
    auto arch    = context->targetArchitecture();

    std::string otherInstruction = GENERATE("", "ds_read_b32", "global_load_dwordx2");

    DYNAMIC_SECTION("Other instruction: " << otherInstruction)
    {

        auto v = context.createRegisters(Register::Type::Vector, DataType::Float, 2, 2);

        if(!otherInstruction.empty())
        {
            auto inst = Instruction(otherInstruction, {v[0]}, {v[1]}, {}, "");

            context->schedule(inst);
        }

        SECTION("Queue not empty: s_waitcnt lgkmcnt(0) is emitted")
        {
            auto s    = context.createRegisters(Register::Type::Scalar, DataType::UInt32, 2, 2);
            auto zero = Register::Value::Literal(0);

            auto s_load = Instruction("s_load_dwordx2", {s[1]}, {s[0], zero}, {}, "");
            context->schedule(s_load);

            auto emptyQueueWait
                = Instruction::Wait(WaitCount::EmptyQueue(arch, GPUWaitQueueType::SMemQueue));
            context->schedule(emptyQueueWait);

            CHECK_THAT(context.output(), ContainsSubstring("s_waitcnt"));
            CHECK_THAT(context.output(), ContainsSubstring("lgkmcnt(0)"));
        }

        SECTION("Queue empty: no s_waitcnt is added")
        {

            auto emptyQueueWait
                = Instruction::Wait(WaitCount::EmptyQueue(arch, GPUWaitQueueType::SMemQueue));
            context->schedule(emptyQueueWait);

            CHECK_THAT(context.output(), !ContainsSubstring("s_waitcnt"));
        }
    }
}

// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
                = Instruction::Wait(WaitCount::SyncQueue(arch, GPUWaitQueueType::SMemQueue));
            context->schedule(emptyQueueWait);

            CHECK_THAT(context.output(), ContainsSubstring("s_waitcnt"));
            CHECK_THAT(context.output(), ContainsSubstring("lgkmcnt(0)"));
        }

        SECTION("Queue empty: no s_waitcnt is added")
        {
            auto emptyQueueWait
                = Instruction::Wait(WaitCount::SyncQueue(arch, GPUWaitQueueType::SMemQueue));
            context->schedule(emptyQueueWait);

            CHECK_THAT(context.output(), !ContainsSubstring("s_waitcnt"));
        }
    }
}

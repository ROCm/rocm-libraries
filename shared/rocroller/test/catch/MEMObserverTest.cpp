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

#include <cmath>
#include <memory>

#include "CustomMatchers.hpp"
#include "CustomSections.hpp"
#include "TestContext.hpp"
#include "TestKernels.hpp"

#include <common/SourceMatcher.hpp>
#include <common/TestValues.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/Scheduling/Observers/FunctionalUnit/MEMObserver.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_contains.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

using namespace rocRoller;

namespace MEMObserverTest
{
    using Catch::Matchers::ContainsSubstring;

    void peekAndSchedule(TestContext& context, Instruction& inst, uint expectedStalls = 0)
    {
        auto peeked = context->observer()->peek(inst);
        CHECK(peeked.stallCycles == expectedStalls);
        context->schedule(inst);
    }

    TEST_CASE("MEMObservers predicts Stall Cycles", "[observer]")
    {
        SUPPORTED_ARCH_SECTION(arch)
        {
            if(arch.isRDNAGPU())
            {
                SKIP("RDNA not supported yet");
            }

            SECTION("VMEM Instructions stall")
            {
                auto context = TestContext::ForTarget(arch);
                auto weights = Scheduling::VMEMObserver::getWeights(context.get());

                if(weights.vmemQueueSize != 3)
                {
                    SKIP("Test tailored to vmemQueueSize == 3");
                }

                auto s    = context.createRegisters(Register::Type::Scalar, DataType::UInt32, 8);
                auto zero = Register::Value::Literal(0);

                std::vector<Instruction> insts = {
                    Instruction("buffer_load_dwordx2", {s[1]}, {s[0], zero}, {}, ""),
                    Instruction("buffer_load_dwordx2", {s[3]}, {s[2], zero}, {}, ""),
                    Instruction("buffer_load_dwordx2", {s[5]}, {s[4], zero}, {}, ""),
                    Instruction("buffer_load_dwordx2", {s[7]}, {s[6], zero}, {}, ""),
                };

                peekAndSchedule(context, insts[0]);
                peekAndSchedule(context, insts[1]);
                peekAndSchedule(context, insts[2]);
                peekAndSchedule(context, insts[3], weights.vmemCycles - 2); // e.g. 384 - 3 + 1

                CHECK_THAT(context.output(), ContainsSubstring("CBNW: 0"));
                CHECK_THAT(context.output(), ContainsSubstring("Inc: 3"));
            }

            SECTION("DS Instructions stall")
            {
                auto context = TestContext::ForTarget(arch);
                auto weights = Scheduling::DSMEMObserver::getWeights(context.get());

                if(weights.vmemQueueSize != 3)
                {
                    SKIP("Test tailored to vmemQueueSize == 3");
                }

                auto s    = context.createRegisters(Register::Type::Scalar, DataType::UInt32, 8);
                auto zero = Register::Value::Literal(0);

                std::vector<Instruction> insts = {
                    Instruction("ds_read_b32", {s[1]}, {s[0], zero}, {}, ""),
                    Instruction("ds_read_b32", {s[3]}, {s[2], zero}, {}, ""),
                    Instruction("ds_read_b32", {s[5]}, {s[4], zero}, {}, ""),
                    Instruction("ds_read_b32", {s[7]}, {s[6], zero}, {}, ""),
                };

                peekAndSchedule(context, insts[0]);
                peekAndSchedule(context, insts[1]);
                peekAndSchedule(context, insts[2]);
                peekAndSchedule(context, insts[3], weights.dsmemCycles - 2); // e.g. 384 - 3 + 1

                CHECK_THAT(context.output(), ContainsSubstring("CBNW: 0"));
                CHECK_THAT(context.output(), ContainsSubstring("Inc: 3"));
            }

            SECTION("No stalls if waitcounts used for loads")
            {
                auto context = TestContext::ForTarget(arch);
                auto s       = context.createRegisters(Register::Type::Scalar, DataType::UInt32, 8);
                auto zero    = Register::Value::Literal(0);

                std::vector<Instruction> insts = {
                    Instruction("buffer_load_dwordx2", {s[1]}, {s[0], zero}, {}, ""),
                    Instruction::Wait(WaitCount::LoadCnt(context->targetArchitecture(), 0)),
                    Instruction("buffer_load_dwordx2", {s[3]}, {s[2], zero}, {}, ""),
                    Instruction("buffer_load_dwordx2", {s[5]}, {s[4], zero}, {}, ""),
                    Instruction("buffer_load_dwordx2", {s[7]}, {s[6], zero}, {}, ""),
                };

                peekAndSchedule(context, insts[0]);
                peekAndSchedule(context, insts[1]);
                peekAndSchedule(context, insts[2]);
                peekAndSchedule(context, insts[3]);
                peekAndSchedule(context, insts[4]);
            }

            SECTION("No stalls if waitcounts used for stores")
            {
                auto context = TestContext::ForTarget(arch);
                auto s       = context.createRegisters(Register::Type::Scalar, DataType::UInt32, 8);
                auto zero    = Register::Value::Literal(0);

                std::vector<Instruction> insts = {
                    Instruction("buffer_store_dwordx2", {s[1]}, {s[0], zero}, {}, ""),
                    Instruction::Wait(WaitCount::StoreCnt(context->targetArchitecture(), 0)),
                    Instruction("buffer_store_dwordx2", {s[3]}, {s[2], zero}, {}, ""),
                    Instruction("buffer_store_dwordx2", {s[5]}, {s[4], zero}, {}, ""),
                    Instruction("buffer_store_dwordx2", {s[7]}, {s[6], zero}, {}, ""),
                };

                peekAndSchedule(context, insts[0]);
                peekAndSchedule(context, insts[1]);
                peekAndSchedule(context, insts[2]);
                peekAndSchedule(context, insts[3]);
                peekAndSchedule(context, insts[4]);
            }
        }
    }
}

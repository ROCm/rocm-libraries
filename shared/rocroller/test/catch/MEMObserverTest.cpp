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

            SECTION("Instructions with latency affect tracked cycles")
            {
                auto context = TestContext::ForTarget(arch);
                auto s       = context.createRegisters(Register::Type::Scalar, DataType::UInt32, 4);
                auto v_f16   = context.createRegisters(Register::Type::Vector, DataType::Half, 3);
                auto a = context.createRegisters(Register::Type::Accumulator, DataType::Float, 2);
                auto zero = Register::Value::Literal(0);

                std::string const& mfma = "v_mfma_f32_32x32x8f16";

                std::vector<Instruction> insts = {
                    Instruction("buffer_store_dwordx2", {s[1]}, {s[0], zero}, {}, ""),
                    Instruction(mfma, {a[0]}, {v_f16[0], v_f16[1], a[0]}, {}, ""),
                    Instruction(mfma, {a[1]}, {v_f16[1], v_f16[2], a[1]}, {}, ""),
                    Instruction("buffer_store_dwordx2", {s[3]}, {s[2], zero}, {}, ""),
                };

                auto const& architecture = context.get()->targetArchitecture();
                auto        cycles       = architecture.GetInstructionInfo(mfma).getLatency();

                peekAndSchedule(context, insts[0]);
                peekAndSchedule(context, insts[1]);
                peekAndSchedule(context, insts[2], cycles); // latency caused by insts[1]
                peekAndSchedule(context, insts[3]);

                CHECK_THAT(context.output(),
                           ContainsSubstring("current "
                                             + std::to_string(4 + cycles))); // 4 insts + latency
            }

            SECTION("VMEM Instructions with dependency")
            {
                auto context = TestContext::ForTarget(arch);
                auto weights = Scheduling::VMEMObserver::getWeights(context.get());

                if(weights.vmemQueueSize != 3)
                {
                    SKIP("Test tailored to vmemQueueSize == 3");
                }

                auto s    = context.createRegisters(Register::Type::Scalar, DataType::UInt32, 13);
                auto zero = Register::Value::Literal(0);

                std::vector<Instruction> insts = {
                    Instruction("buffer_load_dwordx2", {s[1]}, {s[0], zero}, {}, ""), // pc+=1
                    Instruction("buffer_load_dwordx2", {s[3]}, {s[2], zero}, {}, ""), // pc+=1
                    Instruction("buffer_load_dwordx2", {s[5]}, {s[4], zero}, {}, ""), // pc+=1
                    Instruction("buffer_load_dwordx2",
                                {s[7]},
                                {s[6], zero},
                                {},
                                ""), // pc=inst[0].expected+1, expected=pc+vmemCycles

                    Instruction("buffer_load_dwordx2",
                                {s[8]},
                                {s[3], zero},
                                {},
                                ""), // pc+=2 (including s_waitcnt)

                    Instruction("buffer_load_dwordx2", {s[10]}, {s[9], zero}, {}, ""), // pc+=1
                    Instruction("buffer_load_dwordx2",
                                {s[12]},
                                {s[11], zero},
                                {},
                                ""), // pc=inst[3].expected+1
                };

                peekAndSchedule(context, insts[0]);
                peekAndSchedule(context, insts[1]);
                peekAndSchedule(context, insts[2]);
                peekAndSchedule(context, insts[3], weights.vmemCycles - 2); // inst[0].expected - pc
                peekAndSchedule(context, insts[4]); // dependency. generate waitcount.

                peekAndSchedule(context, insts[5]);
                peekAndSchedule(context, insts[6], weights.vmemCycles - 3); // inst[3].expected - pc

                CHECK_THAT(context.output(), ContainsSubstring("CBNW: 0, Inc: 3"));
                CHECK_THAT(context.output(), ContainsSubstring("CBNW: 1, Inc: 3"));

                CHECK_THAT(context.output(), ContainsSubstring("s_waitcnt vmcnt(2)"));
            }
        }
    }
}

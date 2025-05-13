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

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <common/SourceMatcher.hpp>

#include "CustomSections.hpp"
#include "TestContext.hpp"

#include <rocRoller/CodeGen/BranchGenerator.hpp>
#include <rocRoller/InstructionValues/LabelAllocator.hpp>
#include <rocRoller/Scheduling/Scheduler.hpp>

using namespace rocRoller;

namespace BranchGeneratorTest
{
    TEST_CASE("fail to generate invalid branches", "[branches][codegen]")
    {
        auto ctx      = TestContext::ForDefaultTarget().get();
        auto brancher = BranchGenerator(ctx);

        auto l0 = Register::Value::Label("l0");
        auto s0 = Register::Value::Placeholder(ctx, Register::Type::Scalar, DataType::UInt32, 1);
        s0->allocateNow();

        CHECK_THROWS_AS(ctx->schedule(brancher.branch(s0)), FatalError);
        CHECK_THROWS_AS(ctx->schedule(brancher.branchConditional(s0, s0, true)), FatalError);
    }

    TEST_CASE("basic branch code generation", "[branches][codegen]")
    {
        SUPPORTED_ARCH_SECTION(arch)
        {
            auto ctx = TestContext::ForTarget(arch).get();
            auto k   = ctx->kernel();

            ctx->schedule(k->preamble());
            ctx->schedule(k->prolog());

            auto kb = [&]() -> Generator<Instruction> {
                auto l0 = ctx->labelAllocator()->label("l0");
                co_yield Instruction::Label(l0);

                auto wavefront_size = k->wavefront_size();

                auto scc = ctx->getSCC();
                auto vcc = ctx->getVCC();

                auto s0
                    = Register::Value::Placeholder(ctx,
                                                   Register::Type::Scalar,
                                                   DataType::UInt32,
                                                   wavefront_size / 32,
                                                   Register::AllocationOptions::FullyContiguous());
                co_yield s0->allocate();

                co_yield ctx->brancher()->branch(l0);

                co_yield ctx->brancher()->branchConditional(l0, s0, true);
                co_yield ctx->brancher()->branchConditional(l0, s0, false);
                co_yield ctx->brancher()->branchIfZero(l0, s0);
                co_yield ctx->brancher()->branchIfNonZero(l0, s0);

                co_yield ctx->brancher()->branchConditional(l0, scc, true);
                co_yield ctx->brancher()->branchConditional(l0, scc, false);
                co_yield ctx->brancher()->branchIfZero(l0, scc);
                co_yield ctx->brancher()->branchIfNonZero(l0, scc);

                co_yield ctx->brancher()->branchConditional(l0, vcc, true);
                co_yield ctx->brancher()->branchConditional(l0, vcc, false);
                co_yield ctx->brancher()->branchIfZero(l0, vcc);
                co_yield ctx->brancher()->branchIfNonZero(l0, vcc);
            };

            ctx->schedule(kb());
            ctx->schedule(k->postamble());
            ctx->schedule(k->amdgpu_metadata());

            std::vector<char> assembledKernel = ctx->instructions()->assemble();
            CHECK(assembledKernel.size() > 0);
        }
    }

    TEST_CASE("expect waits before branch if alwaysWaitBeforeBranch is true", "[branches][codegen]")
    {
        SUPPORTED_ARCH_SECTION(arch)
        {
            auto testCtx = TestContext::ForTarget(arch, {.alwaysWaitBeforeBranch = true});
            auto ctx     = testCtx.get();
            auto k       = ctx->kernel();

            ctx->schedule(k->preamble());
            ctx->schedule(k->prolog());

            auto kb = [&]() -> Generator<Instruction> {
                auto l0 = ctx->labelAllocator()->label("l0");
                co_yield Instruction::Lock(Scheduling::Dependency::Branch);
                co_yield Instruction::Label(l0);
                co_yield ctx->brancher()->branch(l0);
                co_yield Instruction::Unlock();
            };

            auto const& arch     = ctx->targetArchitecture();
            std::string expected = std::string("s_waitcnt vmcnt(0) lgkmcnt(0)");
            if(arch.HasCapability(GPUCapability::HasExpcnt))
            {
                expected += std::string(" expcnt(0)");
            }
            expected += std::string("\n") + std::string("s_branch");
            if(arch.HasCapability(GPUCapability::HasSplitWaitCounters))
            {
                expected = std::string("s_wait_loadcnt 0\ns_wait_storecnt 0\ns_wait_kmcnt "
                                       "0\ns_wait_dscnt 0");
                if(arch.HasCapability(GPUCapability::HasExpcnt))
                {
                    expected += std::string("\ns_wait_expcnt 0");
                }
                expected += std::string("\n") + std::string("s_branch");
            }

            auto scheduler = Component::GetNew<Scheduling::Scheduler>(
                Scheduling::SchedulerProcedure::Sequential, Scheduling::CostFunction::None, ctx);
            std::vector<Generator<Instruction>> generators;
            generators.push_back(kb());
            ctx->schedule((*scheduler)(generators));
            CHECK(NormalizedSource(testCtx.output()).find(expected) != std::string::npos);
        }
    }

    TEST_CASE("DO NOT wait before branch if alwaysWaitBeforeBranch is false", "[branches][codegen]")
    {
        SUPPORTED_ARCH_SECTION(arch)
        {
            auto testCtx = TestContext::ForTarget(arch, {.alwaysWaitBeforeBranch = false});
            auto ctx     = testCtx.get();
            auto k       = ctx->kernel();

            ctx->schedule(k->preamble());
            ctx->schedule(k->prolog());
            auto kb = [&]() -> Generator<Instruction> {
                auto l0 = ctx->labelAllocator()->label("l0");
                co_yield Instruction::Lock(Scheduling::Dependency::Branch);
                co_yield Instruction::Label(l0);
                co_yield ctx->brancher()->branch(l0);
                co_yield Instruction::Unlock();
            };

            auto const& arch     = ctx->targetArchitecture();
            std::string expected = std::string("s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)");
            if(arch.HasCapability(GPUCapability::HasExpcnt))
            {
                expected += std::string(" expcnt(0)");
            }
            expected += std::string("\n") + std::string("s_branch");
            if(arch.HasCapability(GPUCapability::HasSplitWaitCounters))
            {
                expected = std::string("s_wait_loadcnt 0\ns_wait_storecnt 0\ns_wait_kmcnt "
                                       "0\ns_wait_dscnt 0\ns_wait_expcnt 0");
                if(arch.HasCapability(GPUCapability::HasExpcnt))
                {
                    expected += std::string("s_wait_expcnt 0");
                }
                expected += std::string("\n") + std::string("s_branch");
            }

            auto scheduler = Component::GetNew<Scheduling::Scheduler>(
                Scheduling::SchedulerProcedure::Sequential, Scheduling::CostFunction::None, ctx);
            std::vector<Generator<Instruction>> generators;
            generators.push_back(kb());
            ctx->schedule((*scheduler)(generators));
            CHECK_FALSE(NormalizedSource(testCtx.output()).find(expected) != std::string::npos);
        }
    }
}

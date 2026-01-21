// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Scheduling/Observers/FunctionalUnit/WMMAObserver.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "TestContext.hpp"

using namespace rocRoller;

namespace WMMAObserverUnitTests
{
    class WMMAObserverUnitTest : public TestContext
    {
    public:
        WMMAObserverUnitTest(GPUArchitectureGFX gfx)
            : TestContext(TestContext::ForTarget({gfx})){};
    };

    TEST_CASE("Unit test observer regarding WMMA hazards on GFX1200/1201", "[observer]")
    {
        auto target = GENERATE(GPUArchitectureGFX::GFX1200, GPUArchitectureGFX::GFX1201);
        WMMAObserverUnitTest t{target};
        std::string          opCode{"v_wmma_f32_16x16x16_f16"};
        auto                 ctx = t.get();

        const auto v0
            = Register::Value::Placeholder(ctx, Register::Type::Vector, DataType::Float, 8);
        const auto v1
            = Register::Value::Placeholder(ctx, Register::Type::Vector, DataType::Half, 4);
        const auto v2
            = Register::Value::Placeholder(ctx, Register::Type::Vector, DataType::Half, 4);
        v0->allocateNow();
        v1->allocateNow();
        v2->allocateNow();

        auto wmmaInst = Instruction(opCode, {v0}, {v1, v2, v0}, {}, "");
        auto valuInst = Instruction("v_add_f32", {v1}, {v1, v2}, {}, "");

        const auto info    = ctx->targetArchitecture().GetInstructionInfo(wmmaInst.getOpCode());
        const auto latency = info.getLatency();

        SECTION("Use observer object directly")
        {
            Scheduling::WMMAObserver observer(ctx);

            CHECK(0 == observer.peek(wmmaInst).stallCycles);
            CHECK(0 == observer.peek(valuInst).stallCycles);

            observer.observe(wmmaInst);

            CHECK(latency == observer.peek(wmmaInst).stallCycles);
            CHECK(0 == observer.peek(valuInst).stallCycles);

            observer.observe(valuInst);

            CHECK(latency - 1 == observer.peek(wmmaInst).stallCycles);
            CHECK(0 == observer.peek(valuInst).stallCycles);

            observer.observe(wmmaInst);

            CHECK(latency == observer.peek(wmmaInst).stallCycles);
            CHECK(0 == observer.peek(valuInst).stallCycles);
        }

        SECTION("Use observer object through context")
        {
            CHECK(0 == ctx->peek(wmmaInst).stallCycles);
            CHECK(0 == ctx->peek(valuInst).stallCycles);

            ctx->schedule(wmmaInst);

            CHECK(latency == ctx->peek(wmmaInst).stallCycles);
            CHECK(0 == ctx->peek(valuInst).stallCycles);

            ctx->schedule(valuInst);

            CHECK(latency - 1 == ctx->peek(wmmaInst).stallCycles);
            CHECK(0 == ctx->peek(valuInst).stallCycles);

            ctx->schedule(wmmaInst);

            CHECK(latency == ctx->peek(wmmaInst).stallCycles);
            CHECK(0 == ctx->peek(valuInst).stallCycles);
        }
    }
}

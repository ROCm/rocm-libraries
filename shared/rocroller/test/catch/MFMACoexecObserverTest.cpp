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

#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Scheduling/Observers/FunctionalUnit/MFMAObserver.hpp>

#include <catch2/catch_test_macros.hpp>

using namespace rocRoller;

namespace MFMACoexecObserverTest
{
    TEST_CASE("MFMACoexecObserver predicts Stall Cycles", "[observer]")
    {
        Settings::getInstance()->set(Settings::SchedulerCost,
                                     Scheduling::CostFunction::LinearWeightedSimple);

        auto context = TestContext::ForTarget({GPUArchitectureGFX::GFX950});

        Scheduling::MFMACoexecObserver observer(context.get());

        auto schedule = [&](auto inst) {
            auto status = observer.peek(inst);
            inst.setPeekedStatus(status);
            observer.observe(inst);
            return inst;
        };

        auto agpr = Register::Value::Placeholder(
            context.get(), Register::Type::Accumulator, DataType::Float, 16);
        agpr->allocateNow();

        auto v0 = Register::Value::Placeholder(
            context.get(), Register::Type::Vector, DataType::Half, 4);
        v0->allocateNow();

        auto v1 = Register::Value::Placeholder(
            context.get(), Register::Type::Vector, DataType::Half, 4);
        v1->allocateNow();

        auto v2 = Register::Value::Placeholder(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        v1->allocateNow();

        auto bufDesc
            = Register::Value::Placeholder(context.get(),
                                           Register::Type::Scalar,
                                           VariableType(DataType::Float, PointerType::Buffer),
                                           1);
        bufDesc->allocateNow();

        auto s0 = Register::Value::Placeholder(
            context.get(), Register::Type::Scalar, DataType::Int32, 1);
        s0->allocateNow();

        auto mfmaInst
            = Instruction("v_mfma_scale_f32_16x16x128_f8f6f4", {agpr}, {v0, v1, agpr}, {}, "");
        auto valuInst = Instruction("v_add_f32", {v0}, {v0, v1}, {}, "");
        auto saluInst = Instruction("s_add_b32", {v0}, {v0, v1}, {}, "");

        auto dsInst     = Instruction("ds_read_b128", {v0}, {v2}, {"offset:1024"}, "");
        auto bufferInst = Instruction(
            "buffer_load_dwordx4", {v0}, {bufDesc, s0}, {"offen", "offset:0", "lds"}, "");

        CHECK(Scheduling::MFMACoexecObserver::isTargetedInstruction(mfmaInst));

        CHECK(observer.peek(mfmaInst).stallCycles == 0);
        CHECK(observer.peek(valuInst).stallCycles == 0);

        schedule(mfmaInst);

        CHECK(observer.peek(mfmaInst).stallCycles == 2);
        CHECK(observer.peek(valuInst).stallCycles == 1);
        CHECK(observer.peek(dsInst).stallCycles == 1);
        CHECK(observer.peek(bufferInst).stallCycles == 1);
        CHECK(observer.peek(saluInst).stallCycles == 0);

        // Each of these sections run after starting the test over.

        SECTION("SALU after MFMA")
        {
            schedule(saluInst);

            CHECK(observer.peek(mfmaInst).stallCycles == 1);
            CHECK(observer.peek(valuInst).stallCycles == 0);
            CHECK(observer.peek(dsInst).stallCycles == 0);
            CHECK(observer.peek(bufferInst).stallCycles == 0);
            CHECK(observer.peek(saluInst).stallCycles == 0);
        }

        SECTION("VALU after MFMA")
        {
            schedule(valuInst);

            CHECK(observer.peek(mfmaInst).stallCycles == 0);
            CHECK(observer.peek(valuInst).stallCycles == 0);
            CHECK(observer.peek(dsInst).stallCycles == 0);
            CHECK(observer.peek(bufferInst).stallCycles == 0);
            CHECK(observer.peek(saluInst).stallCycles == 0);
        }

        SECTION("DS after MFMA")
        {
            schedule(dsInst);

            CHECK(observer.peek(mfmaInst).stallCycles == 0);
            CHECK(observer.peek(valuInst).stallCycles == 0);
            CHECK(observer.peek(dsInst).stallCycles == 0);
            CHECK(observer.peek(bufferInst).stallCycles == 0);
            CHECK(observer.peek(saluInst).stallCycles == 0);
        }

        SECTION("Buffer after MFMA")
        {
            schedule(bufferInst);

            CHECK(observer.peek(mfmaInst).stallCycles == 0);
            CHECK(observer.peek(valuInst).stallCycles == 0);
            CHECK(observer.peek(dsInst).stallCycles == 0);
            CHECK(observer.peek(bufferInst).stallCycles == 0);
            CHECK(observer.peek(saluInst).stallCycles == 0);
        }

        SECTION("MFMA after MFMA")
        {
            schedule(mfmaInst);

            CHECK(observer.peek(mfmaInst).stallCycles == 2);
            CHECK(observer.peek(valuInst).stallCycles == 1);
            CHECK(observer.peek(dsInst).stallCycles == 1);
            CHECK(observer.peek(bufferInst).stallCycles == 1);
            CHECK(observer.peek(saluInst).stallCycles == 0);
        }
    }
}

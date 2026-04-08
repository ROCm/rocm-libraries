// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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

    TEST_CASE("MFMACoexecObserver half-speed (FP8) Stall Cycles", "[observer]")
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

        // Use FP8x4 sources to trigger the half-speed path.
        auto v0 = Register::Value::Placeholder(
            context.get(), Register::Type::Vector, DataType::FP8x4, 4);
        v0->allocateNow();

        auto v1 = Register::Value::Placeholder(
            context.get(), Register::Type::Vector, DataType::FP8x4, 4);
        v1->allocateNow();

        // Non-FP8 registers for other instruction types.
        auto v2 = Register::Value::Placeholder(
            context.get(), Register::Type::Vector, DataType::Half, 4);
        v2->allocateNow();

        auto v3 = Register::Value::Placeholder(
            context.get(), Register::Type::Vector, DataType::Int32, 1);
        v3->allocateNow();

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
        auto valuInst = Instruction("v_add_f32", {v2}, {v2, v2}, {}, "");
        auto saluInst = Instruction("s_add_b32", {v2}, {v2, v2}, {}, "");

        auto dsInst     = Instruction("ds_read_b128", {v2}, {v3}, {"offset:1024"}, "");
        auto bufferInst = Instruction(
            "buffer_load_dwordx4", {v2}, {bufDesc, s0}, {"offen", "offset:0", "lds"}, "");

        CHECK(Scheduling::MFMACoexecObserver::isTargetedInstruction(mfmaInst));

        CHECK(observer.peek(mfmaInst).stallCycles == 0);
        CHECK(observer.peek(valuInst).stallCycles == 0);

        schedule(mfmaInst);

        // Half-speed extends the disallowed window: XDL blocked for 6 cycles
        // (vs 2 for non-half-speed), VALU blocked for 2 (vs 1).
        CHECK(observer.peek(mfmaInst).stallCycles == 6);
        CHECK(observer.peek(valuInst).stallCycles == 2);
        CHECK(observer.peek(dsInst).stallCycles == 1);
        CHECK(observer.peek(bufferInst).stallCycles == 1);
        CHECK(observer.peek(saluInst).stallCycles == 0);

        SECTION("SALU after half-speed MFMA")
        {
            schedule(saluInst);

            CHECK(observer.peek(mfmaInst).stallCycles == 5);
            CHECK(observer.peek(valuInst).stallCycles == 1);
            CHECK(observer.peek(dsInst).stallCycles == 0);
            CHECK(observer.peek(bufferInst).stallCycles == 0);
            CHECK(observer.peek(saluInst).stallCycles == 0);
        }

        SECTION("VALU after half-speed MFMA")
        {
            schedule(valuInst);

            CHECK(observer.peek(mfmaInst).stallCycles == 3);
            CHECK(observer.peek(valuInst).stallCycles == 0);
            CHECK(observer.peek(dsInst).stallCycles == 0);
            CHECK(observer.peek(bufferInst).stallCycles == 0);
            CHECK(observer.peek(saluInst).stallCycles == 0);
        }

        SECTION("DS after half-speed MFMA")
        {
            schedule(dsInst);

            CHECK(observer.peek(mfmaInst).stallCycles == 4);
            CHECK(observer.peek(valuInst).stallCycles == 0);
            CHECK(observer.peek(dsInst).stallCycles == 0);
            CHECK(observer.peek(bufferInst).stallCycles == 0);
            CHECK(observer.peek(saluInst).stallCycles == 0);
        }

        SECTION("Buffer after half-speed MFMA")
        {
            schedule(bufferInst);

            CHECK(observer.peek(mfmaInst).stallCycles == 4);
            CHECK(observer.peek(valuInst).stallCycles == 0);
            CHECK(observer.peek(dsInst).stallCycles == 0);
            CHECK(observer.peek(bufferInst).stallCycles == 0);
            CHECK(observer.peek(saluInst).stallCycles == 0);
        }

        SECTION("MFMA after half-speed MFMA")
        {
            schedule(mfmaInst);

            CHECK(observer.peek(mfmaInst).stallCycles == 6);
            CHECK(observer.peek(valuInst).stallCycles == 2);
            CHECK(observer.peek(dsInst).stallCycles == 1);
            CHECK(observer.peek(bufferInst).stallCycles == 1);
            CHECK(observer.peek(saluInst).stallCycles == 0);
        }
    }
}

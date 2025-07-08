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

#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/Utilities/Settings_fwd.hpp>

#include <catch2/matchers/catch_matchers_string.hpp>

#include "TestContext.hpp"

#include <common/SourceMatcher.hpp>

using namespace rocRoller;

TEST_CASE("Instruction class works", "[codegen][instruction]")
{
    auto context = TestContext::ForDefaultTarget();

    auto dst = std::make_shared<Register::Value>(
        context.get(), Register::Type::Vector, DataType::Float, 1);
    dst->setName("C");

    auto src1 = std::make_shared<Register::Value>(
        context.get(), Register::Type::Vector, DataType::Float, 1);
    src1->setName("A");

    auto src2 = Register::Value::Literal(5);
    src2->setName("B");

    dst->allocateNow();
    src1->allocateNow();

    auto inst = Instruction("v_add_f32", {dst}, {src1, src2}, {}, "C = A + 5");

    CHECK("v_add_f32 v0, v1, 5 // C = A + 5\n" == inst.toString(LogLevel::Verbose));
    CHECK(inst.numExecutedInstructions() == 1);
}

// TEST_F(InstructionTest, ImplicitAllocation)
TEST_CASE("Instruction will implicitly allocate destination registers", "[codegen][instruction]")
{
    auto context = TestContext::ForDefaultTarget();

    auto dst = std::make_shared<Register::Value>(
        context.get(), Register::Type::Vector, DataType::Float, 1);
    dst->setName("C");

    auto src1 = std::make_shared<Register::Value>(
        context.get(), Register::Type::Vector, DataType::Float, 1);
    src1->setName("A");

    auto src2 = Register::Value::Literal(5);
    src2->setName("B");

    src1->allocateNow();

    auto inst = Instruction("v_add_f32", {dst}, {src1, src2}, {}, "C = A + 5");

    // dst is not allocated, scheduling an instruction writing to it should implicitly allocate it.
    context->schedule(inst);

    CHECK(dst->allocationState() == Register::AllocationState::Allocated);

    std::string coreString = "v_add_f32 v1, v0, 5 // C = A + 5\n";

    CHECK_THAT(context.output(), Catch::Matchers::ContainsSubstring(coreString));
}

// TEST_F(InstructionTest, NoSourceAllocation)
TEST_CASE("Instruction won't implicitly allocate source registers", "[codegen][instruction]")
{
    auto context = TestContext::ForDefaultTarget();

    auto dst = std::make_shared<Register::Value>(
        context.get(), Register::Type::Vector, DataType::Float, 1);
    dst->setName("C");

    auto src1 = std::make_shared<Register::Value>(
        context.get(), Register::Type::Vector, DataType::Float, 1);
    src1->setName("A");

    auto src2 = Register::Value::Literal(5);
    src2->setName("B");

    dst->allocateNow();

    auto inst = Instruction("v_add_f32", {dst}, {src1, src2}, {}, "C = A + 5");

    // src1 is not allocated, but we can't implicitly allocate a source operand for an instruction
    CHECK_THROWS(context->schedule(inst));
}

TEST_CASE("Instruction will allocate destination registers", "[codegen][instruction]")
{
    auto context = TestContext::ForDefaultTarget();

    auto reg = std::make_shared<Register::Value>(
        context.get(), Register::Type::Vector, DataType::Double, 1);
    reg->setName("C");

    auto allocInst = reg->allocate();

    CHECK(allocInst.numExecutedInstructions() == 0);

    CHECK(reg->allocationState() == Register::AllocationState::Unallocated);

    context->schedule(allocInst);

    CHECK(reg->allocationState() == Register::AllocationState::Allocated);

    CHECK(context.output() == "// Allocated C: 2 VGPRs (Value: Double): v0, v1\n");
}

TEST_CASE("Instruction class will handle comments.", "[codegen][instruction]")
{
    auto context = TestContext::ForDefaultTarget();

    auto inst = Instruction::Comment("Hello, World!");
    inst.addComment("Hello again!");
    context->schedule(inst);

    CHECK(context.output() == " // Hello, World!\n// Hello again!\n");
    CHECK(inst.toString(LogLevel::Verbose) == " // Hello, World!\n// Hello again!\n");
}

TEST_CASE("Instruction class will handle warnings.", "[codegen][instruction]")
{
    auto context = TestContext::ForDefaultTarget();

    auto inst = Instruction::Warning("There is a problem!");
    inst.addWarning("Another problem!");
    context->schedule(inst);

    CHECK_THAT(context.output(), Catch::Matchers::ContainsSubstring("There is a problem!"));
    CHECK_THAT(context.output(), Catch::Matchers::ContainsSubstring("Another problem!"));
    CHECK(inst.toString(LogLevel::Warning) == "// There is a problem!\n// Another problem!\n");
}

TEST_CASE("Instruction class will handle nops.", "[codegen][instruction]")
{
    auto context = TestContext::ForDefaultTarget();

    SECTION("NOPs will combine")
    {
        auto inst = Instruction::Nop();
        CHECK(inst.numExecutedInstructions() == 1);
        context->schedule(inst);

        inst = Instruction::Nop(5);
        inst.addNop();
        inst.addNop(3);
        CHECK(inst.numExecutedInstructions() == 9);
        context->schedule(inst);

        CHECK(context.output() == "s_nop 0\n\ns_nop 8\n\n");
    }

    // clearOutput();

    SECTION("Too many NOPs for one instruction will be split.")
    {
        auto inst = Instruction::Nop(17);
        context->schedule(inst);

        CHECK(context.output() == "s_nop 0xf\ns_nop 0\n\n");
    }
}

TEST_CASE("Instruction class will handle nops on a regular instruction.", "[codegen][instruction]")
{
    auto context = TestContext::ForDefaultTarget();

    auto dst = std::make_shared<Register::Value>(
        context.get(), Register::Type::Vector, DataType::Float, 1);
    auto src = std::make_shared<Register::Value>(
        context.get(), Register::Type::Vector, DataType::Float, 1);
    dst->allocateNow();
    src->allocateNow();

    auto inst = Instruction("v_add_f32", {dst}, {src, Register::Value::Literal(5)}, {}, "Addition");
    inst.setNopMin(3);
    CHECK(inst.numExecutedInstructions() == 4);

    context->schedule(inst);

    CHECK_THAT(context.output(), Catch::Matchers::ContainsSubstring("s_nop 2"));
}

TEST_CASE("Instruction class will handle labels.", "[codegen][instruction]")
{

    SECTION("Label from string")
    {
        auto inst = Instruction::Label("main_loop", "Main loop");
        CHECK(inst.isLabel());
        CHECK(inst.getLabel() == "main_loop");
    }

    SECTION("Label from Register::Value::Label")
    {
        auto label = Register::Value::Label("main_loop");
        auto inst  = Instruction::Label(label, "Main loop");
        CHECK(inst.isLabel());
        CHECK(inst.getLabel() == "main_loop");
    }

    SECTION("String rendering")
    {
        auto context = TestContext::ForDefaultTarget();

        const std::string label2 = "next loop";
        auto              inst   = Instruction::Label("main_loop");
        auto              inst2  = Instruction::Label(label2);
        context->schedule(inst);
        context->schedule(inst2);

        CHECK(inst.isLabel());
        CHECK(inst.getLabel() == "main_loop");

        CHECK(context.output() == "main_loop:\n\nnext loop:\n\n");
    }
}

TEST_CASE("Instruction class will handle waitcnt.", "[codegen][instruction]")
{
    auto context = TestContext::ForDefaultTarget();

    auto            arch  = context->targetArchitecture();
    auto            inst  = Instruction::Wait(WaitCount::EXPCnt(arch, 2));
    const WaitCount wait2 = WaitCount::EXPCnt(arch, 3);
    auto            inst2 = Instruction::Wait(wait2);
    context->schedule(inst);
    context->schedule(inst2);

    CHECK(inst.numExecutedInstructions() == 1);

    CHECK(context.output() == "s_waitcnt expcnt(2)\n\ns_waitcnt expcnt(3)\n\n");
}

TEST_CASE("Instruction class will handle wait zero.", "[codegen][instruction]")
{
    auto context = TestContext::ForDefaultTarget();

    auto inst = Instruction::Wait(WaitCount::Zero(context.get()->targetArchitecture()));
    context->schedule(inst);

    CHECK_THAT(context.output(), Catch::Matchers::ContainsSubstring("s_waitcnt"));

    CHECK_THAT(context.output(), Catch::Matchers::ContainsSubstring("vmcnt(0)"));
    CHECK_THAT(context.output(), Catch::Matchers::ContainsSubstring("lgkmcnt(0)"));
    CHECK_THAT(context.output(), Catch::Matchers::ContainsSubstring("expcnt(0)"));
}

TEST_CASE("Instruction class will handle a wait on a regular instruction.",
          "[codegen][instruction]")
{
    auto context = TestContext::ForDefaultTarget();

    auto dst = std::make_shared<Register::Value>(
        context.get(), Register::Type::Vector, DataType::Float, 1);
    auto src = std::make_shared<Register::Value>(
        context.get(), Register::Type::Vector, DataType::Float, 1);
    dst->allocateNow();
    src->allocateNow();

    auto inst = Instruction("v_add_f32", {dst}, {src, Register::Value::Literal(5)}, {}, "Addition");

    auto const& arch = context->targetArchitecture();
    inst.addWaitCount(WaitCount::KMCnt(arch, 1));
    context->schedule(inst);

    CHECK_THAT(context.output(), Catch::Matchers::ContainsSubstring("s_waitcnt lgkmcnt(1)\n"));
}

TEST_CASE("Instruction class will handle assembler directives.", "[codegen][instruction]")
{
    CHECK(Instruction::Directive(".amdgcn_target \"some-target\"").toString(LogLevel::Verbose)
          == ".amdgcn_target \"some-target\"\n");

    auto inst = Instruction::Directive(".set .amdgcn.next_free_vgpr, 0", "Comment");

    CHECK(inst.toString(LogLevel::Verbose) == ".set .amdgcn.next_free_vgpr, 0 // Comment\n");

    CHECK(inst.numExecutedInstructions() == 0);
}

TEST_CASE("Instruction class works with GPUInstructionInfo.", "[codegen][instruction]")
{
    auto context = TestContext::ForDefaultTarget();

    auto dst = std::make_shared<Register::Value>(
        context.get(), Register::Type::Vector, DataType::Float, 1);
    auto src = std::make_shared<Register::Value>(
        context.get(), Register::Type::Vector, DataType::Float, 1);
    dst->allocateNow();
    src->allocateNow();

    auto inst = Instruction("s_cmov_b32", {dst}, {src}, {}, "Not VALU");
    CHECK_FALSE(GPUInstructionInfo::isVALU(inst.getOpCode()));
}

TEST_CASE("Instruction class handles modifiers.", "[codegen][instruction]")
{
    auto context = TestContext::ForDefaultTarget();

    auto inst = Instruction("s_waitcnt", {}, {}, {"lgkmcnt(1)"}, "");
    context->schedule(inst);

    CHECK_THAT(context.output(), Catch::Matchers::ContainsSubstring("s_waitcnt  lgkmcnt(1)\n"));
}

TEST_CASE("Instruction class readsSpecialRegisters works correctly.", "[codegen][instruction]")
{
    auto context = TestContext::ForDefaultTarget();

    SECTION("As input operand")
    {
        auto inst = Instruction("s_arbitrary_instruction", {}, {context->getVCC()}, {}, "");
        CHECK(inst.readsSpecialRegisters());
    }

    SECTION("As output operand")
    {
        auto inst = Instruction("s_arbitrary_instruction", {context->getVCC()}, {}, {}, "");
        CHECK_FALSE(inst.readsSpecialRegisters());
    }

    SECTION("As input operand mixed with other operands")
    {
        auto dst = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Float, 1);
        auto src = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Float, 1);
        dst->allocateNow();
        src->allocateNow();
        auto inst = Instruction("s_arbitrary_instruction", {dst}, {src, context->getSCC()}, {}, "");
        CHECK(inst.readsSpecialRegisters());
    }

    SECTION("vcc_hi")
    {
        auto inst = Instruction("s_arbitrary_instruction", {}, {context->getVCC_HI()}, {}, "");
        CHECK(inst.readsSpecialRegisters());
    }

    SECTION("Regular operands aren't special")
    {
        auto dst = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Float, 1);
        auto src = std::make_shared<Register::Value>(
            context.get(), Register::Type::Vector, DataType::Float, 1);
        dst->allocateNow();
        src->allocateNow();
        auto inst = Instruction("s_arbitrary_instruction", {dst}, {src}, {}, "");
        CHECK_FALSE(inst.readsSpecialRegisters());
    }

    SECTION("Modifiers aren't checked")
    {
        auto inst = Instruction("s_arbitrary_instruction", {}, {}, {"vcc"}, "");
        CHECK_FALSE(inst.readsSpecialRegisters());
    }

    SECTION("Comments aren't checked")
    {
        auto inst = Instruction("s_arbitrary_instruction", {}, {}, {}, "vcc");
        CHECK_FALSE(inst.readsSpecialRegisters());
    }
}

TEST_CASE("Instruction class handles inout operands.", "[codegen][instruction]")
{
    auto context = TestContext::ForDefaultTarget();

    auto v0 = std::make_shared<Register::Value>(
        context.get(), Register::Type::Vector, DataType::Float, 1);
    auto v1 = std::make_shared<Register::Value>(
        context.get(), Register::Type::Vector, DataType::Float, 1);
    v0->allocateNow();
    v1->allocateNow();
    auto inst = Instruction::InoutInstruction("v_swap_b32", {v0, v1}, {}, "swap");

    CHECK(inst.hasRegisters());

    CHECK("v_swap_b32 v0, v1 // swap\n" == inst.toString(LogLevel::Verbose));
    Register::ValuePtr nullreg;

    CHECK(inst.getSrcs() == std::array{v0, v1, nullreg, nullreg, nullreg});
    CHECK(inst.getDsts() == std::array{v0, v1});
}

TEST_CASE("Instruction class handles extra operands.", "[codegen][instruction]")
{
    auto context = TestContext::ForDefaultTarget();

    auto v0 = std::make_shared<Register::Value>(
        context.get(), Register::Type::Vector, DataType::Float, 1);
    auto v1 = std::make_shared<Register::Value>(
        context.get(), Register::Type::Vector, DataType::Float, 1);

    v0->allocateNow();
    v1->allocateNow();

    Register::ValuePtr nullreg;

    auto bufDesc
        = std::make_shared<Register::Value>(context.get(),
                                            Register::Type::Scalar,
                                            DataType::Raw32,
                                            4,
                                            Register::AllocationOptions::FullyContiguous());
    bufDesc->allocateNow();

    auto lds = Register::Value::AllocateLDS(context.get(), DataType::BF6x16, 4);

    SECTION("Instruction with no regular operands.")
    {
        auto inst = Instruction("s_barrier", {}, {}, {}, "Barrier");

        SECTION("Empty")
        {
            CHECK(inst.getAllDsts().empty());
            CHECK(inst.getAllSrcs().empty());

            CHECK(inst.toString(LogLevel::Verbose) == "s_barrier  // Barrier\n");
            CHECK(NormalizedSource(inst.toString(LogLevel::Verbose))
                  == NormalizedSource("s_barrier"));

            CHECK_FALSE(inst.hasRegisters());
        }

        auto initialRepresentation = NormalizedSource(inst.toString(LogLevel::Verbose));

        SECTION("Extra dsts")
        {
            inst.addExtraDst(v0);
            CHECK(inst.getAllDsts().to<std::vector>() == std::vector{v0});
            CHECK(inst.hasRegisters());

            CHECK(inst.getExtraDsts() == std::array{v0, nullreg, nullreg, nullreg, nullreg});

            CHECK(NormalizedSource(inst.toString(LogLevel::Verbose)) == initialRepresentation);

            inst.addExtraSrc(v0);
            CHECK(inst.getAllDsts().to<std::vector>() == std::vector{v0});

            inst.addExtraDst(v1);
            CHECK(inst.getAllDsts().to<std::vector>() == std::vector{v0, v1});

            CHECK(inst.getExtraDsts() == std::array{v0, v1, nullreg, nullreg, nullreg});

            CHECK(NormalizedSource(inst.toString(LogLevel::Verbose)) == initialRepresentation);

            inst.addExtraDst(lds);
            CHECK(inst.getAllDsts().to<std::vector>() == std::vector{v0, v1, lds});

            CHECK(NormalizedSource(inst.toString(LogLevel::Verbose)) == initialRepresentation);

            CHECK(inst.getAllOperands().to<std::vector>() == std::vector{v0, v1, lds, v0});
        }

        SECTION("Extra srcs")
        {
            inst.addExtraSrc(v0);
            CHECK(inst.getAllSrcs().to<std::vector>() == std::vector{v0});

            CHECK(inst.getExtraSrcs() == std::array{v0, nullreg, nullreg, nullreg, nullreg});

            CHECK(NormalizedSource(inst.toString(LogLevel::Verbose)) == initialRepresentation);

            inst.addExtraDst(v0);
            CHECK(inst.getAllSrcs().to<std::vector>() == std::vector{v0});

            inst.addExtraSrc(v1);
            CHECK(inst.getAllSrcs().to<std::vector>() == std::vector{v0, v1});

            CHECK(NormalizedSource(inst.toString(LogLevel::Verbose)) == initialRepresentation);

            CHECK(inst.getExtraSrcs() == std::array{v0, v1, nullreg, nullreg, nullreg});

            inst.addExtraSrc(lds);
            CHECK(inst.getAllSrcs().to<std::vector>() == std::vector{v0, v1, lds});

            CHECK(NormalizedSource(inst.toString(LogLevel::Verbose)) == initialRepresentation);

            CHECK(inst.getAllOperands().to<std::vector>() == std::vector{v0, v0, v1, lds});
        }
    }

    SECTION("Instruction with regular operands.")
    {
        auto inst = Instruction("buffer_load_dword", {v0}, {bufDesc}, {}, "Load data");

        SECTION("Empty")
        {
            CHECK(inst.getAllDsts().to<std::vector>() == std::vector{v0});
            CHECK(inst.getAllSrcs().to<std::vector>() == std::vector{bufDesc});

            CHECK(inst.toString(LogLevel::Verbose)
                  == "buffer_load_dword v0, s[0:3] // Load data\n");
            CHECK(NormalizedSource(inst.toString(LogLevel::Verbose))
                  == NormalizedSource("buffer_load_dword v0, s[0:3]"));
        }

        auto initialRepresentation = NormalizedSource(inst.toString(LogLevel::Verbose));

        SECTION("Extra dsts")
        {
            CHECK(inst.getExtraDsts() == std::array{nullreg, nullreg, nullreg, nullreg, nullreg});

            inst.addExtraDst(v1);
            CHECK(inst.getAllDsts().to<std::vector>() == std::vector{v0, v1});

            CHECK(inst.getExtraDsts() == std::array{v1, nullreg, nullreg, nullreg, nullreg});

            CHECK(NormalizedSource(inst.toString(LogLevel::Verbose)) == initialRepresentation);

            inst.addExtraSrc(v0);
            CHECK(inst.getAllDsts().to<std::vector>() == std::vector{v0, v1});

            inst.addExtraDst(v0);
            CHECK(inst.getAllDsts().to<std::vector>() == std::vector{v0, v1, v0});

            CHECK(inst.getExtraDsts() == std::array{v1, v0, nullreg, nullreg, nullreg});

            CHECK(NormalizedSource(inst.toString(LogLevel::Verbose)) == initialRepresentation);

            CHECK(inst.getAllOperands().to<std::vector>() == std::vector{v0, v1, v0, bufDesc, v0});
        }

        SECTION("Extra srcs")
        {
            CHECK(inst.getExtraSrcs() == std::array{nullreg, nullreg, nullreg, nullreg, nullreg});

            inst.addExtraSrc(v0);
            CHECK(inst.getAllSrcs().to<std::vector>() == std::vector{bufDesc, v0});

            CHECK(inst.getExtraSrcs() == std::array{v0, nullreg, nullreg, nullreg, nullreg});

            CHECK(NormalizedSource(inst.toString(LogLevel::Verbose)) == initialRepresentation);

            inst.addExtraDst(v0);
            CHECK(inst.getAllSrcs().to<std::vector>() == std::vector{bufDesc, v0});

            inst.addExtraSrc(v1);
            CHECK(inst.getAllSrcs().to<std::vector>() == std::vector{bufDesc, v0, v1});

            CHECK(inst.getExtraSrcs() == std::array{v0, v1, nullreg, nullreg, nullreg});

            CHECK(NormalizedSource(inst.toString(LogLevel::Verbose)) == initialRepresentation);
        }
    }
}

TEST_CASE("Instruction isAfterWriteDependency works.", "[codegen][instruction]")
{
    using DestArray = std::array<Register::ValuePtr, Instruction::MaxDstRegisters>;

    auto context = TestContext::ForDefaultTarget();

    auto v0 = std::make_shared<Register::Value>(
        context.get(), Register::Type::Vector, DataType::Float, 1);
    v0->allocateNow();

    auto v1 = std::make_shared<Register::Value>(
        context.get(), Register::Type::Vector, DataType::Float, 1);
    v1->allocateNow();

    auto v2 = std::make_shared<Register::Value>(
        context.get(), Register::Type::Vector, DataType::Float, 1);
    v2->allocateNow();

    Register::ValuePtr nullreg;

    auto bufDesc
        = std::make_shared<Register::Value>(context.get(),
                                            Register::Type::Scalar,
                                            DataType::Raw32,
                                            4,
                                            Register::AllocationOptions::FullyContiguous());
    bufDesc->allocateNow();

    auto lds1 = Register::Value::AllocateLDS(context.get(), DataType::BF6x16, 4);
    auto lds2 = Register::Value::AllocateLDS(context.get(), DataType::BF6x16, 4);

    SECTION("Simple registers")
    {
        Instruction inst1("buffer_load_dword", {v0}, {bufDesc}, {}, "");

        Instruction inst2("v_add_u32", {v1}, {v0, v2}, {}, "");

        Instruction inst3("v_add_u32", {v2}, {v2, v2}, {}, "");

        CHECK(inst2.isAfterWriteDependency(inst1.getDsts()));
        CHECK_FALSE(inst3.isAfterWriteDependency(inst1.getDsts()));

        CHECK_FALSE(inst3.isAfterWriteDependency(std::vector{v0}));
        CHECK_FALSE(inst3.isAfterWriteDependency(std::vector{v1}));
        CHECK(inst3.isAfterWriteDependency(std::vector{v2}));

        SECTION("Without adding extra reg")
        {
            DestArray dsts;
            append(dsts, inst1.getAllDsts());

            CHECK(inst2.isAfterWriteDependency(dsts));
            CHECK_FALSE(inst3.isAfterWriteDependency(dsts));
        }

        SECTION("With adding extra to first inst")
        {
            inst1.addExtraDst(v2);

            DestArray dsts;
            append(dsts, inst1.getAllDsts());

            CHECK(inst2.isAfterWriteDependency(dsts));
            CHECK(inst3.isAfterWriteDependency(dsts));
        }

        SECTION("With adding extra to second inst")
        {
            inst3.addExtraSrc(v0);

            DestArray dsts;
            append(dsts, inst1.getAllDsts());

            CHECK(inst3.isAfterWriteDependency(dsts));

            CHECK(inst3.isAfterWriteDependency(std::vector{v0}));
            CHECK_FALSE(inst3.isAfterWriteDependency(std::vector{v1}));
            CHECK(inst3.isAfterWriteDependency(std::vector{v2}));
        }
    }

    SECTION("LDS 'registers'")
    {
        Instruction inst1("buffer_load_dword", {v0}, {bufDesc}, {}, "");

        Instruction inst2("v_add_u32", {v1}, {v0, v2}, {}, "");

        Instruction inst3("v_add_u32", {v2}, {v2, v2}, {}, "");

        CHECK(inst2.isAfterWriteDependency(inst1.getDsts()));
        CHECK_FALSE(inst3.isAfterWriteDependency(inst1.getDsts()));

        CHECK_FALSE(inst3.isAfterWriteDependency(std::vector{v0}));
        CHECK_FALSE(inst3.isAfterWriteDependency(std::vector{v1}));
        CHECK(inst3.isAfterWriteDependency(std::vector{v2}));

        SECTION("With adding extra to first inst")
        {
            inst1.addExtraDst(lds1);

            DestArray dsts;
            append(dsts, inst1.getAllDsts());

            CHECK(inst2.isAfterWriteDependency(dsts));
            CHECK_FALSE(inst3.isAfterWriteDependency(dsts));

            inst2.addExtraSrc(lds1);
            inst3.addExtraSrc(lds2);

            CAPTURE(inst1.toString(LogLevel::Debug),
                    inst2.toString(LogLevel::Debug),
                    inst3.toString(LogLevel::Debug));

            CHECK(inst2.isAfterWriteDependency(dsts));
            CHECK_FALSE(inst3.isAfterWriteDependency(dsts));

            CHECK_FALSE(inst1.isAfterWriteDependency(std::vector{lds1}));
            CHECK_FALSE(inst1.isAfterWriteDependency(std::vector{lds2}));

            CHECK(inst2.isAfterWriteDependency(std::vector{lds1}));
            CHECK_FALSE(inst2.isAfterWriteDependency(std::vector{lds2}));

            CHECK_FALSE(inst3.isAfterWriteDependency(std::vector{lds1}));
            CHECK(inst3.isAfterWriteDependency(std::vector{lds2}));
        }
    }
}

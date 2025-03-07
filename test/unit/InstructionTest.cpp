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

#include "GenericContextFixture.hpp"

using namespace rocRoller;

class InstructionTest : public GenericContextFixture
{
    void SetUp() override
    {
        Settings::getInstance()->set(Settings::AllowUnkownInstructions, true);
        GenericContextFixture::SetUp();
    }
};

TEST_F(InstructionTest, Basic)
{
    auto dst
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Float, 1);
    dst->setName("C");

    auto src1
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Float, 1);
    src1->setName("A");

    auto src2 = Register::Value::Literal(5);
    src2->setName("B");

    dst->allocateNow();
    src1->allocateNow();

    auto inst = Instruction("v_add_f32", {dst}, {src1, src2}, {}, "C = A + 5");

    EXPECT_EQ("v_add_f32 v0, v1, 5 // C = A + 5\n", inst.toString(LogLevel::Verbose));
    EXPECT_EQ(1, inst.numExecutedInstructions());
}

TEST_F(InstructionTest, ImplicitAllocation)
{
    auto dst
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Float, 1);
    dst->setName("C");

    auto src1
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Float, 1);
    src1->setName("A");

    auto src2 = Register::Value::Literal(5);
    src2->setName("B");

    src1->allocateNow();

    auto inst = Instruction("v_add_f32", {dst}, {src1, src2}, {}, "C = A + 5");

    // dst is not allocated, scheduling an instruction writing to it should implicitly allocate it.
    m_context->schedule(inst);

    EXPECT_EQ(Register::AllocationState::Allocated, dst->allocationState());

    std::string coreString = "v_add_f32 v1, v0, 5 // C = A + 5\n";

    EXPECT_THAT(output(), testing::HasSubstr(coreString));
}

TEST_F(InstructionTest, NoSourceAllocation)
{
    auto dst
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Float, 1);
    dst->setName("C");

    auto src1
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Float, 1);
    src1->setName("A");

    auto src2 = Register::Value::Literal(5);
    src2->setName("B");

    dst->allocateNow();

    auto inst = Instruction("v_add_f32", {dst}, {src1, src2}, {}, "C = A + 5");

    // src1 is not allocated, but we can't implicitly allocate a source operand for an instruction
    EXPECT_ANY_THROW(m_context->schedule(inst));
}

TEST_F(InstructionTest, Allocate)
{
    auto reg
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Double, 1);
    reg->setName("C");

    auto allocInst = reg->allocate();

    EXPECT_EQ(0, allocInst.numExecutedInstructions());

    EXPECT_EQ(Register::AllocationState::Unallocated, reg->allocationState());

    m_context->schedule(allocInst);

    EXPECT_EQ(Register::AllocationState::Allocated, reg->allocationState());

    EXPECT_EQ("// Allocated C: 2 VGPRs (Value: Double): v0, v1\n", output());
}

TEST_F(InstructionTest, Comment)
{
    auto inst = Instruction::Comment("Hello, World!");
    inst.addComment("Hello again!");
    m_context->schedule(inst);

    EXPECT_EQ(output(), " // Hello, World!\n// Hello again!\n");
    EXPECT_EQ(inst.toString(LogLevel::Verbose), " // Hello, World!\n// Hello again!\n");
}

TEST_F(InstructionTest, Warning)
{
    auto inst = Instruction::Warning("There is a problem!");
    inst.addWarning("Another problem!");
    m_context->schedule(inst);

    EXPECT_THAT(output(), testing::HasSubstr("There is a problem!"));
    EXPECT_THAT(output(), testing::HasSubstr("Another problem!"));
    EXPECT_THAT(inst.toString(LogLevel::Warning), "// There is a problem!\n// Another problem!\n");
}

TEST_F(InstructionTest, Nop)
{
    auto inst = Instruction::Nop();
    EXPECT_EQ(1, inst.numExecutedInstructions());
    m_context->schedule(inst);

    inst = Instruction::Nop(5);
    inst.addNop();
    inst.addNop(3);
    EXPECT_EQ(9, inst.numExecutedInstructions());
    m_context->schedule(inst);

    EXPECT_EQ("s_nop 0\n\ns_nop 8\n\n", output());

    clearOutput();

    inst = Instruction::Nop(17);
    m_context->schedule(inst);

    EXPECT_EQ("s_nop 0xf\ns_nop 0\n\n", output());
}

TEST_F(InstructionTest, NopOnRegularInstruction)
{
    auto dst
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Float, 1);
    auto src
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Float, 1);
    dst->allocateNow();
    src->allocateNow();

    auto inst = Instruction("v_add_f32", {dst}, {src, Register::Value::Literal(5)}, {}, "Addition");
    inst.setNopMin(3);
    EXPECT_EQ(4, inst.numExecutedInstructions());

    m_context->schedule(inst);

    EXPECT_THAT(output(), testing::HasSubstr("s_nop 2"));
}

TEST_F(InstructionTest, Label)
{
    {
        auto inst = Instruction::Label("main_loop", "Main loop");
        EXPECT_TRUE(inst.isLabel());
        EXPECT_EQ(inst.getLabel(), "main_loop");
    }

    {
        auto label = Register::Value::Label("main_loop");
        auto inst  = Instruction::Label(label, "Main loop");
        EXPECT_TRUE(inst.isLabel());
        EXPECT_EQ(inst.getLabel(), "main_loop");
    }

    {
        const std::string label2 = "next loop";
        auto              inst   = Instruction::Label("main_loop");
        auto              inst2  = Instruction::Label(label2);
        m_context->schedule(inst);
        m_context->schedule(inst2);

        EXPECT_TRUE(inst.isLabel());
        EXPECT_EQ(inst.getLabel(), "main_loop");

        EXPECT_EQ("main_loop:\n\nnext loop:\n\n", output());
    }
}

TEST_F(InstructionTest, Wait)
{
    auto            arch  = m_context->targetArchitecture();
    auto            inst  = Instruction::Wait(WaitCount::EXPCnt(arch, 2));
    const WaitCount wait2 = WaitCount::EXPCnt(arch, 3);
    auto            inst2 = Instruction::Wait(wait2);
    m_context->schedule(inst);
    m_context->schedule(inst2);
    EXPECT_EQ(1, inst.numExecutedInstructions());

    EXPECT_EQ("s_waitcnt expcnt(2)\n\ns_waitcnt expcnt(3)\n\n", output());
}

TEST_F(InstructionTest, WaitZero)
{
    auto inst = Instruction::Wait(WaitCount::Zero(m_context->targetArchitecture()));
    m_context->schedule(inst);

    EXPECT_THAT(output(), testing::HasSubstr("s_waitcnt"));

    EXPECT_THAT(output(), testing::HasSubstr("vmcnt(0)"));
    EXPECT_THAT(output(), testing::HasSubstr("lgkmcnt(0)"));
    EXPECT_THAT(output(), testing::HasSubstr("expcnt(0)"));
}

TEST_F(InstructionTest, WaitOnRegularInstruction)
{
    auto dst
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Float, 1);
    auto src
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Float, 1);
    dst->allocateNow();
    src->allocateNow();

    auto inst = Instruction("v_add_f32", {dst}, {src, Register::Value::Literal(5)}, {}, "Addition");

    auto arch = m_context->targetArchitecture();
    inst.addWaitCount(WaitCount::KMCnt(arch, 1));
    m_context->schedule(inst);

    EXPECT_THAT(output(), testing::HasSubstr("s_waitcnt lgkmcnt(1)\n"));
}

TEST_F(InstructionTest, Directive)
{
    EXPECT_EQ(".amdgcn_target \"some-target\"\n",
              Instruction::Directive(".amdgcn_target \"some-target\"").toString(LogLevel::Verbose));

    auto inst = Instruction::Directive(".set .amdgcn.next_free_vgpr, 0", "Comment");

    EXPECT_EQ(".set .amdgcn.next_free_vgpr, 0 // Comment\n", inst.toString(LogLevel::Verbose));

    EXPECT_EQ(0, inst.numExecutedInstructions());
}

TEST_F(InstructionTest, Classifications)
{
    auto dst
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Float, 1);
    auto src
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Float, 1);
    dst->allocateNow();
    src->allocateNow();

    auto inst = Instruction("s_cmov_b32", {dst}, {src}, {}, "Not VALU");
    EXPECT_FALSE(GPUInstructionInfo::isVALU(inst.getOpCode()));
}

TEST_F(InstructionTest, Special)
{
    // This pattern is used in the GuidePosts
    auto inst = Instruction("s_waitcnt", {}, {}, {"lgkmcnt(1)"}, "");
    m_context->schedule(inst);

    EXPECT_THAT(output(), testing::HasSubstr("s_waitcnt  lgkmcnt(1)\n"));
}

TEST_F(InstructionTest, ReadsSpecial)
{
    {
        auto inst = Instruction("s_arbitrary_instruction", {}, {m_context->getVCC()}, {}, "");
        EXPECT_TRUE(inst.readsSpecialRegisters());
    }
    {
        auto dst = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Float, 1);
        auto src = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Float, 1);
        dst->allocateNow();
        src->allocateNow();
        auto inst
            = Instruction("s_arbitrary_instruction", {dst}, {src, m_context->getSCC()}, {}, "");
        EXPECT_TRUE(inst.readsSpecialRegisters());
    }
    {
        auto inst = Instruction("s_arbitrary_instruction", {}, {m_context->getVCC_HI()}, {}, "");
        EXPECT_TRUE(inst.readsSpecialRegisters());
    }
    {
        auto inst = Instruction("s_arbitrary_instruction", {m_context->getVCC()}, {}, {}, "");
        EXPECT_FALSE(inst.readsSpecialRegisters());
    }
    {
        auto dst = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Float, 1);
        auto src = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Float, 1);
        dst->allocateNow();
        src->allocateNow();
        auto inst = Instruction("s_arbitrary_instruction", {dst}, {src}, {}, "");
        EXPECT_FALSE(inst.readsSpecialRegisters());
    }
    {
        auto inst = Instruction("s_arbitrary_instruction", {}, {}, {"vcc"}, "");
        EXPECT_FALSE(inst.readsSpecialRegisters());
    }
    {
        auto inst = Instruction("s_arbitrary_instruction", {}, {}, {}, "vcc");
        EXPECT_FALSE(inst.readsSpecialRegisters());
    }
}

TEST_F(InstructionTest, InoutInstructions)
{
    auto v0
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Float, 1);
    auto v1
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Float, 1);
    v0->allocateNow();
    v1->allocateNow();
    auto inst = Instruction::InoutInstruction("v_swap_b32", {v0, v1}, {}, "swap");

    EXPECT_EQ("v_swap_b32 v0, v1 // swap\n", inst.toString(LogLevel::Verbose));
    EXPECT_TRUE(inst.hasRegisters());
    EXPECT_EQ(inst.getSrcs()[0], v0);
    EXPECT_EQ(inst.getSrcs()[1], v1);
    EXPECT_EQ(inst.getSrcs()[2], nullptr);
    EXPECT_EQ(inst.getDsts()[0], v0);
    EXPECT_EQ(inst.getDsts()[1], v1);
}

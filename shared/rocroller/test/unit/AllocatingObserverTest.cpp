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

#include <rocRoller/CodeGen/BranchGenerator.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/Utilities/Utils.hpp>

#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"

using namespace rocRoller;

namespace rocRollerTest
{
    class AllocatingObserverTest : public GenericContextFixture
    {
    protected:
        GPUArchitectureTarget targetArchitecture() override
        {
            return {GPUArchitectureGFX::GFX90A};
        }
    };

    TEST_F(AllocatingObserverTest, Basic)
    {
        auto vgprIndex = static_cast<size_t>(Register::Type::Vector);
        rocRoller::Scheduling::InstructionStatus peeked;

        auto src1 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Float, 1);

        auto src2 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Float, 1);

        auto src3 = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Float, 1);

        auto zero = Register::Value::Literal(0);

        auto inst1 = Instruction("v_mov_b32", {src1}, {zero}, {}, "");
        peeked     = m_context->observer()->peek(inst1);
        EXPECT_EQ(peeked.allocatedRegisters[vgprIndex], 1);
        EXPECT_EQ(peeked.highWaterMarkRegistersDelta[vgprIndex], 1);
        EXPECT_EQ(255, peeked.remainingRegisters[vgprIndex]) << peeked.remainingRegisters;
        EXPECT_EQ(false, peeked.outOfRegisters.any());

        m_context->schedule(inst1);

        auto inst2 = Instruction("v_mov_b32", {src2}, {zero}, {}, "");
        peeked     = m_context->observer()->peek(inst1);
        EXPECT_EQ(peeked.allocatedRegisters[vgprIndex], 1);
        EXPECT_EQ(peeked.highWaterMarkRegistersDelta[vgprIndex], 1);
        EXPECT_EQ(254, peeked.remainingRegisters[vgprIndex]);
        EXPECT_EQ(false, peeked.outOfRegisters.any());

        m_context->schedule(inst2);

        auto inst3 = Instruction("v_add_f32", {src3}, {src1, src2}, {}, "");
        peeked     = m_context->observer()->peek(inst1);
        EXPECT_EQ(peeked.allocatedRegisters[vgprIndex], 1);
        EXPECT_EQ(peeked.highWaterMarkRegistersDelta[vgprIndex], 1);
        EXPECT_EQ(253, peeked.remainingRegisters[vgprIndex]);
        EXPECT_EQ(false, peeked.outOfRegisters.any());

        m_context->schedule(inst3);

        auto bigValue
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Int32, 300);
        peeked = m_context->peek(bigValue->allocate());
        EXPECT_EQ(peeked.allocatedRegisters[vgprIndex], 300);
        EXPECT_EQ(peeked.highWaterMarkRegistersDelta[vgprIndex], 0);
        EXPECT_EQ(-1, peeked.remainingRegisters[vgprIndex]);
        EXPECT_EQ(1, peeked.outOfRegisters.count());
        EXPECT_EQ(true, peeked.outOfRegisters[Register::Type::Vector]);

        auto inst_end = Instruction("s_endpgm", {}, {}, {}, "");
        peeked        = m_context->observer()->peek(inst_end);
        EXPECT_EQ(peeked.allocatedRegisters[vgprIndex], 0);
        EXPECT_EQ(peeked.highWaterMarkRegistersDelta[vgprIndex], 0);
        EXPECT_EQ(253, peeked.remainingRegisters[vgprIndex]);

        m_context->schedule(inst_end);

        std::string expected = R"(
                                   v_mov_b32 v0, 0
                                   v_mov_b32 v1, 0
                                   v_add_f32 v2, v0, v1
                                   s_endpgm
                                )";
        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(expected));
    }
}

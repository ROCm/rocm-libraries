
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rocRoller/CodeGen/BranchGenerator.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/InstructionValues/Register.hpp>

#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"

using namespace rocRoller;

namespace rocRollerTest
{
    class AllocatingObserverTest : public GenericContextFixture
    {
    protected:
        std::string targetArchitecture() override
        {
            return "gfx90a";
        }
    };

    TEST_F(AllocatingObserverTest, Basic)
    {
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
        EXPECT_EQ(peeked.allocatedRegisters[static_cast<size_t>(Register::Type::Vector)], 1);
        EXPECT_EQ(peeked.highWaterMarkRegistersDelta[static_cast<size_t>(Register::Type::Vector)],
                  1);

        m_context->schedule(inst1);

        auto inst2 = Instruction("v_mov_b32", {src2}, {zero}, {}, "");
        peeked     = m_context->observer()->peek(inst1);
        EXPECT_EQ(peeked.allocatedRegisters[static_cast<size_t>(Register::Type::Vector)], 1);
        EXPECT_EQ(peeked.highWaterMarkRegistersDelta[static_cast<size_t>(Register::Type::Vector)],
                  1);

        m_context->schedule(inst2);

        auto inst3 = Instruction("v_add_f32", {src3}, {src1, src2}, {}, "");
        peeked     = m_context->observer()->peek(inst1);
        EXPECT_EQ(peeked.allocatedRegisters[static_cast<size_t>(Register::Type::Vector)], 1);
        EXPECT_EQ(peeked.highWaterMarkRegistersDelta[static_cast<size_t>(Register::Type::Vector)],
                  1);

        m_context->schedule(inst3);

        auto inst_end = Instruction("s_endpgm", {}, {}, {}, "");
        peeked        = m_context->observer()->peek(inst_end);
        EXPECT_EQ(peeked.allocatedRegisters[static_cast<size_t>(Register::Type::Vector)], 0);
        EXPECT_EQ(peeked.highWaterMarkRegistersDelta[static_cast<size_t>(Register::Type::Vector)],
                  0);

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

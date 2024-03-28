#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/InstructionValues/Register.hpp>

#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"

using namespace rocRoller;
using ::testing::HasSubstr;

namespace rocRollerTest
{
    class Hazard90aObserverTest : public GenericContextFixture
    {
    protected:
        std::string targetArchitecture()
        {
            return "gfx90a";
        }

        void peekAndSchedule(Instruction& inst, uint expectedNops = 0)
        {
            auto peeked = m_context->observer()->peek(inst);
            EXPECT_EQ(peeked.nops, expectedNops);
            m_context->schedule(inst);
        }
    };

    class Hazard908ObserverTest : public GenericContextFixture
    {
    protected:
        std::string targetArchitecture()
        {
            return "gfx908";
        }

        void peekAndSchedule(Instruction inst, uint expectedNops = 0)
        {
            auto peeked = m_context->peek(inst);
            EXPECT_EQ(peeked.nops, expectedNops);
            m_context->schedule(inst);
        }
    };

    class Hazard942ObserverTest : public GenericContextFixture
    {
    protected:
        std::string targetArchitecture()
        {
            return "gfx942";
        }

        void peekAndSchedule(Instruction& inst, uint expectedNops = 0)
        {
            auto peeked = m_context->observer()->peek(inst);
            EXPECT_EQ(peeked.nops, expectedNops);
            m_context->schedule(inst);
        }
    };

    TEST_F(Hazard942ObserverTest, VALUWriteVCC)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::UInt32, 5);

        // This example can be found in basic I32 division
        {
            std::vector<Instruction> insts
                = {Instruction("v_cmp_ge_u32", {m_context->getVCC()}, {v[0], v[1]}, {}, ""),
                   Instruction("v_cndmask_b32", {v[0]}, {v[2], v[3], m_context->getVCC()}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 2);

            EXPECT_THAT(output(), HasSubstr("s_nop 1"));
            clearOutput();
        }

        // Try with instruction encoding `_e32`
        {
            std::vector<Instruction> insts = {
                Instruction("v_cmp_ge_u32_e32", {m_context->getVCC()}, {v[0], v[1]}, {}, ""),
                Instruction("v_cndmask_b32_e32", {v[0]}, {v[2], v[3], m_context->getVCC()}, {}, ""),
                Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 2);

            EXPECT_THAT(output(), HasSubstr("s_nop 1"));
            clearOutput();
        }

        // TODO Fix implicit VCC observer and enable this test
#if 0
        // Implicit VCC write
        {
            std::vector<Instruction> insts
                = {Instruction("v_add_co_u32", {v[3]}, {v[0], v[1]}, {}, ""),
                   Instruction("v_cndmask_b32", {v[4]}, {v[4], v[3], m_context->getVCC()}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 2);

            EXPECT_THAT(output(), HasSubstr("s_nop 1"));
            clearOutput();
        }
#endif
    }

    TEST_F(Hazard90aObserverTest, VALUWriteVCC)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::UInt32, 4);

        // No hazard
        {
            std::vector<Instruction> insts
                = {Instruction("v_cmp_ge_u32", {m_context->getVCC()}, {v[0], v[1]}, {}, ""),
                   Instruction("v_cndmask_b32", {v[0]}, {v[2], v[3], m_context->getVCC()}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1]);

            EXPECT_THAT(output(), Not(HasSubstr("s_nop")));
            clearOutput();
        }
    }

    TEST_F(Hazard942ObserverTest, VALUTrans)
    {
        auto v  = createRegisters(Register::Type::Vector, DataType::Float, 1);
        auto vu = createRegisters(Register::Type::Vector, DataType::UInt32, 2);

        {
            std::vector<Instruction> insts = {
                Instruction("v_rcp_iflag_f32_e32", {v[0]}, {v[0]}, {}, ""),
                Instruction(
                    "v_mul_f32_e32", {v[0]}, {Register::Value::Literal(0x4f7ffffe), v[0]}, {}, ""),
                Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 1);

            EXPECT_THAT(output(), HasSubstr("s_nop 0"));
            clearOutput();
        }

        // No hazard
        {
            std::vector<Instruction> insts = {
                Instruction("v_rcp_iflag_f32_e32", {v[0]}, {v[0]}, {}, ""),
                Instruction("v_add_u32_e32", {vu[0]}, {vu[0], vu[1]}, {}, ""),
                Instruction(
                    "v_mul_f32_e32", {v[0]}, {Register::Value::Literal(0x4f7ffffe), v[0]}, {}, ""),
                Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1]);
            peekAndSchedule(insts[2]);

            EXPECT_THAT(output(), Not(HasSubstr("s_nop")));
            clearOutput();
        }
    }

    TEST_F(Hazard942ObserverTest, VALUWriteReadlane)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::UInt32, 1);
        auto s = createRegisters(Register::Type::Scalar, DataType::UInt32, 2);

        {
            std::vector<Instruction> insts = {
                Instruction("v_subrev_u32", {v[0]}, {s[0], v[0]}, {}, ""),
                Instruction("v_readlane_b32", {s[1]}, {v[0], Register::Value::Literal(0)}, {}, ""),
                Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 1);

            EXPECT_THAT(output(), HasSubstr("s_nop 0"));
            clearOutput();
        }
    }

    TEST_F(Hazard942ObserverTest, VCMPX_EXEC)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::UInt32, 2);
        auto s = createRegisters(Register::Type::Scalar, DataType::UInt32, 2);

        {
            std::vector<Instruction> insts = {
                Instruction("v_cmpx_eq_u32", {}, {s[0], v[0]}, {}, ""),
                Instruction("v_readlane_b32", {s[1]}, {v[0], Register::Value::Literal(0)}, {}, ""),
                Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 2);

            EXPECT_THAT(output(), HasSubstr("s_nop 1"));
            clearOutput();
        }

        {
            std::vector<Instruction> insts
                = {Instruction("v_cmpx_eq_u32", {}, {s[0], v[0]}, {}, ""),
                   Instruction("v_xor_b32", {v[1]}, {v[0], m_context->getExec()}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 4);

            EXPECT_THAT(output(), HasSubstr("s_nop 3"));
            clearOutput();
        }

        {
            std::vector<Instruction> insts
                = {Instruction("v_cmpx_eq_u32", {}, {s[0], v[0]}, {}, ""),
                   Instruction("v_xor_b32", {v[1]}, {v[0], v[1]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1]);

            EXPECT_THAT(output(), Not(HasSubstr("s_nop")));
            clearOutput();
        }
    }

    TEST_F(Hazard942ObserverTest, OPSEL_SDWA)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::UInt32, 2);

        {
            std::vector<Instruction> insts
                = {Instruction("v_xor_b32_sdwa", {v[1]}, {v[0], v[0]}, {}, ""),
                   Instruction("v_mov_b32", {v[0]}, {v[1]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 1);

            EXPECT_THAT(output(), HasSubstr("s_nop 0"));
            clearOutput();
        }

        {
            std::vector<Instruction> insts
                = {Instruction("v_fma_f16", {v[1]}, {v[0], v[0], v[0]}, {"op_sel:[0,1,1]"}, ""),
                   Instruction("v_mov_b32", {v[0]}, {v[1]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 1);

            EXPECT_THAT(output(), HasSubstr("s_nop 0"));
            clearOutput();
        }
    }
}

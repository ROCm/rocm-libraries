#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/InstructionValues/Register.hpp>

#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"

using namespace rocRoller;
using ::testing::HasSubstr;

namespace rocRollerTest
{
    struct Gfx908
    {
    };
    struct Gfx90a
    {
    };
    struct Gfx940
    {
    };
    struct Gfx941
    {
    };
    struct Gfx942
    {
    };

    template <typename Arch>
    class HazardObserverTest : public GenericContextFixture
    {
    protected:
        GPUArchitectureTarget targetArchitecture() override
        {
            // clang-format off
            static_assert(
                std::is_same_v<Arch, Gfx908> ||
        std::is_same_v<Arch, Gfx90a> ||
        std::is_same_v<Arch, Gfx940> ||
        std::is_same_v<Arch, Gfx941> ||
        std::is_same_v<Arch, Gfx942>,
                "Unsupported Arch");
            // clang-format on

            if constexpr(std::is_same_v<Arch, Gfx908>)
                return {GPUArchitectureGFX::GFX908};
            else if constexpr(std::is_same_v<Arch, Gfx90a>)
                return {GPUArchitectureGFX::GFX90A};
            else if constexpr(std::is_same_v<Arch, Gfx940>)
                return {GPUArchitectureGFX::GFX940};
            else if constexpr(std::is_same_v<Arch, Gfx941>)
                return {GPUArchitectureGFX::GFX941};
            else
                return {GPUArchitectureGFX::GFX942};
        }

        void peekAndSchedule(Instruction& inst, uint expectedNops = 0)
        {
            auto peeked = m_context->observer()->peek(inst);
            EXPECT_EQ(peeked.nops, expectedNops);
            m_context->schedule(inst);
        }
    };

    using ArchTypes = ::testing::Types<Gfx908, Gfx90a, Gfx940, Gfx941, Gfx942>;
    TYPED_TEST_SUITE(HazardObserverTest, ArchTypes);

    TYPED_TEST(HazardObserverTest, VALUWriteVCC)
    {
        auto v = this->createRegisters(Register::Type::Vector, DataType::UInt32, 5);
        auto s = this->createRegisters(Register::Type::Scalar, DataType::UInt32, 2);

        // v_readlane (2nd op) read as laneselect
        {
            std::vector<Instruction> insts = {
                Instruction("v_readlane_b32", {s[0]}, {v[0], Register::Value::Literal(0)}, {}, ""),
                Instruction("v_readlane_b32", {s[1]}, {v[1], s[0]}, {}, ""),
                Instruction("s_endpgm", {}, {}, {}, "")};

            if constexpr(std::is_same_v<TypeParam, Gfx908> || std::is_same_v<TypeParam, Gfx90a>)
            {
                this->peekAndSchedule(insts[0]);
                this->peekAndSchedule(insts[1]);

                EXPECT_THAT(this->output(), Not(HasSubstr("s_nop")));
            }
            else
            {
                // NOPs are required on 94X arch
                this->peekAndSchedule(insts[0]);
                this->peekAndSchedule(insts[1], 4);

                EXPECT_THAT(this->output(), HasSubstr("s_nop 3"));
            }
            this->clearOutput();
        }

        // v_* (2nd op) read as constant
        {
            std::vector<Instruction> insts = {
                Instruction("v_readlane_b32", {s[0]}, {v[0], Register::Value::Literal(0)}, {}, ""),
                Instruction(
                    "v_cndmask_b32", {v[0]}, {s[0], v[3], this->m_context->getVCC()}, {}, ""),
                Instruction("s_endpgm", {}, {}, {}, "")};

            if constexpr(std::is_same_v<TypeParam, Gfx908> || std::is_same_v<TypeParam, Gfx90a>)
            {
                this->peekAndSchedule(insts[0]);
                this->peekAndSchedule(insts[1]);

                EXPECT_THAT(this->output(), Not(HasSubstr("s_nop")));
            }
            else
            {
                // NOPs are required on 94X arch
                this->peekAndSchedule(insts[0]);
                this->peekAndSchedule(insts[1], 2);

                EXPECT_THAT(this->output(), HasSubstr("s_nop 1"));
            }
            this->clearOutput();
        }

        // This example can be found in basic I32 division
        {
            std::vector<Instruction> insts
                = {Instruction("v_cmp_ge_u32", {this->m_context->getVCC()}, {v[0], v[1]}, {}, ""),
                   Instruction(
                       "v_cndmask_b32", {v[0]}, {v[2], v[3], this->m_context->getVCC()}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            if constexpr(std::is_same_v<TypeParam, Gfx908> || std::is_same_v<TypeParam, Gfx90a>)
            {
                this->peekAndSchedule(insts[0]);
                this->peekAndSchedule(insts[1]);

                EXPECT_THAT(this->output(), Not(HasSubstr("s_nop")));
            }
            else
            {
                // NOPs are required on 94X arch
                this->peekAndSchedule(insts[0]);
                this->peekAndSchedule(insts[1], 2);

                EXPECT_THAT(this->output(), HasSubstr("s_nop 1"));
            }
            this->clearOutput();
        }

        // Try with instruction encoding `_e32`
        {
            std::vector<Instruction> insts = {
                Instruction("v_cmp_ge_u32_e32", {this->m_context->getVCC()}, {v[0], v[1]}, {}, ""),
                Instruction(
                    "v_cndmask_b32_e32", {v[0]}, {v[2], v[3], this->m_context->getVCC()}, {}, ""),
                Instruction("s_endpgm", {}, {}, {}, "")};

            if constexpr(std::is_same_v<TypeParam, Gfx908> || std::is_same_v<TypeParam, Gfx90a>)
            {
                this->peekAndSchedule(insts[0]);
                this->peekAndSchedule(insts[1]);

                EXPECT_THAT(this->output(), Not(HasSubstr("s_nop")));
            }
            else
            {
                // NOPs are required on 94X arch
                this->peekAndSchedule(insts[0]);
                this->peekAndSchedule(insts[1], 2);

                EXPECT_THAT(this->output(), HasSubstr("s_nop 1"));
            }
            this->clearOutput();
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

    TYPED_TEST(HazardObserverTest, VALUTrans94X)
    {
        auto v  = this->createRegisters(Register::Type::Vector, DataType::Float, 2);
        auto vu = this->createRegisters(Register::Type::Vector, DataType::UInt32, 2);

        {
            std::vector<Instruction> insts = {
                Instruction("v_rcp_iflag_f32_e32", {v[0]}, {v[0]}, {}, ""),
                Instruction(
                    "v_mul_f32_e32", {v[0]}, {Register::Value::Literal(0x4f7ffffe), v[0]}, {}, ""),
                Instruction("s_endpgm", {}, {}, {}, "")};

            if constexpr(std::is_same_v<TypeParam, Gfx908> || std::is_same_v<TypeParam, Gfx90a>)
            {
                this->peekAndSchedule(insts[0]);
                this->peekAndSchedule(insts[1]);

                EXPECT_THAT(this->output(), Not(HasSubstr("s_nop")));
            }
            else
            {
                // NOPs are required on 94X arch
                this->peekAndSchedule(insts[0]);
                this->peekAndSchedule(insts[1], 1);

                EXPECT_THAT(this->output(), HasSubstr("s_nop 0"));
            }
            this->clearOutput();
        }

        // No hazard (2nd non-trans op accessing different VGPR)
        {
            std::vector<Instruction> insts = {
                Instruction("v_rcp_iflag_f32_e32", {v[0]}, {v[0]}, {}, ""),
                Instruction(
                    "v_mul_f32_e32", {v[1]}, {Register::Value::Literal(0x4f7ffffe), v[1]}, {}, ""),
                Instruction("s_endpgm", {}, {}, {}, "")};

            this->peekAndSchedule(insts[0]);
            this->peekAndSchedule(insts[1]);

            EXPECT_THAT(this->output(), Not(HasSubstr("s_nop")));
            this->clearOutput();
        }

        // No hazard (2nd op is a trans op)
        {
            std::vector<Instruction> insts
                = {Instruction("v_rcp_iflag_f32_e32", {v[0]}, {v[0]}, {}, ""),
                   Instruction("v_rcp_f32_e32", {v[0]}, {v[0]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            this->peekAndSchedule(insts[0]);
            this->peekAndSchedule(insts[1]);

            EXPECT_THAT(this->output(), Not(HasSubstr("s_nop")));
            this->clearOutput();
        }

        // No hazard
        {
            std::vector<Instruction> insts = {
                Instruction("v_rcp_iflag_f32_e32", {v[0]}, {v[0]}, {}, ""),
                Instruction("v_add_u32_e32", {vu[0]}, {vu[0], vu[1]}, {}, ""),
                Instruction(
                    "v_mul_f32_e32", {v[0]}, {Register::Value::Literal(0x4f7ffffe), v[0]}, {}, ""),
                Instruction("s_endpgm", {}, {}, {}, "")};

            this->peekAndSchedule(insts[0]);
            this->peekAndSchedule(insts[1]);
            this->peekAndSchedule(insts[2]);

            EXPECT_THAT(this->output(), Not(HasSubstr("s_nop")));
            this->clearOutput();
        }
    }

    TYPED_TEST(HazardObserverTest, VALUWriteReadlane94X)
    {
        auto v = this->createRegisters(Register::Type::Vector, DataType::UInt32, 2);
        auto s = this->createRegisters(Register::Type::Scalar, DataType::UInt32, 2);

        {
            std::vector<Instruction> insts = {
                Instruction("v_subrev_u32", {v[0]}, {s[0], v[0]}, {}, ""),
                Instruction("v_readlane_b32", {s[1]}, {v[0], Register::Value::Literal(0)}, {}, ""),
                Instruction("s_endpgm", {}, {}, {}, "")};

            if constexpr(std::is_same_v<TypeParam, Gfx908> || std::is_same_v<TypeParam, Gfx90a>)
            {
                this->peekAndSchedule(insts[0]);
                this->peekAndSchedule(insts[1]);

                EXPECT_THAT(this->output(), Not(HasSubstr("s_nop")));
            }
            else
            {
                // NOPs are required on 94X arch
                this->peekAndSchedule(insts[0]);
                this->peekAndSchedule(insts[1], 1);

                EXPECT_THAT(this->output(), HasSubstr("s_nop 0"));
            }
            this->clearOutput();
        }

        // No hazard (accessing different VGPRs)
        {
            std::vector<Instruction> insts = {
                Instruction("v_subrev_u32", {v[0]}, {s[0], v[0]}, {}, ""),
                Instruction("v_readlane_b32", {s[1]}, {v[1], Register::Value::Literal(0)}, {}, ""),
                Instruction("s_endpgm", {}, {}, {}, "")};

            this->peekAndSchedule(insts[0]);
            this->peekAndSchedule(insts[1]);

            EXPECT_THAT(this->output(), Not(HasSubstr("s_nop")));
            this->clearOutput();
        }
    }

    TYPED_TEST(HazardObserverTest, VCMPX_EXEC94X)
    {
        auto v = this->createRegisters(Register::Type::Vector, DataType::UInt32, 2);
        auto s = this->createRegisters(Register::Type::Scalar, DataType::UInt32, 2);

        {
            std::vector<Instruction> insts = {
                Instruction("v_cmpx_eq_u32", {}, {s[0], v[0]}, {}, ""),
                Instruction("v_readlane_b32", {s[1]}, {v[0], Register::Value::Literal(0)}, {}, ""),
                Instruction("s_endpgm", {}, {}, {}, "")};

            if constexpr(std::is_same_v<TypeParam, Gfx908> || std::is_same_v<TypeParam, Gfx90a>)
            {
                this->peekAndSchedule(insts[0]);
                this->peekAndSchedule(insts[1]);

                EXPECT_THAT(this->output(), Not(HasSubstr("s_nop")));
            }
            else
            {
                // NOPs are required on 94X arch
                this->peekAndSchedule(insts[0]);
                this->peekAndSchedule(insts[1], 4);

                EXPECT_THAT(this->output(), HasSubstr("s_nop 3"));
            }
            this->clearOutput();
        }

        {
            std::vector<Instruction> insts
                = {Instruction("v_cmpx_eq_u32", {}, {s[0], v[0]}, {}, ""),
                   Instruction("v_xor_b32", {v[1]}, {v[0], this->m_context->getExec()}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            if constexpr(std::is_same_v<TypeParam, Gfx908> || std::is_same_v<TypeParam, Gfx90a>)
            {
                this->peekAndSchedule(insts[0]);
                this->peekAndSchedule(insts[1]);

                EXPECT_THAT(this->output(), Not(HasSubstr("s_nop")));
            }
            else
            {
                // NOPs are required on 94X arch
                this->peekAndSchedule(insts[0]);
                this->peekAndSchedule(insts[1], 2);

                EXPECT_THAT(this->output(), HasSubstr("s_nop 1"));
            }
            this->clearOutput();
        }

        {
            std::vector<Instruction> insts
                = {Instruction("v_cmpx_eq_u32", {}, {s[0], v[0]}, {}, ""),
                   Instruction("v_xor_b32", {v[1]}, {v[0], v[1]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            this->peekAndSchedule(insts[0]);
            this->peekAndSchedule(insts[1]);

            EXPECT_THAT(this->output(), Not(HasSubstr("s_nop")));
            this->clearOutput();
        }
    }

    TYPED_TEST(HazardObserverTest, OPSELxSDWA94X)
    {
        auto v = this->createRegisters(Register::Type::Vector, DataType::UInt32, 2);

        {
            // SDWA
            std::vector<Instruction> insts
                = {Instruction("v_xor_b32_sdwa", {v[1]}, {v[0], v[0]}, {}, ""),
                   Instruction("v_mov_b32", {v[0]}, {v[1]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            if constexpr(std::is_same_v<TypeParam, Gfx908> || std::is_same_v<TypeParam, Gfx90a>)
            {
                this->peekAndSchedule(insts[0]);
                this->peekAndSchedule(insts[1]);

                EXPECT_THAT(this->output(), Not(HasSubstr("s_nop")));
            }
            else
            {
                // NOPs are required on 94X arch
                this->peekAndSchedule(insts[0]);
                this->peekAndSchedule(insts[1], 1);

                EXPECT_THAT(this->output(), HasSubstr("s_nop 0"));
            }
            this->clearOutput();
        }

        {
            // OPSEL
            std::vector<Instruction> insts
                = {Instruction("v_fma_f16", {v[1]}, {v[0], v[0], v[0]}, {"op_sel:[0,1,1]"}, ""),
                   Instruction("v_mov_b32", {v[0]}, {v[1]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            if constexpr(std::is_same_v<TypeParam, Gfx908> || std::is_same_v<TypeParam, Gfx90a>)
            {
                this->peekAndSchedule(insts[0]);
                this->peekAndSchedule(insts[1]);

                EXPECT_THAT(this->output(), Not(HasSubstr("s_nop")));
            }
            else
            {
                // NOPs are required on 94X arch
                this->peekAndSchedule(insts[0]);
                this->peekAndSchedule(insts[1], 1);

                EXPECT_THAT(this->output(), HasSubstr("s_nop 0"));
            }
            this->clearOutput();
        }
    }

    TYPED_TEST(HazardObserverTest, VALUWriteSGPRVMEM)
    {
        auto v = this->createRegisters(Register::Type::Vector, DataType::UInt32, 2);
        auto s = this->createRegisters(Register::Type::Scalar, DataType::UInt32, 1);
        {
            std::vector<Instruction> insts = {
                Instruction("v_readlane_b32", {s[0]}, {v[0], Register::Value::Literal(0)}, {}, ""),
                Instruction(
                    "buffer_load_dword", {v[1]}, {s[0], Register::Value::Literal(0)}, {}, ""),
                Instruction("s_endpgm", {}, {}, {}, "")};
            this->peekAndSchedule(insts[0]);
            this->peekAndSchedule(insts[1], 5);
            EXPECT_THAT(this->output(), HasSubstr("s_nop 4"));
            this->clearOutput();
        }
    }
}

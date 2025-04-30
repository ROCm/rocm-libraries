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

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/InstructionValues/Register.hpp>

#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"

using namespace rocRoller;
using ::testing::HasSubstr;

namespace rocRollerTest
{
    class MFMA90aObserverTest : public GenericContextFixture
    {
    protected:
        GPUArchitectureTarget targetArchitecture() override
        {
            return {GPUArchitectureGFX::GFX90A};
        }

        void peekAndSchedule(Instruction& inst, uint expectedNops = 0)
        {
            auto peeked = m_context->observer()->peek(inst);
            EXPECT_EQ(peeked.nops, expectedNops) << inst.toString(LogLevel::Debug);
            m_context->schedule(inst);
        }
    };

    class MFMA908ObserverTest : public GenericContextFixture
    {
    protected:
        GPUArchitectureTarget targetArchitecture() override
        {
            return {GPUArchitectureGFX::GFX908};
        }

        void peekAndSchedule(Instruction inst, uint expectedNops = 0)
        {
            auto peeked = m_context->peek(inst);
            EXPECT_EQ(peeked.nops, expectedNops) << inst.toString(LogLevel::Debug);
            m_context->schedule(inst);
        }
    };

    class MFMA942ObserverTest : public GenericContextFixture
    {
    protected:
        GPUArchitectureTarget targetArchitecture() override
        {
            return {GPUArchitectureGFX::GFX942};
        }

        void peekAndSchedule(Instruction& inst, uint expectedNops = 0)
        {
            auto peeked = m_context->observer()->peek(inst);
            EXPECT_EQ(peeked.nops, expectedNops) << inst.toString(LogLevel::Debug);
            m_context->schedule(inst);
        }
    };

    class MFMA950ObserverTest : public GenericContextFixture
    {
    protected:
        GPUArchitectureTarget targetArchitecture()
        {
            return GPUArchitectureTarget{GPUArchitectureGFX::GFX950};
        }

        void peekAndSchedule(Instruction& inst, uint expectedNops = 0)
        {
            auto peeked = m_context->observer()->peek(inst);
            EXPECT_EQ(peeked.nops, expectedNops) << inst.toString(LogLevel::Debug);
            m_context->schedule(inst);
        }
    };

    TEST_F(MFMA90aObserverTest, NoWaitStates)
    {
        auto s = createRegisters(Register::Type::Scalar,
                                 DataType::Float,
                                 3,
                                 2,
                                 Register::AllocationOptions::FullyContiguous());

        auto zero = Register::Value::Literal(0);

        {
            std::vector<Instruction> insts
                = {Instruction("s_load_dwordx2", {s[1]}, {s[0], zero}, {}, ""),
                   Instruction("s_load_dwordx2", {s[2]}, {s[0], zero}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            for(auto inst : insts)
            {
                peekAndSchedule(inst);
            }

            std::string expected = R"(
                                        s_load_dwordx2 s[2:3], s[0:1], 0
                                        s_load_dwordx2 s[4:5], s[0:1], 0
                                        s_endpgm
                                    )";

            EXPECT_EQ(NormalizedSource(output()), NormalizedSource(expected));
        }
    }

    TEST_F(MFMA90aObserverTest, CMPXWaitStates)
    {
        auto v_i64 = createRegisters(Register::Type::Vector, DataType::Int64, 1);
        auto v_f32 = createRegisters(Register::Type::Vector, DataType::Float, 3);
        auto v_f16 = createRegisters(Register::Type::Vector, DataType::Half, 2);
        auto a     = createRegisters(Register::Type::Accumulator, DataType::Float, 1);

        {
            std::vector<Instruction> insts
                = {Instruction("v_cmpx_lt_f32", {v_f32[2]}, {v_f32[0], v_f32[1]}, {}, ""),
                   Instruction("v_mfma_f32_32x32x8f16", {a[0]}, {v_f16[0], v_f16[1], a[0]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 4);

            EXPECT_THAT(output(), HasSubstr("s_nop 3"));
            clearOutput();
        }

        // No hazard if v_cmp doesn't write to exec or a VGPR that is later read
        {
            std::vector<Instruction> insts
                = {Instruction("v_cmp_lt_f32", {v_i64[0]}, {v_f32[0], v_f32[1]}, {}, ""),
                   Instruction("v_mfma_f32_32x32x8f16", {a[0]}, {v_f16[0], v_f16[1], a[0]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1]);

            EXPECT_THAT(output(), Not(HasSubstr("s_nop")));
            clearOutput();
        }
    }

    TEST_F(MFMA90aObserverTest, VALUThenMFMA)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 6);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 1);

        {
            std::vector<Instruction> insts
                = {Instruction("v_or_b32", {v[2]}, {v[0], v[1]}, {}, ""),
                   Instruction("v_mfma_f32_16x16x4f32", {a[0]}, {v[0], v[2], a[0]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 2);

            EXPECT_THAT(output(), HasSubstr("s_nop 1"));
            clearOutput();
        }

        // No hazard if second instruction doesn't read same VGPRs
        {
            std::vector<Instruction> insts
                = {Instruction("v_or_b32", {v[2]}, {v[0], v[1]}, {}, ""),
                   Instruction("v_mfma_f32_16x16x4f32", {a[0]}, {v[3], v[4], a[0]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 0);

            EXPECT_THAT(output(), Not(HasSubstr("s_nop")));
            clearOutput();
        }

        {
            std::vector<Instruction> insts
                = {Instruction("v_or_b32", {v[2]}, {v[0], v[1]}, {}, ""),
                   Instruction("v_or_b32", {v[5]}, {v[3], v[4]}, {}, ""), // Unrelated
                   Instruction("v_mfma_f32_16x16x4f32", {a[0]}, {v[0], v[2], a[0]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1]);
            peekAndSchedule(insts[2], 1);

            EXPECT_THAT(output(), HasSubstr("s_nop 0"));
            clearOutput();
        }

        {
            std::vector<Instruction> insts
                = {Instruction("v_or_b32", {v[2]}, {v[0], v[1]}, {}, ""),
                   Instruction("v_or_b32", {v[5]}, {v[3], v[4]}, {}, ""), // Unrelated
                   Instruction("v_mfma_f32_16x16x4f32", {a[0]}, {v[2], v[5], a[0]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1]);
            peekAndSchedule(insts[2], 2);

            EXPECT_THAT(output(), HasSubstr("s_nop 1"));
            clearOutput();
        }
    }

    TEST_F(MFMA90aObserverTest, XDLOPThenMFMA_SrcCExact)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 4);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 1, 4);

        // No hazard if SrcC is exactly overlapped
        {
            std::vector<Instruction> insts
                = {Instruction("v_mfma_f32_16x16x4f32", {a[0]}, {v[0], v[1], a[0]}, {}, ""),
                   Instruction("v_mfma_f32_16x16x4f32", {a[0]}, {v[2], v[3], a[0]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1]);

            EXPECT_THAT(output(), Not(HasSubstr("s_nop")));
            clearOutput();
        }
    }

    TEST_F(MFMA90aObserverTest, XDLOPThenMFMA_Different)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 4);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 1, 4);

        // No register overlap should not be a hazard
        {
            std::vector<Instruction> insts = {Instruction("v_mfma_f32_16x16x4f32",
                                                          {a[0]->subset({0, 1})},
                                                          {v[0], v[1], a[0]->subset({0, 1})},
                                                          {},
                                                          ""),
                                              Instruction("v_mfma_f32_16x16x4f32",
                                                          {a[0]->subset({2, 3})},
                                                          {v[2], v[3], a[0]->subset({2, 3})},
                                                          {},
                                                          ""),
                                              Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1]);

            EXPECT_THAT(output(), Not(HasSubstr("s_nop")));
            clearOutput();
        }
    }

    TEST_F(MFMA90aObserverTest, XDLOPThenMFMA_SrcCOverlap)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 4);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 1, 4);

        // If SrcC is partially overlapped
        {
            std::vector<Instruction> insts = {Instruction("v_mfma_f32_16x16x4f32",
                                                          {a[0]->subset({0, 1})},
                                                          {v[0], v[1], a[0]->subset({0, 1})},
                                                          {},
                                                          ""),
                                              Instruction("v_mfma_f32_16x16x4f32",
                                                          {a[0]->subset({1, 2})},
                                                          {v[2], v[3], a[0]->subset({1, 2})},
                                                          {},
                                                          ""),
                                              Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 8);

            EXPECT_THAT(output(), HasSubstr("s_nop 7"));
            clearOutput();
        }
    }

    TEST_F(MFMA90aObserverTest, XDLOPThenMFMA_ReadA)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 5);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 2, 4);

        {
            std::vector<Instruction> insts
                = {Instruction("v_mfma_f32_16x16x4f32", {v[2]}, {v[0], v[1], a[0]}, {}, ""),
                   Instruction("v_mfma_f32_16x16x4f32", {a[1]}, {v[2], v[3], a[1]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 11);

            EXPECT_THAT(output(), HasSubstr("s_nop 10"));
            clearOutput();
        }
    }

    TEST_F(MFMA90aObserverTest, XDLOPThenFlat)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 5);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 2, 4);

        {
            std::vector<Instruction> insts
                = {Instruction("v_mfma_f32_16x16x4f32", {v[2]}, {v[0], v[1], a[0]}, {}, ""),
                   Instruction("flat_store_dword", {}, {v[2], v[3]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 11);

            EXPECT_THAT(output(), HasSubstr("s_nop 10"));
            clearOutput();
        }
    }

    TEST_F(MFMA90aObserverTest, XDLOPThenVALU_Read)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 5);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 2, 4);

        {
            std::vector<Instruction> insts
                = {Instruction("v_mfma_f32_16x16x4f32", {v[2]}, {v[0], v[1], a[0]}, {}, ""),
                   Instruction("v_or_b32", {v[4]}, {v[2], v[3]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 11);

            EXPECT_THAT(output(), HasSubstr("s_nop 10"));
            clearOutput();
        }
    }

    TEST_F(MFMA90aObserverTest, XDLOPThenVALU_Write)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 5);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 2, 4);

        {
            std::vector<Instruction> insts
                = {Instruction("v_mfma_f32_16x16x4f32", {v[2]}, {v[0], v[1], a[0]}, {}, ""),
                   Instruction("v_or_b32", {v[2]}, {v[3], v[4]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 11);

            EXPECT_THAT(output(), HasSubstr("s_nop 10"));
            clearOutput();
        }
    }

    TEST_F(MFMA90aObserverTest, XDLOPThenVALU_WAR)
    {
        auto                     v = createRegisters(Register::Type::Vector, DataType::Float, 6);
        std::vector<Instruction> insts
            = {Instruction("v_mfma_f32_16x16x4f32", {v[2]}, {v[0], v[1], v[3]}, {}, ""),
               Instruction("v_or_b32", {v[3]}, {v[4], v[5]}, {}, ""),
               Instruction("s_endpgm", {}, {}, {}, "")};

        peekAndSchedule(insts[0]);
        peekAndSchedule(insts[1], 11);

        EXPECT_THAT(output(), HasSubstr("s_nop 10"));
    }

    TEST_F(MFMA90aObserverTest, XDLOPThenVALU_WAR_Partial)
    {
        auto                     v = createRegisters(Register::Type::Vector, DataType::Float, 6, 4);
        std::vector<Instruction> insts
            = {Instruction(
                   "v_mfma_f32_16x16x4f32", {v[2]}, {v[0], v[1], v[3]->subset({2, 3})}, {}, ""),
               Instruction("v_or_b32", {v[3]->subset({1, 2})}, {v[4], v[5]}, {}, ""),
               Instruction("s_endpgm", {}, {}, {}, "")};

        peekAndSchedule(insts[0]);
        peekAndSchedule(insts[1], 11);

        EXPECT_THAT(output(), HasSubstr("s_nop 10"));
        clearOutput();
    }

    TEST_F(MFMA90aObserverTest, DLOPWriteScrC)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Half, 7);

        {
            std::vector<Instruction> insts
                = {Instruction("v_dot2_f32_f16", {v[3]}, {v[0], v[1], v[2]}, {}, ""),
                   Instruction("v_dot2_f32_f16", {v[6]}, {v[4], v[5], v[3]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1]);

            EXPECT_THAT(output(), Not(HasSubstr("s_nop")));
            clearOutput();
        }
    }

    TEST_F(MFMA90aObserverTest, DLOPWriteSrcA)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Half, 7);

        {
            std::vector<Instruction> insts
                = {Instruction("v_dot2_f32_f16", {v[3]}, {v[0], v[1], v[2]}, {}, ""),
                   Instruction("v_dot2_f32_f16", {v[6]}, {v[3], v[4], v[5]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 3);

            EXPECT_THAT(output(), HasSubstr("s_nop 2"));
            clearOutput();
        }
    }

    TEST_F(MFMA90aObserverTest, DLOPWriteSrcB)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Half, 7);

        {
            std::vector<Instruction> insts
                = {Instruction("v_dot2_f32_f16", {v[3]}, {v[0], v[1], v[2]}, {}, ""),
                   Instruction("v_dot2_f32_f16", {v[6]}, {v[4], v[3], v[5]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 3);

            EXPECT_THAT(output(), HasSubstr("s_nop 2"));
            clearOutput();
        }
    }

    TEST_F(MFMA90aObserverTest, DLOPWriteDifferent)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Half, 7);

        {
            std::vector<Instruction> insts
                = {Instruction("v_dot2_f32_f16", {v[3]}, {v[0], v[1], v[2]}, {}, ""),
                   Instruction("v_dot2c_f32_f16", {v[6]}, {v[0], v[1], v[3]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 3);

            EXPECT_THAT(output(), HasSubstr("s_nop 2"));
            clearOutput();
        }
    }

    TEST_F(MFMA90aObserverTest, DGEMM16x16x4ThenMFMA_SrcCExact)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 4);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 1, 4);

        // No hazard if SrcC is exactly overlapped
        {
            std::vector<Instruction> insts
                = {Instruction("v_mfma_f64_16x16x4f64", {a[0]}, {v[0], v[1], a[0]}, {}, ""),
                   Instruction("v_mfma_f64_16x16x4f64", {a[0]}, {v[2], v[3], a[0]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1]);

            EXPECT_THAT(output(), Not(HasSubstr("s_nop")));
            clearOutput();
        }
    }

    TEST_F(MFMA90aObserverTest, DGEMM16x16x4ThenDGEMM_Different)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 4);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 1, 4);

        // No register overlap should not be a hazard
        {
            std::vector<Instruction> insts = {Instruction("v_mfma_f64_16x16x4f64",
                                                          {a[0]->subset({0, 1})},
                                                          {v[0], v[1], a[0]->subset({0, 1})},
                                                          {},
                                                          ""),
                                              Instruction("v_mfma_f64_16x16x4f64",
                                                          {a[0]->subset({2, 3})},
                                                          {v[2], v[3], a[0]->subset({2, 3})},
                                                          {},
                                                          ""),
                                              Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1]);

            EXPECT_THAT(output(), Not(HasSubstr("s_nop")));
            clearOutput();
        }
    }

    TEST_F(MFMA90aObserverTest, DGEMM16x16x4ThenDGEMM_SrcCOverlap)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 4);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 1, 4);

        // If SrcC is partially overlapped
        {
            std::vector<Instruction> insts = {Instruction("v_mfma_f64_16x16x4f64",
                                                          {a[0]->subset({0, 1})},
                                                          {v[0], v[1], a[0]->subset({0, 1})},
                                                          {},
                                                          ""),
                                              Instruction("v_mfma_f64_16x16x4f64",
                                                          {a[0]->subset({1, 2})},
                                                          {v[2], v[3], a[0]->subset({1, 2})},
                                                          {},
                                                          ""),
                                              Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 9);

            EXPECT_THAT(output(), HasSubstr("s_nop 8"));
            clearOutput();
        }
    }

    TEST_F(MFMA90aObserverTest, DGEMM16x16x4ThenMFMA_ReadA)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 5);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 2, 4);

        {
            std::vector<Instruction> insts
                = {Instruction("v_mfma_f64_16x16x4f64", {v[2]}, {v[0], v[1], a[0]}, {}, ""),
                   Instruction("v_mfma_f64_16x16x4f64", {a[1]}, {v[2], v[3], a[1]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 11);

            EXPECT_THAT(output(), HasSubstr("s_nop 10"));
            clearOutput();
        }
    }

    TEST_F(MFMA90aObserverTest, DGEMM16x16x4ThenACCRead)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 5);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 2, 4);

        {
            std::vector<Instruction> insts
                = {Instruction("v_mfma_f64_16x16x4f64", {a[1]}, {v[0], v[1], a[0]}, {}, ""),
                   Instruction::Comment("Comments shouldn't change NOP counts"),
                   Instruction("v_accvgpr_read", {v[2]}, {a[1]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1]);
            peekAndSchedule(insts[2], 11);

            EXPECT_THAT(output(), HasSubstr("s_nop 10"));
            clearOutput();
        }
    }

    TEST_F(MFMA90aObserverTest, DGEMM16x16x4ThenFlat)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 5);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 2, 4);

        {
            std::vector<Instruction> insts
                = {Instruction("v_mfma_f64_16x16x4f64", {v[2]}, {v[0], v[1], a[0]}, {}, ""),
                   Instruction("flat_store_dword", {}, {v[2], v[3]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 18);

            clearOutput();
        }
    }

    TEST_F(MFMA90aObserverTest, DGEMM16x16x4ThenVALU_Read)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 5);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 2, 4);

        {
            std::vector<Instruction> insts
                = {Instruction("v_mfma_f64_16x16x4f64", {v[2]}, {v[0], v[1], a[0]}, {}, ""),
                   Instruction("v_or_b32", {v[4]}, {v[2], v[3]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 11);

            EXPECT_THAT(output(), HasSubstr("s_nop 10"));
            clearOutput();
        }
    }

    TEST_F(MFMA90aObserverTest, DGEMM16x16x4ThenVALU_Write)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 5);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 2, 4);

        {
            std::vector<Instruction> insts
                = {Instruction("v_mfma_f64_16x16x4f64", {v[2]}, {v[0], v[1], a[0]}, {}, ""),
                   Instruction("v_or_b32", {v[2]}, {v[3], v[4]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 11);

            EXPECT_THAT(output(), HasSubstr("s_nop 10"));
            clearOutput();
        }
    }

    TEST_F(MFMA908ObserverTest, CMPXWaitStates)
    {
        auto v_i64 = createRegisters(Register::Type::Vector, DataType::Int64, 1);
        auto v_f32 = createRegisters(Register::Type::Vector, DataType::Float, 3);
        auto v_f16 = createRegisters(Register::Type::Vector, DataType::Half, 2);
        auto a     = createRegisters(Register::Type::Accumulator, DataType::Float, 1);

        {
            std::vector<Instruction> insts
                = {Instruction("v_cmpx_lt_f32", {v_f32[2]}, {v_f32[0], v_f32[1]}, {}, ""),
                   Instruction("v_mfma_f32_32x32x8f16", {a[0]}, {v_f16[0], v_f16[1], a[0]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 4);

            EXPECT_THAT(output(), HasSubstr("s_nop 3"));
            clearOutput();
        }

        // No hazard if v_cmp doesn't write to exec or a VGPR that is later read
        {
            std::vector<Instruction> insts
                = {Instruction("v_cmp_lt_f32", {v_i64[0]}, {v_f32[0], v_f32[1]}, {}, ""),
                   Instruction("v_mfma_f32_32x32x8f16", {a[0]}, {v_f16[0], v_f16[1], a[0]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1]);

            EXPECT_THAT(output(), Not(HasSubstr("s_nop")));
            clearOutput();
        }
    }

    TEST_F(MFMA908ObserverTest, CMPXThenACCVGPRWrite)
    {
        auto v_f32 = createRegisters(Register::Type::Vector, DataType::Float, 3);
        auto v_f16 = createRegisters(Register::Type::Vector, DataType::Half, 2);
        auto a     = createRegisters(Register::Type::Accumulator, DataType::Float, 1);
        {
            std::vector<Instruction> insts
                = {Instruction("v_cmpx_lt_f32", {v_f32[2]}, {v_f32[0], v_f32[1]}, {}, ""),
                   Instruction("v_accvgpr_write", {a[0]}, {v_f16[0]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 4);

            EXPECT_THAT(output(), HasSubstr("s_nop 3"));
            clearOutput();
        }
    }

    TEST_F(MFMA908ObserverTest, VALUThenMFMA)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 6);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 1);

        {
            std::vector<Instruction> insts
                = {Instruction("v_or_b32", {v[2]}, {v[0], v[1]}, {}, ""),
                   Instruction("v_mfma_f32_16x16x4f32", {a[0]}, {v[0], v[2], a[0]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 2);

            EXPECT_THAT(output(), HasSubstr("s_nop 1"));
            clearOutput();
        }

        // No hazard if second instruction doesn't read same VGPRs
        {
            std::vector<Instruction> insts
                = {Instruction("v_or_b32", {v[2]}, {v[0], v[1]}, {}, ""),
                   Instruction("v_mfma_f32_16x16x4f32", {a[0]}, {v[3], v[4], a[0]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 0);

            EXPECT_THAT(output(), Not(HasSubstr("s_nop")));
            clearOutput();
        }

        {
            std::vector<Instruction> insts
                = {Instruction("v_or_b32", {v[2]}, {v[0], v[1]}, {}, ""),
                   Instruction("v_or_b32", {v[5]}, {v[3], v[4]}, {}, ""), // Unrelated
                   Instruction("v_mfma_f32_16x16x4f32", {a[0]}, {v[0], v[2], a[0]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1]);
            peekAndSchedule(insts[2], 1);

            EXPECT_THAT(output(), HasSubstr("s_nop 0"));
            clearOutput();
        }
    }

    TEST_F(MFMA908ObserverTest, VALUThenACCVGPRWrite)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 6);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 1);

        {
            std::vector<Instruction> insts
                = {Instruction("v_or_b32", {v[2]}, {v[0], v[1]}, {}, ""),
                   Instruction("v_accvgpr_write", {a[0]}, {v[2]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 2);

            EXPECT_THAT(output(), HasSubstr("s_nop 1"));
            clearOutput();
        }
    }

    TEST_F(MFMA942ObserverTest, XDLOPThenMFMA_SrcCExact)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 4);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 1, 1);

        // No hazard if SrcC is exactly overlapped
        {
            std::vector<Instruction> insts
                = {Instruction("v_mfma_f32_16x16x4_f32", {a[0]}, {v[0], v[1], a[0]}, {}, ""),
                   Instruction("v_mfma_f32_16x16x4_f32", {a[0]}, {v[2], v[3], a[0]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1]);

            EXPECT_THAT(output(), Not(HasSubstr("s_nop")));
            clearOutput();
        }
    }

    TEST_F(MFMA942ObserverTest, XDLOPThenMFMA_SrcCExact2Pass)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 4);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 1, 1);
        // Exception: Hazard for 2 pass MFMAs
        {
            std::vector<Instruction> insts
                = {Instruction("v_mfma_f32_4x4x1f32", {a[0]}, {v[0], v[1], a[0]}, {}, ""),
                   Instruction("v_mfma_f32_4x4x1f32", {a[0]}, {v[2], v[3], a[0]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 2);

            EXPECT_THAT(output(), HasSubstr("s_nop 1"));
            clearOutput();
        }
    }

    TEST_F(MFMA942ObserverTest, XDLOPThenMFMA_Different)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 4);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 1, 4);

        // No register overlap should not be a hazard
        {
            std::vector<Instruction> insts = {Instruction("v_mfma_f32_16x16x4_f32",
                                                          {a[0]->subset({0, 1})},
                                                          {v[0], v[1], a[0]->subset({0, 1})},
                                                          {},
                                                          ""),
                                              Instruction("v_mfma_f32_16x16x4_f32",
                                                          {a[0]->subset({2, 3})},
                                                          {v[2], v[3], a[0]->subset({2, 3})},
                                                          {},
                                                          ""),
                                              Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1]);

            EXPECT_THAT(output(), Not(HasSubstr("s_nop")));
            clearOutput();
        }
    }

    TEST_F(MFMA942ObserverTest, XDLOPThenMFMA_SrcCOverlap)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 4);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 1, 4);

        // If SrcC is partially overlapped
        {
            std::vector<Instruction> insts = {Instruction("v_mfma_f32_16x16x4_f32",
                                                          {a[0]->subset({0, 1})},
                                                          {v[0], v[1], a[0]->subset({0, 1})},
                                                          {},
                                                          ""),
                                              Instruction("v_mfma_f32_16x16x4_f32",
                                                          {a[0]->subset({1, 2})},
                                                          {v[2], v[3], a[0]->subset({1, 2})},
                                                          {},
                                                          ""),
                                              Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 9);

            EXPECT_THAT(output(), HasSubstr("s_nop 8"));
            clearOutput();
        }
    }

    TEST_F(MFMA942ObserverTest, XDLOPThenMFMA_ReadA)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 5);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 2, 4);

        {
            std::vector<Instruction> insts
                = {Instruction("v_mfma_f32_16x16x4_f32", {v[2]}, {v[0], v[1], a[0]}, {}, ""),
                   Instruction("v_mfma_f32_16x16x4_f32", {a[1]}, {v[2], v[3], a[1]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 11);

            EXPECT_THAT(output(), HasSubstr("s_nop 10"));
            clearOutput();
        }
    }

    TEST_F(MFMA942ObserverTest, XDLOPThenFlat)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 5);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 2, 4);

        {
            std::vector<Instruction> insts
                = {Instruction("v_mfma_f32_16x16x4_f32", {v[2]}, {v[0], v[1], a[0]}, {}, ""),
                   Instruction("flat_store_dword", {}, {v[2], v[3]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 11);

            EXPECT_THAT(output(), HasSubstr("s_nop 10"));
            clearOutput();
        }
    }

    TEST_F(MFMA942ObserverTest, XDLOPThenVALU_Read)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 5);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 2, 4);

        {
            std::vector<Instruction> insts
                = {Instruction("v_mfma_f32_16x16x4_f32", {v[2]}, {v[0], v[1], a[0]}, {}, ""),
                   Instruction("v_or_b32", {v[4]}, {v[2], v[3]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 11);

            EXPECT_THAT(output(), HasSubstr("s_nop 10"));
            clearOutput();
        }
    }

    TEST_F(MFMA942ObserverTest, XDLOPThenVALU_Write)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 5);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 2, 4);

        {
            std::vector<Instruction> insts
                = {Instruction("v_mfma_f32_16x16x4_f32", {v[2]}, {v[0], v[1], a[0]}, {}, ""),
                   Instruction("v_or_b32", {v[2]}, {v[3], v[4]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 11);

            EXPECT_THAT(output(), HasSubstr("s_nop 10"));
            clearOutput();
        }
    }

    TEST_F(MFMA942ObserverTest, XDLOPThenVALU_WAR)
    {
        auto                     v = createRegisters(Register::Type::Vector, DataType::Float, 6);
        std::vector<Instruction> insts
            = {Instruction("v_mfma_f32_16x16x4_f32", {v[2]}, {v[0], v[1], v[3]}, {}, ""),
               Instruction("v_or_b32", {v[3]}, {v[4], v[5]}, {}, ""),
               Instruction("s_endpgm", {}, {}, {}, "")};

        peekAndSchedule(insts[0]);
        peekAndSchedule(insts[1], 7);

        EXPECT_THAT(output(), HasSubstr("s_nop 6"));
    }

    TEST_F(MFMA942ObserverTest, XDLOPThenVALU_WAR_Partial)
    {
        auto                     v = createRegisters(Register::Type::Vector, DataType::Float, 6, 4);
        std::vector<Instruction> insts
            = {Instruction(
                   "v_mfma_f32_16x16x4_f32", {v[2]}, {v[0], v[1], v[3]->subset({2, 3})}, {}, ""),
               Instruction("v_or_b32", {v[3]->subset({1, 2})}, {v[4], v[5]}, {}, ""),
               Instruction("s_endpgm", {}, {}, {}, "")};

        peekAndSchedule(insts[0]);
        peekAndSchedule(insts[1], 7);

        EXPECT_THAT(output(), HasSubstr("s_nop 6"));
        clearOutput();
    }

    TEST_F(MFMA950ObserverTest, XDLOPThenMFMA_SrcCOverlap)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 4);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 1, 4);

        // If SrcC is partially overlapped
        {
            std::vector<Instruction> insts = {Instruction("v_mfma_f32_16x16x4_f32",
                                                          {a[0]->subset({0, 1})},
                                                          {v[0], v[1], a[0]->subset({0, 1})},
                                                          {},
                                                          ""),
                                              Instruction("v_mfma_f32_16x16x4_f32",
                                                          {a[0]->subset({1, 2})},
                                                          {v[2], v[3], a[0]->subset({1, 2})},
                                                          {},
                                                          ""),
                                              Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 10);

            EXPECT_THAT(output(), HasSubstr("s_nop 9"));
            clearOutput();
        }
    }

    TEST_F(MFMA950ObserverTest, XDLOPThenMFMA_SrcCExact)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 4);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 1, 1);

        // No hazard if SrcC is exactly overlapped
        {
            std::vector<Instruction> insts
                = {Instruction("v_mfma_f32_16x16x4_f32", {a[0]}, {v[0], v[1], a[0]}, {}, ""),
                   Instruction("v_mfma_f32_16x16x4_f32", {a[0]}, {v[2], v[3], a[0]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1]);

            EXPECT_THAT(output(), Not(HasSubstr("s_nop")));
            clearOutput();
        }
    }

    TEST_F(MFMA950ObserverTest, XDLOPThenMFMA_SrcCExact2Pass)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 4);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 1, 1);

        // Exception: Hazard for 2 pass MFMAs
        {
            std::vector<Instruction> insts
                = {Instruction("v_mfma_f32_4x4x1f32", {a[0]}, {v[0], v[1], a[0]}, {}, ""),
                   Instruction("v_mfma_f32_4x4x1f32", {a[0]}, {v[2], v[3], a[0]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 2);

            EXPECT_THAT(output(), HasSubstr("s_nop 1"));
            clearOutput();
        }
    }

    TEST_F(MFMA950ObserverTest, XDLOPThenMFMA_ReadA)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 5);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 2, 4);

        {
            std::vector<Instruction> insts
                = {Instruction("v_mfma_f32_16x16x4_f32", {v[2]}, {v[0], v[1], a[0]}, {}, ""),
                   Instruction("v_mfma_f32_16x16x4_f32", {a[1]}, {v[2], v[3], a[1]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 12);

            EXPECT_THAT(output(), HasSubstr("s_nop 11"));
            clearOutput();
        }
    }

    TEST_F(MFMA950ObserverTest, XDLOPThenFlat)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 5);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 2, 4);

        {
            std::vector<Instruction> insts
                = {Instruction("v_mfma_f32_16x16x4_f32", {v[2]}, {v[0], v[1], a[0]}, {}, ""),
                   Instruction("flat_store_dword", {}, {v[2], v[3]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 12);

            EXPECT_THAT(output(), HasSubstr("s_nop 11"));
            clearOutput();
        }
    }

    TEST_F(MFMA950ObserverTest, XDLOPThenVALU_Read)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 5);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 2, 4);

        {
            std::vector<Instruction> insts
                = {Instruction("v_mfma_f32_16x16x4_f32", {v[2]}, {v[0], v[1], a[0]}, {}, ""),
                   Instruction("v_or_b32", {v[4]}, {v[2], v[3]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 12);

            EXPECT_THAT(output(), HasSubstr("s_nop 11"));
            clearOutput();
        }
    }

    TEST_F(MFMA950ObserverTest, XDLOPThenVALU_Write)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 5);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 2, 4);

        {
            std::vector<Instruction> insts
                = {Instruction("v_mfma_f32_16x16x4_f32", {v[2]}, {v[0], v[1], a[0]}, {}, ""),
                   Instruction("v_or_b32", {v[2]}, {v[3], v[4]}, {}, ""),
                   Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 12);

            EXPECT_THAT(output(), HasSubstr("s_nop 11"));
            clearOutput();
        }
    }
}

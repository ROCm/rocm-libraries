/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

using namespace rocRoller;

namespace rocRollerTest
{
    class WMMA1200ObserverTest : public GenericContextFixture
    {
    protected:
        GPUArchitectureTarget targetArchitecture() override
        {
            return {GPUArchitectureGFX::GFX1200};
        }

        void peekAndSchedule(Instruction& inst, uint expectedNops = 0)
        {
            auto peeked = m_context->observer()->peek(inst);
            EXPECT_EQ(peeked.nops, expectedNops);
            m_context->schedule(inst);
        }
    };

    class WMMA1201ObserverTest : public GenericContextFixture
    {
    protected:
        GPUArchitectureTarget targetArchitecture() override
        {
            return {GPUArchitectureGFX::GFX1201};
        }

        void peekAndSchedule(Instruction& inst, uint expectedNops = 0)
        {
            auto peeked = m_context->observer()->peek(inst);
            EXPECT_EQ(peeked.nops, expectedNops);
            m_context->schedule(inst);
        }
    };

    TEST_F(WMMA1200ObserverTest, WMMAThenWMMA_AIsSameAsPrevD)
    {
        const auto v0 = createRegisters(Register::Type::Vector, DataType::Half, 5);
        const auto v1 = createRegisters(Register::Type::Vector, DataType::Half, 1, 2);

        // If the second WMMA's A is identical to the first WMMA's D then one V_NOP is expected
        {
            std::vector<Instruction> insts = {Instruction("v_wmma_f16_16x16x16_f16",
                                                          {v1[0]->subset({0, 1})},
                                                          {v0[0], v0[1], v0[2]},
                                                          {},
                                                          ""),
                                              Instruction("v_wmma_f16_16x16x16_f16",
                                                          {v0[3]},
                                                          {v1[0]->subset({0, 1}), v0[4], v0[3]},
                                                          {},
                                                          ""),
                                              Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 1);

            EXPECT_THAT(1, countSubstring(output(), "v_nop"));
            clearOutput();
        }
    }

    TEST_F(WMMA1200ObserverTest, WMMAThenWMMA_BIsSameAsPrevD)
    {
        const auto v0 = createRegisters(Register::Type::Vector, DataType::Half, 5);
        const auto v1 = createRegisters(Register::Type::Vector, DataType::Half, 1, 2);

        // If the second WMMA's B is identical to the first WMMA's D then one V_NOP is expected
        {
            std::vector<Instruction> insts = {Instruction("v_wmma_f16_16x16x16_f16",
                                                          {v1[0]->subset({0, 1})},
                                                          {v0[0], v0[1], v0[2]},
                                                          {},
                                                          ""),
                                              Instruction("v_wmma_f16_16x16x16_f16",
                                                          {v0[3]},
                                                          {v1[0]->subset({0, 1}), v0[4], v0[3]},
                                                          {},
                                                          ""),
                                              Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 1);

            EXPECT_THAT(1, countSubstring(output(), "v_nop"));
            clearOutput();
        }
    }

    TEST_F(WMMA1200ObserverTest, WMMAThenWMMA_AOverlapsPrevD)
    {
        const auto v0 = createRegisters(Register::Type::Vector, DataType::Half, 6);
        const auto v1 = createRegisters(Register::Type::Vector, DataType::Half, 1, 3);

        // If the second WMMA's A overlaps the first WMMA's D then one V_NOP is expected
        {
            std::vector<Instruction> insts = {Instruction("v_wmma_f16_16x16x16_f16",
                                                          {v1[0]->subset({0, 1})},
                                                          {v0[0], v0[1], v0[2]},
                                                          {},
                                                          ""),
                                              Instruction("v_wmma_f16_16x16x16_f16",
                                                          {v0[3]},
                                                          {v1[0]->subset({1, 2}), v0[4], v0[5]},
                                                          {},
                                                          ""),
                                              Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 1);

            EXPECT_THAT(1, countSubstring(output(), "v_nop"));
            clearOutput();
        }
    }

    TEST_F(WMMA1200ObserverTest, WMMAThenWMMA_BOverlapsPrevD)
    {
        const auto v0 = createRegisters(Register::Type::Vector, DataType::Half, 6);
        const auto v1 = createRegisters(Register::Type::Vector, DataType::Half, 1, 3);

        // If the second WMMA's B overlaps the first WMMA's D then one V_NOP is expected
        {
            std::vector<Instruction> insts = {Instruction("v_wmma_f16_16x16x16_f16",
                                                          {v1[0]->subset({0, 1})},
                                                          {v0[0], v0[1], v0[2]},
                                                          {},
                                                          ""),
                                              Instruction("v_wmma_f16_16x16x16_f16",
                                                          {v0[3]},
                                                          {v0[4], v1[0]->subset({1, 2}), v0[5]},
                                                          {},
                                                          ""),
                                              Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 1);

            EXPECT_THAT(1, countSubstring(output(), "v_nop"));
            clearOutput();
        }
    }

    TEST_F(WMMA1200ObserverTest, WMMAThenWMMA_Different)
    {
        const auto v0 = createRegisters(Register::Type::Vector, DataType::Half, 4);
        const auto v1 = createRegisters(Register::Type::Vector, DataType::Half, 1, 4);

        // No register overlap should not be a hazard
        {
            std::vector<Instruction> insts = {Instruction("v_wmma_f16_16x16x16_f16",
                                                          {v1[0]->subset({0, 1})},
                                                          {v0[0], v0[1], v1[0]->subset({0, 1})},
                                                          {},
                                                          ""),
                                              Instruction("v_wmma_f16_16x16x16_f16",
                                                          {v1[0]->subset({2, 3})},
                                                          {v0[2], v0[3], v1[0]->subset({2, 3})},
                                                          {},
                                                          ""),
                                              Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1]);

            EXPECT_THAT(0, countSubstring(output(), "v_nop"));
            clearOutput();
        }
    }

    TEST_F(WMMA1201ObserverTest, WMMAThenWMMA_AIsSameAsPrevD)
    {
        const auto v0 = createRegisters(Register::Type::Vector, DataType::Half, 6);
        const auto v1 = createRegisters(Register::Type::Vector, DataType::Half, 1, 2);

        // If the second WMMA's A is identical to the first WMMA's D then one V_NOP is expected
        {
            std::vector<Instruction> insts = {Instruction("v_wmma_f16_16x16x16_f16",
                                                          {v1[0]->subset({0, 1})},
                                                          {v0[0], v0[1], v0[2]},
                                                          {},
                                                          ""),
                                              Instruction("v_wmma_f16_16x16x16_f16",
                                                          {v0[3]},
                                                          {v1[0]->subset({0, 1}), v0[4], v0[5]},
                                                          {},
                                                          ""),
                                              Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 1);

            EXPECT_THAT(1, countSubstring(output(), "v_nop"));
            clearOutput();
        }
    }

    TEST_F(WMMA1201ObserverTest, WMMAThenWMMA_BIsSameAsPrevD)
    {
        const auto v0 = createRegisters(Register::Type::Vector, DataType::Half, 5);
        const auto v1 = createRegisters(Register::Type::Vector, DataType::Half, 1, 2);

        // If the second WMMA's B is identical to the first WMMA's D then one V_NOP is expected
        {
            std::vector<Instruction> insts = {Instruction("v_wmma_f16_16x16x16_f16",
                                                          {v1[0]->subset({0, 1})},
                                                          {v0[0], v0[1], v0[2]},
                                                          {},
                                                          ""),
                                              Instruction("v_wmma_f16_16x16x16_f16",
                                                          {v0[3]},
                                                          {v1[0]->subset({0, 1}), v0[4], v0[3]},
                                                          {},
                                                          ""),
                                              Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 1);

            EXPECT_THAT(1, countSubstring(output(), "v_nop"));
            clearOutput();
        }
    }

    TEST_F(WMMA1201ObserverTest, WMMAThenWMMA_AOverlapsPrevD)
    {
        const auto v0 = createRegisters(Register::Type::Vector, DataType::Half, 6);
        const auto v1 = createRegisters(Register::Type::Vector, DataType::Half, 1, 3);

        // If the second WMMA's A overlaps the first WMMA's D then one V_NOP is expected
        {
            std::vector<Instruction> insts = {Instruction("v_wmma_f16_16x16x16_f16",
                                                          {v1[0]->subset({0, 1})},
                                                          {v0[0], v0[1], v0[2]},
                                                          {},
                                                          ""),
                                              Instruction("v_wmma_f16_16x16x16_f16",
                                                          {v0[3]},
                                                          {v1[0]->subset({1, 2}), v0[4], v0[5]},
                                                          {},
                                                          ""),
                                              Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 1);

            EXPECT_THAT(1, countSubstring(output(), "v_nop"));
            clearOutput();
        }
    }

    TEST_F(WMMA1201ObserverTest, WMMAThenWMMA_BOverlapsPrevD)
    {
        const auto v0 = createRegisters(Register::Type::Vector, DataType::Half, 6);
        const auto v1 = createRegisters(Register::Type::Vector, DataType::Half, 1, 3);

        // If the second WMMA's B overlaps the first WMMA's D then one V_NOP is expected
        {
            std::vector<Instruction> insts = {Instruction("v_wmma_f16_16x16x16_f16",
                                                          {v1[0]->subset({0, 1})},
                                                          {v0[0], v0[1], v0[2]},
                                                          {},
                                                          ""),
                                              Instruction("v_wmma_f16_16x16x16_f16",
                                                          {v0[3]},
                                                          {v0[4], v1[0]->subset({1, 2}), v0[5]},
                                                          {},
                                                          ""),
                                              Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1], 1);

            EXPECT_THAT(1, countSubstring(output(), "v_nop"));
            clearOutput();
        }
    }

    TEST_F(WMMA1201ObserverTest, WMMAThenWMMA_Different)
    {
        const auto v0 = createRegisters(Register::Type::Vector, DataType::Half, 4);
        const auto v1 = createRegisters(Register::Type::Vector, DataType::Half, 1, 4);

        // No register overlap should not be a hazard
        {
            std::vector<Instruction> insts = {Instruction("v_wmma_f16_16x16x16_f16",
                                                          {v1[0]->subset({0, 1})},
                                                          {v0[0], v0[1], v1[0]->subset({0, 1})},
                                                          {},
                                                          ""),
                                              Instruction("v_wmma_f16_16x16x16_f16",
                                                          {v1[0]->subset({2, 3})},
                                                          {v0[2], v0[3], v1[0]->subset({2, 3})},
                                                          {},
                                                          ""),
                                              Instruction("s_endpgm", {}, {}, {}, "")};

            peekAndSchedule(insts[0]);
            peekAndSchedule(insts[1]);

            EXPECT_THAT(0, countSubstring(output(), "v_nop"));
            clearOutput();
        }
    }

}

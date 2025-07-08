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
#include <rocRoller/Scheduling/Observers/RegisterLivenessObserver.hpp>

#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"

using namespace rocRoller;

namespace rocRollerTest
{
    class RegisterLivenessObserverTest : public GenericContextFixture
    {
    protected:
        GPUArchitectureTarget targetArchitecture() override
        {
            return {GPUArchitectureGFX::GFX90A};
        }

        void SetUp() override
        {
            Settings::getInstance()->set(Settings::AllowUnknownInstructions, true);
            Settings::getInstance()->set(Settings::AssemblyFile, testKernelName());
            Settings::getInstance()->set(Settings::KernelAnalysis, true);
            GenericContextFixture::SetUp();
        }
    };

    TEST_F(RegisterLivenessObserverTest, SaveLivenessFile)
    {
        std::string expected_file = Settings::getInstance()->get(Settings::AssemblyFile) + ".live";

        if(std::filesystem::exists(expected_file))
            std::remove(expected_file.c_str());

        EXPECT_EQ(std::filesystem::exists(expected_file), false)
            << "The liveness file shouldn't exist before any instructions are emitted.";

        for(int i = 1; i <= 10; i++)
        {
            auto inst = Instruction("test_ins", {}, {}, {}, concatenate("Testing", i));
            m_context->schedule(inst);

            EXPECT_EQ(std::filesystem::exists(expected_file), false)
                << "The liveness file should not exist until the s_endpgm is emitted.";
        }

        auto inst_end = Instruction("s_endpgm", {}, {}, {}, "");
        m_context->schedule(inst_end);

        EXPECT_EQ(std::filesystem::exists(expected_file), true)
            << "The liveness file should exist since s_endpgm was emitted.";

        // Delete the file that was created
        if(std::filesystem::exists(expected_file))
            std::remove(expected_file.c_str());
    }

    TEST_F(RegisterLivenessObserverTest, BasicLiveness)
    {

        auto v = createRegisters(Register::Type::Vector, DataType::Float, 10);
        auto s = createRegisters(Register::Type::Scalar, DataType::Float, 10);

        auto test_gen = [&]() -> Generator<Instruction> {
            co_yield_(Instruction("setupv", {v[0], v[1]}, {}, {}, ""));
            co_yield_(Instruction("setups", {s[0], s[1]}, {}, {}, ""));
            co_yield_(Instruction("readv", {}, {v[0]}, {}, ""));
            co_yield_(Instruction("reads", {}, {s[0]}, {}, ""));
            co_yield_(Instruction("readwrites", {s[1]}, {s[1]}, {}, ""));
            co_yield_(Instruction("readv", {}, {v[1]}, {}, ""));
            co_yield_(Instruction("reads", {}, {s[1]}, {}, ""));
            co_yield_(Instruction("s_endpgm", {}, {}, {}, ""));
        };

        m_context->schedule(test_gen());

        std::string expected = R"(  VGPR      |SGPR      |Instruction
                                    ^^________|__________| 1. setupv v0, v1;
                                    ::________|^^________| 2. setups s0, s1;
                                    v:________|::________| 3. readv v0;
                                    _:________|v:________| 4. reads s0;
                                    _:________|_x________| 5. readwrites s1, s1;
                                    _v________|_:________| 6. readv v1;
                                    __________|_v________| 7. reads s1;
                                    __________|__________| 8. s_endpgm ;
                                )";

        std::string expected_file = Settings::getInstance()->get(Settings::AssemblyFile) + ".live";
        std::ifstream ifs(expected_file);
        std::string   liveness(std::istreambuf_iterator<char>{ifs}, {});
        ifs.close();

        EXPECT_EQ(NormalizedSource(liveness), NormalizedSource(expected));

        std::remove(expected_file.c_str());
    }

    TEST_F(RegisterLivenessObserverTest, AdvancedLiveness)
    {

        {
            auto v = createRegisters(Register::Type::Vector, DataType::Float, 6);
            auto s = createRegisters(Register::Type::Scalar, DataType::Float, 2);

            auto gen = [&]() -> Generator<Instruction> {
                co_yield_(Instruction("setupv", {v[0], v[1]}, {}, {}, ""));
                co_yield_(Instruction("setups", {s[0], s[1]}, {}, {}, ""));
                co_yield_(Instruction("readv", {}, {v[0]}, {}, ""));
                co_yield_(Instruction("reads", {}, {s[0]}, {}, ""));
                co_yield_(Instruction("readwrites", {s[1]}, {s[1]}, {}, ""));
                co_yield_(Instruction("readv", {}, {v[1]}, {}, ""));
                co_yield_(Instruction("reads", {}, {s[1]}, {}, ""));
            };

            m_context->schedule(gen());
        }

        {
            auto s = createRegisters(Register::Type::Scalar, DataType::Float, 6);
            auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 6);

            auto gen = [&]() -> Generator<Instruction> {
                co_yield_(Instruction("setupa", {a[0], a[1]}, {}, {}, ""));
                co_yield_(Instruction("setups", {s[0], s[1]}, {}, {}, ""));
                co_yield_(Instruction("reada", {}, {a[0]}, {}, ""));
                co_yield_(Instruction("reads", {}, {s[0]}, {}, ""));
                co_yield_(Instruction("readwritea", {a[1]}, {a[1]}, {}, ""));
                co_yield_(Instruction("reada", {}, {a[1]}, {}, ""));
                co_yield_(Instruction("reads", {}, {s[1]}, {}, ""));
                co_yield_(Instruction("s_endpgm", {}, {}, {}, ""));
            };

            m_context->schedule(gen());
        }

        std::string expected1 = R"( ACCVGPR|VGPR..|SGPR..|Instruction
                                    ......|^^____|__....|.1..setupv.v0,.v1;
                                    ......|::____|^^....|.2..setups.s0,.s1;
                                    ......|v:____|::....|.3..readv.v0;
                                    ......|_:____|v:....|.4..reads.s0;
                                    ......|_:____|_x....|.5..readwrites.s1,.s1;
                                    ......|_v____|_:....|.6..readv.v1;
                                    ......|______|_v....|.7..reads.s1;
                                )";

        std::string expected2 = R"( ^^____|......|______|.24..setupa.a0,.a1;
                                    ::____|......|^^____|.25..setups.s0,.s1;
                                    v:____|......|::____|.26..reada.a0;
                                    _:____|......|v:____|.27..reads.s0;
                                    _x____|......|_:____|.28..readwritea.a1,.a1;
                                    _v____|......|_:____|.29..reada.a1;
                                    ______|......|_v____|.30..reads.s1;
                                    ______|......|______|.31..s_endpgm.;
                                )";

        std::string expected_file = Settings::getInstance()->get(Settings::AssemblyFile) + ".live";
        std::ifstream ifs(expected_file);
        std::string   liveness(std::istreambuf_iterator<char>{ifs}, {});
        ifs.close();
        std::replace(liveness.begin(), liveness.end(), ' ', '.');
        liveness = NormalizedSource(liveness);

        EXPECT_THAT(liveness, testing::HasSubstr(NormalizedSource(expected1)));
        EXPECT_THAT(liveness, testing::HasSubstr(NormalizedSource(expected2)));

        std::remove(expected_file.c_str());
    }

    TEST_F(RegisterLivenessObserverTest, BranchLiveness)
    {

        {
            auto v = createRegisters(Register::Type::Vector, DataType::Float, 10);
            auto s = createRegisters(Register::Type::Scalar, DataType::Float, 10);

            auto gen = [&]() -> Generator<Instruction> {
                co_yield_(Instruction::Label("KERNEL_NAME"));
                co_yield_(Instruction("setupv", {v[0], v[1]}, {}, {}, ""));
                co_yield_(Instruction("setups", {s[0], s[1]}, {}, {}, ""));
                co_yield_(Instruction::Label("LOOP_TOP"));
                co_yield_(Instruction("readv", {}, {v[0]}, {}, ""));
                co_yield_(Instruction("reads", {}, {s[0]}, {}, ""));
                co_yield_(Instruction("unrelated", {}, {v[1]}, {}, ""));
                co_yield_(Instruction("unrelated", {}, {s[1]}, {}, ""));
                co_yield_(Instruction("writev", {v[0]}, {}, {}, ""));
                co_yield_(Instruction("writes", {s[0]}, {}, {}, ""));
                co_yield_(Instruction(
                    "s_cbranch_vccnz", {}, {Register::Value::Label("LOOP_TOP")}, {}, ""));
                co_yield_(Instruction::Label("LOOP_BOTTOM"));
                co_yield_(Instruction("readv", {}, {v[0]}, {}, ""));
                co_yield_(Instruction("reads", {}, {s[0]}, {}, ""));

                co_yield_(Instruction("setupv", {v[4], v[5]}, {}, {}, ""));
                co_yield_(Instruction("setups", {s[4], s[5]}, {}, {}, ""));
                co_yield_(Instruction("unrelated", {}, {v[5]}, {}, ""));
                co_yield_(Instruction("unrelated", {}, {s[5]}, {}, ""));
                co_yield_(Instruction(
                    "s_cbranch_vccnz", {}, {Register::Value::Label("LOOP_BOTTOM2")}, {}, ""));
                co_yield_(Instruction::Label("LOOP_TOP2"));
                co_yield_(Instruction("unrelated", {}, {v[5]}, {}, ""));
                co_yield_(Instruction("unrelated", {}, {s[5]}, {}, ""));
                co_yield_(Instruction("writev", {v[4]}, {}, {}, ""));
                co_yield_(Instruction("writes", {s[4]}, {}, {}, ""));
                co_yield_(Instruction("unrelated", {}, {v[5]}, {}, ""));
                co_yield_(Instruction("unrelated", {}, {s[5]}, {}, ""));
                co_yield_(Instruction("writev", {v[4]}, {}, {}, ""));
                co_yield_(Instruction("writes", {s[4]}, {}, {}, ""));
                co_yield_(Instruction(
                    "s_cbranch_vccnz", {}, {Register::Value::Label("LOOP_TOP2")}, {}, ""));
                co_yield_(Instruction::Label("LOOP_BOTTOM2"));
                co_yield_(Instruction("readv", {}, {v[4]}, {}, ""));
                co_yield_(Instruction("reads", {}, {s[4]}, {}, ""));
                co_yield_(Instruction("s_endpgm", {}, {}, {}, ""));
            };

            m_context->schedule(gen());
        }

        std::string expected1 = R"( ^^________|__________|.3..setupv.v0,.v1;
                                    ::________|^^________|.4..setups.s0,.s1;
                                    ::________|::________|.5..LOOP_TOP:;;
                                    v:________|::________|.7..readv.v0;
                                    _:________|v:________|.8..reads.s0;
                                    _v________|_:________|.9..unrelated.v1;
                                    _:________|_v________|.10..unrelated.s1;
                                    ^:________|_:________|.11..writev.v0;
                                    ::________|^:________|.12..writes.s0;
                                    ::________|::________|.13..s_cbranch_vccnz.LOOP_TOP;
                                    :_________|:_________|.14..LOOP_BOTTOM:;;
                                    v_________|:_________|.16..readv.v0;
                                    __________|v_________|.17..reads.s0;
                                )";

        std::string expected2 = R"( ____^^____|__________|.18..setupv.v4,.v5;
                                    ____::____|____^^____|.19..setups.s4,.s5;
                                    ____:v____|____::____|.20..unrelated.v5;
                                    ____::____|____:v____|.21..unrelated.s5;
                                    ____::____|____::____|.22..s_cbranch_vccnz.LOOP_BOTTOM2;
                                    _____:____|_____:____|.23..LOOP_TOP2:;;
                                    _____v____|_____:____|.25..unrelated.v5;
                                    _____:____|_____v____|.26..unrelated.s5;
                                    ____^:____|_____:____|.27..writev.v4;
                                    _____:____|____^:____|.28..writes.s4;
                                    _____v____|_____:____|.29..unrelated.v5;
                                    _____:____|_____v____|.30..unrelated.s5;
                                    ____^:____|_____:____|.31..writev.v4;
                                    ____::____|____^:____|.32..writes.s4;
                                    ____::____|____::____|.33..s_cbranch_vccnz.LOOP_TOP2;
                                    ____:_____|____:_____|.34..LOOP_BOTTOM2:;;
                                    ____v_____|____:_____|.36..readv.v4;
                                    __________|____v_____|.37..reads.s4;
                                )";

        std::string expected_file = Settings::getInstance()->get(Settings::AssemblyFile) + ".live";
        std::ifstream ifs(expected_file);
        std::string   liveness(std::istreambuf_iterator<char>{ifs}, {});
        ifs.close();
        std::replace(liveness.begin(), liveness.end(), ' ', '.');
        liveness = NormalizedSource(liveness);

        EXPECT_THAT(liveness, testing::HasSubstr(NormalizedSource(expected1)));
        EXPECT_THAT(liveness, testing::HasSubstr(NormalizedSource(expected2)));

        std::remove(expected_file.c_str());
    }

    class RegisterLivenessObserverNegativeTest : public GenericContextFixture
    {
    protected:
        GPUArchitectureTarget targetArchitecture() override
        {
            return {GPUArchitectureGFX::GFX90A};
        }

        void SetUp() override
        {
            Settings::getInstance()->set(Settings::AllowUnknownInstructions, true);
            Settings::getInstance()->set(Settings::AssemblyFile, testKernelName());
            Settings::getInstance()->set(Settings::KernelAnalysis, false);
            GenericContextFixture::SetUp();
        }
    };

    TEST_F(RegisterLivenessObserverNegativeTest, DontSaveLivenessFile)
    {
        std::string expected_file = Settings::getInstance()->get(Settings::AssemblyFile) + ".live";

        if(std::filesystem::exists(expected_file))
            std::remove(expected_file.c_str());

        EXPECT_EQ(std::filesystem::exists(expected_file), false)
            << "The liveness file should not exist.";

        for(int i = 1; i <= 10; i++)
        {
            auto inst = Instruction("test_inst", {}, {}, {}, concatenate("Testing", i));
            m_context->schedule(inst);

            EXPECT_EQ(std::filesystem::exists(expected_file), false)
                << "The liveness file should not exist.";
        }

        auto inst_end = Instruction("s_endpgm", {}, {}, {}, "");
        m_context->schedule(inst_end);

        EXPECT_EQ(std::filesystem::exists(expected_file), false)
            << "The liveness file should not exist.";

        // Delete the file that was created
        if(std::filesystem::exists(expected_file))
            std::remove(expected_file.c_str());
    }
}

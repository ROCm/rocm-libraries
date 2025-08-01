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

#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Scheduling/Scheduler.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"

#define Inst(opcode) Instruction(opcode, {}, {}, {}, "")

using namespace rocRoller;

namespace rocRollerTest
{
    struct SchedulerTest : public GenericContextFixture
    {
        GPUArchitectureTarget targetArchitecture() override
        {
            return {GPUArchitectureGFX::GFX90A};
        }

        void SetUp() override
        {
            Settings::getInstance()->set(Settings::AllowUnknownInstructions, true);
            GenericContextFixture::SetUp();
        }

        Generator<Instruction> testGeneratorWithComments(bool includeComments = true);
    };

    Generator<Instruction> SchedulerTest::testGeneratorWithComments(bool includeComments)
    {
        co_yield_(Instruction("s_sub_u32", {}, {}, {}, "Comment on an instruction"));
        if(includeComments)
        {
            co_yield Instruction::Comment("Pure comment 1");
            co_yield Instruction::Comment("Pure comment 2");
        }

        co_yield_(Instruction("s_add_u32", {}, {}, {}, "Comment on an instruction"));
        if(includeComments)
        {
            co_yield Instruction::Comment("Pure comment 3");
        }
        co_yield Instruction::Nop("Nop Comment");

        co_yield Instruction::Lock(Scheduling::Dependency::SCC, "Lock instruction");
        co_yield Instruction::Unlock("Unlock instruction");

        if(includeComments)
        {
            co_yield Instruction::Comment("Pure comment 4");
            co_yield Instruction::Comment("Pure comment 5");
        }

        auto reg = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Float, 1);
        co_yield reg->allocate();

        if(includeComments)
        {
            co_yield Instruction::Comment("Pure comment 6");
        }
        co_yield Instruction::Directive(".set .amdgcn.next_free_vgpr, 0");

        co_yield Instruction::Label("FooLabel");

        if(includeComments)
        {
            co_yield Instruction::Comment("Pure comment 7");
        }
        co_yield Instruction::Wait(WaitCount::Zero(m_context->targetArchitecture()));

        if(includeComments)
        {
            co_yield Instruction::Comment("Pure comment 8");
        }
    }

    template <typename Begin, typename End>
    Instruction next(Begin& begin, End const& end)
    {
        AssertFatal(begin != end);
        auto inst = *begin;
        ++begin;
        return std::move(inst);
    }

    TEST_F(SchedulerTest, Enum)
    {
        using namespace rocRoller::Scheduling;
        EXPECT_EQ(toString(SchedulerProcedure::Sequential), "Sequential");
        EXPECT_EQ(toString(SchedulerProcedure::RoundRobin), "RoundRobin");
        EXPECT_EQ(toString(SchedulerProcedure::Random), "Random");
        EXPECT_EQ(toString(SchedulerProcedure::Cooperative), "Cooperative");
        EXPECT_EQ(toString(SchedulerProcedure::Priority), "Priority");
        EXPECT_EQ(toString(SchedulerProcedure::Count), "Count");

        EXPECT_EQ(fromString<SchedulerProcedure>("Sequential"), SchedulerProcedure::Sequential);
        EXPECT_EQ(fromString<SchedulerProcedure>("RoundRobin"), SchedulerProcedure::RoundRobin);
        EXPECT_EQ(fromString<SchedulerProcedure>("Random"), SchedulerProcedure::Random);
        EXPECT_EQ(fromString<SchedulerProcedure>("Cooperative"), SchedulerProcedure::Cooperative);
        EXPECT_EQ(fromString<SchedulerProcedure>("Priority"), SchedulerProcedure::Priority);
        EXPECT_ANY_THROW(fromString<SchedulerProcedure>("Count"));
        EXPECT_ANY_THROW(fromString<SchedulerProcedure>("fjdksl"));
    }

    TEST_F(SchedulerTest, ConsumeCommentsTest)
    {
        using namespace rocRoller::Scheduling;

        auto gen   = testGeneratorWithComments();
        auto begin = gen.begin();
        auto end   = gen.end();

        auto inst = next(begin, end);
        EXPECT_EQ("s_sub_u32", inst.getOpCode());

        EXPECT_TRUE(begin->isCommentOnly());

        {
            m_context->schedule(consumeComments(begin, end));
            std::string expected = R"(// Pure comment 1
                                      // Pure comment 2)";

            EXPECT_EQ(NormalizedSource(expected, true), NormalizedSource(output(), true));

            auto inst = next(begin, end);
            EXPECT_EQ(inst.getOpCode(), "s_add_u32");
        }

        EXPECT_TRUE(begin != end);

        {
            clearOutput();
            m_context->schedule(consumeComments(begin, end));
            std::string expected = "// Pure comment 3\n";

            EXPECT_EQ(NormalizedSource(expected, true), NormalizedSource(output(), true));

            auto inst = next(begin, end);
            EXPECT_EQ(inst.nopCount(), 1);
            EXPECT_EQ(inst.toString(LogLevel::Debug), "s_nop 0\n // Nop Comment\n");
        }

        {
            // Next is a lock, so there should be no comments consumed.
            clearOutput();
            m_context->schedule(consumeComments(begin, end));
            std::string expected = "";

            EXPECT_EQ(NormalizedSource(expected, true), NormalizedSource(output(), true));

            auto inst = next(begin, end);
            EXPECT_EQ(inst.toString(LogLevel::Debug), " // Lock instruction\n");
        }

        {
            // Next is an unlock, so there should be no comments consumed.
            clearOutput();
            m_context->schedule(consumeComments(begin, end));
            std::string expected = "";

            EXPECT_EQ(NormalizedSource(expected, true), NormalizedSource(output(), true));

            auto inst = next(begin, end);
            EXPECT_EQ(inst.toString(LogLevel::Debug), " // Unlock instruction\n");
        }

        {
            clearOutput();
            m_context->schedule(consumeComments(begin, end));
            std::string expected = "// Pure comment 4\n// Pure comment 5\n";

            EXPECT_EQ(NormalizedSource(expected, true), NormalizedSource(output(), true));

            auto inst = next(begin, end);
            EXPECT_EQ(inst.toString(LogLevel::Debug), "// Allocated : 1 VGPR (Value: Float)\n");
        }

        {
            clearOutput();
            m_context->schedule(consumeComments(begin, end));
            std::string expected = "// Pure comment 6\n";

            EXPECT_EQ(NormalizedSource(expected, true), NormalizedSource(output(), true));

            auto inst = next(begin, end);
            EXPECT_EQ(inst.toString(LogLevel::Debug), ".set .amdgcn.next_free_vgpr, 0\n");
        }

        {
            // Next is a label, so there should be no comments consumed.
            clearOutput();
            m_context->schedule(consumeComments(begin, end));
            std::string expected = "";

            EXPECT_EQ(NormalizedSource(expected, true), NormalizedSource(output(), true));

            auto inst = next(begin, end);
            EXPECT_EQ(inst.toString(LogLevel::Debug), "FooLabel:\n\n");
        }

        {
            clearOutput();
            m_context->schedule(consumeComments(begin, end));
            std::string expected = "// Pure comment 7\n";

            EXPECT_EQ(NormalizedSource(expected, true), NormalizedSource(output(), true));

            auto inst       = next(begin, end);
            auto instString = R"(s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0))";
            EXPECT_EQ(NormalizedSource(inst.toString(LogLevel::Debug)),
                      NormalizedSource(instString));
        }

        {
            clearOutput();
            m_context->schedule(consumeComments(begin, end));
            std::string expected = "// Pure comment 8\n";

            EXPECT_EQ(NormalizedSource(expected, true), NormalizedSource(output(), true));

            EXPECT_EQ(begin, end);
        }
    }

    TEST_F(SchedulerTest, SequentialSchedulerTest)
    {
        auto generator_one = [&]() -> Generator<Instruction> {
            co_yield_(Inst("Instruction 1, Generator 1"));
            co_yield_(Inst("Instruction 2, Generator 1"));
            co_yield_(Inst("Instruction 3, Generator 1"));
        };
        auto generator_two = [&]() -> Generator<Instruction> {
            co_yield_(Inst("Instruction 1, Generator 2"));
            co_yield_(Inst("Instruction 2, Generator 2"));
            co_yield_(Inst("Instruction 3, Generator 2"));
            co_yield_(Inst("Instruction 4, Generator 2"));
        };

        std::vector<Generator<Instruction>> generators;
        generators.push_back(generator_one());
        generators.push_back(generator_two());

        std::string expected = R"( Instruction 1, Generator 1
                                    Instruction 2, Generator 1
                                    Instruction 3, Generator 1
                                    Instruction 1, Generator 2
                                    Instruction 2, Generator 2
                                    Instruction 3, Generator 2
                                    Instruction 4, Generator 2
                                )";

        auto scheduler = Component::GetNew<Scheduling::Scheduler>(
            Scheduling::SchedulerProcedure::Sequential, Scheduling::CostFunction::None, m_context);
        m_context->schedule((*scheduler)(generators));
        EXPECT_EQ(NormalizedSource(output(), true), NormalizedSource(expected, true));
    }

    TEST_F(SchedulerTest, RoundRobinSchedulerTest)
    {
        auto generator_one = [&]() -> Generator<Instruction> {
            co_yield_(Inst("Instruction 1, Generator 1"));
            co_yield_(Inst("Instruction 2, Generator 1"));
            co_yield_(Inst("Instruction 3, Generator 1"));
        };
        auto generator_two = [&]() -> Generator<Instruction> {
            co_yield_(Inst("Instruction 1, Generator 2"));
            co_yield_(Inst("Instruction 2, Generator 2"));
            co_yield_(Inst("Instruction 3, Generator 2"));
            co_yield_(Inst("Instruction 4, Generator 2"));
        };

        std::vector<Generator<Instruction>> generators;
        generators.push_back(generator_one());
        generators.push_back(generator_two());

        std::string expected = R"( Instruction 1, Generator 1
                                    Instruction 1, Generator 2
                                    Instruction 2, Generator 1
                                    Instruction 2, Generator 2
                                    Instruction 3, Generator 1
                                    Instruction 3, Generator 2
                                    Instruction 4, Generator 2
                                )";

        auto scheduler = Component::GetNew<Scheduling::Scheduler>(
            Scheduling::SchedulerProcedure::RoundRobin, Scheduling::CostFunction::None, m_context);
        m_context->schedule((*scheduler)(generators));
        EXPECT_EQ(NormalizedSource(output(), true), NormalizedSource(expected, true));
    }

    // Can be run with the --gtest_also_run_disabled_tests option
    TEST_F(SchedulerTest, DISABLED_SchedulerWaitStressTest)
    {
        auto const& arch      = m_context->targetArchitecture();
        auto        generator = [arch]() -> Generator<Instruction> {
            for(size_t i = 0; i < 1000000; i++)
            {
                co_yield Instruction::Wait(WaitCount::LoadCnt(arch, 1, "Comment"));
            }
        };

        m_context->schedule(generator());
    }

    // Can be run with the --gtest_also_run_disabled_tests option
    TEST_F(SchedulerTest, DISABLED_SchedulerCopyStressTest)
    {
        auto v_a
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Int32, 1);

        auto v_b
            = Register::Value::Placeholder(m_context, Register::Type::Vector, DataType::Int32, 1);

        auto generator = [&]() -> Generator<Instruction> {
            co_yield v_a->allocate();
            co_yield v_b->allocate();

            for(size_t i = 0; i < 1000000; i++)
            {
                co_yield m_context->copier()->copy(v_a, v_b, "Comment");
            }
        };

        m_context->schedule(generator());
    }

    TEST_F(SchedulerTest, DoubleUnlocking)
    {
        auto scheduler = Component::GetNew<Scheduling::Scheduler>(
            Scheduling::SchedulerProcedure::RoundRobin, Scheduling::CostFunction::None, m_context);

        std::vector<Generator<Instruction>> sequences;

        auto noUnlock = [&]() -> Generator<Instruction> {
            co_yield(Instruction::Lock(Scheduling::Dependency::SCC));
            co_yield(Instruction::Unlock());
            co_yield(Inst("Test"));
            co_yield(Instruction::Unlock());
        };

        sequences.push_back(noUnlock());

        EXPECT_THROW(m_context->schedule((*scheduler)(sequences));, FatalError);
    }

    TEST_F(SchedulerTest, noUnlocking)
    {
        auto scheduler = Component::GetNew<Scheduling::Scheduler>(
            Scheduling::SchedulerProcedure::RoundRobin, Scheduling::CostFunction::None, m_context);

        std::vector<Generator<Instruction>> sequences;

        auto noUnlock = [&]() -> Generator<Instruction> {
            co_yield(Instruction::Lock(Scheduling::Dependency::SCC));
            co_yield(Inst("Test"));
        };

        sequences.push_back(noUnlock());

        EXPECT_THROW(m_context->schedule((*scheduler)(sequences));, FatalError);
    }

    TEST_F(SchedulerTest, SchedulerDepth)
    {
        auto schedulerA = Component::GetNew<Scheduling::Scheduler>(
            Scheduling::SchedulerProcedure::RoundRobin, Scheduling::CostFunction::None, m_context);
        auto schedulerB = Component::GetNew<Scheduling::Scheduler>(
            Scheduling::SchedulerProcedure::RoundRobin, Scheduling::CostFunction::None, m_context);
        auto schedulerC = Component::GetNew<Scheduling::Scheduler>(
            Scheduling::SchedulerProcedure::RoundRobin, Scheduling::CostFunction::None, m_context);

        std::vector<Generator<Instruction>> a_sequences;
        std::vector<Generator<Instruction>> b_sequences;
        std::vector<Generator<Instruction>> c_sequences;

        auto opB = [&]() -> Generator<Instruction> {
            co_yield(Inst("(C) Op B Begin"));
            co_yield(Inst("(C) Op B Instruction"));
            co_yield(Inst("(C) Op B End"));
        };

        auto ifBlock = [&]() -> Generator<Instruction> {
            EXPECT_EQ(schedulerA->getLockState().getTopDependency(1), Scheduling::Dependency::VCC);
            EXPECT_EQ(schedulerB->getLockState().getTopDependency(0), Scheduling::Dependency::VCC);
            EXPECT_EQ(schedulerC->getLockState().getTopDependency(1), Scheduling::Dependency::None);

            co_yield(
                Inst("(C) If Begin").lock(Scheduling::Dependency::SCC, "(C) Scheduler C Lock"));

            EXPECT_EQ(schedulerA->getLockState().getTopDependency(1), Scheduling::Dependency::SCC);
            EXPECT_EQ(schedulerB->getLockState().getTopDependency(0), Scheduling::Dependency::SCC);
            EXPECT_EQ(schedulerC->getLockState().getTopDependency(1), Scheduling::Dependency::SCC);

            co_yield(Inst("+++ Scheduler A Stream 1 Lock Depth: "
                          + std::to_string(schedulerA->getLockState().getLockDepth(1))));
            co_yield(Inst("+++ Scheduler B Stream 0 Lock Depth: "
                          + std::to_string(schedulerB->getLockState().getLockDepth(0))));
            co_yield(Inst("+++ Scheduler C Stream 1 Lock Depth: "
                          + std::to_string(schedulerC->getLockState().getLockDepth(1))));
            co_yield(Inst("(C) If Instruction"));
            co_yield(Inst("(C) If End").unlock("(C) Scheduler C Unlock"));

            EXPECT_EQ(schedulerA->getLockState().getTopDependency(1), Scheduling::Dependency::VCC);
            EXPECT_EQ(schedulerB->getLockState().getTopDependency(0), Scheduling::Dependency::VCC);
            EXPECT_EQ(schedulerC->getLockState().getTopDependency(1), Scheduling::Dependency::None);

            co_yield(Inst("+++ Scheduler A Stream 1 Lock Depth: "
                          + std::to_string(schedulerA->getLockState().getLockDepth(1))));
            co_yield(Inst("+++ Scheduler B Stream 0 Lock Depth: "
                          + std::to_string(schedulerB->getLockState().getLockDepth(0))));
            co_yield(Inst("+++ Scheduler C Stream 1 Lock Depth: "
                          + std::to_string(schedulerC->getLockState().getLockDepth(1))));
        };

        c_sequences.push_back(opB());
        c_sequences.push_back(ifBlock());

        auto unroll0 = [&]() -> Generator<Instruction> {
            EXPECT_EQ(schedulerA->getLockState().getTopDependency(1),
                      Scheduling::Dependency::Branch);
            EXPECT_EQ(schedulerB->getLockState().getTopDependency(0), Scheduling::Dependency::None);

            co_yield(Inst("(B) Unroll 0 Begin")
                         .lock(Scheduling::Dependency::VCC, "(B) Scheduler B Lock"));

            EXPECT_EQ(schedulerA->getLockState().getTopDependency(1), Scheduling::Dependency::VCC);
            EXPECT_EQ(schedulerB->getLockState().getTopDependency(0), Scheduling::Dependency::VCC);

            co_yield(Inst("+++ Scheduler A Stream 1 Lock Depth: "
                          + std::to_string(schedulerA->getLockState().getLockDepth(1))));
            co_yield(Inst("+++ Scheduler B Stream 0 Lock Depth: "
                          + std::to_string(schedulerB->getLockState().getLockDepth(0))));
            co_yield((*schedulerC)(c_sequences));
            co_yield(Inst("(B) Unroll 0 End")).unlock("(B) Scheduler B Unlock");

            EXPECT_EQ(schedulerA->getLockState().getTopDependency(1),
                      Scheduling::Dependency::Branch);
            EXPECT_EQ(schedulerB->getLockState().getTopDependency(0), Scheduling::Dependency::None);

            co_yield(Inst("+++ Scheduler A Stream 1 Lock Depth: "
                          + std::to_string(schedulerA->getLockState().getLockDepth(1))));
            co_yield(Inst("+++ Scheduler B Stream 0 Lock Depth: "
                          + std::to_string(schedulerB->getLockState().getLockDepth(0))));
        };

        auto unroll1 = [&]() -> Generator<Instruction> {
            co_yield(Inst("(B) Unroll 1 Begin"));
            co_yield(Inst("(B) Unroll 1 Instruction"));
            co_yield(Inst("(B) Unroll 1 End"));
        };

        b_sequences.push_back(unroll0());
        b_sequences.push_back(unroll1());

        auto opA = [&]() -> Generator<Instruction> {
            co_yield(Inst("(A) Op A Begin"));
            co_yield(Inst("(A) Op A Instruction"));
            co_yield(Inst("(A) Op A End"));
        };

        auto forloop = [&]() -> Generator<Instruction> {
            co_yield(Inst("(A) For Loop Begin")
                         .lock(Scheduling::Dependency::Branch, "(A) Scheduler A Lock"));

            EXPECT_EQ(schedulerA->getLockState().getTopDependency(1),
                      Scheduling::Dependency::Branch);

            co_yield(Inst("+++ Scheduler A Stream 1 Lock Depth: "
                          + std::to_string(schedulerA->getLockState().getLockDepth(1))));
            co_yield((*schedulerB)(b_sequences));
            co_yield(Inst("(A) For Loop End").unlock("(A) Scheduler A Unlock"));
            co_yield(Inst("+++ Scheduler A Stream 1 Lock Depth: "
                          + std::to_string(schedulerA->getLockState().getLockDepth(1))));

            EXPECT_EQ(schedulerA->getLockState().getTopDependency(1), Scheduling::Dependency::None);
        };

        a_sequences.push_back(opA());
        a_sequences.push_back(forloop());

        m_context->schedule((*schedulerA)(a_sequences));

        std::string expected = R"( (A) Op A Begin
                                    (A) For Loop Begin  // (A) Scheduler A Lock
                                    +++ Scheduler A Stream 1 Lock Depth: 1
                                    (B) Unroll 0 Begin  // (B) Scheduler B Lock
                                    (B) Unroll 1 Begin
                                    +++ Scheduler A Stream 1 Lock Depth: 2
                                    (B) Unroll 1 Instruction
                                    +++ Scheduler B Stream 0 Lock Depth: 1
                                    (B) Unroll 1 End
                                    (C) Op B Begin
                                    (C) If Begin  // (C) Scheduler C Lock
                                    +++ Scheduler A Stream 1 Lock Depth: 3
                                    +++ Scheduler B Stream 0 Lock Depth: 2
                                    +++ Scheduler C Stream 1 Lock Depth: 1
                                    (C) If Instruction
                                    (C) If End  // (C) Scheduler C Unlock
                                    (C) Op B Instruction
                                    +++ Scheduler A Stream 1 Lock Depth: 2
                                    (C) Op B End
                                    +++ Scheduler B Stream 0 Lock Depth: 1
                                    +++ Scheduler C Stream 1 Lock Depth: 0
                                    (B) Unroll 0 End  // (B) Scheduler B Unlock
                                    +++ Scheduler A Stream 1 Lock Depth: 1
                                    +++ Scheduler B Stream 0 Lock Depth: 0
                                    (A) For Loop End  // (A) Scheduler A Unlock
                                    (A) Op A Instruction
                                    +++ Scheduler A Stream 1 Lock Depth: 0
                                    (A) Op A End
                                )";

        EXPECT_EQ(NormalizedSource(output(), true), NormalizedSource(expected, true));
    }

    struct RandomSchedulerTest : public SchedulerTest, public testing::WithParamInterface<int>
    {
        void SetUp() override
        {
            Settings::getInstance()->set(Settings::AllowUnknownInstructions, true);
            GenericContextFixture::SetUp();
            int seed = GetParam();

            RecordProperty("random_seed", seed);

            m_context->setRandomSeed(seed);
        }
    };

    Generator<Instruction> noComments()
    {
        for(int i = 0; i < 100; i++)
            co_yield_(Inst(concatenate("I", i)));
    }

    TEST_P(RandomSchedulerTest, RandomScheduler)
    {
        std::string output1, output2, output3, output4, output5;

        {
            std::vector<Generator<Instruction>> gens;
            gens.push_back(testGeneratorWithComments());
            gens.push_back(noComments());
            gens.push_back(noComments());

            auto scheduler = Component::GetNew<Scheduling::Scheduler>(
                Scheduling::SchedulerProcedure::Random, Scheduling::CostFunction::None, m_context);
            m_context->schedule((*scheduler)(gens));

            output1 = NormalizedSource(output(), true);
        }

        {
            m_context->setRandomSeed(GetParam());
            clearOutput();

            std::vector<Generator<Instruction>> gens;
            gens.push_back(testGeneratorWithComments());
            gens.push_back(noComments());
            gens.push_back(noComments());

            auto scheduler = Component::GetNew<Scheduling::Scheduler>(
                Scheduling::SchedulerProcedure::Random, Scheduling::CostFunction::None, m_context);
            m_context->schedule((*scheduler)(gens));

            output2 = NormalizedSource(output(), true);
        }

        {
            m_context->setRandomSeed(GetParam());
            clearOutput();

            std::vector<Generator<Instruction>> gens;
            gens.push_back(testGeneratorWithComments(false));
            gens.push_back(noComments());
            gens.push_back(noComments());

            auto scheduler = Component::GetNew<Scheduling::Scheduler>(
                Scheduling::SchedulerProcedure::Random, Scheduling::CostFunction::None, m_context);
            m_context->schedule((*scheduler)(gens));

            output3 = NormalizedSource(output(), true);
        }

        {
            m_context->setRandomSeed(GetParam() + 1);
            clearOutput();

            std::vector<Generator<Instruction>> gens;
            gens.push_back(testGeneratorWithComments());
            gens.push_back(noComments());
            gens.push_back(noComments());

            auto scheduler = Component::GetNew<Scheduling::Scheduler>(
                Scheduling::SchedulerProcedure::Random, Scheduling::CostFunction::None, m_context);
            m_context->schedule((*scheduler)(gens));

            output4 = NormalizedSource(output(), true);
        }

        Settings::getInstance()->set(Settings::RandomSeed, GetParam());
        {
            m_context->setRandomSeed(GetParam() + 1);
            clearOutput();

            std::vector<Generator<Instruction>> gens;
            gens.push_back(testGeneratorWithComments());
            gens.push_back(noComments());
            gens.push_back(noComments());

            auto scheduler = Component::GetNew<Scheduling::Scheduler>(
                Scheduling::SchedulerProcedure::Random, Scheduling::CostFunction::None, m_context);
            m_context->schedule((*scheduler)(gens));

            output5 = NormalizedSource(output(), true);
        }

        // The same generators with the same random seed should produce the same output.
        EXPECT_EQ(output1, output2);

        // Not including the comments should still produce the same code.
        EXPECT_EQ(NormalizedSource(output1), NormalizedSource(output3));

        // The same generators with a different random seed should produce different output.
        EXPECT_NE(output1, output4);

        // The settings override should take precedence.
        EXPECT_EQ(output1, output5);

        // The scheduler should not break up comments or locked sections of instructions.
        auto block1 = NormalizedSource(R"(
        // Pure comment 1
        // Pure comment 2
        )",
                                       true);

        auto block2 = NormalizedSource(R"(
        // Lock instruction
        // Unlock instruction
        )",
                                       true);

        std::string block3 = NormalizedSource(R"(
        // Pure comment 4
        // Pure comment 5
        )",
                                              true);

        EXPECT_THAT(output1, testing::HasSubstr(block1));
        EXPECT_THAT(output1, testing::HasSubstr(block2));
        EXPECT_THAT(output1, testing::HasSubstr(block3));

        EXPECT_THAT(output4, testing::HasSubstr(block1));
        EXPECT_THAT(output4, testing::HasSubstr(block2));
        EXPECT_THAT(output4, testing::HasSubstr(block3));
    }

    INSTANTIATE_TEST_SUITE_P(RandomSchedulerTest,
                             RandomSchedulerTest,
                             ::testing::Values(0, 1, 4, 8, 15, 16, 23, 42));

    TEST_F(SchedulerTest, CooperativeSchedulerTest)
    {
        {
            clearOutput();
            auto v = createRegisters(Register::Type::Vector, DataType::Float, 6);
            auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 1);

            auto generator_one = [&]() -> Generator<Instruction> {
                co_yield_(Instruction("v_or_b32", {v[2]}, {v[0], v[1]}, {}, ""));
                // Since this requires a nop, all the instructions from the other stream will be scheduled here, since none of them require nops.
                co_yield_(Instruction("v_mfma_f32_16x16x4f32", {a[0]}, {v[0], v[2], a[0]}, {}, ""));
            };
            auto generator_two = [&]() -> Generator<Instruction> {
                co_yield_(Inst("Instruction 1, Generator 2"));
                co_yield_(Inst("Instruction 2, Generator 2"));
                co_yield_(Inst("Instruction 3, Generator 2"));
                co_yield_(Inst("Instruction 4, Generator 2"));
            };

            std::vector<Generator<Instruction>> generators;
            generators.push_back(generator_one());
            generators.push_back(generator_two());

            std::string expected = R"( v_or_b32 v2, v0, v1
                                    Instruction 1, Generator 2
                                    Instruction 2, Generator 2
                                    Instruction 3, Generator 2
                                    Instruction 4, Generator 2
                                    v_mfma_f32_16x16x4f32 a0, v0, v2, a0
                                    )";

            auto scheduler = Component::GetNew<Scheduling::Scheduler>(
                Scheduling::SchedulerProcedure::Cooperative,
                Scheduling::CostFunction::MinNops,
                m_context);
            m_context->schedule((*scheduler)(generators));
            EXPECT_EQ(NormalizedSource(output(), true), NormalizedSource(expected, true));
        }
        {
            clearOutput();
            auto v = createRegisters(Register::Type::Vector, DataType::Float, 6);
            auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 2);

            auto generator_one = [&]() -> Generator<Instruction> {
                co_yield_(Instruction("v_or_b32", {v[2]}, {v[0], v[1]}, {}, ""));
            };
            auto generator_two = [&]() -> Generator<Instruction> {
                co_yield_(Instruction("unrelated_op_2", {}, {}, {}, ""));
                co_yield_(Instruction("v_mfma_f32_16x16x4f32", {a[0]}, {v[0], v[2], a[0]}, {}, ""));
            };
            auto generator_three = [&]() -> Generator<Instruction> {
                co_yield_(Instruction("unrelated_op_3", {}, {}, {}, ""));
                co_yield_(Instruction("v_mfma_f32_16x16x4f32", {a[1]}, {v[0], v[2], a[1]}, {}, ""));
                //This instruction will be the first mfma, because of momentum in the coop scheduler.
            };

            std::vector<Generator<Instruction>> generators;
            generators.push_back(generator_one());
            generators.push_back(generator_two());
            generators.push_back(generator_three());

            std::string expected = R"( v_or_b32 v2, v0, v1
                                       unrelated_op_2
                                       unrelated_op_3
                                       v_mfma_f32_16x16x4f32 a1, v0, v2, a1
                                       v_mfma_f32_16x16x4f32 a0, v0, v2, a0
                                    )";

            auto scheduler = Component::GetNew<Scheduling::Scheduler>(
                Scheduling::SchedulerProcedure::Cooperative,
                Scheduling::CostFunction::MinNops,
                m_context);
            m_context->schedule((*scheduler)(generators));
            EXPECT_EQ(NormalizedSource(output(), true), NormalizedSource(expected, true));
        }
        {
            clearOutput();
            auto mfma_v = createRegisters(Register::Type::Vector, DataType::Float, 16);
            auto or_v   = createRegisters(Register::Type::Vector, DataType::Float, 4);

            auto generator_one = [&]() -> Generator<Instruction> {
                std::string comment = "stream1";
                co_yield_(Instruction("v_mfma_f32_32x32x1f32",
                                      {mfma_v[0]},
                                      {mfma_v[1], mfma_v[2], mfma_v[3]},
                                      {},
                                      comment));
                co_yield_(Instruction("v_mfma_f32_32x32x1f32",
                                      {mfma_v[4]},
                                      {mfma_v[5], mfma_v[6], mfma_v[7]},
                                      {},
                                      comment));
                co_yield_(Instruction("v_mfma_f32_32x32x1f32",
                                      {mfma_v[8]},
                                      {mfma_v[9], mfma_v[10], mfma_v[11]},
                                      {},
                                      comment));
            };
            auto generator_two = [&]() -> Generator<Instruction> {
                std::string comment = "stream2";
                co_yield_(Instruction("unrelated_op_2", {}, {}, {}, comment));
                co_yield_(Instruction("v_or_b32", {or_v[0]}, {mfma_v[0], mfma_v[1]}, {}, comment));
                co_yield_(Instruction("unrelated_op_3", {}, {}, {}, comment));
                co_yield_(Instruction("v_or_b32", {or_v[1]}, {mfma_v[8], mfma_v[9]}, {}, comment));
            };
            auto generator_three = [&]() -> Generator<Instruction> {
                std::string comment = "stream3";
                co_yield_(Instruction("unrelated_op_4", {}, {}, {}, comment));
                co_yield_(Instruction("v_mfma_f32_32x32x1f32",
                                      {mfma_v[12]},
                                      {mfma_v[13], mfma_v[14], mfma_v[15]},
                                      {},
                                      comment));
                co_yield_(Instruction("v_or_b32", {or_v[2]}, {mfma_v[4], mfma_v[5]}, {}, comment));
                co_yield_(Instruction("unrelated_op_5", {}, {}, {}, comment));
                co_yield_(
                    Instruction("v_or_b32", {or_v[3]}, {mfma_v[12], mfma_v[13]}, {}, comment));
            };

            std::vector<Generator<Instruction>> generators;
            generators.push_back(generator_one());
            generators.push_back(generator_two());
            generators.push_back(generator_three());

            std::string expected = R"(
                        v_mfma_f32_32x32x1f32 v0, v1, v2, v3 // stream1
                        v_mfma_f32_32x32x1f32 v4, v5, v6, v7 // stream1
                        v_mfma_f32_32x32x1f32 v8, v9, v10, v11 // stream1
                        unrelated_op_2 // stream2
                        unrelated_op_4 // stream3
                        v_mfma_f32_32x32x1f32 v12, v13, v14, v15 // stream3
                        s_nop 13
                        v_or_b32 v16, v0, v1 // stream2
                        // Wait state hazard: XDL Write Hazard
                        unrelated_op_3 // stream2
                        v_or_b32 v17, v8, v9 // stream2
                        v_or_b32 v18, v4, v5 // stream3
                        unrelated_op_5 // stream3
                        v_or_b32 v19, v12, v13 // stream3
                                    )";

            auto scheduler = Component::GetNew<Scheduling::Scheduler>(
                Scheduling::SchedulerProcedure::Cooperative,
                Scheduling::CostFunction::MinNops,
                m_context);
            m_context->schedule((*scheduler)(generators));
            EXPECT_EQ(NormalizedSource(output(), true), NormalizedSource(expected, true));
        }
    }

    TEST_F(SchedulerTest, PriorityCooperativeUniformSchedulerTest)
    {
        //The result of using the Sequential scheduler and the coop or priority scheduler with uniform cost should be the same.
        std::string seqOutput;
        std::string coopUniformOutput;
        std::string priorityUniformOutput;

        auto v = createRegisters(Register::Type::Vector, DataType::Float, 6);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 1);

        auto gen1 = [&]() -> Generator<Instruction> {
            co_yield_(Instruction("v_or_b32", {v[2]}, {v[0], v[1]}, {}, ""));
            co_yield_(Instruction("v_mfma_f32_16x16x4f32", {a[0]}, {v[0], v[2], a[0]}, {}, ""));
        };
        auto gen2 = [&]() -> Generator<Instruction> {
            co_yield_(Inst("Instruction 1, Generator 2"));
            co_yield_(Inst("Instruction 2, Generator 2"));
            co_yield_(Inst("Instruction 3, Generator 2"));
            co_yield_(Inst("Instruction 4, Generator 2"));
        };
        auto gen3 = [&]() -> Generator<Instruction> {
            co_yield_(Instruction("v_mfma_f32_32x32x1f32", {v[0]}, {v[1], v[2], v[3]}, {}, ""));
        };
        auto gen4 = [&]() -> Generator<Instruction> {
            co_yield_(Instruction("unrelated_op_2", {}, {}, {}, ""));
            co_yield_(Instruction("v_or_b32", {v[4]}, {v[0], v[1]}, {}, ""));
        };
        auto gen5 = [&]() -> Generator<Instruction> {
            co_yield_(Instruction("unrelated_op_3", {}, {}, {}, ""));
            co_yield_(Instruction("v_or_b32", {v[4]}, {v[0], v[1]}, {}, ""));
        };

        {
            clearOutput();
            std::vector<Generator<Instruction>> generators;
            generators.push_back(gen1());
            generators.push_back(gen2());
            generators.push_back(gen3());
            generators.push_back(gen4());
            generators.push_back(gen5());

            auto scheduler = Component::GetNew<Scheduling::Scheduler>(
                Scheduling::SchedulerProcedure::Cooperative,
                Scheduling::CostFunction::Uniform,
                m_context);
            m_context->schedule((*scheduler)(generators));
            coopUniformOutput = output();
        }

        {
            clearOutput();
            std::vector<Generator<Instruction>> generators;
            generators.push_back(gen1());
            generators.push_back(gen2());
            generators.push_back(gen3());
            generators.push_back(gen4());
            generators.push_back(gen5());

            auto scheduler = Component::GetNew<Scheduling::Scheduler>(
                Scheduling::SchedulerProcedure::Sequential,
                Scheduling::CostFunction::None,
                m_context);
            m_context->schedule((*scheduler)(generators));
            seqOutput = output();
        }

        {
            clearOutput();
            std::vector<Generator<Instruction>> generators;
            generators.push_back(gen1());
            generators.push_back(gen2());
            generators.push_back(gen3());
            generators.push_back(gen4());
            generators.push_back(gen5());

            auto scheduler
                = Component::GetNew<Scheduling::Scheduler>(Scheduling::SchedulerProcedure::Priority,
                                                           Scheduling::CostFunction::Uniform,
                                                           m_context);
            m_context->schedule((*scheduler)(generators));
            priorityUniformOutput = output();
        }

        EXPECT_EQ(NormalizedSource(seqOutput, true), NormalizedSource(coopUniformOutput, true));

        EXPECT_EQ(NormalizedSource(seqOutput, true), NormalizedSource(priorityUniformOutput, true));
    }

    TEST_F(SchedulerTest, PrioritySchedulerTest)
    {
        {
            clearOutput();
            auto v = createRegisters(Register::Type::Vector, DataType::Float, 6);
            auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 1);

            auto generator_one = [&]() -> Generator<Instruction> {
                co_yield_(Instruction("v_or_b32", {v[2]}, {v[0], v[1]}, {}, ""));
                // Since this requires a nop, instructions will be scheduled from the other stream until nops are no longer needed, then this next instruction will be scheduled.
                co_yield_(Instruction("v_mfma_f32_16x16x4f32", {a[0]}, {v[0], v[2], a[0]}, {}, ""));
            };
            auto generator_two = [&]() -> Generator<Instruction> {
                co_yield_(Inst("Instruction 1, Generator 2"));
                co_yield_(Inst("Instruction 2, Generator 2"));
                co_yield_(Inst("Instruction 3, Generator 2"));
                co_yield_(Inst("Instruction 4, Generator 2"));
            };

            std::vector<Generator<Instruction>> generators;
            generators.push_back(generator_one());
            generators.push_back(generator_two());

            std::string expected = R"( v_or_b32 v2, v0, v1
                                    Instruction 1, Generator 2
                                    Instruction 2, Generator 2
                                    v_mfma_f32_16x16x4f32 a0, v0, v2, a0
                                    Instruction 3, Generator 2
                                    Instruction 4, Generator 2
                                    )";

            auto scheduler
                = Component::GetNew<Scheduling::Scheduler>(Scheduling::SchedulerProcedure::Priority,
                                                           Scheduling::CostFunction::MinNops,
                                                           m_context);
            m_context->schedule((*scheduler)(generators));
            EXPECT_EQ(NormalizedSource(output(), true), NormalizedSource(expected, true));
        }

        {
            clearOutput();
            auto mfma_v = createRegisters(Register::Type::Vector, DataType::Float, 16);
            auto or_v   = createRegisters(Register::Type::Vector, DataType::Float, 4);

            auto generator_one = [&]() -> Generator<Instruction> {
                std::string comment = "stream1";
                co_yield_(Instruction("v_mfma_f32_32x32x1f32",
                                      {mfma_v[0]},
                                      {mfma_v[1], mfma_v[2], mfma_v[3]},
                                      {},
                                      comment));
                co_yield_(Instruction("unrelated_op_2", {}, {}, {}, comment));
                co_yield_(Instruction("v_or_b32", {or_v[0]}, {mfma_v[0], mfma_v[1]}, {}, comment));
                co_yield_(Instruction("v_mfma_f32_32x32x1f32",
                                      {mfma_v[4]},
                                      {mfma_v[5], mfma_v[6], mfma_v[7]},
                                      {},
                                      comment));
                co_yield_(Instruction("unrelated_op_3", {}, {}, {}, comment));
                co_yield_(Instruction("v_or_b32", {or_v[1]}, {mfma_v[4], mfma_v[5]}, {}, comment));
                co_yield_(Instruction("v_mfma_f32_32x32x1f32",
                                      {mfma_v[8]},
                                      {mfma_v[9], mfma_v[10], mfma_v[11]},
                                      {},
                                      comment));
                co_yield_(Instruction("unrelated_op_4", {}, {}, {}, comment));
                co_yield_(Instruction("v_or_b32", {or_v[2]}, {mfma_v[8], mfma_v[9]}, {}, comment));
            };
            auto generator_two = [&]() -> Generator<Instruction> {
                std::string comment = "stream2";
                co_yield_(Instruction("unrelated_op_5", {}, {}, {}, comment));
                co_yield_(Instruction("v_mfma_f32_32x32x1f32",
                                      {mfma_v[12]},
                                      {mfma_v[13], mfma_v[14], mfma_v[15]},
                                      {},
                                      comment));
                co_yield_(Instruction("v_or_b32", {or_v[2]}, {mfma_v[4], mfma_v[5]}, {}, comment));
                co_yield_(Instruction("unrelated_op_6", {}, {}, {}, comment));
                co_yield_(
                    Instruction("v_or_b32", {or_v[3]}, {mfma_v[12], mfma_v[13]}, {}, comment));
            };
            auto generator_three = [&]() -> Generator<Instruction> {
                std::string comment = "stream3";
                co_yield_(Instruction("unrelated_op_7", {}, {}, {}, comment));
                co_yield_(Instruction("unrelated_op_8", {}, {}, {}, comment));
                co_yield_(Instruction("unrelated_op_9", {}, {}, {}, comment));
                co_yield_(Instruction("unrelated_op_10", {}, {}, {}, comment));
            };

            std::vector<Generator<Instruction>> generators;
            generators.push_back(generator_one());
            generators.push_back(generator_two());
            generators.push_back(generator_three());

            std::string expected = R"(
                v_mfma_f32_32x32x1f32 v0, v1, v2, v3 // stream1
                unrelated_op_2 // stream1
                unrelated_op_5 // stream2
                v_mfma_f32_32x32x1f32 v12, v13, v14, v15 // stream2
                v_or_b32 v18, v4, v5 // stream2
                unrelated_op_6 // stream2
                unrelated_op_7 // stream3
                unrelated_op_8 // stream3
                unrelated_op_9 // stream3
                unrelated_op_10 // stream3
                s_nop 9
                v_or_b32 v16, v0, v1 // stream1
                // Wait state hazard: XDL Write Hazard
                v_mfma_f32_32x32x1f32 v4, v5, v6, v7 // stream1
                unrelated_op_3 // stream1
                v_or_b32 v19, v12, v13 // stream2
                s_nop 0xf
                s_nop 0
                v_or_b32 v17, v4, v5 // stream1
                // Wait state hazard: XDL Write Hazard
                v_mfma_f32_32x32x1f32 v8, v9, v10, v11 // stream1
                unrelated_op_4 // stream1
                s_nop 0xf
                s_nop 1
                v_or_b32 v18, v8, v9 // stream1
                // Wait state hazard: XDL Write Hazard
                )";

            auto scheduler
                = Component::GetNew<Scheduling::Scheduler>(Scheduling::SchedulerProcedure::Priority,
                                                           Scheduling::CostFunction::MinNops,
                                                           m_context);
            m_context->schedule((*scheduler)(generators));
            EXPECT_EQ(NormalizedSource(output(), true), NormalizedSource(expected, true));
        }
    }

    struct LockCheckSchedulerTest : public SchedulerTest,
                                    public testing::WithParamInterface<
                                        std::tuple<Scheduling::SchedulerProcedure, std::string>>
    {
    protected:
        GPUArchitectureTarget targetArchitecture() override
        {
            return {GPUArchitectureGFX::GFX90A};
        }
    };

    TEST_P(LockCheckSchedulerTest, LockCheckTest)
    {
#ifdef NDEBUG
        GTEST_SKIP() << "Skipping LockCheckTest in release mode.";
#endif

        auto gen = [](std::string inst, bool lock) -> Generator<Instruction> {
            if(lock)
            {
                co_yield_(Instruction::Lock(Scheduling::Dependency::SCC));
            }
            auto label = Register::Value::Label("testLabel");
            co_yield_(Instruction(inst, {}, {label}, {}, ""));
            if(lock)
            {
                co_yield_(Instruction::Unlock());
            }
        };

        {
            std::vector<Generator<Instruction>> gens;
            gens.push_back(testGeneratorWithComments(true));
            gens.push_back(gen(std::get<1>(GetParam()), false));
            auto scheduler = Component::GetNew<Scheduling::Scheduler>(
                std::get<0>(GetParam()), Scheduling::CostFunction::MinNops, m_context);
            EXPECT_THROW({ m_context->schedule((*scheduler)(gens)); }, FatalError);
        }

        {
            std::vector<Generator<Instruction>> gens;
            gens.push_back(testGeneratorWithComments(true));
            gens.push_back(gen(std::get<1>(GetParam()), true));
            auto scheduler = Component::GetNew<Scheduling::Scheduler>(
                std::get<0>(GetParam()), Scheduling::CostFunction::MinNops, m_context);
            EXPECT_NO_THROW({ m_context->schedule((*scheduler)(gens)); });
        }
    }

    INSTANTIATE_TEST_SUITE_P(
        LockCheckSchedulerTest,
        LockCheckSchedulerTest,
        testing::Combine(::testing::Values(Scheduling::SchedulerProcedure::Sequential,
                                           Scheduling::SchedulerProcedure::RoundRobin,
                                           Scheduling::SchedulerProcedure::Random,
                                           Scheduling::SchedulerProcedure::Cooperative,
                                           Scheduling::SchedulerProcedure::Priority),
                         ::testing::Values("s_branch",
                                           "s_cbranch_scc0"))); //, "s_addc_u32", "s_subb_u32")));
}

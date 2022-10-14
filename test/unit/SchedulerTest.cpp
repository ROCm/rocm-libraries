
#include <gmock/gmock.h>
#include <gtest/gtest.h>

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
    class SchedulerTest : public GenericContextFixture
    {
    };

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

        auto scheduler = Component::Get<Scheduling::Scheduler>(
            Scheduling::SchedulerProcedure::Sequential, m_context);
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

        auto scheduler = Component::Get<Scheduling::Scheduler>(
            Scheduling::SchedulerProcedure::RoundRobin, m_context);
        m_context->schedule((*scheduler)(generators));
        EXPECT_EQ(NormalizedSource(output(), true), NormalizedSource(expected, true));
    }

    // Can be run with the --gtest_also_run_disabled_tests option
    TEST_F(SchedulerTest, DISABLED_SchedulerWaitStressTest)
    {
        auto generator = []() -> Generator<Instruction> {
            for(size_t i = 0; i < 1000000; i++)
            {
                co_yield_(Instruction::Wait(WaitCount::VMCnt(1, "Comment")));
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
        auto scheduler = Component::Get<Scheduling::Scheduler>(
            Scheduling::SchedulerProcedure::RoundRobin, m_context);

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
        auto scheduler = Component::Get<Scheduling::Scheduler>(
            Scheduling::SchedulerProcedure::RoundRobin, m_context);

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
            Scheduling::SchedulerProcedure::RoundRobin, m_context);
        auto schedulerB = Component::GetNew<Scheduling::Scheduler>(
            Scheduling::SchedulerProcedure::RoundRobin, m_context);
        auto schedulerC = Component::GetNew<Scheduling::Scheduler>(
            Scheduling::SchedulerProcedure::RoundRobin, m_context);

        std::vector<Generator<Instruction>> a_sequences;
        std::vector<Generator<Instruction>> b_sequences;
        std::vector<Generator<Instruction>> c_sequences;

        auto opB = [&]() -> Generator<Instruction> {
            co_yield(Inst("(C) Op B Begin"));
            co_yield(Inst("(C) Op B Instruction"));
            co_yield(Inst("(C) Op B End"));
        };

        auto ifBlock = [&]() -> Generator<Instruction> {
            co_yield(
                Inst("(C) If Begin").lock(Scheduling::Dependency::SCC, "(C) Scheduler C Lock"));

            EXPECT_EQ(schedulerA->getLockState().getDependency(), Scheduling::Dependency::Branch);
            EXPECT_EQ(schedulerB->getLockState().getDependency(), Scheduling::Dependency::VCC);
            EXPECT_EQ(schedulerC->getLockState().getDependency(), Scheduling::Dependency::SCC);

            co_yield(Inst("+++ Scheduler A Lock Depth: "
                          + std::to_string(schedulerA->getLockState().getLockDepth())));
            co_yield(Inst("+++ Scheduler B Lock Depth: "
                          + std::to_string(schedulerB->getLockState().getLockDepth())));
            co_yield(Inst("+++ Scheduler C Lock Depth: "
                          + std::to_string(schedulerC->getLockState().getLockDepth())));
            co_yield(Inst("(C) If Instruction"));
            co_yield(Inst("(C) If End").unlock("(C) Scheduler C Unlock"));

            EXPECT_EQ(schedulerA->getLockState().getDependency(), Scheduling::Dependency::Branch);
            EXPECT_EQ(schedulerB->getLockState().getDependency(), Scheduling::Dependency::VCC);
            EXPECT_EQ(schedulerC->getLockState().getDependency(), Scheduling::Dependency::None);

            co_yield(Inst("+++ Scheduler A Lock Depth: "
                          + std::to_string(schedulerA->getLockState().getLockDepth())));
            co_yield(Inst("+++ Scheduler B Lock Depth: "
                          + std::to_string(schedulerB->getLockState().getLockDepth())));
            co_yield(Inst("+++ Scheduler C Lock Depth: "
                          + std::to_string(schedulerC->getLockState().getLockDepth())));
        };

        c_sequences.push_back(opB());
        c_sequences.push_back(ifBlock());

        auto unroll0 = [&]() -> Generator<Instruction> {
            co_yield(Inst("(B) Unroll 0 Begin")
                         .lock(Scheduling::Dependency::VCC, "(B) Scheduler B Lock"));

            EXPECT_EQ(schedulerA->getLockState().getDependency(), Scheduling::Dependency::Branch);
            EXPECT_EQ(schedulerB->getLockState().getDependency(), Scheduling::Dependency::VCC);

            co_yield(Inst("+++ Scheduler A Lock Depth: "
                          + std::to_string(schedulerA->getLockState().getLockDepth())));
            co_yield(Inst("+++ Scheduler B Lock Depth: "
                          + std::to_string(schedulerB->getLockState().getLockDepth())));
            co_yield((*schedulerC)(c_sequences));
            co_yield(Inst("(B) Unroll 0 End")).unlock("(B) Scheduler B Unlock");

            EXPECT_EQ(schedulerA->getLockState().getDependency(), Scheduling::Dependency::Branch);
            EXPECT_EQ(schedulerB->getLockState().getDependency(), Scheduling::Dependency::None);

            co_yield(Inst("+++ Scheduler A Lock Depth: "
                          + std::to_string(schedulerA->getLockState().getLockDepth())));
            co_yield(Inst("+++ Scheduler B Lock Depth: "
                          + std::to_string(schedulerB->getLockState().getLockDepth())));
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

            EXPECT_EQ(schedulerA->getLockState().getDependency(), Scheduling::Dependency::Branch);

            co_yield(Inst("+++ Scheduler A Lock Depth: "
                          + std::to_string(schedulerA->getLockState().getLockDepth())));
            co_yield((*schedulerB)(b_sequences));
            co_yield(Inst("(A) For Loop End").unlock("(A) Scheduler A Unlock)"));
            co_yield(Inst("+++ Scheduler A Lock Depth: "
                          + std::to_string(schedulerA->getLockState().getLockDepth())));

            EXPECT_EQ(schedulerA->getLockState().getDependency(), Scheduling::Dependency::None);
        };

        a_sequences.push_back(opA());
        a_sequences.push_back(forloop());

        m_context->schedule((*schedulerA)(a_sequences));

        // TODO: Fix when m_comments.push_back() issue is fixed
        std::string expected = R"( (A) Op A Begin
                                    (A) For Loop Begin
                                    // (A) Scheduler A Lock
                                    // (A) Scheduler A Lock
                                    +++ Scheduler A Lock Depth: 1
                                    (B) Unroll 0 Begin
                                    // (B) Scheduler B Lock
                                    // (B) Scheduler B Lock
                                    +++ Scheduler A Lock Depth: 2
                                    +++ Scheduler B Lock Depth: 1
                                    (C) Op B Begin
                                    (C) If Begin
                                    // (C) Scheduler C Lock
                                    // (C) Scheduler C Lock
                                    +++ Scheduler A Lock Depth: 3
                                    +++ Scheduler B Lock Depth: 2
                                    +++ Scheduler C Lock Depth: 1
                                    (C) If Instruction
                                    (C) If End
                                    // (C) Scheduler C Unlock
                                    // (C) Scheduler C Unlock
                                    (C) Op B Instruction
                                    +++ Scheduler A Lock Depth: 2
                                    (C) Op B End
                                    +++ Scheduler B Lock Depth: 1
                                    +++ Scheduler C Lock Depth: 0
                                    (B) Unroll 0 End
                                    // (B) Scheduler B Unlock
                                    // (B) Scheduler B Unlock
                                    (B) Unroll 1 Begin
                                    +++ Scheduler A Lock Depth: 1
                                    (B) Unroll 1 Instruction
                                    +++ Scheduler B Lock Depth: 0
                                    (B) Unroll 1 End
                                    (A) For Loop End
                                    // (A) Scheduler A Unlock)
                                    // (A) Scheduler A Unlock)
                                    (A) Op A Instruction
                                    +++ Scheduler A Lock Depth: 0
                                    (A) Op A End
                                )";

        EXPECT_EQ(NormalizedSource(output(), true), NormalizedSource(expected, true));
    }
}

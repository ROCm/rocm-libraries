
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Scheduling/Scheduler.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"

using namespace rocRoller;

namespace rocRollerTest
{
    class SchedulerTest : public GenericContextFixture
    {
    };

    TEST_F(SchedulerTest, SequentialSchedulerTest)
    {
        auto generator_one = [&]() -> Generator<Instruction> {
            co_yield_(Instruction::Comment("Instruction 1, Generator 1"));
            co_yield_(Instruction::Comment("Instruction 2, Generator 1"));
            co_yield_(Instruction::Comment("Instruction 3, Generator 1"));
        };
        auto generator_two = [&]() -> Generator<Instruction> {
            co_yield_(Instruction::Comment("Instruction 1, Generator 2"));
            co_yield_(Instruction::Comment("Instruction 2, Generator 2"));
            co_yield_(Instruction::Comment("Instruction 3, Generator 2"));
            co_yield_(Instruction::Comment("Instruction 4, Generator 2"));
        };

        std::vector<Generator<Instruction>> generators;
        generators.push_back(generator_one());
        generators.push_back(generator_two());

        std::string expected = R"( // Instruction 1, Generator 1
                                    // Instruction 2, Generator 1
                                    // Instruction 3, Generator 1
                                    // Instruction 1, Generator 2
                                    // Instruction 2, Generator 2
                                    // Instruction 3, Generator 2
                                    // Instruction 4, Generator 2
                                )";

        auto scheduler = Component::Get<Scheduling::Scheduler>(
            Scheduling::SchedulerProcedure::Sequential, m_context);
        m_context->schedule((*scheduler)(generators));
        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(expected));
    }

    TEST_F(SchedulerTest, RoundRobinSchedulerTest)
    {
        auto generator_one = [&]() -> Generator<Instruction> {
            co_yield_(Instruction::Comment("Instruction 1, Generator 1"));
            co_yield_(Instruction::Comment("Instruction 2, Generator 1"));
            co_yield_(Instruction::Comment("Instruction 3, Generator 1"));
        };
        auto generator_two = [&]() -> Generator<Instruction> {
            co_yield_(Instruction::Comment("Instruction 1, Generator 2"));
            co_yield_(Instruction::Comment("Instruction 2, Generator 2"));
            co_yield_(Instruction::Comment("Instruction 3, Generator 2"));
            co_yield_(Instruction::Comment("Instruction 4, Generator 2"));
        };

        std::vector<Generator<Instruction>> generators;
        generators.push_back(generator_one());
        generators.push_back(generator_two());

        std::string expected = R"( // Instruction 1, Generator 1
                                    // Instruction 1, Generator 2
                                    // Instruction 2, Generator 1
                                    // Instruction 2, Generator 2
                                    // Instruction 3, Generator 1
                                    // Instruction 3, Generator 2
                                    // Instruction 4, Generator 2
                                )";

        auto scheduler = Component::Get<Scheduling::Scheduler>(
            Scheduling::SchedulerProcedure::RoundRobin, m_context);
        m_context->schedule((*scheduler)(generators));
        EXPECT_EQ(NormalizedSource(output()), NormalizedSource(expected));
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
}

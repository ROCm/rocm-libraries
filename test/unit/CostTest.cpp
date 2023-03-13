
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Scheduling/Costs/Cost.hpp>
#include <rocRoller/Scheduling/Costs/MinNopsCost.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"

#define Inst(opcode) Instruction(opcode, {}, {}, {}, "")

using namespace rocRoller;

namespace rocRollerTest
{
    struct CostTest : public GenericContextFixture
    {
        std::string targetArchitecture()
        {
            return "gfx90a";
        }
    };

    TEST_F(CostTest, WaitCntNopCostTest)
    {
        auto cost
            = Component::Get<Scheduling::Cost>(Scheduling::CostProcedure::WaitCntNop, m_context);
        {
            auto status = Scheduling::InstructionStatus();
            EXPECT_NEAR(cost->cost(status), 0.0, 1e-12);
        }
        {
            auto status = Scheduling::InstructionStatus::Nops(1);
            EXPECT_GT(cost->cost(status), 0.0);
        }
        {
            auto status = Scheduling::InstructionStatus::Wait(WaitCount::VMCnt(3));
            EXPECT_GT(cost->cost(status), 0.0);
        }
        {
            auto status = Scheduling::InstructionStatus::Wait(WaitCount::EXPCnt(3));
            EXPECT_GT(cost->cost(status), 0.0);
        }
        {
            auto status = Scheduling::InstructionStatus::Wait(WaitCount::LGKMCnt(3));
            EXPECT_GT(cost->cost(status), 0.0);
        }
    }

    TEST_F(CostTest, MinNopsCostTest)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 6);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 2);

        auto generator_pre = [&]() -> Generator<Instruction> {
            co_yield_(Instruction("v_or_b32", {v[2]}, {v[0], v[1]}, {}, ""));
        };
        auto generator_one = [&]() -> Generator<Instruction> {
            co_yield_(Instruction("unrelated_op_2", {}, {}, {}, ""));
            co_yield_(Instruction("v_mfma_f32_16x16x4f32", {a[0]}, {v[0], v[2], a[0]}, {}, ""));
        };
        auto generator_two = [&]() -> Generator<Instruction> {
            co_yield_(Instruction("v_mfma_f32_16x16x4f32", {a[1]}, {v[0], v[2], a[1]}, {}, ""));
        };

        auto cost = Component::Get<Scheduling::Cost>(Scheduling::CostProcedure::MinNops, m_context);

        {
            std::vector<Generator<Instruction>> generators;
            generators.push_back(generator_one());
            generators.push_back(generator_two());

            std::vector<Generator<Instruction>::iterator> iterators;

            size_t n = generators.size();

            iterators.reserve(n);
            for(auto& seq : generators)
            {
                iterators.emplace_back(seq.begin());
            }

            m_context->schedule(generator_pre());

            auto result = (*cost)(iterators);
            EXPECT_EQ(result.size(), generators.size());
            EXPECT_EQ(std::get<0>(result[0]), 0);
            EXPECT_EQ(std::get<0>(result[1]), 1);
            EXPECT_EQ(std::get<1>(result[0]), 0);
            EXPECT_GT(std::get<1>(result[1]), 0);

            ++iterators[0];

            result = (*cost)(iterators);
            EXPECT_EQ(result.size(), generators.size());
            EXPECT_EQ(std::get<0>(result[0]), 0);
            EXPECT_EQ(std::get<0>(result[1]), 1);
            EXPECT_GT(std::get<1>(result[0]), 0);
            EXPECT_GT(std::get<1>(result[1]), 0);
        }

        {
            std::vector<Generator<Instruction>> generators;
            generators.push_back(generator_two());
            generators.push_back(generator_one());

            std::vector<Generator<Instruction>::iterator> iterators;

            size_t n = generators.size();

            iterators.reserve(n);
            for(auto& seq : generators)
            {
                iterators.emplace_back(seq.begin());
            }

            m_context->schedule(generator_pre());

            auto result = (*cost)(iterators);
            EXPECT_EQ(result.size(), generators.size());
            EXPECT_EQ(std::get<0>(result[0]), 1);
            EXPECT_EQ(std::get<0>(result[1]), 0);
            EXPECT_EQ(std::get<1>(result[0]), 0);
            EXPECT_GT(std::get<1>(result[1]), 0);

            ++iterators[1];

            result = (*cost)(iterators);
            EXPECT_EQ(result.size(), generators.size());
            EXPECT_EQ(std::get<0>(result[0]), 0);
            EXPECT_EQ(std::get<0>(result[1]), 1);
            EXPECT_GT(std::get<1>(result[0]), 0);
            EXPECT_GT(std::get<1>(result[1]), 0);
        }
    }

    TEST_F(CostTest, UniformCostTest)
    {
        auto v = createRegisters(Register::Type::Vector, DataType::Float, 6);
        auto a = createRegisters(Register::Type::Accumulator, DataType::Float, 2);

        auto generator_pre = [&]() -> Generator<Instruction> {
            co_yield_(Instruction("v_or_b32", {v[2]}, {v[0], v[1]}, {}, ""));
        };
        auto generator_one = [&]() -> Generator<Instruction> {
            co_yield_(Instruction("unrelated_op_2", {}, {}, {}, ""));
            co_yield_(Instruction("v_mfma_f32_16x16x4f32", {a[0]}, {v[0], v[2], a[0]}, {}, ""));
        };
        auto generator_two = [&]() -> Generator<Instruction> {
            co_yield_(Instruction("v_mfma_f32_16x16x4f32", {a[1]}, {v[0], v[2], a[1]}, {}, ""));
        };

        auto cost = Component::Get<Scheduling::Cost>(Scheduling::CostProcedure::Uniform, m_context);

        std::vector<Generator<Instruction>> generators;
        generators.push_back(generator_one());
        generators.push_back(generator_two());

        std::vector<Generator<Instruction>::iterator> iterators;

        size_t n = generators.size();

        iterators.reserve(n);
        for(auto& seq : generators)
        {
            iterators.emplace_back(seq.begin());
        }

        m_context->schedule(generator_pre());

        auto result = (*cost)(iterators);

        EXPECT_EQ(result.size(), n);
        for(int i = 0; i < n; i++)
        {
            EXPECT_EQ(std::get<0>(result[i]), i);
            EXPECT_EQ(std::get<1>(result[i]), 0.0);
        }
    }

    TEST_F(CostTest, NoneCostTest)
    {
        std::shared_ptr<Scheduling::Cost> cost;
        EXPECT_THROW(cost
                     = Component::Get<Scheduling::Cost>(Scheduling::CostProcedure::None, m_context),
                     FatalError);
    }
}

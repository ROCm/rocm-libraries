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
#include <rocRoller/Scheduling/Costs/Cost.hpp>
#include <rocRoller/Scheduling/Costs/LinearWeightedCost.hpp>
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
        GPUArchitectureTarget targetArchitecture() override
        {
            return {GPUArchitectureGFX::GFX90A};
        }
    };

    TEST_F(CostTest, WaitCntNopCostTest)
    {
        Instruction inst;
        auto        cost
            = Component::Get<Scheduling::Cost>(Scheduling::CostFunction::WaitCntNop, m_context);
        auto const& arch = m_context->targetArchitecture();
        {
            auto status = Scheduling::InstructionStatus();
            EXPECT_NEAR(cost->cost(inst, status), 0.0, 1e-12);
        }
        {
            auto status = Scheduling::InstructionStatus::Nops(1);
            EXPECT_GT(cost->cost(inst, status), 0.0);
        }
        {
            auto status = Scheduling::InstructionStatus::Wait(WaitCount::LoadCnt(arch, 3));
            EXPECT_GT(cost->cost(inst, status), 0.0);
        }
        {
            auto status = Scheduling::InstructionStatus::Wait(WaitCount::EXPCnt(arch, 3));
            EXPECT_GT(cost->cost(inst, status), 0.0);
        }
        {
            auto status = Scheduling::InstructionStatus::Wait(WaitCount::KMCnt(arch, 3));
            EXPECT_GT(cost->cost(inst, status), 0.0);
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

        auto cost = Component::Get<Scheduling::Cost>(Scheduling::CostFunction::MinNops, m_context);

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

        auto cost = Component::Get<Scheduling::Cost>(Scheduling::CostFunction::Uniform, m_context);

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
                     = Component::Get<Scheduling::Cost>(Scheduling::CostFunction::None, m_context),
                     FatalError);
    }

    TEST_F(CostTest, NonexistentSchedulerWeightsFile)
    {
        Component::ComponentFactoryBase::ClearAllCaches();
        Settings::getInstance()->set(Settings::SchedulerWeights, "/dev/null/foo");
        EXPECT_THROW(
            Component::Get<Scheduling::Cost>(Scheduling::CostFunction::LinearWeighted, m_context),
            FatalError);
    }
}

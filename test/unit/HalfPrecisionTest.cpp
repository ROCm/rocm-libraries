
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Context.hpp>

#include "GPUContextFixture.hpp"

using namespace rocRoller;

namespace rocRollerTest
{
    class HalfPrecisionTest : public CurrentGPUContextFixture
    {
    };

    void genHalfPrecisionCopy(std::shared_ptr<rocRoller::Context> m_context, int N)
    {
        auto k = m_context->kernel();

        k->setKernelDimensions(1);
        auto command = std::make_shared<Command>();

        auto result_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Half, PointerType::PointerGlobal}));
        auto a_exp = std::make_shared<Expression::Expression>(
            command->allocateArgument({DataType::Half, PointerType::PointerGlobal}));

        auto one  = std::make_shared<Expression::Expression>(1u);
        auto zero = std::make_shared<Expression::Expression>(0u);

        k->addArgument({"result",
                        {DataType::Half, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        result_exp});
        k->addArgument(
            {"a", {DataType::Half, PointerType::PointerGlobal}, DataDirection::ReadOnly, a_exp});

        k->setWorkgroupSize({1, 1, 1});
        k->setWorkitemCount({one, one, one});
        k->setDynamicSharedMemBytes(zero);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr s_result, s_a;
            co_yield m_context->argLoader()->getValue("result", s_result);
            co_yield m_context->argLoader()->getValue("a", s_a);

            auto v_result = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Raw32, 2);

            auto v_ptr = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Raw32, 2);

            auto v_a = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::Int32, 1);

            co_yield v_a->allocate();
            co_yield v_ptr->allocate();
            co_yield v_result->allocate();

            co_yield m_context->copier()->copy(v_result, s_result, "Move pointer.");

            co_yield m_context->copier()->copy(v_ptr, s_a, "Move pointer.");

            for(int i = 0; i < N / 2; i++)
            {
                co_yield m_context->mem()->loadFlat(v_a, v_ptr, std::to_string(i * 4), 4);
                co_yield m_context->mem()->storeFlat(v_result, v_a, std::to_string(i * 4), 4);
            }
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());
    }

    void executeHalfPrecsionCopy(std::shared_ptr<rocRoller::Context> m_context, int N)
    {
        genHalfPrecisionCopy(m_context, N);

        CommandKernel     commandKernel(m_context);
        RandomGenerator   random(314273u);
        auto              a = random.vector<Half>(N, -1.0, 1.0);
        std::vector<Half> result(N);

        auto d_a      = make_shared_device(a);
        auto d_result = make_shared_device<Half>(N);

        KernelArguments runtimeArgs;
        runtimeArgs.append("result", d_result.get());
        runtimeArgs.append("a", d_a.get());

        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        ASSERT_THAT(hipMemcpy(result.data(), d_result.get(), sizeof(Half) * N, hipMemcpyDefault),
                    HasHipSuccess(0));

        for(int i = 0; i < N; i++)
            EXPECT_EQ(result[i], a[i]);
    }

    TEST_F(HalfPrecisionTest, GPU_ExecuteHalfPrecisionCopy)
    {
        executeHalfPrecsionCopy(m_context, 8);
    }
}

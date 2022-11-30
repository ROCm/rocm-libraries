
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <random>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/HIPTimer.hpp>
#include <rocRoller/Utilities/Timer.hpp>

#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "Scheduling/Observers/FileWritingObserver.hpp"
#include "SourceMatcher.hpp"
#include "Utilities.hpp"

using namespace rocRoller;

namespace VectorAddBenchmark
{

    class VectorAddBenchmarkGPU : public CurrentGPUContextFixture,
                                  public ::testing::WithParamInterface<int>
    {
        void SetUp()
        {
            CurrentGPUContextFixture::SetUp();
        }
    };

    void VectorAddGraph(std::shared_ptr<Context> context, size_t nx)
    {
        RandomGenerator random(31415u);

        auto a = random.vector<int>(nx, -100, 100);
        auto b = random.vector<int>(nx, -100, 100);

        auto d_a     = make_shared_device(a);
        auto d_b     = make_shared_device(b);
        auto d_c     = make_shared_device<int>(nx);
        auto d_alpha = make_shared_device<int>();

        int alpha = 22;

        std::vector<int> r(nx), x(nx);

        ASSERT_THAT(hipMemcpy(d_alpha.get(), &alpha, 1 * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        auto command = std::make_shared<Command>();

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Linear(DataType::Int32, 1, 0))); // a
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Linear(DataType::Int32, 1, 1))); // b
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Scalar({DataType::Int32, PointerType::PointerGlobal},
                                                 2))); // alpha

        auto execute = rocRoller::Operations::T_Execute();
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Add(3, 0, 1))); // a + b
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Mul(4, 2, 3))); // alpha * (a + b)

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(execute));

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Store_Linear(1, 4)));

        CommandKernel commandKernel(command, "VectorAdd");

        KernelArguments runtimeArgs;

        runtimeArgs.append("user0", d_a.get());
        runtimeArgs.append("d_a_limit", nx);
        runtimeArgs.append("d_a_size", nx);
        runtimeArgs.append("d_a_stride", (size_t)1);

        runtimeArgs.append("user1", d_b.get());
        runtimeArgs.append("d_b_limit", nx);
        runtimeArgs.append("d_b_size", nx);
        runtimeArgs.append("d_b_stride", (size_t)1);

        runtimeArgs.append("user2", d_alpha.get());

        runtimeArgs.append("user6", d_c.get());
        runtimeArgs.append("d_c_limit", nx);
        runtimeArgs.append("d_c_stride", (size_t)1);

        HIP_TIMER(t_kernel, "VectorAddKernel_Graph");
        commandKernel.launchKernel(runtimeArgs.runtimeArguments());
        HIP_TOC(t_kernel);
        HIP_SYNC(t_kernel);

        ASSERT_THAT(hipMemcpy(r.data(), d_c.get(), nx * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        // reference solution
        for(size_t i = 0; i < nx; ++i)
            x[i] = alpha * (a[i] + b[i]);

        double rnorm = relativeNorm(r, x);

        ASSERT_LT(rnorm, 1.e-12);
    }

    // delete this after graph rearch complete
    void VectorAddGraph2(std::shared_ptr<Context> context, size_t nx)
    {
        RandomGenerator random(31415u);

        auto a = random.vector<int>(nx, -100, 100);
        auto b = random.vector<int>(nx, -100, 100);

        auto d_a     = make_shared_device(a);
        auto d_b     = make_shared_device(b);
        auto d_c     = make_shared_device<int>(nx);
        auto d_alpha = make_shared_device<int>();

        int alpha = 22;

        std::vector<int> r(nx), x(nx);

        ASSERT_THAT(hipMemcpy(d_alpha.get(), &alpha, 1 * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        auto command = std::make_shared<Command>();

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Linear(DataType::Int32, 1, 0))); // a
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Linear(DataType::Int32, 1, 1))); // b
        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Load_Scalar({DataType::Int32, PointerType::PointerGlobal},
                                                 2))); // alpha

        auto execute = rocRoller::Operations::T_Execute();
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Add(3, 0, 1))); // a + b
        execute.addXOp(std::make_shared<rocRoller::Operations::XOp>(
            rocRoller::Operations::E_Mul(4, 2, 3))); // alpha * (a + b)

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(execute));

        command->addOperation(std::make_shared<rocRoller::Operations::Operation>(
            rocRoller::Operations::T_Store_Linear(1, 4)));

        KernelArguments runtimeArgs;

        runtimeArgs.append("user0", d_a.get());
        runtimeArgs.append("d_a_limit", nx);
        runtimeArgs.append("d_a_size", nx);
        runtimeArgs.append("d_a_stride", (size_t)1);

        runtimeArgs.append("user1", d_b.get());
        runtimeArgs.append("d_b_limit", nx);
        runtimeArgs.append("d_b_size", nx);
        runtimeArgs.append("d_b_stride", (size_t)1);

        runtimeArgs.append("user2", d_alpha.get());

        runtimeArgs.append("user6", d_c.get());
        runtimeArgs.append("d_c_limit", nx);
        runtimeArgs.append("d_c_stride", (size_t)1);

        auto NX  = Expression::literal(nx);
        auto one = Expression::literal(1);

        context->kernel()->setKernelDimensions(1);
        context->kernel()->setWorkgroupSize({64, 1, 1});
        context->kernel()->setWorkitemCount({NX, one, one});

        context->kernel()->addCommandArguments(command->getArguments());
        auto kgraph = KernelGraph::translate2(command);
        kgraph      = KernelGraph::lowerLinear(kgraph, context);
        kgraph      = KernelGraph::cleanArguments(kgraph, context->kernel());

        context->schedule(context->kernel()->preamble());
        context->schedule(context->kernel()->prolog());
        context->schedule(KernelGraph::generate(kgraph, context->kernel()));
        context->schedule(context->kernel()->postamble());
        context->schedule(context->kernel()->amdgpu_metadata());

        auto executableKernel = context->instructions()->getExecutableKernel();

        KernelArguments kargs;
        for(auto& arg : context->kernel()->arguments())
        {
            auto value = evaluate(arg.expression, runtimeArgs.runtimeArguments());
            kargs.append(arg.name, value);
        }

        KernelInvocation kinv;
        kinv.workgroupSize    = context->kernel()->workgroupSize();
        kinv.workitemCount[0] = nx;

        executableKernel->executeKernel(kargs, kinv);

        ASSERT_THAT(hipMemcpy(r.data(), d_c.get(), nx * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        // reference solution
        for(size_t i = 0; i < nx; ++i)
            x[i] = alpha * (a[i] + b[i]);

        double rnorm = relativeNorm(r, x);

        ASSERT_LT(rnorm, 1.e-12);
    }

    TEST_P(VectorAddBenchmarkGPU, VectorAddBenchmark_GPU_Graph)
    {
        auto nx = GetParam();

        VectorAddGraph(m_context, nx);

        std::cout << TimerPool::summary() << std::endl;
        std::cout << TimerPool::CSV() << std::endl;
    }

    // delete this after graph rearch complete
    TEST_P(VectorAddBenchmarkGPU, VectorAddBenchmark_GPU_Graph2)
    {
        auto nx = GetParam();

        VectorAddGraph2(m_context, nx);

        std::cout << TimerPool::summary() << std::endl;
        std::cout << TimerPool::CSV() << std::endl;
    }

    INSTANTIATE_TEST_SUITE_P(
        VectorAddBenchmarks,
        VectorAddBenchmarkGPU,
        ::testing::ValuesIn({16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576}));
}

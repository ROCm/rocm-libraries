
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <random>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/Arithmetic.hpp>
#include <rocRoller/CodeGen/Arithmetic/Double.hpp>
#include <rocRoller/CodeGen/Arithmetic/Float.hpp>
#include <rocRoller/CodeGen/Arithmetic/Int32.hpp>
#include <rocRoller/CodeGen/Arithmetic/Int64.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Utilities/HIPTimer.hpp>
#include <rocRoller/Utilities/Timer.hpp>

#include "DataTypes/DataTypes.hpp"
#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "Scheduling/Observers/FileWritingObserver.hpp"
#include "SourceMatcher.hpp"
#include "Utilities/Error.hpp"

using namespace rocRoller;

namespace VectorAddBenchmark
{

    class VectorAddBenchmarkGPU : public CurrentGPUContextFixture,
                                  public ::testing::WithParamInterface<int>
    {
        char* saveAssembly;

        void SetUp()
        {
            // Save environment variable state
            saveAssembly = getenv(rocRoller::ENV_SAVE_ASSEMBLY.c_str());

            // Set the SAVE_ASSEMBLY environment variable
            setenv(rocRoller::ENV_SAVE_ASSEMBLY.c_str(), "1", 1);

            CurrentGPUContextFixture::SetUp();
        }

        void TearDown()
        {
            // Reset environment variable
            if(!saveAssembly)
                unsetenv(rocRoller::ENV_SAVE_ASSEMBLY.c_str());
            else
                setenv(rocRoller::ENV_SAVE_ASSEMBLY.c_str(), saveAssembly, 1);

            CurrentGPUContextFixture::TearDown();
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

    TEST_P(VectorAddBenchmarkGPU, VectorAddBenchmark_GPU_Graph)
    {
        auto nx = GetParam();

        VectorAddGraph(m_context, nx);

        std::cout << TimerPool::summary() << std::endl;
        std::cout << TimerPool::CSV() << std::endl;
    }

    INSTANTIATE_TEST_SUITE_P(
        VectorAddBenchmarks,
        VectorAddBenchmarkGPU,
        ::testing::ValuesIn({16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576}));
}

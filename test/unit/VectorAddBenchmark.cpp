
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <rocRoller/Utilities/HIPTimer.hpp>

#include "CommonGraphs.hpp"
#include "GPUContextFixture.hpp"
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

    TEST_P(VectorAddBenchmarkGPU, VectorAddBenchmark_GPU_Graph)
    {
        auto nx = GetParam();

        RandomGenerator random(31415u);

        auto a = random.vector<int>(nx, -100, 100);
        auto b = random.vector<int>(nx, -100, 100);

        auto d_a     = make_shared_device(a);
        auto d_b     = make_shared_device(b);
        auto d_c     = make_shared_device<int>(nx);
        auto d_alpha = make_shared_device<int>();

        int alpha = 22;

        std::vector<int> r(nx);

        ASSERT_THAT(hipMemcpy(d_alpha.get(), &alpha, 1 * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        auto vectorAdd = rocRollerTest::Graphs::VectorAdd<int>();

        CommandKernel commandKernel(vectorAdd.getCommand(), "VectorAdd");

        auto runtimeArgs
            = vectorAdd.getRuntimeArguments(nx, d_alpha.get(), d_a.get(), d_b.get(), d_c.get());

        HIP_TIMER(t_kernel, "VectorAddKernel_Graph", 1);
        commandKernel.launchKernel(runtimeArgs.runtimeArguments(), t_kernel, 0);
        HIP_SYNC(t_kernel);

        ASSERT_THAT(hipMemcpy(r.data(), d_c.get(), nx * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        // reference solution
        double rnorm = relativeNorm(r, vectorAdd.referenceSolution(alpha, a, b));

        ASSERT_LT(rnorm, 1.e-12);
        EXPECT_GT(t_kernel->elapsed(), std::chrono::steady_clock::duration(0));

        std::cout << TimerPool::summary() << std::endl;
        std::cout << TimerPool::CSV() << std::endl;
    }

    INSTANTIATE_TEST_SUITE_P(
        VectorAddBenchmarks,
        VectorAddBenchmarkGPU,
        ::testing::ValuesIn({16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576}));
}

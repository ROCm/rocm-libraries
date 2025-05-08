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

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include <rocRoller/Utilities/HIPTimer.hpp>

#include <common/CommonGraphs.hpp>

#include "GPUContextFixture.hpp"
#include "Utilities.hpp"

using namespace rocRoller;

namespace VectorAddBenchmark
{

    class VectorAddBenchmarkGPU : public CurrentGPUContextFixture,
                                  public ::testing::WithParamInterface<int>
    {
        void SetUp() override
        {
            CurrentGPUContextFixture::SetUp();
        }
    };

    TEST_P(VectorAddBenchmarkGPU, GPU_VectorAddBenchmark_Graph)
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
        commandKernel.setContext(m_context);
        commandKernel.generateKernel();

        auto runtimeArgs
            = vectorAdd.getRuntimeArguments(nx, d_alpha.get(), d_a.get(), d_b.get(), d_c.get());

        HIP_TIMER(t_kernel, "VectorAddKernel_Graph", 1);
        commandKernel.launchKernel(runtimeArgs.runtimeArguments(), t_kernel, 0);
        HIP_SYNC(t_kernel);

        ASSERT_THAT(hipMemcpy(r.data(), d_c.get(), nx * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        // reference solution
        auto tol = AcceptableError{5 * std::sqrt(nx) * epsilon<double>(), "Sqrt(N) * epsilon"};
        auto res = compare(r, vectorAdd.referenceSolution(alpha, a, b), tol);

        EXPECT_TRUE(res.ok) << res.message();
        EXPECT_GT(t_kernel->elapsed(), std::chrono::steady_clock::duration(0));

        std::cout << TimerPool::summary() << std::endl;
        std::cout << TimerPool::CSV() << std::endl;
    }

    INSTANTIATE_TEST_SUITE_P(
        VectorAddBenchmarks,
        VectorAddBenchmarkGPU,
        ::testing::ValuesIn({16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576}));
}

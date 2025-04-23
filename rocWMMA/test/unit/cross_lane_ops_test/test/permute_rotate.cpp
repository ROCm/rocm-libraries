/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <tuple>
#include <type_traits>

#include "detail/cross_lane_ops.hpp"
#include "kernel_generator.hpp"
#include "unit_test.hpp"

namespace rocwmma
{

    struct TestParams : public UnitTestParams
    {
        using Base = UnitTestParams;

        using Types = typename std::tuple<uint32_t, uint64_t>;

        using PermuteOps32 = std::tuple<PermuteImpl::OpsBase::RotateR<1, 32>,
                                        PermuteImpl::OpsBase::RotateR<5, 32>,
                                        PermuteImpl::OpsBase::RotateL<8, 32>,
                                        PermuteImpl::OpsBase::RotateL<15, 32>>;

        using PermuteOps64 = std::tuple<PermuteImpl::OpsBase::RotateR<1, 64>,
                                        PermuteImpl::OpsBase::RotateR<5, 64>,
                                        PermuteImpl::OpsBase::RotateL<8, 64>,
                                        PermuteImpl::OpsBase::RotateL<15, 64>>;

        using KernelParams32 = typename CombineLists<Types, PermuteOps32>::Result;
        using KernelParams64 = typename CombineLists<Types, PermuteOps64>::Result;

        // Assemble the kernel generator
        // Kernel: VectorIterator
        using GeneratorImpl     = PermuteOpsGenerator;
        using KernelGenerator32 = KernelGenerator<KernelParams32, GeneratorImpl>;
        using KernelGenerator64 = KernelGenerator<KernelParams64, GeneratorImpl>;
        static_assert(std::is_same_v<KernelGenerator64::ResultT, KernelGenerator64::ResultT>,
                      "KernelGenerator32 and KernelGenerator64 should have the same ResultT");
        using KernelResultT = KernelGenerator32::ResultT;

        // Sanity check for kernel generator
        static_assert(std::is_same<typename GeneratorImpl::ResultT, typename Base::KernelT>::value,
                      "Kernels from this generator do not match testing interface");

        // Must be TBlockY must be 1.
        static inline std::vector<ThreadBlockT> threadBlocks()
        {
            auto warpSize = HipDevice::instance()->warpSize();
            return {{warpSize, 1}};
        }

        static inline std::vector<ProblemSizeT> problemSizes()
        {
            auto warpSize = HipDevice::instance()->warpSize();
            return {{warpSize, 1}};
        }

        // 'prev' values
        static inline std::vector<Param1T> param1s()
        {
            return {5.0};
        }

        static inline KernelResultT kernels()
        {
            auto warpSize = HipDevice::instance()->warpSize();
            if(warpSize == 32)
            {
                return KernelGenerator32::generate();
            }
            else
            {
                return KernelGenerator64::generate();
            }
        }
    };

} // namespace rocwmma

// Test suite for unique parameterization
class PermuteRotateTest : public rocwmma::UnitTest
{
};

TEST_P(PermuteRotateTest, RunKernel)
{
    this->RunKernel();
}

INSTANTIATE_TEST_SUITE_P(
    CrossLaneOpTests,
    PermuteRotateTest,
    ::testing::Combine(::testing::ValuesIn(rocwmma::TestParams::kernels()),
                       ::testing::ValuesIn(rocwmma::TestParams::threadBlocks()),
                       ::testing::ValuesIn(rocwmma::TestParams::problemSizes()),
                       ::testing::ValuesIn(rocwmma::TestParams::param1s()),
                       ::testing::ValuesIn(rocwmma::TestParams::param2s())));

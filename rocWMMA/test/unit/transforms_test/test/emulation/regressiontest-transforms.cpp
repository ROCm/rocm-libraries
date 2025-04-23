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

#include "detail/transforms.hpp"
#include "kernel_generator.hpp"
#include "unit_test.hpp"
#include "unit_test_macros.hpp"

namespace rocwmma
{

    template <typename GeneratorImpl, typename VWs>
    struct TestParams : public UnitTestParams
    {
        using Base = UnitTestParams;

        using Types = typename Base::TestAllSizeTypes;

        // size of BlockDim dimension
        using BlockDim = std::tuple<I<16>, I<32>, I<64>, I<128>, I<256>>;

        using KernelParams = typename CombineLists<BlockDim, VWs, Types>::Result;

        // Assemble the kernel generator
        // Kernel: VectorUtil
        using KernelGenerator = KernelGenerator<KernelParams, GeneratorImpl>;

        // Sanity check for kernel generator
        static_assert(std::is_same<typename GeneratorImpl::ResultT, typename Base::KernelT>::value,
                      "Kernels from this generator do not match testing interface");

        static inline std::vector<ThreadBlockT> threadBlocks()
        {
            auto warpSize = HipDevice::instance()->warpSize();
            // clang-format off
            return { {warpSize, 1}, {warpSize * 2, 1}, {warpSize * 4, 1}};
            // return { {warpSize, 1} };
            // clang-format on
        }

        static inline std::vector<ProblemSizeT> problemSizes()
        {
            // clang-format off
            return { {1, 1} };
            // clang-format on
        }

        static inline typename KernelGenerator::ResultT kernels()
        {
            return KernelGenerator::generate();
        }
    };

    using AossoaTestParams = TestParams<AossoaGenerator, std::tuple<I<2>, I<4>, I<8>, I<16>>>;
    using SoaaosTestParams = TestParams<SoaaosGenerator, std::tuple<I<2>, I<4>, I<8>, I<16>>>;

} // namespace rocwmma

ROCWMMA_GENERATE_UNIT_GTEST_SUITE(EmulationRegressionAossoaTest, AossoaTestParams)
ROCWMMA_GENERATE_UNIT_GTEST_SUITE(EmulationRegressionSoaaosTest, SoaaosTestParams)

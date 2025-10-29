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

#include "detail/unpackutil.hpp"
#include "kernel_generator.hpp"
#include "unit_test.hpp"
#include "unit_test_macros.hpp"

namespace rocwmma
{

    template <typename GeneratorImpl>
    struct TestParams : public UnitTestParams
    {
        using Base = UnitTestParams;

        using Types = typename Base::TestAllSizeTypes;

        // Vector Width.
        using VWs = std::tuple<I<2>, I<4>, I<8>, I<16>>;

        using KernelParams = typename CombineLists<VWs, Types>::Result;

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

    using UnpackLo2TestParams    = TestParams<UnpackLo2Generator>;
    using UnpackLo4TestParams    = TestParams<UnpackLo4Generator>;
    using UnpackLo8TestParams    = TestParams<UnpackLo8Generator>;
    using UnpackHi2TestParams    = TestParams<UnpackHi2Generator>;
    using UnpackHi4TestParams    = TestParams<UnpackHi4Generator>;
    using UnpackHi8TestParams    = TestParams<UnpackHi8Generator>;
    using UnpackLoHi1TestParams  = TestParams<UnpackLoHi1Generator>;
    using UnpackLoHi2TestParams  = TestParams<UnpackLoHi2Generator>;
    using UnpackLoHi4TestParams  = TestParams<UnpackLoHi4Generator>;
    using UnpackLoHi8TestParams  = TestParams<UnpackLoHi8Generator>;
    using UnpackLoHi16TestParams = TestParams<UnpackLoHi16Generator>;
    using UnpackLoHi32TestParams = TestParams<UnpackLoHi32Generator>;
} // namespace rocwmma

ROCWMMA_GENERATE_UNIT_GTEST_SUITE(EmulationRegressionUnpackLo2Test, UnpackLo2TestParams)
ROCWMMA_GENERATE_UNIT_GTEST_SUITE(EmulationRegressionUnpackLo4Test, UnpackLo4TestParams)
ROCWMMA_GENERATE_UNIT_GTEST_SUITE(EmulationRegressionUnpackLo8Test, UnpackLo8TestParams)
ROCWMMA_GENERATE_UNIT_GTEST_SUITE(EmulationRegressionUnpackHi2Test, UnpackHi2TestParams)
ROCWMMA_GENERATE_UNIT_GTEST_SUITE(EmulationRegressionUnpackHi4Test, UnpackHi4TestParams)
ROCWMMA_GENERATE_UNIT_GTEST_SUITE(EmulationRegressionUnpackHi8Test, UnpackHi8TestParams)
ROCWMMA_GENERATE_UNIT_GTEST_SUITE(EmulationRegressionUnpackLoHi1Test, UnpackLoHi1TestParams)
ROCWMMA_GENERATE_UNIT_GTEST_SUITE(EmulationRegressionUnpackLoHi2Test, UnpackLoHi2TestParams)
ROCWMMA_GENERATE_UNIT_GTEST_SUITE(EmulationRegressionUnpackLoHi4Test, UnpackLoHi4TestParams)
ROCWMMA_GENERATE_UNIT_GTEST_SUITE(EmulationRegressionUnpackLoHi8Test, UnpackLoHi8TestParams)
ROCWMMA_GENERATE_UNIT_GTEST_SUITE(EmulationRegressionUnpackLoHi16Test, UnpackLoHi16TestParams)
ROCWMMA_GENERATE_UNIT_GTEST_SUITE(EmulationRegressionUnpackLoHi32Test, UnpackLoHi32TestParams)

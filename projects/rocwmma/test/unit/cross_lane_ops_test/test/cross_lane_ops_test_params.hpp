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

#ifndef CROSS_LANE_OPS_TEST_PARAMS_HPP
#define CROSS_LANE_OPS_TEST_PARAMS_HPP
#include <tuple>
#include <type_traits>

#include "detail/cross_lane_ops.hpp"
#include "kernel_generator.hpp"
#include "unit_test.hpp"

namespace rocwmma
{

    template <typename KernelParams, typename GeneratorImpl>
    struct CrossLaneTestParams : public UnitTestParams
    {
        using Base = UnitTestParams;

        using KernelGenerator = KernelGenerator<KernelParams, GeneratorImpl>;

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

        static inline typename KernelGenerator::ResultT kernels()
        {
            return KernelGenerator::generate();
        }
    };

    // Test random assortment of banks and rows

    template <typename... DppOps>
    using DppKernelParams = typename CombineLists<std::tuple<uint32_t, uint64_t>,
                                                  std::tuple<DppOps...>,
                                                  std::tuple<I<0xF>, I<0x5>, I<0xA>>,
                                                  std::tuple<I<0xF>, I<0x7>, I<0x3>>,
                                                  std::tuple<I<false>, I<true>>>::Result;

    template <typename... BlendOps>
    using BlendKernelParams =
        typename CombineLists<std::tuple<uint32_t, uint64_t>, std::tuple<BlendOps...>>::Result;

    template <typename... SwizzleOps>
    using SwizzleKernelParams =
        typename CombineLists<std::tuple<uint32_t, uint64_t>, std::tuple<SwizzleOps...>>::Result;

    template <typename... PermuteOps>
    using PermuteKernelParams =
        typename CombineLists<std::tuple<uint32_t, uint64_t>, std::tuple<PermuteOps...>>::Result;
} // namespace rocwmma
#endif //CROSS_LANE_OPS_TEST_PARAMS_HPP

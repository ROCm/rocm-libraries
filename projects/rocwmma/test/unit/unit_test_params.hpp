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

#ifndef ROCWMMA_UNIT_TEST_PARAMS_HPP
#define ROCWMMA_UNIT_TEST_PARAMS_HPP

#include <tuple>
#include <vector>

#include <rocwmma/internal/types.hpp>

#include "common.hpp"
#include "kernel_generator.hpp"
#include "unit_kernel_base.hpp"

namespace rocwmma
{

    struct UnitTestParams
    {
        ///
        /// Compile-time params used with KernelGenerator to
        /// instantiate kernel objects
        ///

        // Testing types as Input/Output/Compute (IOC)
        using TestTypesIOC = std::tuple<
#if ROCWMMA_FP8
            float8_t,
            bfloat8_t,
#endif // ROCWMMA_FP8

#if ROCWMMA_FP8_FNUZ
            float8_fnuz_t,
            bfloat8_fnuz_t,
#endif // ROCWMMA_FP8_FNUZ
            bfloat16_t,
            float16_t,
#if !ROCWMMA_TESTS_NO_HALF
            hfloat16_t,
#endif // !ROCWMMA_TESTS_NO_HALF
            float32_t,
            xfloat32_t,
            int8_t,
            int32_t
#if ROCWMMA_EXTENDED_TESTS
            ,
            uint8_t,
            uint32_t
#endif // ROCWMMA_EXTENDED_TESTS
            >;

        // Native double
        using TestTypeDouble = float64_t;

        ///
        /// Grouped compile time kernel parameters
        ///

        // 16 has support for double types
        using TestTypes16 = typename Concat<TestTypesIOC, TestTypeDouble>::Result;

        using TestAllSizeTypes = typename std::tuple<int8_t, float16_t, float32_t, float64_t>;

        // Variances for particular BlockM, BlockN
        using TestBlockSizes16 = std::tuple<std::tuple<I<16>, I<16>>,
                                            std::tuple<I<16>, I<32>>,
                                            std::tuple<I<16>, I<64>>
#if ROCWMMA_EXTENDED_TESTS
                                            ,
                                            std::tuple<I<16>, I<128>>,
                                            std::tuple<I<16>, I<256>>
#endif // ROCWMMA_EXTENDED_TESTS
                                            >;

        using TestBlockSizes32 = std::tuple<std::tuple<I<32>, I<8>>,
                                            std::tuple<I<32>, I<16>>,
                                            std::tuple<I<32>, I<32>>,
                                            std::tuple<I<32>, I<64>>
#if ROCWMMA_EXTENDED_TESTS
                                            ,
                                            std::tuple<I<32>, I<128>>,
                                            std::tuple<I<32>, I<256>>
#endif // ROCWMMA_EXTENDED_TESTS
                                            >;

        using TestBlockSizes64 = std::tuple<std::tuple<I<64>, I<8>>,
                                            std::tuple<I<64>, I<16>>,
                                            std::tuple<I<64>, I<32>>,
                                            std::tuple<I<64>, I<64>>
#if ROCWMMA_EXTENDED_TESTS
                                            ,
                                            std::tuple<I<64>, I<128>>,
                                            std::tuple<I<64>, I<256>>
#endif // ROCWMMA_EXTENDED_TESTS
                                            >;

        using TestBlockSizes128 = std::tuple<std::tuple<I<128>, I<8>>,
                                             std::tuple<I<128>, I<16>>,
                                             std::tuple<I<128>, I<32>>,
                                             std::tuple<I<128>, I<64>>
#if ROCWMMA_EXTENDED_TESTS
                                             ,
                                             std::tuple<I<128>, I<128>>,
                                             std::tuple<I<128>, I<256>>
#endif // ROCWMMA_EXTENDED_TESTS
                                             >;

        using TestBlockSizes256 = std::tuple<std::tuple<I<256>, I<8>>,
                                             std::tuple<I<256>, I<16>>,
                                             std::tuple<I<256>, I<32>>,
                                             std::tuple<I<256>, I<64>>
#if ROCWMMA_EXTENDED_TESTS
                                             ,
                                             std::tuple<I<256>, I<128>>,
                                             std::tuple<I<256>, I<256>>
#endif // ROCWMMA_EXTENDED_TESTS
                                             >;

        // Layout groupings
        using TestLayoutsN   = col_major;
        using TestLayoutsT   = row_major;
        using TestLayoutsAll = typename Concat<TestLayoutsN, TestLayoutsT>::Result;

        ///
        /// Run-time kernel argument parameters
        ///

        // Types of parameters
        using KernelT      = std::shared_ptr<KernelI>; // Kernel test interface
        using ThreadBlockT = std::pair<int64_t, int64_t>;
        using ProblemSizeT = std::pair<int64_t, int64_t>;
        using Param1T      = float64_t;
        using Param2T      = float64_t;

        static inline std::vector<ThreadBlockT> threadBlocks()
        {
            auto warpSize = HipDevice::instance()->warpSize();

            // clang-format off
        return { {warpSize, 1},  // 1 Wave
                 {warpSize, 2}, {warpSize * 2, 1}, // 2 Waves
                 {warpSize, 4}, {warpSize * 2, 2}, {warpSize * 4, 1}, // 4 Waves
#if ROCWMMA_EXTENDED_TESTS
                 {warpSize, 8}, {warpSize * 2, 4}, {warpSize * 4, 2}, {warpSize * 8, 1} // 8 waves
#endif // ROCWMMA_EXTENDED_TESTS
            };
            // clang-format on
        }

        static inline std::vector<ProblemSizeT> problemSizes()
        {
            // clang-format off
        // Test at least all the 1-wave and rectangular sizes
        return { {16, 16},  {16, 32},   {16, 64},   {16, 128},  {16, 256},
                 {32, 8},   {32, 16},   {32, 32},   {32, 64},   {32, 128},   {32, 256},
                 {64, 8},   {64, 16},   {64, 32},   {64, 64},   {64, 128},   {64, 256},
                 {128, 8},  {128, 16},  {128, 32},  {128, 64},  {128, 128},  {128, 256},
                 {256, 8},  {256, 16},  {256, 32},  {256, 64},  {256, 128},  {256, 256},
                 {512, 8},  {512, 16},  {512, 32},  {512, 64},  {512, 128},  {512, 256},  {512, 512},
                 {1024, 8}, {1024, 16}, {1024, 32}, {1024, 64}, {1024, 128}, {1024, 256}, {1024, 512},

#if ROCWMMA_EXTENDED_TESTS
                 {1024, 1024},
                 {2048, 2048},
                 {2560, 2560},
                 {3072, 3072},
                 {3584, 3584},
                 {4096, 4096},
#endif // ROCWMMA_EXTENDED_TESTS
        };
            // clang-format on
        }

        static inline std::vector<Param1T> param1s()
        {
            return {0.0};
        }

        static inline std::vector<Param2T> param2s()
        {
            return {0.0};
        }
    };

    struct EmulationUnitTestParams : public UnitTestParams
    {
        static inline std::vector<ThreadBlockT> threadBlocks()
        {
            auto warpSize = HipDevice::instance()->warpSize();

            return {{warpSize * 2, 2}};
        }

        static inline std::vector<ProblemSizeT> problemSizes()
        {
            return {{512, 512}};
        }
    };
} // namespace rocwmma

#endif // ROCWMMA_UNIT_TEST_PARAMS_HPP

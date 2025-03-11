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

#ifndef ROCWMMA_GEMM_COMMON_TEST_PARAMS_HPP
#define ROCWMMA_GEMM_COMMON_TEST_PARAMS_HPP

#include <tuple>
#include <vector>

#include <rocwmma/internal/types.hpp>

#include "common.hpp"
#include "gemm_kernel_base.hpp"
#include "kernel_generator.hpp"

namespace rocwmma
{
    ///
    /// Generalized kernel params for most tests
    ///
    struct GemmCommonTestParams
    {
        ///
        /// Testing types as Input/Output/Compute (IOC)
        ///

        // Native int8
        using TestTypesI8 = std::tuple<
#if ROCWMMA_EXTENDED_TESTS
            std::tuple<int8_t, int8_t, int32_t>,
#endif // ROCWMMA_EXTENDED_TESTS
            std::tuple<int8_t, int32_t, int32_t>>;

        // float8
        using TestTypesF8 = std::tuple<
#if ROCWMMA_FP8
            std::tuple<float8_t, float32_t, float32_t>
#endif // ROCWMMA_FP8

// Only host will ever include both
#if ROCWMMA_FP8 && ROCWMMA_FP8_FNUZ
            ,
#endif // ROCWMMA_FP8 && ROCWMMA_FP8_FNUZ

#if ROCWMMA_FP8_FNUZ
            std::tuple<float8_fnuz_t, float32_t, float32_t>
#endif // ROCWMMA_FP8_FNUZ
            >;

        using TestTypesBF8 = std::tuple<
#if ROCWMMA_FP8
            std::tuple<bfloat8_t, float32_t, float32_t>
#endif // ROCWMMA_FP8

// Only host will ever include both
#if ROCWMMA_FP8 && ROCWMMA_FP8_FNUZ
            ,
#endif // ROCWMMA_FP8 && ROCWMMA_FP8_FNUZ

#if ROCWMMA_FP8_FNUZ
            std::tuple<bfloat8_fnuz_t, float32_t, float32_t>
#endif // ROCWMMA_FP8_FNUZ
            >;

        // Non-native bfloat16_t
        using TestTypesBF16 = std::tuple<
#if ROCWMMA_EXTENDED_TESTS
            std::tuple<bfloat16_t, bfloat16_t, bfloat16_t>,
            std::tuple<bfloat16_t, bfloat16_t, float32_t>,
#endif // ROCWMMA_EXTENDED_TESTS
            std::tuple<bfloat16_t, float32_t, float32_t>>;

        // Native f16
        using TestTypesF16 = std::tuple<
#if ROCWMMA_EXTENDED_TESTS
            std::tuple<float16_t, float16_t, float16_t>,
            std::tuple<float16_t, float16_t, float32_t>,
#endif // ROCWMMA_EXTENDED_TESTS
            std::tuple<float16_t, float32_t, float32_t>>;

#if !ROCWMMA_TESTS_NO_HALF
        // Non-native hfloat16_t (i.e. __half)
        using TestTypesH16 = std::tuple<
#if ROCWMMA_EXTENDED_TESTS
            std::tuple<hfloat16_t, hfloat16_t, hfloat16_t>,
            std::tuple<hfloat16_t, hfloat16_t, float32_t>,
#endif // ROCWMMA_EXTENDED_TESTS
            std::tuple<hfloat16_t, float32_t, float32_t>>;
#endif // !ROCWMMA_TESTS_NO_HALF

        // Native single f32
        using TestTypesF32 = std::tuple<std::tuple<float32_t, float32_t, float32_t>>;

        // xfloat32
        using TestTypesXF32 = std::tuple<std::tuple<xfloat32_t, float32_t, float32_t>>;

        // Native double f64
        using TestTypesF64 = std::tuple<std::tuple<float64_t, float64_t, float64_t>>;

        // Aggregate types <= 8 bit
        using TestTypesTiny = typename Concat<TestTypesF8, TestTypesBF8, TestTypesI8>::Result;

        // Aggregate types <= 16 bit
        using TestTypesSmall = typename Concat<TestTypesTiny,
                                               TestTypesBF16,
                                               TestTypesF16
#if !ROCWMMA_TESTS_NO_HALF
                                               ,
                                               TestTypesH16
#endif // !ROCWMMA_TESTS_NO_HALF
                                               >::Result;

        // Aggregate types <= 32 bit
        using TestTypesMedium =
            typename Concat<TestTypesSmall, TestTypesF32, TestTypesXF32>::Result;

        // Aggregate types <= 64 bit
        using TestTypesLarge = typename Concat<TestTypesMedium, TestTypesF64>::Result;

        // 16 x 16 supports up to f64
        using TestTypes16x16 = TestTypesLarge;

        // 32 x 32 supports up to f32
        using TestTypes32x32 = TestTypesMedium;

        ///
        /// Data Layout types: col_major (N) or row_major (T)
        /// Matrices layouts: A / B / C / D
        /// Lds Layouts: Lds data
        ///

        // Supported generalized data layouts
        using TestDataLayouts = std::tuple<row_major, col_major>;

        // Lds data layouts
        using TestLdsDataLayouts = TestDataLayouts;

        ///
        /// Aggregate data layout combinations A / B / CD
        /// Note: for the following sets, assume C = D = col_major
        /// Extended tests will test both col_major and row_major.
        ///

        using TestLayoutsNN =
#if ROCWMMA_EXTENDED_TESTS
            typename CombineOne<std::tuple<col_major, col_major>, TestDataLayouts>::Result;
#else
            std::tuple<std::tuple<col_major, col_major, col_major>>;
#endif // ROCWMMA_EXTENDED_TESTS

        using TestLayoutsNT =
#if ROCWMMA_EXTENDED_TESTS
            typename CombineOne<std::tuple<col_major, row_major>, TestDataLayouts>::Result;
#else
            std::tuple<std::tuple<col_major, row_major, col_major>>;
#endif // ROCWMMA_EXTENDED_TESTS

        using TestLayoutsTN =
#if ROCWMMA_EXTENDED_TESTS
            typename CombineOne<std::tuple<row_major, col_major>, TestDataLayouts>::Result;
#else
            std::tuple<std::tuple<row_major, col_major, col_major>>;
#endif // ROCWMMA_EXTENDED_TESTS

        using TestLayoutsTT =
#if ROCWMMA_EXTENDED_TESTS
            typename CombineOne<std::tuple<row_major, row_major>, TestDataLayouts>::Result;
#else
            std::tuple<std::tuple<row_major, row_major, col_major>>;
#endif // ROCWMMA_EXTENDED_TESTS

        using TestLayoutsAll =
            typename Concat<TestLayoutsNN, TestLayoutsNT, TestLayoutsNT, TestLayoutsTT>::Result;

        ///
        /// MFMA block sizes
        /// BlockK variances for particular BlockM, BlockN
        ///

        // Aggregate combinations BlockK == 16
        using TestBlockSizes16x16TinyBlockK = std::tuple<std::tuple<I<16>, I<16>, I<16>>>;

        // Aggregate combinations BlockK <= 32
        using TestBlockSizes16x16SmallBlockK = std::tuple<std::tuple<I<16>, I<16>, I<16>>
#if ROCWMMA_EXTENDED_TESTS
                                                          ,
                                                          std::tuple<I<16>, I<16>, I<32>>
#endif // ROCWMMA_EXTENDED_TESTS
                                                          >;

        // Aggregate combinations BlockK <= 64
        using TestBlockSizes16x16MediumBlockK = std::tuple<std::tuple<I<16>, I<16>, I<16>>,
                                                           std::tuple<I<16>, I<16>, I<32>>
#if ROCWMMA_EXTENDED_TESTS
                                                           ,
                                                           std::tuple<I<16>, I<16>, I<64>>
#endif // ROCWMMA_EXTENDED_TESTS
                                                           >;

        // Aggregate combinations BlockK <= 128
        using TestBlockSizes16x16LargeBlockK = std::tuple<std::tuple<I<16>, I<16>, I<16>>,
                                                          std::tuple<I<16>, I<16>, I<32>>,
                                                          std::tuple<I<16>, I<16>, I<64>>
#if ROCWMMA_EXTENDED_TESTS
                                                          ,
                                                          std::tuple<I<16>, I<16>, I<128>>
#endif // ROCWMMA_EXTENDED_TESTS
                                                          >;

        // Aggregate combinations BlockK <= 256
        using TestBlockSizes16x16HugeBlockK = std::tuple<std::tuple<I<16>, I<16>, I<16>>,
                                                         std::tuple<I<16>, I<16>, I<32>>,
                                                         std::tuple<I<16>, I<16>, I<64>>
#if ROCWMMA_EXTENDED_TESTS
                                                         ,
                                                         std::tuple<I<16>, I<16>, I<128>>,
                                                         std::tuple<I<16>, I<16>, I<256>>
#endif // ROCWMMA_EXTENDED_TESTS
                                                         >;
        // Aggregate combinations BlockK == 8
        using TestBlockSizes32x32TinyBlockK = std::tuple<std::tuple<I<32>, I<32>, I<8>>>;

        // Aggregate combinations BlockK <= 16
        using TestBlockSizes32x32SmallBlockK = std::tuple<std::tuple<I<32>, I<32>, I<8>>
#if ROCWMMA_EXTENDED_TESTS
                                                          ,
                                                          std::tuple<I<32>, I<32>, I<16>>
#endif // ROCWMMA_EXTENDED_TESTS
                                                          >;

        // Aggregate combinations BlockK <= 32
        using TestBlockSizes32x32MediumBlockK = std::tuple<std::tuple<I<32>, I<32>, I<8>>,
                                                           std::tuple<I<32>, I<32>, I<16>>
#if ROCWMMA_EXTENDED_TESTS
                                                           ,
                                                           std::tuple<I<32>, I<32>, I<32>>
#endif // ROCWMMA_EXTENDED_TESTS
                                                           >;

        // Aggregate combinations BlockK <= 64
        using TestBlockSizes32x32LargeBlockK = std::tuple<std::tuple<I<32>, I<32>, I<8>>,
                                                          std::tuple<I<32>, I<32>, I<16>>,
                                                          std::tuple<I<32>, I<32>, I<32>>
#if ROCWMMA_EXTENDED_TESTS
                                                          ,
                                                          std::tuple<I<32>, I<32>, I<64>>
#endif // ROCWMMA_EXTENDED_TESTS
                                                          >;

        // Aggregate combinations BlockK <= 128
        using TestBlockSizes32x32HugeBlockK = std::tuple<std::tuple<I<32>, I<32>, I<8>>,
                                                         std::tuple<I<32>, I<32>, I<16>>,
                                                         std::tuple<I<32>, I<32>, I<32>>
#if ROCWMMA_EXTENDED_TESTS
                                                         ,
                                                         std::tuple<I<32>, I<32>, I<64>>,
                                                         std::tuple<I<32>, I<32>, I<128>>
#endif // ROCWMMA_EXTENDED_TESTS
                                                         >;

        using TestBlockSizes16x16 = TestBlockSizes16x16HugeBlockK;
        using TestBlockSizes32x32 = TestBlockSizes32x32HugeBlockK;

        ///
        /// Per-wave output block coverage
        ///
        using TestBlocks1x1 = std::tuple<std::tuple<I<1>, I<1>>>;
        using TestBlocks1x2 = std::tuple<std::tuple<I<1>, I<2>>>;
        using TestBlocks1x4 = std::tuple<std::tuple<I<1>, I<4>>>;
        using TestBlocks1x8 = std::tuple<std::tuple<I<1>, I<8>>>;

        using TestBlocks2x1 = std::tuple<std::tuple<I<2>, I<1>>>;
        using TestBlocks2x2 = std::tuple<std::tuple<I<2>, I<2>>>;
        using TestBlocks2x4 = std::tuple<std::tuple<I<2>, I<4>>>;
        using TestBlocks2x8 = std::tuple<std::tuple<I<2>, I<8>>>;

        using TestBlocks4x1 = std::tuple<std::tuple<I<4>, I<1>>>;
        using TestBlocks4x2 = std::tuple<std::tuple<I<4>, I<2>>>;
        using TestBlocks4x4 = std::tuple<std::tuple<I<4>, I<4>>>;
        using TestBlocks4x8 = std::tuple<std::tuple<I<4>, I<8>>>;

        using TestBlocks8x1 = std::tuple<std::tuple<I<8>, I<1>>>;
        using TestBlocks8x2 = std::tuple<std::tuple<I<8>, I<2>>>;
        using TestBlocks8x4 = std::tuple<std::tuple<I<8>, I<4>>>;
        using TestBlocks8x8 = std::tuple<std::tuple<I<8>, I<8>>>;

        ///
        /// Run-time kernel argument parameters
        ///

        // Types of parameters
        using KernelT      = std::shared_ptr<KernelI>; // Kernel test interface
        using ThreadBlockT = std::pair<int64_t, int64_t>;
        using ProblemSizeT = std::tuple<int64_t, int64_t, int64_t>;
        using AlphaT       = float64_t;
        using BetaT        = float64_t;

        static inline std::vector<ThreadBlockT> threadBlocks()
        {
            auto warpSize = HipDevice::instance()->warpSize();

            return
            {
                // clang-format off
                // Don't benchmark wg less than 4 waves by default
#if ROCWMMA_VALIDATION_TESTS || ROCWMMA_EXTENDED_TESTS
                {warpSize, 1}, // 1 wave
                {warpSize, 2}, {warpSize * 2, 1}, // 2 wave
#endif // ROCWMMA_VALIDATION_TESTS || ROCWMMA_EXTENDED_TESTS
                {warpSize, 4}, {warpSize * 2, 2}, // 4 wave
                {warpSize * 4, 1}  // 4 wave
                // clang-format on
            };
        }

        static inline std::vector<ProblemSizeT> problemSizes()
        {
            return
            {
                // clang-format off
                {64, 64, 1024},
                {32, 64, 1024},
                {64, 32, 1024},
                {256, 256, 1024},
                {2048, 64, 1024},
                {64, 2048, 1024},
                {512, 512, 512},
                // Skip validation on larger sizes
                // due to very slow.
#if !ROCWMMA_VALIDATION_TESTS
                {1024, 1024, 1024},
                {2048, 2048, 2048},
                {2560, 2560, 2560},
                {3072, 3072, 3072},
                {3584, 3584, 3584},
                {4096, 4096, 4096},
                {5120, 5120, 5120},
#if ROCWMMA_EXTENDED_TESTS
                {6144, 6144, 6144},
                {7168, 7168, 7168},
                {8192, 8192, 8192},
#endif // ROCWMMA_EXTENDED_TESTS
#endif // !ROCWMMA_VALIDATION_TESTS \
    // clang-format on
            };
        }

        static inline std::vector<AlphaT> alphas()
        {
            return {static_cast<AlphaT>(2)};
        }

        static inline std::vector<BetaT> betas()
        {
            return {static_cast<BetaT>(2)};
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_GEMM_COMMON_TEST_PARAMS_HPP

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

#include "common_test_emulation_params.hpp"
#include "test/test_includes.hpp"

namespace rocwmma
{

#if ROCWMMA_FP8_FNUZ
    ROCWMMA_GENERATE_GEMM_GTEST_SUITE_PARAMS(
        TestParamsF8fnuzF32,
        EmulationCommonTestParams,
        KernelGeneratorImpl,
        std::tuple<std::tuple<float8_fnuz_t, float32_t, float32_t>>,
        std::tuple<std::tuple<I<16>, I<16>, I<32>>>,
        TestLayoutsAll,
        TestLdsDataLayouts,
        TestGemmConfigsBlockLevel,
        TestBlocks2x2);

    ROCWMMA_GENERATE_GEMM_GTEST_SUITE_PARAMS(
        TestParamsBF8fnuzF32,
        EmulationCommonTestParams,
        KernelGeneratorImpl,
        std::tuple<std::tuple<bfloat8_fnuz_t, float32_t, float32_t>>,
        std::tuple<std::tuple<I<16>, I<16>, I<32>>>,
        TestLayoutsAll,
        TestLdsDataLayouts,
        TestGemmConfigsBlockLevel,
        TestBlocks2x2);
#endif

} // namespace rocwmma

#if(ROCWMMA_FP8_FNUZ)
ROCWMMA_INSTANTIATE_GEMM_GTEST_SUITE_NO_WARMUP(Gemm_PGR1_LB2_MP0_MB_CP,
                                               Emulation_Smoke_Block_LK_16x16_2x2_F8fnuzF32,
                                               rocwmma::TestParamsF8fnuzF32);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(Emulation_Smoke_Block_LK_16x16_2x2_F8fnuzF32);
ROCWMMA_INSTANTIATE_GEMM_GTEST_SUITE_NO_WARMUP(Gemm_PGR1_LB2_MP0_MB_CP,
                                               Emulation_Smoke_Block_LK_16x16_2x2_BF8fnuzF32,
                                               rocwmma::TestParamsBF8fnuzF32);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(Emulation_Smoke_Block_LK_16x16_2x2_BF8fnuzF32);
#endif

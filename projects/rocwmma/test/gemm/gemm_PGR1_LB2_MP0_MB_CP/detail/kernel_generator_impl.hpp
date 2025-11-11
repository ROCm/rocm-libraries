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

#ifndef ROCWMMA_GEMM_TEST_DETAIL_KERNEL_GENERATOR
#define ROCWMMA_GEMM_TEST_DETAIL_KERNEL_GENERATOR

#include <memory>
#include <tuple>

#include "kernel_impl.hpp"

namespace rocwmma
{
    struct KernelGenerator_PGR1_LB2_MP0_MB_CP
    {
        // Indices to test parameters
        enum : uint32_t
        {
            InputT     = 0,
            OutputT    = 1,
            ComputeT   = 2,
            BlockM     = 3,
            BlockN     = 4,
            BlockK     = 5,
            LayoutA    = 6,
            LayoutB    = 7,
            LayoutCD   = 8,
            LayoutLds  = 9,
            GemmConfig = 10,
            BlocksX    = 11,
            BlocksY    = 12
        };

        using ResultT = std::shared_ptr<KernelI>;

        template <typename... Ts>
        static ResultT generate(std::tuple<Ts...> testParams)
        {
            using TestParamsT = std::tuple<Ts...>;
            using KernelT
                = Kernel_PGR1_LB2_MP0_MB_CP<std::tuple_element_t<BlockM, TestParamsT>::value,
                                            std::tuple_element_t<BlockN, TestParamsT>::value,
                                            std::tuple_element_t<BlockK, TestParamsT>::value,
                                            std::tuple_element_t<InputT, TestParamsT>,
                                            std::tuple_element_t<OutputT, TestParamsT>,
                                            std::tuple_element_t<ComputeT, TestParamsT>,
                                            std::tuple_element_t<LayoutA, TestParamsT>,
                                            std::tuple_element_t<LayoutB, TestParamsT>,
                                            std::tuple_element_t<LayoutCD, TestParamsT>,
                                            std::tuple_element_t<LayoutCD, TestParamsT>,
                                            std::tuple_element_t<LayoutLds, TestParamsT>,
                                            std::tuple_element_t<GemmConfig, TestParamsT>,
                                            std::tuple_element_t<BlocksX, TestParamsT>::value,
                                            std::tuple_element_t<BlocksY, TestParamsT>::value>;

            return std::make_shared<KernelT>();
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_GEMM_TEST_DETAIL_KERNEL_GENERATOR

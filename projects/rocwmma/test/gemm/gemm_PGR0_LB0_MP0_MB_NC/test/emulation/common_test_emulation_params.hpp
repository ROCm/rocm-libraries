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

#ifndef ROCWMMA_GEMM_COMMON_TEST_EMULATION_PARAMS
#define ROCWMMA_GEMM_COMMON_TEST_EMULATION_PARAMS

#include "../common_test_params.hpp"

namespace rocwmma
{
    ///
    /// Generalized kernel params for smoke tests
    ///
    struct EmulationCommonTestParams : public GemmCommonTestParams
    {
        ///
        /// Kernel generator impl objects
        ///
        using KernelGeneratorImpl = KernelGenerator_PGR0_LB0_MP0_MB_NC;

        static inline std::vector<ThreadBlockT> threadBlocks()
        {
            auto warpSize = HipDevice::instance()->warpSize();

            return {{warpSize * 2, 2}};
        }
        static inline std::vector<ProblemSizeT> problemSizes()
        {
            return {{256, 256, 256}};
        }
    };
} // namespace rocwmma

#endif // ROCWMMA_GEMM_COMMON_TEST_EMULATION_PARAMS

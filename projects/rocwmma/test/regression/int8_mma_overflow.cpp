/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
// int8 MMA wraps on int32 overflow on every architecture, matching MFMA, the
// uint8 path and the host reference. A single mma_sync overflows here.
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include <cstdint>

namespace
{
    using namespace rocwmma;

    constexpr uint32_t BlockK = 32;

    int8_t valueA(uint32_t row)
    {
        // Rows cycle through overflowing up, overflowing down and not overflowing.
        return row % 3u == 0u ? int8_t(127) : (row % 3u == 1u ? int8_t(-128) : int8_t(1));
    }

    int8_t valueB(uint32_t col)
    {
        return int8_t(127 - int32_t(col));
    }

    int32_t valueC(uint32_t row, uint32_t col)
    {
        return row % 3u == 0u   ? INT32_MAX - int32_t(1000 + col)
               : row % 3u == 1u ? INT32_MIN + int32_t(1000 + col)
                                : int32_t(col);
    }

    __global__ void int8Mma(int8_t const* a, int8_t const* b, int32_t* c)
    {
#if ROCWMMA_BLOCK_DIM_16_SUPPORTED
        fragment<matrix_a, 16, 16, BlockK, int8_t, row_major> fa;
        fragment<matrix_b, 16, 16, BlockK, int8_t, col_major> fb;
        fragment<accumulator, 16, 16, BlockK, int32_t>        fc;
        load_matrix_sync(fa, a, BlockK);
        load_matrix_sync(fb, b, BlockK);
        load_matrix_sync(fc, c, 16, mem_row_major);
        mma_sync(fc, fa, fb, fc);
        store_matrix_sync(c, fc, 16, mem_row_major);
#endif
    }
}

TEST(Int8MmaOverflow, Wraps)
{
    int  count  = 0;
    auto status = hipGetDeviceCount(&count);
    if(status == hipErrorNoDevice || (status == hipSuccess && count == 0))
        GTEST_SKIP() << "No ROCm-capable device available";
    ASSERT_EQ(status, hipSuccess);
    int             device = 0;
    hipDeviceProp_t props{};
    ASSERT_EQ(hipGetDevice(&device), hipSuccess);
    ASSERT_EQ(hipGetDeviceProperties(&props, device), hipSuccess);

    int8_t*  a = nullptr;
    int8_t*  b = nullptr;
    int32_t* c = nullptr;
    ASSERT_EQ(hipMallocManaged(&a, 16 * BlockK), hipSuccess);
    ASSERT_EQ(hipMallocManaged(&b, 16 * BlockK), hipSuccess);
    ASSERT_EQ(hipMallocManaged(&c, sizeof(int32_t) * 256), hipSuccess);
    for(uint32_t i = 0; i < 16; ++i)
        for(uint32_t k = 0; k < BlockK; ++k)
        {
            a[i * BlockK + k] = valueA(i);
            b[i * BlockK + k] = valueB(i);
        }
    for(uint32_t i = 0; i < 16; ++i)
        for(uint32_t j = 0; j < 16; ++j)
            c[i * 16 + j] = valueC(i, j);

    hipLaunchKernelGGL(int8Mma, dim3(1), dim3(props.warpSize), 0, 0, a, b, c);
    EXPECT_EQ(hipGetLastError(), hipSuccess);
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);

    for(uint32_t i = 0; i < 16; ++i)
        for(uint32_t j = 0; j < 16; ++j)
        {
            // Two's complement wrap-around, computed without signed overflow
            uint32_t expected = uint32_t(valueC(i, j));
            for(uint32_t k = 0; k < BlockK; ++k)
                expected += uint32_t(int32_t(valueA(i)) * int32_t(valueB(j)));
            EXPECT_EQ(c[i * 16 + j], int32_t(expected)) << "row=" << i << ", col=" << j;
        }
    EXPECT_EQ(hipFree(a), hipSuccess);
    EXPECT_EQ(hipFree(b), hipSuccess);
    EXPECT_EQ(hipFree(c), hipSuccess);
}

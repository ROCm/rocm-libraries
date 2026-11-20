/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

//
// Unit tests for the BSR / GEBSR value-offset macros in rocsparse_common.hpp.
// With 32-bit indices the block offset block_dim * block_dim * j must still be
// computed in 64-bit, since nnzb * block_dim^2 can exceed INT32_MAX while nnzb
// itself fits (AISPARSE-826). The test offsets are above 2^32 so a product that
// wraps in 32-bit unsigned arithmetic fails too. A real matrix crossing that
// threshold needs tens of GB of values, so the offset expression is pinned here
// instead.
//
// Builds into rocsparse-unit-test-device because rocsparse_common.hpp needs
// `-x hip`.
//
#include "unit_test_utils.hpp"

#include "rocsparse_common.hpp"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace
{
    constexpr int32_t s_block_dim     = 32;
    constexpr int32_t s_row_block_dim = 32;
    constexpr int32_t s_col_block_dim = 64;
    constexpr int32_t s_j             = 4200001;
    constexpr int32_t s_bi            = 3;
    constexpr int32_t s_bj            = 5;

    constexpr int64_t s_bsr_r   = int64_t(32) * 32 * 4200001 + 3 * 32 + 5;
    constexpr int64_t s_bsr_c   = int64_t(32) * 32 * 4200001 + 3 + 5 * 32;
    constexpr int64_t s_gebsr_r = int64_t(32) * 64 * 4200001 + 3 * 64 + 5;
    constexpr int64_t s_gebsr_c = int64_t(32) * 64 * 4200001 + 3 + 5 * 32;

    __global__ void bsr_ind_kernel(int32_t  block_dim,
                                   int32_t  row_block_dim,
                                   int32_t  col_block_dim,
                                   int32_t  j,
                                   int32_t  bi,
                                   int32_t  bj,
                                   int64_t* out)
    {
        out[0] = BSR_IND(j, bi, bj, rocsparse_direction_row);
        out[1] = BSR_IND(j, bi, bj, rocsparse_direction_column);
        out[2] = GEBSR_IND(j, bi, bj, rocsparse_direction_row);
        out[3] = GEBSR_IND(j, bi, bj, rocsparse_direction_column);
    }
}

TEST(internal_common_bsr_ind, host_int32_offset_above_uint32_max)
{
    static_assert(s_bsr_r > UINT32_MAX, "test offset must exceed UINT32_MAX");
    static_assert(s_bsr_c > UINT32_MAX, "test offset must exceed UINT32_MAX");
    static_assert(s_gebsr_r > UINT32_MAX, "test offset must exceed UINT32_MAX");
    static_assert(s_gebsr_c > UINT32_MAX, "test offset must exceed UINT32_MAX");

    // Read through volatile so the compiler cannot constant-fold the products.
    volatile int32_t v_block_dim = s_block_dim, v_row_block_dim = s_row_block_dim,
                     v_col_block_dim = s_col_block_dim, v_j = s_j;

    const int32_t block_dim     = v_block_dim;
    const int32_t row_block_dim = v_row_block_dim;
    const int32_t col_block_dim = v_col_block_dim;
    const int32_t j             = v_j;

    EXPECT_EQ(int64_t(BSR_IND_R(j, s_bi, s_bj)), s_bsr_r);
    EXPECT_EQ(int64_t(BSR_IND_C(j, s_bi, s_bj)), s_bsr_c);
    EXPECT_EQ(int64_t(GEBSR_IND_R(j, s_bi, s_bj)), s_gebsr_r);
    EXPECT_EQ(int64_t(GEBSR_IND_C(j, s_bi, s_bj)), s_gebsr_c);
}

TEST(internal_common_bsr_ind, device_int32_offset_above_uint32_max)
{
    rocsparse_ut::device_vector<int64_t> d_out(4);
    ASSERT_NE(d_out.ptr, nullptr);

    hipLaunchKernelGGL(bsr_ind_kernel,
                       dim3(1),
                       dim3(1),
                       0,
                       0,
                       s_block_dim,
                       s_row_block_dim,
                       s_col_block_dim,
                       s_j,
                       s_bi,
                       s_bj,
                       d_out.ptr);
    UT_CHECK_HIP(hipGetLastError());

    int64_t out[4] = {};
    UT_CHECK_HIP(hipMemcpy(out, d_out.ptr, sizeof(out), hipMemcpyDeviceToHost));

    EXPECT_EQ(out[0], s_bsr_r);
    EXPECT_EQ(out[1], s_bsr_c);
    EXPECT_EQ(out[2], s_gebsr_r);
    EXPECT_EQ(out[3], s_gebsr_c);
}

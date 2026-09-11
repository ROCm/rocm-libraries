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
// Device (GPU) unit tests for rocsparse::read_first_lane (AISPARSE-759).
//
// __builtin_amdgcn_readfirstlane is a 32-bit op. The library overloads
// currently forward every type through it, so 64-bit integers lose their
// upper half and floating-point values are rounded to int. This TU launches
// one wavefront on the active device (wave32 on gfx1201) with tiny buffers —
// it does not allocate a 2^31-nnz matrix.
//
// The nnzsplit kernel computes
//   start_nnz_index = read_first_lane(I(startingId0 * NNZ_PER_BLOCK))
// with NNZ_PER_BLOCK = 2048. That value crosses 2^31 at startingId0 == 1048576.
//
#include "unit_test_utils.hpp"

#include "unit_test_internal_collectives_common.hpp"

#include "rocsparse_common.hpp"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <vector>

using rocsparse_ut::device_vector;
using rocsparse_ut::launch_single_warp;
using rocsparse_ut::to_host;

using namespace rocsparse_ut_collectives;

namespace
{
    constexpr uint32_t NNZ_PER_BLOCK = 2048;

    // Lane 0 holds `first`; every other lane holds `poison`. After
    // read_first_lane, every lane must observe `first`.
    template <typename T>
    __global__ void k_read_first_lane(const T* in, T* out)
    {
        const int lane = threadIdx.x;
        out[lane]      = rocsparse::read_first_lane(in[lane]);
    }

    // Same expression the nnzsplit kernel uses for the block's first nonzero.
    __global__ void k_nnzsplit_start_index(int32_t starting_id0, int64_t* out)
    {
        const int64_t start = rocsparse::read_first_lane(int64_t(starting_id0 * NNZ_PER_BLOCK));
        out[threadIdx.x]    = start;
    }

    template <typename T>
    void expect_broadcast(T first, T poison)
    {
        const uint32_t wf = require_wavefront_size();
        std::vector<T> in(wf, poison);
        in[0] = first;
        device_vector<T> d_in(in), d_out(size_t{wf});
        ASSERT_NE(d_in.ptr, nullptr);
        ASSERT_NE(d_out.ptr, nullptr);
        ASSERT_EQ(launch_single_warp(k_read_first_lane<T>, d_in.ptr, d_out.ptr), hipSuccess);
        auto h = to_host(d_out);
        for(uint32_t l = 0; l < wf; ++l)
            expect_close(h[l], first);
    }

    void expect_nnzsplit_start(int32_t starting_id0, int64_t expected)
    {
        const uint32_t         wf = require_wavefront_size();
        device_vector<int64_t> d_out(size_t{wf});
        ASSERT_NE(d_out.ptr, nullptr);
        ASSERT_EQ(launch_single_warp(k_nnzsplit_start_index, starting_id0, d_out.ptr), hipSuccess);
        auto h = to_host(d_out);
        for(uint32_t l = 0; l < wf; ++l)
            EXPECT_EQ(h[l], expected) << "lane " << l << " starting_id0=" << starting_id0;
    }
} // namespace

// 32-bit integer overloads are already correct (control).
TEST(internal_collectives_readfirstlane, i32)
{
    expect_broadcast<int32_t>(123456789, -1);
}

TEST(internal_collectives_readfirstlane, u32)
{
    expect_broadcast<uint32_t>(0x80000000u, 0xdeadbeefu);
}

// Last nnzsplit block whose offset still fits in 31 bits: 1048575 * 2048.
TEST(internal_collectives_readfirstlane, i64_below_2_31)
{
    expect_broadcast<int64_t>(2147481600LL, int64_t(0x0123456789abcdefLL));
}

// AISPARSE-759: 2^31 is truncated to -2^31 by the 32-bit builtin.
TEST(internal_collectives_readfirstlane, i64_at_2_31)
{
    expect_broadcast<int64_t>(2147483648LL, int64_t(0x0123456789abcdefLL));
}

TEST(internal_collectives_readfirstlane, i64_next_nnzsplit_block)
{
    expect_broadcast<int64_t>(2147485696LL, int64_t(0x0123456789abcdefLL));
}

// Customer matrix nnz from AISPARSE-759.
TEST(internal_collectives_readfirstlane, u64_customer_nnz)
{
    expect_broadcast<uint64_t>(3032311773ull, 0x0123456789abcdefull);
}

TEST(internal_collectives_readfirstlane, f32_pi)
{
    expect_broadcast<float>(3.14159265358979f, -1.0f);
}

TEST(internal_collectives_readfirstlane, f64_pi)
{
    expect_broadcast<double>(3.14159265358979, -1.0);
}

TEST(internal_collectives_readfirstlane, nnzsplit_start_below_2_31)
{
    expect_nnzsplit_start(1048575, 2147481600LL);
}

TEST(internal_collectives_readfirstlane, nnzsplit_start_at_2_31)
{
    expect_nnzsplit_start(1048576, 2147483648LL);
}

TEST(internal_collectives_readfirstlane, nnzsplit_start_next_block)
{
    expect_nnzsplit_start(1048577, 2147485696LL);
}

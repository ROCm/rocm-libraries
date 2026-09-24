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
// Device (GPU) unit tests for rocsparse::read_first_lane.
//
// __builtin_amdgcn_readfirstlane is a 32-bit op. The library overloads
// currently forward every type through it, so 64-bit integers lose their
// upper half and floating-point values are rounded to int. This TU launches
// one wavefront on the active device (wave32 on gfx1201) with tiny buffers --
// it does not allocate a 2^31-nnz matrix.
//
// Every broadcast test sweeps the source lane across the whole wavefront: the
// value under test is placed in lane i and only lanes >= i are left active, so
// lane i is the wavefront's first active lane and the broadcast is exercised
// from every lane rather than from lane 0 alone.
//
// The nnzsplit kernel computes
//   start_nnz_index = read_first_lane(I(startingId0) * NNZ_PER_BLOCK)
// where NNZ_PER_BLOCK = BLOCKSIZE * NNZ_PER_THREAD. That product crosses 2^31
// at startingId0 == 2^31 / NNZ_PER_BLOCK, which is the boundary the nnzsplit
// tests below straddle for each block size the dispatch can select.
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
    // Lane `src_lane` holds the value under test; every other lane holds
    // `poison`. Lanes below `src_lane` are masked off, so `src_lane` is the
    // wavefront's first active lane and read_first_lane must broadcast its
    // value to all remaining lanes.
    template <typename T>
    __global__ void k_read_first_lane(const T* in, T* out, int32_t src_lane)
    {
        const int32_t lane = static_cast<int32_t>(threadIdx.x);
        if(lane >= src_lane)
        {
            out[lane] = rocsparse::read_first_lane(in[lane]);
        }
    }

    // Broadcast `value` from lane `src_lane` and require every active lane to
    // observe it unchanged. `poison` fills the lanes that must not win the
    // broadcast; the 64-bit cases pass one that differs from `value` in both
    // 32-bit halves, so an implementation that drops a half cannot match by
    // accident. The output buffer is pre-filled with `poison` as well, so a
    // lane the kernel never writes fails instead of reading back a zero.
    template <typename T>
    void expect_broadcast(int32_t src_lane, T value, T poison)
    {
        const uint32_t wf = require_wavefront_size();
        ASSERT_LT(src_lane, static_cast<int32_t>(wf));

        std::vector<T> in(wf, poison);
        in[static_cast<size_t>(src_lane)] = value;

        device_vector<T> d_in(in);
        device_vector<T> d_out(std::vector<T>(wf, poison));
        ASSERT_NE(d_in.ptr, nullptr);
        ASSERT_NE(d_out.ptr, nullptr);
        ASSERT_EQ(launch_single_warp(k_read_first_lane<T>, d_in.ptr, d_out.ptr, src_lane),
                  hipSuccess);

        auto h = to_host(d_out);
        for(uint32_t l = static_cast<uint32_t>(src_lane); l < wf; ++l)
        {
            SCOPED_TRACE(testing::Message() << "src_lane=" << src_lane << " lane=" << l);
            expect_close(h[l], value);
        }
    }

    // Run `expect_broadcast` once per source lane of the active wavefront.
    template <typename T>
    void expect_broadcast_from_every_lane(T value, T poison)
    {
        const int32_t wf = static_cast<int32_t>(require_wavefront_size());
        for(int32_t src_lane = 0; src_lane < wf; ++src_lane)
        {
            expect_broadcast<T>(src_lane, value, poison);
        }
    }

    // Same expression the nnzsplit kernel uses for the block's first nonzero.
    template <uint32_t NNZ_PER_BLOCK>
    __global__ void k_nnzsplit_start_index(int32_t starting_id0, int64_t* out)
    {
        const int64_t start = rocsparse::read_first_lane(static_cast<int64_t>(starting_id0)
                                                         * static_cast<int64_t>(NNZ_PER_BLOCK));
        out[threadIdx.x]    = start;
    }

    // Require every lane to see the full 64-bit first-nonzero index of block
    // `starting_id0`. The expectation is computed here rather than spelled out
    // by the caller, and it is computed in int64_t on purpose: the product is
    // exactly what overflows 32 bits, so an expectation evaluated in 32-bit
    // arithmetic would wrap the same way the bug under test does and the test
    // would agree with a broken implementation.
    template <uint32_t NNZ_PER_BLOCK>
    void expect_nnzsplit_start(int32_t starting_id0)
    {
        const int64_t expected
            = static_cast<int64_t>(starting_id0) * static_cast<int64_t>(NNZ_PER_BLOCK);

        const uint32_t         wf = require_wavefront_size();
        device_vector<int64_t> d_out(size_t{wf});
        ASSERT_NE(d_out.ptr, nullptr);
        ASSERT_EQ(
            launch_single_warp(k_nnzsplit_start_index<NNZ_PER_BLOCK>, starting_id0, d_out.ptr),
            hipSuccess);

        auto h = to_host(d_out);
        for(uint32_t l = 0; l < wf; ++l)
        {
            EXPECT_EQ(h[l], expected) << "lane " << l << " starting_id0=" << starting_id0
                                      << " NNZ_PER_BLOCK=" << NNZ_PER_BLOCK;
        }
    }

    // Straddle the 2^31 boundary for a given block size: the last block whose
    // first-nonzero index still fits in 31 bits, the first block that does not,
    // and the one after it. The block indices are derived from NNZ_PER_BLOCK so
    // nothing here has to be recomputed by hand when the block size changes.
    template <uint32_t NNZ_PER_BLOCK>
    void expect_nnzsplit_start_across_2_31()
    {
        constexpr int64_t boundary = int64_t{1} << 31;
        constexpr int32_t at_2_31  = static_cast<int32_t>(boundary / NNZ_PER_BLOCK);
        static_assert(at_2_31 * static_cast<int64_t>(NNZ_PER_BLOCK) == boundary,
                      "NNZ_PER_BLOCK must divide 2^31 for the boundary block to be exact.");

        expect_nnzsplit_start<NNZ_PER_BLOCK>(at_2_31 - 1);
        expect_nnzsplit_start<NNZ_PER_BLOCK>(at_2_31);
        expect_nnzsplit_start<NNZ_PER_BLOCK>(at_2_31 + 1);
    }
} // namespace

// 32-bit integer overloads are already correct (control).
TEST(internal_collectives_readfirstlane, i32)
{
    expect_broadcast_from_every_lane<int32_t>(123456789, -1);
}

TEST(internal_collectives_readfirstlane, u32)
{
    expect_broadcast_from_every_lane<uint32_t>(0x80000000u, 0xdeadbeefu);
}

// Last nnzsplit block whose offset still fits in 31 bits: 1048575 * 2048.
TEST(internal_collectives_readfirstlane, i64_below_2_31)
{
    expect_broadcast_from_every_lane<int64_t>(2147481600LL, int64_t(0x0123456789abcdefLL));
}

// 2^31 is truncated to -2^31 by the 32-bit builtin.
TEST(internal_collectives_readfirstlane, i64_at_2_31)
{
    expect_broadcast_from_every_lane<int64_t>(2147483648LL, int64_t(0x0123456789abcdefLL));
}

TEST(internal_collectives_readfirstlane, i64_next_nnzsplit_block)
{
    expect_broadcast_from_every_lane<int64_t>(2147485696LL, int64_t(0x0123456789abcdefLL));
}

// Customer matrix nnz (int64, above 2^31).
TEST(internal_collectives_readfirstlane, u64_customer_nnz)
{
    expect_broadcast_from_every_lane<uint64_t>(3032311773ull, 0x0123456789abcdefull);
}

TEST(internal_collectives_readfirstlane, f32_pi)
{
    expect_broadcast_from_every_lane<float>(3.14159265358979f, -1.0f);
}

TEST(internal_collectives_readfirstlane, f64_pi)
{
    expect_broadcast_from_every_lane<double>(3.14159265358979, -1.0);
}

// NNZ_PER_BLOCK is BLOCKSIZE * NNZ_PER_THREAD, with BLOCKSIZE spanning four
// wavefronts (128 on wave32, 256 on wave64) and NNZ_PER_THREAD in {1, 4, 8}.
TEST(internal_collectives_readfirstlane, nnzsplit_start_128)
{
    expect_nnzsplit_start_across_2_31<128>();
}

TEST(internal_collectives_readfirstlane, nnzsplit_start_1024)
{
    expect_nnzsplit_start_across_2_31<1024>();
}

TEST(internal_collectives_readfirstlane, nnzsplit_start_2048)
{
    expect_nnzsplit_start_across_2_31<2048>();
}

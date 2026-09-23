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
// Device (GPU) unit tests for rocSPARSE internal BLOCK reductions
// (rocsparse::blockreduce_sum / _max / _min). Split out of
// unit_test_internal_collectives.cpp by collective family; see
// unit_test_internal_collectives_common.hpp for the shared helpers and
// unit_test_utils.hpp for the launch/readback harness.
//
#include "unit_test_utils.hpp"

#include "unit_test_internal_collectives_common.hpp"

#include "rocsparse_common.hpp" // blockreduce_*

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

using rocsparse_ut::device_vector;
using rocsparse_ut::launch_single_block;
using rocsparse_ut::to_host;

using namespace rocsparse_ut_collectives;

namespace
{
    // ---- block reductions (rocsparse_common.hpp) ---------------------------
    template <uint32_t BS, typename T>
    __global__ void k_blockreduce_sum(const T* in, T* out)
    {
        __shared__ T s[BS];
        const int    tid = threadIdx.x;
        s[tid]           = in[tid];
        __syncthreads();
        rocsparse::blockreduce_sum<BS>(tid, s);
        if(tid == 0)
        {
            out[0] = s[0];
        }
    }
    template <uint32_t BS, typename T>
    __global__ void k_blockreduce_max(const T* in, T* out)
    {
        __shared__ T s[BS];
        const int    tid = threadIdx.x;
        s[tid]           = in[tid];
        __syncthreads();
        rocsparse::blockreduce_max<BS>(tid, s);
        if(tid == 0)
        {
            out[0] = s[0];
        }
    }
    template <uint32_t BS, typename T>
    __global__ void k_blockreduce_min(const T* in, T* out)
    {
        __shared__ T s[BS];
        const int    tid = threadIdx.x;
        s[tid]           = in[tid];
        __syncthreads();
        rocsparse::blockreduce_min<BS>(tid, s);
        if(tid == 0)
        {
            out[0] = s[0];
        }
    }
    template <uint32_t BS, typename T>
    void run_blockreduce_sum(const std::vector<T>& in)
    {
        ASSERT_EQ(in.size(), size_t{BS});
        T ref = T(0);
        for(auto v : in)
        {
            ref = ref + v;
        }
        device_vector<T> d_in(in), d_out(size_t{1});
        ASSERT_NE(d_in.ptr, nullptr);
        ASSERT_NE(d_out.ptr, nullptr);
        ASSERT_EQ(launch_single_block(k_blockreduce_sum<BS, T>, BS, d_in.ptr, d_out.ptr),
                  hipSuccess);
        expect_close(to_host(d_out)[0], ref);
    }
    template <uint32_t BS, typename T>
    void run_blockreduce_max(const std::vector<T>& in)
    {
        ASSERT_EQ(in.size(), size_t{BS});
        T                ref = *std::max_element(in.begin(), in.end());
        device_vector<T> d_in(in), d_out(size_t{1});
        ASSERT_NE(d_in.ptr, nullptr);
        ASSERT_NE(d_out.ptr, nullptr);
        ASSERT_EQ(launch_single_block(k_blockreduce_max<BS, T>, BS, d_in.ptr, d_out.ptr),
                  hipSuccess);
        expect_close(to_host(d_out)[0], ref);
    }
    template <uint32_t BS, typename T>
    void run_blockreduce_min(const std::vector<T>& in)
    {
        ASSERT_EQ(in.size(), size_t{BS});
        T                ref = *std::min_element(in.begin(), in.end());
        device_vector<T> d_in(in), d_out(size_t{1});
        ASSERT_NE(d_in.ptr, nullptr);
        ASSERT_NE(d_out.ptr, nullptr);
        ASSERT_EQ(launch_single_block(k_blockreduce_min<BS, T>, BS, d_in.ptr, d_out.ptr),
                  hipSuccess);
        expect_close(to_host(d_out)[0], ref);
    }
    // Build a 256-element well-conditioned integer-valued vector: a deterministic
    // shuffle of 0..255 (distinct, all exact in every supported type) rather than
    // a modulo pattern, so max/min land on interior lanes, not lane 0 / the last.
    template <typename T>
    std::vector<T> perm256()
    {
        std::vector<T> v(256);
        std::iota(v.begin(), v.end(), T(0));
        std::mt19937 rng(0x9E3779B9u);
        std::shuffle(v.begin(), v.end(), rng);
        return v;
    }
} // namespace

// ===========================================================================
// block reductions
// ===========================================================================
// blockreduce_sum<BS>(tid, shared): in-place tree sum of BS shared values; lane
// 0 holds the total. Input: exact small integers -> exact float/double sums.
TEST(internal_collectives_blockreduce, sum_float)
{
    std::vector<float> in(256);
    for(int i = 0; i < 256; ++i)
        in[i] = static_cast<float>((i % 17) + 1); // small exact integers
    run_blockreduce_sum<256, float>(in);
}
TEST(internal_collectives_blockreduce, sum_double)
{
    std::vector<double> in(256);
    for(int i = 0; i < 256; ++i)
        in[i] = static_cast<double>((i % 29) + 1);
    run_blockreduce_sum<256, double>(in);
}
TEST(internal_collectives_blockreduce, sum_int32)
{
    std::vector<int32_t> in(256);
    for(int i = 0; i < 256; ++i)
        in[i] = (i % 13) + 1;
    run_blockreduce_sum<256, int32_t>(in);
}
TEST(internal_collectives_blockreduce, sum_int64)
{
    std::vector<int64_t> in(256);
    for(int i = 0; i < 256; ++i)
        in[i] = static_cast<int64_t>(i) + 1;
    run_blockreduce_sum<256, int64_t>(in);
}
// blockreduce_max<BS>: lane 0 holds the maximum of BS shared values.
TEST(internal_collectives_blockreduce, max_float)
{
    run_blockreduce_max<256, float>(perm256<float>());
}
TEST(internal_collectives_blockreduce, max_double)
{
    run_blockreduce_max<256, double>(perm256<double>());
}
TEST(internal_collectives_blockreduce, max_int32)
{
    run_blockreduce_max<256, int32_t>(perm256<int32_t>());
}
// blockreduce_min<BS>: lane 0 holds the minimum of BS shared values.
TEST(internal_collectives_blockreduce, min_float)
{
    run_blockreduce_min<256, float>(perm256<float>());
}
TEST(internal_collectives_blockreduce, min_int32)
{
    run_blockreduce_min<256, int32_t>(perm256<int32_t>());
}
TEST(internal_collectives_blockreduce, min_int64)
{
    run_blockreduce_min<256, int64_t>(perm256<int64_t>());
}
// Non-power-of-two block size still reduces correctly (guards the i+stride<BS logic).
TEST(internal_collectives_blockreduce, sum_int32_bs192)
{
    std::vector<int32_t> in(192);
    for(int i = 0; i < 192; ++i)
        in[i] = (i % 11) + 1;
    run_blockreduce_sum<192, int32_t>(in);
}

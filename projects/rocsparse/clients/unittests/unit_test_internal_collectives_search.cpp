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
// Device (GPU) unit tests for rocSPARSE internal search / bit-scan collectives
// (rocsparse::dichotomic_search and rocsparse::popc). Split out of
// unit_test_internal_collectives.cpp by collective family.
//
#include "unit_test_utils.hpp"

#include "unit_test_internal_collectives_common.hpp"

#include "rocsparse_common.hpp" // popc
#include "rocsparse_dichotomic_search.hpp" // rocsparse::dichotomic_search

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

using rocsparse_ut::device_vector;
using rocsparse_ut::launch_single_block;
using rocsparse_ut::launch_warp_by_size;
using rocsparse_ut::to_host;

using namespace rocsparse_ut_collectives;

namespace
{
    // ---- dichotomic search -------------------------------------------------
    template <typename I, typename J>
    __global__ void k_dichotomic(J left, J right, const I* vals, I max_val, const I* arr, J* out)
    {
        const int tid = threadIdx.x;
        out[tid]      = rocsparse::dichotomic_search<I, J>(left, right, vals[tid], max_val, arr);
    }

    // ---- popc (inclusive bit-scan) -----------------------------------------
    template <uint32_t WFSZ>
    __global__ void k_popc(uint64_t mask, uint32_t* out)
    {
        const int lane = threadIdx.x;
        out[lane]      = rocsparse::popc<WFSZ>(mask, static_cast<uint32_t>(lane));
    }
    // Host reference for dichotomic_search: for a sorted "row offset" array with
    // n segments (n+1 entries), the segment index of `val` is the standard
    // upper_bound lookup (first offset strictly greater than val, minus one).
    // Values >= max_val fall in no segment and map to 0, matching the device.
    template <typename I, typename J>
    J host_dichotomic(J n, I val, I max_val, const std::vector<I>& arr)
    {
        if(val >= max_val)
            return static_cast<J>(0);
        const auto it = std::upper_bound(arr.begin(), arr.begin() + (n + 1), val);
        return static_cast<J>((it - arr.begin()) - 1);
    }

    // dichotomic_search<I,J>: for each query value, the segment index in a sorted
    // "row offset" array `arr` (n = arr.size()-1 segments). The caller supplies
    // the array and the query values; the host reference is computed per query.
    template <typename I, typename J>
    void run_dichotomic(const std::vector<I>& arr, const std::vector<I>& vals)
    {
        const J      n       = static_cast<J>(arr.size()) - 1;
        const I      max_val = arr[n];
        const size_t nq      = vals.size();

        std::vector<J> ref(nq);
        for(size_t q = 0; q < nq; ++q)
            ref[q] = host_dichotomic<I, J>(n, vals[q], max_val, arr);

        device_vector<I> d_arr(arr), d_vals(vals);
        device_vector<J> d_out(nq);
        ASSERT_NE(d_arr.ptr, nullptr);
        ASSERT_NE(d_vals.ptr, nullptr);
        ASSERT_NE(d_out.ptr, nullptr);
        ASSERT_EQ(launch_single_block(k_dichotomic<I, J>,
                                      static_cast<unsigned int>(nq),
                                      J(0),
                                      n,
                                      d_vals.ptr,
                                      max_val,
                                      d_arr.ptr,
                                      d_out.ptr),
                  hipSuccess);
        auto h = to_host(d_out);
        for(size_t q = 0; q < nq; ++q)
            expect_close(h[q], ref[q]);
    }
    // popc<WFSIZE>(mask, lid): number of set bits of `mask` in bit positions
    // [0, lid]. Host reference uses __builtin_popcountll over the same low-bit
    // window; validated for every lane of the device's wavefront.
    void run_popc(uint64_t mask)
    {
        const uint32_t          wf = require_wavefront_size();
        device_vector<uint32_t> d_out(size_t{wf});
        ASSERT_NE(d_out.ptr, nullptr);
        ASSERT_EQ(launch_warp_by_size(k_popc<32>, k_popc<64>, mask, d_out.ptr), hipSuccess);
        auto h = to_host(d_out);
        for(uint32_t lid = 0; lid < wf; ++lid)
        {
            const uint64_t lowmask  = (lid >= 63) ? ~0ull : ((1ull << (lid + 1)) - 1);
            const uint32_t expected = static_cast<uint32_t>(__builtin_popcountll(mask & lowmask));
            expect_close(h[lid], expected);
        }
    }

    // A sorted "row offset" array with a zero-length segment (3==3) and gaps.
    template <typename I>
    std::vector<I> offsets()
    {
        return std::vector<I>{0, 3, 3, 7, 12, 20};
    }
    // Every value in [0, max_val + 1] so both in-range and out-of-range queries
    // (val == max_val and val > max_val, which map to segment 0) are exercised.
    template <typename I>
    std::vector<I> queries(I max_val)
    {
        std::vector<I> v;
        for(I x = 0; x <= max_val + 1; ++x)
            v.push_back(x);
        return v;
    }
} // namespace

TEST(internal_collectives_dichotomic_search, i32_i32)
{
    run_dichotomic<int32_t, int32_t>(offsets<int32_t>(), queries<int32_t>(20));
}
TEST(internal_collectives_dichotomic_search, i64_i64)
{
    run_dichotomic<int64_t, int64_t>(offsets<int64_t>(), queries<int64_t>(20));
}
// popc<WFSIZE>(mask, lid): inclusive set-bit count over lanes [0, lid].
TEST(internal_collectives_popc, mixed_pattern)
{
    run_popc(0xB6D1E4A5F0C3927Bull); // low 32 bits used on wf32, all 64 on wf64
}
TEST(internal_collectives_popc, all_ones)
{
    run_popc(~0ull);
}

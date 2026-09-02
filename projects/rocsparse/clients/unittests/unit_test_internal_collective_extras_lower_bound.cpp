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
// Device (GPU) unit tests for rocSPARSE internal coo2csr rocsparse::lower_bound
// (binary lower-bound over a sorted array). Split out of
// unit_test_internal_collective_extras.cpp by family.
//
#include "unit_test_utils.hpp"

#include "rocsparse_common.hpp"

// coo2csr lower_bound lives in the conversion device header. The device
// unit-test target only puts library/src/{include,level1,level3} on the
// include path, so we reach it with a source-relative include (this TU lives
// in clients/unittests/). This keeps the addition local to this file and
// avoids a shared CMakeLists.txt include-dir change.
#include "../../library/src/conversion/coo2csr_device.h" // rocsparse::lower_bound

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

using rocsparse_ut::device_vector;
using rocsparse_ut::launch_single_block;
using rocsparse_ut::to_host;

namespace
{
    template <typename I, typename J>
    __global__ void k_lower_bound(const J* arr, const J* keys, I low, I high, I* out)
    {
        const int t = threadIdx.x;
        out[t]      = rocsparse::lower_bound<I, J>(arr, keys[t], low, high);
    }

    // lower_bound<I,J>(arr, key, low, high): index of the first element in the
    // sorted range [low, high) that is not less than `key` (std::lower_bound
    // semantics). The caller supplies the sorted array and the query keys; the
    // host reference is std::lower_bound over the same array.
    template <typename I, typename J>
    void run_lower_bound(const std::vector<J>& arr, const std::vector<J>& keys)
    {
        const I      high = static_cast<I>(arr.size());
        const size_t nq   = keys.size();

        std::vector<I> ref(nq);
        for(size_t q = 0; q < nq; ++q)
            ref[q]
                = static_cast<I>(std::lower_bound(arr.begin(), arr.end(), keys[q]) - arr.begin());

        device_vector<J> d_arr(arr), d_keys(keys);
        device_vector<I> d_out(nq);
        ASSERT_NE(d_arr.ptr, nullptr);
        ASSERT_NE(d_keys.ptr, nullptr);
        ASSERT_NE(d_out.ptr, nullptr);
        ASSERT_EQ(launch_single_block(k_lower_bound<I, J>,
                                      static_cast<unsigned int>(nq),
                                      d_arr.ptr,
                                      d_keys.ptr,
                                      static_cast<I>(0),
                                      high,
                                      d_out.ptr),
                  hipSuccess);
        auto h = to_host(d_out);
        for(size_t q = 0; q < nq; ++q)
            EXPECT_EQ(h[q], ref[q]) << "key=" << keys[q];
    }

    // Query every integer in [lo, hi] so below-range, in-range and above-range
    // keys are all exercised.
    template <typename J>
    std::vector<J> keys_range(J lo, J hi)
    {
        std::vector<J> k;
        for(J v = lo; v <= hi; ++v)
            k.push_back(v);
        return k;
    }
} // namespace

// Sorted array with duplicates and gaps to exercise the <-vs-<= boundary logic.
TEST(internal_collective_extras_lower_bound, i32_j32)
{
    const std::vector<int32_t> arr{0, 2, 2, 2, 5, 9, 9, 14};
    run_lower_bound<int32_t, int32_t>(arr, keys_range<int32_t>(-1, 16));
}
TEST(internal_collective_extras_lower_bound, i64_j32)
{
    const std::vector<int32_t> arr{0, 2, 2, 2, 5, 9, 9, 14};
    run_lower_bound<int64_t, int32_t>(arr, keys_range<int32_t>(-1, 16));
}
TEST(internal_collective_extras_lower_bound, i64_j64)
{
    const std::vector<int64_t> arr{0, 2, 2, 2, 5, 9, 9, 14};
    run_lower_bound<int64_t, int64_t>(arr, keys_range<int64_t>(-1, 16));
}
// All-equal array: every key <= the value lands at index 0, every key strictly
// above lands at size (past the end).
TEST(internal_collective_extras_lower_bound, all_same_array)
{
    const std::vector<int32_t> arr(8, 5);
    run_lower_bound<int32_t, int32_t>(arr, keys_range<int32_t>(3, 7));
}
// Strictly increasing (no duplicates): lower_bound is the exact position.
TEST(internal_collective_extras_lower_bound, strictly_increasing_array)
{
    const std::vector<int32_t> arr{0, 1, 2, 3, 4, 5, 6, 7};
    run_lower_bound<int32_t, int32_t>(arr, keys_range<int32_t>(-1, 8));
}

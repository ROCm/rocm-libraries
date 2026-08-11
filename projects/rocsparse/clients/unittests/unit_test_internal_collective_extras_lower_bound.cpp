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

    template <typename I, typename J>
    void run_lower_bound()
    {
        // Sorted, with duplicates and gaps to exercise <-vs-<= boundary logic.
        const std::vector<J> arr{0, 2, 2, 2, 5, 9, 9, 14};
        const I              high = static_cast<I>(arr.size());
        std::vector<J>       keys;
        for(J k = -1; k <= 16; ++k)
            keys.push_back(k);
        const size_t nq = keys.size();

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
} // namespace

TEST(internal_collective_extras_lower_bound, i32_j32)
{
    run_lower_bound<int32_t, int32_t>();
}
TEST(internal_collective_extras_lower_bound, i64_j32)
{
    run_lower_bound<int64_t, int32_t>();
}
TEST(internal_collective_extras_lower_bound, i64_j64)
{
    run_lower_bound<int64_t, int64_t>();
}

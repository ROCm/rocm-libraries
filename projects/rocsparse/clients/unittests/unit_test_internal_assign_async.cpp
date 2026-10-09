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
// Unit tests for rocSPARSE's internal batch-assign helpers (assign_async /
// assign_device_async) and the ceil_div helper in rocsparse_grid.hpp. The
// per-axis grid clamps themselves are covered by unit_test_internal_grid.cpp.
//
// FOCUS (AISPARSE-696/697): the assign kernels are a 1-D grid-stride fill. They
// launch at most assign_max_blocks (1024) blocks of assign_blocksize (256)
// threads, so one pass of the grid writes 262,144 entries and every entry past
// that is reached only by the grid-stride loop. This suite fills arrays of
// 262,144, 262,145 and more than two grid passes (2 * 262,144 + 17) entries and
// checks that every one is written: a regression in the block cap or in the
// stride leaves the tail unwritten. The largest `dest` is about 2 MB, so this
// runs on any GPU (including the 15 GB gfx1201).
//
// It also checks the size contract: n == 0 is a no-op that succeeds and n < 0
// is rejected with rocsparse_status_invalid_size.
//
// TARGET: rocsparse_assign_async.cpp is compiled into rocsparse-unit-test-device
// (ROCSPARSE_UNIT_TEST_DEVICE_LIB_SOURCES) and rocsparse_common.hpp pulls in HIP
// device intrinsics, so this file builds into the device (GPU) unit-test binary,
// NOT the host-only rocsparse-unit-test.
//

#include "unit_test_utils.hpp"

#include "rocsparse_assign_async.hpp"
#include "rocsparse_common.hpp"
#include "rocsparse_grid.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace
{
    // Entries one pass of the assign grid writes: 1024 blocks x 256 threads.
    constexpr int64_t assign_one_pass = 1024 * 256;

    // Scan a host readback and report the first index whose value differs from
    // `expected`, or -1 if all match. Cheaper (and produces a single, precise
    // gtest failure) versus wrapping hundreds of thousands of EXPECT_EQ macros
    // in a loop.
    template <typename T>
    int64_t first_mismatch(const std::vector<T>& got, T expected)
    {
        for(int64_t i = 0; i < static_cast<int64_t>(got.size()); ++i)
        {
            if(got[i] != expected)
            {
                return i;
            }
        }
        return -1;
    }
}

// ---------------------------------------------------------------------------
// Host-pure ceil helper (rocsparse_grid.hpp).
// ---------------------------------------------------------------------------

TEST(internal_assign_async, ceil_div_is_overflow_safe)
{
    EXPECT_EQ(rocsparse::ceil_div(0, 256), 0);
    EXPECT_EQ(rocsparse::ceil_div(-1, 256), 0); // count <= 0 short-circuits to 0
    EXPECT_EQ(rocsparse::ceil_div(1, 256), 1);
    EXPECT_EQ(rocsparse::ceil_div(256, 256), 1);
    EXPECT_EQ(rocsparse::ceil_div(257, 256), 2);

    // A non-positive block size is invalid; it yields 0 rather than a division
    // by zero or a negative extent.
    EXPECT_EQ(rocsparse::ceil_div(1, 0), 0);
    EXPECT_EQ(rocsparse::ceil_div(256, -1), 0);

    // The (count + block_size - 1) form would overflow int64 for a count near
    // the type maximum; the (count - 1) / block_size + 1 form used by ceil_div
    // does not.
    const int64_t max = std::numeric_limits<int64_t>::max();
    EXPECT_EQ(rocsparse::ceil_div(max, 256), (max - 1) / 256 + 1);

    // Usable in constant expressions.
    static_assert(rocsparse::ceil_div(1000, 256) == 4, "ceil_div must be constexpr");
}

// ---------------------------------------------------------------------------
// End-to-end: block cap + grid-stride on the GPU.
// ---------------------------------------------------------------------------

template <typename T>
static void run_assign_async(int64_t n)
{
    // Pre-fill with a sentinel so an unwritten entry is caught.
    rocsparse_ut::device_vector<T> d(std::vector<T>(n, static_cast<T>(-1)));
    ASSERT_NE(d.ptr, nullptr);

    const T value = static_cast<T>(42);
    ASSERT_EQ(rocsparse::assign_async<T>(n, d.ptr, value, /*stream=*/nullptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    const std::vector<T> got = rocsparse_ut::to_host(d);
    ASSERT_EQ(static_cast<int64_t>(got.size()), n);
    const int64_t bad = first_mismatch(got, value);
    EXPECT_EQ(bad, -1) << "entry not written by assign_async at index " << bad << " (n=" << n
                       << ")";
}

template <typename T>
static void run_assign_device_async(int64_t n)
{
    rocsparse_ut::device_vector<T> d(std::vector<T>(n, static_cast<T>(-1)));
    rocsparse_ut::device_vector<T> d_value(std::vector<T>(1, static_cast<T>(7)));
    ASSERT_NE(d.ptr, nullptr);
    ASSERT_NE(d_value.ptr, nullptr);

    ASSERT_EQ(rocsparse::assign_device_async<T>(n, d.ptr, d_value.ptr, /*stream=*/nullptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    const std::vector<T> got = rocsparse_ut::to_host(d);
    ASSERT_EQ(static_cast<int64_t>(got.size()), n);
    const int64_t bad = first_mismatch(got, static_cast<T>(7));
    EXPECT_EQ(bad, -1) << "entry not written by assign_device_async at index " << bad << " (n=" << n
                       << ")";
}

// Sizes: below, at and just above one grid pass, and more than two passes so
// that some threads take a third iteration of the stride loop.
static const int64_t assign_sizes[]
    = {1, 257, assign_one_pass, assign_one_pass + 1, 2 * assign_one_pass + 17};

TEST(internal_assign_async, assign_async_grid_stride_i32)
{
    for(int64_t n : assign_sizes)
    {
        SCOPED_TRACE(n);
        run_assign_async<int32_t>(n);
    }
}

TEST(internal_assign_async, assign_async_grid_stride_i64)
{
    for(int64_t n : assign_sizes)
    {
        SCOPED_TRACE(n);
        run_assign_async<int64_t>(n);
    }
}

TEST(internal_assign_async, assign_device_async_grid_stride_i32)
{
    for(int64_t n : assign_sizes)
    {
        SCOPED_TRACE(n);
        run_assign_device_async<int32_t>(n);
    }
}

TEST(internal_assign_async, assign_device_async_grid_stride_i64)
{
    for(int64_t n : assign_sizes)
    {
        SCOPED_TRACE(n);
        run_assign_device_async<int64_t>(n);
    }
}

// ---------------------------------------------------------------------------
// Size contract.
// ---------------------------------------------------------------------------

TEST(internal_assign_async, zero_size_is_a_successful_no_op)
{
    rocsparse_ut::device_vector<int32_t> d(std::vector<int32_t>(4, -1));
    rocsparse_ut::device_vector<int32_t> d_value(std::vector<int32_t>(1, 7));
    ASSERT_NE(d.ptr, nullptr);

    EXPECT_EQ(rocsparse::assign_async<int32_t>(0, d.ptr, 42, nullptr), rocsparse_status_success);
    EXPECT_EQ(rocsparse::assign_device_async<int32_t>(0, d.ptr, d_value.ptr, nullptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(first_mismatch(rocsparse_ut::to_host(d), static_cast<int32_t>(-1)), -1);
}

TEST(internal_assign_async, negative_size_is_invalid_size)
{
    rocsparse_ut::device_vector<int32_t> d(std::vector<int32_t>(4, -1));
    rocsparse_ut::device_vector<int32_t> d_value(std::vector<int32_t>(1, 7));
    ASSERT_NE(d.ptr, nullptr);

    EXPECT_EQ(rocsparse::assign_async<int32_t>(-1, d.ptr, 42, nullptr),
              rocsparse_status_invalid_size);
    EXPECT_EQ(rocsparse::assign_device_async<int32_t>(-1, d.ptr, d_value.ptr, nullptr),
              rocsparse_status_invalid_size);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(first_mismatch(rocsparse_ut::to_host(d), static_cast<int32_t>(-1)), -1);
}

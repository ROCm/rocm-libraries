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
// assign_device_async) and the shared grid-size clamp helpers they rely on
// (get_grid_size / get_batch_grid_size / ceil_div in rocsparse_common.hpp).
//
// FOCUS (AISPARSE-696/697): the assign kernels launch on grid.y and CLAMP that
// extent to max_batch_grid_size (65535, the hardware grid.y/z limit). To stay
// correct for batch counts ABOVE that clamp, the kernels grid-stride over the
// batch index (batch_index += hipGridDim_y). This suite drives n = 70000 >
// 65535 so the launch grid is clamped and the tail [65535, n) is reached ONLY
// by the grid-stride loop: a regression in either the clamp or the stride
// leaves entries past 65535 unwritten. `dest` holds n scalars, so the footprint
// is well under a megabyte and this runs on any GPU (including the 15 GB
// gfx1201).
//
// TARGET: rocsparse_assign_async.cpp is compiled into rocsparse-unit-test-device
// (ROCSPARSE_UNIT_TEST_DEVICE_LIB_SOURCES) and rocsparse_common.hpp pulls in HIP
// device intrinsics, so this file builds into the device (GPU) unit-test binary,
// NOT the host-only rocsparse-unit-test.
//

#include "unit_test_utils.hpp"

#include "rocsparse_assign_async.hpp"
#include "rocsparse_common.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace
{
    // A batch count strictly larger than the grid.y clamp, so the launch grid is
    // clamped to max_batch_grid_size and the tail is covered only by the kernel's
    // grid-stride loop.
    constexpr int64_t beyond_clamp = rocsparse::max_batch_grid_size + 4465; // 70000

    // Scan a host readback and report the first index whose value differs from
    // `expected`, or -1 if all match. Cheaper (and produces a single, precise
    // gtest failure) versus wrapping 70000 EXPECT_EQ macros in a loop.
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
// Host-pure clamp / ceil helpers (rocsparse_common.hpp).
// ---------------------------------------------------------------------------

TEST(internal_assign_async, get_grid_size_clamps_to_axis_maximum)
{
    // At or below the limit: passed through unchanged.
    EXPECT_EQ(rocsparse::get_grid_size(int64_t{0}, rocsparse::max_batch_grid_size), 0u);
    EXPECT_EQ(rocsparse::get_grid_size(int64_t{1}, rocsparse::max_batch_grid_size), 1u);
    EXPECT_EQ(rocsparse::get_grid_size(int64_t{65535}, rocsparse::max_batch_grid_size), 65535u);

    // Above the limit: clamped down to the limit.
    EXPECT_EQ(rocsparse::get_grid_size(int64_t{65536}, rocsparse::max_batch_grid_size), 65535u);
    EXPECT_EQ(rocsparse::get_grid_size(int64_t{1} << 40, rocsparse::max_batch_grid_size), 65535u);

    // The grid.x axis clamps at its own (much larger) maximum.
    EXPECT_EQ(rocsparse::get_grid_size(int64_t{5}, rocsparse::max_grid_size_x), 5u);
    EXPECT_EQ(rocsparse::get_grid_size(int64_t{1} << 40, rocsparse::max_grid_size_x),
              static_cast<uint32_t>(rocsparse::max_grid_size_x));
}

TEST(internal_assign_async, get_batch_grid_size_clamps_at_65535)
{
    EXPECT_EQ(rocsparse::get_batch_grid_size(1), 1u);
    EXPECT_EQ(rocsparse::get_batch_grid_size(65535), 65535u);
    EXPECT_EQ(rocsparse::get_batch_grid_size(65536), 65535u);
    EXPECT_EQ(rocsparse::get_batch_grid_size(int64_t{1} << 32), 65535u);
}

TEST(internal_assign_async, ceil_div_is_overflow_safe)
{
    EXPECT_EQ(rocsparse::ceil_div(0, 256), 0);
    EXPECT_EQ(rocsparse::ceil_div(-1, 256), 0); // count <= 0 short-circuits to 0
    EXPECT_EQ(rocsparse::ceil_div(1, 256), 1);
    EXPECT_EQ(rocsparse::ceil_div(256, 256), 1);
    EXPECT_EQ(rocsparse::ceil_div(257, 256), 2);

    // The (count + block_size - 1) form would overflow int64 for a count near
    // the type maximum; the (count - 1) / block_size + 1 form used by ceil_div
    // does not.
    const int64_t max = std::numeric_limits<int64_t>::max();
    EXPECT_EQ(rocsparse::ceil_div(max, 256), (max - 1) / 256 + 1);
}

// ---------------------------------------------------------------------------
// End-to-end: grid.y clamp + grid-stride on the GPU.
// ---------------------------------------------------------------------------

template <typename T>
static void run_assign_async_beyond_clamp()
{
    const int64_t n = beyond_clamp;
    ASSERT_GT(n, rocsparse::max_batch_grid_size);

    // Pre-fill with a sentinel so an unwritten tail entry is caught.
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
                       << ", grid.y clamp=" << rocsparse::max_batch_grid_size << ")";
}

TEST(internal_assign_async, assign_async_grid_stride_beyond_clamp_i32)
{
    run_assign_async_beyond_clamp<int32_t>();
}

TEST(internal_assign_async, assign_async_grid_stride_beyond_clamp_i64)
{
    run_assign_async_beyond_clamp<int64_t>();
}

template <typename T>
static void run_assign_device_async_beyond_clamp()
{
    const int64_t n = beyond_clamp;
    ASSERT_GT(n, rocsparse::max_batch_grid_size);

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
                       << ", grid.y clamp=" << rocsparse::max_batch_grid_size << ")";
}

TEST(internal_assign_async, assign_device_async_grid_stride_beyond_clamp_i32)
{
    run_assign_device_async_beyond_clamp<int32_t>();
}

TEST(internal_assign_async, assign_device_async_grid_stride_beyond_clamp_i64)
{
    run_assign_device_async_beyond_clamp<int64_t>();
}

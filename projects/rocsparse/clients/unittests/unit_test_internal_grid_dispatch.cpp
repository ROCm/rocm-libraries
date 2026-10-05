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
// Unit tests for rocsparse::dispatch_grid_stride_x in rocsparse_grid.hpp.
//
// dispatch_grid_stride_x picks the straight-line variant (std::false_type)
// while count fits the grid.x cap and the grid-stride variant (std::true_type)
// once it does not. These tests check that choice and the grid it passes at
// the boundary, count == cap and count == cap + 1, against both caps: the
// derived (2^32 - 1) / block_size bound, which binds at block size 32 and up,
// and maxGridSize[0], which binds at block size 1. No kernel is launched.
//
#include "unit_test_utils.hpp"

#include "rocsparse_grid.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

using namespace rocsparse_ut;

namespace
{
    constexpr int64_t BLOCK_SIZES[] = {1, 32, 64, 128, 256, 512, 1024};

    struct Dispatch
    {
        int      calls       = 0;
        bool     grid_stride = false;
        uint32_t grid        = 0;
    };

    template <typename J>
    Dispatch dispatch(rocsparse_handle handle, J count, int64_t block_size)
    {
        Dispatch         d;
        rocsparse_status status = rocsparse::dispatch_grid_stride_x(
            handle, count, block_size, [&](auto grid_stride, uint32_t grid) {
                ++d.calls;
                d.grid_stride = decltype(grid_stride)::value;
                d.grid        = grid;
                return rocsparse_status_success;
            });
        EXPECT_EQ(status, rocsparse_status_success);
        return d;
    }

    int64_t grid_x_cap(rocsparse_handle handle, int64_t block_size)
    {
        return std::min(static_cast<int64_t>(handle->properties.maxGridSize[0]),
                        rocsparse::dispatch_limit_x(block_size));
    }
}

class internal_grid_dispatch : public HandleTest
{
};

// count == cap launches the straight-line variant on the full grid; cap + 1
// launches the grid-stride variant on the clamped grid.
TEST_F(internal_grid_dispatch, boundary)
{
    for(const int64_t bs : BLOCK_SIZES)
    {
        const int64_t cap = grid_x_cap(handle, bs);

        const Dispatch at = dispatch(handle, cap, bs);
        EXPECT_EQ(at.calls, 1) << "block size " << bs;
        EXPECT_FALSE(at.grid_stride) << "block size " << bs << ", count " << cap;
        EXPECT_EQ(static_cast<int64_t>(at.grid), cap) << "block size " << bs;

        const Dispatch over = dispatch(handle, cap + 1, bs);
        EXPECT_EQ(over.calls, 1) << "block size " << bs;
        EXPECT_TRUE(over.grid_stride) << "block size " << bs << ", count " << cap + 1;
        EXPECT_EQ(static_cast<int64_t>(over.grid), cap) << "block size " << bs;
    }
}

// The derived bound binds below maxGridSize[0] at 256 threads, and the device
// cap binds at 1 thread.
TEST_F(internal_grid_dispatch, boundary_concrete_caps)
{
    ASSERT_GT(static_cast<int64_t>(handle->properties.maxGridSize[0]), 16777215);
    EXPECT_EQ(grid_x_cap(handle, 256), 16777215);
    EXPECT_EQ(grid_x_cap(handle, 1), static_cast<int64_t>(handle->properties.maxGridSize[0]));

    EXPECT_FALSE(dispatch(handle, int32_t(16777215), 256).grid_stride);
    EXPECT_TRUE(dispatch(handle, int32_t(16777216), 256).grid_stride);
}

// The choice does not depend on the count type. A 32-bit count crosses the
// bound as readily as a 64-bit one.
TEST_F(internal_grid_dispatch, count_types)
{
    constexpr int64_t bs  = 1024;
    const int64_t     cap = grid_x_cap(handle, bs);

    EXPECT_FALSE(dispatch(handle, static_cast<int32_t>(cap), bs).grid_stride);
    EXPECT_TRUE(dispatch(handle, static_cast<int32_t>(cap + 1), bs).grid_stride);
    EXPECT_FALSE(dispatch(handle, static_cast<uint32_t>(cap), bs).grid_stride);
    EXPECT_TRUE(dispatch(handle, static_cast<uint32_t>(cap + 1), bs).grid_stride);
    EXPECT_FALSE(dispatch(handle, static_cast<int64_t>(cap), bs).grid_stride);
    EXPECT_TRUE(dispatch(handle, static_cast<int64_t>(cap + 1), bs).grid_stride);
    EXPECT_FALSE(dispatch(handle, static_cast<uint64_t>(cap), bs).grid_stride);
    EXPECT_TRUE(dispatch(handle, static_cast<uint64_t>(cap + 1), bs).grid_stride);
}

// Small counts and a zero count stay on the straight-line variant with the
// grid equal to count.
TEST_F(internal_grid_dispatch, small_counts)
{
    for(const int64_t bs : BLOCK_SIZES)
    {
        for(const int64_t count : {int64_t(0), int64_t(1), int64_t(1000)})
        {
            const Dispatch d = dispatch(handle, count, bs);
            EXPECT_FALSE(d.grid_stride) << "block size " << bs << ", count " << count;
            EXPECT_EQ(static_cast<int64_t>(d.grid), count) << "block size " << bs;
        }
    }
}

// The launcher's status is returned to the caller.
TEST_F(internal_grid_dispatch, returns_launch_status)
{
    EXPECT_EQ(rocsparse::dispatch_grid_stride_x(
                  handle,
                  int64_t(1),
                  256,
                  [](auto, uint32_t) { return rocsparse_status_internal_error; }),
              rocsparse_status_internal_error);
}

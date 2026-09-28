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
// Unit tests for the per-axis grid-size helpers in rocsparse_grid.hpp:
//   clamp_grid_extent, dispatch_limit_x, get_grid_size_x/y/z.
//
// The host tests check the returned extents against the handle's device
// properties. The device tests launch the largest grid.x that get_grid_size_x
// returns and check that its last block runs, then check that a grid-stride
// kernel on a clamped grid covers every element exactly once. They do not
// launch an over-limit grid: whether the runtime rejects or silently truncates
// such a launch depends on the ROCm version.
//
#include "unit_test_utils.hpp"

#include "rocsparse_grid.hpp"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    // The dispatch packet stores grid_size_x in work-items as a uint32_t.
    constexpr int64_t DISPATCH_WORK_ITEMS = 4294967295LL;

    constexpr int64_t BLOCK_SIZES[] = {1, 32, 64, 128, 256, 512, 1024};

    constexpr int64_t HUGE_COUNT = int64_t(1) << 40;

    int64_t max_grid(rocsparse_handle handle, int axis)
    {
        return static_cast<int64_t>(handle->properties.maxGridSize[axis]);
    }

    int64_t expected_grid_x(rocsparse_handle handle, int64_t block_size)
    {
        return std::min(max_grid(handle, 0), DISPATCH_WORK_ITEMS / block_size);
    }

    template <unsigned int BLOCKSIZE>
    __launch_bounds__(BLOCKSIZE) __global__
        void mark_blocks_kernel(uint32_t expected_last, uint32_t* ran_last, uint32_t* block_count)
    {
        if(hipThreadIdx_x != 0)
        {
            return;
        }

        atomicAdd(block_count, 1u);

        if(hipBlockIdx_x == expected_last)
        {
            *ran_last = 1;
        }
    }

    template <unsigned int BLOCKSIZE>
    __launch_bounds__(BLOCKSIZE) __global__ void grid_stride_kernel(int64_t count, uint32_t* hits)
    {
        const int64_t stride = static_cast<int64_t>(hipGridDim_x) * BLOCKSIZE;
        for(int64_t i = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x; i < count;
            i += stride)
        {
            atomicAdd(&hits[i], 1u);
        }
    }

    // Launch get_grid_size_x(huge) blocks of BLOCKSIZE threads and check that
    // every block, including the last, ran. expected_last comes from the host:
    // a truncated dispatch would also report a truncated gridDim.x.
    template <unsigned int BLOCKSIZE>
    void check_largest_grid_x_runs(rocsparse_handle handle)
    {
        const uint32_t grid = rocsparse::get_grid_size_x(handle, HUGE_COUNT, BLOCKSIZE);
        ASSERT_EQ(static_cast<int64_t>(grid), expected_grid_x(handle, BLOCKSIZE));
        ASSERT_GT(grid, 0u);

        device_vector<uint32_t> flags{std::vector<uint32_t>{0, 0}};
        ASSERT_TRUE(flags.ptr);

        (void)hipGetLastError();
        hipLaunchKernelGGL((mark_blocks_kernel<BLOCKSIZE>),
                           dim3(grid),
                           dim3(BLOCKSIZE),
                           0,
                           0,
                           grid - 1,
                           flags.ptr,
                           flags.ptr + 1);
        ASSERT_EQ(hipGetLastError(), hipSuccess);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const std::vector<uint32_t> host = to_host(flags);
        EXPECT_EQ(host[0], 1u) << "last block " << grid - 1 << " did not run";
        EXPECT_EQ(host[1], grid);
    }
}

class internal_grid : public HandleTest
{
};

// ===========================================================================
// dispatch_limit_x / clamp_grid_extent
// ===========================================================================

// dispatch_limit_x is floor((2^32 - 1) / block_size), the table in the header.
TEST(internal_grid_limits, dispatch_limit_x_values)
{
    static_assert(rocsparse::dispatch_limit_x(1) == 4294967295LL, "");
    static_assert(rocsparse::dispatch_limit_x(64) == 67108863LL, "");
    static_assert(rocsparse::dispatch_limit_x(128) == 33554431LL, "");
    static_assert(rocsparse::dispatch_limit_x(256) == 16777215LL, "");
    static_assert(rocsparse::dispatch_limit_x(512) == 8388607LL, "");
    static_assert(rocsparse::dispatch_limit_x(1024) == 4194303LL, "");

    for(int64_t bs = 1; bs <= 1024; ++bs)
    {
        const int64_t limit = rocsparse::dispatch_limit_x(bs);
        EXPECT_LE(limit * bs, DISPATCH_WORK_ITEMS) << "block size " << bs;
        EXPECT_GT((limit + 1) * bs, DISPATCH_WORK_ITEMS) << "block size " << bs;
    }
}

// clamp_grid_extent returns min(count, max_extent) for every count type.
TEST(internal_grid_limits, clamp_grid_extent_values)
{
    EXPECT_EQ(rocsparse::clamp_grid_extent(5, 3), 3u);
    EXPECT_EQ(rocsparse::clamp_grid_extent(3, 5), 3u);
    EXPECT_EQ(rocsparse::clamp_grid_extent(3, 3), 3u);
    EXPECT_EQ(rocsparse::clamp_grid_extent(0, 3), 0u);
    EXPECT_EQ(rocsparse::clamp_grid_extent(int32_t(7), 4), 4u);
    EXPECT_EQ(rocsparse::clamp_grid_extent(uint32_t(7), 4), 4u);
    EXPECT_EQ(rocsparse::clamp_grid_extent(uint64_t(7), 4), 4u);
    EXPECT_EQ(rocsparse::clamp_grid_extent(HUGE_COUNT, DISPATCH_WORK_ITEMS), 4294967295u);
    EXPECT_EQ(
        rocsparse::clamp_grid_extent(std::numeric_limits<uint32_t>::max(), DISPATCH_WORK_ITEMS),
        4294967295u);
}

// ===========================================================================
// get_grid_size_x
// ===========================================================================

// A huge count clamps to min(maxGridSize[0], (2^32 - 1) / block_size).
TEST_F(internal_grid, grid_x_huge_count_clamps)
{
    for(const int64_t bs : BLOCK_SIZES)
    {
        EXPECT_EQ(static_cast<int64_t>(rocsparse::get_grid_size_x(handle, HUGE_COUNT, bs)),
                  expected_grid_x(handle, bs))
            << "block size " << bs;
    }
}

// The derived bound, not maxGridSize[0], is what binds at common block sizes.
TEST_F(internal_grid, grid_x_concrete_values)
{
    ASSERT_GE(max_grid(handle, 0), 16777215);

    EXPECT_EQ(rocsparse::get_grid_size_x(handle, HUGE_COUNT, 256), 16777215u);
    EXPECT_EQ(rocsparse::get_grid_size_x(handle, HUGE_COUNT, 1024), 4194303u);
}

// At block size 1 the derived bound is 2^32 - 1, which no int maxGridSize[0]
// reaches, so the device cap wins.
TEST_F(internal_grid, grid_x_block_size_one_device_cap_wins)
{
    ASSERT_LT(max_grid(handle, 0), rocsparse::dispatch_limit_x(1));

    EXPECT_EQ(static_cast<int64_t>(rocsparse::get_grid_size_x(handle, HUGE_COUNT, 1)),
              max_grid(handle, 0));
}

// Counts at or below the bound pass through; one past the bound clamps to it.
TEST_F(internal_grid, grid_x_small_counts_pass_through)
{
    for(const int64_t bs : BLOCK_SIZES)
    {
        const int64_t bound = expected_grid_x(handle, bs);
        for(const int64_t count : {int64_t(1), int64_t(2), int64_t(255), int64_t(65536), bound})
        {
            EXPECT_EQ(static_cast<int64_t>(rocsparse::get_grid_size_x(handle, count, bs)), count)
                << "block size " << bs << ", count " << count;
        }
        EXPECT_EQ(static_cast<int64_t>(rocsparse::get_grid_size_x(handle, bound + 1, bs)), bound)
            << "block size " << bs;
    }
}

// A zero count returns a zero extent. A zero-sized grid is an invalid launch
// configuration, so callers must skip the launch rather than pass it through.
TEST_F(internal_grid, grid_x_zero_count)
{
    for(const int64_t bs : BLOCK_SIZES)
    {
        EXPECT_EQ(rocsparse::get_grid_size_x(handle, int32_t(0), bs), 0u);
        EXPECT_EQ(rocsparse::get_grid_size_x(handle, int64_t(0), bs), 0u);
        EXPECT_EQ(rocsparse::get_grid_size_x(handle, uint32_t(0), bs), 0u);
    }
}

// get_grid_size_x(huge) * block_size never exceeds 2^32 - 1 work-items.
TEST_F(internal_grid, grid_x_times_block_size_fits_dispatch)
{
    for(int64_t bs = 1; bs <= 1024; ++bs)
    {
        const int64_t grid = rocsparse::get_grid_size_x(handle, HUGE_COUNT, bs);
        EXPECT_LE(grid * bs, DISPATCH_WORK_ITEMS) << "block size " << bs;
    }
}

// ===========================================================================
// get_grid_size_x with each count type
// ===========================================================================
template <typename T>
class internal_grid_count_type : public HandleTest
{
};

using grid_count_types = ::testing::Types<int32_t, int64_t, uint32_t, uint64_t>;
TYPED_TEST_SUITE(internal_grid_count_type, grid_count_types);

// The largest value of every count type clamps to the bound; a small count
// passes through unchanged.
TYPED_TEST(internal_grid_count_type, grid_x)
{
    using J = TypeParam;

    const J large = (sizeof(J) == 8) ? static_cast<J>(HUGE_COUNT) : std::numeric_limits<J>::max();

    for(const int64_t bs : BLOCK_SIZES)
    {
        const int64_t bound = expected_grid_x(this->handle, bs);
        EXPECT_EQ(static_cast<int64_t>(rocsparse::get_grid_size_x(this->handle, large, bs)),
                  std::min(static_cast<int64_t>(large), bound))
            << "block size " << bs;
        EXPECT_EQ(rocsparse::get_grid_size_x(this->handle, J(1000), bs), 1000u)
            << "block size " << bs;
    }
}

// ===========================================================================
// get_grid_size_y / get_grid_size_z
// ===========================================================================

// grid.y and grid.z clamp to maxGridSize[1] and [2]; smaller counts pass through.
TEST_F(internal_grid, grid_yz_clamp_to_max_grid_size)
{
    const int64_t max_y = max_grid(handle, 1);
    const int64_t max_z = max_grid(handle, 2);
    ASSERT_GT(max_y, 1);
    ASSERT_GT(max_z, 1);

    EXPECT_EQ(static_cast<int64_t>(rocsparse::get_grid_size_y(handle, HUGE_COUNT)), max_y);
    EXPECT_EQ(static_cast<int64_t>(rocsparse::get_grid_size_y(handle, max_y + 1)), max_y);
    EXPECT_EQ(static_cast<int64_t>(rocsparse::get_grid_size_y(handle, max_y)), max_y);
    EXPECT_EQ(static_cast<int64_t>(rocsparse::get_grid_size_y(handle, max_y - 1)), max_y - 1);
    EXPECT_EQ(rocsparse::get_grid_size_y(handle, int32_t(1)), 1u);
    EXPECT_EQ(rocsparse::get_grid_size_y(handle, uint32_t(0)), 0u);
    EXPECT_EQ(static_cast<int64_t>(
                  rocsparse::get_grid_size_y(handle, std::numeric_limits<int32_t>::max())),
              std::min<int64_t>(max_y, std::numeric_limits<int32_t>::max()));

    EXPECT_EQ(static_cast<int64_t>(rocsparse::get_grid_size_z(handle, HUGE_COUNT)), max_z);
    EXPECT_EQ(static_cast<int64_t>(rocsparse::get_grid_size_z(handle, max_z + 1)), max_z);
    EXPECT_EQ(static_cast<int64_t>(rocsparse::get_grid_size_z(handle, max_z)), max_z);
    EXPECT_EQ(static_cast<int64_t>(rocsparse::get_grid_size_z(handle, max_z - 1)), max_z - 1);
    EXPECT_EQ(rocsparse::get_grid_size_z(handle, int32_t(1)), 1u);
    EXPECT_EQ(rocsparse::get_grid_size_z(handle, uint32_t(0)), 0u);
    EXPECT_EQ(static_cast<int64_t>(
                  rocsparse::get_grid_size_z(handle, std::numeric_limits<int32_t>::max())),
              std::min<int64_t>(max_z, std::numeric_limits<int32_t>::max()));
}

// ===========================================================================
// Device: the bound is launchable and grid-striding over it is complete
// ===========================================================================

TEST_F(internal_grid, device_largest_grid_x_runs_1024)
{
    check_largest_grid_x_runs<1024>(handle);
}

TEST_F(internal_grid, device_largest_grid_x_runs_256)
{
    check_largest_grid_x_runs<256>(handle);
}

// A count clamped to a tiny extent is still covered exactly once when the
// kernel grid-strides, including the unclamped case.
TEST_F(internal_grid, device_grid_stride_covers_count)
{
    constexpr unsigned int BLOCKSIZE = 64;
    constexpr int64_t      COUNT     = 1000;
    constexpr int64_t      BLOCKS    = (COUNT - 1) / BLOCKSIZE + 1;

    for(const int64_t max_extent : {int64_t(1), int64_t(3), BLOCKS - 1, BLOCKS, BLOCKS + 5})
    {
        const uint32_t grid = rocsparse::clamp_grid_extent(BLOCKS, max_extent);
        ASSERT_EQ(static_cast<int64_t>(grid), std::min(BLOCKS, max_extent));

        device_vector<uint32_t> hits{std::vector<uint32_t>(COUNT, 0)};
        ASSERT_TRUE(hits.ptr);

        (void)hipGetLastError();
        hipLaunchKernelGGL(
            (grid_stride_kernel<BLOCKSIZE>), dim3(grid), dim3(BLOCKSIZE), 0, 0, COUNT, hits.ptr);
        ASSERT_EQ(hipGetLastError(), hipSuccess);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        const std::vector<uint32_t> host = to_host(hits);
        for(int64_t i = 0; i < COUNT; ++i)
        {
            ASSERT_EQ(host[i], 1u) << "element " << i << ", grid " << grid;
        }
    }
}

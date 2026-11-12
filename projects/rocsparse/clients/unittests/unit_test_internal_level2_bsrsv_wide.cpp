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

// Wide bsrsv solve kernels (bsrsv_device_large.h).
//
// The matrix is an 8-by-8 block bidiagonal system of 2-by-2 blocks. Diagonal
// blocks are I and off-diagonal blocks are -I, so each of the two scalar
// chains is a bidiagonal of length 8. With alpha = 1 and x = 1 the solution
// is exact in float:
//
//   lower: y(r) = r / block_dim + 1
//   upper: y(r) = mb - r / block_dim
//
// Direct launches pass an undersized grid, which is the case the host creates
// when get_grid_size_x clamps, and they instantiate (i32, i32), (i64, i32)
// and (i64, i64). The public rocsparse_sbsrsv_solve tests shrink
// maxGridSize[0] so the host is the one that picks the wide kernel, including
// the transposed gather.

#include "unit_test_utils.hpp"

#include "unit_test_grid_clamp.hpp"

#include "bsrsv_device_large.h"

#include "rocsparse.h"

#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <vector>

namespace
{
    using test_T = float;

    constexpr int32_t  kMb       = 8;
    constexpr int32_t  kBlockDim = 2;
    constexpr int32_t  kRows     = kMb * kBlockDim;
    constexpr uint32_t kBlock    = 128;

    void append_block(std::vector<test_T>& val, test_T diag)
    {
        val.push_back(diag);
        val.push_back(static_cast<test_T>(0));
        val.push_back(static_cast<test_T>(0));
        val.push_back(diag);
    }

    template <typename I, typename J>
    void build_matrix(bool                 lower,
                      std::vector<I>&      row_ptr,
                      std::vector<J>&      col_ind,
                      std::vector<test_T>& val,
                      std::vector<J>&      map,
                      rocsparse_index_base base         = rocsparse_index_base_zero,
                      int32_t              singular_row = -1)
    {
        row_ptr.assign(1, static_cast<I>(0));
        col_ind.clear();
        val.clear();
        map.resize(static_cast<size_t>(kMb));

        for(int32_t row = 0; row < kMb; ++row)
        {
            if(lower)
            {
                if(row > 0)
                {
                    col_ind.push_back(static_cast<J>(row - 1));
                    append_block(val, static_cast<test_T>(-1));
                }
                col_ind.push_back(static_cast<J>(row));
                append_block(
                    val, (row == singular_row) ? static_cast<test_T>(0) : static_cast<test_T>(1));
                map[static_cast<size_t>(row)] = static_cast<J>(row);
            }
            else
            {
                col_ind.push_back(static_cast<J>(row));
                append_block(
                    val, (row == singular_row) ? static_cast<test_T>(0) : static_cast<test_T>(1));
                if(row + 1 < kMb)
                {
                    col_ind.push_back(static_cast<J>(row + 1));
                    append_block(val, static_cast<test_T>(-1));
                }
                map[static_cast<size_t>(row)] = static_cast<J>(kMb - 1 - row);
            }
            row_ptr.push_back(static_cast<I>(col_ind.size()));
        }

        if(base == rocsparse_index_base_one)
        {
            for(auto& r : row_ptr)
            {
                r = static_cast<I>(r + 1);
            }
            for(auto& c : col_ind)
            {
                c = static_cast<J>(c + 1);
            }
        }
    }

    test_T expected(bool lower, int32_t row)
    {
        const int32_t block_row = row / kBlockDim;
        return lower ? static_cast<test_T>(block_row + 1) : static_cast<test_T>(kMb - block_row);
    }

    template <uint32_t WFSIZE, typename I, typename J>
    void launch_wide(bool                 lower,
                     uint32_t             grid_x,
                     std::vector<test_T>& y_out,
                     J&                   pivot,
                     rocsparse_index_base base         = rocsparse_index_base_zero,
                     int32_t              singular_row = -1)
    {
        std::vector<I>      row_ptr;
        std::vector<J>      col_ind;
        std::vector<test_T> val;
        std::vector<J>      map;
        build_matrix(lower, row_ptr, col_ind, val, map, base, singular_row);

        const std::vector<test_T> x(static_cast<size_t>(kRows), static_cast<test_T>(1));
        const std::vector<test_T> y(static_cast<size_t>(kRows), static_cast<test_T>(-7));

        rocsparse_ut::device_vector<I>      d_row_ptr(row_ptr);
        rocsparse_ut::device_vector<J>      d_col_ind(col_ind);
        rocsparse_ut::device_vector<test_T> d_val(val);
        rocsparse_ut::device_vector<J>      d_map(map);
        rocsparse_ut::device_vector<test_T> d_x(x);
        rocsparse_ut::device_vector<test_T> d_y(y);
        rocsparse_ut::device_vector<int>    d_done(std::vector<int>(static_cast<size_t>(kMb), 0));
        rocsparse_ut::device_vector<J>      d_pivot(std::vector<J>{std::numeric_limits<J>::max()});

        ASSERT_NE(d_row_ptr.ptr, nullptr);
        ASSERT_NE(d_col_ind.ptr, nullptr);
        ASSERT_NE(d_val.ptr, nullptr);
        ASSERT_NE(d_map.ptr, nullptr);
        ASSERT_NE(d_x.ptr, nullptr);
        ASSERT_NE(d_y.ptr, nullptr);
        ASSERT_NE(d_done.ptr, nullptr);
        ASSERT_NE(d_pivot.ptr, nullptr);

        const test_T alpha       = static_cast<test_T>(1);
        const auto   alpha_union = rocsparse::const_host_device_scalar<test_T>(alpha);

        if(lower)
        {
            hipLaunchKernelGGL(
                (rocsparse::bsrsv_lower_general_wide<kBlock, WFSIZE, false, test_T, I, J>),
                dim3(grid_x),
                dim3(kBlock),
                0,
                0,
                static_cast<J>(kMb),
                alpha_union,
                d_row_ptr.ptr,
                d_col_ind.ptr,
                d_val.ptr,
                kBlockDim,
                d_x.ptr,
                d_y.ptr,
                d_done.ptr,
                d_map.ptr,
                d_pivot.ptr,
                base,
                rocsparse_diag_type_non_unit,
                rocsparse_direction_row,
                true);
        }
        else
        {
            hipLaunchKernelGGL(
                (rocsparse::bsrsv_upper_general_wide<kBlock, WFSIZE, false, test_T, I, J>),
                dim3(grid_x),
                dim3(kBlock),
                0,
                0,
                static_cast<J>(kMb),
                alpha_union,
                d_row_ptr.ptr,
                d_col_ind.ptr,
                d_val.ptr,
                kBlockDim,
                d_x.ptr,
                d_y.ptr,
                d_done.ptr,
                d_map.ptr,
                d_pivot.ptr,
                base,
                rocsparse_diag_type_non_unit,
                rocsparse_direction_row,
                true);
        }

        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        y_out = rocsparse_ut::to_host(d_y);
        pivot = rocsparse_ut::to_host(d_pivot)[0];
    }

    // The wide kernels assume WFSIZE equals the device wavefront, so the tests
    // run the size the device reports. A 32-lane launch on a wave64 device, or
    // the reverse, can leave a warp spinning on a row another warp still owns.
    uint32_t device_wavefront()
    {
        int device = 0;
        int warp   = 0;
        hipGetDevice(&device);
        hipDeviceGetAttribute(&warp, hipDeviceAttributeWarpSize, device);
        return (warp == 64) ? 64u : 32u;
    }

    template <uint32_t WFSIZE, typename I, typename J>
    void expect_wide(bool lower, uint32_t grid_x)
    {
        std::vector<test_T> y;
        J                   pivot{};
        ASSERT_NO_FATAL_FAILURE((launch_wide<WFSIZE, I, J>(lower, grid_x, y, pivot)));
        ASSERT_EQ(static_cast<int>(y.size()), kRows);
        EXPECT_EQ(pivot, std::numeric_limits<J>::max());
        for(int32_t row = 0; row < kRows; ++row)
        {
            EXPECT_EQ(y[static_cast<size_t>(row)], expected(lower, row)) << "row " << row;
        }
    }

    template <uint32_t WFSIZE, typename I, typename J>
    void expect_singular_pivot(bool                 lower,
                               rocsparse_index_base base,
                               int32_t              singular_row,
                               uint32_t             grid_x)
    {
        std::vector<test_T> y;
        J                   pivot{};
        ASSERT_NO_FATAL_FAILURE(
            (launch_wide<WFSIZE, I, J>(lower, grid_x, y, pivot, base, singular_row)));
        // The kernel reports the block row in the caller's index base.
        EXPECT_EQ(pivot, static_cast<J>(singular_row + static_cast<int32_t>(base)));
    }

    template <typename I, typename J>
    void expect_singular_pivot_device(bool                 lower,
                                      rocsparse_index_base base,
                                      int32_t              singular_row,
                                      uint32_t             grid_x)
    {
        if(device_wavefront() == 64)
        {
            expect_singular_pivot<64, I, J>(lower, base, singular_row, grid_x);
        }
        else
        {
            expect_singular_pivot<32, I, J>(lower, base, singular_row, grid_x);
        }
    }

    template <typename I, typename J>
    void expect_wide_device(bool lower, uint32_t grid_x)
    {
        if(device_wavefront() == 64)
        {
            expect_wide<64, I, J>(lower, grid_x);
        }
        else
        {
            expect_wide<32, I, J>(lower, grid_x);
        }
    }

    void solve_public(bool                 lower,
                      rocsparse_operation  trans,
                      int                  max_grid_x,
                      std::vector<test_T>& y_out)
    {
        std::vector<rocsparse_int> row_ptr;
        std::vector<rocsparse_int> col_ind;
        std::vector<test_T>        val;
        std::vector<rocsparse_int> map;
        build_matrix(lower, row_ptr, col_ind, val, map);

        const rocsparse_int       nnzb = static_cast<rocsparse_int>(col_ind.size());
        const std::vector<test_T> x(static_cast<size_t>(kRows), static_cast<test_T>(1));
        const std::vector<test_T> y(static_cast<size_t>(kRows), static_cast<test_T>(-7));

        rocsparse_ut::device_vector<rocsparse_int> d_row_ptr(row_ptr);
        rocsparse_ut::device_vector<rocsparse_int> d_col_ind(col_ind);
        rocsparse_ut::device_vector<test_T>        d_val(val);
        rocsparse_ut::device_vector<test_T>        d_x(x);
        rocsparse_ut::device_vector<test_T>        d_y(y);

        rocsparse_handle    handle = nullptr;
        rocsparse_mat_descr descr  = nullptr;
        rocsparse_mat_info  info   = nullptr;
        ASSERT_EQ(rocsparse_create_handle(&handle), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_fill_mode(
                      descr, lower ? rocsparse_fill_mode_lower : rocsparse_fill_mode_upper),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_diag_type(descr, rocsparse_diag_type_non_unit),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

        size_t buffer_size = 0;
        ASSERT_EQ(rocsparse_sbsrsv_buffer_size(handle,
                                               rocsparse_direction_row,
                                               trans,
                                               kMb,
                                               nnzb,
                                               descr,
                                               d_val.ptr,
                                               d_row_ptr.ptr,
                                               d_col_ind.ptr,
                                               kBlockDim,
                                               info,
                                               &buffer_size),
                  rocsparse_status_success);

        rocsparse_ut::device_vector<char> d_buffer(std::vector<char>(buffer_size + 1, 0));
        ASSERT_NE(d_buffer.ptr, nullptr);

        ASSERT_EQ(rocsparse_sbsrsv_analysis(handle,
                                            rocsparse_direction_row,
                                            trans,
                                            kMb,
                                            nnzb,
                                            descr,
                                            d_val.ptr,
                                            d_row_ptr.ptr,
                                            d_col_ind.ptr,
                                            kBlockDim,
                                            info,
                                            rocsparse_analysis_policy_force,
                                            rocsparse_solve_policy_auto,
                                            d_buffer.ptr),
                  rocsparse_status_success);

        const test_T alpha = static_cast<test_T>(1);
        {
            std::unique_ptr<rocsparse_ut::ScopedMaxGridSizeX> clamp;
            if(max_grid_x > 0)
            {
                clamp = std::make_unique<rocsparse_ut::ScopedMaxGridSizeX>(handle, max_grid_x);
            }
            ASSERT_EQ(rocsparse_sbsrsv_solve(handle,
                                             rocsparse_direction_row,
                                             trans,
                                             kMb,
                                             nnzb,
                                             &alpha,
                                             descr,
                                             d_val.ptr,
                                             d_row_ptr.ptr,
                                             d_col_ind.ptr,
                                             kBlockDim,
                                             info,
                                             d_x.ptr,
                                             d_y.ptr,
                                             rocsparse_solve_policy_auto,
                                             d_buffer.ptr),
                      rocsparse_status_success);
        }

        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        y_out = rocsparse_ut::to_host(d_y);

        rocsparse_destroy_mat_info(info);
        rocsparse_destroy_mat_descr(descr);
        rocsparse_destroy_handle(handle);
    }
}

TEST(internal_level2_bsrsv_wide, general_i32_chunk)
{
    expect_wide_device<int32_t, int32_t>(true, 1);
    expect_wide_device<int32_t, int32_t>(false, 1);
}

TEST(internal_level2_bsrsv_wide, general_i32_full_grid)
{
    // Block 128 and 8 block rows: 2 virtual blocks on wave32, 4 on wave64.
    // A grid of 2 is exact on wave32 and a two-chunk walk on wave64.
    expect_wide_device<int32_t, int32_t>(true, 2);
}

TEST(internal_level2_bsrsv_wide, general_i64_i32_chunk)
{
    expect_wide_device<int64_t, int32_t>(true, 1);
    expect_wide_device<int64_t, int32_t>(false, 1);
}

TEST(internal_level2_bsrsv_wide, general_i64_chunk)
{
    expect_wide_device<int64_t, int64_t>(true, 1);
    expect_wide_device<int64_t, int64_t>(false, 1);
}

TEST(internal_level2_bsrsv_wide, zero_pivot_reports_block_row)
{
    // Block row 3 has a zero diagonal block. The reported pivot is the block
    // row plus the index base, so base one must report 4.
    expect_singular_pivot_device<int32_t, int32_t>(true, rocsparse_index_base_zero, 3, 1);
    expect_singular_pivot_device<int64_t, int32_t>(true, rocsparse_index_base_zero, 5, 1);
    expect_singular_pivot_device<int64_t, int64_t>(true, rocsparse_index_base_one, 3, 1);
    expect_singular_pivot_device<int64_t, int64_t>(true, rocsparse_index_base_one, 6, 2);
}

TEST(internal_level2_bsrsv_wide, needs_wide_predicate_boundaries)
{
    const int64_t i32max = std::numeric_limits<int32_t>::max();

    // Exact grid and small sizes stay narrow.
    EXPECT_FALSE(rocsparse::bsrsv_solve_needs_wide(8, 2, 4, 4, 4));
    // Grid one short of the virtual blocks is wide.
    EXPECT_TRUE(rocsparse::bsrsv_solve_needs_wide(8, 2, 4, 4, 3));
    // Empty problems never need the wide path.
    EXPECT_FALSE(rocsparse::bsrsv_solve_needs_wide(0, 2, 4, 0, 0));
    EXPECT_FALSE(rocsparse::bsrsv_solve_needs_wide(8, 2, 0, 4, 4));

    // mb at and past INT32_MAX.
    EXPECT_FALSE(rocsparse::bsrsv_solve_needs_wide(i32max, 1, 1, i32max, i32max));
    EXPECT_TRUE(rocsparse::bsrsv_solve_needs_wide(i32max + 1, 1, 1, i32max + 1, i32max + 1));

    // mb * block_dim at and past INT32_MAX (1048576 * 2047 fits, 1048576 * 2048 does not).
    EXPECT_FALSE(rocsparse::bsrsv_solve_needs_wide(1048576, 2047, 1, 1048576, 1048576));
    EXPECT_TRUE(rocsparse::bsrsv_solve_needs_wide(1048576, 2048, 1, 1048576, 1048576));

    // Diagonal offset block_dim * block_dim * j, j < nnzb. 40 * 40 * 1342177 fits in
    // INT32_MAX and one more block does not. This is the case the grid alone missed.
    EXPECT_FALSE(rocsparse::bsrsv_solve_needs_wide(10000, 40, 1342177, 10000, 10000));
    EXPECT_TRUE(rocsparse::bsrsv_solve_needs_wide(10000, 40, 1342178, 10000, 10000));

    // block_dim * block_dim alone: 46340^2 fits, 46341^2 does not.
    EXPECT_FALSE(rocsparse::bsrsv_solve_needs_wide(1, 46340, 1, 1, 1));
    EXPECT_TRUE(rocsparse::bsrsv_solve_needs_wide(1, 46341, 1, 1, 1));
}

TEST(internal_level2_bsrsv_wide, public_lower_narrow_and_clamped)
{
    for(int limit : {0, 1})
    {
        std::vector<test_T> y;
        ASSERT_NO_FATAL_FAILURE(solve_public(true, rocsparse_operation_none, limit, y));
        ASSERT_EQ(static_cast<int>(y.size()), kRows);
        for(int32_t row = 0; row < kRows; ++row)
        {
            EXPECT_EQ(y[static_cast<size_t>(row)], expected(true, row)) << "limit " << limit;
        }
    }
}

TEST(internal_level2_bsrsv_wide, public_upper_clamped)
{
    std::vector<test_T> y;
    ASSERT_NO_FATAL_FAILURE(solve_public(false, rocsparse_operation_none, 1, y));
    for(int32_t row = 0; row < kRows; ++row)
    {
        EXPECT_EQ(y[static_cast<size_t>(row)], expected(false, row)) << "row " << row;
    }
}

TEST(internal_level2_bsrsv_wide, public_lower_transpose_clamped)
{
    // A is lower, so A^T is the upper system.
    std::vector<test_T> y;
    ASSERT_NO_FATAL_FAILURE(solve_public(true, rocsparse_operation_transpose, 1, y));
    for(int32_t row = 0; row < kRows; ++row)
    {
        EXPECT_EQ(y[static_cast<size_t>(row)], expected(false, row)) << "row " << row;
    }
}

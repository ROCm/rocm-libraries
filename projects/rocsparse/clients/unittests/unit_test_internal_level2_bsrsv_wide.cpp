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
    void build_matrix(bool                lower,
                      std::vector<I>&     row_ptr,
                      std::vector<J>&     col_ind,
                      std::vector<test_T>& val,
                      std::vector<J>&     map)
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
                append_block(val, static_cast<test_T>(1));
                map[static_cast<size_t>(row)] = static_cast<J>(row);
            }
            else
            {
                col_ind.push_back(static_cast<J>(row));
                append_block(val, static_cast<test_T>(1));
                if(row + 1 < kMb)
                {
                    col_ind.push_back(static_cast<J>(row + 1));
                    append_block(val, static_cast<test_T>(-1));
                }
                map[static_cast<size_t>(row)] = static_cast<J>(kMb - 1 - row);
            }
            row_ptr.push_back(static_cast<I>(col_ind.size()));
        }
    }

    test_T expected(bool lower, int32_t row)
    {
        const int32_t block_row = row / kBlockDim;
        return lower ? static_cast<test_T>(block_row + 1)
                     : static_cast<test_T>(kMb - block_row);
    }

    template <uint32_t WFSIZE, typename I, typename J>
    void launch_wide(bool lower, uint32_t grid_x, std::vector<test_T>& y_out, J& pivot)
    {
        std::vector<I>      row_ptr;
        std::vector<J>      col_ind;
        std::vector<test_T> val;
        std::vector<J>      map;
        build_matrix(lower, row_ptr, col_ind, val, map);

        const std::vector<test_T> x(static_cast<size_t>(kRows), static_cast<test_T>(1));
        const std::vector<test_T> y(static_cast<size_t>(kRows), static_cast<test_T>(-7));

        rocsparse_ut::device_vector<I>      d_row_ptr(row_ptr);
        rocsparse_ut::device_vector<J>      d_col_ind(col_ind);
        rocsparse_ut::device_vector<test_T> d_val(val);
        rocsparse_ut::device_vector<J>      d_map(map);
        rocsparse_ut::device_vector<test_T> d_x(x);
        rocsparse_ut::device_vector<test_T> d_y(y);
        rocsparse_ut::device_vector<int>    d_done(std::vector<int>(static_cast<size_t>(kMb), 0));
        rocsparse_ut::device_vector<J>      d_pivot(
            std::vector<J>{std::numeric_limits<J>::max()});

        ASSERT_NE(d_row_ptr.ptr, nullptr);
        ASSERT_NE(d_col_ind.ptr, nullptr);
        ASSERT_NE(d_val.ptr, nullptr);
        ASSERT_NE(d_map.ptr, nullptr);
        ASSERT_NE(d_x.ptr, nullptr);
        ASSERT_NE(d_y.ptr, nullptr);
        ASSERT_NE(d_done.ptr, nullptr);
        ASSERT_NE(d_pivot.ptr, nullptr);

        const test_T alpha = static_cast<test_T>(1);
        const auto   alpha_union = rocsparse::const_host_device_scalar<test_T>(alpha);

        if(lower)
        {
            hipLaunchKernelGGL((rocsparse::bsrsv_lower_general_wide<kBlock, WFSIZE, false, test_T, I, J>),
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
                               rocsparse_index_base_zero,
                               rocsparse_diag_type_non_unit,
                               rocsparse_direction_row,
                               true);
        }
        else
        {
            hipLaunchKernelGGL((rocsparse::bsrsv_upper_general_wide<kBlock, WFSIZE, false, test_T, I, J>),
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
                               rocsparse_index_base_zero,
                               rocsparse_diag_type_non_unit,
                               rocsparse_direction_row,
                               true);
        }

        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        y_out = rocsparse_ut::to_host(d_y);
        pivot = rocsparse_ut::to_host(d_pivot)[0];
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

    void solve_public(bool lower, rocsparse_operation trans, int max_grid_x, std::vector<test_T>& y_out)
    {
        std::vector<rocsparse_int> row_ptr;
        std::vector<rocsparse_int> col_ind;
        std::vector<test_T>        val;
        std::vector<rocsparse_int> map;
        build_matrix(lower, row_ptr, col_ind, val, map);

        const rocsparse_int        nnzb = static_cast<rocsparse_int>(col_ind.size());
        const std::vector<test_T>  x(static_cast<size_t>(kRows), static_cast<test_T>(1));
        const std::vector<test_T>  y(static_cast<size_t>(kRows), static_cast<test_T>(-7));

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
    expect_wide<32, int32_t, int32_t>(true, 1);
    expect_wide<32, int32_t, int32_t>(false, 1);
}

TEST(internal_level2_bsrsv_wide, general_i32_full_grid)
{
    // wf 32, block 128: 4 warps per block, 8 block rows -> 2 virtual blocks.
    expect_wide<32, int32_t, int32_t>(true, 2);
}

TEST(internal_level2_bsrsv_wide, general_i64_i32_chunk)
{
    expect_wide<32, int64_t, int32_t>(true, 1);
    expect_wide<32, int64_t, int32_t>(false, 1);
}

TEST(internal_level2_bsrsv_wide, general_i64_chunk)
{
    // WFSIZE has to match the device wavefront. The diagonal update is
    // lockstep within one wavefront, same as the narrow kernel, so a
    // 64-lane launch is wrong on gfx1201 (wave32). Index type i64/i64 is
    // what this case covers; the host picks 64 only on wave64 devices.
    expect_wide<32, int64_t, int64_t>(true, 1);
    expect_wide<32, int64_t, int64_t>(false, 1);
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

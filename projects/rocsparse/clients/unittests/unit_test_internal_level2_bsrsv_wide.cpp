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

#include "rocsparse_bsrsv.hpp"
#include "rocsparse_mat_info.hpp"

#include "rocsparse.h"

#include <algorithm>
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
                     int32_t              singular_row = -1,
                     bool                 shared       = false)
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

        // BSRDIM 8 is the specialization the host uses for block_dim <= 8, and
        // it divides both wavefront sizes.
        if(shared && lower)
        {
            hipLaunchKernelGGL(
                (rocsparse::bsrsv_lower_shared_wide<kBlock, WFSIZE, 8, false, test_T, I, J>),
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
        else if(shared)
        {
            hipLaunchKernelGGL(
                (rocsparse::bsrsv_upper_shared_wide<kBlock, WFSIZE, 8, false, test_T, I, J>),
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
        else if(lower)
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
    // A failed query returns 0, which is not a launchable size.
    int device_wavefront()
    {
        const int ws = rocsparse_ut::device_warp_size();
        EXPECT_TRUE(ws == 32 || ws == 64) << "unsupported device wavefront size: " << ws;
        return ws;
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
                               uint32_t             grid_x,
                               bool                 shared = false)
    {
        std::vector<test_T> y;
        J                   pivot{};
        ASSERT_NO_FATAL_FAILURE(
            (launch_wide<WFSIZE, I, J>(lower, grid_x, y, pivot, base, singular_row, shared)));
        // The kernel reports the block row in the caller's index base.
        EXPECT_EQ(pivot, static_cast<J>(singular_row + static_cast<int32_t>(base)));
    }

    template <typename I, typename J>
    void expect_singular_pivot_device(bool                 lower,
                                      rocsparse_index_base base,
                                      int32_t              singular_row,
                                      uint32_t             grid_x,
                                      bool                 shared = false)
    {
        const int ws = device_wavefront();
        ASSERT_TRUE(ws == 32 || ws == 64);
        if(ws == 64)
        {
            expect_singular_pivot<64, I, J>(lower, base, singular_row, grid_x, shared);
        }
        else
        {
            expect_singular_pivot<32, I, J>(lower, base, singular_row, grid_x, shared);
        }
    }

    template <typename I, typename J>
    void expect_wide_device(bool lower, uint32_t grid_x)
    {
        const int ws = device_wavefront();
        ASSERT_TRUE(ws == 32 || ws == 64);
        if(ws == 64)
        {
            expect_wide<64, I, J>(lower, grid_x);
        }
        else
        {
            expect_wide<32, I, J>(lower, grid_x);
        }
    }

    // Releases the rocSPARSE objects when an assertion returns early.
    struct scoped_bsrsv_objects
    {
        rocsparse_handle    handle = nullptr;
        rocsparse_mat_descr descr  = nullptr;
        rocsparse_mat_info  info   = nullptr;

        scoped_bsrsv_objects() = default;

        ~scoped_bsrsv_objects()
        {
            if(info != nullptr)
            {
                rocsparse_destroy_mat_info(info);
            }
            if(descr != nullptr)
            {
                rocsparse_destroy_mat_descr(descr);
            }
            if(handle != nullptr)
            {
                rocsparse_destroy_handle(handle);
            }
        }

        scoped_bsrsv_objects(const scoped_bsrsv_objects&)            = delete;
        scoped_bsrsv_objects& operator=(const scoped_bsrsv_objects&) = delete;
    };

    void solve_public(bool                 lower,
                      rocsparse_operation  trans,
                      int                  max_grid_x,
                      std::vector<test_T>& y_out,
                      int32_t              singular_row = -1,
                      rocsparse_int*       pivot        = nullptr)
    {
        std::vector<rocsparse_int> row_ptr;
        std::vector<rocsparse_int> col_ind;
        std::vector<test_T>        val;
        std::vector<rocsparse_int> map;
        build_matrix(lower, row_ptr, col_ind, val, map, rocsparse_index_base_zero, singular_row);

        const rocsparse_int       nnzb = static_cast<rocsparse_int>(col_ind.size());
        const std::vector<test_T> x(static_cast<size_t>(kRows), static_cast<test_T>(1));
        const std::vector<test_T> y(static_cast<size_t>(kRows), static_cast<test_T>(-7));

        rocsparse_ut::device_vector<rocsparse_int> d_row_ptr(row_ptr);
        rocsparse_ut::device_vector<rocsparse_int> d_col_ind(col_ind);
        rocsparse_ut::device_vector<test_T>        d_val(val);
        rocsparse_ut::device_vector<test_T>        d_x(x);
        rocsparse_ut::device_vector<test_T>        d_y(y);

        scoped_bsrsv_objects objects;
        ASSERT_EQ(rocsparse_create_handle(&objects.handle), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_mat_descr(&objects.descr), rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_fill_mode(
                      objects.descr, lower ? rocsparse_fill_mode_lower : rocsparse_fill_mode_upper),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_diag_type(objects.descr, rocsparse_diag_type_non_unit),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_mat_info(&objects.info), rocsparse_status_success);

        size_t buffer_size = 0;
        ASSERT_EQ(rocsparse_sbsrsv_buffer_size(objects.handle,
                                               rocsparse_direction_row,
                                               trans,
                                               kMb,
                                               nnzb,
                                               objects.descr,
                                               d_val.ptr,
                                               d_row_ptr.ptr,
                                               d_col_ind.ptr,
                                               kBlockDim,
                                               objects.info,
                                               &buffer_size),
                  rocsparse_status_success);

        rocsparse_ut::device_vector<char> d_buffer(std::vector<char>(buffer_size + 1, 0));
        ASSERT_NE(d_buffer.ptr, nullptr);

        ASSERT_EQ(rocsparse_sbsrsv_analysis(objects.handle,
                                            rocsparse_direction_row,
                                            trans,
                                            kMb,
                                            nnzb,
                                            objects.descr,
                                            d_val.ptr,
                                            d_row_ptr.ptr,
                                            d_col_ind.ptr,
                                            kBlockDim,
                                            objects.info,
                                            rocsparse_analysis_policy_force,
                                            rocsparse_solve_policy_auto,
                                            d_buffer.ptr),
                  rocsparse_status_success);

        const test_T alpha = static_cast<test_T>(1);
        {
            std::unique_ptr<rocsparse_ut::ScopedMaxGridSizeX> clamp;
            if(max_grid_x > 0)
            {
                clamp = std::make_unique<rocsparse_ut::ScopedMaxGridSizeX>(objects.handle,
                                                                           max_grid_x);
            }
            ASSERT_EQ(rocsparse_sbsrsv_solve(objects.handle,
                                             rocsparse_direction_row,
                                             trans,
                                             kMb,
                                             nnzb,
                                             &alpha,
                                             objects.descr,
                                             d_val.ptr,
                                             d_row_ptr.ptr,
                                             d_col_ind.ptr,
                                             kBlockDim,
                                             objects.info,
                                             d_x.ptr,
                                             d_y.ptr,
                                             rocsparse_solve_policy_auto,
                                             d_buffer.ptr),
                      rocsparse_status_success);
        }

        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        y_out = rocsparse_ut::to_host(d_y);

        if(pivot != nullptr)
        {
            // No pivot stays at the analysis sentinel, which the query reports as -1.
            rocsparse_int          position = -2;
            const rocsparse_status status
                = rocsparse_bsrsv_zero_pivot(objects.handle, objects.info, &position);
            EXPECT_EQ(status,
                      (singular_row >= 0) ? rocsparse_status_zero_pivot : rocsparse_status_success);
            *pivot = position;
        }
    }

    // bsric0 builds the analysis, bsrsv reuse borrows it, and a later bsrsv
    // force must not delete it. Clearing bsric0 afterwards is the check.
    void expect_borrowed_analysis_survives_force()
    {
        std::vector<rocsparse_int> row_ptr;
        std::vector<rocsparse_int> col_ind;
        std::vector<test_T>        val;
        std::vector<rocsparse_int> map;
        build_matrix(true, row_ptr, col_ind, val, map);

        const rocsparse_int nnzb = static_cast<rocsparse_int>(col_ind.size());
        rocsparse_ut::device_vector<rocsparse_int> d_row_ptr(row_ptr);
        rocsparse_ut::device_vector<rocsparse_int> d_col_ind(col_ind);
        rocsparse_ut::device_vector<test_T>        d_val(val);

        scoped_bsrsv_objects objects;
        ASSERT_EQ(rocsparse_create_handle(&objects.handle), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_mat_descr(&objects.descr), rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_fill_mode(objects.descr, rocsparse_fill_mode_lower),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_diag_type(objects.descr, rocsparse_diag_type_non_unit),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_mat_info(&objects.info), rocsparse_status_success);

        size_t ic0_bytes = 0;
        size_t sv_bytes  = 0;
        ASSERT_EQ(rocsparse_sbsric0_buffer_size(objects.handle,
                                                rocsparse_direction_row,
                                                kMb,
                                                nnzb,
                                                objects.descr,
                                                d_val.ptr,
                                                d_row_ptr.ptr,
                                                d_col_ind.ptr,
                                                kBlockDim,
                                                objects.info,
                                                &ic0_bytes),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_sbsrsv_buffer_size(objects.handle,
                                               rocsparse_direction_row,
                                               rocsparse_operation_none,
                                               kMb,
                                               nnzb,
                                               objects.descr,
                                               d_val.ptr,
                                               d_row_ptr.ptr,
                                               d_col_ind.ptr,
                                               kBlockDim,
                                               objects.info,
                                               &sv_bytes),
                  rocsparse_status_success);

        rocsparse_ut::device_vector<char> d_buffer(
            std::vector<char>(std::max(ic0_bytes, sv_bytes) + 1, 0));
        ASSERT_NE(d_buffer.ptr, nullptr);

        ASSERT_EQ(rocsparse_sbsric0_analysis(objects.handle,
                                             rocsparse_direction_row,
                                             kMb,
                                             nnzb,
                                             objects.descr,
                                             d_val.ptr,
                                             d_row_ptr.ptr,
                                             d_col_ind.ptr,
                                             kBlockDim,
                                             objects.info,
                                             rocsparse_analysis_policy_force,
                                             rocsparse_solve_policy_auto,
                                             d_buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_sbsrsv_analysis(objects.handle,
                                            rocsparse_direction_row,
                                            rocsparse_operation_none,
                                            kMb,
                                            nnzb,
                                            objects.descr,
                                            d_val.ptr,
                                            d_row_ptr.ptr,
                                            d_col_ind.ptr,
                                            kBlockDim,
                                            objects.info,
                                            rocsparse_analysis_policy_reuse,
                                            rocsparse_solve_policy_auto,
                                            d_buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_sbsrsv_analysis(objects.handle,
                                            rocsparse_direction_row,
                                            rocsparse_operation_none,
                                            kMb,
                                            nnzb,
                                            objects.descr,
                                            d_val.ptr,
                                            d_row_ptr.ptr,
                                            d_col_ind.ptr,
                                            kBlockDim,
                                            objects.info,
                                            rocsparse_analysis_policy_force,
                                            rocsparse_solve_policy_auto,
                                            d_buffer.ptr),
                  rocsparse_status_success);

        ASSERT_EQ(rocsparse_bsric0_clear(objects.handle, objects.info), rocsparse_status_success);
    }

    // Drops the diagonal block of block row missing_row from a block-lower matrix.
    // The analysis sees a structural zero there, which the solve does not.
    void remove_diagonal_block(std::vector<rocsparse_int>& row_ptr,
                               std::vector<rocsparse_int>& col_ind,
                               std::vector<test_T>&        val,
                               int32_t                     missing_row)
    {
        std::vector<rocsparse_int> new_row_ptr{0};
        std::vector<rocsparse_int> new_col_ind;
        std::vector<test_T>        new_val;
        for(int32_t row = 0; row < kMb; ++row)
        {
            for(rocsparse_int j = row_ptr[row]; j < row_ptr[row + 1]; ++j)
            {
                if(row == missing_row && col_ind[j] == row)
                {
                    continue;
                }
                new_col_ind.push_back(col_ind[j]);
                new_val.insert(new_val.end(),
                               val.begin() + 4 * static_cast<size_t>(j),
                               val.begin() + 4 * static_cast<size_t>(j) + 4);
            }
            new_row_ptr.push_back(static_cast<rocsparse_int>(new_col_ind.size()));
        }
        row_ptr = new_row_ptr;
        col_ind = new_col_ind;
        val     = new_val;
    }

    // bsric0 builds the analysis, bsrsv reuse borrows it, and a solve follows. With
    // missing_row < 0 the matrix is non-singular and the solve must give the expected
    // result. Otherwise that block row has a structural zero, and the solve must report
    // it through rocsparse_bsrsv_zero_pivot, which needs the donor's pivot.
    void expect_borrowed_reuse_solves(int32_t missing_row = -1)
    {
        std::vector<rocsparse_int> row_ptr;
        std::vector<rocsparse_int> col_ind;
        std::vector<test_T>        val;
        std::vector<rocsparse_int> map;
        build_matrix(true, row_ptr, col_ind, val, map);
        if(missing_row >= 0)
        {
            remove_diagonal_block(row_ptr, col_ind, val, missing_row);
        }

        const rocsparse_int nnzb = static_cast<rocsparse_int>(col_ind.size());
        rocsparse_ut::device_vector<rocsparse_int> d_row_ptr(row_ptr);
        rocsparse_ut::device_vector<rocsparse_int> d_col_ind(col_ind);
        rocsparse_ut::device_vector<test_T>        d_val(val);
        const std::vector<test_T>           x(static_cast<size_t>(kRows), static_cast<test_T>(1));
        const std::vector<test_T>           y0(static_cast<size_t>(kRows), static_cast<test_T>(-7));
        rocsparse_ut::device_vector<test_T> d_x(x);
        rocsparse_ut::device_vector<test_T> d_y(y0);

        scoped_bsrsv_objects objects;
        ASSERT_EQ(rocsparse_create_handle(&objects.handle), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_mat_descr(&objects.descr), rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_fill_mode(objects.descr, rocsparse_fill_mode_lower),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_diag_type(objects.descr, rocsparse_diag_type_non_unit),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_mat_info(&objects.info), rocsparse_status_success);

        size_t ic0_bytes = 0;
        size_t sv_bytes  = 0;
        ASSERT_EQ(rocsparse_sbsric0_buffer_size(objects.handle,
                                                rocsparse_direction_row,
                                                kMb,
                                                nnzb,
                                                objects.descr,
                                                d_val.ptr,
                                                d_row_ptr.ptr,
                                                d_col_ind.ptr,
                                                kBlockDim,
                                                objects.info,
                                                &ic0_bytes),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_sbsrsv_buffer_size(objects.handle,
                                               rocsparse_direction_row,
                                               rocsparse_operation_none,
                                               kMb,
                                               nnzb,
                                               objects.descr,
                                               d_val.ptr,
                                               d_row_ptr.ptr,
                                               d_col_ind.ptr,
                                               kBlockDim,
                                               objects.info,
                                               &sv_bytes),
                  rocsparse_status_success);

        rocsparse_ut::device_vector<char> d_buffer(
            std::vector<char>(std::max(ic0_bytes, sv_bytes) + 1, 0));
        ASSERT_NE(d_buffer.ptr, nullptr);

        ASSERT_EQ(rocsparse_sbsric0_analysis(objects.handle,
                                             rocsparse_direction_row,
                                             kMb,
                                             nnzb,
                                             objects.descr,
                                             d_val.ptr,
                                             d_row_ptr.ptr,
                                             d_col_ind.ptr,
                                             kBlockDim,
                                             objects.info,
                                             rocsparse_analysis_policy_force,
                                             rocsparse_solve_policy_auto,
                                             d_buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_sbsrsv_analysis(objects.handle,
                                            rocsparse_direction_row,
                                            rocsparse_operation_none,
                                            kMb,
                                            nnzb,
                                            objects.descr,
                                            d_val.ptr,
                                            d_row_ptr.ptr,
                                            d_col_ind.ptr,
                                            kBlockDim,
                                            objects.info,
                                            rocsparse_analysis_policy_reuse,
                                            rocsparse_solve_policy_auto,
                                            d_buffer.ptr),
                  rocsparse_status_success);

        const test_T alpha = static_cast<test_T>(1);
        ASSERT_EQ(rocsparse_sbsrsv_solve(objects.handle,
                                         rocsparse_direction_row,
                                         rocsparse_operation_none,
                                         kMb,
                                         nnzb,
                                         &alpha,
                                         objects.descr,
                                         d_val.ptr,
                                         d_row_ptr.ptr,
                                         d_col_ind.ptr,
                                         kBlockDim,
                                         objects.info,
                                         d_x.ptr,
                                         d_y.ptr,
                                         rocsparse_solve_policy_auto,
                                         d_buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        if(missing_row < 0)
        {
            const std::vector<test_T> y = rocsparse_ut::to_host(d_y);
            ASSERT_EQ(static_cast<int>(y.size()), kRows);
            for(int32_t row = 0; row < kRows; ++row)
            {
                EXPECT_EQ(y[static_cast<size_t>(row)], expected(true, row)) << "row " << row;
            }
        }

        rocsparse_int          position = -2;
        const rocsparse_status status
            = rocsparse_bsrsv_zero_pivot(objects.handle, objects.info, &position);
        if(missing_row < 0)
        {
            EXPECT_EQ(status, rocsparse_status_success);
        }
        else
        {
            EXPECT_EQ(status, rocsparse_status_zero_pivot);
            EXPECT_EQ(position, static_cast<rocsparse_int>(missing_row));
        }
        ASSERT_EQ(rocsparse_bsric0_clear(objects.handle, objects.info), rocsparse_status_success);
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

    // Upper general has its own pivot flag. Base one checks the offset there too.
    expect_singular_pivot_device<int32_t, int32_t>(false, rocsparse_index_base_one, 3, 1);
    expect_singular_pivot_device<int64_t, int64_t>(false, rocsparse_index_base_one, 5, 1);
}

TEST(internal_level2_bsrsv_wide, zero_pivot_shared_reports_block_row)
{
    // The host launches the shared kernels only on wave64. These direct launches
    // use the device wavefront, so the shared pivot path runs on wave32 as well.
    expect_singular_pivot_device<int32_t, int32_t>(true, rocsparse_index_base_one, 3, 1, true);
    expect_singular_pivot_device<int64_t, int32_t>(false, rocsparse_index_base_zero, 4, 1, true);
    expect_singular_pivot_device<int64_t, int64_t>(true, rocsparse_index_base_one, 6, 2, true);
    expect_singular_pivot_device<int64_t, int64_t>(false, rocsparse_index_base_one, 2, 1, true);
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

TEST(internal_level2_bsrsv_wide, public_singular_pivot_clamped)
{
    // block_dim is 2, so a clamped wave64 launch takes the shared kernel and a
    // wave32 launch takes the general kernel. Both report the block row.
    std::vector<test_T> y;
    rocsparse_int       pivot = -2;
    ASSERT_NO_FATAL_FAILURE(solve_public(true, rocsparse_operation_none, 1, y, 3, &pivot));
    EXPECT_EQ(pivot, static_cast<rocsparse_int>(3));

    pivot = -2;
    ASSERT_NO_FATAL_FAILURE(solve_public(false, rocsparse_operation_none, 1, y, 4, &pivot));
    EXPECT_EQ(pivot, static_cast<rocsparse_int>(4));
}

void expect_success(rocsparse_status status)
{
    ASSERT_EQ(status, rocsparse_status_success);
}

// Analysis at one index width, then another, on the same mat_info. The pivot
// type has to follow the latest analysis, and a solve of that width has to
// succeed. The reuse case below starts from an i64 pivot left by a different
// fill mode and borrows an i32 bsric0 analysis: reuse must drop the i64 pivot
// and copy the donor's. Removing either drop call fails one of these checks.
void expect_mixed_width_retypes_and_solves()
{
    std::vector<int32_t> row32;
    std::vector<int32_t> col32;
    std::vector<int32_t> map32;
    std::vector<int64_t> row64;
    std::vector<int64_t> col64;
    std::vector<int64_t> map64;
    std::vector<test_T>  val;
    build_matrix<int32_t, int32_t>(true, row32, col32, val, map32);
    build_matrix<int64_t, int64_t>(true, row64, col64, val, map64);
    (void)map32;
    (void)map64;

    const int32_t                        nnzb32 = static_cast<int32_t>(col32.size());
    const int64_t                        nnzb64 = static_cast<int64_t>(col64.size());
    rocsparse_ut::device_vector<int32_t> d_row32(row32);
    rocsparse_ut::device_vector<int32_t> d_col32(col32);
    rocsparse_ut::device_vector<int64_t> d_row64(row64);
    rocsparse_ut::device_vector<int64_t> d_col64(col64);
    rocsparse_ut::device_vector<test_T>  d_val(val);
    const std::vector<test_T>            x(static_cast<size_t>(kRows), static_cast<test_T>(1));
    const std::vector<test_T>            y0(static_cast<size_t>(kRows), static_cast<test_T>(-7));
    rocsparse_ut::device_vector<test_T>  d_x(x);
    rocsparse_ut::device_vector<test_T>  d_y(y0);

    scoped_bsrsv_objects objects;
    ASSERT_EQ(rocsparse_create_handle(&objects.handle), rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_mat_descr(&objects.descr), rocsparse_status_success);
    ASSERT_EQ(rocsparse_set_mat_fill_mode(objects.descr, rocsparse_fill_mode_lower),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_set_mat_diag_type(objects.descr, rocsparse_diag_type_non_unit),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_mat_info(&objects.info), rocsparse_status_success);

    size_t bytes32 = 0;
    size_t bytes64 = 0;
    ASSERT_NO_FATAL_FAILURE(expect_success(
        rocsparse::bsrsv_buffer_size_template<test_T, int32_t, int32_t>(objects.handle,
                                                                        rocsparse_direction_row,
                                                                        rocsparse_operation_none,
                                                                        kMb,
                                                                        nnzb32,
                                                                        objects.descr,
                                                                        d_val.ptr,
                                                                        d_row32.ptr,
                                                                        d_col32.ptr,
                                                                        kBlockDim,
                                                                        objects.info,
                                                                        &bytes32)));
    ASSERT_NO_FATAL_FAILURE(expect_success(
        rocsparse::bsrsv_buffer_size_template<test_T, int64_t, int64_t>(objects.handle,
                                                                        rocsparse_direction_row,
                                                                        rocsparse_operation_none,
                                                                        static_cast<int64_t>(kMb),
                                                                        nnzb64,
                                                                        objects.descr,
                                                                        d_val.ptr,
                                                                        d_row64.ptr,
                                                                        d_col64.ptr,
                                                                        kBlockDim,
                                                                        objects.info,
                                                                        &bytes64)));
    rocsparse_ut::device_vector<char> d_buffer(
        std::vector<char>(std::max(bytes32, bytes64) + 1, 0));
    ASSERT_NE(d_buffer.ptr, nullptr);

    ASSERT_NO_FATAL_FAILURE(
        expect_success(rocsparse::bsrsv_analysis_template<test_T, int32_t, int32_t>(
            objects.handle,
            rocsparse_direction_row,
            rocsparse_operation_none,
            kMb,
            nnzb32,
            objects.descr,
            d_val.ptr,
            d_row32.ptr,
            d_col32.ptr,
            kBlockDim,
            objects.info,
            rocsparse_analysis_policy_force,
            rocsparse_solve_policy_auto,
            d_buffer.ptr)));
    auto* container = objects.info->get_bsrsv_info();
    ASSERT_NE(container, nullptr);
    EXPECT_EQ(container->get_indextype(), rocsparse_indextype_i32);

    ASSERT_NO_FATAL_FAILURE(
        expect_success(rocsparse::bsrsv_analysis_template<test_T, int64_t, int64_t>(
            objects.handle,
            rocsparse_direction_row,
            rocsparse_operation_none,
            static_cast<int64_t>(kMb),
            nnzb64,
            objects.descr,
            d_val.ptr,
            d_row64.ptr,
            d_col64.ptr,
            kBlockDim,
            objects.info,
            rocsparse_analysis_policy_force,
            rocsparse_solve_policy_auto,
            d_buffer.ptr)));
    EXPECT_EQ(container->get_indextype(), rocsparse_indextype_i64);

    const test_T alpha = static_cast<test_T>(1);
    // The public entry reads the pivot as rocsparse_int, so an i64 pivot is rejected.
    EXPECT_EQ(rocsparse_sbsrsv_solve(objects.handle,
                                     rocsparse_direction_row,
                                     rocsparse_operation_none,
                                     kMb,
                                     nnzb32,
                                     &alpha,
                                     objects.descr,
                                     d_val.ptr,
                                     d_row32.ptr,
                                     d_col32.ptr,
                                     kBlockDim,
                                     objects.info,
                                     d_x.ptr,
                                     d_y.ptr,
                                     rocsparse_solve_policy_auto,
                                     d_buffer.ptr),
              rocsparse_status_invalid_value);

    auto* trm = objects.info->get_bsrsv_info(rocsparse_operation_none, rocsparse_fill_mode_lower);
    ASSERT_NE(trm, nullptr);
    ASSERT_NO_FATAL_FAILURE(expect_success(rocsparse::bsrsv_solve_core<test_T, int64_t, int64_t>(
        objects.handle,
        rocsparse_direction_row,
        rocsparse_operation_none,
        static_cast<int64_t>(kMb),
        nnzb64,
        &alpha,
        objects.descr,
        d_val.ptr,
        d_row64.ptr,
        d_col64.ptr,
        kBlockDim,
        trm,
        container->get_indextype(),
        static_cast<int64_t*>(container->get_position()),
        d_x.ptr,
        d_y.ptr,
        rocsparse_solve_policy_auto,
        d_buffer.ptr)));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    const std::vector<test_T> y64 = rocsparse_ut::to_host(d_y);
    ASSERT_EQ(static_cast<int>(y64.size()), kRows);
    for(int32_t row = 0; row < kRows; ++row)
    {
        EXPECT_EQ(y64[static_cast<size_t>(row)], expected(true, row)) << "row " << row;
    }
    int64_t pivot64 = 0;
    ASSERT_EQ(
        hipMemcpy(&pivot64, container->get_position(), sizeof(pivot64), hipMemcpyDeviceToHost),
        hipSuccess);
    EXPECT_EQ(pivot64, std::numeric_limits<int64_t>::max());

    // Upper i32 analysis retypes the shared pivot. Reuse of the lower i64
    // analysis must drop that pivot and create an i64 one. Until it does,
    // the i64 solve rejects the i32 pivot.
    ASSERT_EQ(rocsparse_set_mat_fill_mode(objects.descr, rocsparse_fill_mode_upper),
              rocsparse_status_success);
    ASSERT_NO_FATAL_FAILURE(
        expect_success(rocsparse::bsrsv_analysis_template<test_T, int32_t, int32_t>(
            objects.handle,
            rocsparse_direction_row,
            rocsparse_operation_none,
            kMb,
            nnzb32,
            objects.descr,
            d_val.ptr,
            d_row32.ptr,
            d_col32.ptr,
            kBlockDim,
            objects.info,
            rocsparse_analysis_policy_force,
            rocsparse_solve_policy_auto,
            d_buffer.ptr)));
    EXPECT_EQ(container->get_indextype(), rocsparse_indextype_i32);
    const rocsparse_status mismatched = rocsparse::bsrsv_solve_core<test_T, int64_t, int64_t>(
        objects.handle,
        rocsparse_direction_row,
        rocsparse_operation_none,
        static_cast<int64_t>(kMb),
        nnzb64,
        &alpha,
        objects.descr,
        d_val.ptr,
        d_row64.ptr,
        d_col64.ptr,
        kBlockDim,
        trm,
        container->get_indextype(),
        static_cast<int64_t*>(container->get_position()),
        d_x.ptr,
        d_y.ptr,
        rocsparse_solve_policy_auto,
        d_buffer.ptr);
    EXPECT_EQ(mismatched, rocsparse_status_invalid_value);

    ASSERT_EQ(rocsparse_set_mat_fill_mode(objects.descr, rocsparse_fill_mode_lower),
              rocsparse_status_success);
    ASSERT_NO_FATAL_FAILURE(
        expect_success(rocsparse::bsrsv_analysis_template<test_T, int64_t, int64_t>(
            objects.handle,
            rocsparse_direction_row,
            rocsparse_operation_none,
            static_cast<int64_t>(kMb),
            nnzb64,
            objects.descr,
            d_val.ptr,
            d_row64.ptr,
            d_col64.ptr,
            kBlockDim,
            objects.info,
            rocsparse_analysis_policy_reuse,
            rocsparse_solve_policy_auto,
            d_buffer.ptr)));
    EXPECT_EQ(container->get_indextype(), rocsparse_indextype_i64);
    ASSERT_EQ(hipMemcpy(d_y.ptr, y0.data(), y0.size() * sizeof(test_T), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_NO_FATAL_FAILURE(expect_success(rocsparse::bsrsv_solve_core<test_T, int64_t, int64_t>(
        objects.handle,
        rocsparse_direction_row,
        rocsparse_operation_none,
        static_cast<int64_t>(kMb),
        nnzb64,
        &alpha,
        objects.descr,
        d_val.ptr,
        d_row64.ptr,
        d_col64.ptr,
        kBlockDim,
        trm,
        container->get_indextype(),
        static_cast<int64_t*>(container->get_position()),
        d_x.ptr,
        d_y.ptr,
        rocsparse_solve_policy_auto,
        d_buffer.ptr)));
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    const std::vector<test_T> y64_reuse = rocsparse_ut::to_host(d_y);
    for(int32_t row = 0; row < kRows; ++row)
    {
        EXPECT_EQ(y64_reuse[static_cast<size_t>(row)], expected(true, row)) << "row " << row;
    }

    ASSERT_EQ(hipMemcpy(d_y.ptr, y0.data(), y0.size() * sizeof(test_T), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_NO_FATAL_FAILURE(
        expect_success(rocsparse::bsrsv_analysis_template<test_T, int32_t, int32_t>(
            objects.handle,
            rocsparse_direction_row,
            rocsparse_operation_none,
            kMb,
            nnzb32,
            objects.descr,
            d_val.ptr,
            d_row32.ptr,
            d_col32.ptr,
            kBlockDim,
            objects.info,
            rocsparse_analysis_policy_force,
            rocsparse_solve_policy_auto,
            d_buffer.ptr)));
    EXPECT_EQ(container->get_indextype(), rocsparse_indextype_i32);
    ASSERT_EQ(rocsparse_sbsrsv_solve(objects.handle,
                                     rocsparse_direction_row,
                                     rocsparse_operation_none,
                                     kMb,
                                     nnzb32,
                                     &alpha,
                                     objects.descr,
                                     d_val.ptr,
                                     d_row32.ptr,
                                     d_col32.ptr,
                                     kBlockDim,
                                     objects.info,
                                     d_x.ptr,
                                     d_y.ptr,
                                     rocsparse_solve_policy_auto,
                                     d_buffer.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    const std::vector<test_T> y32 = rocsparse_ut::to_host(d_y);
    for(int32_t row = 0; row < kRows; ++row)
    {
        EXPECT_EQ(y32[static_cast<size_t>(row)], expected(true, row)) << "row " << row;
    }
}

// An i64 analysis of the other fill mode leaves an i64 pivot in the container.
// Reuse of the lower i32 bsric0 analysis must drop it and copy the donor pivot,
// which holds the structural zero at block row 3.
void expect_reuse_drops_mismatched_pivot()
{
    std::vector<int32_t> row32;
    std::vector<int32_t> col32;
    std::vector<int32_t> map32;
    std::vector<test_T>  val;
    build_matrix<int32_t, int32_t>(true, row32, col32, val, map32);
    (void)map32;
    remove_diagonal_block(row32, col32, val, 3);
    std::vector<int64_t> row64(row32.size());
    std::vector<int64_t> col64(col32.size());
    for(size_t i = 0; i < row32.size(); ++i)
    {
        row64[i] = row32[i];
    }
    for(size_t i = 0; i < col32.size(); ++i)
    {
        col64[i] = col32[i];
    }

    const int32_t                        nnzb32 = static_cast<int32_t>(col32.size());
    const int64_t                        nnzb64 = static_cast<int64_t>(col64.size());
    rocsparse_ut::device_vector<int32_t> d_row32(row32);
    rocsparse_ut::device_vector<int32_t> d_col32(col32);
    rocsparse_ut::device_vector<int64_t> d_row64(row64);
    rocsparse_ut::device_vector<int64_t> d_col64(col64);
    rocsparse_ut::device_vector<test_T>  d_val(val);
    const std::vector<test_T>            x(static_cast<size_t>(kRows), static_cast<test_T>(1));
    const std::vector<test_T>            y0(static_cast<size_t>(kRows), static_cast<test_T>(-7));
    rocsparse_ut::device_vector<test_T>  d_x(x);
    rocsparse_ut::device_vector<test_T>  d_y(y0);

    scoped_bsrsv_objects objects;
    ASSERT_EQ(rocsparse_create_handle(&objects.handle), rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_mat_descr(&objects.descr), rocsparse_status_success);
    ASSERT_EQ(rocsparse_set_mat_fill_mode(objects.descr, rocsparse_fill_mode_upper),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_set_mat_diag_type(objects.descr, rocsparse_diag_type_non_unit),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_mat_info(&objects.info), rocsparse_status_success);

    size_t bytes64   = 0;
    size_t ic0_bytes = 0;
    size_t sv_bytes  = 0;
    ASSERT_NO_FATAL_FAILURE(expect_success(
        rocsparse::bsrsv_buffer_size_template<test_T, int64_t, int64_t>(objects.handle,
                                                                        rocsparse_direction_row,
                                                                        rocsparse_operation_none,
                                                                        static_cast<int64_t>(kMb),
                                                                        nnzb64,
                                                                        objects.descr,
                                                                        d_val.ptr,
                                                                        d_row64.ptr,
                                                                        d_col64.ptr,
                                                                        kBlockDim,
                                                                        objects.info,
                                                                        &bytes64)));
    ASSERT_EQ(rocsparse_set_mat_fill_mode(objects.descr, rocsparse_fill_mode_lower),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_sbsric0_buffer_size(objects.handle,
                                            rocsparse_direction_row,
                                            kMb,
                                            nnzb32,
                                            objects.descr,
                                            d_val.ptr,
                                            d_row32.ptr,
                                            d_col32.ptr,
                                            kBlockDim,
                                            objects.info,
                                            &ic0_bytes),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_sbsrsv_buffer_size(objects.handle,
                                           rocsparse_direction_row,
                                           rocsparse_operation_none,
                                           kMb,
                                           nnzb32,
                                           objects.descr,
                                           d_val.ptr,
                                           d_row32.ptr,
                                           d_col32.ptr,
                                           kBlockDim,
                                           objects.info,
                                           &sv_bytes),
              rocsparse_status_success);
    const size_t                      bytes = std::max(bytes64, std::max(ic0_bytes, sv_bytes));
    rocsparse_ut::device_vector<char> d_buffer(std::vector<char>(bytes + 1, 0));
    ASSERT_NE(d_buffer.ptr, nullptr);

    ASSERT_EQ(rocsparse_set_mat_fill_mode(objects.descr, rocsparse_fill_mode_upper),
              rocsparse_status_success);
    ASSERT_NO_FATAL_FAILURE(
        expect_success(rocsparse::bsrsv_analysis_template<test_T, int64_t, int64_t>(
            objects.handle,
            rocsparse_direction_row,
            rocsparse_operation_none,
            static_cast<int64_t>(kMb),
            nnzb64,
            objects.descr,
            d_val.ptr,
            d_row64.ptr,
            d_col64.ptr,
            kBlockDim,
            objects.info,
            rocsparse_analysis_policy_force,
            rocsparse_solve_policy_auto,
            d_buffer.ptr)));
    auto* container = objects.info->get_bsrsv_info();
    ASSERT_NE(container, nullptr);
    EXPECT_EQ(container->get_indextype(), rocsparse_indextype_i64);

    ASSERT_EQ(rocsparse_set_mat_fill_mode(objects.descr, rocsparse_fill_mode_lower),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_sbsric0_analysis(objects.handle,
                                         rocsparse_direction_row,
                                         kMb,
                                         nnzb32,
                                         objects.descr,
                                         d_val.ptr,
                                         d_row32.ptr,
                                         d_col32.ptr,
                                         kBlockDim,
                                         objects.info,
                                         rocsparse_analysis_policy_force,
                                         rocsparse_solve_policy_auto,
                                         d_buffer.ptr),
              rocsparse_status_success);
    ASSERT_NO_FATAL_FAILURE(
        expect_success(rocsparse::bsrsv_analysis_template<test_T, int32_t, int32_t>(
            objects.handle,
            rocsparse_direction_row,
            rocsparse_operation_none,
            kMb,
            nnzb32,
            objects.descr,
            d_val.ptr,
            d_row32.ptr,
            d_col32.ptr,
            kBlockDim,
            objects.info,
            rocsparse_analysis_policy_reuse,
            rocsparse_solve_policy_auto,
            d_buffer.ptr)));
    EXPECT_EQ(container->get_indextype(), rocsparse_indextype_i32);

    const test_T alpha = static_cast<test_T>(1);
    ASSERT_EQ(rocsparse_sbsrsv_solve(objects.handle,
                                     rocsparse_direction_row,
                                     rocsparse_operation_none,
                                     kMb,
                                     nnzb32,
                                     &alpha,
                                     objects.descr,
                                     d_val.ptr,
                                     d_row32.ptr,
                                     d_col32.ptr,
                                     kBlockDim,
                                     objects.info,
                                     d_x.ptr,
                                     d_y.ptr,
                                     rocsparse_solve_policy_auto,
                                     d_buffer.ptr),
              rocsparse_status_success);
    rocsparse_int          position = -2;
    const rocsparse_status status
        = rocsparse_bsrsv_zero_pivot(objects.handle, objects.info, &position);
    EXPECT_EQ(status, rocsparse_status_zero_pivot);
    EXPECT_EQ(position, static_cast<rocsparse_int>(3));
    ASSERT_EQ(rocsparse_bsric0_clear(objects.handle, objects.info), rocsparse_status_success);
}

TEST(internal_level2_bsrsv_wide, force_keeps_borrowed_bsric0_analysis)
{
    ASSERT_NO_FATAL_FAILURE(expect_borrowed_analysis_survives_force());
}

TEST(internal_level2_bsrsv_wide, reuse_borrowed_bsric0_solves)
{
    ASSERT_NO_FATAL_FAILURE(expect_borrowed_reuse_solves());
}

TEST(internal_level2_bsrsv_wide, reuse_borrowed_structural_zero_reported)
{
    // Block row 3 has no diagonal block. The donor's analysis finds it and the
    // borrowed solve must report row 3, not success.
    ASSERT_NO_FATAL_FAILURE(expect_borrowed_reuse_solves(3));
}

TEST(internal_level2_bsrsv_wide, mixed_width_retypes_pivot_and_solves)
{
    ASSERT_NO_FATAL_FAILURE(expect_mixed_width_retypes_and_solves());
}

TEST(internal_level2_bsrsv_wide, reuse_drops_mismatched_pivot)
{
    ASSERT_NO_FATAL_FAILURE(expect_reuse_drops_mismatched_pivot());
}

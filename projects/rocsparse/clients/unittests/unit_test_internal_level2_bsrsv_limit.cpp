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
// Unit tests for the bsrsv solve dispatch-limit check (AISPARSE-656).
//
// The bsrsv solve kernels map one wavefront to one block row and spin-wait on
// the rows they depend on, so they need every block row in a single dispatch:
// a clamped, grid-striding launch can deadlock. bsrsv analysis and solve reject
// a block-row count whose grid exceeds get_grid_size_x instead of clamping. The
// tests shrink handle->properties.maxGridSize[0] to one below the solve grid of
// a small lower block-bidiagonal matrix, check that analysis and solve return
// rocsparse_status_not_implemented and leave y untouched, then check that at
// exactly the solve grid both succeed and match a host forward substitution.
//
// The block dimensions select every solve kernel family on wavefront 64: the
// shared-memory kernels for block_dim <= 8, 16 and 32, and the general kernel
// above. Wavefront 32 always runs the general kernel.
//
#include "unit_test_utils.hpp"

#include "rocsparse.h"
#include "rocsparse_handle.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    constexpr rocsparse_int MB = 1000;

    // Threads per block of every bsrsv solve launch.
    constexpr int64_t SOLVE_BLOCKSIZE = 128;

    // Lower block-bidiagonal BSR matrix: block row 0 holds (0, 0); block row
    // i > 0 holds (i, i - 1) and (i, i). The upper triangle of each diagonal
    // block holds a value the lower solve must ignore.
    struct BlockBidiagonal
    {
        rocsparse_int              bd;
        rocsparse_direction        dir;
        std::vector<rocsparse_int> row_ptr;
        std::vector<rocsparse_int> col_ind;
        std::vector<double>        val;

        static double diag_entry(rocsparse_int i, rocsparse_int bi, rocsparse_int bj, int bd)
        {
            if(bi == bj)
            {
                return 2.0 + 0.5 * (i % 3);
            }
            return (bj < bi) ? 0.125 / bd * (1 + (bi + bj) % 3) : 7.0;
        }

        static double off_entry(rocsparse_int bi, rocsparse_int bj, int bd)
        {
            return 0.25 / bd * (1 + (bi * bd + bj) % 5);
        }

        void push_block(rocsparse_int col, rocsparse_int i, bool diag)
        {
            const size_t base = val.size();
            col_ind.push_back(col);
            val.resize(base + static_cast<size_t>(bd) * bd);
            for(rocsparse_int bi = 0; bi < bd; ++bi)
            {
                for(rocsparse_int bj = 0; bj < bd; ++bj)
                {
                    const size_t k = (dir == rocsparse_direction_row) ? bi * bd + bj : bi + bj * bd;
                    val[base + k]  = diag ? diag_entry(i, bi, bj, bd) : off_entry(bi, bj, bd);
                }
            }
        }

        BlockBidiagonal(rocsparse_int block_dim, rocsparse_direction d)
            : bd(block_dim)
            , dir(d)
        {
            row_ptr.push_back(0);
            for(rocsparse_int i = 0; i < MB; ++i)
            {
                if(i > 0)
                {
                    push_block(i - 1, i, false);
                }
                push_block(i, i, true);
                row_ptr.push_back(static_cast<rocsparse_int>(col_ind.size()));
            }
        }

        rocsparse_int nnzb() const
        {
            return static_cast<rocsparse_int>(col_ind.size());
        }

        // y = alpha * inv(L) * x by block forward substitution.
        std::vector<double> solve(double alpha, const std::vector<double>& x) const
        {
            std::vector<double> y(x.size());
            for(rocsparse_int i = 0; i < MB; ++i)
            {
                for(rocsparse_int bi = 0; bi < bd; ++bi)
                {
                    double r = alpha * x[i * bd + bi];
                    if(i > 0)
                    {
                        for(rocsparse_int bj = 0; bj < bd; ++bj)
                        {
                            r -= off_entry(bi, bj, bd) * y[(i - 1) * bd + bj];
                        }
                    }
                    for(rocsparse_int bj = 0; bj < bi; ++bj)
                    {
                        r -= diag_entry(i, bi, bj, bd) * y[i * bd + bj];
                    }
                    y[i * bd + bi] = r / diag_entry(i, bi, bi, bd);
                }
            }
            return y;
        }
    };

    // Restores maxGridSize[0] even when an assertion returns early.
    struct GridLimitGuard
    {
        rocsparse_handle handle;
        int              saved;

        explicit GridLimitGuard(rocsparse_handle h)
            : handle(h)
            , saved(h->properties.maxGridSize[0])
        {
        }
        ~GridLimitGuard()
        {
            handle->properties.maxGridSize[0] = saved;
        }
        void set(int64_t limit)
        {
            handle->properties.maxGridSize[0] = static_cast<int>(limit);
        }
        void restore()
        {
            handle->properties.maxGridSize[0] = saved;
        }
    };

    struct BsrsvLimitCase
    {
        rocsparse_int       block_dim;
        rocsparse_direction dir;
    };

    class BsrsvLimit : public HandleTest, public ::testing::WithParamInterface<BsrsvLimitCase>
    {
    };
}

TEST_P(BsrsvLimit, rejects_above_dispatch_limit)
{
    const rocsparse_int       bd  = GetParam().block_dim;
    const rocsparse_direction dir = GetParam().dir;
    const BlockBidiagonal     A(bd, dir);

    device_vector<rocsparse_int> d_row_ptr{A.row_ptr};
    device_vector<rocsparse_int> d_col_ind{A.col_ind};
    device_vector<double>        d_val{A.val};
    ASSERT_TRUE(d_row_ptr.ptr && d_col_ind.ptr && d_val.ptr);

    const size_t        n = static_cast<size_t>(MB) * bd;
    std::vector<double> hx(n), hy0(n, -99.0);
    for(size_t i = 0; i < n; ++i)
    {
        hx[i] = static_cast<double>(i % 7 + 1);
    }
    device_vector<double> d_x{hx};
    device_vector<double> d_y{hy0};
    ASSERT_TRUE(d_x.ptr && d_y.ptr);

    rocsparse_mat_descr descr = nullptr;
    rocsparse_mat_info  info  = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);
    ASSERT_EQ(rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_lower),
              rocsparse_status_success);

    const rocsparse_operation trans = rocsparse_operation_none;
    const double              alpha = 1.5;

    size_t buffer_size = 0;
    ASSERT_EQ(rocsparse_dbsrsv_buffer_size(handle,
                                           dir,
                                           trans,
                                           MB,
                                           A.nnzb(),
                                           descr,
                                           d_val,
                                           d_row_ptr,
                                           d_col_ind,
                                           bd,
                                           info,
                                           &buffer_size),
              rocsparse_status_success);
    void* buffer = nullptr;
    ASSERT_EQ(hipMalloc(&buffer, std::max<size_t>(buffer_size, 1)), hipSuccess);

    auto analysis = [&]() {
        return rocsparse_dbsrsv_analysis(handle,
                                         dir,
                                         trans,
                                         MB,
                                         A.nnzb(),
                                         descr,
                                         d_val,
                                         d_row_ptr,
                                         d_col_ind,
                                         bd,
                                         info,
                                         rocsparse_analysis_policy_force,
                                         rocsparse_solve_policy_auto,
                                         buffer);
    };
    auto solve = [&]() {
        return rocsparse_dbsrsv_solve(handle,
                                      dir,
                                      trans,
                                      MB,
                                      A.nnzb(),
                                      &alpha,
                                      descr,
                                      d_val,
                                      d_row_ptr,
                                      d_col_ind,
                                      bd,
                                      info,
                                      d_x,
                                      d_y,
                                      rocsparse_solve_policy_auto,
                                      buffer);
    };

    // One wavefront per block row.
    const int64_t nblocks
        = (static_cast<int64_t>(handle->wavefront_size) * MB - 1) / SOLVE_BLOCKSIZE + 1;
    ASSERT_GE(nblocks, 2);

    GridLimitGuard guard(handle);

    // One below the solve grid: analysis fails.
    guard.set(nblocks - 1);
    EXPECT_EQ(analysis(), rocsparse_status_not_implemented);

    // Analyse within the limit, then shrink it: the solve fails, y untouched.
    guard.restore();
    ASSERT_EQ(analysis(), rocsparse_status_success);
    guard.set(nblocks - 1);
    EXPECT_EQ(solve(), rocsparse_status_not_implemented);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(to_host(d_y), hy0);

    // Exactly the solve grid: the solve succeeds and matches the host.
    guard.set(nblocks);
    ASSERT_EQ(solve(), rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    const std::vector<double> got = to_host(d_y);
    const std::vector<double> ref = A.solve(alpha, hx);
    for(size_t i = 0; i < n; ++i)
    {
        ASSERT_NEAR(got[i], ref[i], 1e-12 * std::max(1.0, std::abs(ref[i]))) << "entry " << i;
    }

    EXPECT_EQ(hipFree(buffer), hipSuccess);
    EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

INSTANTIATE_TEST_SUITE_P(BsrsvLimit,
                         BsrsvLimit,
                         ::testing::Values(BsrsvLimitCase{2, rocsparse_direction_row},
                                           BsrsvLimitCase{2, rocsparse_direction_column},
                                           BsrsvLimitCase{12, rocsparse_direction_row},
                                           BsrsvLimitCase{24, rocsparse_direction_column},
                                           BsrsvLimitCase{40, rocsparse_direction_row},
                                           BsrsvLimitCase{40, rocsparse_direction_column}));

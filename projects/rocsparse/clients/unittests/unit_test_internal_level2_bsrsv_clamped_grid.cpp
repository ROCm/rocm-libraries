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
// Unit tests for the bsrsv solve with a clamped grid (AISPARSE-656).
//
// The bsrsv solve kernels map one wavefront to one block row and spin-wait on
// the rows they depend on. When the solve grid exceeds get_grid_size_x, the
// grid is clamped and each block walks a contiguous chunk of row groups, which
// cannot deadlock where a grid-stride loop would. The tests shrink
// handle->properties.maxGridSize[0] below the solve grid of a block-bidiagonal
// matrix, lower and upper, and check that analysis and solve succeed and match
// a host substitution.
//
// The block dimensions select every solve kernel family on wavefront 64: the
// shared-memory kernels for block_dim <= 8, 16 and 32, and the general kernel
// above. Wavefront 32 always runs the general kernel.
//
// LargeGrid caps a 1,000,000 block-row solve at 200,000 blocks, a grid larger
// than the device holds resident, so each block owns more than one row group.
// A grid-stride loop hangs on it; a hang shows up as the test timing out.
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
    // Threads per block of every bsrsv solve launch.
    constexpr int64_t SOLVE_BLOCKSIZE = 128;

    // Block-bidiagonal BSR matrix. Lower: block row 0 holds (0, 0) and block
    // row i > 0 holds (i, i - 1) and (i, i). Upper: block row i < mb - 1 holds
    // (i, i) and (i, i + 1), and the last holds only its diagonal block. The
    // opposite triangle of each diagonal block holds a value the solve must
    // ignore.
    struct BlockBidiagonal
    {
        rocsparse_int              mb;
        rocsparse_int              bd;
        rocsparse_direction        dir;
        bool                       lower;
        std::vector<rocsparse_int> row_ptr;
        std::vector<rocsparse_int> col_ind;
        std::vector<double>        val;

        double diag_entry(rocsparse_int i, rocsparse_int bi, rocsparse_int bj) const
        {
            if(bi == bj)
            {
                return 2.0 + 0.5 * (i % 3);
            }
            const bool in_triangle = lower ? (bj < bi) : (bj > bi);
            return in_triangle ? 0.125 / bd * (1 + (bi + bj) % 3) : 7.0;
        }

        double off_entry(rocsparse_int bi, rocsparse_int bj) const
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
                    val[base + k]  = diag ? diag_entry(i, bi, bj) : off_entry(bi, bj);
                }
            }
        }

        BlockBidiagonal(rocsparse_int       nblockrows,
                        rocsparse_int       block_dim,
                        rocsparse_direction d,
                        rocsparse_fill_mode fill)
            : mb(nblockrows)
            , bd(block_dim)
            , dir(d)
            , lower(fill == rocsparse_fill_mode_lower)
        {
            row_ptr.reserve(static_cast<size_t>(mb) + 1);
            col_ind.reserve(2 * static_cast<size_t>(mb));
            val.reserve(2 * static_cast<size_t>(mb) * bd * bd);
            row_ptr.push_back(0);
            for(rocsparse_int i = 0; i < mb; ++i)
            {
                if(lower && i > 0)
                {
                    push_block(i - 1, i, false);
                }
                push_block(i, i, true);
                if(!lower && i < mb - 1)
                {
                    push_block(i + 1, i, false);
                }
                row_ptr.push_back(static_cast<rocsparse_int>(col_ind.size()));
            }
        }

        rocsparse_int nnzb() const
        {
            return static_cast<rocsparse_int>(col_ind.size());
        }

        // y = alpha * inv(A) * x by block forward or backward substitution.
        std::vector<double> solve(double alpha, const std::vector<double>& x) const
        {
            std::vector<double> y(x.size());
            for(rocsparse_int k = 0; k < mb; ++k)
            {
                const rocsparse_int i   = lower ? k : mb - 1 - k;
                const bool          has = lower ? (i > 0) : (i < mb - 1);
                const rocsparse_int nb  = lower ? i - 1 : i + 1;
                for(rocsparse_int l = 0; l < bd; ++l)
                {
                    const rocsparse_int bi = lower ? l : bd - 1 - l;
                    double              r  = alpha * x[static_cast<size_t>(i) * bd + bi];
                    if(has)
                    {
                        for(rocsparse_int bj = 0; bj < bd; ++bj)
                        {
                            r -= off_entry(bi, bj) * y[static_cast<size_t>(nb) * bd + bj];
                        }
                    }
                    for(rocsparse_int bj = 0; bj < bd; ++bj)
                    {
                        if(lower ? (bj < bi) : (bj > bi))
                        {
                            r -= diag_entry(i, bi, bj) * y[static_cast<size_t>(i) * bd + bj];
                        }
                    }
                    y[static_cast<size_t>(i) * bd + bi] = r / diag_entry(i, bi, bi);
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
    };

    // Analyses and solves A with maxGridSize[0] set to each of caps in turn,
    // from a poisoned y every time, and checks y against the host solve.
    void check_clamped_solves(rocsparse_handle       handle,
                              const BlockBidiagonal& A,
                              rocsparse_fill_mode    fill,
                              std::vector<int64_t>   caps)
    {
        device_vector<rocsparse_int> d_row_ptr{A.row_ptr};
        device_vector<rocsparse_int> d_col_ind{A.col_ind};
        device_vector<double>        d_val{A.val};
        ASSERT_TRUE(d_row_ptr.ptr && d_col_ind.ptr && d_val.ptr);

        const size_t        n = static_cast<size_t>(A.mb) * A.bd;
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
        ASSERT_EQ(rocsparse_set_mat_fill_mode(descr, fill), rocsparse_status_success);

        const rocsparse_operation trans = rocsparse_operation_none;
        const double              alpha = 1.5;

        size_t buffer_size = 0;
        ASSERT_EQ(rocsparse_dbsrsv_buffer_size(handle,
                                               A.dir,
                                               trans,
                                               A.mb,
                                               A.nnzb(),
                                               descr,
                                               d_val,
                                               d_row_ptr,
                                               d_col_ind,
                                               A.bd,
                                               info,
                                               &buffer_size),
                  rocsparse_status_success);
        void* buffer = nullptr;
        ASSERT_EQ(hipMalloc(&buffer, std::max<size_t>(buffer_size, 1)), hipSuccess);

        const std::vector<double> ref = A.solve(alpha, hx);

        GridLimitGuard guard(handle);
        for(const int64_t cap : caps)
        {
            SCOPED_TRACE(::testing::Message() << "maxGridSize[0] = " << cap);
            guard.set(cap);

            ASSERT_EQ(rocsparse_dbsrsv_analysis(handle,
                                                A.dir,
                                                trans,
                                                A.mb,
                                                A.nnzb(),
                                                descr,
                                                d_val,
                                                d_row_ptr,
                                                d_col_ind,
                                                A.bd,
                                                info,
                                                rocsparse_analysis_policy_force,
                                                rocsparse_solve_policy_auto,
                                                buffer),
                      rocsparse_status_success);

            ASSERT_EQ(hipMemcpy(d_y, hy0.data(), sizeof(double) * n, hipMemcpyHostToDevice),
                      hipSuccess);
            ASSERT_EQ(rocsparse_dbsrsv_solve(handle,
                                             A.dir,
                                             trans,
                                             A.mb,
                                             A.nnzb(),
                                             &alpha,
                                             descr,
                                             d_val,
                                             d_row_ptr,
                                             d_col_ind,
                                             A.bd,
                                             info,
                                             d_x,
                                             d_y,
                                             rocsparse_solve_policy_auto,
                                             buffer),
                      rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

            rocsparse_int zero_pivot = 0;
            EXPECT_EQ(rocsparse_bsrsv_zero_pivot(handle, info, &zero_pivot),
                      rocsparse_status_success);

            const std::vector<double> got = to_host(d_y);
            for(size_t i = 0; i < n; ++i)
            {
                ASSERT_NEAR(got[i], ref[i], 1e-12 * std::max(1.0, std::abs(ref[i])))
                    << "entry " << i;
            }
        }

        EXPECT_EQ(hipFree(buffer), hipSuccess);
        EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
    }

    // Blocks of the unclamped solve grid: one wavefront per block row.
    int64_t solve_grid(rocsparse_handle handle, rocsparse_int mb)
    {
        return (static_cast<int64_t>(handle->wavefront_size) * mb - 1) / SOLVE_BLOCKSIZE + 1;
    }

    struct BsrsvClampedGridCase
    {
        rocsparse_int       block_dim;
        rocsparse_direction dir;
        rocsparse_fill_mode fill;
    };

    class BsrsvClampedGrid : public HandleTest,
                             public ::testing::WithParamInterface<BsrsvClampedGridCase>
    {
    };

    class BsrsvClampedGridLarge : public HandleTest,
                                  public ::testing::WithParamInterface<rocsparse_fill_mode>
    {
    };
}

TEST_P(BsrsvClampedGrid, matches_reference)
{
    constexpr rocsparse_int MB = 1000;

    const BlockBidiagonal A(MB, GetParam().block_dim, GetParam().dir, GetParam().fill);

    const int64_t nblocks = solve_grid(handle, MB);
    ASSERT_GE(nblocks, 5);

    check_clamped_solves(handle, A, GetParam().fill, {1, 7, nblocks / 5, nblocks - 1, nblocks});
}

INSTANTIATE_TEST_SUITE_P(
    BsrsvClampedGrid,
    BsrsvClampedGrid,
    ::testing::Values(
        BsrsvClampedGridCase{2, rocsparse_direction_row, rocsparse_fill_mode_lower},
        BsrsvClampedGridCase{2, rocsparse_direction_column, rocsparse_fill_mode_lower},
        BsrsvClampedGridCase{12, rocsparse_direction_row, rocsparse_fill_mode_lower},
        BsrsvClampedGridCase{24, rocsparse_direction_column, rocsparse_fill_mode_lower},
        BsrsvClampedGridCase{40, rocsparse_direction_row, rocsparse_fill_mode_lower},
        BsrsvClampedGridCase{40, rocsparse_direction_column, rocsparse_fill_mode_lower},
        BsrsvClampedGridCase{2, rocsparse_direction_row, rocsparse_fill_mode_upper},
        BsrsvClampedGridCase{2, rocsparse_direction_column, rocsparse_fill_mode_upper},
        BsrsvClampedGridCase{12, rocsparse_direction_column, rocsparse_fill_mode_upper},
        BsrsvClampedGridCase{24, rocsparse_direction_row, rocsparse_fill_mode_upper},
        BsrsvClampedGridCase{40, rocsparse_direction_row, rocsparse_fill_mode_upper},
        BsrsvClampedGridCase{40, rocsparse_direction_column, rocsparse_fill_mode_upper}));

TEST_P(BsrsvClampedGridLarge, matches_reference)
{
    constexpr rocsparse_int MB  = 1000000;
    constexpr int64_t       CAP = 200000;

    const BlockBidiagonal A(MB, 2, rocsparse_direction_row, GetParam());

    ASSERT_GT(solve_grid(handle, MB), CAP);

    check_clamped_solves(handle, A, GetParam(), {CAP});
}

INSTANTIATE_TEST_SUITE_P(BsrsvClampedGridLarge,
                         BsrsvClampedGridLarge,
                         ::testing::Values(rocsparse_fill_mode_lower, rocsparse_fill_mode_upper));

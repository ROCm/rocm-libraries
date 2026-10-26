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
// Unit tests for the CSR triangular solve with multiple right hand sides
// (library/src/level3/csrsm_device.h).
//
// FOCUS (AISPARSE-669): the solve kernel flattens a two dimensional iteration
// space (RHS panel, row) onto grid.x and recovers BOTH coordinates from the
// flattened index by dividing and remaindering against m. A grid that does not
// cover the whole flattened space therefore does not merely drop trailing work,
// it pairs the wrong row with the wrong RHS panel. rocsparse::csrsm now clamps
// grid.x (csrsm_solve_grid_size) and grid-strides over the remainder.
//
// The clamp only bites at handle->properties.maxGridSize[0] (~2.1e9 blocks),
// which is unreachable in a test: the done_array alone would need 8 GB. So this
// suite launches rocsparse::csrsm DIRECTLY with a deliberately undersized grid,
// which is exactly the situation the clamp creates, and checks the full solution.
// A kernel without the grid-stride loop leaves every panel past the first sweep
// holding its untouched right hand side.
//
// The problem is a 16 x 16 lower bidiagonal matrix with 300 right hand sides, so
// the whole footprint is a few tens of kilobytes and this runs on any GPU
// (including the 15 GB gfx1201). No memory guard: this test always executes.
//
// The solve also has to refuse a problem whose m alone does not fit on grid.x:
// one RHS panel needs m blocks and the kernel cannot stride below that. The last
// cases drive the public rocsparse_scsrsm_solve entry point and expect
// rocsparse_status_invalid_size instead of a launch that fails without an error.
//
// TARGET: csrsm_device.h is device code, so this file builds into
// rocsparse-unit-test-device, NOT the host-only rocsparse-unit-test.
//

#include "unit_test_utils.hpp"

#include "csrsm_device.h"

// ScopedMaxGridSizeX: shrinks handle->properties.maxGridSize[0], the limit
// get_grid_size_x clamps the csrsm solve grid against.
#include "unit_test_grid_clamp.hpp"

#include "rocsparse.h"

#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <vector>

namespace
{
    using test_I = int32_t;
    using test_J = int32_t;
    using test_T = float;

    constexpr uint32_t csrsm_blocksize = 64;

    // A is lower bidiagonal with A(i,i) = 1 and A(i,i-1) = -1. With alpha = 1 and
    // B(i,c) = c + 1, the solve X = A^-1 * B telescopes to X(i,c) = (i+1)*(c+1),
    // which is an exact float for every value used here, so the expected result
    // can be compared bit for bit.
    constexpr test_J csrsm_m    = 16;
    constexpr test_J csrsm_nrhs = 300; // 5 panels of 64, the last one partial

    constexpr int64_t csrsm_npanels = (csrsm_nrhs - 1) / csrsm_blocksize + 1;
    constexpr int64_t csrsm_blocks  = csrsm_npanels * csrsm_m;

    test_T expected_at(test_J row, test_J col)
    {
        return static_cast<test_T>(row + 1) * static_cast<test_T>(col + 1);
    }

    // Run rocsparse::csrsm on `grid_x` blocks and read B back. `grid_x` is the
    // caller's choice rather than the value csrsm_solve_grid_size would pick, so
    // the grid-stride loop can be driven with far fewer blocks than the flattened
    // space needs.
    void solve_on_grid(int64_t grid_x, std::vector<test_T>& result, test_J& pivot)
    {
        // A, in CSR. Row 0 holds only the diagonal; every later row holds the
        // subdiagonal entry followed by the diagonal, ascending as csrsm expects.
        std::vector<test_I> row_ptr{0};
        std::vector<test_J> col_ind;
        std::vector<test_T> val;
        for(test_J i = 0; i < csrsm_m; ++i)
        {
            if(i > 0)
            {
                col_ind.push_back(i - 1);
                val.push_back(static_cast<test_T>(-1));
            }
            col_ind.push_back(i);
            val.push_back(static_cast<test_T>(1));
            row_ptr.push_back(static_cast<test_I>(col_ind.size()));
        }

        // For a lower triangular matrix stored in natural order the identity
        // permutation is already a topological order of the dependency graph,
        // which is what the analysis phase would have produced.
        std::vector<test_J> map(csrsm_m);
        for(test_J i = 0; i < csrsm_m; ++i)
        {
            map[i] = i;
        }

        // B is row major with ldb = nrhs, the layout csrsm_solve_dispatch hands the
        // kernel after transposing.
        std::vector<test_T> B(static_cast<size_t>(csrsm_m) * csrsm_nrhs);
        for(test_J i = 0; i < csrsm_m; ++i)
        {
            for(test_J c = 0; c < csrsm_nrhs; ++c)
            {
                B[static_cast<size_t>(i) * csrsm_nrhs + c] = static_cast<test_T>(c + 1);
            }
        }

        rocsparse_ut::device_vector<test_I> d_row_ptr(row_ptr);
        rocsparse_ut::device_vector<test_J> d_col_ind(col_ind);
        rocsparse_ut::device_vector<test_T> d_val(val);
        rocsparse_ut::device_vector<test_J> d_map(map);
        rocsparse_ut::device_vector<test_T> d_B(B);
        rocsparse_ut::device_vector<int>    d_done(std::vector<int>(csrsm_blocks, 0));
        rocsparse_ut::device_vector<test_J> d_pivot(
            std::vector<test_J>{std::numeric_limits<test_J>::max()});

        ASSERT_NE(d_row_ptr.ptr, nullptr);
        ASSERT_NE(d_col_ind.ptr, nullptr);
        ASSERT_NE(d_val.ptr, nullptr);
        ASSERT_NE(d_map.ptr, nullptr);
        ASSERT_NE(d_B.ptr, nullptr);
        ASSERT_NE(d_done.ptr, nullptr);
        ASSERT_NE(d_pivot.ptr, nullptr);

        const test_T alpha = static_cast<test_T>(1);

        hipLaunchKernelGGL(
            (rocsparse::csrsm<csrsm_blocksize, false, test_I, test_J, test_T>),
            dim3(static_cast<uint32_t>(grid_x)),
            dim3(csrsm_blocksize),
            0,
            0,
            rocsparse_operation_none,
            csrsm_m,
            csrsm_nrhs,
            rocsparse::to_const_host_device_scalar(rocsparse_pointer_mode_host, &alpha),
            d_row_ptr.ptr,
            d_col_ind.ptr,
            d_val.ptr,
            d_B.ptr,
            static_cast<int64_t>(csrsm_nrhs),
            d_done.ptr,
            d_map.ptr,
            d_pivot.ptr,
            rocsparse_index_base_zero,
            rocsparse_fill_mode_lower,
            rocsparse_diag_type_non_unit,
            true);

        ASSERT_EQ(hipGetLastError(), hipSuccess);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        result = rocsparse_ut::to_host(d_B);
        pivot  = rocsparse_ut::to_host(d_pivot)[0];
    }

    // Compare against the closed form solution and report the FIRST offending
    // (row, column) pair, so a failure names the coordinate that was skipped or
    // mispaired instead of drowning the log in 4800 expectations.
    void expect_exact_solution(const std::vector<test_T>& got, int64_t grid_x)
    {
        ASSERT_EQ(static_cast<int64_t>(got.size()), static_cast<int64_t>(csrsm_m) * csrsm_nrhs);

        for(test_J i = 0; i < csrsm_m; ++i)
        {
            for(test_J c = 0; c < csrsm_nrhs; ++c)
            {
                const test_T value = got[static_cast<size_t>(i) * csrsm_nrhs + c];
                ASSERT_EQ(value, expected_at(i, c))
                    << "wrong solution at row " << i << ", rhs " << c << " (grid.x=" << grid_x
                    << ", flattened blocks=" << csrsm_blocks << ", panels=" << csrsm_npanels
                    << "). A grid smaller than the flattened space must be covered by the "
                       "kernel's grid-stride loop.";
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host-pure grid sizing (csrsm_device.h).
// ---------------------------------------------------------------------------

TEST(internal_level3_csrsm, num_blocks_is_computed_in_64_bit)
{
    EXPECT_EQ(rocsparse::csrsm_num_blocks(0, 8, 64), 0);
    EXPECT_EQ(rocsparse::csrsm_num_blocks(10, 1, 64), 10);
    EXPECT_EQ(rocsparse::csrsm_num_blocks(10, 64, 64), 10);
    EXPECT_EQ(rocsparse::csrsm_num_blocks(10, 65, 64), 20);

    // ((nrhs - 1) / blockdim + 1) * m = 40000 * 70000 = 2.8e9, which does not fit
    // in the int32 the original expression evaluated it in.
    EXPECT_EQ(rocsparse::csrsm_num_blocks(70000, 40000 * 64, 64), int64_t{2800000000});
    EXPECT_GT(rocsparse::csrsm_num_blocks(70000, 40000 * 64, 64),
              int64_t{std::numeric_limits<int32_t>::max()});
}

TEST(internal_level3_csrsm, solve_grid_size_clamps_to_whole_panels)
{
    constexpr int64_t max_grid_x = 2147483647;

    // Below the limit the grid is the full flattened space.
    EXPECT_EQ(rocsparse::csrsm_solve_grid_size(1000, 64 * 8, 64, max_grid_x), 8000);

    // Above the limit it is clamped, and rounded DOWN to a whole number of RHS
    // panels so the kernel's stride stays a multiple of m.
    EXPECT_EQ(rocsparse::csrsm_solve_grid_size(1000, 64 * 8, 64, 2500), 2000);
    EXPECT_EQ(rocsparse::csrsm_solve_grid_size(1000, 64 * 8, 64, 999), 1000);

    // The overflowing configuration: 2.8e9 blocks clamped to the grid limit, then
    // trimmed to 30678 whole panels of 70000 rows.
    EXPECT_EQ(rocsparse::csrsm_solve_grid_size(70000, 40000 * 64, 64, max_grid_x),
              int64_t{30678} * 70000);
    EXPECT_LE(rocsparse::csrsm_solve_grid_size(70000, 40000 * 64, 64, max_grid_x), max_grid_x);

    // Degenerate m is not a division by zero.
    EXPECT_EQ(rocsparse::csrsm_solve_grid_size(0, 8, 64, max_grid_x), 0);
}

TEST(internal_level3_csrsm, solve_grid_size_is_always_a_multiple_of_m)
{
    // The kernel derives its stride as (hipGridDim_x / m) * m. If grid.x were not
    // a whole number of panels the stride would no longer pin a block to a single
    // row, and a block could wait on a done_array flag owned by a block that has
    // not been dispatched yet.
    for(int64_t m = 1; m <= 37; ++m)
    {
        for(int64_t max_grid_x = 1; max_grid_x <= 200; max_grid_x += 7)
        {
            const int64_t grid = rocsparse::csrsm_solve_grid_size(m, 64 * 5, 64, max_grid_x);
            EXPECT_EQ(grid % m, 0) << "m=" << m << ", max_grid_x=" << max_grid_x;
            EXPECT_GE(grid, m) << "m=" << m << ", max_grid_x=" << max_grid_x;
            EXPECT_LE(grid, rocsparse::csrsm_num_blocks(m, 64 * 5, 64));
        }
    }
}

// ---------------------------------------------------------------------------
// End-to-end: the flattened grid-stride loop on the GPU.
// ---------------------------------------------------------------------------

// Control: the grid covers the whole flattened space, so the stride loop runs
// exactly one iteration. This is the pre-existing behaviour and must not change.
TEST(internal_level3_csrsm, solve_full_grid)
{
    std::vector<test_T> got;
    test_J              pivot = 0;
    ASSERT_NO_FATAL_FAILURE(solve_on_grid(csrsm_blocks, got, pivot));
    EXPECT_EQ(pivot, std::numeric_limits<test_J>::max());
    expect_exact_solution(got, csrsm_blocks);
}

// Two RHS panels per sweep: 80 flattened blocks covered by 32, so three sweeps.
TEST(internal_level3_csrsm, solve_undersized_grid_two_panels)
{
    const int64_t grid_x = 2 * csrsm_m;
    ASSERT_LT(grid_x, csrsm_blocks);

    std::vector<test_T> got;
    test_J              pivot = 0;
    ASSERT_NO_FATAL_FAILURE(solve_on_grid(grid_x, got, pivot));
    EXPECT_EQ(pivot, std::numeric_limits<test_J>::max());
    expect_exact_solution(got, grid_x);
}

// The tightest grid the clamp can produce: a single RHS panel, so all five panels
// are reached only by striding. Also the case where hipGridDim_x == m exactly.
TEST(internal_level3_csrsm, solve_undersized_grid_single_panel)
{
    const int64_t grid_x = csrsm_m;
    ASSERT_LT(grid_x, csrsm_blocks);

    std::vector<test_T> got;
    test_J              pivot = 0;
    ASSERT_NO_FATAL_FAILURE(solve_on_grid(grid_x, got, pivot));
    EXPECT_EQ(pivot, std::numeric_limits<test_J>::max());
    expect_exact_solution(got, grid_x);
}

// A grid that is NOT a whole number of RHS panels. csrsm_solve_grid_size never
// produces one, but the kernel is launchable directly (this file does it), so the
// contract has to hold for any grid.x, and the failure mode is silent: the stride
// is rounded down to whole panels, so without a guard the ragged blocks past the
// last whole panel re-run (panel, row) pairs that a lower numbered block already
// owns. csrsm_block_device is NOT idempotent -- it reads its own B element as the
// right hand side and overwrites it -- so a second visit corrupts the result.
TEST(internal_level3_csrsm, solve_ragged_grid_is_not_a_multiple_of_m)
{
    const int64_t grid_x = 2 * csrsm_m + 5;
    ASSERT_LT(grid_x, csrsm_blocks);
    ASSERT_NE(grid_x % csrsm_m, 0);

    std::vector<test_T> got;
    test_J              pivot = 0;
    ASSERT_NO_FATAL_FAILURE(solve_on_grid(grid_x, got, pivot));
    EXPECT_EQ(pivot, std::numeric_limits<test_J>::max());
    expect_exact_solution(got, grid_x);
}

// ---------------------------------------------------------------------------
// Public rocsparse_scsrsm_solve: m beyond the grid.x limit is invalid_size.
// ---------------------------------------------------------------------------

namespace
{
    // Owns the handle, descriptor and info of a csrsm call so they are released on
    // every return path.
    struct CsrsmContext
    {
        rocsparse_handle    handle = nullptr;
        rocsparse_mat_descr descr  = nullptr;
        rocsparse_mat_info  info   = nullptr;

        CsrsmContext() = default;

        CsrsmContext(const CsrsmContext&) = delete;

        CsrsmContext& operator=(const CsrsmContext&) = delete;

        bool create()
        {
            return rocsparse_create_handle(&handle) == rocsparse_status_success
                   && rocsparse_create_mat_descr(&descr) == rocsparse_status_success
                   && rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_lower)
                          == rocsparse_status_success
                   && rocsparse_create_mat_info(&info) == rocsparse_status_success;
        }

        ~CsrsmContext()
        {
            if(info != nullptr)
            {
                (void)rocsparse_destroy_mat_info(info);
            }
            if(descr != nullptr)
            {
                (void)rocsparse_destroy_mat_descr(descr);
            }
            if(handle != nullptr)
            {
                (void)rocsparse_destroy_handle(handle);
            }
        }
    };

    struct CsrsmSolveResult
    {
        rocsparse_status    status     = rocsparse_status_internal_error;
        hipError_t          last_error = hipSuccess;
        std::vector<test_T> B;
    };

    // Buffer size, analysis and solve of A * X = B for a lower triangular A that is
    // already on the host. When max_grid_x > 0 the grid.x limit is shrunk to it for
    // the solve only. B is read back after the solve, and hipGetLastError() is
    // sampled right after it: a launch that failed without being reported leaves
    // its error there.
    CsrsmSolveResult csrsm_solve_call(rocsparse_operation        trans_B,
                                      test_J                     m,
                                      test_J                     nrhs,
                                      const std::vector<test_I>& row_ptr,
                                      const std::vector<test_J>& col_ind,
                                      const std::vector<test_T>& val,
                                      const std::vector<test_T>& B,
                                      test_J                     ldb,
                                      int                        max_grid_x)
    {
        CsrsmSolveResult result;

        CsrsmContext ctx;
        if(!ctx.create())
        {
            ADD_FAILURE() << "handle, descriptor or info creation failed";
            return result;
        }

        rocsparse_ut::device_vector<test_I> d_row_ptr(row_ptr);
        rocsparse_ut::device_vector<test_J> d_col_ind(col_ind);
        rocsparse_ut::device_vector<test_T> d_val(val);
        rocsparse_ut::device_vector<test_T> d_B(B);
        if(d_row_ptr.ptr == nullptr || d_col_ind.ptr == nullptr || d_val.ptr == nullptr
           || d_B.ptr == nullptr)
        {
            ADD_FAILURE() << "device allocation failed";
            return result;
        }

        const test_T alpha = static_cast<test_T>(1);
        const test_I nnz   = static_cast<test_I>(col_ind.size());

        size_t buffer_size = 0;
        if(rocsparse_scsrsm_buffer_size(ctx.handle,
                                        rocsparse_operation_none,
                                        trans_B,
                                        m,
                                        nrhs,
                                        nnz,
                                        &alpha,
                                        ctx.descr,
                                        d_val.ptr,
                                        d_row_ptr.ptr,
                                        d_col_ind.ptr,
                                        d_B.ptr,
                                        ldb,
                                        ctx.info,
                                        rocsparse_solve_policy_auto,
                                        &buffer_size)
           != rocsparse_status_success)
        {
            ADD_FAILURE() << "rocsparse_scsrsm_buffer_size failed";
            return result;
        }

        rocsparse_ut::device_vector<char> d_buffer(buffer_size);
        if(d_buffer.ptr == nullptr)
        {
            ADD_FAILURE() << "temporary buffer allocation failed";
            return result;
        }

        if(rocsparse_scsrsm_analysis(ctx.handle,
                                     rocsparse_operation_none,
                                     trans_B,
                                     m,
                                     nrhs,
                                     nnz,
                                     &alpha,
                                     ctx.descr,
                                     d_val.ptr,
                                     d_row_ptr.ptr,
                                     d_col_ind.ptr,
                                     d_B.ptr,
                                     ldb,
                                     ctx.info,
                                     rocsparse_analysis_policy_reuse,
                                     rocsparse_solve_policy_auto,
                                     d_buffer.ptr)
           != rocsparse_status_success)
        {
            ADD_FAILURE() << "rocsparse_scsrsm_analysis failed";
            return result;
        }

        {
            std::unique_ptr<rocsparse_ut::ScopedMaxGridSizeX> clamp;
            if(max_grid_x > 0)
            {
                clamp.reset(new rocsparse_ut::ScopedMaxGridSizeX(ctx.handle, max_grid_x));
            }

            result.status = rocsparse_scsrsm_solve(ctx.handle,
                                                   rocsparse_operation_none,
                                                   trans_B,
                                                   m,
                                                   nrhs,
                                                   nnz,
                                                   &alpha,
                                                   ctx.descr,
                                                   d_val.ptr,
                                                   d_row_ptr.ptr,
                                                   d_col_ind.ptr,
                                                   d_B.ptr,
                                                   ldb,
                                                   ctx.info,
                                                   rocsparse_solve_policy_auto,
                                                   d_buffer.ptr);
        }

        result.last_error = hipGetLastError();
        EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
        result.B = rocsparse_ut::to_host(d_B);
        return result;
    }

    // The lower bidiagonal matrix of the kernel tests above, with m rows.
    void make_bidiagonal(test_J               m,
                         std::vector<test_I>& row_ptr,
                         std::vector<test_J>& col_ind,
                         std::vector<test_T>& val)
    {
        row_ptr.assign(1, 0);
        for(test_J i = 0; i < m; ++i)
        {
            if(i > 0)
            {
                col_ind.push_back(i - 1);
                val.push_back(static_cast<test_T>(-1));
            }
            col_ind.push_back(i);
            val.push_back(static_cast<test_T>(1));
            row_ptr.push_back(static_cast<test_I>(col_ind.size()));
        }
    }

    // The solve block size is picked from nrhs (64 up to 64 right hand sides, then
    // 128, 256, 512 and 1024 above 512), so one case per block size.
    constexpr test_J csrsm_nrhs_per_blocksize[] = {2, 65, 129, 257, 513};
}

// grid.x is limited to get_grid_size_x(handle, m, blockdim) blocks. One RHS panel
// needs m of them, so m past the limit has to be rejected: the launch would fail
// and, without ROCSPARSE_DEBUG_KERNEL_LAUNCH, nothing would report it.
//
// The limit is shrunk to m - 1 instead of growing m to the real threshold. The
// grid is then exactly one block short, and limit == m is the control that the
// check is not off by one.
TEST(internal_level3_csrsm, solve_rejects_m_above_grid_limit)
{
    constexpr test_J m = 16;

    std::vector<test_I> row_ptr;
    std::vector<test_J> col_ind;
    std::vector<test_T> val;
    make_bidiagonal(m, row_ptr, col_ind, val);

    for(const test_J nrhs : csrsm_nrhs_per_blocksize)
    {
        SCOPED_TRACE(::testing::Message() << "nrhs=" << nrhs);

        // Column major B with ldb = m and B(i, c) = c + 1.
        std::vector<test_T> B(static_cast<size_t>(m) * nrhs);
        for(test_J c = 0; c < nrhs; ++c)
        {
            for(test_J i = 0; i < m; ++i)
            {
                B[static_cast<size_t>(c) * m + i] = static_cast<test_T>(c + 1);
            }
        }

        const CsrsmSolveResult rejected = csrsm_solve_call(
            rocsparse_operation_none, m, nrhs, row_ptr, col_ind, val, B, m, m - 1);
        EXPECT_EQ(rejected.status, rocsparse_status_invalid_size);
        EXPECT_EQ(rejected.last_error, hipSuccess);
        EXPECT_EQ(rejected.B, B) << "B must be left untouched";

        const CsrsmSolveResult accepted
            = csrsm_solve_call(rocsparse_operation_none, m, nrhs, row_ptr, col_ind, val, B, m, m);
        EXPECT_EQ(accepted.status, rocsparse_status_success);
        EXPECT_EQ(accepted.last_error, hipSuccess);
        ASSERT_EQ(accepted.B.size(), B.size());
        for(test_J c = 0; c < nrhs; ++c)
        {
            for(test_J i = 0; i < m; ++i)
            {
                ASSERT_EQ(accepted.B[static_cast<size_t>(c) * m + i], expected_at(i, c))
                    << "wrong solution at row " << i << ", rhs " << c;
            }
        }
    }
}

// The real thresholds, with the device limit left alone: (2^32 - 1) / blockdim
// grid.x blocks, i.e. m > 4194303 for nrhs > 512 and m > 8388607 for
// 256 < nrhs <= 512. A diagonal A keeps the footprint to a few hundred megabytes.
// The size is rejected before any work is queued, so B is never read; with
// trans_B == transpose only ldb >= nrhs is required, which lets B be a single row
// instead of the m * nrhs floats (8.6 GB for the first case) it nominally holds.
TEST(internal_level3_csrsm, solve_rejects_m_above_dispatch_limit)
{
    struct Case
    {
        test_J m;
        test_J nrhs;
    };
    const Case cases[] = {{4194304, 513}, {8388608, 512}};

    for(const Case& c : cases)
    {
        SCOPED_TRACE(::testing::Message() << "m=" << c.m << " nrhs=" << c.nrhs);

        std::vector<test_I> row_ptr(static_cast<size_t>(c.m) + 1);
        std::vector<test_J> col_ind(c.m);
        std::vector<test_T> val(c.m, static_cast<test_T>(1));
        for(test_J i = 0; i < c.m; ++i)
        {
            row_ptr[i] = i;
            col_ind[i] = i;
        }
        row_ptr[c.m] = c.m;

        const std::vector<test_T> B(c.nrhs, static_cast<test_T>(1));

        const CsrsmSolveResult res = csrsm_solve_call(
            rocsparse_operation_transpose, c.m, c.nrhs, row_ptr, col_ind, val, B, c.nrhs, 0);
        EXPECT_EQ(res.status, rocsparse_status_invalid_size);
        EXPECT_EQ(res.last_error, hipSuccess);
        EXPECT_EQ(res.B, B);
    }
}

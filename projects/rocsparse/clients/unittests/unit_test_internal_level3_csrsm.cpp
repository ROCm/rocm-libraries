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
// TARGET: csrsm_device.h is device code, so this file builds into
// rocsparse-unit-test-device, NOT the host-only rocsparse-unit-test.
//

#include "unit_test_utils.hpp"

#include "csrsm_device.h"

#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
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

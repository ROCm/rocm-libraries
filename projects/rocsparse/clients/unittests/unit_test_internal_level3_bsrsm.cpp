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
// Unit tests for the BSR triangular solve with multiple right hand sides
// (library/src/level3/bsrsm_device_large.h, bsrsm_device.h).
//
// FOCUS (AISPARSE-670): rocsparse_bsrsm_template_large.cpp carried three signed
// 32-bit product overflows in its launch grids. The fix forms each product in
// 64 bit, clamps it against handle->properties.maxGridSize[0] and adds a
// grid-stride loop so the clamped grid still covers the whole iteration space:
//
//   1. the solve grid ((nrhs - 1) / NCOL + 1) * mb, which flattens a two
//      dimensional space (RHS panel, block row) onto grid.x and recovers BOTH
//      coordinates from the flattened index by dividing and remaindering against
//      mb -- so an undersized grid does not merely drop trailing work, it pairs
//      the wrong block row with the wrong RHS panel;
//   2. the transpose gather grid wfsize * nnzb (rocsparse::bsr_gather, shared
//      with bsrsv/AISPARSE-656);
//   3. the right hand side setup grid mb * block_dim
//      (rocsparse::bsrsm_copy_scale).
//
// Each clamp only bites at maxGridSize[0] (~2.1e9 blocks), which is unreachable
// in a test: the solve's done_array alone would need 8 GB. So this suite launches
// the three kernels DIRECTLY with deliberately undersized grids, which is exactly
// the situation the clamps create, and checks the full numerical output. Without
// the grid-stride loops the panels/elements past the first sweep keep their
// untouched input.
//
// The solve problem is an 8 x 8 block-bidiagonal matrix of 2 x 2 blocks (16 scalar
// rows) with 100 right hand sides, so the whole footprint is a few tens of
// kilobytes and this runs on any GPU (including the 15 GB gfx1201). No memory
// guard: these tests always execute.
//
// TARGET: these are device headers, so this file builds into
// rocsparse-unit-test-device, NOT the host-only rocsparse-unit-test.
//

#include "unit_test_utils.hpp"

#include "bsrsm_device.h"
#include "bsrsm_device_large.h"

#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace
{
    using test_T = float;

    // The production launch picks NCOL = 16 columns per block and
    // M_ = max(4, fnp2(block_dim)) rows per column group, i.e. BLOCKSIZE 64 and
    // WFSIZE 4 for block_dim = 2.
    constexpr uint32_t bsrsm_ncols     = 16;
    constexpr uint32_t bsrsm_blocksize = 64;

    constexpr rocsparse_int bsrsm_mb        = 8;
    constexpr rocsparse_int bsrsm_block_dim = 2;
    constexpr rocsparse_int bsrsm_nrhs      = 100; // 7 panels of 16, the last partial
    constexpr rocsparse_int bsrsm_rows      = bsrsm_mb * bsrsm_block_dim;

    constexpr int64_t bsrsm_npanels = (bsrsm_nrhs - 1) / bsrsm_ncols + 1;
    constexpr int64_t bsrsm_blocks  = bsrsm_npanels * bsrsm_mb;

    // A is block bidiagonal with diagonal blocks I and off-diagonal blocks -I, so
    // the scalar system splits into block_dim independent chains of length mb. With
    // alpha = 1 and X(r, c) = c + 1 on entry:
    //
    //   lower: x(2i + k) - x(2i - 2 + k) = c + 1  =>  x(r, c) = (r / 2 + 1) * (c + 1)
    //   upper: x(2i + k) - x(2i + 2 + k) = c + 1  =>  x(r, c) = (mb - r / 2) * (c + 1)
    //
    // Both are exact floats for every value used here (at most 8 * 100), so the
    // expected result can be compared bit for bit.
    test_T expected_lower(rocsparse_int row, rocsparse_int col)
    {
        return static_cast<test_T>(row / bsrsm_block_dim + 1) * static_cast<test_T>(col + 1);
    }

    test_T expected_upper(rocsparse_int row, rocsparse_int col)
    {
        return static_cast<test_T>(bsrsm_mb - row / bsrsm_block_dim) * static_cast<test_T>(col + 1);
    }

    // BSR structure of A. `lower` selects the sub- or super-diagonal block. Column
    // indices stay ascending within a row, which is what both kernels expect: the
    // lower kernel walks the row forwards and stops at the diagonal, the upper
    // kernel walks it backwards and stops at the diagonal.
    void build_matrix(bool                        lower,
                      std::vector<rocsparse_int>& row_ptr,
                      std::vector<rocsparse_int>& col_ind,
                      std::vector<test_T>&        val,
                      std::vector<rocsparse_int>& map)
    {
        row_ptr.assign(1, 0);
        col_ind.clear();
        val.clear();

        for(rocsparse_int i = 0; i < bsrsm_mb; ++i)
        {
            const rocsparse_int off = lower ? (i - 1) : (i + 1);

            if(lower && off >= 0)
            {
                col_ind.push_back(off);
            }
            col_ind.push_back(i);
            if(!lower && off < bsrsm_mb)
            {
                col_ind.push_back(off);
            }

            row_ptr.push_back(static_cast<rocsparse_int>(col_ind.size()));
        }

        // Block values, row-major within each block (rocsparse_direction_row):
        // the diagonal block is the identity, the off-diagonal block is -I.
        val.assign(static_cast<size_t>(col_ind.size()) * bsrsm_block_dim * bsrsm_block_dim,
                   static_cast<test_T>(0));

        for(rocsparse_int i = 0; i < bsrsm_mb; ++i)
        {
            for(rocsparse_int j = row_ptr[i]; j < row_ptr[i + 1]; ++j)
            {
                const test_T diag
                    = (col_ind[j] == i) ? static_cast<test_T>(1) : static_cast<test_T>(-1);

                for(rocsparse_int b = 0; b < bsrsm_block_dim; ++b)
                {
                    val[static_cast<size_t>(j) * bsrsm_block_dim * bsrsm_block_dim
                        + b * bsrsm_block_dim + b]
                        = diag;
                }
            }
        }

        // Topological order of the dependency graph, which is what the analysis
        // phase produces: natural order for a lower triangular matrix, reversed for
        // an upper triangular one. A dependency always has a SMALLER map index than
        // its dependent, which is what makes the spin-wait deadlock-free.
        map.resize(bsrsm_mb);
        for(rocsparse_int i = 0; i < bsrsm_mb; ++i)
        {
            map[i] = lower ? i : (bsrsm_mb - 1 - i);
        }
    }

    // Run one of the two solve kernels on `grid_x` blocks and read X back. `grid_x`
    // is the caller's choice rather than the value bsrsm_solve_grid_size would
    // pick, so the grid-stride loop can be driven with far fewer blocks than the
    // flattened space needs.
    void
        solve_on_grid(bool lower, int64_t grid_x, std::vector<test_T>& result, rocsparse_int& pivot)
    {
        std::vector<rocsparse_int> row_ptr;
        std::vector<rocsparse_int> col_ind;
        std::vector<test_T>        val;
        std::vector<rocsparse_int> map;
        build_matrix(lower, row_ptr, col_ind, val, map);

        // X is row major with ldx = nrhs, the layout bsrsm_solve_template_large
        // hands the kernel after transposing, and holds alpha * B on entry.
        std::vector<test_T> X(static_cast<size_t>(bsrsm_rows) * bsrsm_nrhs);
        for(rocsparse_int r = 0; r < bsrsm_rows; ++r)
        {
            for(rocsparse_int c = 0; c < bsrsm_nrhs; ++c)
            {
                X[static_cast<size_t>(r) * bsrsm_nrhs + c] = static_cast<test_T>(c + 1);
            }
        }

        rocsparse_ut::device_vector<rocsparse_int> d_row_ptr(row_ptr);
        rocsparse_ut::device_vector<rocsparse_int> d_col_ind(col_ind);
        rocsparse_ut::device_vector<test_T>        d_val(val);
        rocsparse_ut::device_vector<rocsparse_int> d_map(map);
        rocsparse_ut::device_vector<test_T>        d_X(X);
        rocsparse_ut::device_vector<int>           d_done(
            std::vector<int>(static_cast<size_t>(bsrsm_blocks), 0));
        rocsparse_ut::device_vector<rocsparse_int> d_pivot(
            std::vector<rocsparse_int>{std::numeric_limits<rocsparse_int>::max()});

        ASSERT_NE(d_row_ptr.ptr, nullptr);
        ASSERT_NE(d_col_ind.ptr, nullptr);
        ASSERT_NE(d_val.ptr, nullptr);
        ASSERT_NE(d_map.ptr, nullptr);
        ASSERT_NE(d_X.ptr, nullptr);
        ASSERT_NE(d_done.ptr, nullptr);
        ASSERT_NE(d_pivot.ptr, nullptr);

        if(lower)
        {
            hipLaunchKernelGGL(
                (rocsparse::bsrsm_lower_large_kernel<bsrsm_blocksize, bsrsm_ncols, false, test_T>),
                dim3(static_cast<uint32_t>(grid_x)),
                dim3(bsrsm_blocksize),
                0,
                0,
                bsrsm_mb,
                bsrsm_nrhs,
                d_row_ptr.ptr,
                d_col_ind.ptr,
                d_val.ptr,
                bsrsm_block_dim,
                d_X.ptr,
                bsrsm_nrhs,
                d_done.ptr,
                d_map.ptr,
                d_pivot.ptr,
                rocsparse_index_base_zero,
                rocsparse_diag_type_non_unit,
                rocsparse_direction_row);
        }
        else
        {
            hipLaunchKernelGGL(
                (rocsparse::bsrsm_upper_large_kernel<bsrsm_blocksize, bsrsm_ncols, false, test_T>),
                dim3(static_cast<uint32_t>(grid_x)),
                dim3(bsrsm_blocksize),
                0,
                0,
                bsrsm_mb,
                bsrsm_nrhs,
                d_row_ptr.ptr,
                d_col_ind.ptr,
                d_val.ptr,
                bsrsm_block_dim,
                d_X.ptr,
                bsrsm_nrhs,
                d_done.ptr,
                d_map.ptr,
                d_pivot.ptr,
                rocsparse_index_base_zero,
                rocsparse_diag_type_non_unit,
                rocsparse_direction_row);
        }

        ASSERT_EQ(hipGetLastError(), hipSuccess);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        result = rocsparse_ut::to_host(d_X);
        pivot  = rocsparse_ut::to_host(d_pivot)[0];
    }

    // Compare against the closed form solution and report the FIRST offending
    // (row, column) pair, so a failure names the coordinate that was skipped or
    // mispaired instead of drowning the log in 1600 expectations.
    void expect_exact_solution(bool lower, const std::vector<test_T>& got, int64_t grid_x)
    {
        ASSERT_EQ(static_cast<int64_t>(got.size()), static_cast<int64_t>(bsrsm_rows) * bsrsm_nrhs);

        for(rocsparse_int r = 0; r < bsrsm_rows; ++r)
        {
            for(rocsparse_int c = 0; c < bsrsm_nrhs; ++c)
            {
                const test_T value    = got[static_cast<size_t>(r) * bsrsm_nrhs + c];
                const test_T expected = lower ? expected_lower(r, c) : expected_upper(r, c);

                ASSERT_EQ(value, expected)
                    << "wrong solution at row " << r << ", rhs " << c << " ("
                    << (lower ? "lower" : "upper") << ", grid.x=" << grid_x
                    << ", flattened blocks=" << bsrsm_blocks << ", panels=" << bsrsm_npanels
                    << "). A grid smaller than the flattened space must be covered by the "
                       "kernel's grid-stride loop.";
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host-pure grid sizing (bsrsm_device_large.h).
// ---------------------------------------------------------------------------

TEST(internal_level3_bsrsm, num_blocks_is_computed_in_64_bit)
{
    EXPECT_EQ(rocsparse::bsrsm_num_blocks(0, 8, 16), 0);
    EXPECT_EQ(rocsparse::bsrsm_num_blocks(10, 1, 16), 10);
    EXPECT_EQ(rocsparse::bsrsm_num_blocks(10, 16, 16), 10);
    EXPECT_EQ(rocsparse::bsrsm_num_blocks(10, 17, 16), 20);

    // The configuration named in AISPARSE-670: 50,000 block rows and 700,000 right
    // hand sides give ((700000 - 1) / 16 + 1) * 50000 = 2,187,500,000 blocks, which
    // does not fit in the signed int32 the original expression evaluated it in.
    EXPECT_EQ(rocsparse::bsrsm_num_blocks(50000, 700000, 16), int64_t{2187500000});
    EXPECT_GT(rocsparse::bsrsm_num_blocks(50000, 700000, 16),
              int64_t{std::numeric_limits<int32_t>::max()});
}

TEST(internal_level3_bsrsm, solve_grid_size_clamps_to_whole_panels)
{
    constexpr int64_t max_grid_x = 2147483647;

    // Below the limit the grid is the full flattened space.
    EXPECT_EQ(rocsparse::bsrsm_solve_grid_size(1000, 16 * 8, 16, max_grid_x), 8000);

    // Above the limit it is clamped, and rounded DOWN to a whole number of RHS
    // panels so the kernels' stride stays a multiple of mb.
    EXPECT_EQ(rocsparse::bsrsm_solve_grid_size(1000, 16 * 8, 16, 2500), 2000);

    // mb alone above the limit: return mb so the launch fails loudly with
    // hipErrorInvalidConfiguration rather than running a grid the kernels cannot
    // stride.
    EXPECT_EQ(rocsparse::bsrsm_solve_grid_size(1000, 16 * 8, 16, 999), 1000);

    // The overflowing configuration: 2,187,500,000 blocks clamped to the grid
    // limit, then trimmed to 42,949 whole panels of 50,000 block rows.
    EXPECT_EQ(rocsparse::bsrsm_solve_grid_size(50000, 700000, 16, max_grid_x),
              int64_t{42949} * 50000);
    EXPECT_LE(rocsparse::bsrsm_solve_grid_size(50000, 700000, 16, max_grid_x), max_grid_x);

    // Degenerate mb is not a division by zero.
    EXPECT_EQ(rocsparse::bsrsm_solve_grid_size(0, 8, 16, max_grid_x), 0);
}

TEST(internal_level3_bsrsm, solve_grid_size_is_always_a_multiple_of_mb)
{
    // The kernels derive their stride as (gridDim.x / mb) * mb. If grid.x were not
    // a whole number of panels the stride would no longer pin a block to a single
    // row-map slot, and a block could wait on a done_array flag owned by a block
    // that has not been dispatched yet.
    for(int64_t mb = 1; mb <= 37; ++mb)
    {
        for(int64_t max_grid_x = 1; max_grid_x <= 200; max_grid_x += 7)
        {
            const int64_t grid = rocsparse::bsrsm_solve_grid_size(mb, 16 * 5, 16, max_grid_x);
            EXPECT_EQ(grid % mb, 0) << "mb=" << mb << ", max_grid_x=" << max_grid_x;
            EXPECT_GE(grid, mb) << "mb=" << mb << ", max_grid_x=" << max_grid_x;
            EXPECT_LE(grid, rocsparse::bsrsm_num_blocks(mb, 16 * 5, 16));
        }
    }
}

// ---------------------------------------------------------------------------
// Defect 1, end-to-end: the flattened solve grid-stride loop on the GPU.
// ---------------------------------------------------------------------------

// Control: the grid covers the whole flattened space, so the stride loop runs
// exactly one iteration. This is the pre-existing behaviour and must not change.
TEST(internal_level3_bsrsm, solve_lower_full_grid)
{
    std::vector<test_T> got;
    rocsparse_int       pivot = 0;
    ASSERT_NO_FATAL_FAILURE(solve_on_grid(true, bsrsm_blocks, got, pivot));
    EXPECT_EQ(pivot, std::numeric_limits<rocsparse_int>::max());
    expect_exact_solution(true, got, bsrsm_blocks);
}

// Two RHS panels per sweep: 56 flattened blocks covered by 16, so four sweeps.
TEST(internal_level3_bsrsm, solve_lower_undersized_grid_two_panels)
{
    const int64_t grid_x = 2 * bsrsm_mb;
    ASSERT_LT(grid_x, bsrsm_blocks);

    std::vector<test_T> got;
    rocsparse_int       pivot = 0;
    ASSERT_NO_FATAL_FAILURE(solve_on_grid(true, grid_x, got, pivot));
    EXPECT_EQ(pivot, std::numeric_limits<rocsparse_int>::max());
    expect_exact_solution(true, got, grid_x);
}

// The tightest grid the clamp can produce: a single RHS panel, so all seven panels
// are reached only by striding. Also the case where gridDim.x == mb exactly.
TEST(internal_level3_bsrsm, solve_lower_undersized_grid_single_panel)
{
    const int64_t grid_x = bsrsm_mb;
    ASSERT_LT(grid_x, bsrsm_blocks);

    std::vector<test_T> got;
    rocsparse_int       pivot = 0;
    ASSERT_NO_FATAL_FAILURE(solve_on_grid(true, grid_x, got, pivot));
    EXPECT_EQ(pivot, std::numeric_limits<rocsparse_int>::max());
    expect_exact_solution(true, got, grid_x);
}

// A grid that is NOT a whole number of RHS panels. bsrsm_solve_grid_size never
// produces one, but the kernels are launchable directly (this file does it), so
// the contract has to hold for any grid.x, and the failure mode is silent: the
// stride is rounded down to whole panels, so without a guard the ragged blocks
// past the last whole panel re-run (panel, block row) pairs that a lower numbered
// block already owns. The block body is NOT idempotent -- it reads its own X
// elements as the right hand side and overwrites them -- so a second visit
// corrupts the result.
TEST(internal_level3_bsrsm, solve_lower_ragged_grid_is_not_a_multiple_of_mb)
{
    const int64_t grid_x = 2 * bsrsm_mb + 3;
    ASSERT_LT(grid_x, bsrsm_blocks);
    ASSERT_NE(grid_x % bsrsm_mb, 0);

    std::vector<test_T> got;
    rocsparse_int       pivot = 0;
    ASSERT_NO_FATAL_FAILURE(solve_on_grid(true, grid_x, got, pivot));
    EXPECT_EQ(pivot, std::numeric_limits<rocsparse_int>::max());
    expect_exact_solution(true, got, grid_x);
}

// The upper kernel carries the same flattened grid and the same reversed row map,
// so it gets the control plus the tightest undersized grid.
TEST(internal_level3_bsrsm, solve_upper_full_grid)
{
    std::vector<test_T> got;
    rocsparse_int       pivot = 0;
    ASSERT_NO_FATAL_FAILURE(solve_on_grid(false, bsrsm_blocks, got, pivot));
    EXPECT_EQ(pivot, std::numeric_limits<rocsparse_int>::max());
    expect_exact_solution(false, got, bsrsm_blocks);
}

TEST(internal_level3_bsrsm, solve_upper_undersized_grid_single_panel)
{
    const int64_t grid_x = bsrsm_mb;
    ASSERT_LT(grid_x, bsrsm_blocks);

    std::vector<test_T> got;
    rocsparse_int       pivot = 0;
    ASSERT_NO_FATAL_FAILURE(solve_on_grid(false, grid_x, got, pivot));
    EXPECT_EQ(pivot, std::numeric_limits<rocsparse_int>::max());
    expect_exact_solution(false, got, grid_x);
}

TEST(internal_level3_bsrsm, solve_upper_undersized_grid_two_panels)
{
    const int64_t grid_x = 2 * bsrsm_mb;
    ASSERT_LT(grid_x, bsrsm_blocks);

    std::vector<test_T> got;
    rocsparse_int       pivot = 0;
    ASSERT_NO_FATAL_FAILURE(solve_on_grid(false, grid_x, got, pivot));
    EXPECT_EQ(pivot, std::numeric_limits<rocsparse_int>::max());
    expect_exact_solution(false, got, grid_x);
}

// ---------------------------------------------------------------------------
// Defect 2: the transpose gather (rocsparse::bsr_gather, shared with bsrsv).
// ---------------------------------------------------------------------------

namespace
{
    constexpr uint32_t      gather_wfsize    = 4;
    constexpr uint32_t      gather_dimy      = 64;
    constexpr rocsparse_int gather_block_dim = 2;
    constexpr rocsparse_int gather_nnzb      = 500;

    constexpr int64_t gather_full_grid = (gather_nnzb - 1) / gather_dimy + 1;

    // bsr_gather writes bsr_val_T[BSR_IND(j, bi, bj)] = bsr_val_A[BSR_IND(p, bj, bi)]
    // with BSR_IND_R(j, bi, bj) = block_dim * block_dim * j + bi * block_dim + bj,
    // i.e. it transposes each block while permuting the block order.
    void gather_on_grid(int64_t grid_x, std::vector<test_T>& result)
    {
        constexpr size_t block_elems = static_cast<size_t>(gather_block_dim) * gather_block_dim;

        std::vector<rocsparse_int> perm(gather_nnzb);
        std::vector<test_T>        A(static_cast<size_t>(gather_nnzb) * block_elems);
        for(rocsparse_int j = 0; j < gather_nnzb; ++j)
        {
            perm[j] = gather_nnzb - 1 - j;
            for(size_t q = 0; q < block_elems; ++q)
            {
                A[static_cast<size_t>(j) * block_elems + q]
                    = static_cast<test_T>(j * block_elems + q);
            }
        }

        rocsparse_ut::device_vector<rocsparse_int> d_perm(perm);
        rocsparse_ut::device_vector<test_T>        d_A(A);
        rocsparse_ut::device_vector<test_T>        d_T(
            std::vector<test_T>(A.size(), static_cast<test_T>(-1)));

        ASSERT_NE(d_perm.ptr, nullptr);
        ASSERT_NE(d_A.ptr, nullptr);
        ASSERT_NE(d_T.ptr, nullptr);

        hipLaunchKernelGGL(
            (rocsparse::
                 bsr_gather<gather_wfsize, gather_dimy, gather_block_dim, rocsparse_int, test_T>),
            dim3(static_cast<uint32_t>(grid_x)),
            dim3(gather_wfsize, gather_dimy),
            0,
            0,
            rocsparse_direction_row,
            gather_nnzb,
            d_perm.ptr,
            d_A.ptr,
            d_T.ptr,
            gather_block_dim);

        ASSERT_EQ(hipGetLastError(), hipSuccess);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        result = rocsparse_ut::to_host(d_T);
    }

    void expect_exact_gather(const std::vector<test_T>& got, int64_t grid_x)
    {
        constexpr rocsparse_int block_dim = gather_block_dim;

        for(rocsparse_int j = 0; j < gather_nnzb; ++j)
        {
            const rocsparse_int p = gather_nnzb - 1 - j;

            for(rocsparse_int bi = 0; bi < gather_block_dim; ++bi)
            {
                for(rocsparse_int bj = 0; bj < gather_block_dim; ++bj)
                {
                    const size_t dst = static_cast<size_t>(BSR_IND_R(j, bi, bj));
                    const size_t src = static_cast<size_t>(BSR_IND_R(p, bj, bi));

                    ASSERT_EQ(got[dst], static_cast<test_T>(src))
                        << "wrong gather at nnz " << j << ", block entry (" << bi << ", " << bj
                        << ") with grid.x=" << grid_x << " of " << gather_full_grid
                        << ". A clamped grid must be covered by the kernel's grid-stride loop.";
                }
            }
        }
    }
}

TEST(internal_level3_bsrsm, gather_full_grid)
{
    std::vector<test_T> got;
    ASSERT_NO_FATAL_FAILURE(gather_on_grid(gather_full_grid, got));
    expect_exact_gather(got, gather_full_grid);
}

TEST(internal_level3_bsrsm, gather_undersized_grid)
{
    constexpr int64_t grid_x = 2;
    static_assert(grid_x < gather_full_grid, "grid must be undersized");

    std::vector<test_T> got;
    ASSERT_NO_FATAL_FAILURE(gather_on_grid(grid_x, got));
    expect_exact_gather(got, grid_x);
}

TEST(internal_level3_bsrsm, gather_single_block_grid)
{
    std::vector<test_T> got;
    ASSERT_NO_FATAL_FAILURE(gather_on_grid(1, got));
    expect_exact_gather(got, 1);
}

// ---------------------------------------------------------------------------
// Defect 3: the right hand side setup (rocsparse::bsrsm_copy_scale).
// ---------------------------------------------------------------------------

namespace
{
    constexpr uint32_t      copy_blocksize = 1024; // the production instantiation
    constexpr int64_t       copy_rows      = 5000; // mb * block_dim
    constexpr rocsparse_int copy_ncols     = 3;

    constexpr int64_t copy_full_grid = (copy_rows - 1) / copy_blocksize + 1;

    void copy_scale_on_grid(int64_t grid_x, std::vector<test_T>& result)
    {
        const test_T alpha = static_cast<test_T>(2);

        std::vector<test_T> B(static_cast<size_t>(copy_rows) * copy_ncols);
        for(size_t i = 0; i < B.size(); ++i)
        {
            B[i] = static_cast<test_T>(i + 1);
        }

        rocsparse_ut::device_vector<test_T> d_B(B);
        rocsparse_ut::device_vector<test_T> d_X(
            std::vector<test_T>(B.size(), static_cast<test_T>(-1)));

        ASSERT_NE(d_B.ptr, nullptr);
        ASSERT_NE(d_X.ptr, nullptr);

        hipLaunchKernelGGL(
            (rocsparse::bsrsm_copy_scale<copy_blocksize, test_T>),
            dim3(static_cast<uint32_t>(grid_x)),
            dim3(copy_blocksize),
            0,
            0,
            copy_rows,
            copy_ncols,
            rocsparse::to_const_host_device_scalar(rocsparse_pointer_mode_host, &alpha),
            d_B.ptr,
            static_cast<int64_t>(copy_ncols),
            d_X.ptr,
            static_cast<int64_t>(copy_ncols),
            true);

        ASSERT_EQ(hipGetLastError(), hipSuccess);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        result = rocsparse_ut::to_host(d_X);
    }

    void expect_exact_copy(const std::vector<test_T>& got, int64_t grid_x)
    {
        ASSERT_EQ(static_cast<int64_t>(got.size()), copy_rows * copy_ncols);

        for(size_t i = 0; i < got.size(); ++i)
        {
            ASSERT_EQ(got[i], static_cast<test_T>(2 * (i + 1)))
                << "wrong copy at element " << i << " (row " << i / copy_ncols
                << ") with grid.x=" << grid_x << " of " << copy_full_grid
                << ". A clamped grid must be covered by the kernel's grid-stride loop.";
        }
    }
}

TEST(internal_level3_bsrsm, copy_scale_full_grid)
{
    std::vector<test_T> got;
    ASSERT_NO_FATAL_FAILURE(copy_scale_on_grid(copy_full_grid, got));
    expect_exact_copy(got, copy_full_grid);
}

TEST(internal_level3_bsrsm, copy_scale_undersized_grid)
{
    constexpr int64_t grid_x = 2;
    static_assert(grid_x < copy_full_grid, "grid must be undersized");

    std::vector<test_T> got;
    ASSERT_NO_FATAL_FAILURE(copy_scale_on_grid(grid_x, got));
    expect_exact_copy(got, grid_x);
}

TEST(internal_level3_bsrsm, copy_scale_single_block_grid)
{
    std::vector<test_T> got;
    ASSERT_NO_FATAL_FAILURE(copy_scale_on_grid(1, got));
    expect_exact_copy(got, 1);
}

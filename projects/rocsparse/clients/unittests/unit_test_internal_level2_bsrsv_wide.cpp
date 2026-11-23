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

// Wide bsrsv solve kernels (bsrsv_device_large.h), launched directly on the
// block bidiagonal system of unit_test_internal_level2_bsrsv_test_matrix.hpp.
//
// The launches pass an undersized grid, which is the case the host creates
// when get_grid_size_x clamps, so they cover the chunk walk of the wide
// kernels. They instantiate (i32, i32), (i64, i32) and (i64, i64), and use the
// device wavefront, so the shared kernels run on wave32 as well. The host path
// at the 64-bit index types runs in L2DeviceBsrsv.wide_index_types, and the
// analysis reuse paths in unit_test_internal_level2_bsrsv_analysis_reuse.cpp.

#include "unit_test_internal_level2_bsrsv_test_matrix.hpp"

#include "bsrsv_device_large.h"

#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>
#include <vector>

namespace
{
    using namespace bsrsv_test;

    constexpr uint32_t kBlock = 128;

    template <uint32_t WFSIZE, typename I, typename J>
    void launch_wide(const test_matrix<I, J>& a,
                     uint32_t                 grid_x,
                     rocsparse_diag_type      diag,
                     bool                     shared,
                     std::vector<test_T>&     y_out,
                     J&                       pivot)
    {
        const std::vector<test_T> x = make_x();
        const std::vector<test_T> y(static_cast<size_t>(kRows), static_cast<test_T>(-7));

        rocsparse_ut::device_vector<I>      d_row_ptr(a.row_ptr);
        rocsparse_ut::device_vector<J>      d_col_ind(a.col_ind);
        rocsparse_ut::device_vector<test_T> d_val(a.val);
        rocsparse_ut::device_vector<J>      d_map(a.map);
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

        const auto alpha_union = rocsparse::const_host_device_scalar<test_T>(kAlpha);

        // BSRDIM 8 is the specialization the host uses for block_dim <= 8, and
        // it divides both wavefront sizes.
        if(shared && a.lower)
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
                a.base,
                diag,
                a.dir,
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
                a.base,
                diag,
                a.dir,
                true);
        }
        else if(a.lower)
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
                a.base,
                diag,
                a.dir,
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
                a.base,
                diag,
                a.dir,
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
    template <typename F>
    void with_device_wavefront(F&& f)
    {
        const int ws = rocsparse_ut::device_warp_size();
        ASSERT_TRUE(ws == 32 || ws == 64) << "unsupported device wavefront size: " << ws;
        if(ws == 64)
        {
            f(std::integral_constant<uint32_t, 64>{});
        }
        else
        {
            f(std::integral_constant<uint32_t, 32>{});
        }
    }

    // Both kernel families and both diagonal types against the host solve.
    template <typename I, typename J>
    void expect_wide(bool lower, uint32_t grid_x)
    {
        const test_matrix<I, J> a = build_matrix<I, J>(lower);
        for(const bool shared : {false, true})
        {
            for(const auto diag : {rocsparse_diag_type_non_unit, rocsparse_diag_type_unit})
            {
                SCOPED_TRACE(::testing::Message()
                             << (lower ? "lower" : "upper") << (shared ? " shared" : " general")
                             << " diag " << diag << " grid " << grid_x);
                std::vector<test_T> y;
                J                   pivot{};
                with_device_wavefront([&](auto wf) {
                    ASSERT_NO_FATAL_FAILURE((
                        launch_wide<decltype(wf)::value, I, J>(a, grid_x, diag, shared, y, pivot)));
                });
                if(::testing::Test::HasFatalFailure())
                {
                    return;
                }
                EXPECT_EQ(pivot, std::numeric_limits<J>::max());
                expect_solution(y, reference(a, lower, rocsparse_operation_none, diag));
            }
        }
    }

    template <typename I, typename J>
    void expect_singular_pivot(bool                 lower,
                               rocsparse_index_base base,
                               int32_t              singular_row,
                               uint32_t             grid_x,
                               bool                 shared = false)
    {
        const test_matrix<I, J> a = build_matrix<I, J>(lower, base, singular_row);
        std::vector<test_T>     y;
        J                       pivot{};
        with_device_wavefront([&](auto wf) {
            ASSERT_NO_FATAL_FAILURE((launch_wide<decltype(wf)::value, I, J>(
                a, grid_x, rocsparse_diag_type_non_unit, shared, y, pivot)));
        });
        // The kernel reports the block row in the caller's index base.
        EXPECT_EQ(pivot, static_cast<J>(singular_row + static_cast<int32_t>(base)));
    }

    void expect_wide_direction(bool lower, bool shared, rocsparse_direction dir)
    {
        SCOPED_TRACE(::testing::Message()
                     << (lower ? "lower" : "upper") << (shared ? " shared" : " general")
                     << (dir == rocsparse_direction_row ? " row" : " column"));
        const auto a
            = build_matrix<int64_t, int64_t>(lower, rocsparse_index_base_zero, -1, kOffAsym, dir);
        std::vector<test_T> y;
        int64_t             pivot{};
        with_device_wavefront([&](auto wf) {
            ASSERT_NO_FATAL_FAILURE((launch_wide<decltype(wf)::value, int64_t, int64_t>(
                a, 1, rocsparse_diag_type_non_unit, shared, y, pivot)));
        });
        EXPECT_EQ(pivot, std::numeric_limits<int64_t>::max());
        expect_solution(
            y, reference(a, lower, rocsparse_operation_none, rocsparse_diag_type_non_unit));
    }
}

TEST(internal_level2_bsrsv_wide, solve_i32_chunk)
{
    expect_wide<int32_t, int32_t>(true, 1);
    expect_wide<int32_t, int32_t>(false, 1);
}

TEST(internal_level2_bsrsv_wide, solve_i32_full_grid)
{
    // Block 128 and 8 block rows: 2 virtual blocks on wave32, 4 on wave64.
    // A grid of 2 is exact on wave32 and a two-chunk walk on wave64.
    expect_wide<int32_t, int32_t>(true, 2);
}

TEST(internal_level2_bsrsv_wide, solve_i64_i32_chunk)
{
    expect_wide<int64_t, int32_t>(true, 1);
    expect_wide<int64_t, int32_t>(false, 1);
}

TEST(internal_level2_bsrsv_wide, solve_i64_chunk)
{
    expect_wide<int64_t, int64_t>(true, 1);
    expect_wide<int64_t, int64_t>(false, 1);
}

TEST(internal_level2_bsrsv_wide, direction_nonsymmetric_blocks)
{
    // A kernel that swapped or ignored dir would solve with the transposed
    // off-diagonal block, which gives a different solution.
    for(const bool lower : {true, false})
    {
        const auto asym
            = build_matrix<int64_t, int64_t>(lower, rocsparse_index_base_zero, -1, kOffAsym);
        const auto asym_t
            = build_matrix<int64_t, int64_t>(lower, rocsparse_index_base_zero, -1, kOffAsymT);
        ASSERT_NE(reference(asym, lower, rocsparse_operation_none, rocsparse_diag_type_non_unit),
                  reference(asym_t, lower, rocsparse_operation_none, rocsparse_diag_type_non_unit));
        for(const bool shared : {false, true})
        {
            for(const rocsparse_direction dir :
                {rocsparse_direction_row, rocsparse_direction_column})
            {
                expect_wide_direction(lower, shared, dir);
            }
        }
    }
}

TEST(internal_level2_bsrsv_wide, zero_pivot_reports_block_row)
{
    // Block row 3 has a zero diagonal block. The reported pivot is the block
    // row plus the index base, so base one must report 4.
    expect_singular_pivot<int32_t, int32_t>(true, rocsparse_index_base_zero, 3, 1);
    expect_singular_pivot<int64_t, int32_t>(true, rocsparse_index_base_zero, 5, 1);
    expect_singular_pivot<int64_t, int64_t>(true, rocsparse_index_base_one, 3, 1);
    expect_singular_pivot<int64_t, int64_t>(true, rocsparse_index_base_one, 6, 2);

    // Upper general has its own pivot flag. Base one checks the offset there too.
    expect_singular_pivot<int32_t, int32_t>(false, rocsparse_index_base_one, 3, 1);
    expect_singular_pivot<int64_t, int64_t>(false, rocsparse_index_base_one, 5, 1);
}

TEST(internal_level2_bsrsv_wide, zero_pivot_shared_reports_block_row)
{
    // The host launches the shared kernels only on wave64. These direct launches
    // use the device wavefront, so the shared pivot path runs on wave32 as well.
    expect_singular_pivot<int32_t, int32_t>(true, rocsparse_index_base_one, 3, 1, true);
    expect_singular_pivot<int64_t, int32_t>(false, rocsparse_index_base_zero, 4, 1, true);
    expect_singular_pivot<int64_t, int64_t>(true, rocsparse_index_base_one, 6, 2, true);
    expect_singular_pivot<int64_t, int64_t>(false, rocsparse_index_base_one, 2, 1, true);
}

TEST(internal_level2_bsrsv_wide, needs_wide_predicate_boundaries)
{
    const int64_t i32max = std::numeric_limits<int32_t>::max();

    // Small sizes stay narrow. The launch grid does not decide: the narrow
    // kernels walk a clamped grid in chunks.
    EXPECT_FALSE(rocsparse::bsrsv_solve_needs_wide(8, 2));
    // Empty problems never need the wide path.
    EXPECT_FALSE(rocsparse::bsrsv_solve_needs_wide(0, 2));

    // mb at and past INT32_MAX.
    EXPECT_FALSE(rocsparse::bsrsv_solve_needs_wide(i32max, 1));
    EXPECT_TRUE(rocsparse::bsrsv_solve_needs_wide(i32max + 1, 1));

    // mb * block_dim past INT32_MAX stays narrow: those kernels form the
    // vector offsets in 64 bits.
    EXPECT_FALSE(rocsparse::bsrsv_solve_needs_wide(1048576, 2047));
    EXPECT_FALSE(rocsparse::bsrsv_solve_needs_wide(1048576, 2048));

    // block_dim * block_dim alone: 46340^2 fits, 46341^2 does not.
    EXPECT_FALSE(rocsparse::bsrsv_solve_needs_wide(1, 46340));
    EXPECT_TRUE(rocsparse::bsrsv_solve_needs_wide(1, 46341));
}

namespace
{
    // The argument types are the ones the general wide kernels pass: the
    // block index has type I, the in-block indices are int and block_dim is
    // rocsparse_int, so the test does not widen anything itself.
    __global__ void bsrsv_wide_ind_kernel(
        int32_t j, int bi, int bj, rocsparse_int block_dim, rocsparse_direction dir, int64_t* out)
    {
        out[0] = rocsparse::bsrsv_wide_ind(j, bi, bj, block_dim, dir);
    }
}

// From block_dim 46341, which bsrsv_solve_needs_wide sends to the general wide
// kernels, bi * block_dim alone passes INT32_MAX, so the in-block offset must
// be formed in 64 bits on the host and on the device.
TEST(internal_level2_bsrsv_wide, wide_ind_forms_in_block_offset_in_64_bit)
{
    struct offset_case
    {
        int32_t             j;
        int                 bi;
        int                 bj;
        rocsparse_int       block_dim;
        rocsparse_direction dir;
        int64_t             want;
    };
    const offset_case cases[] = {
        // Row 46341 of block 1 at block_dim 46342, which is 148290 in 32 bits.
        {1, 46341, 0, 46342, rocsparse_direction_row, 4295115586},
        {1, 0, 46341, 46342, rocsparse_direction_column, 4295115586},
        // The last diagonal entry of block 0 at the first wide block_dim.
        {0, 46340, 46340, 46341, rocsparse_direction_row, 2147488280},
        {0, 46340, 46340, 46341, rocsparse_direction_column, 2147488280},
        // One strict-triangle entry in both storage directions.
        {2, 3, 46340, 46341, rocsparse_direction_row, 4295161925},
        {2, 3, 46340, 46341, rocsparse_direction_column, 6442418505},
    };

    for(const offset_case& c : cases)
    {
        SCOPED_TRACE(::testing::Message() << "j " << c.j << " bi " << c.bi << " bj " << c.bj
                                          << " block_dim " << c.block_dim << " dir " << c.dir);
        EXPECT_EQ(rocsparse::bsrsv_wide_ind(c.j, c.bi, c.bj, c.block_dim, c.dir), c.want);

        rocsparse_ut::device_vector<int64_t> d_out(std::vector<int64_t>(1, -1));
        ASSERT_NE(d_out.ptr, nullptr);
        ASSERT_EQ(rocsparse_ut::launch_single_block(
                      bsrsv_wide_ind_kernel, 1, c.j, c.bi, c.bj, c.block_dim, c.dir, d_out.ptr),
                  hipSuccess);
        EXPECT_EQ(rocsparse_ut::to_host(d_out)[0], c.want);
    }
}

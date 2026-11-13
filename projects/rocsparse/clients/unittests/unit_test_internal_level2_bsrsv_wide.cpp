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

// Wide bsrsv solve kernels (bsrsv_device_large.h) and the bsrsv analysis reuse
// paths they depend on.
//
// The matrix is an 8-by-8 block bidiagonal system of 2-by-2 blocks. The
// diagonal blocks are triangular with diagonal 2 and 4 and an in-block entry
// of 1, and hold 37 in the triangle the solve must not read. With alpha = -2
// and a varying x, the diagonal division, the in-block elimination, alpha and
// the diagonal type all change the result, which a host solve in double
// checks. L2DeviceBsrsv.clamped_grid_wide_kernels runs the wider sweep of
// unit_test_device_level2_bsrsv.cpp through the wide kernels.
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
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

namespace
{
    using test_T = float;

    constexpr int32_t  kMb       = 8;
    constexpr int32_t  kBlockDim = 2;
    constexpr int32_t  kRows     = kMb * kBlockDim;
    constexpr uint32_t kBlock    = 128;
    constexpr test_T   kAlpha    = -2;

    // Logical 2-by-2 blocks, stored in the order dir gives.
    using block2              = test_T[2][2];
    constexpr block2 kOffDiag = {{-1, 0}, {0, -1}};
    // kOffAsym is not symmetric, so a kernel that reads it with the wrong
    // direction solves with kOffAsymT instead.
    constexpr block2 kOffAsym  = {{-1, 0}, {1, -1}};
    constexpr block2 kOffAsymT = {{-1, 1}, {0, -1}};

    void append_block(std::vector<test_T>& val, const block2& b, rocsparse_direction dir)
    {
        for(int32_t i = 0; i < kBlockDim; ++i)
        {
            for(int32_t j = 0; j < kBlockDim; ++j)
            {
                val.push_back(dir == rocsparse_direction_row ? b[i][j] : b[j][i]);
            }
        }
    }

    // Diagonal block of a lower or upper matrix. A singular block has a zero
    // diagonal, which the kernels report as a zero pivot.
    void append_diag_block(std::vector<test_T>& val,
                           bool                 lower,
                           bool                 singular,
                           rocsparse_direction  dir)
    {
        const test_T d0 = singular ? 0 : 2;
        const test_T d1 = singular ? 0 : 4;
        const block2 lower_block{{d0, 37}, {1, d1}};
        const block2 upper_block{{d0, 1}, {37, d1}};
        append_block(val, lower ? lower_block : upper_block, dir);
    }

    template <typename I, typename J>
    struct test_matrix
    {
        bool                 lower = true;
        rocsparse_index_base base  = rocsparse_index_base_zero;
        rocsparse_direction  dir   = rocsparse_direction_row;
        std::vector<I>       row_ptr;
        std::vector<J>       col_ind;
        std::vector<test_T>  val;
        // Block rows in solve order, as the analysis builds it.
        std::vector<J> map;

        I nnzb() const
        {
            return static_cast<I>(col_ind.size());
        }
    };

    template <typename I, typename J>
    test_matrix<I, J> build_matrix(bool                 lower,
                                   rocsparse_index_base base         = rocsparse_index_base_zero,
                                   int32_t              singular_row = -1,
                                   const block2&        off          = kOffDiag,
                                   rocsparse_direction  dir          = rocsparse_direction_row)
    {
        test_matrix<I, J> a;
        a.lower = lower;
        a.base  = base;
        a.dir   = dir;
        a.row_ptr.assign(1, static_cast<I>(0));
        a.map.resize(static_cast<size_t>(kMb));

        for(int32_t row = 0; row < kMb; ++row)
        {
            if(lower && row > 0)
            {
                a.col_ind.push_back(static_cast<J>(row - 1));
                append_block(a.val, off, dir);
            }
            a.col_ind.push_back(static_cast<J>(row));
            append_diag_block(a.val, lower, row == singular_row, dir);
            if(!lower && row + 1 < kMb)
            {
                a.col_ind.push_back(static_cast<J>(row + 1));
                append_block(a.val, off, dir);
            }
            a.map[static_cast<size_t>(row)] = static_cast<J>(lower ? row : kMb - 1 - row);
            a.row_ptr.push_back(static_cast<I>(a.col_ind.size()));
        }

        if(base == rocsparse_index_base_one)
        {
            for(auto& r : a.row_ptr)
            {
                r = static_cast<I>(r + 1);
            }
            for(auto& c : a.col_ind)
            {
                c = static_cast<J>(c + 1);
            }
        }
        return a;
    }

    // Drops the diagonal block of block row missing_row from a zero-based matrix.
    // The analysis sees a structural zero there, which the solve does not.
    template <typename I, typename J>
    void remove_diagonal_block(test_matrix<I, J>& a, int32_t missing_row)
    {
        constexpr size_t  blk = static_cast<size_t>(kBlockDim) * kBlockDim;
        test_matrix<I, J> b   = a;
        b.row_ptr.assign(1, static_cast<I>(0));
        b.col_ind.clear();
        b.val.clear();
        for(int32_t row = 0; row < kMb; ++row)
        {
            for(I j = a.row_ptr[row]; j < a.row_ptr[row + 1]; ++j)
            {
                if(row == missing_row && a.col_ind[j] == static_cast<J>(row))
                {
                    continue;
                }
                b.col_ind.push_back(a.col_ind[j]);
                b.val.insert(b.val.end(),
                             a.val.begin() + blk * static_cast<size_t>(j),
                             a.val.begin() + blk * static_cast<size_t>(j + 1));
            }
            b.row_ptr.push_back(static_cast<I>(b.col_ind.size()));
        }
        a = b;
    }

    test_matrix<int64_t, int64_t> widen(const test_matrix<int32_t, int32_t>& a)
    {
        test_matrix<int64_t, int64_t> b;
        b.lower = a.lower;
        b.base  = a.base;
        b.dir   = a.dir;
        b.row_ptr.assign(a.row_ptr.begin(), a.row_ptr.end());
        b.col_ind.assign(a.col_ind.begin(), a.col_ind.end());
        b.val = a.val;
        b.map.assign(a.map.begin(), a.map.end());
        return b;
    }

    std::vector<test_T> make_x()
    {
        std::vector<test_T> x(static_cast<size_t>(kRows));
        for(int32_t r = 0; r < kRows; ++r)
        {
            x[static_cast<size_t>(r)] = static_cast<test_T>(1 + r % 3);
        }
        return x;
    }

    // Host solve of op(A) y = kAlpha x in double, reading only the triangle of A
    // given by lower. The 37s in the other triangle of the diagonal blocks are
    // ignored, as the solve must ignore them.
    template <typename I, typename J>
    std::vector<double> reference(const test_matrix<I, J>& a,
                                  bool                     lower,
                                  rocsparse_operation      trans,
                                  rocsparse_diag_type      diag)
    {
        constexpr size_t          blk  = static_cast<size_t>(kBlockDim) * kBlockDim;
        const std::vector<test_T> x    = make_x();
        const int64_t             base = static_cast<int64_t>(a.base);

        std::vector<double> op(static_cast<size_t>(kRows) * kRows, 0.0);
        for(int64_t br = 0; br < kMb; ++br)
        {
            for(int64_t k = a.row_ptr[br] - base; k < a.row_ptr[br + 1] - base; ++k)
            {
                const int64_t bc = static_cast<int64_t>(a.col_ind[k]) - base;
                for(int64_t i = 0; i < kBlockDim; ++i)
                {
                    for(int64_t j = 0; j < kBlockDim; ++j)
                    {
                        const int64_t r = br * kBlockDim + i;
                        const int64_t c = bc * kBlockDim + j;
                        if(lower ? c > r : c < r)
                        {
                            continue;
                        }
                        const int64_t pos = (a.dir == rocsparse_direction_row) ? i * kBlockDim + j
                                                                               : j * kBlockDim + i;
                        const double  v
                            = a.val[blk * static_cast<size_t>(k) + static_cast<size_t>(pos)];
                        if(trans == rocsparse_operation_none)
                        {
                            op[r * kRows + c] = v;
                        }
                        else
                        {
                            op[c * kRows + r] = v;
                        }
                    }
                }
            }
        }

        const bool          op_lower = (lower == (trans == rocsparse_operation_none));
        std::vector<double> y(static_cast<size_t>(kRows), 0.0);
        for(int32_t s = 0; s < kRows; ++s)
        {
            const int32_t r   = op_lower ? s : kRows - 1 - s;
            double        sum = static_cast<double>(kAlpha) * x[static_cast<size_t>(r)];
            for(int32_t c = 0; c < kRows; ++c)
            {
                if(c != r)
                {
                    sum -= op[static_cast<size_t>(r) * kRows + c] * y[static_cast<size_t>(c)];
                }
            }
            y[static_cast<size_t>(r)] = (diag == rocsparse_diag_type_unit)
                                            ? sum
                                            : sum / op[static_cast<size_t>(r) * kRows + r];
        }
        return y;
    }

    void expect_solution(const std::vector<test_T>& y, const std::vector<double>& ref)
    {
        ASSERT_EQ(y.size(), ref.size());
        for(size_t row = 0; row < ref.size(); ++row)
        {
            EXPECT_NEAR(y[row], ref[row], 1e-5 * std::max(1.0, std::abs(ref[row])))
                << "row " << row;
        }
    }

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

        scoped_bsrsv_objects(const scoped_bsrsv_objects&) = delete;
        scoped_bsrsv_objects& operator=(const scoped_bsrsv_objects&) = delete;
    };

    // Public analysis and solve of the matrix build_matrix creates. pivot, when
    // not null, receives the zero pivot rocsparse_bsrsv_zero_pivot must report.
    void solve_public(const test_matrix<rocsparse_int, rocsparse_int>& a,
                      rocsparse_operation                              trans,
                      int                                              max_grid_x,
                      std::vector<test_T>&                             y_out,
                      rocsparse_int*                                   pivot = nullptr)
    {
        const rocsparse_int       nnzb = a.nnzb();
        const std::vector<test_T> x    = make_x();
        const std::vector<test_T> y(static_cast<size_t>(kRows), static_cast<test_T>(-7));

        rocsparse_ut::device_vector<rocsparse_int> d_row_ptr(a.row_ptr);
        rocsparse_ut::device_vector<rocsparse_int> d_col_ind(a.col_ind);
        rocsparse_ut::device_vector<test_T>        d_val(a.val);
        rocsparse_ut::device_vector<test_T>        d_x(x);
        rocsparse_ut::device_vector<test_T>        d_y(y);

        scoped_bsrsv_objects objects;
        ASSERT_EQ(rocsparse_create_handle(&objects.handle), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_mat_descr(&objects.descr), rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_fill_mode(objects.descr,
                                              a.lower ? rocsparse_fill_mode_lower
                                                      : rocsparse_fill_mode_upper),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_diag_type(objects.descr, rocsparse_diag_type_non_unit),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_mat_info(&objects.info), rocsparse_status_success);

        size_t buffer_size = 0;
        ASSERT_EQ(rocsparse_sbsrsv_buffer_size(objects.handle,
                                               a.dir,
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
                                            a.dir,
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

        const test_T alpha = kAlpha;
        {
            std::unique_ptr<rocsparse_ut::ScopedMaxGridSizeX> clamp;
            if(max_grid_x > 0)
            {
                clamp = std::make_unique<rocsparse_ut::ScopedMaxGridSizeX>(objects.handle,
                                                                           max_grid_x);
            }
            ASSERT_EQ(rocsparse_sbsrsv_solve(objects.handle,
                                             a.dir,
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
            rocsparse_int position = -2;
            EXPECT_EQ(rocsparse_bsrsv_zero_pivot(objects.handle, objects.info, &position),
                      rocsparse_status_zero_pivot);
            *pivot = position;
        }
    }

    void expect_public_solve(bool lower, rocsparse_operation trans, int max_grid_x)
    {
        SCOPED_TRACE(::testing::Message() << (lower ? "lower" : "upper") << " trans " << trans
                                          << " max_grid_x " << max_grid_x);
        const auto          a = build_matrix<rocsparse_int, rocsparse_int>(lower);
        std::vector<test_T> y;
        ASSERT_NO_FATAL_FAILURE(solve_public(a, trans, max_grid_x, y));
        expect_solution(y, reference(a, lower, trans, rocsparse_diag_type_non_unit));
    }

    rocsparse_int public_singular_pivot(bool lower, int32_t singular_row)
    {
        const auto a = build_matrix<rocsparse_int, rocsparse_int>(
            lower, rocsparse_index_base_zero, singular_row);
        std::vector<test_T> y;
        rocsparse_int       pivot = -2;
        solve_public(a, rocsparse_operation_none, 1, y, &pivot);
        return pivot;
    }

    // One block-lower matrix with 32-bit and 64-bit indices on one handle, matrix
    // descriptor and mat_info, and a buffer large enough for bsric0 and for bsrsv
    // at either width and fill mode. The bsrsv analyses of both fill modes share
    // one zero pivot, which is how an analysis of one width retypes the pivot that
    // another analysis left.
    class internal_level2_bsrsv_wide : public ::testing::Test
    {
    protected:
        test_matrix<int32_t, int32_t> a32;
        test_matrix<int64_t, int64_t> a64;
        std::vector<test_T>           x = make_x();

        scoped_bsrsv_objects objects;

        std::unique_ptr<rocsparse_ut::device_vector<int32_t>> d_row32;
        std::unique_ptr<rocsparse_ut::device_vector<int32_t>> d_col32;
        std::unique_ptr<rocsparse_ut::device_vector<int64_t>> d_row64;
        std::unique_ptr<rocsparse_ut::device_vector<int64_t>> d_col64;
        std::unique_ptr<rocsparse_ut::device_vector<test_T>>  d_val;
        std::unique_ptr<rocsparse_ut::device_vector<test_T>>  d_x;
        std::unique_ptr<rocsparse_ut::device_vector<test_T>>  d_y;
        std::unique_ptr<rocsparse_ut::device_vector<char>>    d_buffer;

        // With missing_row >= 0 that block row has no diagonal block, which the
        // analyses report as a structural zero.
        void build(int32_t missing_row = -1)
        {
            a32 = build_matrix<int32_t, int32_t>(true);
            if(missing_row >= 0)
            {
                remove_diagonal_block(a32, missing_row);
            }
            a64 = widen(a32);

            d_row32 = std::make_unique<rocsparse_ut::device_vector<int32_t>>(a32.row_ptr);
            d_col32 = std::make_unique<rocsparse_ut::device_vector<int32_t>>(a32.col_ind);
            d_row64 = std::make_unique<rocsparse_ut::device_vector<int64_t>>(a64.row_ptr);
            d_col64 = std::make_unique<rocsparse_ut::device_vector<int64_t>>(a64.col_ind);
            d_val   = std::make_unique<rocsparse_ut::device_vector<test_T>>(a32.val);
            d_x     = std::make_unique<rocsparse_ut::device_vector<test_T>>(x);
            d_y     = std::make_unique<rocsparse_ut::device_vector<test_T>>(
                std::vector<test_T>(static_cast<size_t>(kRows), static_cast<test_T>(-7)));
            ASSERT_NE(d_row32->ptr, nullptr);
            ASSERT_NE(d_col32->ptr, nullptr);
            ASSERT_NE(d_row64->ptr, nullptr);
            ASSERT_NE(d_col64->ptr, nullptr);
            ASSERT_NE(d_val->ptr, nullptr);
            ASSERT_NE(d_x->ptr, nullptr);
            ASSERT_NE(d_y->ptr, nullptr);

            ASSERT_EQ(rocsparse_create_handle(&objects.handle), rocsparse_status_success);
            ASSERT_EQ(rocsparse_create_mat_descr(&objects.descr), rocsparse_status_success);
            ASSERT_EQ(rocsparse_set_mat_diag_type(objects.descr, rocsparse_diag_type_non_unit),
                      rocsparse_status_success);
            ASSERT_EQ(rocsparse_create_mat_info(&objects.info), rocsparse_status_success);

            size_t bytes = 0;
            for(const auto fill : {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper})
            {
                ASSERT_EQ(rocsparse_set_mat_fill_mode(objects.descr, fill),
                          rocsparse_status_success);
                size_t b32 = 0;
                size_t b64 = 0;
                ASSERT_EQ((buffer_size<int32_t, int32_t>(&b32)), rocsparse_status_success);
                ASSERT_EQ((buffer_size<int64_t, int64_t>(&b64)), rocsparse_status_success);
                bytes = std::max(bytes, std::max(b32, b64));
            }
            ASSERT_EQ(rocsparse_set_mat_fill_mode(objects.descr, rocsparse_fill_mode_lower),
                      rocsparse_status_success);
            size_t ic0_bytes = 0;
            ASSERT_EQ(rocsparse_sbsric0_buffer_size(objects.handle,
                                                    rocsparse_direction_row,
                                                    kMb,
                                                    a32.nnzb(),
                                                    objects.descr,
                                                    d_val->ptr,
                                                    d_row32->ptr,
                                                    d_col32->ptr,
                                                    kBlockDim,
                                                    objects.info,
                                                    &ic0_bytes),
                      rocsparse_status_success);
            bytes = std::max(bytes, ic0_bytes);

            d_buffer = std::make_unique<rocsparse_ut::device_vector<char>>(
                std::vector<char>(bytes + 1, 0));
            ASSERT_NE(d_buffer->ptr, nullptr);
        }

        template <typename I>
        const I* row_ptr() const
        {
            if constexpr(std::is_same<I, int64_t>::value)
            {
                return d_row64->ptr;
            }
            else
            {
                return d_row32->ptr;
            }
        }

        template <typename J>
        const J* col_ind() const
        {
            if constexpr(std::is_same<J, int64_t>::value)
            {
                return d_col64->ptr;
            }
            else
            {
                return d_col32->ptr;
            }
        }

        template <typename I, typename J>
        rocsparse_status buffer_size(size_t* bytes)
        {
            return rocsparse::bsrsv_buffer_size_template<test_T, I, J>(objects.handle,
                                                                       rocsparse_direction_row,
                                                                       rocsparse_operation_none,
                                                                       static_cast<J>(kMb),
                                                                       static_cast<I>(a32.nnzb()),
                                                                       objects.descr,
                                                                       d_val->ptr,
                                                                       row_ptr<I>(),
                                                                       col_ind<J>(),
                                                                       kBlockDim,
                                                                       objects.info,
                                                                       bytes);
        }

        template <typename I, typename J>
        rocsparse_status analysis(rocsparse_fill_mode fill, rocsparse_analysis_policy policy)
        {
            const rocsparse_status status = rocsparse_set_mat_fill_mode(objects.descr, fill);
            if(status != rocsparse_status_success)
            {
                return status;
            }
            return rocsparse::bsrsv_analysis_template<test_T, I, J>(objects.handle,
                                                                    rocsparse_direction_row,
                                                                    rocsparse_operation_none,
                                                                    static_cast<J>(kMb),
                                                                    static_cast<I>(a32.nnzb()),
                                                                    objects.descr,
                                                                    d_val->ptr,
                                                                    row_ptr<I>(),
                                                                    col_ind<J>(),
                                                                    kBlockDim,
                                                                    objects.info,
                                                                    policy,
                                                                    rocsparse_solve_policy_auto,
                                                                    d_buffer->ptr);
        }

        rocsparse_status bsric0_analysis()
        {
            const rocsparse_status status
                = rocsparse_set_mat_fill_mode(objects.descr, rocsparse_fill_mode_lower);
            if(status != rocsparse_status_success)
            {
                return status;
            }
            return rocsparse_sbsric0_analysis(objects.handle,
                                              rocsparse_direction_row,
                                              kMb,
                                              a32.nnzb(),
                                              objects.descr,
                                              d_val->ptr,
                                              d_row32->ptr,
                                              d_col32->ptr,
                                              kBlockDim,
                                              objects.info,
                                              rocsparse_analysis_policy_force,
                                              rocsparse_solve_policy_auto,
                                              d_buffer->ptr);
        }

        // Lower solve through bsrsv_solve_core, the entry rocsparse_sptrsv is going
        // to use, with the analysis and pivot given explicitly. y is reset first.
        template <typename I, typename J>
        rocsparse_status
            solve_core(rocsparse::trm_info_t* trm, rocsparse_indextype pivot_type, J* pivot)
        {
            if(hipMemcpy(
                   d_y->ptr,
                   std::vector<test_T>(static_cast<size_t>(kRows), static_cast<test_T>(-7)).data(),
                   sizeof(test_T) * kRows,
                   hipMemcpyHostToDevice)
               != hipSuccess)
            {
                return rocsparse_status_internal_error;
            }
            const rocsparse_status status
                = rocsparse_set_mat_fill_mode(objects.descr, rocsparse_fill_mode_lower);
            if(status != rocsparse_status_success)
            {
                return status;
            }
            const test_T alpha = kAlpha;
            return rocsparse::bsrsv_solve_core<test_T, I, J>(objects.handle,
                                                             rocsparse_direction_row,
                                                             rocsparse_operation_none,
                                                             static_cast<J>(kMb),
                                                             static_cast<I>(a32.nnzb()),
                                                             &alpha,
                                                             objects.descr,
                                                             d_val->ptr,
                                                             row_ptr<I>(),
                                                             col_ind<J>(),
                                                             kBlockDim,
                                                             trm,
                                                             pivot_type,
                                                             pivot,
                                                             d_x->ptr,
                                                             d_y->ptr,
                                                             rocsparse_solve_policy_auto,
                                                             d_buffer->ptr);
        }

        // The same solve with the lower analysis and the pivot that mat_info holds.
        template <typename I, typename J>
        rocsparse_status solve_stored()
        {
            auto* container = objects.info->get_bsrsv_info();
            return solve_core<I, J>(
                objects.info->get_bsrsv_info(rocsparse_operation_none, rocsparse_fill_mode_lower),
                container->get_indextype(),
                static_cast<J*>(container->get_position()));
        }

        rocsparse_status solve_public()
        {
            const rocsparse_status status
                = rocsparse_set_mat_fill_mode(objects.descr, rocsparse_fill_mode_lower);
            if(status != rocsparse_status_success)
            {
                return status;
            }
            const test_T alpha = kAlpha;
            return rocsparse_sbsrsv_solve(objects.handle,
                                          rocsparse_direction_row,
                                          rocsparse_operation_none,
                                          kMb,
                                          a32.nnzb(),
                                          &alpha,
                                          objects.descr,
                                          d_val->ptr,
                                          d_row32->ptr,
                                          d_col32->ptr,
                                          kBlockDim,
                                          objects.info,
                                          d_x->ptr,
                                          d_y->ptr,
                                          rocsparse_solve_policy_auto,
                                          d_buffer->ptr);
        }

        void expect_lower_solution()
        {
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            expect_solution(
                rocsparse_ut::to_host(*d_y),
                reference(a32, true, rocsparse_operation_none, rocsparse_diag_type_non_unit));
        }

        rocsparse_indextype pivot_type()
        {
            return objects.info->get_bsrsv_info()->get_indextype();
        }

        // The shared pivot, read with type J, which it must have.
        template <typename J>
        J stored_pivot()
        {
            auto* container = objects.info->get_bsrsv_info();
            EXPECT_EQ(container->get_indextype(), rocsparse::get_indextype<J>());
            J value = 0;
            EXPECT_NE(container->get_position(), nullptr);
            if(container->get_position() != nullptr)
            {
                EXPECT_EQ(
                    hipMemcpy(&value, container->get_position(), sizeof(J), hipMemcpyDeviceToHost),
                    hipSuccess);
            }
            return value;
        }

        void expect_public_zero_pivot(rocsparse_int row)
        {
            rocsparse_int position = -2;
            EXPECT_EQ(rocsparse_bsrsv_zero_pivot(objects.handle, objects.info, &position),
                      rocsparse_status_zero_pivot);
            EXPECT_EQ(position, row);
        }
    };
}

TEST_F(internal_level2_bsrsv_wide, general_i32_chunk)
{
    expect_wide<int32_t, int32_t>(true, 1);
    expect_wide<int32_t, int32_t>(false, 1);
}

TEST_F(internal_level2_bsrsv_wide, general_i32_full_grid)
{
    // Block 128 and 8 block rows: 2 virtual blocks on wave32, 4 on wave64.
    // A grid of 2 is exact on wave32 and a two-chunk walk on wave64.
    expect_wide<int32_t, int32_t>(true, 2);
}

TEST_F(internal_level2_bsrsv_wide, general_i64_i32_chunk)
{
    expect_wide<int64_t, int32_t>(true, 1);
    expect_wide<int64_t, int32_t>(false, 1);
}

TEST_F(internal_level2_bsrsv_wide, general_i64_chunk)
{
    expect_wide<int64_t, int64_t>(true, 1);
    expect_wide<int64_t, int64_t>(false, 1);
}

TEST_F(internal_level2_bsrsv_wide, direction_nonsymmetric_blocks)
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

TEST_F(internal_level2_bsrsv_wide, zero_pivot_reports_block_row)
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

TEST_F(internal_level2_bsrsv_wide, zero_pivot_shared_reports_block_row)
{
    // The host launches the shared kernels only on wave64. These direct launches
    // use the device wavefront, so the shared pivot path runs on wave32 as well.
    expect_singular_pivot<int32_t, int32_t>(true, rocsparse_index_base_one, 3, 1, true);
    expect_singular_pivot<int64_t, int32_t>(false, rocsparse_index_base_zero, 4, 1, true);
    expect_singular_pivot<int64_t, int64_t>(true, rocsparse_index_base_one, 6, 2, true);
    expect_singular_pivot<int64_t, int64_t>(false, rocsparse_index_base_one, 2, 1, true);
}

TEST_F(internal_level2_bsrsv_wide, needs_wide_predicate_boundaries)
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

TEST_F(internal_level2_bsrsv_wide, public_lower_narrow_and_clamped)
{
    for(int limit : {0, 1})
    {
        expect_public_solve(true, rocsparse_operation_none, limit);
    }
}

TEST_F(internal_level2_bsrsv_wide, public_upper_clamped)
{
    expect_public_solve(false, rocsparse_operation_none, 1);
}

TEST_F(internal_level2_bsrsv_wide, public_lower_transpose_clamped)
{
    // A is lower, so A^T is the upper system.
    expect_public_solve(true, rocsparse_operation_transpose, 1);
}

TEST_F(internal_level2_bsrsv_wide, public_singular_pivot_clamped)
{
    // block_dim is 2, so a clamped wave64 launch takes the shared kernel and a
    // wave32 launch takes the general kernel. Both report the block row. A
    // 128-thread block solves the first 4 map slots on wave32 and 2 on wave64,
    // so the singular rows sit past them: lower row 6 is map slot 6, and upper
    // row 1 is map slot 6 as well. A narrow kernel on the clamped grid never
    // reaches them.
    EXPECT_EQ(public_singular_pivot(true, 6), static_cast<rocsparse_int>(6));
    EXPECT_EQ(public_singular_pivot(false, 1), static_cast<rocsparse_int>(1));
}

// bsric0 builds the analysis, bsrsv reuse borrows it, and a later bsrsv force
// must build its own analysis instead of deleting the borrowed one. The force
// would otherwise free the borrowed analysis and allocate the new one, often at
// the same address, so the check is that both analyses are alive at once.
TEST_F(internal_level2_bsrsv_wide, force_keeps_borrowed_bsric0_analysis)
{
    ASSERT_NO_FATAL_FAILURE(build());
    ASSERT_EQ(bsric0_analysis(), rocsparse_status_success);
    ASSERT_EQ(
        (analysis<int32_t, int32_t>(rocsparse_fill_mode_lower, rocsparse_analysis_policy_reuse)),
        rocsparse_status_success);

    rocsparse::trm_info_t* const borrowed
        = objects.info->get_bsric0_info(rocsparse_operation_none, rocsparse_fill_mode_lower);
    ASSERT_NE(borrowed, nullptr);
    EXPECT_EQ(objects.info->get_bsrsv_info(rocsparse_operation_none, rocsparse_fill_mode_lower),
              borrowed);

    ASSERT_EQ(
        (analysis<int32_t, int32_t>(rocsparse_fill_mode_lower, rocsparse_analysis_policy_force)),
        rocsparse_status_success);
    rocsparse::trm_info_t* const own
        = objects.info->get_bsrsv_info(rocsparse_operation_none, rocsparse_fill_mode_lower);
    EXPECT_NE(own, nullptr);
    EXPECT_NE(own, borrowed);
    EXPECT_EQ(objects.info->get_bsric0_info(rocsparse_operation_none, rocsparse_fill_mode_lower),
              borrowed);

    ASSERT_EQ(rocsparse_bsric0_clear(objects.handle, objects.info), rocsparse_status_success);
}

TEST_F(internal_level2_bsrsv_wide, reuse_borrowed_bsric0_solves)
{
    ASSERT_NO_FATAL_FAILURE(build());
    ASSERT_EQ(bsric0_analysis(), rocsparse_status_success);
    ASSERT_EQ(
        (analysis<int32_t, int32_t>(rocsparse_fill_mode_lower, rocsparse_analysis_policy_reuse)),
        rocsparse_status_success);
    ASSERT_EQ(solve_public(), rocsparse_status_success);
    expect_lower_solution();

    rocsparse_int position = -2;
    EXPECT_EQ(rocsparse_bsrsv_zero_pivot(objects.handle, objects.info, &position),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_bsric0_clear(objects.handle, objects.info), rocsparse_status_success);
}

TEST_F(internal_level2_bsrsv_wide, reuse_borrowed_structural_zero_reported)
{
    // Block row 3 has no diagonal block. The donor's analysis finds it and the
    // borrowed solve must report row 3, not success.
    ASSERT_NO_FATAL_FAILURE(build(3));
    ASSERT_EQ(bsric0_analysis(), rocsparse_status_success);
    ASSERT_EQ(
        (analysis<int32_t, int32_t>(rocsparse_fill_mode_lower, rocsparse_analysis_policy_reuse)),
        rocsparse_status_success);
    ASSERT_EQ(solve_public(), rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    expect_public_zero_pivot(3);
    ASSERT_EQ(rocsparse_bsric0_clear(objects.handle, objects.info), rocsparse_status_success);
}

// An analysis at one width, then at another, on the same mat_info. The pivot
// type follows the latest analysis, and a solve of that width succeeds.
TEST_F(internal_level2_bsrsv_wide, mixed_width_retypes_pivot_and_solves)
{
    ASSERT_NO_FATAL_FAILURE(build());
    ASSERT_EQ(
        (analysis<int32_t, int32_t>(rocsparse_fill_mode_lower, rocsparse_analysis_policy_force)),
        rocsparse_status_success);
    EXPECT_EQ(pivot_type(), rocsparse_indextype_i32);

    ASSERT_EQ(
        (analysis<int64_t, int64_t>(rocsparse_fill_mode_lower, rocsparse_analysis_policy_force)),
        rocsparse_status_success);
    EXPECT_EQ(pivot_type(), rocsparse_indextype_i64);
    // The public entry reads the pivot as rocsparse_int, so an i64 pivot is rejected.
    EXPECT_EQ(solve_public(), rocsparse_status_invalid_value);
    ASSERT_EQ((solve_stored<int64_t, int64_t>()), rocsparse_status_success);
    ASSERT_NO_FATAL_FAILURE(expect_lower_solution());
    EXPECT_EQ(stored_pivot<int64_t>(), std::numeric_limits<int64_t>::max());

    ASSERT_EQ(
        (analysis<int32_t, int32_t>(rocsparse_fill_mode_lower, rocsparse_analysis_policy_force)),
        rocsparse_status_success);
    EXPECT_EQ(pivot_type(), rocsparse_indextype_i32);
    ASSERT_EQ(solve_public(), rocsparse_status_success);
    ASSERT_NO_FATAL_FAILURE(expect_lower_solution());
}

// An upper i32 analysis retypes the shared pivot. Until the lower i64 analysis
// is reused, an i64 solve rejects that i32 pivot, and the reuse gives it an i64
// pivot again.
TEST_F(internal_level2_bsrsv_wide, mixed_width_own_reuse_retypes_pivot)
{
    ASSERT_NO_FATAL_FAILURE(build());
    ASSERT_EQ(
        (analysis<int64_t, int64_t>(rocsparse_fill_mode_lower, rocsparse_analysis_policy_force)),
        rocsparse_status_success);
    ASSERT_EQ(
        (analysis<int32_t, int32_t>(rocsparse_fill_mode_upper, rocsparse_analysis_policy_force)),
        rocsparse_status_success);
    EXPECT_EQ(pivot_type(), rocsparse_indextype_i32);
    EXPECT_EQ((solve_stored<int64_t, int64_t>()), rocsparse_status_invalid_value);

    ASSERT_EQ(
        (analysis<int64_t, int64_t>(rocsparse_fill_mode_lower, rocsparse_analysis_policy_reuse)),
        rocsparse_status_success);
    EXPECT_EQ(pivot_type(), rocsparse_indextype_i64);
    ASSERT_EQ((solve_stored<int64_t, int64_t>()), rocsparse_status_success);
    ASSERT_NO_FATAL_FAILURE(expect_lower_solution());
    EXPECT_EQ(stored_pivot<int64_t>(), std::numeric_limits<int64_t>::max());
}

// Block row 3 has no diagonal block, a structural zero that only the analysis
// records, in the shared pivot. The upper i32 analysis replaces that pivot with
// an i32 one, so the reuse of the lower i64 analysis has to run the analysis
// again to record the zero in an i64 pivot.
TEST_F(internal_level2_bsrsv_wide, mixed_width_own_reuse_keeps_structural_zero)
{
    ASSERT_NO_FATAL_FAILURE(build(3));
    ASSERT_EQ(
        (analysis<int64_t, int64_t>(rocsparse_fill_mode_lower, rocsparse_analysis_policy_force)),
        rocsparse_status_success);
    EXPECT_EQ(stored_pivot<int64_t>(), 3);
    ASSERT_EQ(
        (analysis<int32_t, int32_t>(rocsparse_fill_mode_upper, rocsparse_analysis_policy_force)),
        rocsparse_status_success);
    EXPECT_EQ(stored_pivot<int32_t>(), 3);

    ASSERT_EQ(
        (analysis<int64_t, int64_t>(rocsparse_fill_mode_lower, rocsparse_analysis_policy_reuse)),
        rocsparse_status_success);
    EXPECT_EQ(stored_pivot<int64_t>(), 3);
    ASSERT_EQ((solve_stored<int64_t, int64_t>()), rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(stored_pivot<int64_t>(), 3);
}

// bsrsv_solve_core must reject an analysis of another width even when the
// pivot it is given has the right type. The i64 analysis comes first: read as
// i32, its row map still holds valid block rows, so a solve that skipped the
// check would return success instead of reading out of bounds.
TEST_F(internal_level2_bsrsv_wide, solve_core_rejects_analysis_of_other_width)
{
    ASSERT_NO_FATAL_FAILURE(build());
    rocsparse_ut::device_vector<int32_t> d_pivot32(
        std::vector<int32_t>{std::numeric_limits<int32_t>::max()});
    rocsparse_ut::device_vector<int64_t> d_pivot64(
        std::vector<int64_t>{std::numeric_limits<int64_t>::max()});
    ASSERT_NE(d_pivot32.ptr, nullptr);
    ASSERT_NE(d_pivot64.ptr, nullptr);

    ASSERT_EQ(
        (analysis<int64_t, int64_t>(rocsparse_fill_mode_lower, rocsparse_analysis_policy_force)),
        rocsparse_status_success);
    ASSERT_EQ((solve_core<int32_t, int32_t>(
                  objects.info->get_bsrsv_info(rocsparse_operation_none, rocsparse_fill_mode_lower),
                  rocsparse_indextype_i32,
                  d_pivot32.ptr)),
              rocsparse_status_invalid_value);

    ASSERT_EQ(
        (analysis<int32_t, int32_t>(rocsparse_fill_mode_lower, rocsparse_analysis_policy_force)),
        rocsparse_status_success);
    EXPECT_EQ((solve_core<int64_t, int64_t>(
                  objects.info->get_bsrsv_info(rocsparse_operation_none, rocsparse_fill_mode_lower),
                  rocsparse_indextype_i64,
                  d_pivot64.ptr)),
              rocsparse_status_invalid_value);
}

// Reuse of an analysis of another width rebuilds it. The upper i64 analysis
// leaves an i64 pivot, so the pivot type alone does not force the rebuild.
TEST_F(internal_level2_bsrsv_wide, reuse_rebuilds_analysis_of_other_width)
{
    ASSERT_NO_FATAL_FAILURE(build());
    ASSERT_EQ(
        (analysis<int32_t, int32_t>(rocsparse_fill_mode_lower, rocsparse_analysis_policy_force)),
        rocsparse_status_success);
    ASSERT_EQ(
        (analysis<int64_t, int64_t>(rocsparse_fill_mode_upper, rocsparse_analysis_policy_force)),
        rocsparse_status_success);
    ASSERT_EQ(
        (analysis<int64_t, int64_t>(rocsparse_fill_mode_lower, rocsparse_analysis_policy_reuse)),
        rocsparse_status_success);

    rocsparse::trm_info_t* const trm
        = objects.info->get_bsrsv_info(rocsparse_operation_none, rocsparse_fill_mode_lower);
    ASSERT_NE(trm, nullptr);
    EXPECT_TRUE(trm->has_indextypes(rocsparse_indextype_i64, rocsparse_indextype_i64));
    ASSERT_EQ((solve_stored<int64_t, int64_t>()), rocsparse_status_success);
    ASSERT_NO_FATAL_FAILURE(expect_lower_solution());
}

// An i64 analysis of the other fill mode leaves an i64 pivot. Reuse of the
// lower i32 bsric0 analysis must drop it and copy the donor pivot, which holds
// the structural zero at block row 3.
TEST_F(internal_level2_bsrsv_wide, reuse_drops_mismatched_pivot)
{
    ASSERT_NO_FATAL_FAILURE(build(3));
    ASSERT_EQ(
        (analysis<int64_t, int64_t>(rocsparse_fill_mode_upper, rocsparse_analysis_policy_force)),
        rocsparse_status_success);
    EXPECT_EQ(pivot_type(), rocsparse_indextype_i64);

    ASSERT_EQ(bsric0_analysis(), rocsparse_status_success);
    ASSERT_EQ(
        (analysis<int32_t, int32_t>(rocsparse_fill_mode_lower, rocsparse_analysis_policy_reuse)),
        rocsparse_status_success);
    EXPECT_EQ(pivot_type(), rocsparse_indextype_i32);

    ASSERT_EQ(solve_public(), rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    expect_public_zero_pivot(3);
    ASSERT_EQ(rocsparse_bsric0_clear(objects.handle, objects.info), rocsparse_status_success);
}

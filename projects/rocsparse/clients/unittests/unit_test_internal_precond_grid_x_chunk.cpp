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
// Device unit tests for the chunk loop in the BSR incomplete factorization
// kernels (library/src/precond/bsr{ilu,ic}0/*.cpp), general and block-dim
// specific alike.
//
// Every launch clamps grid.x with rocsparse::get_grid_size_x and every block
// walks the contiguous chunk rocsparse::grid_x_chunk gives it. The clamp only
// binds at millions of blocks, so on any real problem every chunk is one item
// and the loop runs once. These tests make it bind on a small matrix by
// shrinking handle->properties.maxGridSize[0], which get_grid_size_x reads, to
// 1, 2 and 3 blocks. Each block then walks many rows, and rows spin on the done
// flags of rows owned by other chunks and by earlier items of their own chunk.
//
// The factorization of every row is a fixed sequence of operations on final
// values of the rows it depends on, so the result does not depend on the grid:
// the clamped runs must match the full-grid run bit for bit.
//
// The general kernels are exercised with a double complex matrix with
// block_dim 33, which takes them on wave32 and wave64 devices alike (see
// bsr{ilu,ic}0_kernel_launch). The block-dim specific kernels (2_8, 9_32,
// 33_64 for bsrilu0; 2_8, 2_8_unrolled, 9_16, 17_32 for bsric0) are only
// launched on wave64 devices; on a wave32 device the launcher never selects
// them and those tests skip. The structure couples each block row to rows 1, 2,
// 7, 23 and 41 before it, so the dependencies cross chunk boundaries for every
// grid tested. 67 block rows do not divide evenly by 2 or 3, so the last chunk
// is ragged.
//
// A deadlock in the chunk loop shows up as a hang, not a failure: run this
// binary under a timeout.
//

#include "unit_test_utils.hpp"

#include "rocsparse_handle.hpp"

#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <type_traits>
#include <vector>

namespace
{
    constexpr int small_grids[] = {1, 2, 3};

    enum class factorization
    {
        ilu0,
        ic0
    };

    const char* name(factorization f)
    {
        return (f == factorization::ilu0) ? "bsrilu0" : "bsric0";
    }

    struct problem
    {
        const char*                label;
        rocsparse_int              mb;
        rocsparse_int              block_dim;
        std::vector<rocsparse_int> offsets;
        bool                       wave64_only;
    };

    // Block rows coupled to row i: i - offset and i + offset for every offset.
    const std::vector<rocsparse_int> default_offsets = {1, 2, 7, 23, 41};

    std::vector<rocsparse_int> consecutive_offsets(rocsparse_int count)
    {
        std::vector<rocsparse_int> offsets;
        for(rocsparse_int k = 1; k <= count; ++k)
        {
            offsets.push_back(k);
        }
        return offsets;
    }

    template <typename T>
    struct bsr_matrix
    {
        std::vector<rocsparse_int> row_ptr;
        std::vector<rocsparse_int> col_ind;
        std::vector<T>             val;
    };

    // Entry (a, b) of the block at block row i, block column j, for j < i. The
    // block at (j, i) is its conjugate transpose, so A is Hermitian.
    template <typename T>
    T lower_entry(rocsparse_int i, rocsparse_int j, rocsparse_int a, rocsparse_int b)
    {
        using R     = decltype(std::real(T{}));
        const int h = (i * 7 + j * 3 + a * 5 + b) % 11;
        return T(static_cast<R>(0.01 * (1 + h)), static_cast<R>(0.005 * (h - 5)));
    }

    // Entry (a, b) of the diagonal block of block row i: Hermitian, with a real
    // diagonal that dominates everything else in its row, so neither
    // factorization meets a zero or tiny pivot.
    template <typename T>
    T diag_entry(rocsparse_int i, rocsparse_int a, rocsparse_int b)
    {
        using R = decltype(std::real(T{}));
        if(a == b)
        {
            return T(static_cast<R>(100.0 + i % 5), static_cast<R>(0));
        }

        const rocsparse_int lo = (a < b) ? a : b;
        const rocsparse_int hi = (a < b) ? b : a;
        const int           h  = (i + lo * 3 + hi * 5) % 7;
        const double        im = 0.002 * (h - 3);
        return T(static_cast<R>(0.01 * (1 + h)), static_cast<R>((a < b) ? im : -im));
    }

    template <typename T>
    bsr_matrix<T> build_matrix(const problem& p, rocsparse_direction dir)
    {
        bsr_matrix<T> A;
        A.row_ptr.push_back(0);

        const size_t bb = static_cast<size_t>(p.block_dim) * p.block_dim;

        for(rocsparse_int i = 0; i < p.mb; ++i)
        {
            std::vector<rocsparse_int> cols;
            for(auto k = p.offsets.rbegin(); k != p.offsets.rend(); ++k)
            {
                if(i - *k >= 0)
                {
                    cols.push_back(i - *k);
                }
            }
            cols.push_back(i);
            for(rocsparse_int off : p.offsets)
            {
                if(i + off < p.mb)
                {
                    cols.push_back(i + off);
                }
            }

            for(rocsparse_int j : cols)
            {
                const size_t base = A.val.size();
                A.val.resize(base + bb);

                for(rocsparse_int a = 0; a < p.block_dim; ++a)
                {
                    for(rocsparse_int b = 0; b < p.block_dim; ++b)
                    {
                        T v;
                        if(j == i)
                        {
                            v = diag_entry<T>(i, a, b);
                        }
                        else if(j < i)
                        {
                            v = lower_entry<T>(i, j, a, b);
                        }
                        else
                        {
                            const T t = lower_entry<T>(j, i, b, a);
                            v         = T(std::real(t), -std::imag(t));
                        }

                        const size_t pos  = (dir == rocsparse_direction_row)
                                                ? static_cast<size_t>(a) * p.block_dim + b
                                                : static_cast<size_t>(b) * p.block_dim + a;
                        A.val[base + pos] = v;
                    }
                }

                A.col_ind.push_back(j);
            }

            A.row_ptr.push_back(static_cast<rocsparse_int>(A.col_ind.size()));
        }

        return A;
    }

    // The precision-specific entry points, so the harness can be a template.
    template <typename T, typename... P>
    rocsparse_status bsrilu0_buffer_size(P... p)
    {
        if constexpr(std::is_same_v<T, rocsparse_double_complex>)
        {
            return rocsparse_zbsrilu0_buffer_size(p...);
        }
        else
        {
            return rocsparse_cbsrilu0_buffer_size(p...);
        }
    }

    template <typename T, typename... P>
    rocsparse_status bsrilu0_analysis(P... p)
    {
        if constexpr(std::is_same_v<T, rocsparse_double_complex>)
        {
            return rocsparse_zbsrilu0_analysis(p...);
        }
        else
        {
            return rocsparse_cbsrilu0_analysis(p...);
        }
    }

    template <typename T, typename... P>
    rocsparse_status bsrilu0_solve(P... p)
    {
        if constexpr(std::is_same_v<T, rocsparse_double_complex>)
        {
            return rocsparse_zbsrilu0(p...);
        }
        else
        {
            return rocsparse_cbsrilu0(p...);
        }
    }

    template <typename T, typename... P>
    rocsparse_status bsric0_buffer_size(P... p)
    {
        if constexpr(std::is_same_v<T, rocsparse_double_complex>)
        {
            return rocsparse_zbsric0_buffer_size(p...);
        }
        else
        {
            return rocsparse_cbsric0_buffer_size(p...);
        }
    }

    template <typename T, typename... P>
    rocsparse_status bsric0_analysis(P... p)
    {
        if constexpr(std::is_same_v<T, rocsparse_double_complex>)
        {
            return rocsparse_zbsric0_analysis(p...);
        }
        else
        {
            return rocsparse_cbsric0_analysis(p...);
        }
    }

    template <typename T, typename... P>
    rocsparse_status bsric0_solve(P... p)
    {
        if constexpr(std::is_same_v<T, rocsparse_double_complex>)
        {
            return rocsparse_zbsric0(p...);
        }
        else
        {
            return rocsparse_cbsric0(p...);
        }
    }

    // Factorize A in place through the public API and read the values back.
    // max_grid_x > 0 replaces handle->properties.maxGridSize[0] for the numeric
    // phase only, which is what the bsr{ilu,ic}0 launches clamp grid.x against;
    // 0 leaves the device limit in place.
    template <typename T>
    void factorize(factorization        f,
                   const problem&       p,
                   rocsparse_direction  dir,
                   int                  max_grid_x,
                   const bsr_matrix<T>& A,
                   std::vector<T>&      result,
                   rocsparse_int&       pivot)
    {
        const rocsparse_int nnzb = static_cast<rocsparse_int>(A.col_ind.size());

        rocsparse_ut::device_vector<rocsparse_int> d_row_ptr(A.row_ptr);
        rocsparse_ut::device_vector<rocsparse_int> d_col_ind(A.col_ind);
        rocsparse_ut::device_vector<T>             d_val(A.val);
        ASSERT_NE(d_row_ptr.ptr, nullptr);
        ASSERT_NE(d_col_ind.ptr, nullptr);
        ASSERT_NE(d_val.ptr, nullptr);

        rocsparse_handle    handle = nullptr;
        rocsparse_mat_descr descr  = nullptr;
        rocsparse_mat_info  info   = nullptr;
        ASSERT_EQ(rocsparse_create_handle(&handle), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

        size_t buffer_size = 0;
        ASSERT_EQ((f == factorization::ilu0) ? bsrilu0_buffer_size<T>(handle,
                                                                      dir,
                                                                      p.mb,
                                                                      nnzb,
                                                                      descr,
                                                                      d_val.ptr,
                                                                      d_row_ptr.ptr,
                                                                      d_col_ind.ptr,
                                                                      p.block_dim,
                                                                      info,
                                                                      &buffer_size)
                                             : bsric0_buffer_size<T>(handle,
                                                                     dir,
                                                                     p.mb,
                                                                     nnzb,
                                                                     descr,
                                                                     d_val.ptr,
                                                                     d_row_ptr.ptr,
                                                                     d_col_ind.ptr,
                                                                     p.block_dim,
                                                                     info,
                                                                     &buffer_size),
                  rocsparse_status_success);

        rocsparse_ut::device_vector<char> d_buffer(buffer_size > 0 ? buffer_size : 1);
        ASSERT_NE(d_buffer.ptr, nullptr);

        ASSERT_EQ((f == factorization::ilu0) ? bsrilu0_analysis<T>(handle,
                                                                   dir,
                                                                   p.mb,
                                                                   nnzb,
                                                                   descr,
                                                                   d_val.ptr,
                                                                   d_row_ptr.ptr,
                                                                   d_col_ind.ptr,
                                                                   p.block_dim,
                                                                   info,
                                                                   rocsparse_analysis_policy_force,
                                                                   rocsparse_solve_policy_auto,
                                                                   d_buffer.ptr)
                                             : bsric0_analysis<T>(handle,
                                                                  dir,
                                                                  p.mb,
                                                                  nnzb,
                                                                  descr,
                                                                  d_val.ptr,
                                                                  d_row_ptr.ptr,
                                                                  d_col_ind.ptr,
                                                                  p.block_dim,
                                                                  info,
                                                                  rocsparse_analysis_policy_force,
                                                                  rocsparse_solve_policy_auto,
                                                                  d_buffer.ptr),
                  rocsparse_status_success);

        if(max_grid_x > 0)
        {
            handle->properties.maxGridSize[0] = max_grid_x;
        }

        ASSERT_EQ((f == factorization::ilu0) ? bsrilu0_solve<T>(handle,
                                                                dir,
                                                                p.mb,
                                                                nnzb,
                                                                descr,
                                                                d_val.ptr,
                                                                d_row_ptr.ptr,
                                                                d_col_ind.ptr,
                                                                p.block_dim,
                                                                info,
                                                                rocsparse_solve_policy_auto,
                                                                d_buffer.ptr)
                                             : bsric0_solve<T>(handle,
                                                               dir,
                                                               p.mb,
                                                               nnzb,
                                                               descr,
                                                               d_val.ptr,
                                                               d_row_ptr.ptr,
                                                               d_col_ind.ptr,
                                                               p.block_dim,
                                                               info,
                                                               rocsparse_solve_policy_auto,
                                                               d_buffer.ptr),
                  rocsparse_status_success);

        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        pivot = 0;
        const rocsparse_status pivot_status
            = (f == factorization::ilu0) ? rocsparse_bsrilu0_zero_pivot(handle, info, &pivot)
                                         : rocsparse_bsric0_zero_pivot(handle, info, &pivot);
        EXPECT_EQ(pivot_status, rocsparse_status_success);

        result = rocsparse_ut::to_host(d_val);

        EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_handle(handle), rocsparse_status_success);
    }

    bool device_is_wave64()
    {
        rocsparse_handle handle = nullptr;
        EXPECT_EQ(rocsparse_create_handle(&handle), rocsparse_status_success);
        const bool wave64 = (handle->wavefront_size == 64);
        EXPECT_EQ(rocsparse_destroy_handle(handle), rocsparse_status_success);
        return wave64;
    }

    template <typename T>
    void expect_same_result_on_small_grids(factorization       f,
                                           const problem&      p,
                                           rocsparse_direction dir)
    {
        if(p.wave64_only && !device_is_wave64())
        {
            GTEST_SKIP() << p.label << " is only launched on wave64 devices";
        }

        const bsr_matrix<T> A = build_matrix<T>(p, dir);

        std::vector<T> reference;
        rocsparse_int  reference_pivot = 0;
        ASSERT_NO_FATAL_FAILURE(factorize<T>(f, p, dir, 0, A, reference, reference_pivot));
        ASSERT_EQ(reference_pivot, -1) << name(f) << " " << p.label << " full grid";
        ASSERT_EQ(reference.size(), A.val.size());
        ASSERT_NE(std::memcmp(reference.data(), A.val.data(), A.val.size() * sizeof(T)), 0)
            << name(f) << " " << p.label << " full grid left the matrix untouched";

        for(int grid : small_grids)
        {
            std::vector<T> got;
            rocsparse_int  pivot = 0;
            ASSERT_NO_FATAL_FAILURE(factorize<T>(f, p, dir, grid, A, got, pivot));
            EXPECT_EQ(pivot, -1) << name(f) << " " << p.label << " grid.x " << grid;
            ASSERT_EQ(got.size(), reference.size());

            //
            // Report the first differing block row rather than every entry, so
            // a failure names the chunk that was skipped or computed early.
            //
            for(rocsparse_int i = 0; i < p.mb; ++i)
            {
                const size_t first = static_cast<size_t>(A.row_ptr[i]) * p.block_dim * p.block_dim;
                const size_t count = static_cast<size_t>(A.row_ptr[i + 1] - A.row_ptr[i])
                                     * p.block_dim * p.block_dim;

                ASSERT_EQ(
                    std::memcmp(got.data() + first, reference.data() + first, count * sizeof(T)), 0)
                    << name(f) << " " << p.label << ": block row " << i
                    << " differs from the full-grid result with grid.x clamped to " << grid << " ("
                    << p.mb << " block rows, "
                    << (dir == rocsparse_direction_row ? "row" : "column") << " direction)";
            }
        }
    }

    const problem general_problem{"general", 67, 33, default_offsets, false};

    // Block dimensions that pick each block-dim specific kernel. bsrilu0 takes
    // 2_8 up to 8, 9_32 up to 32 (16 or 32 threads along x) and 33_64 above,
    // where double complex goes to the general kernel, hence single precision.
    // bsric0 takes 2_8_unrolled while the lower triangle has at most 32 blocks
    // per row, 2_8 up to 128, then 9_16 and 17_32 by block dimension.
    const problem ilu0_2_8{"2_8", 67, 5, default_offsets, true};
    const problem ilu0_9_32_x16{"9_32 (16x4 threads)", 67, 12, default_offsets, true};
    const problem ilu0_9_32_x32{"9_32 (32x2 threads)", 67, 20, default_offsets, true};
    const problem ilu0_33_64{"33_64", 67, 40, default_offsets, true};
    const problem ic0_2_8_unrolled{"2_8_unrolled", 67, 5, default_offsets, true};
    const problem ic0_2_8{"2_8", 100, 5, consecutive_offsets(40), true};
    const problem ic0_9_16{"9_16", 67, 12, default_offsets, true};
    const problem ic0_17_32{"17_32", 67, 20, default_offsets, true};
}

TEST(internal_precond_grid_x_chunk, bsrilu0_general_small_grid_row)
{
    expect_same_result_on_small_grids<rocsparse_double_complex>(
        factorization::ilu0, general_problem, rocsparse_direction_row);
}

TEST(internal_precond_grid_x_chunk, bsrilu0_general_small_grid_column)
{
    expect_same_result_on_small_grids<rocsparse_double_complex>(
        factorization::ilu0, general_problem, rocsparse_direction_column);
}

TEST(internal_precond_grid_x_chunk, bsric0_general_small_grid_row)
{
    expect_same_result_on_small_grids<rocsparse_double_complex>(
        factorization::ic0, general_problem, rocsparse_direction_row);
}

TEST(internal_precond_grid_x_chunk, bsric0_general_small_grid_column)
{
    expect_same_result_on_small_grids<rocsparse_double_complex>(
        factorization::ic0, general_problem, rocsparse_direction_column);
}

#define SMALL_GRID_TESTS(NAME, FACTORIZATION, PROBLEM)              \
    TEST(internal_precond_grid_x_chunk, NAME##_small_grid_row)      \
    {                                                               \
        expect_same_result_on_small_grids<rocsparse_float_complex>( \
            FACTORIZATION, PROBLEM, rocsparse_direction_row);       \
    }                                                               \
    TEST(internal_precond_grid_x_chunk, NAME##_small_grid_column)   \
    {                                                               \
        expect_same_result_on_small_grids<rocsparse_float_complex>( \
            FACTORIZATION, PROBLEM, rocsparse_direction_column);    \
    }

SMALL_GRID_TESTS(bsrilu0_2_8, factorization::ilu0, ilu0_2_8)
SMALL_GRID_TESTS(bsrilu0_9_32_x16, factorization::ilu0, ilu0_9_32_x16)
SMALL_GRID_TESTS(bsrilu0_9_32_x32, factorization::ilu0, ilu0_9_32_x32)
SMALL_GRID_TESTS(bsrilu0_33_64, factorization::ilu0, ilu0_33_64)
SMALL_GRID_TESTS(bsric0_2_8_unrolled, factorization::ic0, ic0_2_8_unrolled)
SMALL_GRID_TESTS(bsric0_2_8, factorization::ic0, ic0_2_8)
SMALL_GRID_TESTS(bsric0_9_16, factorization::ic0, ic0_9_16)
SMALL_GRID_TESTS(bsric0_17_32, factorization::ic0, ic0_17_32)
#undef SMALL_GRID_TESTS

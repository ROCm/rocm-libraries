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
// Device unit tests for the chunk loop in bsrilu0_kernel_general and
// bsric0_kernel_general (library/src/precond/bsr{ilu,ic}0/*_kernel_general.cpp).
//
// Both launches clamp grid.x with rocsparse::get_grid_size_x and every block
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
// A double complex matrix with block_dim 33 takes the general kernel for both
// routines on wave32 and wave64 devices alike (see bsr{ilu,ic}0_kernel_launch).
// The structure couples each block row to rows 1, 2, 7, 23 and 41 before it, so
// the dependencies cross chunk boundaries for every grid tested. 67 block rows
// do not divide evenly by 2 or 3, so the last chunk is ragged.
//
// A deadlock in the chunk loop shows up as a hang, not a failure: run this
// binary under a timeout.
//

#include "unit_test_utils.hpp"

#include "rocsparse_handle.hpp"

#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <iterator>
#include <vector>

namespace
{
    using test_T = rocsparse_double_complex;

    constexpr rocsparse_int mb        = 67;
    constexpr rocsparse_int block_dim = 33;
    constexpr rocsparse_int offsets[] = {1, 2, 7, 23, 41};

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

    struct bsr_matrix
    {
        std::vector<rocsparse_int> row_ptr;
        std::vector<rocsparse_int> col_ind;
        std::vector<test_T>        val;
    };

    // Entry (a, b) of the block at block row i, block column j, for j < i. The
    // block at (j, i) is its conjugate transpose, so A is Hermitian.
    test_T lower_entry(rocsparse_int i, rocsparse_int j, rocsparse_int a, rocsparse_int b)
    {
        const int h = (i * 7 + j * 3 + a * 5 + b) % 11;
        return test_T(0.01 * (1 + h), 0.005 * (h - 5));
    }

    // Entry (a, b) of the diagonal block of block row i: Hermitian, with a real
    // diagonal that dominates everything else in its row, so neither
    // factorization meets a zero or tiny pivot.
    test_T diag_entry(rocsparse_int i, rocsparse_int a, rocsparse_int b)
    {
        if(a == b)
        {
            return test_T(100.0 + i % 5, 0.0);
        }

        const rocsparse_int lo = (a < b) ? a : b;
        const rocsparse_int hi = (a < b) ? b : a;
        const int           h  = (i + lo * 3 + hi * 5) % 7;
        const double        im = 0.002 * (h - 3);
        return test_T(0.01 * (1 + h), (a < b) ? im : -im);
    }

    bsr_matrix build_matrix(rocsparse_direction dir)
    {
        bsr_matrix A;
        A.row_ptr.push_back(0);

        const size_t bb = static_cast<size_t>(block_dim) * block_dim;

        for(rocsparse_int i = 0; i < mb; ++i)
        {
            std::vector<rocsparse_int> cols;
            for(rocsparse_int k = static_cast<rocsparse_int>(std::size(offsets)) - 1; k >= 0; --k)
            {
                if(i - offsets[k] >= 0)
                {
                    cols.push_back(i - offsets[k]);
                }
            }
            cols.push_back(i);
            for(rocsparse_int off : offsets)
            {
                if(i + off < mb)
                {
                    cols.push_back(i + off);
                }
            }

            for(rocsparse_int j : cols)
            {
                const size_t base = A.val.size();
                A.val.resize(base + bb);

                for(rocsparse_int a = 0; a < block_dim; ++a)
                {
                    for(rocsparse_int b = 0; b < block_dim; ++b)
                    {
                        test_T v;
                        if(j == i)
                        {
                            v = diag_entry(i, a, b);
                        }
                        else if(j < i)
                        {
                            v = lower_entry(i, j, a, b);
                        }
                        else
                        {
                            const test_T t = lower_entry(j, i, b, a);
                            v              = test_T(std::real(t), -std::imag(t));
                        }

                        const size_t pos  = (dir == rocsparse_direction_row)
                                                ? static_cast<size_t>(a) * block_dim + b
                                                : static_cast<size_t>(b) * block_dim + a;
                        A.val[base + pos] = v;
                    }
                }

                A.col_ind.push_back(j);
            }

            A.row_ptr.push_back(static_cast<rocsparse_int>(A.col_ind.size()));
        }

        return A;
    }

    // Factorize A in place through the public API and read the values back.
    // max_grid_x > 0 replaces handle->properties.maxGridSize[0] for the numeric
    // phase only, which is what bsr{ilu,ic}0_kernel_general_launch clamp grid.x
    // against; 0 leaves the device limit in place.
    void factorize(factorization        f,
                   rocsparse_direction  dir,
                   int                  max_grid_x,
                   const bsr_matrix&    A,
                   std::vector<test_T>& result,
                   rocsparse_int&       pivot)
    {
        const rocsparse_int nnzb = static_cast<rocsparse_int>(A.col_ind.size());

        rocsparse_ut::device_vector<rocsparse_int> d_row_ptr(A.row_ptr);
        rocsparse_ut::device_vector<rocsparse_int> d_col_ind(A.col_ind);
        rocsparse_ut::device_vector<test_T>        d_val(A.val);
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
        if(f == factorization::ilu0)
        {
            ASSERT_EQ(rocsparse_zbsrilu0_buffer_size(handle,
                                                     dir,
                                                     mb,
                                                     nnzb,
                                                     descr,
                                                     d_val.ptr,
                                                     d_row_ptr.ptr,
                                                     d_col_ind.ptr,
                                                     block_dim,
                                                     info,
                                                     &buffer_size),
                      rocsparse_status_success);
        }
        else
        {
            ASSERT_EQ(rocsparse_zbsric0_buffer_size(handle,
                                                    dir,
                                                    mb,
                                                    nnzb,
                                                    descr,
                                                    d_val.ptr,
                                                    d_row_ptr.ptr,
                                                    d_col_ind.ptr,
                                                    block_dim,
                                                    info,
                                                    &buffer_size),
                      rocsparse_status_success);
        }

        rocsparse_ut::device_vector<char> d_buffer(buffer_size > 0 ? buffer_size : 1);
        ASSERT_NE(d_buffer.ptr, nullptr);

        if(f == factorization::ilu0)
        {
            ASSERT_EQ(rocsparse_zbsrilu0_analysis(handle,
                                                  dir,
                                                  mb,
                                                  nnzb,
                                                  descr,
                                                  d_val.ptr,
                                                  d_row_ptr.ptr,
                                                  d_col_ind.ptr,
                                                  block_dim,
                                                  info,
                                                  rocsparse_analysis_policy_force,
                                                  rocsparse_solve_policy_auto,
                                                  d_buffer.ptr),
                      rocsparse_status_success);
        }
        else
        {
            ASSERT_EQ(rocsparse_zbsric0_analysis(handle,
                                                 dir,
                                                 mb,
                                                 nnzb,
                                                 descr,
                                                 d_val.ptr,
                                                 d_row_ptr.ptr,
                                                 d_col_ind.ptr,
                                                 block_dim,
                                                 info,
                                                 rocsparse_analysis_policy_force,
                                                 rocsparse_solve_policy_auto,
                                                 d_buffer.ptr),
                      rocsparse_status_success);
        }

        if(max_grid_x > 0)
        {
            handle->properties.maxGridSize[0] = max_grid_x;
        }

        if(f == factorization::ilu0)
        {
            ASSERT_EQ(rocsparse_zbsrilu0(handle,
                                         dir,
                                         mb,
                                         nnzb,
                                         descr,
                                         d_val.ptr,
                                         d_row_ptr.ptr,
                                         d_col_ind.ptr,
                                         block_dim,
                                         info,
                                         rocsparse_solve_policy_auto,
                                         d_buffer.ptr),
                      rocsparse_status_success);
        }
        else
        {
            ASSERT_EQ(rocsparse_zbsric0(handle,
                                        dir,
                                        mb,
                                        nnzb,
                                        descr,
                                        d_val.ptr,
                                        d_row_ptr.ptr,
                                        d_col_ind.ptr,
                                        block_dim,
                                        info,
                                        rocsparse_solve_policy_auto,
                                        d_buffer.ptr),
                      rocsparse_status_success);
        }

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

    void expect_same_result_on_small_grids(factorization f, rocsparse_direction dir)
    {
        const bsr_matrix A = build_matrix(dir);

        std::vector<test_T> reference;
        rocsparse_int       reference_pivot = 0;
        ASSERT_NO_FATAL_FAILURE(factorize(f, dir, 0, A, reference, reference_pivot));
        ASSERT_EQ(reference_pivot, -1) << name(f) << " full grid";
        ASSERT_EQ(reference.size(), A.val.size());
        ASSERT_NE(std::memcmp(reference.data(), A.val.data(), A.val.size() * sizeof(test_T)), 0)
            << name(f) << " full grid left the matrix untouched";

        for(int grid : small_grids)
        {
            std::vector<test_T> got;
            rocsparse_int       pivot = 0;
            ASSERT_NO_FATAL_FAILURE(factorize(f, dir, grid, A, got, pivot));
            EXPECT_EQ(pivot, -1) << name(f) << " grid.x " << grid;
            ASSERT_EQ(got.size(), reference.size());

            //
            // Report the first differing block row rather than every entry, so
            // a failure names the chunk that was skipped or computed early.
            //
            for(rocsparse_int i = 0; i < mb; ++i)
            {
                const size_t first = static_cast<size_t>(A.row_ptr[i]) * block_dim * block_dim;
                const size_t count
                    = static_cast<size_t>(A.row_ptr[i + 1] - A.row_ptr[i]) * block_dim * block_dim;

                ASSERT_EQ(std::memcmp(
                              got.data() + first, reference.data() + first, count * sizeof(test_T)),
                          0)
                    << name(f) << " block row " << i << " differs from the full-grid result with "
                    << "grid.x clamped to " << grid << " (" << mb << " block rows, "
                    << (dir == rocsparse_direction_row ? "row" : "column") << " direction)";
            }
        }
    }
}

TEST(internal_precond_grid_x_chunk, bsrilu0_general_small_grid_row)
{
    expect_same_result_on_small_grids(factorization::ilu0, rocsparse_direction_row);
}

TEST(internal_precond_grid_x_chunk, bsrilu0_general_small_grid_column)
{
    expect_same_result_on_small_grids(factorization::ilu0, rocsparse_direction_column);
}

TEST(internal_precond_grid_x_chunk, bsric0_general_small_grid_row)
{
    expect_same_result_on_small_grids(factorization::ic0, rocsparse_direction_row);
}

TEST(internal_precond_grid_x_chunk, bsric0_general_small_grid_column)
{
    expect_same_result_on_small_grids(factorization::ic0, rocsparse_direction_column);
}

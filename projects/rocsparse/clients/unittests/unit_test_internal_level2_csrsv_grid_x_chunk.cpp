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
// Device unit tests for the chunk loop in the csrsv analysis and solve kernels
// (library/src/level2/csrsv_device.h and rocsparse_csrsv_solve_kernels.cpp).
//
// Both launch one wavefront per row and clamp grid.x with
// rocsparse::get_grid_size_x. When the clamp binds, every block walks the
// contiguous range of row blocks rocsparse::grid_x_chunk gives it. The clamp
// only binds from about 67 million rows on wave64 devices, so these tests make
// it bind on a small matrix by shrinking handle->properties.maxGridSize[0],
// which get_grid_size_x reads, for the analysis and the solve.
//
// Each row is reduced by one wavefront in a fixed order whatever the grid, so
// the clamped runs must match the full-grid run bit for bit. They are also
// checked against a host triangular solve.
//
// The large case has 100,003 rows, coupled to the row before, so the dependency
// chain is serial and crosses every chunk boundary. With grid.x clamped to 2000
// blocks, every block walks two to four row blocks and several times more
// blocks are launched than a device holds resident. A grid-stride loop would
// hang there, because the first blocks would wait on rows owned by blocks that
// cannot start: run this binary under a timeout.
//

#include "unit_test_utils.hpp"

#include "rocsparse_handle.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

namespace
{
    struct csr_matrix
    {
        rocsparse_int              m;
        std::vector<rocsparse_int> row_ptr;
        std::vector<rocsparse_int> col_ind;
        std::vector<double>        val;
    };

    // Row i couples to rows i - offset and i + offset for every offset, so the
    // lower and upper triangles are mirror images. The diagonal dominates its
    // row and column. missing_diagonal >= 0 drops that row's diagonal entry.
    csr_matrix build_matrix(rocsparse_int                     m,
                            const std::vector<rocsparse_int>& offsets,
                            rocsparse_int                     missing_diagonal = -1)
    {
        csr_matrix A;
        A.m = m;
        A.row_ptr.push_back(0);

        for(rocsparse_int i = 0; i < m; ++i)
        {
            for(auto k = offsets.rbegin(); k != offsets.rend(); ++k)
            {
                if(i - *k >= 0)
                {
                    A.col_ind.push_back(i - *k);
                    A.val.push_back(0.01 * (1 + (i * 7 + *k * 3) % 11));
                }
            }

            if(i != missing_diagonal)
            {
                A.col_ind.push_back(i);
                A.val.push_back(4.0 + i % 5);
            }

            for(rocsparse_int off : offsets)
            {
                if(i + off < m)
                {
                    A.col_ind.push_back(i + off);
                    A.val.push_back(-0.01 * (1 + (i * 5 + off) % 13));
                }
            }

            A.row_ptr.push_back(static_cast<rocsparse_int>(A.col_ind.size()));
        }

        return A;
    }

    // y = op(T)^-1 * alpha * x, with T the triangle of A that fill selects,
    // diagonal included.
    std::vector<double> host_solve(const csr_matrix&          A,
                                   rocsparse_fill_mode        fill,
                                   rocsparse_operation        trans,
                                   double                     alpha,
                                   const std::vector<double>& x)
    {
        const rocsparse_int m = A.m;
        std::vector<double> y(m);
        for(rocsparse_int i = 0; i < m; ++i)
        {
            y[i] = alpha * x[i];
        }

        const bool lower = (fill == rocsparse_fill_mode_lower);
        auto       in_triangle
            = [&](rocsparse_int i, rocsparse_int j) { return lower ? (j < i) : (j > i); };
        auto diagonal = [&](rocsparse_int i) {
            for(rocsparse_int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k)
            {
                if(A.col_ind[k] == i)
                {
                    return A.val[k];
                }
            }
            return 1.0;
        };

        if(trans == rocsparse_operation_none)
        {
            // Row oriented substitution, forward for lower, backward for upper.
            for(rocsparse_int n = 0; n < m; ++n)
            {
                const rocsparse_int i = lower ? n : m - 1 - n;
                for(rocsparse_int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k)
                {
                    if(in_triangle(i, A.col_ind[k]))
                    {
                        y[i] -= A.val[k] * y[A.col_ind[k]];
                    }
                }
                y[i] /= diagonal(i);
            }
        }
        else
        {
            // Column oriented substitution on T^T, backward for lower, forward
            // for upper.
            for(rocsparse_int n = 0; n < m; ++n)
            {
                const rocsparse_int i = lower ? m - 1 - n : n;
                y[i] /= diagonal(i);
                for(rocsparse_int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k)
                {
                    if(in_triangle(i, A.col_ind[k]))
                    {
                        y[A.col_ind[k]] -= A.val[k] * y[i];
                    }
                }
            }
        }

        return y;
    }

    std::vector<double> right_hand_side(rocsparse_int m)
    {
        std::vector<double> x(m);
        for(rocsparse_int i = 0; i < m; ++i)
        {
            x[i] = 1.0 + 0.125 * ((i * 37) % 17);
        }
        return x;
    }

    // Run analysis and solve through the public API. max_grid_x > 0 replaces
    // handle->properties.maxGridSize[0] for both; 0 leaves the device limit.
    void csrsv(const csr_matrix&          A,
               rocsparse_fill_mode        fill,
               rocsparse_operation        trans,
               double                     alpha,
               const std::vector<double>& x,
               int                        max_grid_x,
               std::vector<double>&       y,
               rocsparse_int&             pivot)
    {
        const rocsparse_int nnz = static_cast<rocsparse_int>(A.col_ind.size());

        rocsparse_ut::device_vector<rocsparse_int> d_row_ptr(A.row_ptr);
        rocsparse_ut::device_vector<rocsparse_int> d_col_ind(A.col_ind);
        rocsparse_ut::device_vector<double>        d_val(A.val);
        rocsparse_ut::device_vector<double>        d_x(x);
        rocsparse_ut::device_vector<double>        d_y(static_cast<size_t>(A.m));
        ASSERT_NE(d_row_ptr.ptr, nullptr);
        ASSERT_NE(d_col_ind.ptr, nullptr);
        ASSERT_NE(d_val.ptr, nullptr);
        ASSERT_NE(d_x.ptr, nullptr);
        ASSERT_NE(d_y.ptr, nullptr);

        rocsparse_handle    handle = nullptr;
        rocsparse_mat_descr descr  = nullptr;
        rocsparse_mat_info  info   = nullptr;
        ASSERT_EQ(rocsparse_create_handle(&handle), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_fill_mode(descr, fill), rocsparse_status_success);

        if(max_grid_x > 0)
        {
            handle->properties.maxGridSize[0] = max_grid_x;
        }

        size_t buffer_size = 0;
        ASSERT_EQ(rocsparse_dcsrsv_buffer_size(handle,
                                               trans,
                                               A.m,
                                               nnz,
                                               descr,
                                               d_val.ptr,
                                               d_row_ptr.ptr,
                                               d_col_ind.ptr,
                                               info,
                                               &buffer_size),
                  rocsparse_status_success);

        rocsparse_ut::device_vector<char> d_buffer(buffer_size > 0 ? buffer_size : 1);
        ASSERT_NE(d_buffer.ptr, nullptr);

        ASSERT_EQ(rocsparse_dcsrsv_analysis(handle,
                                            trans,
                                            A.m,
                                            nnz,
                                            descr,
                                            d_val.ptr,
                                            d_row_ptr.ptr,
                                            d_col_ind.ptr,
                                            info,
                                            rocsparse_analysis_policy_force,
                                            rocsparse_solve_policy_auto,
                                            d_buffer.ptr),
                  rocsparse_status_success);

        ASSERT_EQ(rocsparse_dcsrsv_solve(handle,
                                         trans,
                                         A.m,
                                         nnz,
                                         &alpha,
                                         descr,
                                         d_val.ptr,
                                         d_row_ptr.ptr,
                                         d_col_ind.ptr,
                                         info,
                                         d_x.ptr,
                                         d_y.ptr,
                                         rocsparse_solve_policy_auto,
                                         d_buffer.ptr),
                  rocsparse_status_success);

        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        pivot                         = -1;
        const rocsparse_status status = rocsparse_csrsv_zero_pivot(handle, descr, info, &pivot);
        EXPECT_TRUE(status == rocsparse_status_success || status == rocsparse_status_zero_pivot);

        y = rocsparse_ut::to_host(d_y);

        EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_handle(handle), rocsparse_status_success);
    }

    const char* label(rocsparse_fill_mode fill, rocsparse_operation trans)
    {
        const bool lower = (fill == rocsparse_fill_mode_lower);
        return (trans == rocsparse_operation_none) ? (lower ? "lower N" : "upper N")
                                                   : (lower ? "lower T" : "upper T");
    }

    void expect_same_result_on_small_grids(const csr_matrix&       A,
                                           rocsparse_fill_mode     fill,
                                           rocsparse_operation     trans,
                                           const std::vector<int>& grids)
    {
        const double              alpha = 1.5;
        const std::vector<double> x     = right_hand_side(A.m);
        const std::vector<double> ref   = host_solve(A, fill, trans, alpha, x);

        std::vector<double> full;
        rocsparse_int       full_pivot = 0;
        ASSERT_NO_FATAL_FAILURE(csrsv(A, fill, trans, alpha, x, 0, full, full_pivot));
        ASSERT_EQ(full_pivot, -1) << label(fill, trans) << " full grid";
        ASSERT_EQ(full.size(), ref.size());

        for(rocsparse_int i = 0; i < A.m; ++i)
        {
            ASSERT_NEAR(full[i], ref[i], 1e-12 * (1.0 + std::abs(ref[i])))
                << label(fill, trans) << " full grid, row " << i;
        }

        for(int grid : grids)
        {
            std::vector<double> got;
            rocsparse_int       pivot = 0;
            ASSERT_NO_FATAL_FAILURE(csrsv(A, fill, trans, alpha, x, grid, got, pivot));
            EXPECT_EQ(pivot, -1) << label(fill, trans) << " grid.x " << grid;
            ASSERT_EQ(got.size(), full.size());

            // Report the first differing row rather than every entry, so a
            // failure names the chunk that was skipped or computed early.
            for(rocsparse_int i = 0; i < A.m; ++i)
            {
                ASSERT_EQ(std::memcmp(&got[i], &full[i], sizeof(double)), 0)
                    << label(fill, trans) << ": row " << i << " is " << got[i]
                    << " with grid.x clamped to " << grid << " and " << full[i]
                    << " on the full grid (" << A.m << " rows)";
            }
        }
    }

    // 1000 rows do not divide evenly into row blocks of 16 or 32 rows, nor the
    // row blocks into 2, 3 or 7 chunks. Without an offset of 1, rows share
    // levels, so the solve walks a row map that is not the identity.
    const std::vector<int> small_grids = {1, 2, 3, 7};

    csr_matrix small_matrix()
    {
        return build_matrix(1000, {2, 3, 7, 23, 41});
    }

    csr_matrix large_matrix()
    {
        return build_matrix(100003, {1, 17, 300});
    }
}

#define CSRSV_GRID_TESTS(NAME, FILL, TRANS)                                                     \
    TEST(internal_level2_csrsv_grid_x_chunk, NAME##_small_grids)                                \
    {                                                                                           \
        expect_same_result_on_small_grids(small_matrix(), FILL, TRANS, small_grids);            \
    }                                                                                           \
    TEST(internal_level2_csrsv_grid_x_chunk, NAME##_more_blocks_than_resident)                  \
    {                                                                                           \
        expect_same_result_on_small_grids(large_matrix(), FILL, TRANS, std::vector<int>{2000}); \
    }

CSRSV_GRID_TESTS(lower_non_transpose, rocsparse_fill_mode_lower, rocsparse_operation_none)
CSRSV_GRID_TESTS(upper_non_transpose, rocsparse_fill_mode_upper, rocsparse_operation_none)
CSRSV_GRID_TESTS(lower_transpose, rocsparse_fill_mode_lower, rocsparse_operation_transpose)
CSRSV_GRID_TESTS(upper_transpose, rocsparse_fill_mode_upper, rocsparse_operation_transpose)
#undef CSRSV_GRID_TESTS

// A structural zero pivot found by the chunked analysis, in a chunk other than
// the first, for both analysis kernels.
TEST(internal_level2_csrsv_grid_x_chunk, zero_pivot_on_small_grid)
{
    const rocsparse_int       missing = 613;
    const csr_matrix          A       = build_matrix(1000, {2, 3, 7, 23, 41}, missing);
    const std::vector<double> x       = right_hand_side(A.m);

    for(rocsparse_fill_mode fill : {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper})
    {
        for(rocsparse_operation trans : {rocsparse_operation_none, rocsparse_operation_transpose})
        {
            for(int grid : small_grids)
            {
                std::vector<double> y;
                rocsparse_int       pivot = -1;
                ASSERT_NO_FATAL_FAILURE(csrsv(A, fill, trans, 1.0, x, grid, y, pivot));
                EXPECT_EQ(pivot, missing) << label(fill, trans) << " grid.x " << grid;
            }
        }
    }
}

// rocsparse_spsv with 64-bit row offsets and column indices, which takes the
// csrsv kernels instantiated for int64_t indices, on a clamped grid.
TEST(internal_level2_csrsv_grid_x_chunk, spsv_int64_small_grids)
{
    const csr_matrix          A     = small_matrix();
    const double              alpha = 1.5;
    const std::vector<double> x     = right_hand_side(A.m);

    std::vector<int64_t> row_ptr(A.row_ptr.begin(), A.row_ptr.end());
    std::vector<int64_t> col_ind(A.col_ind.begin(), A.col_ind.end());

    rocsparse_ut::device_vector<int64_t> d_row_ptr(row_ptr);
    rocsparse_ut::device_vector<int64_t> d_col_ind(col_ind);
    rocsparse_ut::device_vector<double>  d_val(A.val);
    rocsparse_ut::device_vector<double>  d_x(x);
    rocsparse_ut::device_vector<double>  d_y(static_cast<size_t>(A.m));
    ASSERT_NE(d_row_ptr.ptr, nullptr);
    ASSERT_NE(d_col_ind.ptr, nullptr);
    ASSERT_NE(d_val.ptr, nullptr);
    ASSERT_NE(d_x.ptr, nullptr);
    ASSERT_NE(d_y.ptr, nullptr);

    for(rocsparse_fill_mode fill : {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper})
    {
        for(rocsparse_operation trans : {rocsparse_operation_none, rocsparse_operation_transpose})
        {
            const std::vector<double> ref = host_solve(A, fill, trans, alpha, x);

            for(int grid : small_grids)
            {
                rocsparse_handle handle = nullptr;
                ASSERT_EQ(rocsparse_create_handle(&handle), rocsparse_status_success);
                handle->properties.maxGridSize[0] = grid;

                rocsparse_spmat_descr mat = nullptr;
                rocsparse_dnvec_descr vx  = nullptr;
                rocsparse_dnvec_descr vy  = nullptr;
                ASSERT_EQ(rocsparse_create_csr_descr(&mat,
                                                     A.m,
                                                     A.m,
                                                     static_cast<int64_t>(A.col_ind.size()),
                                                     d_row_ptr.ptr,
                                                     d_col_ind.ptr,
                                                     d_val.ptr,
                                                     rocsparse_indextype_i64,
                                                     rocsparse_indextype_i64,
                                                     rocsparse_index_base_zero,
                                                     rocsparse_datatype_f64_r),
                          rocsparse_status_success);
                ASSERT_EQ(rocsparse_spmat_set_attribute(
                              mat, rocsparse_spmat_fill_mode, &fill, sizeof(fill)),
                          rocsparse_status_success);
                ASSERT_EQ(rocsparse_create_dnvec_descr(&vx, A.m, d_x.ptr, rocsparse_datatype_f64_r),
                          rocsparse_status_success);
                ASSERT_EQ(rocsparse_create_dnvec_descr(&vy, A.m, d_y.ptr, rocsparse_datatype_f64_r),
                          rocsparse_status_success);

                size_t buffer_size = 0;
                ASSERT_EQ(rocsparse_spsv(handle,
                                         trans,
                                         &alpha,
                                         mat,
                                         vx,
                                         vy,
                                         rocsparse_datatype_f64_r,
                                         rocsparse_spsv_alg_default,
                                         rocsparse_spsv_stage_buffer_size,
                                         &buffer_size,
                                         nullptr),
                          rocsparse_status_success);
                rocsparse_ut::device_vector<char> d_buffer(buffer_size > 0 ? buffer_size : 1);
                ASSERT_NE(d_buffer.ptr, nullptr);
                ASSERT_EQ(rocsparse_spsv(handle,
                                         trans,
                                         &alpha,
                                         mat,
                                         vx,
                                         vy,
                                         rocsparse_datatype_f64_r,
                                         rocsparse_spsv_alg_default,
                                         rocsparse_spsv_stage_preprocess,
                                         &buffer_size,
                                         d_buffer.ptr),
                          rocsparse_status_success);
                ASSERT_EQ(rocsparse_spsv(handle,
                                         trans,
                                         &alpha,
                                         mat,
                                         vx,
                                         vy,
                                         rocsparse_datatype_f64_r,
                                         rocsparse_spsv_alg_default,
                                         rocsparse_spsv_stage_compute,
                                         &buffer_size,
                                         d_buffer.ptr),
                          rocsparse_status_success);
                ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

                const std::vector<double> got = rocsparse_ut::to_host(d_y);
                ASSERT_EQ(got.size(), ref.size());
                for(rocsparse_int i = 0; i < A.m; ++i)
                {
                    ASSERT_NEAR(got[i], ref[i], 1e-12 * (1.0 + std::abs(ref[i])))
                        << "spsv i64 " << label(fill, trans) << " grid.x " << grid << ", row " << i;
                }

                EXPECT_EQ(rocsparse_destroy_dnvec_descr(vy), rocsparse_status_success);
                EXPECT_EQ(rocsparse_destroy_dnvec_descr(vx), rocsparse_status_success);
                EXPECT_EQ(rocsparse_destroy_spmat_descr(mat), rocsparse_status_success);
                EXPECT_EQ(rocsparse_destroy_handle(handle), rocsparse_status_success);
            }
        }
    }
}

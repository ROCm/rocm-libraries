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
// checked against a host triangular solve. The solution is the same for any
// valid row order, so max_nnz, diag_ind and the row map from the analysis are
// compared with the full-grid run as well.
//
// The large case has 100,003 rows, coupled to the row before, so the dependency
// chain is serial and crosses every chunk boundary. It has 3126 row blocks on
// wave32 devices and 6251 on wave64 devices. With grid.x clamped to 2000
// blocks, blocks 0 to 1562 walk 2 (wave32) or 4 (wave64) row blocks and blocks
// 1563 and up get an empty range, and several times more blocks are launched
// than a device holds resident. A grid-stride loop would hang there, because
// the first blocks would wait on rows owned by blocks that cannot start: run
// this binary under a timeout.
//
// The output vector is filled with NaN before every solve, so a row that a
// clamped run skips cannot inherit the value of an earlier run.
//

#include "unit_test_utils.hpp"

#include "rocsparse_handle.hpp"
#include "rocsparse_mat_info.hpp"
#include "rocsparse_trm_info.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace
{
    // Owns a handle or descriptor, so an ASSERT that returns early does not
    // leak it.
    template <typename H, auto DESTROY>
    struct guard
    {
        H obj = nullptr;

        guard()             = default;
        guard(const guard&) = delete;
        guard& operator=(const guard&) = delete;
        ~guard()
        {
            if(obj != nullptr)
            {
                EXPECT_EQ(DESTROY(obj), rocsparse_status_success);
            }
        }

        operator H() const
        {
            return obj;
        }
        H operator->() const
        {
            return obj;
        }
    };

    using handle_guard       = guard<rocsparse_handle, rocsparse_destroy_handle>;
    using mat_descr_guard    = guard<rocsparse_mat_descr, rocsparse_destroy_mat_descr>;
    using mat_info_guard     = guard<rocsparse_mat_info, rocsparse_destroy_mat_info>;
    using spmat_guard        = guard<rocsparse_spmat_descr, rocsparse_destroy_spmat_descr>;
    using dnvec_guard        = guard<rocsparse_dnvec_descr, rocsparse_destroy_dnvec_descr>;
    using sptrsv_descr_guard = guard<rocsparse_sptrsv_descr, rocsparse_destroy_sptrsv_descr>;

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

    // What the analysis leaves for the solve and for csrilu0 and csric0. The
    // solve reads only the row map, csrilu0 and csric0 read diag_ind, and
    // max_nnz selects their kernel.
    struct analysis_data
    {
        int64_t                    max_nnz = -1;
        std::vector<rocsparse_int> diag_ind;
        std::vector<rocsparse_int> row_map;
    };

    // Index of the first entry where a and b differ, or -1.
    int64_t first_difference(const std::vector<rocsparse_int>& a,
                             const std::vector<rocsparse_int>& b)
    {
        for(size_t i = 0; i < a.size() || i < b.size(); ++i)
        {
            if(i >= a.size() || i >= b.size() || a[i] != b[i])
            {
                return static_cast<int64_t>(i);
            }
        }
        return -1;
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
               rocsparse_int&             pivot,
               analysis_data&             analysis)
    {
        const rocsparse_int nnz = static_cast<rocsparse_int>(A.col_ind.size());

        rocsparse_ut::device_vector<rocsparse_int> d_row_ptr(A.row_ptr);
        rocsparse_ut::device_vector<rocsparse_int> d_col_ind(A.col_ind);
        rocsparse_ut::device_vector<double>        d_val(A.val);
        rocsparse_ut::device_vector<double>        d_x(x);
        rocsparse_ut::device_vector<double>        d_y(
            std::vector<double>(A.m, std::numeric_limits<double>::quiet_NaN()));

        handle_guard    handle;
        mat_descr_guard descr;
        mat_info_guard  info;
        ASSERT_EQ(rocsparse_create_handle(&handle.obj), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_mat_descr(&descr.obj), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_mat_info(&info.obj), rocsparse_status_success);
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

        const rocsparse::trm_info_t* trm = info->get_csrsv_info(trans, fill);
        ASSERT_NE(trm, nullptr);
        analysis.max_nnz  = trm->get_max_nnz();
        analysis.diag_ind = rocsparse_ut::to_host(
            static_cast<const rocsparse_int*>(trm->get_diag_ind()), static_cast<size_t>(A.m));
        analysis.row_map = rocsparse_ut::to_host(
            static_cast<const rocsparse_int*>(trm->get_row_map()), static_cast<size_t>(A.m));

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
        analysis_data       full_analysis;
        ASSERT_NO_FATAL_FAILURE(
            csrsv(A, fill, trans, alpha, x, 0, full, full_pivot, full_analysis));
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
            analysis_data       analysis;
            ASSERT_NO_FATAL_FAILURE(csrsv(A, fill, trans, alpha, x, grid, got, pivot, analysis));
            EXPECT_EQ(pivot, -1) << label(fill, trans) << " grid.x " << grid;
            EXPECT_EQ(analysis.max_nnz, full_analysis.max_nnz)
                << label(fill, trans) << ": max_nnz with grid.x clamped to " << grid;
            EXPECT_EQ(first_difference(analysis.diag_ind, full_analysis.diag_ind), -1)
                << label(fill, trans) << ": first differing diag_ind entry with grid.x clamped to "
                << grid;
            EXPECT_EQ(first_difference(analysis.row_map, full_analysis.row_map), -1)
                << label(fill, trans) << ": first differing row map entry with grid.x clamped to "
                << grid;
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

    // Row blocks of the 1000-row matrix: 32 of 32 rows on wave32 devices, 63 of
    // 16 rows on wave64 devices, the last one partial either way. They split
    // unevenly into some of the 2, 3 and 7 chunks (3 and 7 on wave32, 2 on
    // wave64). Without an offset of 1, rows share levels, so the solve walks a
    // row map that is not the identity.
    const std::vector<int> small_grids = {1, 2, 3, 7};

    rocsparse_int row_blocks(rocsparse_int m)
    {
        handle_guard handle;
        EXPECT_EQ(rocsparse_create_handle(&handle.obj), rocsparse_status_success);
        const rocsparse_int rows = 1024 / handle->wavefront_size;
        return (m - 1) / rows + 1;
    }

    // The small grids plus the edge where the launch switches kernels: one block
    // per row block runs the unchunked kernel, one fewer the chunked kernel with
    // a chunk of 2 row blocks for the first blocks and empty ranges at the end.
    std::vector<int> small_and_edge_grids(rocsparse_int m)
    {
        std::vector<int> grids = small_grids;
        grids.push_back(row_blocks(m) - 1);
        grids.push_back(row_blocks(m));
        return grids;
    }

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
        expect_same_result_on_small_grids(                                                      \
            small_matrix(), FILL, TRANS, small_and_edge_grids(1000));                           \
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

// A structural zero pivot found by the chunked analysis in a chunk other than
// the first, for both analysis kernels. The lower kernel numbers row blocks
// from the first row and the upper kernel from the last, so each gets its own
// missing diagonal: row 613 and row 386 lie in row block 19 (wave32) or 38
// (wave64) of their kernel, past the first chunk for grid.x 2, 3 and 7.
TEST(internal_level2_csrsv_grid_x_chunk, zero_pivot_on_small_grid)
{
    for(rocsparse_fill_mode fill : {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper})
    {
        for(rocsparse_operation trans : {rocsparse_operation_none, rocsparse_operation_transpose})
        {
            // Lower non-transposed and upper transposed take the lower kernel.
            const bool lower_kernel
                = (fill == rocsparse_fill_mode_lower) == (trans == rocsparse_operation_none);
            const rocsparse_int       missing = lower_kernel ? 613 : 1000 - 1 - 613;
            const csr_matrix          A       = build_matrix(1000, {2, 3, 7, 23, 41}, missing);
            const std::vector<double> x       = right_hand_side(A.m);

            for(int grid : {2, 3, 7})
            {
                std::vector<double> y;
                rocsparse_int       pivot = -1;
                analysis_data       analysis;
                ASSERT_NO_FATAL_FAILURE(csrsv(A, fill, trans, 1.0, x, grid, y, pivot, analysis));
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
    const std::vector<double>            nan(A.m, std::numeric_limits<double>::quiet_NaN());

    for(rocsparse_fill_mode fill : {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper})
    {
        for(rocsparse_operation trans : {rocsparse_operation_none, rocsparse_operation_transpose})
        {
            const std::vector<double> ref = host_solve(A, fill, trans, alpha, x);

            for(int grid : small_grids)
            {
                handle_guard handle;
                ASSERT_EQ(rocsparse_create_handle(&handle.obj), rocsparse_status_success);
                handle->properties.maxGridSize[0] = grid;

                spmat_guard mat;
                dnvec_guard vx;
                dnvec_guard vy;
                ASSERT_EQ(rocsparse_create_csr_descr(&mat.obj,
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
                ASSERT_EQ(
                    rocsparse_create_dnvec_descr(&vx.obj, A.m, d_x.ptr, rocsparse_datatype_f64_r),
                    rocsparse_status_success);
                ASSERT_EQ(
                    rocsparse_create_dnvec_descr(&vy.obj, A.m, d_y.ptr, rocsparse_datatype_f64_r),
                    rocsparse_status_success);

                ASSERT_EQ(
                    hipMemcpy(d_y.ptr, nan.data(), sizeof(double) * A.m, hipMemcpyHostToDevice),
                    hipSuccess);

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
            }
        }
    }
}

namespace
{
    // Batched rocsparse_sptrsv on CSR: batch_count instances of the same
    // pattern, instance b with its values scaled by 1 + b. max_grid_x > 0
    // replaces handle->properties.maxGridSize[0] for the analysis and the solve.
    void sptrsv_batched(const csr_matrix&          A,
                        int64_t                    batch_count,
                        double                     alpha,
                        const std::vector<double>& x,
                        int                        max_grid_x,
                        std::vector<double>&       y)
    {
        const int64_t m   = A.m;
        const int64_t nnz = static_cast<int64_t>(A.col_ind.size());

        std::vector<rocsparse_int> row_ptr;
        std::vector<rocsparse_int> col_ind;
        std::vector<double>        val;
        for(int64_t b = 0; b < batch_count; ++b)
        {
            row_ptr.insert(row_ptr.end(), A.row_ptr.begin(), A.row_ptr.end());
            col_ind.insert(col_ind.end(), A.col_ind.begin(), A.col_ind.end());
            for(double v : A.val)
            {
                val.push_back(v * (1.0 + b));
            }
        }

        rocsparse_ut::device_vector<rocsparse_int> d_row_ptr(row_ptr);
        rocsparse_ut::device_vector<rocsparse_int> d_col_ind(col_ind);
        rocsparse_ut::device_vector<double>        d_val(val);
        rocsparse_ut::device_vector<double>        d_x(x);
        rocsparse_ut::device_vector<double>        d_y(
            std::vector<double>(m * batch_count, std::numeric_limits<double>::quiet_NaN()));

        handle_guard handle;
        ASSERT_EQ(rocsparse_create_handle(&handle.obj), rocsparse_status_success);
        if(max_grid_x > 0)
        {
            handle->properties.maxGridSize[0] = max_grid_x;
        }

        spmat_guard mat;
        dnvec_guard vx;
        dnvec_guard vy;
        ASSERT_EQ(rocsparse_create_csr_descr(&mat.obj,
                                             m,
                                             m,
                                             nnz,
                                             d_row_ptr.ptr,
                                             d_col_ind.ptr,
                                             d_val.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f64_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_csr_set_strided_batch(mat, batch_count, m + 1, nnz),
                  rocsparse_status_success);
        const rocsparse_fill_mode fill = rocsparse_fill_mode_lower;
        ASSERT_EQ(
            rocsparse_spmat_set_attribute(mat, rocsparse_spmat_fill_mode, &fill, sizeof(fill)),
            rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vx.obj, m, d_x.ptr, rocsparse_datatype_f64_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_dnvec_set_strided_batch(vx, batch_count, m), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&vy.obj, m, d_y.ptr, rocsparse_datatype_f64_r),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_dnvec_set_strided_batch(vy, batch_count, m), rocsparse_status_success);

        sptrsv_descr_guard descr;
        ASSERT_EQ(rocsparse_create_sptrsv_descr(&descr.obj), rocsparse_status_success);
        const rocsparse_sptrsv_alg      alg  = rocsparse_sptrsv_alg_default;
        const rocsparse_operation       op   = rocsparse_operation_none;
        const rocsparse_datatype        type = rocsparse_datatype_f64_r;
        const rocsparse_analysis_policy apol = rocsparse_analysis_policy_force;
        ASSERT_EQ(rocsparse_sptrsv_set_input(
                      handle, descr, rocsparse_sptrsv_input_alg, &alg, sizeof(alg), nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_sptrsv_set_input(
                      handle, descr, rocsparse_sptrsv_input_operation, &op, sizeof(op), nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_sptrsv_set_input(handle,
                                             descr,
                                             rocsparse_sptrsv_input_scalar_datatype,
                                             &type,
                                             sizeof(type),
                                             nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_sptrsv_set_input(handle,
                                             descr,
                                             rocsparse_sptrsv_input_compute_datatype,
                                             &type,
                                             sizeof(type),
                                             nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_sptrsv_set_input(handle,
                                             descr,
                                             rocsparse_sptrsv_input_analysis_policy,
                                             &apol,
                                             sizeof(apol),
                                             nullptr),
                  rocsparse_status_success);

        for(rocsparse_sptrsv_stage stage :
            {rocsparse_sptrsv_stage_analysis, rocsparse_sptrsv_stage_compute})
        {
            if(stage == rocsparse_sptrsv_stage_compute)
            {
                ASSERT_EQ(rocsparse_sptrsv_set_input(handle,
                                                     descr,
                                                     rocsparse_sptrsv_input_scalar_alpha,
                                                     &alpha,
                                                     sizeof(&alpha),
                                                     nullptr),
                          rocsparse_status_success);
            }
            size_t buffer_size = 0;
            ASSERT_EQ(rocsparse_sptrsv_buffer_size(
                          handle, descr, mat, vx, vy, stage, &buffer_size, nullptr),
                      rocsparse_status_success);
            rocsparse_ut::device_vector<char> d_buffer(buffer_size > 0 ? buffer_size : 1);
            ASSERT_EQ(rocsparse_sptrsv(handle,
                                       descr,
                                       mat,
                                       vx,
                                       vy,
                                       stage,
                                       buffer_size,
                                       buffer_size > 0 ? d_buffer.ptr : nullptr,
                                       nullptr),
                      rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        }

        y = rocsparse_ut::to_host(d_y);
    }
}

// Batched rocsparse_sptrsv on a clamped grid: the chunked solve with grid.y
// over the batch instances, each with its own values, right-hand side, done
// flags and singularity position.
TEST(internal_level2_csrsv_grid_x_chunk, sptrsv_batched_small_grids)
{
    const csr_matrix    A           = small_matrix();
    const int64_t       batch_count = 3;
    const double        alpha       = 1.5;
    std::vector<double> x;
    for(int64_t b = 0; b < batch_count; ++b)
    {
        const std::vector<double> xb = right_hand_side(A.m);
        for(double v : xb)
        {
            x.push_back(v * (1.0 + 0.5 * b));
        }
    }

    std::vector<double> full;
    ASSERT_NO_FATAL_FAILURE(sptrsv_batched(A, batch_count, alpha, x, 0, full));
    ASSERT_EQ(full.size(), x.size());

    for(int64_t b = 0; b < batch_count; ++b)
    {
        csr_matrix Ab = A;
        for(double& v : Ab.val)
        {
            v *= 1.0 + b;
        }
        const std::vector<double> xb(x.begin() + b * A.m, x.begin() + (b + 1) * A.m);
        const std::vector<double> ref
            = host_solve(Ab, rocsparse_fill_mode_lower, rocsparse_operation_none, alpha, xb);
        for(rocsparse_int i = 0; i < A.m; ++i)
        {
            ASSERT_NEAR(full[b * A.m + i], ref[i], 1e-12 * (1.0 + std::abs(ref[i])))
                << "full grid, batch " << b << ", row " << i;
        }
    }

    for(int grid : {2, row_blocks(A.m) - 1})
    {
        std::vector<double> got;
        ASSERT_NO_FATAL_FAILURE(sptrsv_batched(A, batch_count, alpha, x, grid, got));
        ASSERT_EQ(got.size(), full.size());
        for(size_t k = 0; k < full.size(); ++k)
        {
            ASSERT_EQ(std::memcmp(&got[k], &full[k], sizeof(double)), 0)
                << "batch " << k / A.m << ", row " << k % A.m << " is " << got[k]
                << " with grid.x clamped to " << grid << " and " << full[k] << " on the full grid";
        }
    }
}

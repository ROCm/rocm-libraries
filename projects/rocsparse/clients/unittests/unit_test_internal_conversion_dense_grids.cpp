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
// Forced-clamp tests for the dense conversion paths (AISPARSE-687).
//
// FOCUS. rocsparse_dense_to_sparse to CSR or CSC first counts the non-zeros
// with nnz_kernel_row or nnz_kernel_col, then fills with dense2csr_kernel or
// dense2csc_kernel. rocsparse_Xcoo2dense zero fills and scatters with
// coo2dense_kernel. Every one of these launches clamps grid.x with
// rocsparse::get_grid_size_x and must grid-stride over whatever the clamp
// drops.
//
// WHY NOT TEST THE REAL THRESHOLD. nnz_kernel_col runs one 256-thread block per
// column, so the clamp only binds past 16,777,215 columns. The tests shrink
// handle->properties.maxGridSize[0] with ScopedMaxGridSizeX to 1, 3 and 7
// blocks instead, so a 3000 x 700 matrix takes the clamped, looping path of
// every launch above. The memory-guarded dense_to_sparse_bell_extra case covers
// a real clamp.
//
// WHAT MAKES EACH CASE LOAD-BEARING. A launch that does not grid-stride on the
// clamped grid leaves the tail of the per-row (per-column) counts, the index
// arrays or the dense matrix unwritten. All outputs are poisoned first so that
// shows. An unclamped run is the control.
//
// TARGET: rocsparse-unit-test-device.
//
#include "unit_test_utils.hpp"

// ScopedMaxGridSizeX: shrinks handle->properties.maxGridSize[0], the limit
// get_grid_size_x clamps grid.x against.
#include "unit_test_grid_clamp.hpp"

#include "rocsparse.h"

#include <cstdint>
#include <memory>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    // grid.x limits the tests shrink maxGridSize[0] to. 0 means the device limit.
    constexpr int limits[] = {0, 1, 3, 7};

    constexpr rocsparse_int m = 3000;
    constexpr rocsparse_int n = 700;

    bool is_nonzero(int64_t i, int64_t j)
    {
        return (i * 7 + j * 3) % 5 == 0;
    }

    template <typename T>
    T value_at(int64_t i, int64_t j)
    {
        return scalar<T>(static_cast<float>(1 + (i + 2 * j) % 9));
    }

    // Dense m x n matrix with leading dimension ld in the given order.
    template <typename T>
    std::vector<T> make_dense(rocsparse_order order, int64_t ld)
    {
        const int64_t  outer = (order == rocsparse_order_column) ? n : m;
        std::vector<T> a(static_cast<size_t>(ld * outer), scalar<T>(0.0f));
        for(int64_t i = 0; i < m; ++i)
        {
            for(int64_t j = 0; j < n; ++j)
            {
                if(is_nonzero(i, j))
                {
                    const int64_t e = (order == rocsparse_order_column) ? i + ld * j : j + ld * i;
                    a[e]            = value_at<T>(i, j);
                }
            }
        }
        return a;
    }

    template <typename T>
    struct HostCsx
    {
        std::vector<rocsparse_int> ptr;
        std::vector<rocsparse_int> ind;
        std::vector<T>             val;
    };

    // Compressed rows (csr == true) or columns of the dense matrix, zero based.
    template <typename T>
    HostCsx<T> make_csx(bool csr)
    {
        const int64_t outer = csr ? m : n;
        const int64_t inner = csr ? n : m;
        HostCsx<T>    c;
        c.ptr.push_back(0);
        for(int64_t o = 0; o < outer; ++o)
        {
            for(int64_t k = 0; k < inner; ++k)
            {
                const int64_t i = csr ? o : k;
                const int64_t j = csr ? k : o;
                if(is_nonzero(i, j))
                {
                    c.ind.push_back(static_cast<rocsparse_int>(k));
                    c.val.push_back(value_at<T>(i, j));
                }
            }
            c.ptr.push_back(static_cast<rocsparse_int>(c.ind.size()));
        }
        return c;
    }

    template <typename T>
    void check_dense_to_sparse(rocsparse_handle handle, bool csr, rocsparse_order order)
    {
        const int64_t        ld    = (order == rocsparse_order_column) ? m + 3 : n + 3;
        const std::vector<T> h_A   = make_dense<T>(order, ld);
        const HostCsx<T>     want  = make_csx<T>(csr);
        const int64_t        outer = csr ? m : n;
        const int64_t        nnz   = want.ptr.back();

        // Every limit must clamp the column-count launch, one block per column.
        ASSERT_GT(n, limits[3]);

        device_vector<T> d_A(h_A);

        for(const int limit : limits)
        {
            std::unique_ptr<ScopedMaxGridSizeX> clamp;
            if(limit > 0)
            {
                clamp.reset(new ScopedMaxGridSizeX(handle, limit));
            }

            device_vector<rocsparse_int> d_ptr(static_cast<size_t>(outer + 1));
            device_vector<rocsparse_int> d_ind(static_cast<size_t>(nnz));
            device_vector<T>             d_val(static_cast<size_t>(nnz));
            UT_CHECK_HIP(hipMemset(d_ptr.ptr, 0xFF, d_ptr.n * sizeof(rocsparse_int)));
            UT_CHECK_HIP(hipMemset(d_ind.ptr, 0xFF, d_ind.n * sizeof(rocsparse_int)));
            UT_CHECK_HIP(hipMemset(d_val.ptr, 0xFF, d_val.n * sizeof(T)));

            rocsparse_dnmat_descr mat_A = nullptr;
            rocsparse_spmat_descr mat_B = nullptr;
            ASSERT_EQ(rocsparse_create_dnmat_descr(&mat_A, m, n, ld, d_A.ptr, dt_of<T>(), order),
                      rocsparse_status_success);
            if(csr)
            {
                ASSERT_EQ(rocsparse_create_csr_descr(&mat_B,
                                                     m,
                                                     n,
                                                     0,
                                                     d_ptr.ptr,
                                                     nullptr,
                                                     nullptr,
                                                     rocsparse_indextype_i32,
                                                     rocsparse_indextype_i32,
                                                     rocsparse_index_base_zero,
                                                     dt_of<T>()),
                          rocsparse_status_success);
            }
            else
            {
                ASSERT_EQ(rocsparse_create_csc_descr(&mat_B,
                                                     m,
                                                     n,
                                                     0,
                                                     d_ptr.ptr,
                                                     nullptr,
                                                     nullptr,
                                                     rocsparse_indextype_i32,
                                                     rocsparse_indextype_i32,
                                                     rocsparse_index_base_zero,
                                                     dt_of<T>()),
                          rocsparse_status_success);
            }

            const rocsparse_dense_to_sparse_alg alg = rocsparse_dense_to_sparse_alg_default;

            size_t buffer_size = 0;
            EXPECT_EQ(rocsparse_dense_to_sparse(handle, mat_A, mat_B, alg, &buffer_size, nullptr),
                      rocsparse_status_success);
            device_vector<char> d_buffer(buffer_size);
            UT_CHECK_HIP(hipMemset(d_buffer.ptr, 0xFF, buffer_size));
            EXPECT_EQ(rocsparse_dense_to_sparse(handle, mat_A, mat_B, alg, nullptr, d_buffer.ptr),
                      rocsparse_status_success);

            int64_t rows = 0, cols = 0, got_nnz = 0;
            EXPECT_EQ(rocsparse_spmat_get_size(mat_B, &rows, &cols, &got_nnz),
                      rocsparse_status_success);
            EXPECT_EQ(got_nnz, nnz) << "grid.x clamped to " << limit << " blocks";

            if(got_nnz == nnz)
            {
                if(csr)
                {
                    EXPECT_EQ(rocsparse_csr_set_pointers(mat_B, d_ptr.ptr, d_ind.ptr, d_val.ptr),
                              rocsparse_status_success);
                }
                else
                {
                    EXPECT_EQ(rocsparse_csc_set_pointers(mat_B, d_ptr.ptr, d_ind.ptr, d_val.ptr),
                              rocsparse_status_success);
                }
                EXPECT_EQ(rocsparse_dense_to_sparse(
                              handle, mat_A, mat_B, alg, &buffer_size, d_buffer.ptr),
                          rocsparse_status_success);
                UT_CHECK_HIP(hipDeviceSynchronize());

                EXPECT_EQ(to_host(d_ptr), want.ptr) << "grid.x clamped to " << limit << " blocks";
                EXPECT_EQ(to_host(d_ind), want.ind) << "grid.x clamped to " << limit << " blocks";
                EXPECT_TRUE(to_host(d_val) == want.val)
                    << "grid.x clamped to " << limit << " blocks";
            }

            EXPECT_EQ(rocsparse_destroy_spmat_descr(mat_B), rocsparse_status_success);
            EXPECT_EQ(rocsparse_destroy_dnmat_descr(mat_A), rocsparse_status_success);
        }
    }

    using DenseConversionGrids = HandleTest;
}

TEST_F(DenseConversionGrids, csr_column_f32)
{
    check_dense_to_sparse<float>(handle, true, rocsparse_order_column);
}

TEST_F(DenseConversionGrids, csr_row_c64)
{
    check_dense_to_sparse<rocsparse_double_complex>(handle, true, rocsparse_order_row);
}

TEST_F(DenseConversionGrids, csc_column_f64)
{
    check_dense_to_sparse<double>(handle, false, rocsparse_order_column);
}

TEST_F(DenseConversionGrids, csc_row_f32)
{
    check_dense_to_sparse<float>(handle, false, rocsparse_order_row);
}

TEST_F(DenseConversionGrids, coo2dense)
{
    const rocsparse_int ld = m + 3;

    std::vector<rocsparse_int> h_row, h_col;
    std::vector<float>         h_val;
    for(int64_t j = 0; j < n; ++j)
    {
        for(int64_t i = 0; i < m; ++i)
        {
            if(is_nonzero(i, j))
            {
                h_row.push_back(static_cast<rocsparse_int>(i));
                h_col.push_back(static_cast<rocsparse_int>(j));
                h_val.push_back(value_at<float>(i, j));
            }
        }
    }
    const rocsparse_int          nnz  = static_cast<rocsparse_int>(h_val.size());
    std::vector<float>           want = make_dense<float>(rocsparse_order_column, ld);
    device_vector<rocsparse_int> d_row(h_row), d_col(h_col);
    device_vector<float>         d_val(h_val);

    // Every limit must clamp the 512-thread scatter launch.
    ASSERT_GT((nnz - 1) / 512 + 1, limits[3]);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);

    for(const int limit : limits)
    {
        std::unique_ptr<ScopedMaxGridSizeX> clamp;
        if(limit > 0)
        {
            clamp.reset(new ScopedMaxGridSizeX(handle, limit));
        }

        device_vector<float> d_A(want.size());
        UT_CHECK_HIP(hipMemset(d_A.ptr, 0xFF, d_A.n * sizeof(float)));

        EXPECT_EQ(rocsparse_scoo2dense(handle, m, n, nnz, descr, d_val, d_row, d_col, d_A, ld),
                  rocsparse_status_success);
        UT_CHECK_HIP(hipDeviceSynchronize());

        // The padding rows below m are not written; compare the m x n part.
        std::vector<float> got = to_host(d_A);
        for(int64_t j = 0; j < n; ++j)
        {
            for(int64_t i = m; i < ld; ++i)
            {
                got[i + ld * j] = want[i + ld * j];
            }
        }
        EXPECT_TRUE(got == want) << "grid.x clamped to " << limit << " blocks";
    }

    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

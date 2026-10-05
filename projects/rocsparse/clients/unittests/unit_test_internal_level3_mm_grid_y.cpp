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
// Regression tests for the grid.y column-panel grid-stride loops of the csrmm
// row_split and coomm kernels (AISPARSE-668, AISPARSE-696).
//
// FOCUS. These kernels put the dense column panel index on grid.y. The
// launchers clamp that extent with rocsparse::get_grid_size_y to
// handle->properties.maxGridSize[1], and every kernel grid-strides over the
// panels, so a launch with more panels than the limit still covers every
// column. The real limit is 65535 blocks, i.e. hundreds of thousands of dense
// columns, so these tests shrink maxGridSize[1] to 1, 2 and 3 blocks instead
// and run a small problem through the same code path:
//
//   kernel                            algorithm, operations        panels
//   csrmmnn_row_split_shared_kernel   csr_row_split, A n, n <= 32  4 at n = 30
//   csrmmnn_row_split_kernel          csr_row_split, A n, n > 32   10 + 3 at n = 83
//   csrmmtn_row_split_kernel          csr_row_split, A t, B n      8 at n = 30
//   csrmmtt_row_split_kernel          csr_row_split, A t, B t      8 at n = 30
//   coommtn_atomic_main               coo_atomic, A t              13 at n = 13
//   coommnn_segmented_main_kernel     coo_segmented, A n           5 at n = 43
//   coommnn_segmented_atomic          coo_segmented_atomic, A n    5 at n = 43
//
// B and C are column order throughout. Every panel count exceeds the limits 1
// and 2, and at least one of the three limits does not divide it, so the last
// stride step is a partial one. A kernel whose panel loop does not stride
// leaves the columns past the first `limit` panels unwritten.
//
// BATCHES. Every case also runs with A, B and C strided-batched over three
// batches with different values, so the batch loop nested in the panel loop
// has to pick the right per-batch pointers for every panel.
//
// ARITHMETIC. All values are small integers, so the host reference is exact in
// either precision and does not depend on the order in which atomics land.
//
// TARGET: rocsparse-unit-test-device. The tests drive the public rocsparse_spmm
// entry point and need a real device and the complete handle type. The largest
// allocation is a few kilobytes, so there is no device-memory guard.
//
#include "unit_test_utils.hpp"

// Internal handle definition: the complete _rocsparse_handle type is required to
// reach handle->properties.
#include "rocsparse_handle.hpp"

#include "rocsparse.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace rocsparse_ut;

namespace
{
    // grid.y counterpart of rocsparse_ut::ScopedMaxGridSizeX (unit_test_grid_clamp.hpp).
    struct ScopedMaxGridSizeY
    {
        rocsparse_handle handle;
        int              saved;

        ScopedMaxGridSizeY(rocsparse_handle h, int limit)
            : handle(h)
            , saved(h->properties.maxGridSize[1])
        {
            handle->properties.maxGridSize[1] = limit;
        }

        ~ScopedMaxGridSizeY()
        {
            handle->properties.maxGridSize[1] = saved;
        }

        ScopedMaxGridSizeY(const ScopedMaxGridSizeY&) = delete;

        ScopedMaxGridSizeY& operator=(const ScopedMaxGridSizeY&) = delete;
    };

    // 0 runs with the real limit, as a control for the clamped runs.
    constexpr int grid_y_limits[] = {0, 1, 2, 3};

    constexpr int32_t batch_counts[] = {1, 3};

    constexpr int32_t mat_m = 37;
    constexpr int32_t mat_k = 29;

    enum class Format
    {
        csr,
        coo
    };

    struct Case
    {
        Format              format;
        rocsparse_spmm_alg  alg;
        rocsparse_operation trans_A;
        rocsparse_operation trans_B;
        int32_t             n;
    };

    // Fixed sparsity pattern, sorted by row and then column, about 2 in 7 dense.
    struct Pattern
    {
        std::vector<int32_t> row_ptr;
        std::vector<int32_t> row_ind;
        std::vector<int32_t> col_ind;
    };

    Pattern make_pattern()
    {
        Pattern p;
        p.row_ptr.push_back(0);
        for(int32_t i = 0; i < mat_m; ++i)
        {
            for(int32_t j = 0; j < mat_k; ++j)
            {
                if((3 * i + 5 * j) % 7 < 2)
                {
                    p.row_ind.push_back(i);
                    p.col_ind.push_back(j);
                }
            }
            p.row_ptr.push_back(static_cast<int32_t>(p.col_ind.size()));
        }
        return p;
    }

    struct SpMatDescr
    {
        rocsparse_spmat_descr d = nullptr;
        ~SpMatDescr()
        {
            if(d)
                (void)rocsparse_destroy_spmat_descr(d);
        }
    };

    struct DnMatDescr
    {
        rocsparse_dnmat_descr d = nullptr;
        ~DnMatDescr()
        {
            if(d)
                (void)rocsparse_destroy_dnmat_descr(d);
        }
    };

    template <typename T>
    void check_case(rocsparse_handle handle, const Case& c)
    {
        const Pattern p   = make_pattern();
        const int64_t nnz = static_cast<int64_t>(p.col_ind.size());

        // op(A) is mo x ko, op(B) is ko x n and C is mo x n; B and C are column
        // order, and B is stored transposed when trans_B is.
        const bool    trans_A  = (c.trans_A != rocsparse_operation_none);
        const bool    trans_B  = (c.trans_B != rocsparse_operation_none);
        const int32_t n        = c.n;
        const int32_t mo       = trans_A ? mat_k : mat_m;
        const int32_t ko       = trans_A ? mat_m : mat_k;
        const int64_t ldb      = trans_B ? n : ko;
        const int64_t b_cols   = trans_B ? ko : n;
        const int64_t ldc      = mo;
        const int64_t stride_B = ldb * b_cols;
        const int64_t stride_C = ldc * n;

        const T alpha = static_cast<T>(2);
        const T beta  = static_cast<T>(-1);

        for(int32_t batch_count : batch_counts)
        {
            std::vector<int32_t> row_ptr;
            std::vector<int32_t> row_ind;
            std::vector<int32_t> col_ind;
            std::vector<T>       val;
            for(int32_t b = 0; b < batch_count; ++b)
            {
                row_ptr.insert(row_ptr.end(), p.row_ptr.begin(), p.row_ptr.end());
                row_ind.insert(row_ind.end(), p.row_ind.begin(), p.row_ind.end());
                col_ind.insert(col_ind.end(), p.col_ind.begin(), p.col_ind.end());
                for(int64_t q = 0; q < nnz; ++q)
                {
                    val.push_back(static_cast<T>(1 + (q + 2 * b) % 5));
                }
            }

            std::vector<T> dense_B(stride_B * batch_count);
            for(size_t i = 0; i < dense_B.size(); ++i)
            {
                dense_B[i] = static_cast<T>(static_cast<int>(i % 7) - 3);
            }

            std::vector<T> dense_C(stride_C * batch_count);
            for(size_t i = 0; i < dense_C.size(); ++i)
            {
                dense_C[i] = static_cast<T>(static_cast<int>((i * 3) % 5) - 2);
            }

            // Host reference, C = alpha * op(A) * op(B) + beta * C per batch.
            std::vector<double> want(dense_C.size());
            for(int32_t b = 0; b < batch_count; ++b)
            {
                std::vector<double> op_A(static_cast<size_t>(mo) * ko, 0.0);
                for(int32_t i = 0; i < mat_m; ++i)
                {
                    for(int32_t q = p.row_ptr[i]; q < p.row_ptr[i + 1]; ++q)
                    {
                        const int32_t j = p.col_ind[q];
                        const double  v = static_cast<double>(val[b * nnz + q]);
                        if(trans_A)
                        {
                            op_A[static_cast<size_t>(j) * ko + i] = v;
                        }
                        else
                        {
                            op_A[static_cast<size_t>(i) * ko + j] = v;
                        }
                    }
                }

                for(int32_t col = 0; col < n; ++col)
                {
                    for(int32_t row = 0; row < mo; ++row)
                    {
                        double sum = 0.0;
                        for(int32_t r = 0; r < ko; ++r)
                        {
                            const int64_t bi = trans_B ? (col + r * ldb) : (r + col * ldb);
                            sum += op_A[static_cast<size_t>(row) * ko + r]
                                   * static_cast<double>(dense_B[b * stride_B + bi]);
                        }
                        const int64_t ci = b * stride_C + row + col * ldc;
                        want[ci]         = static_cast<double>(alpha) * sum
                                   + static_cast<double>(beta) * static_cast<double>(dense_C[ci]);
                    }
                }
            }

            for(int limit : grid_y_limits)
            {
                SCOPED_TRACE(testing::Message()
                             << "n = " << n << ", batch_count = " << batch_count
                             << ", maxGridSize[1] = "
                             << (limit == 0 ? std::string("unclamped") : std::to_string(limit)));

                device_vector<int32_t> d_row_ptr(row_ptr);
                device_vector<int32_t> d_row_ind(row_ind);
                device_vector<int32_t> d_col_ind(col_ind);
                device_vector<T>       d_val(val);
                device_vector<T>       d_B(dense_B);
                device_vector<T>       d_C(dense_C);
                ASSERT_TRUE(d_row_ptr.ptr && d_row_ind.ptr && d_col_ind.ptr && d_val.ptr && d_B.ptr
                            && d_C.ptr);

                SpMatDescr mat_A;
                DnMatDescr mat_B;
                DnMatDescr mat_C;
                if(c.format == Format::csr)
                {
                    ASSERT_EQ(rocsparse_create_csr_descr(&mat_A.d,
                                                         mat_m,
                                                         mat_k,
                                                         nnz,
                                                         d_row_ptr.ptr,
                                                         d_col_ind.ptr,
                                                         d_val.ptr,
                                                         rocsparse_indextype_i32,
                                                         rocsparse_indextype_i32,
                                                         rocsparse_index_base_zero,
                                                         dt_of<T>()),
                              rocsparse_status_success);
                }
                else
                {
                    ASSERT_EQ(rocsparse_create_coo_descr(&mat_A.d,
                                                         mat_m,
                                                         mat_k,
                                                         nnz,
                                                         d_row_ind.ptr,
                                                         d_col_ind.ptr,
                                                         d_val.ptr,
                                                         rocsparse_indextype_i32,
                                                         rocsparse_index_base_zero,
                                                         dt_of<T>()),
                              rocsparse_status_success);
                }
                ASSERT_EQ(
                    rocsparse_create_dnmat_descr(
                        &mat_B.d, ldb, b_cols, ldb, d_B.ptr, dt_of<T>(), rocsparse_order_column),
                    rocsparse_status_success);
                ASSERT_EQ(rocsparse_create_dnmat_descr(
                              &mat_C.d, mo, n, ldc, d_C.ptr, dt_of<T>(), rocsparse_order_column),
                          rocsparse_status_success);

                if(batch_count > 1)
                {
                    if(c.format == Format::csr)
                    {
                        ASSERT_EQ(
                            rocsparse_csr_set_strided_batch(mat_A.d, batch_count, mat_m + 1, nnz),
                            rocsparse_status_success);
                    }
                    else
                    {
                        ASSERT_EQ(rocsparse_coo_set_strided_batch(mat_A.d, batch_count, nnz),
                                  rocsparse_status_success);
                    }
                    ASSERT_EQ(rocsparse_dnmat_set_strided_batch(mat_B.d, batch_count, stride_B),
                              rocsparse_status_success);
                    ASSERT_EQ(rocsparse_dnmat_set_strided_batch(mat_C.d, batch_count, stride_C),
                              rocsparse_status_success);
                }

                {
                    std::unique_ptr<ScopedMaxGridSizeY> clamp;
                    if(limit > 0)
                    {
                        clamp.reset(new ScopedMaxGridSizeY(handle, limit));
                    }

                    size_t buffer_size = 0;
                    ASSERT_EQ(rocsparse_spmm(handle,
                                             c.trans_A,
                                             c.trans_B,
                                             &alpha,
                                             mat_A.d,
                                             mat_B.d,
                                             &beta,
                                             mat_C.d,
                                             dt_of<T>(),
                                             c.alg,
                                             rocsparse_spmm_stage_buffer_size,
                                             &buffer_size,
                                             nullptr),
                              rocsparse_status_success);

                    device_vector<char> d_buffer(std::max<size_t>(buffer_size, 1));
                    ASSERT_NE(d_buffer.ptr, nullptr);

                    ASSERT_EQ(rocsparse_spmm(handle,
                                             c.trans_A,
                                             c.trans_B,
                                             &alpha,
                                             mat_A.d,
                                             mat_B.d,
                                             &beta,
                                             mat_C.d,
                                             dt_of<T>(),
                                             c.alg,
                                             rocsparse_spmm_stage_preprocess,
                                             &buffer_size,
                                             d_buffer.ptr),
                              rocsparse_status_success);

                    ASSERT_EQ(rocsparse_spmm(handle,
                                             c.trans_A,
                                             c.trans_B,
                                             &alpha,
                                             mat_A.d,
                                             mat_B.d,
                                             &beta,
                                             mat_C.d,
                                             dt_of<T>(),
                                             c.alg,
                                             rocsparse_spmm_stage_compute,
                                             &buffer_size,
                                             d_buffer.ptr),
                              rocsparse_status_success);
                    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
                }

                const std::vector<T> got = to_host(d_C);
                ASSERT_EQ(got.size(), want.size());

                const double tol = (sizeof(T) == sizeof(float)) ? 1e-5 : 1e-12;
                for(size_t i = 0; i < got.size(); ++i)
                {
                    const double diff = std::abs(static_cast<double>(got[i]) - want[i]);
                    if(diff > tol * std::max(1.0, std::abs(want[i])))
                    {
                        const int64_t local = static_cast<int64_t>(i) % stride_C;
                        ADD_FAILURE()
                            << "first mismatch at batch " << static_cast<int64_t>(i) / stride_C
                            << ", row " << local % ldc << ", column " << local / ldc << ": got "
                            << static_cast<double>(got[i]) << ", want " << want[i];
                        break;
                    }
                }
            }
        }
    }

    void check_all(rocsparse_handle handle, const Case& c)
    {
        check_case<float>(handle, c);
        check_case<double>(handle, c);
    }
}

class MmGridY : public HandleTest
{
};

// ---------------------------------------------------------------------------
// csrmm row_split
// ---------------------------------------------------------------------------

// csrmmnn_row_split_shared_kernel: n <= 32, ceil(30 / 8) = 4 panels of 8 columns.
TEST_F(MmGridY, csrmm_row_split_nn_shared)
{
    check_all(handle,
              {Format::csr,
               rocsparse_spmm_alg_csr_row_split,
               rocsparse_operation_none,
               rocsparse_operation_none,
               30});
}

// csrmmnn_row_split_kernel: n > 32, 80 / 8 = 10 panels of 8 columns in the main
// launch and 3 one-column panels in the remainder launch.
TEST_F(MmGridY, csrmm_row_split_nn_main_and_remainder)
{
    check_all(handle,
              {Format::csr,
               rocsparse_spmm_alg_csr_row_split,
               rocsparse_operation_none,
               rocsparse_operation_none,
               83});
}

// csrmmtn_row_split_kernel: ceil(30 / 4) = 8 panels of 4 columns.
TEST_F(MmGridY, csrmm_row_split_tn)
{
    check_all(handle,
              {Format::csr,
               rocsparse_spmm_alg_csr_row_split,
               rocsparse_operation_transpose,
               rocsparse_operation_none,
               30});
}

// csrmmtt_row_split_kernel: ceil(30 / 4) = 8 panels of 4 columns.
TEST_F(MmGridY, csrmm_row_split_tt)
{
    check_all(handle,
              {Format::csr,
               rocsparse_spmm_alg_csr_row_split,
               rocsparse_operation_transpose,
               rocsparse_operation_transpose,
               30});
}

// ---------------------------------------------------------------------------
// coomm
// ---------------------------------------------------------------------------

// coommtn_atomic_main<.., false, ..>: one column per grid.y block, 13 panels. The
// atomic algorithm only puts columns on grid.y for a transposed A.
TEST_F(MmGridY, coomm_atomic_tn)
{
    check_all(handle,
              {Format::coo,
               rocsparse_spmm_alg_coo_atomic,
               rocsparse_operation_transpose,
               rocsparse_operation_none,
               13});
}

// coommtn_atomic_main<.., true, ..>.
TEST_F(MmGridY, coomm_atomic_tt)
{
    check_all(handle,
              {Format::coo,
               rocsparse_spmm_alg_coo_atomic,
               rocsparse_operation_transpose,
               rocsparse_operation_transpose,
               13});
}

// coommnn_segmented_main_kernel<.., 8, .., false>: 40 / 8 = 5 panels, plus the
// 3 remainder columns, which run on a grid.y of 1.
TEST_F(MmGridY, coomm_segmented_nn)
{
    check_all(handle,
              {Format::coo,
               rocsparse_spmm_alg_coo_segmented,
               rocsparse_operation_none,
               rocsparse_operation_none,
               43});
}

// coommnn_segmented_main_kernel<.., 8, .., true>.
TEST_F(MmGridY, coomm_segmented_nt)
{
    check_all(handle,
              {Format::coo,
               rocsparse_spmm_alg_coo_segmented,
               rocsparse_operation_none,
               rocsparse_operation_transpose,
               43});
}

// coommnn_segmented_atomic<.., COLS = 8, ..>: 40 / 8 = 5 panels, plus the 3
// remainder columns, which run on a grid.y of 1.
TEST_F(MmGridY, coomm_segmented_atomic_nn)
{
    check_all(handle,
              {Format::coo,
               rocsparse_spmm_alg_coo_segmented_atomic,
               rocsparse_operation_none,
               rocsparse_operation_none,
               43});
}

// coommnn_segmented_atomic<.., COLS = 8, NT = true, ..>.
TEST_F(MmGridY, coomm_segmented_atomic_nt)
{
    check_all(handle,
              {Format::coo,
               rocsparse_spmm_alg_coo_segmented_atomic,
               rocsparse_operation_none,
               rocsparse_operation_transpose,
               43});
}

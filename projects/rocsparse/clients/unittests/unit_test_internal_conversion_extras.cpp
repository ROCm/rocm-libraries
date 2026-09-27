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
// Host-path unit tests for previously-untested conversion C APIs:
//   rocsparse_Xgebsr2gebsc (+ _buffer_size),
//   rocsparse_Xprune_csr2csr_by_percentage (+ _nnz / _buffer_size),
//   rocsparse_Xprune_dense2csr_by_percentage (+ _nnz / _buffer_size).
//
// Existing tests covered the plain prune_* and gebsr2csr paths; the percentage
// variants and the gebsr->gebsc transpose lived at 0%. Each test runs the full
// buffer-size / nnz / compute pipeline on a tiny matrix plus a few bad-argument
// branches.
//
#include "unit_test_utils.hpp"

#include "rocsparse.h"

using namespace rocsparse_ut;

namespace
{
    constexpr rocsparse_index_base BASE = rocsparse_index_base_zero;

    // RAII for a mat descr.
    struct MatDescr
    {
        rocsparse_mat_descr d = nullptr;
        MatDescr()
        {
            (void)rocsparse_create_mat_descr(&d);
        }
        ~MatDescr()
        {
            if(d)
                (void)rocsparse_destroy_mat_descr(d);
        }
    };
    struct MatInfo
    {
        rocsparse_mat_info i = nullptr;
        MatInfo()
        {
            (void)rocsparse_create_mat_info(&i);
        }
        ~MatInfo()
        {
            if(i)
                (void)rocsparse_destroy_mat_info(i);
        }
    };
}

// ===========================================================================
// gebsr2gebsc : transpose a GEBSR matrix into GEBSC layout.
//
// With 1x1 blocks this is exactly a CSR -> CSC transpose. The input GEBSR
// matrix is the non-symmetric
//
//     A = | 0  2 |     (row-major GEBSR: row 0 -> col 1 = 2, row 1 -> col 0 = 3)
//         | 3  0 |
//
// so the transpose is observable in the output layout:
//   bsc_col_ptr = {0, 1, 2}, bsc_row_ind = {1, 0}, bsc_val = {3, 2}.
// ===========================================================================
namespace
{
    // Shared, hoisted description of the tiny GEBSR input used by the tests below
    // (mb = nb = nnzb = 2, 1x1 blocks, zero-based).
    const std::vector<int32_t> gebsr_row_ptr{0, 1, 2};
    const std::vector<int32_t> gebsr_col_ind{1, 0};
    const std::vector<float>   gebsr_val{2.0f, 3.0f};

    // Expected GEBSC (transpose) of the matrix above.
    const std::vector<int32_t> expected_bsc_col_ptr{0, 1, 2};
    const std::vector<int32_t> expected_bsc_row_ind{1, 0};
    const std::vector<float>   expected_bsc_val{3.0f, 2.0f};
}

class Gebsr2Gebsc : public HandleTest
{
};

// gebsr2gebsc_buffer_size: the buffer-size query alone must succeed and report a
// size (it drives the sort scratch estimate). Split out from the convert case.
TEST_F(Gebsr2Gebsc, buffer_size)
{
    device_vector<int32_t> row_ptr{gebsr_row_ptr};
    device_vector<int32_t> col_ind{gebsr_col_ind};
    device_vector<float>   val{gebsr_val};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

    size_t buffer_size = 0;
    ASSERT_EQ(rocsparse_sgebsr2gebsc_buffer_size(
                  handle, 2, 2, 2, val, row_ptr, col_ind, 1, 1, &buffer_size),
              rocsparse_status_success);
}

// gebsr2gebsc (numeric): transpose the 2x2 matrix and verify the produced
// GEBSC arrays (col_ptr / row_ind / val) equal the analytic transpose.
TEST_F(Gebsr2Gebsc, transpose_values)
{
    device_vector<int32_t> row_ptr{gebsr_row_ptr};
    device_vector<int32_t> col_ind{gebsr_col_ind};
    device_vector<float>   val{gebsr_val};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

    size_t buffer_size = 0;
    ASSERT_EQ(rocsparse_sgebsr2gebsc_buffer_size(
                  handle, 2, 2, 2, val, row_ptr, col_ind, 1, 1, &buffer_size),
              rocsparse_status_success);

    device_vector<char> tmp{buffer_size ? buffer_size : size_t(1)};
    ASSERT_TRUE(tmp.ptr);

    device_vector<float>   bsc_val{size_t(2)};
    device_vector<int32_t> bsc_row_ind{size_t(2)};
    device_vector<int32_t> bsc_col_ptr{size_t(3)};
    ASSERT_TRUE(bsc_val.ptr && bsc_row_ind.ptr && bsc_col_ptr.ptr);

    ASSERT_EQ(rocsparse_sgebsr2gebsc(handle,
                                     2,
                                     2,
                                     2,
                                     val,
                                     row_ptr,
                                     col_ind,
                                     1,
                                     1,
                                     bsc_val,
                                     bsc_row_ind,
                                     bsc_col_ptr,
                                     rocsparse_action_numeric,
                                     BASE,
                                     tmp.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    EXPECT_EQ(to_host(bsc_col_ptr.ptr, 3), expected_bsc_col_ptr);
    EXPECT_EQ(to_host(bsc_row_ind.ptr, 2), expected_bsc_row_ind);
    EXPECT_EQ(to_host(bsc_val.ptr, 2), expected_bsc_val);
}

// gebsr2gebsc bad args: null handle and null buffer-size pointer are rejected.
TEST_F(Gebsr2Gebsc, bad_args)
{
    device_vector<int32_t> row_ptr{gebsr_row_ptr};
    device_vector<int32_t> col_ind{gebsr_col_ind};
    device_vector<float>   val{gebsr_val};
    size_t                 bs = 0;

    EXPECT_EQ(
        rocsparse_sgebsr2gebsc_buffer_size(nullptr, 2, 2, 2, val, row_ptr, col_ind, 1, 1, &bs),
        rocsparse_status_invalid_handle);
    EXPECT_EQ(
        rocsparse_sgebsr2gebsc_buffer_size(handle, 2, 2, 2, val, row_ptr, col_ind, 1, 1, nullptr),
        rocsparse_status_invalid_pointer);
    // Null value / index arrays are rejected by the buffer-size query.
    EXPECT_EQ(
        rocsparse_sgebsr2gebsc_buffer_size(handle, 2, 2, 2, nullptr, row_ptr, col_ind, 1, 1, &bs),
        rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_sgebsr2gebsc_buffer_size(handle, 2, 2, 2, val, nullptr, col_ind, 1, 1, &bs),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_sgebsr2gebsc_buffer_size(handle, 2, 2, 2, val, row_ptr, nullptr, 1, 1, &bs),
              rocsparse_status_invalid_pointer);
}

// ===========================================================================
// prune_csr2csr_by_percentage : keep the largest-magnitude entries of a CSR.
// ===========================================================================
class PruneCsr2CsrByPct : public HandleTest
{
};

// prune_csr2csr_by_percentage full pipeline: buffer_size -> nnz -> compute on
// A = diag(1, 4) pruning the smallest 50% of the magnitudes. The single largest
// entry (4 at row 1, col 1) must survive: nnz_C = 1, val_C = {4}, col_ind_C = {1},
// row_ptr_C = {0, 0, 1}.
TEST_F(PruneCsr2CsrByPct, full_pipeline)
{
    // A = 2x2 with entries {1, 4} (identity pattern), prune bottom 50%.
    device_vector<int32_t> row_ptr_A{std::vector<int32_t>{0, 1, 2}};
    device_vector<int32_t> col_ind_A{std::vector<int32_t>{0, 1}};
    device_vector<float>   val_A{std::vector<float>{1.0f, 4.0f}};
    ASSERT_TRUE(row_ptr_A.ptr && col_ind_A.ptr && val_A.ptr);

    MatDescr descr_A, descr_C;
    MatInfo  info;
    ASSERT_TRUE(descr_A.d && descr_C.d && info.i);

    const float            percentage = 50.0f;
    device_vector<int32_t> row_ptr_C{size_t(3)};
    ASSERT_TRUE(row_ptr_C.ptr);

    size_t buffer_size = 0;
    ASSERT_EQ(rocsparse_sprune_csr2csr_by_percentage_buffer_size(handle,
                                                                 2,
                                                                 2,
                                                                 2,
                                                                 descr_A.d,
                                                                 val_A,
                                                                 row_ptr_A,
                                                                 col_ind_A,
                                                                 percentage,
                                                                 descr_C.d,
                                                                 nullptr,
                                                                 row_ptr_C,
                                                                 nullptr,
                                                                 info.i,
                                                                 &buffer_size),
              rocsparse_status_success);

    device_vector<char> tmp{buffer_size ? buffer_size : size_t(1)};
    ASSERT_TRUE(tmp.ptr);

    rocsparse_int nnz_C = 0;
    ASSERT_EQ(rocsparse_sprune_csr2csr_nnz_by_percentage(handle,
                                                         2,
                                                         2,
                                                         2,
                                                         descr_A.d,
                                                         val_A,
                                                         row_ptr_A,
                                                         col_ind_A,
                                                         percentage,
                                                         descr_C.d,
                                                         row_ptr_C,
                                                         &nnz_C,
                                                         info.i,
                                                         tmp.ptr),
              rocsparse_status_success);

    device_vector<int32_t> col_ind_C{size_t(2)};
    device_vector<float>   val_C{size_t(2)};
    ASSERT_TRUE(col_ind_C.ptr && val_C.ptr);

    ASSERT_EQ(rocsparse_sprune_csr2csr_by_percentage(handle,
                                                     2,
                                                     2,
                                                     2,
                                                     descr_A.d,
                                                     val_A,
                                                     row_ptr_A,
                                                     col_ind_A,
                                                     percentage,
                                                     descr_C.d,
                                                     val_C,
                                                     row_ptr_C,
                                                     col_ind_C,
                                                     info.i,
                                                     tmp.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // Pruning the smaller-magnitude half of {1, 4} keeps only the 4.
    EXPECT_EQ(nnz_C, 1);
    EXPECT_EQ(to_host(row_ptr_C.ptr, 3), (std::vector<int32_t>{0, 0, 1}));
    ASSERT_GE(nnz_C, 1);
    EXPECT_EQ(to_host(col_ind_C.ptr, 1), (std::vector<int32_t>{1}));
    EXPECT_FLOAT_EQ(to_host(val_C.ptr, 1)[0], 4.0f);
}

TEST_F(PruneCsr2CsrByPct, bad_args)
{
    device_vector<int32_t> row_ptr_A{std::vector<int32_t>{0, 1, 2}};
    device_vector<int32_t> col_ind_A{std::vector<int32_t>{0, 1}};
    device_vector<float>   val_A{std::vector<float>{1.0f, 4.0f}};
    MatDescr               descr_A, descr_C;
    MatInfo                info;
    device_vector<int32_t> row_ptr_C{size_t(3)};
    size_t                 bs = 0;

    EXPECT_EQ(rocsparse_sprune_csr2csr_by_percentage_buffer_size(nullptr,
                                                                 2,
                                                                 2,
                                                                 2,
                                                                 descr_A.d,
                                                                 val_A,
                                                                 row_ptr_A,
                                                                 col_ind_A,
                                                                 50.0f,
                                                                 descr_C.d,
                                                                 nullptr,
                                                                 row_ptr_C,
                                                                 nullptr,
                                                                 info.i,
                                                                 &bs),
              rocsparse_status_invalid_handle);

    rocsparse_int nnz_C = 0;
    // percentage outside [0,100] is rejected.
    EXPECT_EQ(rocsparse_sprune_csr2csr_nnz_by_percentage(handle,
                                                         2,
                                                         2,
                                                         2,
                                                         descr_A.d,
                                                         val_A,
                                                         row_ptr_A,
                                                         col_ind_A,
                                                         -1.0f,
                                                         descr_C.d,
                                                         row_ptr_C,
                                                         &nnz_C,
                                                         info.i,
                                                         nullptr),
              rocsparse_status_invalid_value);
}

// ===========================================================================
// prune_dense2csr_by_percentage : prune a dense matrix into CSR.
// ===========================================================================
class PruneDense2CsrByPct : public HandleTest
{
};

// prune_dense2csr_by_percentage full pipeline: buffer_size -> nnz -> compute on
// dense A = [[1,0],[0,4]] pruning the smallest 50% of the magnitudes. Unlike the
// csr2csr variant, the percentile here is taken over *all* m*n = 4 dense entries
// (including the two zeros), so the 50% cut removes exactly those two zeros and
// keeps both nonzeros: nnz = 2, row_ptr = {0,1,2}, col_ind = {0,1}, val = {1,4}.
TEST_F(PruneDense2CsrByPct, full_pipeline)
{
    // 2x2 dense column-major A = [[1,0],[0,4]] (lda = 2).
    device_vector<float> A{std::vector<float>{1.0f, 0.0f, 0.0f, 4.0f}};
    ASSERT_TRUE(A.ptr);

    MatDescr descr;
    MatInfo  info;
    ASSERT_TRUE(descr.d && info.i);

    const float            percentage = 50.0f;
    device_vector<int32_t> row_ptr{size_t(3)};
    ASSERT_TRUE(row_ptr.ptr);

    size_t buffer_size = 0;
    ASSERT_EQ(rocsparse_sprune_dense2csr_by_percentage_buffer_size(handle,
                                                                   2,
                                                                   2,
                                                                   A,
                                                                   2,
                                                                   percentage,
                                                                   descr.d,
                                                                   nullptr,
                                                                   row_ptr,
                                                                   nullptr,
                                                                   info.i,
                                                                   &buffer_size),
              rocsparse_status_success);

    device_vector<char> tmp{buffer_size ? buffer_size : size_t(1)};
    ASSERT_TRUE(tmp.ptr);

    rocsparse_int nnz = 0;
    ASSERT_EQ(rocsparse_sprune_dense2csr_nnz_by_percentage(
                  handle, 2, 2, A, 2, percentage, descr.d, row_ptr, &nnz, info.i, tmp.ptr),
              rocsparse_status_success);

    device_vector<int32_t> col_ind{size_t(4)};
    device_vector<float>   val{size_t(4)};
    ASSERT_TRUE(col_ind.ptr && val.ptr);

    ASSERT_EQ(rocsparse_sprune_dense2csr_by_percentage(
                  handle, 2, 2, A, 2, percentage, descr.d, val, row_ptr, col_ind, info.i, tmp.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // The 50% cut over all 4 dense entries drops the two zeros; both nonzeros stay.
    EXPECT_EQ(nnz, 2);
    EXPECT_EQ(to_host(row_ptr.ptr, 3), (std::vector<int32_t>{0, 1, 2}));
    ASSERT_GE(nnz, 2);
    EXPECT_EQ(to_host(col_ind.ptr, 2), (std::vector<int32_t>{0, 1}));
    EXPECT_EQ(to_host(val.ptr, 2), (std::vector<float>{1.0f, 4.0f}));
}

TEST_F(PruneDense2CsrByPct, bad_args)
{
    device_vector<float>   A{std::vector<float>{1.0f, 0.0f, 0.0f, 4.0f}};
    MatDescr               descr;
    MatInfo                info;
    device_vector<int32_t> row_ptr{size_t(3)};
    size_t                 bs = 0;

    EXPECT_EQ(rocsparse_sprune_dense2csr_by_percentage_buffer_size(
                  nullptr, 2, 2, A, 2, 50.0f, descr.d, nullptr, row_ptr, nullptr, info.i, &bs),
              rocsparse_status_invalid_handle);

    rocsparse_int nnz = 0;
    EXPECT_EQ(rocsparse_sprune_dense2csr_nnz_by_percentage(
                  handle, 2, 2, A, 2, 150.0f, descr.d, row_ptr, &nnz, info.i, nullptr),
              rocsparse_status_invalid_value);
}

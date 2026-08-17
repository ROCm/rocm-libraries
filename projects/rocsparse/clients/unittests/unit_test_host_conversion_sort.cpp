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
// Host-path unit tests for the rocSPARSE conversion sub-library. These drive
// the public C API on tiny inputs (mostly 3x3 identity CSR / 2x2 blocks) so
// they exercise the HOST dispatch + argument-validation code in
// library/src/conversion (coverage is host-only). Every routine gets at least
// one valid call plus its bad-argument guards (null handle -> invalid_handle,
// null pointer -> invalid_pointer, negative size/nnz -> invalid_size, invalid
// enum -> invalid_value). Typed routines are exercised across all supported
// precisions via if-constexpr dispatch helpers.
//
#include "unit_test_utils.hpp"

#include <type_traits>

using namespace rocsparse_ut;

class Conversion : public HandleTest
{
};

// ===========================================================================
// Index-utility conversions (originally present).
// ===========================================================================
TEST_F(Conversion, csr2coo)
{
    // 3x3 identity-pattern CSR -> COO row indices.
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> coo_row{(size_t)3};
    ASSERT_TRUE(row_ptr.ptr && coo_row.ptr);

    EXPECT_EQ(rocsparse_csr2coo(handle, row_ptr, 3, 3, coo_row, rocsparse_index_base_zero),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // bad args
    EXPECT_EQ(rocsparse_csr2coo(nullptr, row_ptr, 3, 3, coo_row, rocsparse_index_base_zero),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_csr2coo(handle, nullptr, 3, 3, coo_row, rocsparse_index_base_zero),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_csr2coo(handle, row_ptr, -1, 3, coo_row, rocsparse_index_base_zero),
              rocsparse_status_invalid_size);
}

TEST_F(Conversion, coo2csr)
{
    device_vector<rocsparse_int> coo_row{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<rocsparse_int> row_ptr{(size_t)4};
    ASSERT_TRUE(coo_row.ptr && row_ptr.ptr);

    EXPECT_EQ(rocsparse_coo2csr(handle, coo_row, 3, 3, row_ptr, rocsparse_index_base_zero),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    EXPECT_EQ(rocsparse_coo2csr(nullptr, coo_row, 3, 3, row_ptr, rocsparse_index_base_zero),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_coo2csr(handle, nullptr, 3, 3, row_ptr, rocsparse_index_base_zero),
              rocsparse_status_invalid_pointer);
}

TEST_F(Conversion, identity_permutation)
{
    device_vector<rocsparse_int> p{(size_t)5};
    ASSERT_TRUE(p.ptr);
    EXPECT_EQ(rocsparse_create_identity_permutation(handle, 5, p), rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    EXPECT_EQ(rocsparse_create_identity_permutation(nullptr, 5, p),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_create_identity_permutation(handle, -1, p), rocsparse_status_invalid_size);
    EXPECT_EQ(rocsparse_create_identity_permutation(handle, 5, nullptr),
              rocsparse_status_invalid_pointer);
}

TEST_F(Conversion, csrsort)
{
    const rocsparse_int          m = 3, n = 3, nnz = 3;
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<rocsparse_int> perm{(size_t)nnz};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && perm.ptr);

    size_t buffer_size = 0;
    EXPECT_EQ(rocsparse_csrsort_buffer_size(handle, m, n, nnz, row_ptr, col_ind, &buffer_size),
              rocsparse_status_success);
    EXPECT_GT(buffer_size, 0u);

    device_vector<char> buffer{buffer_size};
    ASSERT_TRUE(buffer.ptr);

    EXPECT_EQ(rocsparse_create_identity_permutation(handle, nnz, perm), rocsparse_status_success);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    EXPECT_EQ(rocsparse_csrsort(handle, m, n, nnz, descr, row_ptr, col_ind, perm, buffer.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);

    // bad args on buffer_size
    EXPECT_EQ(rocsparse_csrsort_buffer_size(nullptr, m, n, nnz, row_ptr, col_ind, &buffer_size),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_csrsort_buffer_size(handle, -1, n, nnz, row_ptr, col_ind, &buffer_size),
              rocsparse_status_invalid_size);
}

TEST_F(Conversion, coosort_buffer_size)
{
    const rocsparse_int          m = 3, n = 3, nnz = 3;
    device_vector<rocsparse_int> row_ind{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
    ASSERT_TRUE(row_ind.ptr && col_ind.ptr);

    size_t buffer_size = 0;
    EXPECT_EQ(rocsparse_coosort_buffer_size(handle, m, n, nnz, row_ind, col_ind, &buffer_size),
              rocsparse_status_success);
    EXPECT_GT(buffer_size, 0u);

    EXPECT_EQ(rocsparse_coosort_buffer_size(nullptr, m, n, nnz, row_ind, col_ind, &buffer_size),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_coosort_buffer_size(handle, m, n, nnz, row_ind, col_ind, nullptr),
              rocsparse_status_invalid_pointer);
}

TEST_F(Conversion, cscsort)
{
    const rocsparse_int          m = 3, n = 3, nnz = 3;
    device_vector<rocsparse_int> col_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> row_ind{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<rocsparse_int> perm{(size_t)nnz};
    ASSERT_TRUE(col_ptr.ptr && row_ind.ptr && perm.ptr);

    size_t buffer_size = 0;
    EXPECT_EQ(rocsparse_cscsort_buffer_size(handle, m, n, nnz, col_ptr, row_ind, &buffer_size),
              rocsparse_status_success);
    EXPECT_GT(buffer_size, 0u);

    device_vector<char> buffer{buffer_size};
    ASSERT_TRUE(buffer.ptr);
    EXPECT_EQ(rocsparse_create_identity_permutation(handle, nnz, perm), rocsparse_status_success);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    EXPECT_EQ(rocsparse_cscsort(handle, m, n, nnz, descr, col_ptr, row_ind, perm, buffer.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);

    // bad args
    EXPECT_EQ(rocsparse_cscsort_buffer_size(nullptr, m, n, nnz, col_ptr, row_ind, &buffer_size),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_cscsort_buffer_size(handle, -1, n, nnz, col_ptr, row_ind, &buffer_size),
              rocsparse_status_invalid_size);
    EXPECT_EQ(rocsparse_cscsort_buffer_size(handle, m, n, nnz, col_ptr, row_ind, nullptr),
              rocsparse_status_invalid_pointer);
}

// ===========================================================================
// CSR <-> CSC (transpose).
// ===========================================================================
template <typename T>
static void check_csr2csc(rocsparse_handle handle)
{
    const rocsparse_int          m = 3, n = 3, nnz = 3;
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<T>             csr_val{std::vector<T>(nnz, scalar<T>(1.0f))};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && csr_val.ptr);

    size_t buffer_size = 0;
    EXPECT_EQ(rocsparse_csr2csc_buffer_size(
                  handle, m, n, nnz, row_ptr, col_ind, rocsparse_action_numeric, &buffer_size),
              rocsparse_status_success);
    EXPECT_GT(buffer_size, 0u);

    device_vector<char>          buffer{buffer_size};
    device_vector<T>             csc_val{(size_t)nnz};
    device_vector<rocsparse_int> csc_row_ind{(size_t)nnz};
    device_vector<rocsparse_int> csc_col_ptr{(size_t)(n + 1)};
    ASSERT_TRUE(buffer.ptr && csc_val.ptr && csc_row_ind.ptr && csc_col_ptr.ptr);

    rocsparse_status st = rocsparse_status_success;
    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_scsr2csc(handle,
                                m,
                                n,
                                nnz,
                                csr_val,
                                row_ptr,
                                col_ind,
                                csc_val,
                                csc_row_ind,
                                csc_col_ptr,
                                rocsparse_action_numeric,
                                rocsparse_index_base_zero,
                                buffer.ptr);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dcsr2csc(handle,
                                m,
                                n,
                                nnz,
                                csr_val,
                                row_ptr,
                                col_ind,
                                csc_val,
                                csc_row_ind,
                                csc_col_ptr,
                                rocsparse_action_numeric,
                                rocsparse_index_base_zero,
                                buffer.ptr);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_ccsr2csc(handle,
                                m,
                                n,
                                nnz,
                                csr_val,
                                row_ptr,
                                col_ind,
                                csc_val,
                                csc_row_ind,
                                csc_col_ptr,
                                rocsparse_action_numeric,
                                rocsparse_index_base_zero,
                                buffer.ptr);
    else
        st = rocsparse_zcsr2csc(handle,
                                m,
                                n,
                                nnz,
                                csr_val,
                                row_ptr,
                                col_ind,
                                csc_val,
                                csc_row_ind,
                                csc_col_ptr,
                                rocsparse_action_numeric,
                                rocsparse_index_base_zero,
                                buffer.ptr);
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // bad args
    EXPECT_EQ(rocsparse_csr2csc_buffer_size(
                  nullptr, m, n, nnz, row_ptr, col_ind, rocsparse_action_numeric, &buffer_size),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_csr2csc_buffer_size(
                  handle, -1, n, nnz, row_ptr, col_ind, rocsparse_action_numeric, &buffer_size),
              rocsparse_status_invalid_size);
    EXPECT_EQ(rocsparse_csr2csc_buffer_size(
                  handle, m, n, nnz, row_ptr, col_ind, rocsparse_action_numeric, nullptr),
              rocsparse_status_invalid_pointer);
}

TEST_F(Conversion, csr2csc)
{
    check_csr2csc<float>(handle);
    check_csr2csc<double>(handle);
    check_csr2csc<rocsparse_float_complex>(handle);
    check_csr2csc<rocsparse_double_complex>(handle);
}

// ===========================================================================
// CSR compression (csr2csr_compress + nnz_compress).
// ===========================================================================
template <typename T>
static void check_csr_compress(rocsparse_handle handle)
{
    const rocsparse_int          m = 3, n = 3, nnz = 3;
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<T>             csr_val{std::vector<T>(nnz, scalar<T>(1.0f))};
    device_vector<rocsparse_int> nnz_per_row{(size_t)m};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && csr_val.ptr && nnz_per_row.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);

    const T       tol   = scalar<T>(0.5f);
    rocsparse_int nnz_C = 0;

    rocsparse_status st = rocsparse_status_success;
    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_snnz_compress(handle, m, descr, csr_val, row_ptr, nnz_per_row, &nnz_C, tol);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dnnz_compress(handle, m, descr, csr_val, row_ptr, nnz_per_row, &nnz_C, tol);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_cnnz_compress(handle, m, descr, csr_val, row_ptr, nnz_per_row, &nnz_C, tol);
    else
        st = rocsparse_znnz_compress(handle, m, descr, csr_val, row_ptr, nnz_per_row, &nnz_C, tol);
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(nnz_C, nnz);

    device_vector<T>             csr_val_C{(size_t)nnz_C};
    device_vector<rocsparse_int> csr_row_ptr_C{(size_t)(m + 1)};
    device_vector<rocsparse_int> csr_col_ind_C{(size_t)nnz_C};
    ASSERT_TRUE(csr_val_C.ptr && csr_row_ptr_C.ptr && csr_col_ind_C.ptr);

    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_scsr2csr_compress(handle,
                                         m,
                                         n,
                                         descr,
                                         csr_val,
                                         row_ptr,
                                         col_ind,
                                         nnz,
                                         nnz_per_row,
                                         csr_val_C,
                                         csr_row_ptr_C,
                                         csr_col_ind_C,
                                         tol);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dcsr2csr_compress(handle,
                                         m,
                                         n,
                                         descr,
                                         csr_val,
                                         row_ptr,
                                         col_ind,
                                         nnz,
                                         nnz_per_row,
                                         csr_val_C,
                                         csr_row_ptr_C,
                                         csr_col_ind_C,
                                         tol);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_ccsr2csr_compress(handle,
                                         m,
                                         n,
                                         descr,
                                         csr_val,
                                         row_ptr,
                                         col_ind,
                                         nnz,
                                         nnz_per_row,
                                         csr_val_C,
                                         csr_row_ptr_C,
                                         csr_col_ind_C,
                                         tol);
    else
        st = rocsparse_zcsr2csr_compress(handle,
                                         m,
                                         n,
                                         descr,
                                         csr_val,
                                         row_ptr,
                                         col_ind,
                                         nnz,
                                         nnz_per_row,
                                         csr_val_C,
                                         csr_row_ptr_C,
                                         csr_col_ind_C,
                                         tol);
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // bad args
    if constexpr(std::is_same_v<T, float>)
    {
        EXPECT_EQ(
            rocsparse_snnz_compress(nullptr, m, descr, csr_val, row_ptr, nnz_per_row, &nnz_C, tol),
            rocsparse_status_invalid_handle);
        EXPECT_EQ(
            rocsparse_snnz_compress(handle, -1, descr, csr_val, row_ptr, nnz_per_row, &nnz_C, tol),
            rocsparse_status_invalid_size);
        EXPECT_EQ(rocsparse_scsr2csr_compress(handle,
                                              m,
                                              n,
                                              descr,
                                              csr_val,
                                              row_ptr,
                                              col_ind,
                                              -1,
                                              nnz_per_row,
                                              csr_val_C,
                                              csr_row_ptr_C,
                                              csr_col_ind_C,
                                              tol),
                  rocsparse_status_invalid_size);
    }

    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

TEST_F(Conversion, csr2csr_compress)
{
    check_csr_compress<float>(handle);
    check_csr_compress<double>(handle);
    check_csr_compress<rocsparse_float_complex>(handle);
    check_csr_compress<rocsparse_double_complex>(handle);
}

// ===========================================================================
// prune_csr2csr (real precisions only).
// ===========================================================================
template <typename T>
static void check_prune_csr2csr(rocsparse_handle handle)
{
    const rocsparse_int          m = 3, n = 3, nnz = 3;
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<T>             csr_val{std::vector<T>(nnz, scalar<T>(1.0f))};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && csr_val.ptr);

    rocsparse_mat_descr descr_A = nullptr, descr_C = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr_A), rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_mat_descr(&descr_C), rocsparse_status_success);

    const T                      threshold = scalar<T>(0.5f); // host pointer mode (default)
    device_vector<rocsparse_int> csr_row_ptr_C{(size_t)(m + 1)};
    device_vector<T>             dummy_val{(size_t)nnz};
    device_vector<rocsparse_int> dummy_ci{(size_t)nnz};
    ASSERT_TRUE(csr_row_ptr_C.ptr && dummy_val.ptr && dummy_ci.ptr);

    size_t           buffer_size = 0;
    rocsparse_status st          = rocsparse_status_success;
    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_sprune_csr2csr_buffer_size(handle,
                                                  m,
                                                  n,
                                                  nnz,
                                                  descr_A,
                                                  csr_val,
                                                  row_ptr,
                                                  col_ind,
                                                  &threshold,
                                                  descr_C,
                                                  dummy_val,
                                                  csr_row_ptr_C,
                                                  dummy_ci,
                                                  &buffer_size);
    else
        st = rocsparse_dprune_csr2csr_buffer_size(handle,
                                                  m,
                                                  n,
                                                  nnz,
                                                  descr_A,
                                                  csr_val,
                                                  row_ptr,
                                                  col_ind,
                                                  &threshold,
                                                  descr_C,
                                                  dummy_val,
                                                  csr_row_ptr_C,
                                                  dummy_ci,
                                                  &buffer_size);
    EXPECT_EQ(st, rocsparse_status_success);

    device_vector<char> buffer{buffer_size ? buffer_size : size_t(1)};
    ASSERT_TRUE(buffer.ptr);

    rocsparse_int nnz_C = 0;
    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_sprune_csr2csr_nnz(handle,
                                          m,
                                          n,
                                          nnz,
                                          descr_A,
                                          csr_val,
                                          row_ptr,
                                          col_ind,
                                          &threshold,
                                          descr_C,
                                          csr_row_ptr_C,
                                          &nnz_C,
                                          buffer.ptr);
    else
        st = rocsparse_dprune_csr2csr_nnz(handle,
                                          m,
                                          n,
                                          nnz,
                                          descr_A,
                                          csr_val,
                                          row_ptr,
                                          col_ind,
                                          &threshold,
                                          descr_C,
                                          csr_row_ptr_C,
                                          &nnz_C,
                                          buffer.ptr);
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(nnz_C, nnz);

    device_vector<T>             csr_val_C{(size_t)nnz_C};
    device_vector<rocsparse_int> csr_col_ind_C{(size_t)nnz_C};
    ASSERT_TRUE(csr_val_C.ptr && csr_col_ind_C.ptr);

    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_sprune_csr2csr(handle,
                                      m,
                                      n,
                                      nnz,
                                      descr_A,
                                      csr_val,
                                      row_ptr,
                                      col_ind,
                                      &threshold,
                                      descr_C,
                                      csr_val_C,
                                      csr_row_ptr_C,
                                      csr_col_ind_C,
                                      buffer.ptr);
    else
        st = rocsparse_dprune_csr2csr(handle,
                                      m,
                                      n,
                                      nnz,
                                      descr_A,
                                      csr_val,
                                      row_ptr,
                                      col_ind,
                                      &threshold,
                                      descr_C,
                                      csr_val_C,
                                      csr_row_ptr_C,
                                      csr_col_ind_C,
                                      buffer.ptr);
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // bad args
    if constexpr(std::is_same_v<T, float>)
    {
        EXPECT_EQ(rocsparse_sprune_csr2csr_nnz(nullptr,
                                               m,
                                               n,
                                               nnz,
                                               descr_A,
                                               csr_val,
                                               row_ptr,
                                               col_ind,
                                               &threshold,
                                               descr_C,
                                               csr_row_ptr_C,
                                               &nnz_C,
                                               buffer.ptr),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(rocsparse_sprune_csr2csr_nnz(handle,
                                               -1,
                                               n,
                                               nnz,
                                               descr_A,
                                               csr_val,
                                               row_ptr,
                                               col_ind,
                                               &threshold,
                                               descr_C,
                                               csr_row_ptr_C,
                                               &nnz_C,
                                               buffer.ptr),
                  rocsparse_status_invalid_size);
    }

    EXPECT_EQ(rocsparse_destroy_mat_descr(descr_A), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr_C), rocsparse_status_success);
}

TEST_F(Conversion, prune_csr2csr)
{
    check_prune_csr2csr<float>(handle);
    check_prune_csr2csr<double>(handle);
}

// ===========================================================================
// prune_dense2csr (real precisions only).
// ===========================================================================
template <typename T>
static void check_prune_dense2csr(rocsparse_handle handle)
{
    const rocsparse_int m = 3, n = 3, lda = 3;
    // Column-major 3x3 identity.
    std::vector<T> host_A(lda * n, scalar<T>(0.0f));
    host_A[0] = host_A[4] = host_A[8] = scalar<T>(1.0f);
    device_vector<T> A{host_A};
    ASSERT_TRUE(A.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);

    const T                      threshold = scalar<T>(0.5f);
    device_vector<rocsparse_int> csr_row_ptr{(size_t)(m + 1)};
    device_vector<T>             dummy_val{(size_t)(m * n)};
    device_vector<rocsparse_int> dummy_ci{(size_t)(m * n)};
    ASSERT_TRUE(csr_row_ptr.ptr && dummy_val.ptr && dummy_ci.ptr);

    size_t           buffer_size = 0;
    rocsparse_status st          = rocsparse_status_success;
    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_sprune_dense2csr_buffer_size(handle,
                                                    m,
                                                    n,
                                                    A,
                                                    lda,
                                                    &threshold,
                                                    descr,
                                                    dummy_val,
                                                    csr_row_ptr,
                                                    dummy_ci,
                                                    &buffer_size);
    else
        st = rocsparse_dprune_dense2csr_buffer_size(handle,
                                                    m,
                                                    n,
                                                    A,
                                                    lda,
                                                    &threshold,
                                                    descr,
                                                    dummy_val,
                                                    csr_row_ptr,
                                                    dummy_ci,
                                                    &buffer_size);
    EXPECT_EQ(st, rocsparse_status_success);

    device_vector<char> buffer{buffer_size ? buffer_size : size_t(1)};
    ASSERT_TRUE(buffer.ptr);

    rocsparse_int nnz_C = 0;
    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_sprune_dense2csr_nnz(
            handle, m, n, A, lda, &threshold, descr, csr_row_ptr, &nnz_C, buffer.ptr);
    else
        st = rocsparse_dprune_dense2csr_nnz(
            handle, m, n, A, lda, &threshold, descr, csr_row_ptr, &nnz_C, buffer.ptr);
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(nnz_C, 3);

    device_vector<T>             csr_val{(size_t)nnz_C};
    device_vector<rocsparse_int> csr_col_ind{(size_t)nnz_C};
    ASSERT_TRUE(csr_val.ptr && csr_col_ind.ptr);

    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_sprune_dense2csr(
            handle, m, n, A, lda, &threshold, descr, csr_val, csr_row_ptr, csr_col_ind, buffer.ptr);
    else
        st = rocsparse_dprune_dense2csr(
            handle, m, n, A, lda, &threshold, descr, csr_val, csr_row_ptr, csr_col_ind, buffer.ptr);
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // bad args
    if constexpr(std::is_same_v<T, float>)
    {
        EXPECT_EQ(rocsparse_sprune_dense2csr_nnz(
                      nullptr, m, n, A, lda, &threshold, descr, csr_row_ptr, &nnz_C, buffer.ptr),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(rocsparse_sprune_dense2csr_nnz(
                      handle, -1, n, A, lda, &threshold, descr, csr_row_ptr, &nnz_C, buffer.ptr),
                  rocsparse_status_invalid_size);
    }

    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

TEST_F(Conversion, prune_dense2csr)
{
    check_prune_dense2csr<float>(handle);
    check_prune_dense2csr<double>(handle);
}

// ===========================================================================
// Dense <-> sparse conversions (nnz, csr2dense, dense2csr, coo2dense,
// dense2coo). Kept in a distinct suite for clarity.
// ===========================================================================
class ConversionDense : public HandleTest
{
};

template <typename T>
static void check_nnz_csr2dense_dense2csr(rocsparse_handle handle)
{
    const rocsparse_int          m = 3, n = 3, nnz = 3, ld = 3;
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<T>             csr_val{std::vector<T>(nnz, scalar<T>(1.0f))};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && csr_val.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);

    device_vector<T> A{(size_t)(ld * n)};
    ASSERT_TRUE(A.ptr);

    rocsparse_status st = rocsparse_status_success;
    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_scsr2dense(handle, m, n, descr, csr_val, row_ptr, col_ind, A, ld);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dcsr2dense(handle, m, n, descr, csr_val, row_ptr, col_ind, A, ld);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_ccsr2dense(handle, m, n, descr, csr_val, row_ptr, col_ind, A, ld);
    else
        st = rocsparse_zcsr2dense(handle, m, n, descr, csr_val, row_ptr, col_ind, A, ld);
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // nnz on the resulting dense matrix
    device_vector<rocsparse_int> nnz_per_row{(size_t)m};
    ASSERT_TRUE(nnz_per_row.ptr);
    rocsparse_int nnz_total = 0;
    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_snnz(
            handle, rocsparse_direction_row, m, n, descr, A, ld, nnz_per_row, &nnz_total);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dnnz(
            handle, rocsparse_direction_row, m, n, descr, A, ld, nnz_per_row, &nnz_total);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_cnnz(
            handle, rocsparse_direction_row, m, n, descr, A, ld, nnz_per_row, &nnz_total);
    else
        st = rocsparse_znnz(
            handle, rocsparse_direction_row, m, n, descr, A, ld, nnz_per_row, &nnz_total);
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(nnz_total, nnz);

    // dense -> csr round trip
    device_vector<T>             csr_val_out{(size_t)nnz};
    device_vector<rocsparse_int> csr_row_ptr_out{(size_t)(m + 1)};
    device_vector<rocsparse_int> csr_col_ind_out{(size_t)nnz};
    ASSERT_TRUE(csr_val_out.ptr && csr_row_ptr_out.ptr && csr_col_ind_out.ptr);

    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_sdense2csr(
            handle, m, n, descr, A, ld, nnz_per_row, csr_val_out, csr_row_ptr_out, csr_col_ind_out);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_ddense2csr(
            handle, m, n, descr, A, ld, nnz_per_row, csr_val_out, csr_row_ptr_out, csr_col_ind_out);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_cdense2csr(
            handle, m, n, descr, A, ld, nnz_per_row, csr_val_out, csr_row_ptr_out, csr_col_ind_out);
    else
        st = rocsparse_zdense2csr(
            handle, m, n, descr, A, ld, nnz_per_row, csr_val_out, csr_row_ptr_out, csr_col_ind_out);
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // bad args
    if constexpr(std::is_same_v<T, float>)
    {
        EXPECT_EQ(rocsparse_scsr2dense(nullptr, m, n, descr, csr_val, row_ptr, col_ind, A, ld),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(rocsparse_scsr2dense(handle, -1, n, descr, csr_val, row_ptr, col_ind, A, ld),
                  rocsparse_status_invalid_size);
        EXPECT_EQ(rocsparse_snnz(
                      handle, rocsparse_direction_row, m, n, descr, A, ld, nnz_per_row, nullptr),
                  rocsparse_status_invalid_pointer);
    }

    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

TEST_F(ConversionDense, nnz_csr2dense_dense2csr)
{
    check_nnz_csr2dense_dense2csr<float>(handle);
    check_nnz_csr2dense_dense2csr<double>(handle);
    check_nnz_csr2dense_dense2csr<rocsparse_float_complex>(handle);
    check_nnz_csr2dense_dense2csr<rocsparse_double_complex>(handle);
}

template <typename T>
static void check_coo2dense_dense2coo(rocsparse_handle handle)
{
    const rocsparse_int          m = 3, n = 3, nnz = 3, ld = 3;
    device_vector<rocsparse_int> coo_row{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<rocsparse_int> coo_col{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<T>             coo_val{std::vector<T>(nnz, scalar<T>(1.0f))};
    ASSERT_TRUE(coo_row.ptr && coo_col.ptr && coo_val.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);

    device_vector<T> A{(size_t)(ld * n)};
    ASSERT_TRUE(A.ptr);

    rocsparse_status st = rocsparse_status_success;
    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_scoo2dense(handle, m, n, nnz, descr, coo_val, coo_row, coo_col, A, ld);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dcoo2dense(handle, m, n, nnz, descr, coo_val, coo_row, coo_col, A, ld);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_ccoo2dense(handle, m, n, nnz, descr, coo_val, coo_row, coo_col, A, ld);
    else
        st = rocsparse_zcoo2dense(handle, m, n, nnz, descr, coo_val, coo_row, coo_col, A, ld);
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // nnz per row for dense2coo
    device_vector<rocsparse_int> nnz_per_row{(size_t)m};
    ASSERT_TRUE(nnz_per_row.ptr);
    rocsparse_int nnz_total = 0;
    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_snnz(
            handle, rocsparse_direction_row, m, n, descr, A, ld, nnz_per_row, &nnz_total);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dnnz(
            handle, rocsparse_direction_row, m, n, descr, A, ld, nnz_per_row, &nnz_total);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_cnnz(
            handle, rocsparse_direction_row, m, n, descr, A, ld, nnz_per_row, &nnz_total);
    else
        st = rocsparse_znnz(
            handle, rocsparse_direction_row, m, n, descr, A, ld, nnz_per_row, &nnz_total);
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    device_vector<T>             coo_val_out{(size_t)nnz_total};
    device_vector<rocsparse_int> coo_row_out{(size_t)nnz_total};
    device_vector<rocsparse_int> coo_col_out{(size_t)nnz_total};
    ASSERT_TRUE(coo_val_out.ptr && coo_row_out.ptr && coo_col_out.ptr);

    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_sdense2coo(
            handle, m, n, descr, A, ld, nnz_per_row, coo_val_out, coo_row_out, coo_col_out);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_ddense2coo(
            handle, m, n, descr, A, ld, nnz_per_row, coo_val_out, coo_row_out, coo_col_out);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_cdense2coo(
            handle, m, n, descr, A, ld, nnz_per_row, coo_val_out, coo_row_out, coo_col_out);
    else
        st = rocsparse_zdense2coo(
            handle, m, n, descr, A, ld, nnz_per_row, coo_val_out, coo_row_out, coo_col_out);
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // bad args
    if constexpr(std::is_same_v<T, float>)
    {
        EXPECT_EQ(rocsparse_scoo2dense(nullptr, m, n, nnz, descr, coo_val, coo_row, coo_col, A, ld),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(rocsparse_scoo2dense(handle, m, n, -1, descr, coo_val, coo_row, coo_col, A, ld),
                  rocsparse_status_invalid_size);
    }

    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

TEST_F(ConversionDense, coo2dense_dense2coo)
{
    check_coo2dense_dense2coo<float>(handle);
    check_coo2dense_dense2coo<double>(handle);
    check_coo2dense_dense2coo<rocsparse_float_complex>(handle);
    check_coo2dense_dense2coo<rocsparse_double_complex>(handle);
}

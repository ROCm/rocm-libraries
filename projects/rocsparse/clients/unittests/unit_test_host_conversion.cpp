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
// CSR <-> ELL.
// ===========================================================================
template <typename T>
static void check_csr2ell_ell2csr(rocsparse_handle handle)
{
    const rocsparse_int          m = 3, n = 3, nnz = 3;
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<T>             csr_val{std::vector<T>(nnz, scalar<T>(1.0f))};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && csr_val.ptr);

    rocsparse_mat_descr csr_descr = nullptr, ell_descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&csr_descr), rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_mat_descr(&ell_descr), rocsparse_status_success);

    rocsparse_int ell_width = 0;
    EXPECT_EQ(rocsparse_csr2ell_width(handle, m, csr_descr, row_ptr, ell_descr, &ell_width),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(ell_width, 1);

    device_vector<T>             ell_val{(size_t)(m * ell_width)};
    device_vector<rocsparse_int> ell_col_ind{(size_t)(m * ell_width)};
    ASSERT_TRUE(ell_val.ptr && ell_col_ind.ptr);

    rocsparse_status st = rocsparse_status_success;
    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_scsr2ell(handle,
                                m,
                                csr_descr,
                                csr_val,
                                row_ptr,
                                col_ind,
                                ell_descr,
                                ell_width,
                                ell_val,
                                ell_col_ind);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dcsr2ell(handle,
                                m,
                                csr_descr,
                                csr_val,
                                row_ptr,
                                col_ind,
                                ell_descr,
                                ell_width,
                                ell_val,
                                ell_col_ind);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_ccsr2ell(handle,
                                m,
                                csr_descr,
                                csr_val,
                                row_ptr,
                                col_ind,
                                ell_descr,
                                ell_width,
                                ell_val,
                                ell_col_ind);
    else
        st = rocsparse_zcsr2ell(handle,
                                m,
                                csr_descr,
                                csr_val,
                                row_ptr,
                                col_ind,
                                ell_descr,
                                ell_width,
                                ell_val,
                                ell_col_ind);
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // ell -> csr round trip
    device_vector<rocsparse_int> csr_row_ptr_out{(size_t)(m + 1)};
    ASSERT_TRUE(csr_row_ptr_out.ptr);
    rocsparse_int csr_nnz = 0;
    EXPECT_EQ(
        rocsparse_ell2csr_nnz(
            handle, m, n, ell_descr, ell_width, ell_col_ind, csr_descr, csr_row_ptr_out, &csr_nnz),
        rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(csr_nnz, nnz);

    device_vector<T>             csr_val_out{(size_t)csr_nnz};
    device_vector<rocsparse_int> csr_col_ind_out{(size_t)csr_nnz};
    ASSERT_TRUE(csr_val_out.ptr && csr_col_ind_out.ptr);

    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_sell2csr(handle,
                                m,
                                n,
                                ell_descr,
                                ell_width,
                                ell_val,
                                ell_col_ind,
                                csr_descr,
                                csr_val_out,
                                csr_row_ptr_out,
                                csr_col_ind_out);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dell2csr(handle,
                                m,
                                n,
                                ell_descr,
                                ell_width,
                                ell_val,
                                ell_col_ind,
                                csr_descr,
                                csr_val_out,
                                csr_row_ptr_out,
                                csr_col_ind_out);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_cell2csr(handle,
                                m,
                                n,
                                ell_descr,
                                ell_width,
                                ell_val,
                                ell_col_ind,
                                csr_descr,
                                csr_val_out,
                                csr_row_ptr_out,
                                csr_col_ind_out);
    else
        st = rocsparse_zell2csr(handle,
                                m,
                                n,
                                ell_descr,
                                ell_width,
                                ell_val,
                                ell_col_ind,
                                csr_descr,
                                csr_val_out,
                                csr_row_ptr_out,
                                csr_col_ind_out);
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // bad args
    EXPECT_EQ(rocsparse_csr2ell_width(nullptr, m, csr_descr, row_ptr, ell_descr, &ell_width),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_csr2ell_width(handle, -1, csr_descr, row_ptr, ell_descr, &ell_width),
              rocsparse_status_invalid_size);
    EXPECT_EQ(rocsparse_csr2ell_width(handle, m, csr_descr, row_ptr, ell_descr, nullptr),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(
        rocsparse_ell2csr_nnz(
            nullptr, m, n, ell_descr, ell_width, ell_col_ind, csr_descr, csr_row_ptr_out, &csr_nnz),
        rocsparse_status_invalid_handle);

    EXPECT_EQ(rocsparse_destroy_mat_descr(csr_descr), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(ell_descr), rocsparse_status_success);
}

TEST_F(Conversion, csr2ell_ell2csr)
{
    check_csr2ell_ell2csr<float>(handle);
    check_csr2ell_ell2csr<double>(handle);
    check_csr2ell_ell2csr<rocsparse_float_complex>(handle);
    check_csr2ell_ell2csr<rocsparse_double_complex>(handle);
}

// ===========================================================================
// CSR <-> HYB.
// ===========================================================================
template <typename T>
static void check_csr2hyb_hyb2csr(rocsparse_handle handle)
{
    const rocsparse_int          m = 3, n = 3, nnz = 3;
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<T>             csr_val{std::vector<T>(nnz, scalar<T>(1.0f))};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && csr_val.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    rocsparse_hyb_mat hyb = nullptr;
    ASSERT_EQ(rocsparse_create_hyb_mat(&hyb), rocsparse_status_success);

    rocsparse_status st = rocsparse_status_success;
    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_scsr2hyb(
            handle, m, n, descr, csr_val, row_ptr, col_ind, hyb, 0, rocsparse_hyb_partition_auto);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dcsr2hyb(
            handle, m, n, descr, csr_val, row_ptr, col_ind, hyb, 0, rocsparse_hyb_partition_auto);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_ccsr2hyb(
            handle, m, n, descr, csr_val, row_ptr, col_ind, hyb, 0, rocsparse_hyb_partition_auto);
    else
        st = rocsparse_zcsr2hyb(
            handle, m, n, descr, csr_val, row_ptr, col_ind, hyb, 0, rocsparse_hyb_partition_auto);
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // hyb -> csr round trip
    device_vector<rocsparse_int> csr_row_ptr_out{(size_t)(m + 1)};
    ASSERT_TRUE(csr_row_ptr_out.ptr);
    size_t buffer_size = 0;
    EXPECT_EQ(rocsparse_hyb2csr_buffer_size(handle, descr, hyb, csr_row_ptr_out, &buffer_size),
              rocsparse_status_success);
    EXPECT_GT(buffer_size, 0u);

    device_vector<char>          buffer{buffer_size};
    device_vector<T>             csr_val_out{(size_t)nnz};
    device_vector<rocsparse_int> csr_col_ind_out{(size_t)nnz};
    ASSERT_TRUE(buffer.ptr && csr_val_out.ptr && csr_col_ind_out.ptr);

    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_shyb2csr(
            handle, descr, hyb, csr_val_out, csr_row_ptr_out, csr_col_ind_out, buffer.ptr);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dhyb2csr(
            handle, descr, hyb, csr_val_out, csr_row_ptr_out, csr_col_ind_out, buffer.ptr);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_chyb2csr(
            handle, descr, hyb, csr_val_out, csr_row_ptr_out, csr_col_ind_out, buffer.ptr);
    else
        st = rocsparse_zhyb2csr(
            handle, descr, hyb, csr_val_out, csr_row_ptr_out, csr_col_ind_out, buffer.ptr);
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // bad args
    EXPECT_EQ(rocsparse_hyb2csr_buffer_size(nullptr, descr, hyb, csr_row_ptr_out, &buffer_size),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_hyb2csr_buffer_size(handle, descr, hyb, csr_row_ptr_out, nullptr),
              rocsparse_status_invalid_pointer);

    EXPECT_EQ(rocsparse_destroy_hyb_mat(hyb), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

TEST_F(Conversion, csr2hyb_hyb2csr)
{
    check_csr2hyb_hyb2csr<float>(handle);
    check_csr2hyb_hyb2csr<double>(handle);
    check_csr2hyb_hyb2csr<rocsparse_float_complex>(handle);
    check_csr2hyb_hyb2csr<rocsparse_double_complex>(handle);
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

// ===========================================================================
// BSR / GEBSR conversions.
// ===========================================================================
class ConversionBsr : public HandleTest
{
};

template <typename T>
static void check_csr2bsr_bsr2csr(rocsparse_handle handle)
{
    // 2x2 identity CSR, block_dim 2 -> mb=nb=1.
    const rocsparse_int          m = 2, n = 2, block_dim = 2;
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1}};
    device_vector<T>             csr_val{std::vector<T>(2, scalar<T>(1.0f))};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && csr_val.ptr);

    rocsparse_mat_descr csr_descr = nullptr, bsr_descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&csr_descr), rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_mat_descr(&bsr_descr), rocsparse_status_success);

    const rocsparse_int          mb = (m + block_dim - 1) / block_dim;
    device_vector<rocsparse_int> bsr_row_ptr{(size_t)(mb + 1)};
    ASSERT_TRUE(bsr_row_ptr.ptr);
    rocsparse_int bsr_nnzb = 0;
    EXPECT_EQ(rocsparse_csr2bsr_nnz(handle,
                                    rocsparse_direction_row,
                                    m,
                                    n,
                                    csr_descr,
                                    row_ptr,
                                    col_ind,
                                    block_dim,
                                    bsr_descr,
                                    bsr_row_ptr,
                                    &bsr_nnzb),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(bsr_nnzb, 1);

    device_vector<T>             bsr_val{(size_t)(bsr_nnzb * block_dim * block_dim)};
    device_vector<rocsparse_int> bsr_col_ind{(size_t)bsr_nnzb};
    ASSERT_TRUE(bsr_val.ptr && bsr_col_ind.ptr);

    rocsparse_status st = rocsparse_status_success;
    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_scsr2bsr(handle,
                                rocsparse_direction_row,
                                m,
                                n,
                                csr_descr,
                                csr_val,
                                row_ptr,
                                col_ind,
                                block_dim,
                                bsr_descr,
                                bsr_val,
                                bsr_row_ptr,
                                bsr_col_ind);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dcsr2bsr(handle,
                                rocsparse_direction_row,
                                m,
                                n,
                                csr_descr,
                                csr_val,
                                row_ptr,
                                col_ind,
                                block_dim,
                                bsr_descr,
                                bsr_val,
                                bsr_row_ptr,
                                bsr_col_ind);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_ccsr2bsr(handle,
                                rocsparse_direction_row,
                                m,
                                n,
                                csr_descr,
                                csr_val,
                                row_ptr,
                                col_ind,
                                block_dim,
                                bsr_descr,
                                bsr_val,
                                bsr_row_ptr,
                                bsr_col_ind);
    else
        st = rocsparse_zcsr2bsr(handle,
                                rocsparse_direction_row,
                                m,
                                n,
                                csr_descr,
                                csr_val,
                                row_ptr,
                                col_ind,
                                block_dim,
                                bsr_descr,
                                bsr_val,
                                bsr_row_ptr,
                                bsr_col_ind);
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // bsr -> csr round trip
    const rocsparse_int          nnz_csr = bsr_nnzb * block_dim * block_dim;
    device_vector<T>             csr_val_out{(size_t)nnz_csr};
    device_vector<rocsparse_int> csr_row_ptr_out{(size_t)(m + 1)};
    device_vector<rocsparse_int> csr_col_ind_out{(size_t)nnz_csr};
    ASSERT_TRUE(csr_val_out.ptr && csr_row_ptr_out.ptr && csr_col_ind_out.ptr);

    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_sbsr2csr(handle,
                                rocsparse_direction_row,
                                mb,
                                mb,
                                bsr_descr,
                                bsr_val,
                                bsr_row_ptr,
                                bsr_col_ind,
                                block_dim,
                                csr_descr,
                                csr_val_out,
                                csr_row_ptr_out,
                                csr_col_ind_out);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dbsr2csr(handle,
                                rocsparse_direction_row,
                                mb,
                                mb,
                                bsr_descr,
                                bsr_val,
                                bsr_row_ptr,
                                bsr_col_ind,
                                block_dim,
                                csr_descr,
                                csr_val_out,
                                csr_row_ptr_out,
                                csr_col_ind_out);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_cbsr2csr(handle,
                                rocsparse_direction_row,
                                mb,
                                mb,
                                bsr_descr,
                                bsr_val,
                                bsr_row_ptr,
                                bsr_col_ind,
                                block_dim,
                                csr_descr,
                                csr_val_out,
                                csr_row_ptr_out,
                                csr_col_ind_out);
    else
        st = rocsparse_zbsr2csr(handle,
                                rocsparse_direction_row,
                                mb,
                                mb,
                                bsr_descr,
                                bsr_val,
                                bsr_row_ptr,
                                bsr_col_ind,
                                block_dim,
                                csr_descr,
                                csr_val_out,
                                csr_row_ptr_out,
                                csr_col_ind_out);
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // bad args
    if constexpr(std::is_same_v<T, float>)
    {
        EXPECT_EQ(rocsparse_csr2bsr_nnz(nullptr,
                                        rocsparse_direction_row,
                                        m,
                                        n,
                                        csr_descr,
                                        row_ptr,
                                        col_ind,
                                        block_dim,
                                        bsr_descr,
                                        bsr_row_ptr,
                                        &bsr_nnzb),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(rocsparse_csr2bsr_nnz(handle,
                                        rocsparse_direction_row,
                                        -1,
                                        n,
                                        csr_descr,
                                        row_ptr,
                                        col_ind,
                                        block_dim,
                                        bsr_descr,
                                        bsr_row_ptr,
                                        &bsr_nnzb),
                  rocsparse_status_invalid_size);
        EXPECT_EQ(rocsparse_csr2bsr_nnz(handle,
                                        rocsparse_direction_row,
                                        m,
                                        n,
                                        csr_descr,
                                        nullptr,
                                        col_ind,
                                        block_dim,
                                        bsr_descr,
                                        bsr_row_ptr,
                                        &bsr_nnzb),
                  rocsparse_status_invalid_pointer);
    }

    EXPECT_EQ(rocsparse_destroy_mat_descr(csr_descr), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(bsr_descr), rocsparse_status_success);
}

TEST_F(ConversionBsr, csr2bsr_bsr2csr)
{
    check_csr2bsr_bsr2csr<float>(handle);
    check_csr2bsr_bsr2csr<double>(handle);
    check_csr2bsr_bsr2csr<rocsparse_float_complex>(handle);
    check_csr2bsr_bsr2csr<rocsparse_double_complex>(handle);
}

template <typename T>
static void check_csr2gebsr(rocsparse_handle handle)
{
    const rocsparse_int          m = 2, n = 2, rbd = 2, cbd = 2;
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1}};
    device_vector<T>             csr_val{std::vector<T>(2, scalar<T>(1.0f))};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && csr_val.ptr);

    rocsparse_mat_descr csr_descr = nullptr, bsr_descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&csr_descr), rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_mat_descr(&bsr_descr), rocsparse_status_success);

    size_t           buffer_size = 0;
    rocsparse_status st          = rocsparse_status_success;
    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_scsr2gebsr_buffer_size(handle,
                                              rocsparse_direction_row,
                                              m,
                                              n,
                                              csr_descr,
                                              csr_val,
                                              row_ptr,
                                              col_ind,
                                              rbd,
                                              cbd,
                                              &buffer_size);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dcsr2gebsr_buffer_size(handle,
                                              rocsparse_direction_row,
                                              m,
                                              n,
                                              csr_descr,
                                              csr_val,
                                              row_ptr,
                                              col_ind,
                                              rbd,
                                              cbd,
                                              &buffer_size);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_ccsr2gebsr_buffer_size(handle,
                                              rocsparse_direction_row,
                                              m,
                                              n,
                                              csr_descr,
                                              csr_val,
                                              row_ptr,
                                              col_ind,
                                              rbd,
                                              cbd,
                                              &buffer_size);
    else
        st = rocsparse_zcsr2gebsr_buffer_size(handle,
                                              rocsparse_direction_row,
                                              m,
                                              n,
                                              csr_descr,
                                              csr_val,
                                              row_ptr,
                                              col_ind,
                                              rbd,
                                              cbd,
                                              &buffer_size);
    EXPECT_EQ(st, rocsparse_status_success);

    device_vector<char>          buffer{buffer_size ? buffer_size : size_t(1)};
    const rocsparse_int          mb = (m + rbd - 1) / rbd;
    device_vector<rocsparse_int> bsr_row_ptr{(size_t)(mb + 1)};
    ASSERT_TRUE(buffer.ptr && bsr_row_ptr.ptr);

    rocsparse_int bsr_nnzb = 0;
    EXPECT_EQ(rocsparse_csr2gebsr_nnz(handle,
                                      rocsparse_direction_row,
                                      m,
                                      n,
                                      csr_descr,
                                      row_ptr,
                                      col_ind,
                                      bsr_descr,
                                      bsr_row_ptr,
                                      rbd,
                                      cbd,
                                      &bsr_nnzb,
                                      buffer.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(bsr_nnzb, 1);

    device_vector<T>             bsr_val{(size_t)(bsr_nnzb * rbd * cbd)};
    device_vector<rocsparse_int> bsr_col_ind{(size_t)bsr_nnzb};
    ASSERT_TRUE(bsr_val.ptr && bsr_col_ind.ptr);

    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_scsr2gebsr(handle,
                                  rocsparse_direction_row,
                                  m,
                                  n,
                                  csr_descr,
                                  csr_val,
                                  row_ptr,
                                  col_ind,
                                  bsr_descr,
                                  bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind,
                                  rbd,
                                  cbd,
                                  buffer.ptr);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dcsr2gebsr(handle,
                                  rocsparse_direction_row,
                                  m,
                                  n,
                                  csr_descr,
                                  csr_val,
                                  row_ptr,
                                  col_ind,
                                  bsr_descr,
                                  bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind,
                                  rbd,
                                  cbd,
                                  buffer.ptr);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_ccsr2gebsr(handle,
                                  rocsparse_direction_row,
                                  m,
                                  n,
                                  csr_descr,
                                  csr_val,
                                  row_ptr,
                                  col_ind,
                                  bsr_descr,
                                  bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind,
                                  rbd,
                                  cbd,
                                  buffer.ptr);
    else
        st = rocsparse_zcsr2gebsr(handle,
                                  rocsparse_direction_row,
                                  m,
                                  n,
                                  csr_descr,
                                  csr_val,
                                  row_ptr,
                                  col_ind,
                                  bsr_descr,
                                  bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind,
                                  rbd,
                                  cbd,
                                  buffer.ptr);
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // bad args
    if constexpr(std::is_same_v<T, float>)
    {
        EXPECT_EQ(rocsparse_csr2gebsr_nnz(nullptr,
                                          rocsparse_direction_row,
                                          m,
                                          n,
                                          csr_descr,
                                          row_ptr,
                                          col_ind,
                                          bsr_descr,
                                          bsr_row_ptr,
                                          rbd,
                                          cbd,
                                          &bsr_nnzb,
                                          buffer.ptr),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(rocsparse_csr2gebsr_nnz(handle,
                                          rocsparse_direction_row,
                                          -1,
                                          n,
                                          csr_descr,
                                          row_ptr,
                                          col_ind,
                                          bsr_descr,
                                          bsr_row_ptr,
                                          rbd,
                                          cbd,
                                          &bsr_nnzb,
                                          buffer.ptr),
                  rocsparse_status_invalid_size);
    }

    EXPECT_EQ(rocsparse_destroy_mat_descr(csr_descr), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(bsr_descr), rocsparse_status_success);
}

TEST_F(ConversionBsr, csr2gebsr)
{
    check_csr2gebsr<float>(handle);
    check_csr2gebsr<double>(handle);
    check_csr2gebsr<rocsparse_float_complex>(handle);
    check_csr2gebsr<rocsparse_double_complex>(handle);
}

template <typename T>
static void check_gebsr2csr(rocsparse_handle handle)
{
    // 1x1 block grid, 2x2 blocks -> 2x2 CSR.
    const rocsparse_int          mb = 1, nb = 1, rbd = 2, cbd = 2, nnzb = 1;
    device_vector<rocsparse_int> bsr_row_ptr{std::vector<rocsparse_int>{0, 1}};
    device_vector<rocsparse_int> bsr_col_ind{std::vector<rocsparse_int>{0}};
    device_vector<T>             bsr_val{std::vector<T>(nnzb * rbd * cbd, scalar<T>(1.0f))};
    ASSERT_TRUE(bsr_row_ptr.ptr && bsr_col_ind.ptr && bsr_val.ptr);

    rocsparse_mat_descr bsr_descr = nullptr, csr_descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&bsr_descr), rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_mat_descr(&csr_descr), rocsparse_status_success);

    const rocsparse_int          m = mb * rbd, nnz = nnzb * rbd * cbd;
    device_vector<T>             csr_val{(size_t)nnz};
    device_vector<rocsparse_int> csr_row_ptr{(size_t)(m + 1)};
    device_vector<rocsparse_int> csr_col_ind{(size_t)nnz};
    ASSERT_TRUE(csr_val.ptr && csr_row_ptr.ptr && csr_col_ind.ptr);

    rocsparse_status st = rocsparse_status_success;
    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_sgebsr2csr(handle,
                                  rocsparse_direction_row,
                                  mb,
                                  nb,
                                  bsr_descr,
                                  bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind,
                                  rbd,
                                  cbd,
                                  csr_descr,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dgebsr2csr(handle,
                                  rocsparse_direction_row,
                                  mb,
                                  nb,
                                  bsr_descr,
                                  bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind,
                                  rbd,
                                  cbd,
                                  csr_descr,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_cgebsr2csr(handle,
                                  rocsparse_direction_row,
                                  mb,
                                  nb,
                                  bsr_descr,
                                  bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind,
                                  rbd,
                                  cbd,
                                  csr_descr,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind);
    else
        st = rocsparse_zgebsr2csr(handle,
                                  rocsparse_direction_row,
                                  mb,
                                  nb,
                                  bsr_descr,
                                  bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind,
                                  rbd,
                                  cbd,
                                  csr_descr,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind);
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // bad args
    if constexpr(std::is_same_v<T, float>)
    {
        EXPECT_EQ(rocsparse_sgebsr2csr(nullptr,
                                       rocsparse_direction_row,
                                       mb,
                                       nb,
                                       bsr_descr,
                                       bsr_val,
                                       bsr_row_ptr,
                                       bsr_col_ind,
                                       rbd,
                                       cbd,
                                       csr_descr,
                                       csr_val,
                                       csr_row_ptr,
                                       csr_col_ind),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(rocsparse_sgebsr2csr(handle,
                                       rocsparse_direction_row,
                                       -1,
                                       nb,
                                       bsr_descr,
                                       bsr_val,
                                       bsr_row_ptr,
                                       bsr_col_ind,
                                       rbd,
                                       cbd,
                                       csr_descr,
                                       csr_val,
                                       csr_row_ptr,
                                       csr_col_ind),
                  rocsparse_status_invalid_size);
    }

    EXPECT_EQ(rocsparse_destroy_mat_descr(bsr_descr), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(csr_descr), rocsparse_status_success);
}

TEST_F(ConversionBsr, gebsr2csr)
{
    check_gebsr2csr<float>(handle);
    check_gebsr2csr<double>(handle);
    check_gebsr2csr<rocsparse_float_complex>(handle);
    check_gebsr2csr<rocsparse_double_complex>(handle);
}

template <typename T>
static void check_gebsr2gebsr(rocsparse_handle handle)
{
    // GEBSR A: 1x1 block grid, 2x2 blocks. Convert to same block dims.
    const rocsparse_int          mb = 1, nb = 1, nnzb = 1, rbd = 2, cbd = 2;
    device_vector<rocsparse_int> bsr_row_ptr_A{std::vector<rocsparse_int>{0, 1}};
    device_vector<rocsparse_int> bsr_col_ind_A{std::vector<rocsparse_int>{0}};
    device_vector<T>             bsr_val_A{std::vector<T>(nnzb * rbd * cbd, scalar<T>(1.0f))};
    ASSERT_TRUE(bsr_row_ptr_A.ptr && bsr_col_ind_A.ptr && bsr_val_A.ptr);

    rocsparse_mat_descr descr_A = nullptr, descr_C = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr_A), rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_mat_descr(&descr_C), rocsparse_status_success);

    size_t           buffer_size = 0;
    rocsparse_status st          = rocsparse_status_success;
    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_sgebsr2gebsr_buffer_size(handle,
                                                rocsparse_direction_row,
                                                mb,
                                                nb,
                                                nnzb,
                                                descr_A,
                                                bsr_val_A,
                                                bsr_row_ptr_A,
                                                bsr_col_ind_A,
                                                rbd,
                                                cbd,
                                                rbd,
                                                cbd,
                                                &buffer_size);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dgebsr2gebsr_buffer_size(handle,
                                                rocsparse_direction_row,
                                                mb,
                                                nb,
                                                nnzb,
                                                descr_A,
                                                bsr_val_A,
                                                bsr_row_ptr_A,
                                                bsr_col_ind_A,
                                                rbd,
                                                cbd,
                                                rbd,
                                                cbd,
                                                &buffer_size);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_cgebsr2gebsr_buffer_size(handle,
                                                rocsparse_direction_row,
                                                mb,
                                                nb,
                                                nnzb,
                                                descr_A,
                                                bsr_val_A,
                                                bsr_row_ptr_A,
                                                bsr_col_ind_A,
                                                rbd,
                                                cbd,
                                                rbd,
                                                cbd,
                                                &buffer_size);
    else
        st = rocsparse_zgebsr2gebsr_buffer_size(handle,
                                                rocsparse_direction_row,
                                                mb,
                                                nb,
                                                nnzb,
                                                descr_A,
                                                bsr_val_A,
                                                bsr_row_ptr_A,
                                                bsr_col_ind_A,
                                                rbd,
                                                cbd,
                                                rbd,
                                                cbd,
                                                &buffer_size);
    EXPECT_EQ(st, rocsparse_status_success);

    device_vector<char>          buffer{buffer_size ? buffer_size : size_t(1)};
    device_vector<rocsparse_int> bsr_row_ptr_C{(size_t)(mb + 1)};
    ASSERT_TRUE(buffer.ptr && bsr_row_ptr_C.ptr);

    rocsparse_int nnzb_C = 0;
    EXPECT_EQ(rocsparse_gebsr2gebsr_nnz(handle,
                                        rocsparse_direction_row,
                                        mb,
                                        nb,
                                        nnzb,
                                        descr_A,
                                        bsr_row_ptr_A,
                                        bsr_col_ind_A,
                                        rbd,
                                        cbd,
                                        descr_C,
                                        bsr_row_ptr_C,
                                        rbd,
                                        cbd,
                                        &nnzb_C,
                                        buffer.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(nnzb_C, 1);

    device_vector<T>             bsr_val_C{(size_t)(nnzb_C * rbd * cbd)};
    device_vector<rocsparse_int> bsr_col_ind_C{(size_t)nnzb_C};
    ASSERT_TRUE(bsr_val_C.ptr && bsr_col_ind_C.ptr);

    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_sgebsr2gebsr(handle,
                                    rocsparse_direction_row,
                                    mb,
                                    nb,
                                    nnzb,
                                    descr_A,
                                    bsr_val_A,
                                    bsr_row_ptr_A,
                                    bsr_col_ind_A,
                                    rbd,
                                    cbd,
                                    descr_C,
                                    bsr_val_C,
                                    bsr_row_ptr_C,
                                    bsr_col_ind_C,
                                    rbd,
                                    cbd,
                                    buffer.ptr);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dgebsr2gebsr(handle,
                                    rocsparse_direction_row,
                                    mb,
                                    nb,
                                    nnzb,
                                    descr_A,
                                    bsr_val_A,
                                    bsr_row_ptr_A,
                                    bsr_col_ind_A,
                                    rbd,
                                    cbd,
                                    descr_C,
                                    bsr_val_C,
                                    bsr_row_ptr_C,
                                    bsr_col_ind_C,
                                    rbd,
                                    cbd,
                                    buffer.ptr);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_cgebsr2gebsr(handle,
                                    rocsparse_direction_row,
                                    mb,
                                    nb,
                                    nnzb,
                                    descr_A,
                                    bsr_val_A,
                                    bsr_row_ptr_A,
                                    bsr_col_ind_A,
                                    rbd,
                                    cbd,
                                    descr_C,
                                    bsr_val_C,
                                    bsr_row_ptr_C,
                                    bsr_col_ind_C,
                                    rbd,
                                    cbd,
                                    buffer.ptr);
    else
        st = rocsparse_zgebsr2gebsr(handle,
                                    rocsparse_direction_row,
                                    mb,
                                    nb,
                                    nnzb,
                                    descr_A,
                                    bsr_val_A,
                                    bsr_row_ptr_A,
                                    bsr_col_ind_A,
                                    rbd,
                                    cbd,
                                    descr_C,
                                    bsr_val_C,
                                    bsr_row_ptr_C,
                                    bsr_col_ind_C,
                                    rbd,
                                    cbd,
                                    buffer.ptr);
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // bad args
    if constexpr(std::is_same_v<T, float>)
    {
        EXPECT_EQ(rocsparse_gebsr2gebsr_nnz(nullptr,
                                            rocsparse_direction_row,
                                            mb,
                                            nb,
                                            nnzb,
                                            descr_A,
                                            bsr_row_ptr_A,
                                            bsr_col_ind_A,
                                            rbd,
                                            cbd,
                                            descr_C,
                                            bsr_row_ptr_C,
                                            rbd,
                                            cbd,
                                            &nnzb_C,
                                            buffer.ptr),
                  rocsparse_status_invalid_handle);
        EXPECT_EQ(rocsparse_gebsr2gebsr_nnz(handle,
                                            rocsparse_direction_row,
                                            -1,
                                            nb,
                                            nnzb,
                                            descr_A,
                                            bsr_row_ptr_A,
                                            bsr_col_ind_A,
                                            rbd,
                                            cbd,
                                            descr_C,
                                            bsr_row_ptr_C,
                                            rbd,
                                            cbd,
                                            &nnzb_C,
                                            buffer.ptr),
                  rocsparse_status_invalid_size);
    }

    EXPECT_EQ(rocsparse_destroy_mat_descr(descr_A), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr_C), rocsparse_status_success);
}

TEST_F(ConversionBsr, gebsr2gebsr)
{
    check_gebsr2gebsr<float>(handle);
    check_gebsr2gebsr<double>(handle);
    check_gebsr2gebsr<rocsparse_float_complex>(handle);
    check_gebsr2gebsr<rocsparse_double_complex>(handle);
}

template <typename T>
static void check_bsrpad_value(rocsparse_handle handle)
{
    // BSR with mb=nb=2, block_dim=2 (4x4 blocks) but logical m=3 so the last
    // diagonal block is padded. Two diagonal blocks -> nnzb=2.
    const rocsparse_int          m = 3, mb = 2, block_dim = 2, nnzb = 2;
    device_vector<rocsparse_int> bsr_row_ptr{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<rocsparse_int> bsr_col_ind{std::vector<rocsparse_int>{0, 1}};
    device_vector<T> bsr_val{std::vector<T>(nnzb * block_dim * block_dim, scalar<T>(1.0f))};
    ASSERT_TRUE(bsr_row_ptr.ptr && bsr_col_ind.ptr && bsr_val.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);

    const T          value = scalar<T>(1.0f);
    rocsparse_status st    = rocsparse_status_success;
    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_sbsrpad_value(
            handle, m, mb, nnzb, block_dim, value, descr, bsr_val, bsr_row_ptr, bsr_col_ind);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dbsrpad_value(
            handle, m, mb, nnzb, block_dim, value, descr, bsr_val, bsr_row_ptr, bsr_col_ind);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_cbsrpad_value(
            handle, m, mb, nnzb, block_dim, value, descr, bsr_val, bsr_row_ptr, bsr_col_ind);
    else
        st = rocsparse_zbsrpad_value(
            handle, m, mb, nnzb, block_dim, value, descr, bsr_val, bsr_row_ptr, bsr_col_ind);
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // bad args
    if constexpr(std::is_same_v<T, float>)
    {
        EXPECT_EQ(
            rocsparse_sbsrpad_value(
                nullptr, m, mb, nnzb, block_dim, value, descr, bsr_val, bsr_row_ptr, bsr_col_ind),
            rocsparse_status_invalid_handle);
        EXPECT_EQ(rocsparse_sbsrpad_value(
                      handle, m, mb, nnzb, -1, value, descr, bsr_val, bsr_row_ptr, bsr_col_ind),
                  rocsparse_status_invalid_size);
        EXPECT_EQ(
            rocsparse_sbsrpad_value(
                handle, m, mb, nnzb, block_dim, value, descr, nullptr, bsr_row_ptr, bsr_col_ind),
            rocsparse_status_invalid_pointer);
    }

    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

TEST_F(ConversionBsr, bsrpad_value)
{
    check_bsrpad_value<float>(handle);
    check_bsrpad_value<double>(handle);
    check_bsrpad_value<rocsparse_float_complex>(handle);
    check_bsrpad_value<rocsparse_double_complex>(handle);
}

// ===========================================================================
// dense2csx ORDER / BASE / DIRECTION branch coverage.
//
// The heavily-branched dense2csx dispatch (rocsparse_dense2csx_impl.hpp and
// rocsparse_dense2csx.hpp) keys off the dense memory ORDER, the index BASE,
// the DIRECTION (row=CSR / column=CSC), and the leading dimension. The typed
// C APIs (rocsparse_Xdense2csr / _Xdense2csc / _Xdense2coo) always drive the
// column-oriented path (they hard-code rocsparse_order_column), so those
// suites vary base + direction + ld. The generic rocsparse_dense_to_sparse
// path forwards the dense-matrix-descriptor order verbatim, so it is used to
// reach the rocsparse_order_row branches as well.
// ===========================================================================

namespace
{
    // Fixed logical 3x4 sparsity pattern (a couple of zeros) reused everywhere:
    //   (0,0)=1  (0,3)=4  (1,2)=2  (2,3)=3   -> nnz = 4
    //   per-row nnz    : {2, 1, 1}
    //   per-column nnz : {1, 0, 1, 2}
    template <typename T>
    static T ut_dense_entry(rocsparse_int i, rocsparse_int j)
    {
        if(i == 0 && j == 0)
            return rocsparse_ut::scalar<T>(1.0f);
        if(i == 0 && j == 3)
            return rocsparse_ut::scalar<T>(4.0f);
        if(i == 1 && j == 2)
            return rocsparse_ut::scalar<T>(2.0f);
        if(i == 2 && j == 3)
            return rocsparse_ut::scalar<T>(3.0f);
        return rocsparse_ut::scalar<T>(0.0f);
    }

    template <typename T>
    static std::vector<T>
        ut_make_dense(rocsparse_int m, rocsparse_int n, rocsparse_order order, int64_t ld)
    {
        // column-major: ld >= m, element (i,j) at i + j*ld
        // row-major   : ld >= n, element (i,j) at i*ld + j
        const size_t   sz = (order == rocsparse_order_column) ? (size_t)ld * n : (size_t)ld * m;
        std::vector<T> A(sz, rocsparse_ut::scalar<T>(0.0f));
        for(rocsparse_int i = 0; i < m; ++i)
        {
            for(rocsparse_int j = 0; j < n; ++j)
            {
                const size_t idx = (order == rocsparse_order_column)
                                       ? (size_t)i + (size_t)j * (size_t)ld
                                       : (size_t)i * (size_t)ld + (size_t)j;
                A[idx]           = ut_dense_entry<T>(i, j);
            }
        }
        return A;
    }

    template <typename T>
    static rocsparse_status ut_nnz(rocsparse_handle          handle,
                                   rocsparse_direction       dir,
                                   rocsparse_int             m,
                                   rocsparse_int             n,
                                   const rocsparse_mat_descr descr,
                                   const T*                  A,
                                   rocsparse_int             ld,
                                   rocsparse_int*            nnz_per,
                                   rocsparse_int*            nnz_total)
    {
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_snnz(handle, dir, m, n, descr, A, ld, nnz_per, nnz_total);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dnnz(handle, dir, m, n, descr, A, ld, nnz_per, nnz_total);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_cnnz(handle, dir, m, n, descr, A, ld, nnz_per, nnz_total);
        else
            return rocsparse_znnz(handle, dir, m, n, descr, A, ld, nnz_per, nnz_total);
    }

    template <typename T>
    static rocsparse_status ut_dense2csr(rocsparse_handle          handle,
                                         rocsparse_int             m,
                                         rocsparse_int             n,
                                         const rocsparse_mat_descr descr,
                                         const T*                  A,
                                         rocsparse_int             ld,
                                         const rocsparse_int*      nnz_per_row,
                                         T*                        csr_val,
                                         rocsparse_int*            csr_row_ptr,
                                         rocsparse_int*            csr_col_ind)
    {
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_sdense2csr(
                handle, m, n, descr, A, ld, nnz_per_row, csr_val, csr_row_ptr, csr_col_ind);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_ddense2csr(
                handle, m, n, descr, A, ld, nnz_per_row, csr_val, csr_row_ptr, csr_col_ind);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_cdense2csr(
                handle, m, n, descr, A, ld, nnz_per_row, csr_val, csr_row_ptr, csr_col_ind);
        else
            return rocsparse_zdense2csr(
                handle, m, n, descr, A, ld, nnz_per_row, csr_val, csr_row_ptr, csr_col_ind);
    }

    template <typename T>
    static rocsparse_status ut_dense2csc(rocsparse_handle          handle,
                                         rocsparse_int             m,
                                         rocsparse_int             n,
                                         const rocsparse_mat_descr descr,
                                         const T*                  A,
                                         rocsparse_int             ld,
                                         const rocsparse_int*      nnz_per_col,
                                         T*                        csc_val,
                                         rocsparse_int*            csc_col_ptr,
                                         rocsparse_int*            csc_row_ind)
    {
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_sdense2csc(
                handle, m, n, descr, A, ld, nnz_per_col, csc_val, csc_col_ptr, csc_row_ind);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_ddense2csc(
                handle, m, n, descr, A, ld, nnz_per_col, csc_val, csc_col_ptr, csc_row_ind);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_cdense2csc(
                handle, m, n, descr, A, ld, nnz_per_col, csc_val, csc_col_ptr, csc_row_ind);
        else
            return rocsparse_zdense2csc(
                handle, m, n, descr, A, ld, nnz_per_col, csc_val, csc_col_ptr, csc_row_ind);
    }

    template <typename T>
    static rocsparse_status ut_dense2coo(rocsparse_handle          handle,
                                         rocsparse_int             m,
                                         rocsparse_int             n,
                                         const rocsparse_mat_descr descr,
                                         const T*                  A,
                                         rocsparse_int             ld,
                                         const rocsparse_int*      nnz_per_row,
                                         T*                        coo_val,
                                         rocsparse_int*            coo_row_ind,
                                         rocsparse_int*            coo_col_ind)
    {
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_sdense2coo(
                handle, m, n, descr, A, ld, nnz_per_row, coo_val, coo_row_ind, coo_col_ind);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_ddense2coo(
                handle, m, n, descr, A, ld, nnz_per_row, coo_val, coo_row_ind, coo_col_ind);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_cdense2coo(
                handle, m, n, descr, A, ld, nnz_per_row, coo_val, coo_row_ind, coo_col_ind);
        else
            return rocsparse_zdense2coo(
                handle, m, n, descr, A, ld, nnz_per_row, coo_val, coo_row_ind, coo_col_ind);
    }
} // namespace

class ConversionDense2csxOrder : public HandleTest
{
};

// Typed dense2csr / dense2csc / dense2coo + nnz, sweeping index base and ld.
// These always run the rocsparse_order_column code path (hard-coded by the C
// API) but exercise both DIRECTION_row (csr/coo, nnz per row) and
// DIRECTION_column (csc, nnz per column), each with base zero/one and with the
// leading dimension exactly equal to and strictly larger than the minimum.
template <typename T>
static void
    check_dense2csx_typed(rocsparse_handle handle, rocsparse_index_base base, int64_t ld_extra)
{
    const rocsparse_int m = 3, n = 4;
    const rocsparse_int ld = (rocsparse_int)(m + ld_extra); // column-major -> ld >= m

    std::vector<T>   hA = ut_make_dense<T>(m, n, rocsparse_order_column, ld);
    device_vector<T> A{hA};
    ASSERT_TRUE(A.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    ASSERT_EQ(rocsparse_set_mat_index_base(descr, base), rocsparse_status_success);

    // ---- DIRECTION_row: nnz-per-row + dense2csr ----
    device_vector<rocsparse_int> nnz_per_row{(size_t)m};
    ASSERT_TRUE(nnz_per_row.ptr);
    rocsparse_int nnz_row_total = 0;
    EXPECT_EQ(
        ut_nnz<T>(handle, rocsparse_direction_row, m, n, descr, A, ld, nnz_per_row, &nnz_row_total),
        rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(nnz_row_total, 4);

    device_vector<rocsparse_int> csr_row_ptr{(size_t)(m + 1)};
    device_vector<rocsparse_int> csr_col_ind{(size_t)nnz_row_total};
    device_vector<T>             csr_val{(size_t)nnz_row_total};
    ASSERT_TRUE(csr_row_ptr.ptr && csr_col_ind.ptr && csr_val.ptr);
    EXPECT_EQ(
        ut_dense2csr<T>(handle, m, n, descr, A, ld, nnz_per_row, csr_val, csr_row_ptr, csr_col_ind),
        rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // ---- DIRECTION_column: nnz-per-column + dense2csc ----
    device_vector<rocsparse_int> nnz_per_col{(size_t)n};
    ASSERT_TRUE(nnz_per_col.ptr);
    rocsparse_int nnz_col_total = 0;
    EXPECT_EQ(
        ut_nnz<T>(
            handle, rocsparse_direction_column, m, n, descr, A, ld, nnz_per_col, &nnz_col_total),
        rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(nnz_col_total, 4);

    device_vector<rocsparse_int> csc_col_ptr{(size_t)(n + 1)};
    device_vector<rocsparse_int> csc_row_ind{(size_t)nnz_col_total};
    device_vector<T>             csc_val{(size_t)nnz_col_total};
    ASSERT_TRUE(csc_col_ptr.ptr && csc_row_ind.ptr && csc_val.ptr);
    EXPECT_EQ(
        ut_dense2csc<T>(handle, m, n, descr, A, ld, nnz_per_col, csc_val, csc_col_ptr, csc_row_ind),
        rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // ---- dense2coo (uses nnz-per-row / DIRECTION_row internally) ----
    device_vector<rocsparse_int> coo_row_ind{(size_t)nnz_row_total};
    device_vector<rocsparse_int> coo_col_ind{(size_t)nnz_row_total};
    device_vector<T>             coo_val{(size_t)nnz_row_total};
    ASSERT_TRUE(coo_row_ind.ptr && coo_col_ind.ptr && coo_val.ptr);
    EXPECT_EQ(
        ut_dense2coo<T>(handle, m, n, descr, A, ld, nnz_per_row, coo_val, coo_row_ind, coo_col_ind),
        rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

template <typename T>
static void sweep_dense2csx_typed(rocsparse_handle handle)
{
    for(rocsparse_index_base base : {rocsparse_index_base_zero, rocsparse_index_base_one})
    {
        for(int64_t ld_extra : {int64_t(0), int64_t(2)})
        {
            check_dense2csx_typed<T>(handle, base, ld_extra);
        }
    }
}

TEST_F(ConversionDense2csxOrder, typed_base_ld_direction)
{
    sweep_dense2csx_typed<float>(handle);
    sweep_dense2csx_typed<double>(handle);
    sweep_dense2csx_typed<rocsparse_float_complex>(handle);
    sweep_dense2csx_typed<rocsparse_double_complex>(handle);
}

// Generic dense_to_sparse: forwards the dense-matrix ORDER, so this drives the
// rocsparse_order_row branches (and re-covers order_column) of dense2csx for
// CSR (direction_row), CSC (direction_column) and COO targets, across both
// index bases. Uses the documented buffer-size / analysis / compute sequence.
template <typename T>
static void check_dense_to_sparse(rocsparse_handle     handle,
                                  rocsparse_order      order,
                                  rocsparse_format     format,
                                  rocsparse_index_base base)
{
    const rocsparse_int m = 3, n = 4;
    const int64_t       ld = (order == rocsparse_order_column) ? (int64_t)m : (int64_t)n;

    std::vector<T>   hA = ut_make_dense<T>(m, n, order, ld);
    device_vector<T> A{hA};
    ASSERT_TRUE(A.ptr);

    rocsparse_dnmat_descr dn = nullptr;
    ASSERT_EQ(rocsparse_create_dnmat_descr(&dn, m, n, ld, A.ptr, dt_of<T>(), order),
              rocsparse_status_success);

    // Offsets array (row_ptr for CSR, col_ptr for CSC); unused for COO.
    device_vector<rocsparse_int> offsets{(size_t)((m > n ? m : n) + 1)};
    ASSERT_TRUE(offsets.ptr);

    rocsparse_spmat_descr sp = nullptr;
    if(format == rocsparse_format_csr)
    {
        ASSERT_EQ(rocsparse_create_csr_descr(&sp,
                                             m,
                                             n,
                                             0,
                                             offsets.ptr,
                                             nullptr,
                                             nullptr,
                                             it_of<int32_t>(),
                                             it_of<int32_t>(),
                                             base,
                                             dt_of<T>()),
                  rocsparse_status_success);
    }
    else if(format == rocsparse_format_csc)
    {
        ASSERT_EQ(rocsparse_create_csc_descr(&sp,
                                             m,
                                             n,
                                             0,
                                             offsets.ptr,
                                             nullptr,
                                             nullptr,
                                             it_of<int32_t>(),
                                             it_of<int32_t>(),
                                             base,
                                             dt_of<T>()),
                  rocsparse_status_success);
    }
    else
    {
        ASSERT_EQ(rocsparse_create_coo_descr(
                      &sp, m, n, 0, nullptr, nullptr, nullptr, it_of<int32_t>(), base, dt_of<T>()),
                  rocsparse_status_success);
    }

    // Stage 1: query buffer size.
    size_t buffer_size = 0;
    ASSERT_EQ(rocsparse_dense_to_sparse(
                  handle, dn, sp, rocsparse_dense_to_sparse_alg_default, &buffer_size, nullptr),
              rocsparse_status_success);

    device_vector<char> buffer{buffer_size ? buffer_size : size_t(1)};
    ASSERT_TRUE(buffer.ptr);

    // Stage 2: analysis (computes nnz / fills offsets).
    ASSERT_EQ(rocsparse_dense_to_sparse(
                  handle, dn, sp, rocsparse_dense_to_sparse_alg_default, nullptr, buffer.ptr),
              rocsparse_status_success);

    int64_t rows = 0, cols = 0, nnz = 0;
    ASSERT_EQ(rocsparse_spmat_get_size(sp, &rows, &cols, &nnz), rocsparse_status_success);
    EXPECT_EQ(nnz, 4);

    device_vector<rocsparse_int> idx0{(size_t)nnz};
    device_vector<rocsparse_int> idx1{(size_t)nnz};
    device_vector<T>             val{(size_t)nnz};
    ASSERT_TRUE(idx0.ptr && idx1.ptr && val.ptr);

    if(format == rocsparse_format_csr)
    {
        ASSERT_EQ(rocsparse_csr_set_pointers(sp, offsets.ptr, idx0.ptr, val.ptr),
                  rocsparse_status_success);
    }
    else if(format == rocsparse_format_csc)
    {
        ASSERT_EQ(rocsparse_csc_set_pointers(sp, offsets.ptr, idx0.ptr, val.ptr),
                  rocsparse_status_success);
    }
    else
    {
        ASSERT_EQ(rocsparse_coo_set_pointers(sp, idx0.ptr, idx1.ptr, val.ptr),
                  rocsparse_status_success);
    }

    // Stage 3: compute (actual conversion, drives dense2csx with dn order).
    ASSERT_EQ(rocsparse_dense_to_sparse(
                  handle, dn, sp, rocsparse_dense_to_sparse_alg_default, &buffer_size, buffer.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    EXPECT_EQ(rocsparse_destroy_spmat_descr(sp), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_dnmat_descr(dn), rocsparse_status_success);
}

template <typename T>
static void sweep_dense_to_sparse(rocsparse_handle handle)
{
    for(rocsparse_order order : {rocsparse_order_column, rocsparse_order_row})
    {
        for(rocsparse_format fmt :
            {rocsparse_format_csr, rocsparse_format_csc, rocsparse_format_coo})
        {
            for(rocsparse_index_base base : {rocsparse_index_base_zero, rocsparse_index_base_one})
            {
                check_dense_to_sparse<T>(handle, order, fmt, base);
            }
        }
    }
}

TEST_F(ConversionDense2csxOrder, generic_dense_to_sparse_order)
{
    sweep_dense_to_sparse<float>(handle);
    sweep_dense_to_sparse<double>(handle);
    sweep_dense_to_sparse<rocsparse_float_complex>(handle);
    sweep_dense_to_sparse<rocsparse_double_complex>(handle);
}

// Argument-validation / not-implemented branches of the dense2csx dispatch and
// the dense2coo checkarg.
TEST_F(ConversionDense2csxOrder, invalid_and_quickreturn)
{
    const rocsparse_int  m = 3, n = 4, ld = 3;
    std::vector<float>   hA = ut_make_dense<float>(m, n, rocsparse_order_column, ld);
    device_vector<float> A{hA};
    ASSERT_TRUE(A.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);

    device_vector<rocsparse_int> nnz_per_row{(size_t)m};
    device_vector<rocsparse_int> csr_row_ptr{(size_t)(m + 1)};
    device_vector<rocsparse_int> csr_col_ind{(size_t)(m * n)};
    device_vector<float>         csr_val{(size_t)(m * n)};
    ASSERT_TRUE(nnz_per_row.ptr && csr_row_ptr.ptr && csr_col_ind.ptr && csr_val.ptr);

    // invalid handle
    EXPECT_EQ(rocsparse_sdense2csr(
                  nullptr, m, n, descr, A, ld, nnz_per_row, csr_val, csr_row_ptr, csr_col_ind),
              rocsparse_status_invalid_handle);
    // negative size
    EXPECT_EQ(rocsparse_sdense2csr(
                  handle, -1, n, descr, A, ld, nnz_per_row, csr_val, csr_row_ptr, csr_col_ind),
              rocsparse_status_invalid_size);
    // ld < m (order_column minimum)
    EXPECT_EQ(rocsparse_sdense2csr(
                  handle, m, n, descr, A, m - 1, nnz_per_row, csr_val, csr_row_ptr, csr_col_ind),
              rocsparse_status_invalid_size);
    // null dense pointer
    EXPECT_EQ(rocsparse_sdense2csr(handle,
                                   m,
                                   n,
                                   descr,
                                   (const float*)nullptr,
                                   ld,
                                   nnz_per_row,
                                   csr_val,
                                   csr_row_ptr,
                                   csr_col_ind),
              rocsparse_status_invalid_pointer);

    // requires-sorted-storage branch
    ASSERT_EQ(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_unsorted),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_sdense2csr(
                  handle, m, n, descr, A, ld, nnz_per_row, csr_val, csr_row_ptr, csr_col_ind),
              rocsparse_status_requires_sorted_storage);
    ASSERT_EQ(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_sorted),
              rocsparse_status_success);

    // not-implemented branch (non-general matrix type)
    ASSERT_EQ(rocsparse_set_mat_type(descr, rocsparse_matrix_type_symmetric),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_sdense2csr(
                  handle, m, n, descr, A, ld, nnz_per_row, csr_val, csr_row_ptr, csr_col_ind),
              rocsparse_status_not_implemented);
    ASSERT_EQ(rocsparse_set_mat_type(descr, rocsparse_matrix_type_general),
              rocsparse_status_success);

    // quick return: m == 0 (fills row_ptr[0] with base, returns success)
    EXPECT_EQ(rocsparse_sdense2csr(
                  handle, 0, n, descr, A, 0, nnz_per_row, csr_val, csr_row_ptr, csr_col_ind),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // dense2coo checkarg: ld < m -> invalid_size ; negative n -> invalid_size
    device_vector<rocsparse_int> coo_row{(size_t)(m * n)};
    device_vector<rocsparse_int> coo_col{(size_t)(m * n)};
    device_vector<float>         coo_val{(size_t)(m * n)};
    ASSERT_TRUE(coo_row.ptr && coo_col.ptr && coo_val.ptr);
    EXPECT_EQ(
        rocsparse_sdense2coo(handle, m, n, descr, A, m - 1, nnz_per_row, coo_val, coo_row, coo_col),
        rocsparse_status_invalid_size);
    EXPECT_EQ(
        rocsparse_sdense2coo(handle, m, -1, descr, A, ld, nnz_per_row, coo_val, coo_row, coo_col),
        rocsparse_status_invalid_size);
    EXPECT_EQ(
        rocsparse_sdense2coo(nullptr, m, n, descr, A, ld, nnz_per_row, coo_val, coo_row, coo_col),
        rocsparse_status_invalid_handle);

    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

// ===========================================================================
// sparse_to_sparse: descriptor lifecycle, buffer-size + both stages, and the
// argument-validation branches of rocsparse_sparse_to_sparse.cpp.
// ===========================================================================
class ConversionSparseToSparse : public HandleTest
{
};

namespace
{
    // Build a 3x3 identity-pattern CSR spmat descriptor (owning device arrays
    // kept alive by the caller via the passed device_vectors).
    static rocsparse_status ut_make_csr(rocsparse_spmat_descr*        descr,
                                        device_vector<rocsparse_int>& row_ptr,
                                        device_vector<rocsparse_int>& col_ind,
                                        device_vector<float>&         val)
    {
        return rocsparse_create_csr_descr(descr,
                                          3,
                                          3,
                                          3,
                                          row_ptr.ptr,
                                          col_ind.ptr,
                                          val.ptr,
                                          rocsparse_indextype_i32,
                                          rocsparse_indextype_i32,
                                          rocsparse_index_base_zero,
                                          rocsparse_datatype_f32_r);
    }
} // namespace

static void run_sparse_to_sparse_pair(rocsparse_handle            handle,
                                      rocsparse_const_spmat_descr source,
                                      rocsparse_spmat_descr       target,
                                      bool                        permissive)
{
    rocsparse_sparse_to_sparse_descr s2s = nullptr;
    ASSERT_EQ(rocsparse_create_sparse_to_sparse_descr(
                  &s2s, source, target, rocsparse_sparse_to_sparse_alg_default),
              rocsparse_status_success);

    if(permissive)
    {
        ASSERT_EQ(rocsparse_sparse_to_sparse_permissive(s2s), rocsparse_status_success);
    }

    for(rocsparse_sparse_to_sparse_stage stage :
        {rocsparse_sparse_to_sparse_stage_analysis, rocsparse_sparse_to_sparse_stage_compute})
    {
        size_t buffer_size = 0;
        ASSERT_EQ(rocsparse_sparse_to_sparse_buffer_size(
                      handle, s2s, source, target, stage, &buffer_size),
                  rocsparse_status_success);

        device_vector<char> buffer{buffer_size ? buffer_size : size_t(1)};
        ASSERT_TRUE(buffer.ptr);
        ASSERT_EQ(
            rocsparse_sparse_to_sparse(handle, s2s, source, target, stage, buffer_size, buffer.ptr),
            rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    }

    EXPECT_EQ(rocsparse_destroy_sparse_to_sparse_descr(s2s), rocsparse_status_success);
}

TEST_F(ConversionSparseToSparse, csr_to_coo)
{
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<float>         csr_val{std::vector<float>(3, 1.0f)};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && csr_val.ptr);

    device_vector<rocsparse_int> coo_row{(size_t)3};
    device_vector<rocsparse_int> coo_col{(size_t)3};
    device_vector<float>         coo_val{(size_t)3};
    ASSERT_TRUE(coo_row.ptr && coo_col.ptr && coo_val.ptr);

    rocsparse_spmat_descr csr = nullptr, coo = nullptr;
    ASSERT_EQ(ut_make_csr(&csr, row_ptr, col_ind, csr_val), rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_coo_descr(&coo,
                                         3,
                                         3,
                                         3,
                                         coo_row.ptr,
                                         coo_col.ptr,
                                         coo_val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    run_sparse_to_sparse_pair(handle, csr, coo, /*permissive=*/false);

    EXPECT_EQ(rocsparse_destroy_spmat_descr(csr), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(coo), rocsparse_status_success);
}

TEST_F(ConversionSparseToSparse, coo_to_csr)
{
    device_vector<rocsparse_int> coo_row{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<rocsparse_int> coo_col{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<float>         coo_val{std::vector<float>(3, 1.0f)};
    ASSERT_TRUE(coo_row.ptr && coo_col.ptr && coo_val.ptr);

    device_vector<rocsparse_int> row_ptr{(size_t)4};
    device_vector<rocsparse_int> col_ind{(size_t)3};
    device_vector<float>         csr_val{(size_t)3};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && csr_val.ptr);

    rocsparse_spmat_descr coo = nullptr, csr = nullptr;
    ASSERT_EQ(rocsparse_create_coo_descr(&coo,
                                         3,
                                         3,
                                         3,
                                         coo_row.ptr,
                                         coo_col.ptr,
                                         coo_val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);
    ASSERT_EQ(ut_make_csr(&csr, row_ptr, col_ind, csr_val), rocsparse_status_success);

    run_sparse_to_sparse_pair(handle, coo, csr, /*permissive=*/true);

    EXPECT_EQ(rocsparse_destroy_spmat_descr(coo), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(csr), rocsparse_status_success);
}

TEST_F(ConversionSparseToSparse, csr_to_csc)
{
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<float>         csr_val{std::vector<float>(3, 1.0f)};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && csr_val.ptr);

    device_vector<rocsparse_int> csc_col_ptr{(size_t)4};
    device_vector<rocsparse_int> csc_row_ind{(size_t)3};
    device_vector<float>         csc_val{(size_t)3};
    ASSERT_TRUE(csc_col_ptr.ptr && csc_row_ind.ptr && csc_val.ptr);

    rocsparse_spmat_descr csr = nullptr, csc = nullptr;
    ASSERT_EQ(ut_make_csr(&csr, row_ptr, col_ind, csr_val), rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_csc_descr(&csc,
                                         3,
                                         3,
                                         3,
                                         csc_col_ptr.ptr,
                                         csc_row_ind.ptr,
                                         csc_val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    run_sparse_to_sparse_pair(handle, csr, csc, /*permissive=*/false);

    EXPECT_EQ(rocsparse_destroy_spmat_descr(csr), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(csc), rocsparse_status_success);
}

TEST_F(ConversionSparseToSparse, invalid_args)
{
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<float>         csr_val{std::vector<float>(3, 1.0f)};
    device_vector<rocsparse_int> coo_row{(size_t)3};
    device_vector<rocsparse_int> coo_col{(size_t)3};
    device_vector<float>         coo_val{(size_t)3};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && csr_val.ptr && coo_row.ptr && coo_col.ptr
                && coo_val.ptr);

    rocsparse_spmat_descr csr = nullptr, coo = nullptr;
    ASSERT_EQ(ut_make_csr(&csr, row_ptr, col_ind, csr_val), rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_coo_descr(&coo,
                                         3,
                                         3,
                                         3,
                                         coo_row.ptr,
                                         coo_col.ptr,
                                         coo_val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    rocsparse_sparse_to_sparse_descr s2s = nullptr;

    // create_descr: null out-pointer / source / target, invalid alg enum.
    EXPECT_EQ(rocsparse_create_sparse_to_sparse_descr(
                  nullptr, csr, coo, rocsparse_sparse_to_sparse_alg_default),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_create_sparse_to_sparse_descr(
                  &s2s, nullptr, coo, rocsparse_sparse_to_sparse_alg_default),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_create_sparse_to_sparse_descr(
                  &s2s, csr, nullptr, rocsparse_sparse_to_sparse_alg_default),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_create_sparse_to_sparse_descr(
                  &s2s, csr, coo, (rocsparse_sparse_to_sparse_alg)0x7fffffff),
              rocsparse_status_invalid_value);

    // permissive: null descr.
    EXPECT_EQ(rocsparse_sparse_to_sparse_permissive(nullptr), rocsparse_status_invalid_pointer);

    // A valid descriptor for the remaining checks.
    ASSERT_EQ(rocsparse_create_sparse_to_sparse_descr(
                  &s2s, csr, coo, rocsparse_sparse_to_sparse_alg_default),
              rocsparse_status_success);

    size_t buffer_size = 0;
    // buffer_size: invalid handle.
    EXPECT_EQ(rocsparse_sparse_to_sparse_buffer_size(
                  nullptr, s2s, csr, coo, rocsparse_sparse_to_sparse_stage_analysis, &buffer_size),
              rocsparse_status_invalid_handle);
    // buffer_size: null descr.
    EXPECT_EQ(
        rocsparse_sparse_to_sparse_buffer_size(
            handle, nullptr, csr, coo, rocsparse_sparse_to_sparse_stage_analysis, &buffer_size),
        rocsparse_status_invalid_pointer);
    // compute call: invalid stage enum.
    EXPECT_EQ(rocsparse_sparse_to_sparse(
                  handle, s2s, csr, coo, (rocsparse_sparse_to_sparse_stage)999, 0, nullptr),
              rocsparse_status_invalid_value);
    // compute call: non-null size with null buffer -> invalid_pointer.
    EXPECT_EQ(rocsparse_sparse_to_sparse(
                  handle, s2s, csr, coo, rocsparse_sparse_to_sparse_stage_compute, 128, nullptr),
              rocsparse_status_invalid_pointer);

    EXPECT_EQ(rocsparse_destroy_sparse_to_sparse_descr(s2s), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(csr), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(coo), rocsparse_status_success);
}

// ===========================================================================
// SECOND COVERAGE WAVE
//
// Extra tests aimed at conversion lines still uncovered after the first wave.
//
// Notes on lines that are NOT reachable on this configuration (documented here
// rather than chased with impossible tests):
//   * rocsparse_dense2csx.hpp WF_SIZE==64 kernel-launch `else` blocks
//     (dense2csr_kernel / dense2csc_kernel) are selected only when
//     handle->wavefront_size != 32. gfx1201 is a wave32 part, so only the
//     WF_SIZE==32 branch executes; the else is dead on this GPU (independent of
//     data type -- the type only changes the compile-time NROWS_PER_BLOCK).
//   * rocsparse_dense2csx.hpp trailing `return rocsparse_status_invalid_value`
//     is wrapped in LCOV_EXCL and is unreachable (switch over row/column only).
//   * rocsparse_dense2csx_impl.hpp fallback rocprim allocation branch
//     (buffer_size < temp_storage_bytes) does not trigger: the handle buffer is
//     >= 1 MiB and the inclusive-scan scratch never exceeds it for testable
//     sizes. The symbolic guard (csx_val && csx_col both null with nnz != 0) is
//     rejected earlier by the CHECKARG_ARRAY guards, so it is defensive only.
//   * rocsparse_sparse_to_sparse.cpp enum_utils::to_string(alg/stage) overloads
//     have no callers anywhere in the library (CHECKARG_ENUM uses is_invalid),
//     and are wrapped in LCOV_EXCL; the checkarg/impl quick-return blocks are
//     dead because sparse_to_sparse_quickreturn always returns _continue.
// ===========================================================================

namespace
{
    // Parametrized dense_to_sparse over a diagonal matrix (nnz = min(m,n)).
    // Used to drive dense2csr / dense2csc kernels for f64 and complex types at
    // sizes large enough (>= 32) to launch multiple blocks, in both memory
    // orders.
    template <typename T>
    static void check_dense_to_sparse_sized(rocsparse_handle     handle,
                                            rocsparse_order      order,
                                            rocsparse_format     format,
                                            rocsparse_int        m,
                                            rocsparse_int        n,
                                            rocsparse_index_base base)
    {
        const int64_t ld = (order == rocsparse_order_column) ? (int64_t)m : (int64_t)n;
        const size_t  sz = (order == rocsparse_order_column) ? (size_t)ld * n : (size_t)ld * m;
        const rocsparse_int diag = (m < n) ? m : n;

        std::vector<T> hA(sz, rocsparse_ut::scalar<T>(0.0f));
        for(rocsparse_int i = 0; i < diag; ++i)
        {
            const size_t idx = (order == rocsparse_order_column)
                                   ? (size_t)i + (size_t)i * (size_t)ld
                                   : (size_t)i * (size_t)ld + (size_t)i;
            hA[idx]          = rocsparse_ut::scalar<T>(1.0f);
        }
        device_vector<T> A{hA};
        ASSERT_TRUE(A.ptr);

        rocsparse_dnmat_descr dn = nullptr;
        ASSERT_EQ(rocsparse_create_dnmat_descr(&dn, m, n, ld, A.ptr, dt_of<T>(), order),
                  rocsparse_status_success);

        device_vector<rocsparse_int> offsets{(size_t)((m > n ? m : n) + 1)};
        ASSERT_TRUE(offsets.ptr);

        rocsparse_spmat_descr sp = nullptr;
        if(format == rocsparse_format_csr)
        {
            ASSERT_EQ(rocsparse_create_csr_descr(&sp,
                                                 m,
                                                 n,
                                                 0,
                                                 offsets.ptr,
                                                 nullptr,
                                                 nullptr,
                                                 it_of<int32_t>(),
                                                 it_of<int32_t>(),
                                                 base,
                                                 dt_of<T>()),
                      rocsparse_status_success);
        }
        else
        {
            ASSERT_EQ(rocsparse_create_csc_descr(&sp,
                                                 m,
                                                 n,
                                                 0,
                                                 offsets.ptr,
                                                 nullptr,
                                                 nullptr,
                                                 it_of<int32_t>(),
                                                 it_of<int32_t>(),
                                                 base,
                                                 dt_of<T>()),
                      rocsparse_status_success);
        }

        size_t buffer_size = 0;
        ASSERT_EQ(rocsparse_dense_to_sparse(
                      handle, dn, sp, rocsparse_dense_to_sparse_alg_default, &buffer_size, nullptr),
                  rocsparse_status_success);
        device_vector<char> buffer{buffer_size ? buffer_size : size_t(1)};
        ASSERT_TRUE(buffer.ptr);

        ASSERT_EQ(rocsparse_dense_to_sparse(
                      handle, dn, sp, rocsparse_dense_to_sparse_alg_default, nullptr, buffer.ptr),
                  rocsparse_status_success);

        int64_t rows = 0, cols = 0, nnz = 0;
        ASSERT_EQ(rocsparse_spmat_get_size(sp, &rows, &cols, &nnz), rocsparse_status_success);
        EXPECT_EQ(nnz, diag);

        device_vector<rocsparse_int> ind{(size_t)nnz};
        device_vector<T>             val{(size_t)nnz};
        ASSERT_TRUE(ind.ptr && val.ptr);
        if(format == rocsparse_format_csr)
        {
            ASSERT_EQ(rocsparse_csr_set_pointers(sp, offsets.ptr, ind.ptr, val.ptr),
                      rocsparse_status_success);
        }
        else
        {
            ASSERT_EQ(rocsparse_csc_set_pointers(sp, offsets.ptr, ind.ptr, val.ptr),
                      rocsparse_status_success);
        }

        ASSERT_EQ(
            rocsparse_dense_to_sparse(
                handle, dn, sp, rocsparse_dense_to_sparse_alg_default, &buffer_size, buffer.ptr),
            rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        EXPECT_EQ(rocsparse_destroy_spmat_descr(sp), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_dnmat_descr(dn), rocsparse_status_success);
    }

    template <typename T>
    static void check_dense2coo_quickreturn(rocsparse_handle handle)
    {
        rocsparse_mat_descr descr = nullptr;
        ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);

        // m == 0 quick return (dense2coo_checkarg returns success -> the typed C
        // wrapper takes its `status != continue` success path and returns).
        EXPECT_EQ(
            ut_dense2coo<T>(
                handle, 0, 4, descr, (const T*)nullptr, 0, nullptr, nullptr, nullptr, nullptr),
            rocsparse_status_success);
        // n == 0 quick return (ld >= m required by the earlier ld check).
        EXPECT_EQ(
            ut_dense2coo<T>(
                handle, 3, 0, descr, (const T*)nullptr, 3, nullptr, nullptr, nullptr, nullptr),
            rocsparse_status_success);

        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
    }
} // namespace

// Larger f64 / complex conversions in both memory orders. Drives the
// dense2csr (direction_row) and dense2csc (direction_column) kernels through
// the generic dense_to_sparse path at sizes that launch multiple blocks. (The
// WF_SIZE==64 else branch remains wave64-only; see note above.)
TEST_F(ConversionDense2csxOrder, generic_large_f64_complex)
{
    const rocsparse_int m = 48, n = 40;
    for(rocsparse_order order : {rocsparse_order_column, rocsparse_order_row})
    {
        for(rocsparse_format fmt : {rocsparse_format_csr, rocsparse_format_csc})
        {
            check_dense_to_sparse_sized<double>(
                handle, order, fmt, m, n, rocsparse_index_base_zero);
            check_dense_to_sparse_sized<rocsparse_float_complex>(
                handle, order, fmt, m, n, rocsparse_index_base_one);
            check_dense_to_sparse_sized<rocsparse_double_complex>(
                handle, order, fmt, m, n, rocsparse_index_base_zero);
        }
    }
}

// dense2coo quick-return guard (m == 0 / n == 0) for every precision, hitting
// the checkarg early-out and each typed wrapper's success return.
TEST_F(ConversionDense2csxOrder, dense2coo_quickreturn_all_types)
{
    check_dense2coo_quickreturn<float>(handle);
    check_dense2coo_quickreturn<double>(handle);
    check_dense2coo_quickreturn<rocsparse_float_complex>(handle);
    check_dense2coo_quickreturn<rocsparse_double_complex>(handle);
}

// Batched source descriptor handling in rocsparse_create_sparse_to_sparse_descr:
// a batched source is accepted (batched flag set) when both batch strides are
// zero, and rejected with not_implemented when either stride is positive.
TEST_F(ConversionSparseToSparse, batched_source_create)
{
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<float>         csr_val{std::vector<float>(3, 1.0f)};
    device_vector<rocsparse_int> coo_row{(size_t)3};
    device_vector<rocsparse_int> coo_col{(size_t)3};
    device_vector<float>         coo_val{(size_t)3};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && csr_val.ptr && coo_row.ptr && coo_col.ptr
                && coo_val.ptr);

    rocsparse_spmat_descr csr = nullptr, coo = nullptr;
    ASSERT_EQ(ut_make_csr(&csr, row_ptr, col_ind, csr_val), rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_coo_descr(&coo,
                                         3,
                                         3,
                                         3,
                                         coo_row.ptr,
                                         coo_col.ptr,
                                         coo_val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    // Accepted: batched source with zero strides.
    ASSERT_EQ(rocsparse_csr_set_strided_batch(csr, 2, 0, 0), rocsparse_status_success);
    rocsparse_sparse_to_sparse_descr s2s = nullptr;
    EXPECT_EQ(rocsparse_create_sparse_to_sparse_descr(
                  &s2s, csr, coo, rocsparse_sparse_to_sparse_alg_default),
              rocsparse_status_success);
    ASSERT_NE(s2s, nullptr);
    EXPECT_EQ(rocsparse_destroy_sparse_to_sparse_descr(s2s), rocsparse_status_success);

    // Rejected: positive offsets batch stride -> not_implemented.
    ASSERT_EQ(rocsparse_csr_set_strided_batch(csr, 2, 8, 0), rocsparse_status_success);
    s2s = nullptr;
    EXPECT_EQ(rocsparse_create_sparse_to_sparse_descr(
                  &s2s, csr, coo, rocsparse_sparse_to_sparse_alg_default),
              rocsparse_status_not_implemented);
    if(s2s != nullptr)
        EXPECT_EQ(rocsparse_destroy_sparse_to_sparse_descr(s2s), rocsparse_status_success);

    // Rejected: positive columns/values batch stride -> not_implemented.
    ASSERT_EQ(rocsparse_csr_set_strided_batch(csr, 2, 0, 8), rocsparse_status_success);
    s2s = nullptr;
    EXPECT_EQ(rocsparse_create_sparse_to_sparse_descr(
                  &s2s, csr, coo, rocsparse_sparse_to_sparse_alg_default),
              rocsparse_status_not_implemented);
    if(s2s != nullptr)
        EXPECT_EQ(rocsparse_destroy_sparse_to_sparse_descr(s2s), rocsparse_status_success);

    EXPECT_EQ(rocsparse_destroy_spmat_descr(csr), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(coo), rocsparse_status_success);
}

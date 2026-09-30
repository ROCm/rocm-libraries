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

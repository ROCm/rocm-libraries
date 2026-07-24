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
// Host-path unit tests for conversion-family index utilities that are simple to
// drive with tiny valid inputs: csr2coo, coo2csr, identity permutation, and the
// csrsort / coosort buffer-size + sort entry points. These exercise the host
// dispatch + validation code in library/src/conversion.
//
#include "unit_test_utils.hpp"

using namespace rocsparse_ut;

class Conversion : public HandleTest
{
};

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

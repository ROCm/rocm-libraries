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
// Host-path unit tests for the extra sub-lib. Drives a full CSR csrgeam pipeline
// (C = alpha*A + beta*B) on tiny 3x3 identity matrices plus its validation
// guards. Exercises host dispatch/nnz/compute code in library/src/extra.
//
#include "unit_test_utils.hpp"

using namespace rocsparse_ut;

class Extra : public HandleTest
{
};

TEST_F(Extra, csrgeam_full_pipeline)
{
    const rocsparse_int m = 3, n = 3, nnz = 3;
    // A = B = 3x3 identity in CSR.
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<float>         val{std::vector<float>{1, 1, 1}};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);

    device_vector<rocsparse_int> row_ptr_C{(size_t)(m + 1)};
    ASSERT_TRUE(row_ptr_C.ptr);

    rocsparse_int nnz_C = 0;
    ASSERT_EQ(rocsparse_csrgeam_nnz(handle,
                                    m,
                                    n,
                                    descr,
                                    nnz,
                                    row_ptr,
                                    col_ind,
                                    descr,
                                    nnz,
                                    row_ptr,
                                    col_ind,
                                    descr,
                                    row_ptr_C,
                                    &nnz_C),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(nnz_C, 3);

    device_vector<rocsparse_int> col_ind_C{(size_t)nnz_C};
    device_vector<float>         val_C{(size_t)nnz_C};
    ASSERT_TRUE(col_ind_C.ptr && val_C.ptr);

    const float alpha = 1.0f, beta = 1.0f;
    EXPECT_EQ(rocsparse_scsrgeam(handle,
                                 m,
                                 n,
                                 &alpha,
                                 descr,
                                 nnz,
                                 val,
                                 row_ptr,
                                 col_ind,
                                 &beta,
                                 descr,
                                 nnz,
                                 val,
                                 row_ptr,
                                 col_ind,
                                 descr,
                                 val_C,
                                 row_ptr_C,
                                 col_ind_C),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

TEST_F(Extra, csrgeam_nnz_bad_args)
{
    const rocsparse_int          m = 3, n = 3, nnz = 3;
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<rocsparse_int> row_ptr_C{(size_t)(m + 1)};
    rocsparse_mat_descr          descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    rocsparse_int nnz_C = 0;

    EXPECT_EQ(rocsparse_csrgeam_nnz(nullptr,
                                    m,
                                    n,
                                    descr,
                                    nnz,
                                    row_ptr,
                                    col_ind,
                                    descr,
                                    nnz,
                                    row_ptr,
                                    col_ind,
                                    descr,
                                    row_ptr_C,
                                    &nnz_C),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_csrgeam_nnz(handle,
                                    -1,
                                    n,
                                    descr,
                                    nnz,
                                    row_ptr,
                                    col_ind,
                                    descr,
                                    nnz,
                                    row_ptr,
                                    col_ind,
                                    descr,
                                    row_ptr_C,
                                    &nnz_C),
              rocsparse_status_invalid_size);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

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
// Host-path unit tests for the precond sub-lib. Drives a full csrilu0
// buffer_size -> analysis -> compute -> clear pipeline on a tiny well-conditioned
// diagonal matrix (ILU0 of a diagonal is trivially the diagonal, no zero pivot),
// plus mat_info lifecycle and validation guards. Exercises host dispatch/analysis
// code in library/src/precond.
//
#include "unit_test_utils.hpp"

using namespace rocsparse_ut;

class Precond : public HandleTest
{
};

TEST_F(Precond, mat_info_lifecycle)
{
    rocsparse_mat_info info = nullptr;
    EXPECT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
    EXPECT_EQ(rocsparse_create_mat_info(nullptr), rocsparse_status_invalid_pointer);
}

TEST_F(Precond, csrilu0_full_pipeline)
{
    const rocsparse_int m = 3, nnz = 3;
    // Diagonal matrix diag(2,3,4) in CSR (well-conditioned, no zero pivot).
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<float>         val{std::vector<float>{2, 3, 4}};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    rocsparse_mat_info info = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

    size_t buffer_size = 0;
    ASSERT_EQ(rocsparse_scsrilu0_buffer_size(
                  handle, m, nnz, descr, val, row_ptr, col_ind, info, &buffer_size),
              rocsparse_status_success);
    EXPECT_GT(buffer_size, 0u);

    device_vector<char> buffer{buffer_size};
    ASSERT_TRUE(buffer.ptr);

    ASSERT_EQ(rocsparse_scsrilu0_analysis(handle,
                                          m,
                                          nnz,
                                          descr,
                                          val,
                                          row_ptr,
                                          col_ind,
                                          info,
                                          rocsparse_analysis_policy_reuse,
                                          rocsparse_solve_policy_auto,
                                          buffer.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    EXPECT_EQ(rocsparse_scsrilu0(handle,
                                 m,
                                 nnz,
                                 descr,
                                 val,
                                 row_ptr,
                                 col_ind,
                                 info,
                                 rocsparse_solve_policy_auto,
                                 buffer.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    EXPECT_EQ(rocsparse_csrilu0_clear(handle, info), rocsparse_status_success);

    // bad args on buffer_size
    EXPECT_EQ(rocsparse_scsrilu0_buffer_size(
                  nullptr, m, nnz, descr, val, row_ptr, col_ind, info, &buffer_size),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_scsrilu0_buffer_size(
                  handle, -1, nnz, descr, val, row_ptr, col_ind, info, &buffer_size),
              rocsparse_status_invalid_size);

    EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

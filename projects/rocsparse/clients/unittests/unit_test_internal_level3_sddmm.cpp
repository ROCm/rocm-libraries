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
// Host-path unit tests for rocsparse_sddmm across the *non-CSR* sparse output
// formats: COO, CSC, ELL and COO-AoS. The existing generic suite exercised only
// the CSR path, so the format dispatch in library/src/level3/rocsparse_sddmm_{coo,
// csc,ell,coo_aos}.cpp sat at 0%. Each test runs the full buffer_size ->
// preprocess -> compute pipeline with C = 3x3 identity and dense A,B = ones.
//
#include "unit_test_utils.hpp"

#include "rocsparse.h"

using namespace rocsparse_ut;

namespace
{
    constexpr rocsparse_indextype  IT   = rocsparse_indextype_i32;
    constexpr rocsparse_index_base BASE = rocsparse_index_base_zero;
    constexpr rocsparse_datatype   DT   = rocsparse_datatype_f32_r;

    // Run the sddmm pipeline for a given C spmat descriptor (A,B dense 3x3 ones)
    // and verify the sampled result. C = alpha*(A*B) sampled at C's sparsity
    // pattern + beta*C. With A = B = ones(3x3), (A*B)[i][j] = 3 for every (i,j);
    // with alpha = 1, beta = 0 every stored value of C becomes exactly 3.
    // `c_val` / `c_nnz` reference the value array backing matC so the numeric
    // output can be read back and checked.
    void run_sddmm(rocsparse_handle handle, rocsparse_spmat_descr matC, float* c_val, size_t c_nnz)
    {
        device_vector<float> A{std::vector<float>(3 * 3, 1.0f)};
        device_vector<float> B{std::vector<float>(3 * 3, 1.0f)};
        ASSERT_TRUE(A.ptr && B.ptr);

        rocsparse_dnmat_descr mA = nullptr, mB = nullptr;
        ASSERT_EQ(rocsparse_create_dnmat_descr(&mA, 3, 3, 3, A.ptr, DT, rocsparse_order_column),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnmat_descr(&mB, 3, 3, 3, B.ptr, DT, rocsparse_order_column),
                  rocsparse_status_success);

        const float alpha = 1.0f, beta = 0.0f;
        size_t      buffer_size = 0;
        ASSERT_EQ(rocsparse_sddmm_buffer_size(handle,
                                              rocsparse_operation_none,
                                              rocsparse_operation_none,
                                              &alpha,
                                              mA,
                                              mB,
                                              &beta,
                                              matC,
                                              DT,
                                              rocsparse_sddmm_alg_default,
                                              &buffer_size),
                  rocsparse_status_success);

        device_vector<char> tmp{buffer_size ? buffer_size : size_t(1)};
        ASSERT_TRUE(tmp.ptr);

        ASSERT_EQ(rocsparse_sddmm_preprocess(handle,
                                             rocsparse_operation_none,
                                             rocsparse_operation_none,
                                             &alpha,
                                             mA,
                                             mB,
                                             &beta,
                                             matC,
                                             DT,
                                             rocsparse_sddmm_alg_default,
                                             tmp.ptr),
                  rocsparse_status_success);

        ASSERT_EQ(rocsparse_sddmm(handle,
                                  rocsparse_operation_none,
                                  rocsparse_operation_none,
                                  &alpha,
                                  mA,
                                  mB,
                                  &beta,
                                  matC,
                                  DT,
                                  rocsparse_sddmm_alg_default,
                                  tmp.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        // Every sampled entry equals (A*B)[i][j] = 3.
        auto hc = to_host(c_val, c_nnz);
        for(size_t i = 0; i < c_nnz; ++i)
        {
            EXPECT_FLOAT_EQ(hc[i], 3.0f) << "sampled value " << i;
        }

        EXPECT_EQ(rocsparse_destroy_dnmat_descr(mA), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_dnmat_descr(mB), rocsparse_status_success);
    }
}

class Sddmm : public HandleTest
{
};

// sddmm into a COO C (3x3 diagonal pattern): each of the 3 stored values must
// become 3 (see run_sddmm).
TEST_F(Sddmm, coo)
{
    device_vector<int32_t> row_ind{std::vector<int32_t>{0, 1, 2}};
    device_vector<int32_t> col_ind{std::vector<int32_t>{0, 1, 2}};
    device_vector<float>   val{std::vector<float>{1.0f, 1.0f, 1.0f}};
    ASSERT_TRUE(row_ind.ptr && col_ind.ptr && val.ptr);

    rocsparse_spmat_descr matC = nullptr;
    ASSERT_EQ(rocsparse_create_coo_descr(&matC, 3, 3, 3, row_ind, col_ind, val, IT, BASE, DT),
              rocsparse_status_success);
    run_sddmm(handle, matC, val.ptr, 3);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(matC), rocsparse_status_success);
}

// sddmm into a CSC C (3x3 diagonal pattern): each stored value must become 3.
TEST_F(Sddmm, csc)
{
    device_vector<int32_t> col_ptr{std::vector<int32_t>{0, 1, 2, 3}};
    device_vector<int32_t> row_ind{std::vector<int32_t>{0, 1, 2}};
    device_vector<float>   val{std::vector<float>{1.0f, 1.0f, 1.0f}};
    ASSERT_TRUE(col_ptr.ptr && row_ind.ptr && val.ptr);

    rocsparse_spmat_descr matC = nullptr;
    ASSERT_EQ(rocsparse_create_csc_descr(&matC, 3, 3, 3, col_ptr, row_ind, val, IT, IT, BASE, DT),
              rocsparse_status_success);
    run_sddmm(handle, matC, val.ptr, 3);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(matC), rocsparse_status_success);
}

// sddmm into an ELL C (ell_width 1, diagonal pattern): each stored value -> 3.
TEST_F(Sddmm, ell)
{
    device_vector<int32_t> col_ind{std::vector<int32_t>{0, 1, 2}};
    device_vector<float>   val{std::vector<float>{1.0f, 1.0f, 1.0f}};
    ASSERT_TRUE(col_ind.ptr && val.ptr);

    rocsparse_spmat_descr matC = nullptr;
    ASSERT_EQ(rocsparse_create_ell_descr(&matC, 3, 3, col_ind, val, 1, IT, BASE, DT),
              rocsparse_status_success);
    run_sddmm(handle, matC, val.ptr, 3);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(matC), rocsparse_status_success);
}

// sddmm into a COO-AoS C (interleaved (row,col), diagonal pattern): values -> 3.
TEST_F(Sddmm, coo_aos)
{
    // Array-of-structs COO: interleaved (row,col) pairs.
    device_vector<int32_t> ind{std::vector<int32_t>{0, 0, 1, 1, 2, 2}};
    device_vector<float>   val{std::vector<float>{1.0f, 1.0f, 1.0f}};
    ASSERT_TRUE(ind.ptr && val.ptr);

    rocsparse_spmat_descr matC = nullptr;
    ASSERT_EQ(rocsparse_create_coo_aos_descr(&matC, 3, 3, 3, ind, val, IT, BASE, DT),
              rocsparse_status_success);
    run_sddmm(handle, matC, val.ptr, 3);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(matC), rocsparse_status_success);
}

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
// Extra rocsparse_sddmm coverage:
//   * default algorithm k-dispatch (k = 1, 2, 5), transpose, row order, int64
//     indices, device pointer mode, and a 2-batch CSR launch
//   * alg_dense for CSR, CSC, ELL and COO-AoS. alg_dense converts C to dense
//     (ell2dense / coo2dense_aos / csx2dense) and then calls rocBLAS gemm.
//     A build without rocBLAS returns not_implemented after that conversion;
//     the post-gemm sample kernel runs only when rocBLAS is linked.
//
#include "unit_test_utils.hpp"

#include "rocsparse.h"

#include <vector>

using namespace rocsparse_ut;

namespace
{
    // A and B are dense ones. With trans = none, A is m x k and B is k x n, so
    // every entry of A*B equals k. alpha/beta scale the sampled result.
    rocsparse_status run_ones(rocsparse_handle        handle,
                              rocsparse_operation     trans_A,
                              rocsparse_operation     trans_B,
                              rocsparse_order         order_A,
                              rocsparse_order         order_B,
                              int64_t                 m,
                              int64_t                 n,
                              int64_t                 k,
                              rocsparse_spmat_descr   C,
                              float                   alpha,
                              float                   beta,
                              rocsparse_sddmm_alg     alg,
                              const void*             alpha_ptr = nullptr,
                              const void*             beta_ptr  = nullptr)
    {
        const int64_t a_rows = (trans_A == rocsparse_operation_none) ? m : k;
        const int64_t a_cols = (trans_A == rocsparse_operation_none) ? k : m;
        const int64_t b_rows = (trans_B == rocsparse_operation_none) ? k : n;
        const int64_t b_cols = (trans_B == rocsparse_operation_none) ? n : k;
        const int64_t lda    = (order_A == rocsparse_order_column) ? a_rows : a_cols;
        const int64_t ldb    = (order_B == rocsparse_order_column) ? b_rows : b_cols;

        device_vector<float> A{std::vector<float>(size_t(a_rows * a_cols), 1.0f)};
        device_vector<float> B{std::vector<float>(size_t(b_rows * b_cols), 1.0f)};
        if(!A.ptr || !B.ptr)
            return rocsparse_status_memory_error;

        rocsparse_dnmat_descr mA = nullptr, mB = nullptr;
        rocsparse_status      st
            = rocsparse_create_dnmat_descr(&mA, a_rows, a_cols, lda, A.ptr, rocsparse_datatype_f32_r, order_A);
        if(st != rocsparse_status_success)
            return st;
        st = rocsparse_create_dnmat_descr(&mB, b_rows, b_cols, ldb, B.ptr, rocsparse_datatype_f32_r, order_B);
        if(st != rocsparse_status_success)
        {
            (void)rocsparse_destroy_dnmat_descr(mA);
            return st;
        }

        const float  alpha_host = alpha;
        const float  beta_host  = beta;
        const void*  a_ptr      = alpha_ptr ? alpha_ptr : static_cast<const void*>(&alpha_host);
        const void*  b_ptr      = beta_ptr ? beta_ptr : static_cast<const void*>(&beta_host);
        size_t       buffer_size = 0;
        st                       = rocsparse_sddmm_buffer_size(handle,
                                         trans_A,
                                         trans_B,
                                         a_ptr,
                                         mA,
                                         mB,
                                         b_ptr,
                                         C,
                                         rocsparse_datatype_f32_r,
                                         alg,
                                         &buffer_size);
        device_vector<char> tmp{buffer_size ? buffer_size : size_t(1)};
        if(st == rocsparse_status_success && tmp.ptr)
        {
            st = rocsparse_sddmm_preprocess(handle,
                                            trans_A,
                                            trans_B,
                                            a_ptr,
                                            mA,
                                            mB,
                                            b_ptr,
                                            C,
                                            rocsparse_datatype_f32_r,
                                            alg,
                                            tmp.ptr);
        }
        if(st == rocsparse_status_success)
        {
            st = rocsparse_sddmm(handle,
                                 trans_A,
                                 trans_B,
                                 a_ptr,
                                 mA,
                                 mB,
                                 b_ptr,
                                 C,
                                 rocsparse_datatype_f32_r,
                                 alg,
                                 tmp.ptr);
        }
        (void)rocsparse_destroy_dnmat_descr(mA);
        (void)rocsparse_destroy_dnmat_descr(mB);
        if(st == rocsparse_status_success)
            (void)hipDeviceSynchronize();
        return st;
    }

    void expect_filled(const float* values, size_t n, float expected)
    {
        auto host = to_host(values, n);
        for(size_t i = 0; i < n; ++i)
            EXPECT_FLOAT_EQ(host[i], expected) << "index " << i;
    }
} // namespace

class SddmmBranches : public HandleTest
{
};

// Default algorithm: the k > 4 / k > 2 / k > 1 / else launch ladder on CSR and
// CSC, plus the alpha == 0 && beta == 1 early return.
TEST_F(SddmmBranches, default_k_dispatch)
{
    const int ks[] = {1, 2, 5};
    for(int k : ks)
    {
        device_vector<int32_t> row_ptr{std::vector<int32_t>{0, 1, 2}};
        device_vector<int32_t> col_ind{std::vector<int32_t>{0, 1}};
        device_vector<float>   val{std::vector<float>{0.0f, 0.0f}};
        ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);
        rocsparse_spmat_descr csr = nullptr;
        ASSERT_EQ(rocsparse_create_csr_descr(&csr,
                                             2,
                                             2,
                                             2,
                                             row_ptr.ptr,
                                             col_ind.ptr,
                                             val.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(run_ones(handle,
                           rocsparse_operation_none,
                           rocsparse_operation_none,
                           rocsparse_order_column,
                           rocsparse_order_column,
                           2,
                           2,
                           k,
                           csr,
                           1.0f,
                           0.0f,
                           rocsparse_sddmm_alg_default),
                  rocsparse_status_success);
        expect_filled(val.ptr, 2, float(k));
        EXPECT_EQ(rocsparse_destroy_spmat_descr(csr), rocsparse_status_success);

        device_vector<int32_t> csc_ptr{std::vector<int32_t>{0, 1, 2}};
        device_vector<int32_t> csc_row{std::vector<int32_t>{0, 1}};
        device_vector<float>   csc_val{std::vector<float>{0.0f, 0.0f}};
        ASSERT_TRUE(csc_ptr.ptr && csc_row.ptr && csc_val.ptr);
        rocsparse_spmat_descr csc = nullptr;
        ASSERT_EQ(rocsparse_create_csc_descr(&csc,
                                             2,
                                             2,
                                             2,
                                             csc_ptr.ptr,
                                             csc_row.ptr,
                                             csc_val.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        ASSERT_EQ(run_ones(handle,
                           rocsparse_operation_none,
                           rocsparse_operation_none,
                           rocsparse_order_column,
                           rocsparse_order_column,
                           2,
                           2,
                           k,
                           csc,
                           1.0f,
                           0.0f,
                           rocsparse_sddmm_alg_default),
                  rocsparse_status_success);
        expect_filled(csc_val.ptr, 2, float(k));
        EXPECT_EQ(rocsparse_destroy_spmat_descr(csc), rocsparse_status_success);
    }

    device_vector<int32_t> row_ptr{std::vector<int32_t>{0, 1, 2}};
    device_vector<int32_t> col_ind{std::vector<int32_t>{0, 1}};
    device_vector<float>   val{std::vector<float>{4.0f, 4.0f}};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);
    rocsparse_spmat_descr csr = nullptr;
    ASSERT_EQ(rocsparse_create_csr_descr(&csr,
                                         2,
                                         2,
                                         2,
                                         row_ptr.ptr,
                                         col_ind.ptr,
                                         val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);
    ASSERT_EQ(run_ones(handle,
                       rocsparse_operation_none,
                       rocsparse_operation_none,
                       rocsparse_order_column,
                       rocsparse_order_column,
                       2,
                       2,
                       2,
                       csr,
                       0.0f,
                       1.0f,
                       rocsparse_sddmm_alg_default),
              rocsparse_status_success);
    expect_filled(val.ptr, 2, 4.0f);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(csr), rocsparse_status_success);
}

// Transpose selects mat_A->rows as k. Row-major dense inputs take the other
// order adjustment. int64 indices select that sddmm instantiation.
TEST_F(SddmmBranches, transpose_order_and_i64)
{
    device_vector<int32_t> row_ptr{std::vector<int32_t>{0, 1, 2}};
    device_vector<int32_t> col_ind{std::vector<int32_t>{0, 1}};
    device_vector<float>   val{std::vector<float>{0.0f, 0.0f}};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);
    rocsparse_spmat_descr csr = nullptr;
    ASSERT_EQ(rocsparse_create_csr_descr(&csr,
                                         2,
                                         2,
                                         2,
                                         row_ptr.ptr,
                                         col_ind.ptr,
                                         val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);
    // A is stored k x m = 1 x 2 and transposed, so the product is still ones.
    ASSERT_EQ(run_ones(handle,
                       rocsparse_operation_transpose,
                       rocsparse_operation_none,
                       rocsparse_order_column,
                       rocsparse_order_column,
                       2,
                       2,
                       1,
                       csr,
                       1.0f,
                       0.0f,
                       rocsparse_sddmm_alg_default),
              rocsparse_status_success);
    expect_filled(val.ptr, 2, 1.0f);

    ASSERT_EQ(hipMemset(val.ptr, 0, sizeof(float) * 2), hipSuccess);
    ASSERT_EQ(run_ones(handle,
                       rocsparse_operation_none,
                       rocsparse_operation_transpose,
                       rocsparse_order_row,
                       rocsparse_order_row,
                       2,
                       2,
                       2,
                       csr,
                       1.0f,
                       0.0f,
                       rocsparse_sddmm_alg_default),
              rocsparse_status_success);
    expect_filled(val.ptr, 2, 2.0f);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(csr), rocsparse_status_success);

    device_vector<int64_t> row64{std::vector<int64_t>{0, 1, 2}};
    device_vector<int64_t> col64{std::vector<int64_t>{0, 1}};
    device_vector<float>   val64{std::vector<float>{0.0f, 0.0f}};
    ASSERT_TRUE(row64.ptr && col64.ptr && val64.ptr);
    rocsparse_spmat_descr csr64 = nullptr;
    ASSERT_EQ(rocsparse_create_csr_descr(&csr64,
                                         2,
                                         2,
                                         2,
                                         row64.ptr,
                                         col64.ptr,
                                         val64.ptr,
                                         rocsparse_indextype_i64,
                                         rocsparse_indextype_i64,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);
    ASSERT_EQ(run_ones(handle,
                       rocsparse_operation_none,
                       rocsparse_operation_none,
                       rocsparse_order_column,
                       rocsparse_order_column,
                       2,
                       2,
                       1,
                       csr64,
                       1.0f,
                       0.0f,
                       rocsparse_sddmm_alg_default),
              rocsparse_status_success);
    expect_filled(val64.ptr, 2, 1.0f);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(csr64), rocsparse_status_success);

    // Double, k > 4, so the double instantiation of the CSR default ladder runs.
    device_vector<int32_t> d_ptr{std::vector<int32_t>{0, 1, 2}};
    device_vector<int32_t> d_col{std::vector<int32_t>{0, 1}};
    device_vector<double>  d_val{std::vector<double>{0.0, 0.0}};
    ASSERT_TRUE(d_ptr.ptr && d_col.ptr && d_val.ptr);
    rocsparse_spmat_descr dcsr = nullptr;
    ASSERT_EQ(rocsparse_create_csr_descr(&dcsr,
                                         2,
                                         2,
                                         2,
                                         d_ptr.ptr,
                                         d_col.ptr,
                                         d_val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f64_r),
              rocsparse_status_success);
    device_vector<double> A{std::vector<double>(size_t(2 * 5), 1.0)};
    device_vector<double> B{std::vector<double>(size_t(5 * 2), 1.0)};
    ASSERT_TRUE(A.ptr && B.ptr);
    rocsparse_dnmat_descr mA = nullptr, mB = nullptr;
    ASSERT_EQ(rocsparse_create_dnmat_descr(&mA, 2, 5, 2, A.ptr, rocsparse_datatype_f64_r, rocsparse_order_column),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_dnmat_descr(&mB, 5, 2, 5, B.ptr, rocsparse_datatype_f64_r, rocsparse_order_column),
              rocsparse_status_success);
    const double one = 1.0, zero = 0.0;
    size_t       buffer_size = 0;
    ASSERT_EQ(rocsparse_sddmm_buffer_size(handle,
                                          rocsparse_operation_none,
                                          rocsparse_operation_none,
                                          &one,
                                          mA,
                                          mB,
                                          &zero,
                                          dcsr,
                                          rocsparse_datatype_f64_r,
                                          rocsparse_sddmm_alg_default,
                                          &buffer_size),
              rocsparse_status_success);
    device_vector<char> tmp{buffer_size ? buffer_size : size_t(1)};
    ASSERT_TRUE(tmp.ptr);
    ASSERT_EQ(rocsparse_sddmm_preprocess(handle,
                                        rocsparse_operation_none,
                                        rocsparse_operation_none,
                                        &one,
                                        mA,
                                        mB,
                                        &zero,
                                        dcsr,
                                        rocsparse_datatype_f64_r,
                                        rocsparse_sddmm_alg_default,
                                        tmp.ptr),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_sddmm(handle,
                              rocsparse_operation_none,
                              rocsparse_operation_none,
                              &one,
                              mA,
                              mB,
                              &zero,
                              dcsr,
                              rocsparse_datatype_f64_r,
                              rocsparse_sddmm_alg_default,
                              tmp.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    auto hd = to_host(d_val.ptr, 2);
    EXPECT_DOUBLE_EQ(hd[0], 5.0);
    EXPECT_DOUBLE_EQ(hd[1], 5.0);
    EXPECT_EQ(rocsparse_destroy_dnmat_descr(mA), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_dnmat_descr(mB), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(dcsr), rocsparse_status_success);
}

// Device pointer mode skips the host alpha/beta early-out. A second batch
// exercises the batched grid when ROCSPARSE_WITH_SDDMM_BATCHED is on.
TEST_F(SddmmBranches, pointer_mode_and_batch)
{
    device_vector<int32_t> row_ptr{std::vector<int32_t>{0, 1, 2}};
    device_vector<int32_t> col_ind{std::vector<int32_t>{0, 1}};
    device_vector<float>   val{std::vector<float>{0.0f, 0.0f}};
    device_vector<float>   alpha{std::vector<float>{1.0f}};
    device_vector<float>   beta{std::vector<float>{0.0f}};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr && alpha.ptr && beta.ptr);
    rocsparse_spmat_descr csr = nullptr;
    ASSERT_EQ(rocsparse_create_csr_descr(&csr,
                                         2,
                                         2,
                                         2,
                                         row_ptr.ptr,
                                         col_ind.ptr,
                                         val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device),
              rocsparse_status_success);
    ASSERT_EQ(run_ones(handle,
                       rocsparse_operation_none,
                       rocsparse_operation_none,
                       rocsparse_order_column,
                       rocsparse_order_column,
                       2,
                       2,
                       1,
                       csr,
                       0.0f,
                       0.0f,
                       rocsparse_sddmm_alg_default,
                       alpha.ptr,
                       beta.ptr),
              rocsparse_status_success);
    expect_filled(val.ptr, 2, 1.0f);
    ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(csr), rocsparse_status_success);

    // Two identical batches. Offsets stride is rows+1, values stride is nnz.
    device_vector<int32_t> b_ptr{std::vector<int32_t>{0, 1, 2, 0, 1, 2}};
    device_vector<int32_t> b_col{std::vector<int32_t>{0, 1, 0, 1}};
    device_vector<float>   b_val{std::vector<float>(4, 0.0f)};
    ASSERT_TRUE(b_ptr.ptr && b_col.ptr && b_val.ptr);
    rocsparse_spmat_descr bcsr = nullptr;
    ASSERT_EQ(rocsparse_create_csr_descr(&bcsr,
                                         2,
                                         2,
                                         2,
                                         b_ptr.ptr,
                                         b_col.ptr,
                                         b_val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_csr_set_strided_batch(bcsr, 2, 3, 2), rocsparse_status_success);

    device_vector<float> A{std::vector<float>(4, 1.0f)};
    device_vector<float> B{std::vector<float>(4, 1.0f)};
    ASSERT_TRUE(A.ptr && B.ptr);
    rocsparse_dnmat_descr mA = nullptr, mB = nullptr;
    ASSERT_EQ(rocsparse_create_dnmat_descr(&mA, 2, 1, 2, A.ptr, rocsparse_datatype_f32_r, rocsparse_order_column),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_dnmat_descr(&mB, 1, 2, 1, B.ptr, rocsparse_datatype_f32_r, rocsparse_order_column),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_dnmat_set_strided_batch(mA, 2, 2), rocsparse_status_success);
    ASSERT_EQ(rocsparse_dnmat_set_strided_batch(mB, 2, 2), rocsparse_status_success);
    const float one = 1.0f, zero = 0.0f;
    size_t      buffer_size = 0;
    rocsparse_status bst     = rocsparse_sddmm_buffer_size(handle,
                                                      rocsparse_operation_none,
                                                      rocsparse_operation_none,
                                                      &one,
                                                      mA,
                                                      mB,
                                                      &zero,
                                                      bcsr,
                                                      rocsparse_datatype_f32_r,
                                                      rocsparse_sddmm_alg_default,
                                                      &buffer_size);
    if(bst == rocsparse_status_success)
    {
        device_vector<char> tmp{buffer_size ? buffer_size : size_t(1)};
        ASSERT_TRUE(tmp.ptr);
        ASSERT_EQ(rocsparse_sddmm_preprocess(handle,
                                            rocsparse_operation_none,
                                            rocsparse_operation_none,
                                            &one,
                                            mA,
                                            mB,
                                            &zero,
                                            bcsr,
                                            rocsparse_datatype_f32_r,
                                            rocsparse_sddmm_alg_default,
                                            tmp.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_sddmm(handle,
                                  rocsparse_operation_none,
                                  rocsparse_operation_none,
                                  &one,
                                  mA,
                                  mB,
                                  &zero,
                                  bcsr,
                                  rocsparse_datatype_f32_r,
                                  rocsparse_sddmm_alg_default,
                                  tmp.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        expect_filled(b_val.ptr, 4, 1.0f);
    }
    else
    {
        EXPECT_EQ(bst, rocsparse_status_not_implemented);
    }
    EXPECT_EQ(rocsparse_destroy_dnmat_descr(mA), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_dnmat_descr(mB), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(bcsr), rocsparse_status_success);
}

// alg_dense. Null buffer is rejected before gemm. A real buffer runs the
// conversion (and the sample kernel when rocBLAS is present). Several average
// nnz values select different sample-kernel launch widths.
TEST_F(SddmmBranches, alg_dense)
{
    device_vector<int32_t> row_ptr{std::vector<int32_t>{0, 1, 2, 3}};
    device_vector<int32_t> col_ind{std::vector<int32_t>{0, 1, 2}};
    device_vector<float>   val{std::vector<float>{1.0f, 1.0f, 1.0f}};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);
    rocsparse_spmat_descr csr = nullptr;
    ASSERT_EQ(rocsparse_create_csr_descr(&csr,
                                         3,
                                         3,
                                         3,
                                         row_ptr.ptr,
                                         col_ind.ptr,
                                         val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);

    device_vector<float> A{std::vector<float>(9, 1.0f)};
    device_vector<float> B{std::vector<float>(9, 1.0f)};
    ASSERT_TRUE(A.ptr && B.ptr);
    rocsparse_dnmat_descr mA = nullptr, mB = nullptr;
    ASSERT_EQ(rocsparse_create_dnmat_descr(&mA, 3, 3, 3, A.ptr, rocsparse_datatype_f32_r, rocsparse_order_column),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_dnmat_descr(&mB, 3, 3, 3, B.ptr, rocsparse_datatype_f32_r, rocsparse_order_column),
              rocsparse_status_success);
    const float one = 1.0f, zero = 0.0f;
    size_t      buffer_size = 0;
    ASSERT_EQ(rocsparse_sddmm_buffer_size(handle,
                                          rocsparse_operation_none,
                                          rocsparse_operation_none,
                                          &one,
                                          mA,
                                          mB,
                                          &zero,
                                          csr,
                                          rocsparse_datatype_f32_r,
                                          rocsparse_sddmm_alg_dense,
                                          &buffer_size),
              rocsparse_status_success);
    EXPECT_GT(buffer_size, size_t(0));
    EXPECT_EQ(rocsparse_sddmm(handle,
                              rocsparse_operation_none,
                              rocsparse_operation_none,
                              &one,
                              mA,
                              mB,
                              &zero,
                              csr,
                              rocsparse_datatype_f32_r,
                              rocsparse_sddmm_alg_dense,
                              nullptr),
              rocsparse_status_invalid_pointer);

    device_vector<char> tmp{buffer_size};
    ASSERT_TRUE(tmp.ptr);
    ASSERT_EQ(rocsparse_sddmm_preprocess(handle,
                                        rocsparse_operation_none,
                                        rocsparse_operation_none,
                                        &one,
                                        mA,
                                        mB,
                                        &zero,
                                        csr,
                                        rocsparse_datatype_f32_r,
                                        rocsparse_sddmm_alg_dense,
                                        tmp.ptr),
              rocsparse_status_success);
    rocsparse_status st = rocsparse_sddmm(handle,
                                          rocsparse_operation_none,
                                          rocsparse_operation_none,
                                          &one,
                                          mA,
                                          mB,
                                          &zero,
                                          csr,
                                          rocsparse_datatype_f32_r,
                                          rocsparse_sddmm_alg_dense,
                                          tmp.ptr);
    EXPECT_TRUE(st == rocsparse_status_success || st == rocsparse_status_not_implemented) << st;
    if(st == rocsparse_status_success)
    {
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        expect_filled(val.ptr, 3, 3.0f);
    }
    EXPECT_EQ(rocsparse_destroy_dnmat_descr(mA), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_dnmat_descr(mB), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(csr), rocsparse_status_success);

    // CSC alg_dense, same 3x3 diagonal.
    device_vector<int32_t> csc_ptr{std::vector<int32_t>{0, 1, 2, 3}};
    device_vector<int32_t> csc_row{std::vector<int32_t>{0, 1, 2}};
    device_vector<float>   csc_val{std::vector<float>{1.0f, 1.0f, 1.0f}};
    ASSERT_TRUE(csc_ptr.ptr && csc_row.ptr && csc_val.ptr);
    rocsparse_spmat_descr csc = nullptr;
    ASSERT_EQ(rocsparse_create_csc_descr(&csc,
                                         3,
                                         3,
                                         3,
                                         csc_ptr.ptr,
                                         csc_row.ptr,
                                         csc_val.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);
    st = run_ones(handle,
                  rocsparse_operation_none,
                  rocsparse_operation_none,
                  rocsparse_order_column,
                  rocsparse_order_column,
                  3,
                  3,
                  3,
                  csc,
                  1.0f,
                  0.0f,
                  rocsparse_sddmm_alg_dense);
    EXPECT_TRUE(st == rocsparse_status_success || st == rocsparse_status_not_implemented) << st;
    if(st == rocsparse_status_success)
        expect_filled(csc_val.ptr, 3, 3.0f);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(csc), rocsparse_status_success);

    // ELL alg_dense runs ell2dense before gemm.
    device_vector<int32_t> ell_col{std::vector<int32_t>{0, 1, 2}};
    device_vector<float>   ell_val{std::vector<float>{1.0f, 1.0f, 1.0f}};
    ASSERT_TRUE(ell_col.ptr && ell_val.ptr);
    rocsparse_spmat_descr ell = nullptr;
    ASSERT_EQ(rocsparse_create_ell_descr(&ell,
                                         3,
                                         3,
                                         ell_col.ptr,
                                         ell_val.ptr,
                                         1,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);
    st = run_ones(handle,
                  rocsparse_operation_none,
                  rocsparse_operation_none,
                  rocsparse_order_column,
                  rocsparse_order_column,
                  3,
                  3,
                  3,
                  ell,
                  1.0f,
                  0.0f,
                  rocsparse_sddmm_alg_dense);
    EXPECT_TRUE(st == rocsparse_status_success || st == rocsparse_status_not_implemented) << st;
    if(st == rocsparse_status_success)
        expect_filled(ell_val.ptr, 3, 3.0f);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(ell), rocsparse_status_success);

    // COO-AoS alg_dense runs coo2dense_aos before gemm.
    device_vector<int32_t> ind{std::vector<int32_t>{0, 0, 1, 1, 2, 2}};
    device_vector<float>   aos_val{std::vector<float>{1.0f, 1.0f, 1.0f}};
    ASSERT_TRUE(ind.ptr && aos_val.ptr);
    rocsparse_spmat_descr aos = nullptr;
    ASSERT_EQ(rocsparse_create_coo_aos_descr(&aos,
                                             3,
                                             3,
                                             3,
                                             ind.ptr,
                                             aos_val.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
              rocsparse_status_success);
    st = run_ones(handle,
                  rocsparse_operation_none,
                  rocsparse_operation_none,
                  rocsparse_order_column,
                  rocsparse_order_column,
                  3,
                  3,
                  3,
                  aos,
                  1.0f,
                  0.0f,
                  rocsparse_sddmm_alg_dense);
    EXPECT_TRUE(st == rocsparse_status_success || st == rocsparse_status_not_implemented) << st;
    if(st == rocsparse_status_success)
        expect_filled(aos_val.ptr, 3, 3.0f);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(aos), rocsparse_status_success);

    // nnz == 0 buffer-size query hits the dense quick return inside the template.
    device_vector<int32_t> zptr{std::vector<int32_t>{0, 0}};
    ASSERT_TRUE(zptr.ptr);
    rocsparse_spmat_descr empty = nullptr;
    ASSERT_EQ(rocsparse_create_csr_descr(&empty,
                                         1,
                                         1,
                                         0,
                                         zptr.ptr,
                                         zptr.ptr,
                                         zptr.ptr,
                                         rocsparse_indextype_i32,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocsparse_datatype_f32_r),
              rocsparse_status_success);
    device_vector<float> one_a{std::vector<float>{1.0f}};
    ASSERT_TRUE(one_a.ptr);
    rocsparse_dnmat_descr eA = nullptr, eB = nullptr;
    ASSERT_EQ(rocsparse_create_dnmat_descr(&eA, 1, 1, 1, one_a.ptr, rocsparse_datatype_f32_r, rocsparse_order_column),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_dnmat_descr(&eB, 1, 1, 1, one_a.ptr, rocsparse_datatype_f32_r, rocsparse_order_column),
              rocsparse_status_success);
    buffer_size = 99;
    ASSERT_EQ(rocsparse_sddmm_buffer_size(handle,
                                          rocsparse_operation_none,
                                          rocsparse_operation_none,
                                          &one,
                                          eA,
                                          eB,
                                          &zero,
                                          empty,
                                          rocsparse_datatype_f32_r,
                                          rocsparse_sddmm_alg_dense,
                                          &buffer_size),
              rocsparse_status_success);
    EXPECT_EQ(buffer_size, size_t(0));
    EXPECT_EQ(rocsparse_destroy_dnmat_descr(eA), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_dnmat_descr(eB), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(empty), rocsparse_status_success);

    // Sample-kernel width ladder: avg_nnz = nnz / m. These complete only when
    // rocBLAS gemm succeeds; otherwise the conversion still ran.
    const int avgs[] = {1, 2, 3, 6, 10, 20, 40};
    for(int avg : avgs)
    {
        const int                  m = 1;
        const int                  n = avg;
        std::vector<int32_t>       ptr{0, n};
        std::vector<int32_t>       cols(n);
        std::vector<float>         values(n, 1.0f);
        for(int c = 0; c < n; ++c)
            cols[c] = c;
        device_vector<int32_t> dptr{ptr};
        device_vector<int32_t> dcol{cols};
        device_vector<float>   dval{values};
        ASSERT_TRUE(dptr.ptr && dcol.ptr && dval.ptr);
        rocsparse_spmat_descr mat = nullptr;
        ASSERT_EQ(rocsparse_create_csr_descr(&mat,
                                             m,
                                             n,
                                             n,
                                             dptr.ptr,
                                             dcol.ptr,
                                             dval.ptr,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f32_r),
                  rocsparse_status_success);
        st = run_ones(handle,
                      rocsparse_operation_none,
                      rocsparse_operation_none,
                      rocsparse_order_column,
                      rocsparse_order_column,
                      m,
                      n,
                      1,
                      mat,
                      1.0f,
                      0.0f,
                      rocsparse_sddmm_alg_dense);
        EXPECT_TRUE(st == rocsparse_status_success || st == rocsparse_status_not_implemented)
            << "avg " << avg << " status " << st;
        if(st == rocsparse_status_success)
            expect_filled(dval.ptr, n, 1.0f);
        EXPECT_EQ(rocsparse_destroy_spmat_descr(mat), rocsparse_status_success);
    }
}

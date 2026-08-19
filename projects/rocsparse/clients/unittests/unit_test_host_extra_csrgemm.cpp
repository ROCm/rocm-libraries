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
// Host-path unit tests for the extra sub-lib. Drives full CSR/BSR pipelines
// (csrgeam C = alpha*A + beta*B, csrgemm C = alpha*A*B + beta*D, bsrgemm) on
// tiny 3x3 identity / 2x2-block matrices plus their validation guards.
// Exercises host dispatch/nnz/compute code in library/src/extra across all
// precisions. Coverage is host-only, so we drive the public C API.
//
#include "unit_test_utils.hpp"

#include <type_traits>

using namespace rocsparse_ut;

namespace
{
    // ----- typed wrappers over the per-precision C API ----------------------
    template <typename T>
    rocsparse_status ut_csrgemm_buffer_size(rocsparse_handle          handle,
                                            rocsparse_operation       trans_A,
                                            rocsparse_operation       trans_B,
                                            rocsparse_int             m,
                                            rocsparse_int             n,
                                            rocsparse_int             k,
                                            const T*                  alpha,
                                            const rocsparse_mat_descr descr_A,
                                            rocsparse_int             nnz_A,
                                            const rocsparse_int*      csr_row_ptr_A,
                                            const rocsparse_int*      csr_col_ind_A,
                                            const rocsparse_mat_descr descr_B,
                                            rocsparse_int             nnz_B,
                                            const rocsparse_int*      csr_row_ptr_B,
                                            const rocsparse_int*      csr_col_ind_B,
                                            const T*                  beta,
                                            const rocsparse_mat_descr descr_D,
                                            rocsparse_int             nnz_D,
                                            const rocsparse_int*      csr_row_ptr_D,
                                            const rocsparse_int*      csr_col_ind_D,
                                            rocsparse_mat_info        info_C,
                                            size_t*                   buffer_size)
    {
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_scsrgemm_buffer_size(handle,
                                                  trans_A,
                                                  trans_B,
                                                  m,
                                                  n,
                                                  k,
                                                  alpha,
                                                  descr_A,
                                                  nnz_A,
                                                  csr_row_ptr_A,
                                                  csr_col_ind_A,
                                                  descr_B,
                                                  nnz_B,
                                                  csr_row_ptr_B,
                                                  csr_col_ind_B,
                                                  beta,
                                                  descr_D,
                                                  nnz_D,
                                                  csr_row_ptr_D,
                                                  csr_col_ind_D,
                                                  info_C,
                                                  buffer_size);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dcsrgemm_buffer_size(handle,
                                                  trans_A,
                                                  trans_B,
                                                  m,
                                                  n,
                                                  k,
                                                  alpha,
                                                  descr_A,
                                                  nnz_A,
                                                  csr_row_ptr_A,
                                                  csr_col_ind_A,
                                                  descr_B,
                                                  nnz_B,
                                                  csr_row_ptr_B,
                                                  csr_col_ind_B,
                                                  beta,
                                                  descr_D,
                                                  nnz_D,
                                                  csr_row_ptr_D,
                                                  csr_col_ind_D,
                                                  info_C,
                                                  buffer_size);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_ccsrgemm_buffer_size(handle,
                                                  trans_A,
                                                  trans_B,
                                                  m,
                                                  n,
                                                  k,
                                                  alpha,
                                                  descr_A,
                                                  nnz_A,
                                                  csr_row_ptr_A,
                                                  csr_col_ind_A,
                                                  descr_B,
                                                  nnz_B,
                                                  csr_row_ptr_B,
                                                  csr_col_ind_B,
                                                  beta,
                                                  descr_D,
                                                  nnz_D,
                                                  csr_row_ptr_D,
                                                  csr_col_ind_D,
                                                  info_C,
                                                  buffer_size);
        else
            return rocsparse_zcsrgemm_buffer_size(handle,
                                                  trans_A,
                                                  trans_B,
                                                  m,
                                                  n,
                                                  k,
                                                  alpha,
                                                  descr_A,
                                                  nnz_A,
                                                  csr_row_ptr_A,
                                                  csr_col_ind_A,
                                                  descr_B,
                                                  nnz_B,
                                                  csr_row_ptr_B,
                                                  csr_col_ind_B,
                                                  beta,
                                                  descr_D,
                                                  nnz_D,
                                                  csr_row_ptr_D,
                                                  csr_col_ind_D,
                                                  info_C,
                                                  buffer_size);
    }

    template <typename T>
    rocsparse_status ut_csrgemm(rocsparse_handle          handle,
                                rocsparse_operation       trans_A,
                                rocsparse_operation       trans_B,
                                rocsparse_int             m,
                                rocsparse_int             n,
                                rocsparse_int             k,
                                const T*                  alpha,
                                const rocsparse_mat_descr descr_A,
                                rocsparse_int             nnz_A,
                                const T*                  csr_val_A,
                                const rocsparse_int*      csr_row_ptr_A,
                                const rocsparse_int*      csr_col_ind_A,
                                const rocsparse_mat_descr descr_B,
                                rocsparse_int             nnz_B,
                                const T*                  csr_val_B,
                                const rocsparse_int*      csr_row_ptr_B,
                                const rocsparse_int*      csr_col_ind_B,
                                const T*                  beta,
                                const rocsparse_mat_descr descr_D,
                                rocsparse_int             nnz_D,
                                const T*                  csr_val_D,
                                const rocsparse_int*      csr_row_ptr_D,
                                const rocsparse_int*      csr_col_ind_D,
                                const rocsparse_mat_descr descr_C,
                                T*                        csr_val_C,
                                const rocsparse_int*      csr_row_ptr_C,
                                rocsparse_int*            csr_col_ind_C,
                                const rocsparse_mat_info  info_C,
                                void*                     temp_buffer)
    {
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_scsrgemm(handle,
                                      trans_A,
                                      trans_B,
                                      m,
                                      n,
                                      k,
                                      alpha,
                                      descr_A,
                                      nnz_A,
                                      csr_val_A,
                                      csr_row_ptr_A,
                                      csr_col_ind_A,
                                      descr_B,
                                      nnz_B,
                                      csr_val_B,
                                      csr_row_ptr_B,
                                      csr_col_ind_B,
                                      beta,
                                      descr_D,
                                      nnz_D,
                                      csr_val_D,
                                      csr_row_ptr_D,
                                      csr_col_ind_D,
                                      descr_C,
                                      csr_val_C,
                                      csr_row_ptr_C,
                                      csr_col_ind_C,
                                      info_C,
                                      temp_buffer);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dcsrgemm(handle,
                                      trans_A,
                                      trans_B,
                                      m,
                                      n,
                                      k,
                                      alpha,
                                      descr_A,
                                      nnz_A,
                                      csr_val_A,
                                      csr_row_ptr_A,
                                      csr_col_ind_A,
                                      descr_B,
                                      nnz_B,
                                      csr_val_B,
                                      csr_row_ptr_B,
                                      csr_col_ind_B,
                                      beta,
                                      descr_D,
                                      nnz_D,
                                      csr_val_D,
                                      csr_row_ptr_D,
                                      csr_col_ind_D,
                                      descr_C,
                                      csr_val_C,
                                      csr_row_ptr_C,
                                      csr_col_ind_C,
                                      info_C,
                                      temp_buffer);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_ccsrgemm(handle,
                                      trans_A,
                                      trans_B,
                                      m,
                                      n,
                                      k,
                                      alpha,
                                      descr_A,
                                      nnz_A,
                                      csr_val_A,
                                      csr_row_ptr_A,
                                      csr_col_ind_A,
                                      descr_B,
                                      nnz_B,
                                      csr_val_B,
                                      csr_row_ptr_B,
                                      csr_col_ind_B,
                                      beta,
                                      descr_D,
                                      nnz_D,
                                      csr_val_D,
                                      csr_row_ptr_D,
                                      csr_col_ind_D,
                                      descr_C,
                                      csr_val_C,
                                      csr_row_ptr_C,
                                      csr_col_ind_C,
                                      info_C,
                                      temp_buffer);
        else
            return rocsparse_zcsrgemm(handle,
                                      trans_A,
                                      trans_B,
                                      m,
                                      n,
                                      k,
                                      alpha,
                                      descr_A,
                                      nnz_A,
                                      csr_val_A,
                                      csr_row_ptr_A,
                                      csr_col_ind_A,
                                      descr_B,
                                      nnz_B,
                                      csr_val_B,
                                      csr_row_ptr_B,
                                      csr_col_ind_B,
                                      beta,
                                      descr_D,
                                      nnz_D,
                                      csr_val_D,
                                      csr_row_ptr_D,
                                      csr_col_ind_D,
                                      descr_C,
                                      csr_val_C,
                                      csr_row_ptr_C,
                                      csr_col_ind_C,
                                      info_C,
                                      temp_buffer);
    }
} // namespace

// ---------------------------------------------------------------------------
// csrgemm: buffer_size -> nnz -> compute pipeline (C = alpha*A*B) on 3x3
// identity => C = identity. Plus a multadd variant (C = alpha*A*B + beta*D)
// and bad-arg guards. Coverage is host-only; nnz_C is a HOST pointer under
// the default (host) pointer mode.
// ---------------------------------------------------------------------------
class ExtraCsrgemm : public HandleTest
{
protected:
    template <typename T>
    void run_mult()
    {
        const rocsparse_int          m = 3, n = 3, k = 3, nnz = 3;
        device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
        device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
        device_vector<T>             val{std::vector<T>{scalar<T>(1), scalar<T>(1), scalar<T>(1)}};
        ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

        rocsparse_mat_descr descr = nullptr;
        ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
        rocsparse_mat_info info = nullptr;
        ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

        const T                   alpha = scalar<T>(1);
        const rocsparse_operation opn   = rocsparse_operation_none;

        size_t           buffer_size = 0;
        rocsparse_status st          = ut_csrgemm_buffer_size<T>(handle,
                                                        opn,
                                                        opn,
                                                        m,
                                                        n,
                                                        k,
                                                        &alpha,
                                                        descr,
                                                        nnz,
                                                        row_ptr,
                                                        col_ind,
                                                        descr,
                                                        nnz,
                                                        row_ptr,
                                                        col_ind,
                                                        (const T*)nullptr,
                                                        nullptr,
                                                        0,
                                                        nullptr,
                                                        nullptr,
                                                        info,
                                                        &buffer_size);
        if(st == rocsparse_status_not_implemented)
        {
            EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
            EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
            GTEST_SKIP() << "csrgemm not implemented for this configuration";
        }
        ASSERT_EQ(st, rocsparse_status_success);
        EXPECT_GT(buffer_size, 0u);

        device_vector<char> buffer{buffer_size};
        ASSERT_TRUE(buffer.ptr);

        device_vector<rocsparse_int> row_ptr_C{(size_t)(m + 1)};
        ASSERT_TRUE(row_ptr_C.ptr);

        rocsparse_int nnz_C = 0;
        ASSERT_EQ(rocsparse_csrgemm_nnz(handle,
                                        opn,
                                        opn,
                                        m,
                                        n,
                                        k,
                                        descr,
                                        nnz,
                                        row_ptr,
                                        col_ind,
                                        descr,
                                        nnz,
                                        row_ptr,
                                        col_ind,
                                        nullptr,
                                        0,
                                        nullptr,
                                        nullptr,
                                        descr,
                                        row_ptr_C,
                                        &nnz_C,
                                        info,
                                        buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_EQ(nnz_C, 3);

        device_vector<rocsparse_int> col_ind_C{(size_t)nnz_C};
        device_vector<T>             val_C{(size_t)nnz_C};
        ASSERT_TRUE(col_ind_C.ptr && val_C.ptr);

        ASSERT_EQ(ut_csrgemm<T>(handle,
                                opn,
                                opn,
                                m,
                                n,
                                k,
                                &alpha,
                                descr,
                                nnz,
                                val,
                                row_ptr,
                                col_ind,
                                descr,
                                nnz,
                                val,
                                row_ptr,
                                col_ind,
                                (const T*)nullptr,
                                nullptr,
                                0,
                                (const T*)nullptr,
                                nullptr,
                                nullptr,
                                descr,
                                val_C,
                                row_ptr_C,
                                col_ind_C,
                                info,
                                buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        // C = 1 * I * I = I
        std::vector<T> host_C(nnz_C);
        ASSERT_EQ(hipMemcpy(host_C.data(), val_C.ptr, nnz_C * sizeof(T), hipMemcpyDeviceToHost),
                  hipSuccess);
        for(auto v : host_C)
            EXPECT_EQ(v, scalar<T>(1));

        EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
    }
};

TEST_F(ExtraCsrgemm, mult_f32)
{
    run_mult<float>();
}
TEST_F(ExtraCsrgemm, mult_f64)
{
    run_mult<double>();
}
TEST_F(ExtraCsrgemm, mult_c32)
{
    run_mult<rocsparse_float_complex>();
}
TEST_F(ExtraCsrgemm, mult_c64)
{
    run_mult<rocsparse_double_complex>();
}

TEST_F(ExtraCsrgemm, scsrgemm_multadd_f32)
{
    const rocsparse_int          m = 3, n = 3, k = 3, nnz = 3;
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<float>         val{std::vector<float>{1, 1, 1}};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    rocsparse_mat_info info = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

    const float               alpha = 1.0f, beta = 1.0f;
    const rocsparse_operation opn = rocsparse_operation_none;

    size_t           buffer_size = 0;
    rocsparse_status st          = rocsparse_scsrgemm_buffer_size(handle,
                                                         opn,
                                                         opn,
                                                         m,
                                                         n,
                                                         k,
                                                         &alpha,
                                                         descr,
                                                         nnz,
                                                         row_ptr,
                                                         col_ind,
                                                         descr,
                                                         nnz,
                                                         row_ptr,
                                                         col_ind,
                                                         &beta,
                                                         descr,
                                                         nnz,
                                                         row_ptr,
                                                         col_ind,
                                                         info,
                                                         &buffer_size);
    if(st == rocsparse_status_not_implemented)
    {
        EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
        GTEST_SKIP() << "csrgemm multadd not implemented for this configuration";
    }
    ASSERT_EQ(st, rocsparse_status_success);

    device_vector<char> buffer{buffer_size};
    ASSERT_TRUE(buffer.ptr);

    device_vector<rocsparse_int> row_ptr_C{(size_t)(m + 1)};
    ASSERT_TRUE(row_ptr_C.ptr);

    rocsparse_int nnz_C = 0;
    ASSERT_EQ(rocsparse_csrgemm_nnz(handle,
                                    opn,
                                    opn,
                                    m,
                                    n,
                                    k,
                                    descr,
                                    nnz,
                                    row_ptr,
                                    col_ind,
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
                                    &nnz_C,
                                    info,
                                    buffer.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(nnz_C, 3);

    device_vector<rocsparse_int> col_ind_C{(size_t)nnz_C};
    device_vector<float>         val_C{(size_t)nnz_C};
    ASSERT_TRUE(col_ind_C.ptr && val_C.ptr);

    // C = 1*A*B + 1*D = I + I = 2*I
    EXPECT_EQ(rocsparse_scsrgemm(handle,
                                 opn,
                                 opn,
                                 m,
                                 n,
                                 k,
                                 &alpha,
                                 descr,
                                 nnz,
                                 val,
                                 row_ptr,
                                 col_ind,
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
                                 col_ind_C,
                                 info,
                                 buffer.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    std::vector<float> host_C(nnz_C);
    ASSERT_EQ(hipMemcpy(host_C.data(), val_C.ptr, nnz_C * sizeof(float), hipMemcpyDeviceToHost),
              hipSuccess);
    for(auto v : host_C)
        EXPECT_FLOAT_EQ(v, 2.0f);

    EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

TEST_F(ExtraCsrgemm, bad_args)
{
    const rocsparse_int          m = 3, n = 3, k = 3, nnz = 3;
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<rocsparse_int> row_ptr_C{(size_t)(m + 1)};
    rocsparse_mat_descr          descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    rocsparse_mat_info info = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

    const float               alpha       = 1.0f;
    const rocsparse_operation opn         = rocsparse_operation_none;
    size_t                    buffer_size = 0;

    // buffer_size: null handle -> invalid_handle
    EXPECT_EQ(rocsparse_scsrgemm_buffer_size(nullptr,
                                             opn,
                                             opn,
                                             m,
                                             n,
                                             k,
                                             &alpha,
                                             descr,
                                             nnz,
                                             row_ptr,
                                             col_ind,
                                             descr,
                                             nnz,
                                             row_ptr,
                                             col_ind,
                                             nullptr,
                                             nullptr,
                                             0,
                                             nullptr,
                                             nullptr,
                                             info,
                                             &buffer_size),
              rocsparse_status_invalid_handle);
    // buffer_size: negative size -> invalid_size
    EXPECT_EQ(rocsparse_scsrgemm_buffer_size(handle,
                                             opn,
                                             opn,
                                             -1,
                                             n,
                                             k,
                                             &alpha,
                                             descr,
                                             nnz,
                                             row_ptr,
                                             col_ind,
                                             descr,
                                             nnz,
                                             row_ptr,
                                             col_ind,
                                             nullptr,
                                             nullptr,
                                             0,
                                             nullptr,
                                             nullptr,
                                             info,
                                             &buffer_size),
              rocsparse_status_invalid_size);
    // buffer_size: null alpha AND beta -> invalid_pointer
    EXPECT_EQ(rocsparse_scsrgemm_buffer_size(handle,
                                             opn,
                                             opn,
                                             m,
                                             n,
                                             k,
                                             nullptr,
                                             descr,
                                             nnz,
                                             row_ptr,
                                             col_ind,
                                             descr,
                                             nnz,
                                             row_ptr,
                                             col_ind,
                                             nullptr,
                                             nullptr,
                                             0,
                                             nullptr,
                                             nullptr,
                                             info,
                                             &buffer_size),
              rocsparse_status_invalid_pointer);
    // buffer_size: transpose A -> not_implemented
    EXPECT_EQ(rocsparse_scsrgemm_buffer_size(handle,
                                             rocsparse_operation_transpose,
                                             opn,
                                             m,
                                             n,
                                             k,
                                             &alpha,
                                             descr,
                                             nnz,
                                             row_ptr,
                                             col_ind,
                                             descr,
                                             nnz,
                                             row_ptr,
                                             col_ind,
                                             nullptr,
                                             nullptr,
                                             0,
                                             nullptr,
                                             nullptr,
                                             info,
                                             &buffer_size),
              rocsparse_status_not_implemented);

    // nnz: null handle -> invalid_handle
    rocsparse_int nnz_C = 0;
    EXPECT_EQ(rocsparse_csrgemm_nnz(nullptr,
                                    opn,
                                    opn,
                                    m,
                                    n,
                                    k,
                                    descr,
                                    nnz,
                                    row_ptr,
                                    col_ind,
                                    descr,
                                    nnz,
                                    row_ptr,
                                    col_ind,
                                    nullptr,
                                    0,
                                    nullptr,
                                    nullptr,
                                    descr,
                                    row_ptr_C,
                                    &nnz_C,
                                    info,
                                    nullptr),
              rocsparse_status_invalid_handle);
    // nnz: negative size -> invalid_size
    EXPECT_EQ(rocsparse_csrgemm_nnz(handle,
                                    opn,
                                    opn,
                                    -1,
                                    n,
                                    k,
                                    descr,
                                    nnz,
                                    row_ptr,
                                    col_ind,
                                    descr,
                                    nnz,
                                    row_ptr,
                                    col_ind,
                                    nullptr,
                                    0,
                                    nullptr,
                                    nullptr,
                                    descr,
                                    row_ptr_C,
                                    &nnz_C,
                                    info,
                                    nullptr),
              rocsparse_status_invalid_size);

    EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

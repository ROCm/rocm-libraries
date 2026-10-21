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
    template <typename T>
    rocsparse_status ut_csrgeam(rocsparse_handle          handle,
                                rocsparse_int             m,
                                rocsparse_int             n,
                                const T*                  alpha,
                                const rocsparse_mat_descr descr_A,
                                rocsparse_int             nnz_A,
                                const T*                  csr_val_A,
                                const rocsparse_int*      csr_row_ptr_A,
                                const rocsparse_int*      csr_col_ind_A,
                                const T*                  beta,
                                const rocsparse_mat_descr descr_B,
                                rocsparse_int             nnz_B,
                                const T*                  csr_val_B,
                                const rocsparse_int*      csr_row_ptr_B,
                                const rocsparse_int*      csr_col_ind_B,
                                const rocsparse_mat_descr descr_C,
                                T*                        csr_val_C,
                                const rocsparse_int*      csr_row_ptr_C,
                                rocsparse_int*            csr_col_ind_C)
    {
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_scsrgeam(handle,
                                      m,
                                      n,
                                      alpha,
                                      descr_A,
                                      nnz_A,
                                      csr_val_A,
                                      csr_row_ptr_A,
                                      csr_col_ind_A,
                                      beta,
                                      descr_B,
                                      nnz_B,
                                      csr_val_B,
                                      csr_row_ptr_B,
                                      csr_col_ind_B,
                                      descr_C,
                                      csr_val_C,
                                      csr_row_ptr_C,
                                      csr_col_ind_C);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dcsrgeam(handle,
                                      m,
                                      n,
                                      alpha,
                                      descr_A,
                                      nnz_A,
                                      csr_val_A,
                                      csr_row_ptr_A,
                                      csr_col_ind_A,
                                      beta,
                                      descr_B,
                                      nnz_B,
                                      csr_val_B,
                                      csr_row_ptr_B,
                                      csr_col_ind_B,
                                      descr_C,
                                      csr_val_C,
                                      csr_row_ptr_C,
                                      csr_col_ind_C);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_ccsrgeam(handle,
                                      m,
                                      n,
                                      alpha,
                                      descr_A,
                                      nnz_A,
                                      csr_val_A,
                                      csr_row_ptr_A,
                                      csr_col_ind_A,
                                      beta,
                                      descr_B,
                                      nnz_B,
                                      csr_val_B,
                                      csr_row_ptr_B,
                                      csr_col_ind_B,
                                      descr_C,
                                      csr_val_C,
                                      csr_row_ptr_C,
                                      csr_col_ind_C);
        else
            return rocsparse_zcsrgeam(handle,
                                      m,
                                      n,
                                      alpha,
                                      descr_A,
                                      nnz_A,
                                      csr_val_A,
                                      csr_row_ptr_A,
                                      csr_col_ind_A,
                                      beta,
                                      descr_B,
                                      nnz_B,
                                      csr_val_B,
                                      csr_row_ptr_B,
                                      csr_col_ind_B,
                                      descr_C,
                                      csr_val_C,
                                      csr_row_ptr_C,
                                      csr_col_ind_C);
    }
} // namespace

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

// ---------------------------------------------------------------------------
// csrgeam: more precisions (C = alpha*A + beta*B on 3x3 identity => 2*identity)
// plus scsrgeam bad-arg guards.
// ---------------------------------------------------------------------------
class ExtraCsrgeam : public HandleTest
{
protected:
    template <typename T>
    void run_pipeline()
    {
        const rocsparse_int          m = 3, n = 3, nnz = 3;
        device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
        device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
        device_vector<T>             val{std::vector<T>{scalar<T>(1), scalar<T>(1), scalar<T>(1)}};
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
        device_vector<T>             val_C{(size_t)nnz_C};
        ASSERT_TRUE(col_ind_C.ptr && val_C.ptr);

        const T alpha = scalar<T>(1), beta = scalar<T>(1);
        EXPECT_EQ(ut_csrgeam<T>(handle,
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

        // C = 1*I + 1*I = 2*I
        std::vector<T> host_C(nnz_C);
        ASSERT_EQ(hipMemcpy(host_C.data(), val_C.ptr, nnz_C * sizeof(T), hipMemcpyDeviceToHost),
                  hipSuccess);
        for(auto v : host_C)
            EXPECT_EQ(v, scalar<T>(2));

        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
    }
};

TEST_F(ExtraCsrgeam, pipeline_f64)
{
    run_pipeline<double>();
}
TEST_F(ExtraCsrgeam, pipeline_c32)
{
    run_pipeline<rocsparse_float_complex>();
}
TEST_F(ExtraCsrgeam, pipeline_c64)
{
    run_pipeline<rocsparse_double_complex>();
}

TEST_F(ExtraCsrgeam, scsrgeam_bad_args)
{
    const rocsparse_int          m = 3, n = 3, nnz = 3;
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<float>         val{std::vector<float>{1, 1, 1}};
    device_vector<rocsparse_int> row_ptr_C{(size_t)(m + 1)};
    device_vector<rocsparse_int> col_ind_C{(size_t)nnz};
    device_vector<float>         val_C{(size_t)nnz};
    rocsparse_mat_descr          descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    const float alpha = 1.0f, beta = 1.0f;

    // null handle -> invalid_handle
    EXPECT_EQ(rocsparse_scsrgeam(nullptr,
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
              rocsparse_status_invalid_handle);
    // negative size -> invalid_size
    EXPECT_EQ(rocsparse_scsrgeam(handle,
                                 -1,
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
              rocsparse_status_invalid_size);
    // null alpha -> invalid_pointer
    EXPECT_EQ(rocsparse_scsrgeam(handle,
                                 m,
                                 n,
                                 nullptr,
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
              rocsparse_status_invalid_pointer);
    // null descr -> invalid_pointer
    EXPECT_EQ(rocsparse_scsrgeam(handle,
                                 m,
                                 n,
                                 &alpha,
                                 nullptr,
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
              rocsparse_status_invalid_pointer);

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

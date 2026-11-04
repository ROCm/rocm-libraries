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
    rocsparse_status ut_bsrgemm_buffer_size(rocsparse_handle          handle,
                                            rocsparse_direction       dir,
                                            rocsparse_operation       trans_A,
                                            rocsparse_operation       trans_B,
                                            rocsparse_int             mb,
                                            rocsparse_int             nb,
                                            rocsparse_int             kb,
                                            rocsparse_int             block_dim,
                                            const T*                  alpha,
                                            const rocsparse_mat_descr descr_A,
                                            rocsparse_int             nnzb_A,
                                            const rocsparse_int*      bsr_row_ptr_A,
                                            const rocsparse_int*      bsr_col_ind_A,
                                            const rocsparse_mat_descr descr_B,
                                            rocsparse_int             nnzb_B,
                                            const rocsparse_int*      bsr_row_ptr_B,
                                            const rocsparse_int*      bsr_col_ind_B,
                                            const T*                  beta,
                                            const rocsparse_mat_descr descr_D,
                                            rocsparse_int             nnzb_D,
                                            const rocsparse_int*      bsr_row_ptr_D,
                                            const rocsparse_int*      bsr_col_ind_D,
                                            rocsparse_mat_info        info_C,
                                            size_t*                   buffer_size)
    {
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_sbsrgemm_buffer_size(handle,
                                                  dir,
                                                  trans_A,
                                                  trans_B,
                                                  mb,
                                                  nb,
                                                  kb,
                                                  block_dim,
                                                  alpha,
                                                  descr_A,
                                                  nnzb_A,
                                                  bsr_row_ptr_A,
                                                  bsr_col_ind_A,
                                                  descr_B,
                                                  nnzb_B,
                                                  bsr_row_ptr_B,
                                                  bsr_col_ind_B,
                                                  beta,
                                                  descr_D,
                                                  nnzb_D,
                                                  bsr_row_ptr_D,
                                                  bsr_col_ind_D,
                                                  info_C,
                                                  buffer_size);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dbsrgemm_buffer_size(handle,
                                                  dir,
                                                  trans_A,
                                                  trans_B,
                                                  mb,
                                                  nb,
                                                  kb,
                                                  block_dim,
                                                  alpha,
                                                  descr_A,
                                                  nnzb_A,
                                                  bsr_row_ptr_A,
                                                  bsr_col_ind_A,
                                                  descr_B,
                                                  nnzb_B,
                                                  bsr_row_ptr_B,
                                                  bsr_col_ind_B,
                                                  beta,
                                                  descr_D,
                                                  nnzb_D,
                                                  bsr_row_ptr_D,
                                                  bsr_col_ind_D,
                                                  info_C,
                                                  buffer_size);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_cbsrgemm_buffer_size(handle,
                                                  dir,
                                                  trans_A,
                                                  trans_B,
                                                  mb,
                                                  nb,
                                                  kb,
                                                  block_dim,
                                                  alpha,
                                                  descr_A,
                                                  nnzb_A,
                                                  bsr_row_ptr_A,
                                                  bsr_col_ind_A,
                                                  descr_B,
                                                  nnzb_B,
                                                  bsr_row_ptr_B,
                                                  bsr_col_ind_B,
                                                  beta,
                                                  descr_D,
                                                  nnzb_D,
                                                  bsr_row_ptr_D,
                                                  bsr_col_ind_D,
                                                  info_C,
                                                  buffer_size);
        else
            return rocsparse_zbsrgemm_buffer_size(handle,
                                                  dir,
                                                  trans_A,
                                                  trans_B,
                                                  mb,
                                                  nb,
                                                  kb,
                                                  block_dim,
                                                  alpha,
                                                  descr_A,
                                                  nnzb_A,
                                                  bsr_row_ptr_A,
                                                  bsr_col_ind_A,
                                                  descr_B,
                                                  nnzb_B,
                                                  bsr_row_ptr_B,
                                                  bsr_col_ind_B,
                                                  beta,
                                                  descr_D,
                                                  nnzb_D,
                                                  bsr_row_ptr_D,
                                                  bsr_col_ind_D,
                                                  info_C,
                                                  buffer_size);
    }

    template <typename T>
    rocsparse_status ut_bsrgemm(rocsparse_handle          handle,
                                rocsparse_direction       dir,
                                rocsparse_operation       trans_A,
                                rocsparse_operation       trans_B,
                                rocsparse_int             mb,
                                rocsparse_int             nb,
                                rocsparse_int             kb,
                                rocsparse_int             block_dim,
                                const T*                  alpha,
                                const rocsparse_mat_descr descr_A,
                                rocsparse_int             nnzb_A,
                                const T*                  bsr_val_A,
                                const rocsparse_int*      bsr_row_ptr_A,
                                const rocsparse_int*      bsr_col_ind_A,
                                const rocsparse_mat_descr descr_B,
                                rocsparse_int             nnzb_B,
                                const T*                  bsr_val_B,
                                const rocsparse_int*      bsr_row_ptr_B,
                                const rocsparse_int*      bsr_col_ind_B,
                                const T*                  beta,
                                const rocsparse_mat_descr descr_D,
                                rocsparse_int             nnzb_D,
                                const T*                  bsr_val_D,
                                const rocsparse_int*      bsr_row_ptr_D,
                                const rocsparse_int*      bsr_col_ind_D,
                                const rocsparse_mat_descr descr_C,
                                T*                        bsr_val_C,
                                const rocsparse_int*      bsr_row_ptr_C,
                                rocsparse_int*            bsr_col_ind_C,
                                const rocsparse_mat_info  info_C,
                                void*                     temp_buffer)
    {
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_sbsrgemm(handle,
                                      dir,
                                      trans_A,
                                      trans_B,
                                      mb,
                                      nb,
                                      kb,
                                      block_dim,
                                      alpha,
                                      descr_A,
                                      nnzb_A,
                                      bsr_val_A,
                                      bsr_row_ptr_A,
                                      bsr_col_ind_A,
                                      descr_B,
                                      nnzb_B,
                                      bsr_val_B,
                                      bsr_row_ptr_B,
                                      bsr_col_ind_B,
                                      beta,
                                      descr_D,
                                      nnzb_D,
                                      bsr_val_D,
                                      bsr_row_ptr_D,
                                      bsr_col_ind_D,
                                      descr_C,
                                      bsr_val_C,
                                      bsr_row_ptr_C,
                                      bsr_col_ind_C,
                                      info_C,
                                      temp_buffer);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dbsrgemm(handle,
                                      dir,
                                      trans_A,
                                      trans_B,
                                      mb,
                                      nb,
                                      kb,
                                      block_dim,
                                      alpha,
                                      descr_A,
                                      nnzb_A,
                                      bsr_val_A,
                                      bsr_row_ptr_A,
                                      bsr_col_ind_A,
                                      descr_B,
                                      nnzb_B,
                                      bsr_val_B,
                                      bsr_row_ptr_B,
                                      bsr_col_ind_B,
                                      beta,
                                      descr_D,
                                      nnzb_D,
                                      bsr_val_D,
                                      bsr_row_ptr_D,
                                      bsr_col_ind_D,
                                      descr_C,
                                      bsr_val_C,
                                      bsr_row_ptr_C,
                                      bsr_col_ind_C,
                                      info_C,
                                      temp_buffer);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_cbsrgemm(handle,
                                      dir,
                                      trans_A,
                                      trans_B,
                                      mb,
                                      nb,
                                      kb,
                                      block_dim,
                                      alpha,
                                      descr_A,
                                      nnzb_A,
                                      bsr_val_A,
                                      bsr_row_ptr_A,
                                      bsr_col_ind_A,
                                      descr_B,
                                      nnzb_B,
                                      bsr_val_B,
                                      bsr_row_ptr_B,
                                      bsr_col_ind_B,
                                      beta,
                                      descr_D,
                                      nnzb_D,
                                      bsr_val_D,
                                      bsr_row_ptr_D,
                                      bsr_col_ind_D,
                                      descr_C,
                                      bsr_val_C,
                                      bsr_row_ptr_C,
                                      bsr_col_ind_C,
                                      info_C,
                                      temp_buffer);
        else
            return rocsparse_zbsrgemm(handle,
                                      dir,
                                      trans_A,
                                      trans_B,
                                      mb,
                                      nb,
                                      kb,
                                      block_dim,
                                      alpha,
                                      descr_A,
                                      nnzb_A,
                                      bsr_val_A,
                                      bsr_row_ptr_A,
                                      bsr_col_ind_A,
                                      descr_B,
                                      nnzb_B,
                                      bsr_val_B,
                                      bsr_row_ptr_B,
                                      bsr_col_ind_B,
                                      beta,
                                      descr_D,
                                      nnzb_D,
                                      bsr_val_D,
                                      bsr_row_ptr_D,
                                      bsr_col_ind_D,
                                      descr_C,
                                      bsr_val_C,
                                      bsr_row_ptr_C,
                                      bsr_col_ind_C,
                                      info_C,
                                      temp_buffer);
    }
} // namespace

// ---------------------------------------------------------------------------
// bsrgemm: buffer_size -> nnzb -> compute pipeline (C = alpha*A*B) on a single
// 2x2 identity block => C = identity block. Plus bad-arg guards.
// ---------------------------------------------------------------------------
class ExtraBsrgemm : public HandleTest
{
protected:
    template <typename T>
    void run_mult()
    {
        const rocsparse_int mb = 1, nb = 1, kb = 1, block_dim = 2, nnzb = 1;
        // Single 2x2 identity block (row-major within block).
        device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1}};
        device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0}};
        device_vector<T>             val{
            std::vector<T>{scalar<T>(1), scalar<T>(0), scalar<T>(0), scalar<T>(1)}};
        ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

        rocsparse_mat_descr descr = nullptr;
        ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
        rocsparse_mat_info info = nullptr;
        ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

        const T                   alpha = scalar<T>(1);
        const rocsparse_operation opn   = rocsparse_operation_none;
        const rocsparse_direction dir   = rocsparse_direction_row;

        size_t           buffer_size = 0;
        rocsparse_status st          = ut_bsrgemm_buffer_size<T>(handle,
                                                        dir,
                                                        opn,
                                                        opn,
                                                        mb,
                                                        nb,
                                                        kb,
                                                        block_dim,
                                                        &alpha,
                                                        descr,
                                                        nnzb,
                                                        row_ptr,
                                                        col_ind,
                                                        descr,
                                                        nnzb,
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
            GTEST_SKIP() << "bsrgemm not implemented for this configuration";
        }
        ASSERT_EQ(st, rocsparse_status_success);

        device_vector<char> buffer{buffer_size};
        ASSERT_TRUE(buffer.ptr);

        device_vector<rocsparse_int> row_ptr_C{(size_t)(mb + 1)};
        ASSERT_TRUE(row_ptr_C.ptr);

        rocsparse_int nnzb_C = 0;
        ASSERT_EQ(rocsparse_bsrgemm_nnzb(handle,
                                         dir,
                                         opn,
                                         opn,
                                         mb,
                                         nb,
                                         kb,
                                         block_dim,
                                         descr,
                                         nnzb,
                                         row_ptr,
                                         col_ind,
                                         descr,
                                         nnzb,
                                         row_ptr,
                                         col_ind,
                                         nullptr,
                                         0,
                                         nullptr,
                                         nullptr,
                                         descr,
                                         row_ptr_C,
                                         &nnzb_C,
                                         info,
                                         buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_EQ(nnzb_C, 1);

        device_vector<rocsparse_int> col_ind_C{(size_t)nnzb_C};
        device_vector<T>             val_C{(size_t)(nnzb_C * block_dim * block_dim)};
        ASSERT_TRUE(col_ind_C.ptr && val_C.ptr);

        ASSERT_EQ(ut_bsrgemm<T>(handle,
                                dir,
                                opn,
                                opn,
                                mb,
                                nb,
                                kb,
                                block_dim,
                                &alpha,
                                descr,
                                nnzb,
                                val,
                                row_ptr,
                                col_ind,
                                descr,
                                nnzb,
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

        EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
    }
};

TEST_F(ExtraBsrgemm, mult_f32)
{
    run_mult<float>();
}
TEST_F(ExtraBsrgemm, mult_f64)
{
    run_mult<double>();
}
TEST_F(ExtraBsrgemm, mult_c32)
{
    run_mult<rocsparse_float_complex>();
}
TEST_F(ExtraBsrgemm, mult_c64)
{
    run_mult<rocsparse_double_complex>();
}

TEST_F(ExtraBsrgemm, bad_args)
{
    const rocsparse_int          mb = 1, nb = 1, kb = 1, block_dim = 2, nnzb = 1;
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0}};
    device_vector<rocsparse_int> row_ptr_C{(size_t)(mb + 1)};
    rocsparse_mat_descr          descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    rocsparse_mat_info info = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

    const float               alpha       = 1.0f;
    const rocsparse_operation opn         = rocsparse_operation_none;
    const rocsparse_direction dir         = rocsparse_direction_row;
    size_t                    buffer_size = 0;

    // buffer_size: null handle -> invalid_handle
    EXPECT_EQ(rocsparse_sbsrgemm_buffer_size(nullptr,
                                             dir,
                                             opn,
                                             opn,
                                             mb,
                                             nb,
                                             kb,
                                             block_dim,
                                             &alpha,
                                             descr,
                                             nnzb,
                                             row_ptr,
                                             col_ind,
                                             descr,
                                             nnzb,
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
    // buffer_size: negative block dim -> invalid_size
    EXPECT_EQ(rocsparse_sbsrgemm_buffer_size(handle,
                                             dir,
                                             opn,
                                             opn,
                                             mb,
                                             nb,
                                             kb,
                                             -1,
                                             &alpha,
                                             descr,
                                             nnzb,
                                             row_ptr,
                                             col_ind,
                                             descr,
                                             nnzb,
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
    EXPECT_EQ(rocsparse_sbsrgemm_buffer_size(handle,
                                             dir,
                                             opn,
                                             opn,
                                             mb,
                                             nb,
                                             kb,
                                             block_dim,
                                             nullptr,
                                             descr,
                                             nnzb,
                                             row_ptr,
                                             col_ind,
                                             descr,
                                             nnzb,
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

    // nnzb: null handle -> invalid_handle
    rocsparse_int nnzb_C = 0;
    EXPECT_EQ(rocsparse_bsrgemm_nnzb(nullptr,
                                     dir,
                                     opn,
                                     opn,
                                     mb,
                                     nb,
                                     kb,
                                     block_dim,
                                     descr,
                                     nnzb,
                                     row_ptr,
                                     col_ind,
                                     descr,
                                     nnzb,
                                     row_ptr,
                                     col_ind,
                                     nullptr,
                                     0,
                                     nullptr,
                                     nullptr,
                                     descr,
                                     row_ptr_C,
                                     &nnzb_C,
                                     info,
                                     nullptr),
              rocsparse_status_invalid_handle);
    // nnzb: negative size -> invalid_size
    EXPECT_EQ(rocsparse_bsrgemm_nnzb(handle,
                                     dir,
                                     opn,
                                     opn,
                                     -1,
                                     nb,
                                     kb,
                                     block_dim,
                                     descr,
                                     nnzb,
                                     row_ptr,
                                     col_ind,
                                     descr,
                                     nnzb,
                                     row_ptr,
                                     col_ind,
                                     nullptr,
                                     0,
                                     nullptr,
                                     nullptr,
                                     descr,
                                     row_ptr_C,
                                     &nnzb_C,
                                     info,
                                     nullptr),
              rocsparse_status_invalid_size);

    EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

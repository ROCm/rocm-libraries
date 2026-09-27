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
// Host-path unit tests for the BSR matrix-addition C API:
//   rocsparse_bsrgeam_nnzb and rocsparse_Xbsrgeam.
//
// C = alpha * A + beta * B for block-sparse matrices. The existing suite covered
// csrgeam but not bsrgeam (library/src/extra/rocsparse_bsrgeam.cpp,
// rocsparse_bsrgeam_nnzb.cpp were 0%). The full nnzb -> compute pipeline is
// driven on a tiny 2x2 block-identity plus bad-argument branches.
//
#include "unit_test_utils.hpp"

#include "rocsparse.h"

using namespace rocsparse_ut;

namespace
{
    struct MatDescr
    {
        rocsparse_mat_descr d = nullptr;
        MatDescr()
        {
            (void)rocsparse_create_mat_descr(&d);
        }
        ~MatDescr()
        {
            if(d)
                (void)rocsparse_destroy_mat_descr(d);
        }
    };
}

// ===========================================================================
// bsrgeam : C = alpha * A + beta * B  (2x2 block-identity, 1x1 blocks)
// ===========================================================================
class BsrGeam : public HandleTest
{
};

// bsrgeam C = alpha*A + beta*B on 2x2 block-identity A, B (1x1 blocks) with
// val_A = {1,1}, val_B = {2,2}, alpha = beta = 1. Same pattern => nnzb_C = 2 and
// each diagonal block value is 1*1 + 1*2 = 3.
TEST_F(BsrGeam, full_pipeline)
{
    const rocsparse_direction dir       = rocsparse_direction_row;
    const rocsparse_int       mb        = 2;
    const rocsparse_int       nb        = 2;
    const rocsparse_int       block_dim = 1;

    device_vector<int32_t> row_ptr_A{std::vector<int32_t>{0, 1, 2}};
    device_vector<int32_t> col_ind_A{std::vector<int32_t>{0, 1}};
    device_vector<float>   val_A{std::vector<float>{1.0f, 1.0f}};
    device_vector<int32_t> row_ptr_B{std::vector<int32_t>{0, 1, 2}};
    device_vector<int32_t> col_ind_B{std::vector<int32_t>{0, 1}};
    device_vector<float>   val_B{std::vector<float>{2.0f, 2.0f}};
    ASSERT_TRUE(row_ptr_A.ptr && col_ind_A.ptr && val_A.ptr);
    ASSERT_TRUE(row_ptr_B.ptr && col_ind_B.ptr && val_B.ptr);

    MatDescr descr_A, descr_B, descr_C;
    ASSERT_TRUE(descr_A.d && descr_B.d && descr_C.d);

    device_vector<int32_t> row_ptr_C{size_t(3)};
    ASSERT_TRUE(row_ptr_C.ptr);

    rocsparse_int nnzb_C = 0;
    ASSERT_EQ(rocsparse_bsrgeam_nnzb(handle,
                                     dir,
                                     mb,
                                     nb,
                                     block_dim,
                                     descr_A.d,
                                     2,
                                     row_ptr_A,
                                     col_ind_A,
                                     descr_B.d,
                                     2,
                                     row_ptr_B,
                                     col_ind_B,
                                     descr_C.d,
                                     row_ptr_C,
                                     &nnzb_C),
              rocsparse_status_success);
    EXPECT_EQ(nnzb_C, 2);

    device_vector<int32_t> col_ind_C{size_t(2)};
    device_vector<float>   val_C{size_t(2)};
    ASSERT_TRUE(col_ind_C.ptr && val_C.ptr);

    const float alpha = 1.0f, beta = 1.0f;
    ASSERT_EQ(rocsparse_sbsrgeam(handle,
                                 dir,
                                 mb,
                                 nb,
                                 block_dim,
                                 &alpha,
                                 descr_A.d,
                                 2,
                                 val_A,
                                 row_ptr_A,
                                 col_ind_A,
                                 &beta,
                                 descr_B.d,
                                 2,
                                 val_B,
                                 row_ptr_B,
                                 col_ind_B,
                                 descr_C.d,
                                 val_C,
                                 row_ptr_C,
                                 col_ind_C),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // C = 1*A + 1*B = 3 on the diagonal blocks.
    auto hc = to_host(val_C.ptr, 2);
    EXPECT_FLOAT_EQ(hc[0], 3.0f);
    EXPECT_FLOAT_EQ(hc[1], 3.0f);
}

TEST_F(BsrGeam, bad_args)
{
    const rocsparse_direction dir = rocsparse_direction_row;
    device_vector<int32_t>    row_ptr{std::vector<int32_t>{0, 1, 2}};
    device_vector<int32_t>    col_ind{std::vector<int32_t>{0, 1}};
    device_vector<float>      val{std::vector<float>{1.0f, 1.0f}};
    MatDescr                  descr_A, descr_B, descr_C;
    device_vector<int32_t>    row_ptr_C{size_t(3)};
    rocsparse_int             nnzb_C = 0;

    EXPECT_EQ(rocsparse_bsrgeam_nnzb(nullptr,
                                     dir,
                                     2,
                                     2,
                                     1,
                                     descr_A.d,
                                     2,
                                     row_ptr,
                                     col_ind,
                                     descr_B.d,
                                     2,
                                     row_ptr,
                                     col_ind,
                                     descr_C.d,
                                     row_ptr_C,
                                     &nnzb_C),
              rocsparse_status_invalid_handle);

    // Invalid block_dim (0) is rejected.
    EXPECT_EQ(rocsparse_bsrgeam_nnzb(handle,
                                     dir,
                                     2,
                                     2,
                                     0,
                                     descr_A.d,
                                     2,
                                     row_ptr,
                                     col_ind,
                                     descr_B.d,
                                     2,
                                     row_ptr,
                                     col_ind,
                                     descr_C.d,
                                     row_ptr_C,
                                     &nnzb_C),
              rocsparse_status_invalid_size);

    const float            alpha = 1.0f, beta = 1.0f;
    device_vector<int32_t> col_ind_C{size_t(2)};
    device_vector<float>   val_C{size_t(2)};
    EXPECT_EQ(rocsparse_sbsrgeam(nullptr,
                                 dir,
                                 2,
                                 2,
                                 1,
                                 &alpha,
                                 descr_A.d,
                                 2,
                                 val,
                                 row_ptr,
                                 col_ind,
                                 &beta,
                                 descr_B.d,
                                 2,
                                 val,
                                 row_ptr,
                                 col_ind,
                                 descr_C.d,
                                 val_C,
                                 row_ptr_C,
                                 col_ind_C),
              rocsparse_status_invalid_handle);
}

// ===========================================================================
// spgeam : generic C = alpha * op(A) + beta * op(B)  (CSR, descriptor-based)
// ===========================================================================
namespace
{
    constexpr rocsparse_indextype  GIT   = rocsparse_indextype_i32;
    constexpr rocsparse_index_base GBASE = rocsparse_index_base_zero;
    constexpr rocsparse_datatype   GDT   = rocsparse_datatype_f32_r;
}

class SpGeam : public HandleTest
{
};

// spgeam full pipeline (analysis -> compute) on CSR A = I(3), B = 2*I(3) with
// alpha = beta = 1: C = 1*I + 1*(2I) = 3*I, so nnz_C = 3, row_ptr_C = {0,1,2,3},
// col_ind_C = {0,1,2}, val_C = {3,3,3}.
TEST_F(SpGeam, csr_full_pipeline)
{
    // A = B = 3x3 identity (CSR).
    device_vector<int32_t> row_ptr_A{std::vector<int32_t>{0, 1, 2, 3}};
    device_vector<int32_t> col_ind_A{std::vector<int32_t>{0, 1, 2}};
    device_vector<float>   val_A{std::vector<float>{1.0f, 1.0f, 1.0f}};
    device_vector<int32_t> row_ptr_B{std::vector<int32_t>{0, 1, 2, 3}};
    device_vector<int32_t> col_ind_B{std::vector<int32_t>{0, 1, 2}};
    device_vector<float>   val_B{std::vector<float>{2.0f, 2.0f, 2.0f}};
    ASSERT_TRUE(row_ptr_A.ptr && col_ind_A.ptr && val_A.ptr);
    ASSERT_TRUE(row_ptr_B.ptr && col_ind_B.ptr && val_B.ptr);

    rocsparse_error       p_error[1] = {};
    rocsparse_spmat_descr matA = nullptr, matB = nullptr, matC = nullptr;
    ASSERT_EQ(rocsparse_create_csr_descr(
                  &matA, 3, 3, 3, row_ptr_A, col_ind_A, val_A, GIT, GIT, GBASE, GDT),
              rocsparse_status_success);
    ASSERT_EQ(rocsparse_create_csr_descr(
                  &matB, 3, 3, 3, row_ptr_B, col_ind_B, val_B, GIT, GIT, GBASE, GDT),
              rocsparse_status_success);

    rocsparse_spgeam_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_spgeam_descr(&descr), rocsparse_status_success);

    const rocsparse_spgeam_alg alg     = rocsparse_spgeam_alg_default;
    const rocsparse_operation  trans_A = rocsparse_operation_none;
    const rocsparse_operation  trans_B = rocsparse_operation_none;
    const rocsparse_datatype   sdt     = GDT;
    const rocsparse_datatype   cdt     = GDT;
    ASSERT_EQ(rocsparse_spgeam_set_input(
                  handle, descr, rocsparse_spgeam_input_alg, &alg, sizeof(alg), p_error),
              rocsparse_status_success);
    ASSERT_EQ(
        rocsparse_spgeam_set_input(
            handle, descr, rocsparse_spgeam_input_operation_A, &trans_A, sizeof(trans_A), p_error),
        rocsparse_status_success);
    ASSERT_EQ(
        rocsparse_spgeam_set_input(
            handle, descr, rocsparse_spgeam_input_operation_B, &trans_B, sizeof(trans_B), p_error),
        rocsparse_status_success);
    ASSERT_EQ(
        rocsparse_spgeam_set_input(
            handle, descr, rocsparse_spgeam_input_scalar_datatype, &sdt, sizeof(sdt), p_error),
        rocsparse_status_success);
    ASSERT_EQ(
        rocsparse_spgeam_set_input(
            handle, descr, rocsparse_spgeam_input_compute_datatype, &cdt, sizeof(cdt), p_error),
        rocsparse_status_success);

    // Analysis stage (matC == nullptr).
    size_t buffer_size = 0;
    ASSERT_EQ(rocsparse_spgeam_buffer_size(handle,
                                           descr,
                                           matA,
                                           matB,
                                           nullptr,
                                           rocsparse_spgeam_stage_analysis,
                                           &buffer_size,
                                           p_error),
              rocsparse_status_success);
    {
        device_vector<char> buf{buffer_size ? buffer_size : size_t(1)};
        ASSERT_TRUE(buf.ptr);
        ASSERT_EQ(rocsparse_spgeam(handle,
                                   descr,
                                   matA,
                                   matB,
                                   nullptr,
                                   rocsparse_spgeam_stage_analysis,
                                   buffer_size,
                                   buffer_size ? buf.ptr : nullptr,
                                   p_error),
                  rocsparse_status_success);
    }
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    int64_t nnz_C = 0;
    ASSERT_EQ(rocsparse_spgeam_get_output(
                  handle, descr, rocsparse_spgeam_output_nnz, &nnz_C, sizeof(int64_t), p_error),
              rocsparse_status_success);
    ASSERT_GT(nnz_C, 0);

    device_vector<int32_t> row_ptr_C{size_t(4)};
    device_vector<int32_t> col_ind_C{size_t(nnz_C)};
    device_vector<float>   val_C{size_t(nnz_C)};
    ASSERT_TRUE(row_ptr_C.ptr && col_ind_C.ptr && val_C.ptr);
    ASSERT_EQ(rocsparse_create_csr_descr(
                  &matC, 3, 3, nnz_C, row_ptr_C, col_ind_C, val_C, GIT, GIT, GBASE, GDT),
              rocsparse_status_success);

    ASSERT_EQ(
        rocsparse_spgeam_buffer_size(
            handle, descr, matA, matB, matC, rocsparse_spgeam_stage_compute, &buffer_size, p_error),
        rocsparse_status_success);

    // scalar_alpha/beta store the passed pointer itself as the scalar location
    // (set_scalar_A/B(data)); the expected size is sizeof(void*) and the pointed-to
    // scalars must outlive the compute call.
    const float alpha = 1.0f, beta = 1.0f;
    ASSERT_EQ(
        rocsparse_spgeam_set_input(
            handle, descr, rocsparse_spgeam_input_scalar_alpha, &alpha, sizeof(void*), p_error),
        rocsparse_status_success);
    ASSERT_EQ(rocsparse_spgeam_set_input(
                  handle, descr, rocsparse_spgeam_input_scalar_beta, &beta, sizeof(void*), p_error),
              rocsparse_status_success);

    device_vector<char> buf2{buffer_size ? buffer_size : size_t(1)};
    ASSERT_TRUE(buf2.ptr);
    ASSERT_EQ(rocsparse_spgeam(handle,
                               descr,
                               matA,
                               matB,
                               matC,
                               rocsparse_spgeam_stage_compute,
                               buffer_size,
                               buffer_size ? buf2.ptr : nullptr,
                               p_error),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // C = I + 2I = 3I.
    EXPECT_EQ(nnz_C, 3);
    EXPECT_EQ(to_host(row_ptr_C.ptr, 4), (std::vector<int32_t>{0, 1, 2, 3}));
    EXPECT_EQ(to_host(col_ind_C.ptr, 3), (std::vector<int32_t>{0, 1, 2}));
    EXPECT_EQ(to_host(val_C.ptr, 3), (std::vector<float>{3.0f, 3.0f, 3.0f}));

    EXPECT_EQ(rocsparse_destroy_spmat_descr(matA), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(matB), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(matC), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spgeam_descr(descr), rocsparse_status_success);
}

TEST_F(SpGeam, bad_args)
{
    rocsparse_spgeam_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_spgeam_descr(&descr), rocsparse_status_success);

    const rocsparse_spgeam_alg alg = rocsparse_spgeam_alg_default;
    EXPECT_EQ(rocsparse_spgeam_set_input(
                  nullptr, descr, rocsparse_spgeam_input_alg, &alg, sizeof(alg), nullptr),
              rocsparse_status_invalid_handle);

    EXPECT_EQ(rocsparse_destroy_spgeam_descr(descr), rocsparse_status_success);
}

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
// Host-path unit tests for previously-untested level-2 C APIs:
//   rocsparse_v2_spmv (the new descriptor-based SpMV; CSR + COO),
//   rocsparse_Xbsrxmv (masked block-sparse mv), and
//   rocsparse_Xgemvi (dense matrix * sparse vector).
//
// rocsparse_v2_spmv.cpp sat at ~2% and rocsparse_bsrxmv.cpp at 0%. Each test
// drives the public entry point (buffer-size / analysis / compute where
// applicable) on a tiny problem, so the host dispatch/validation/orchestration
// in library/src/level2 is counted.
//
#include "unit_test_utils.hpp"

#include "rocsparse.h"

using namespace rocsparse_ut;

namespace
{
    constexpr rocsparse_indextype  IT   = rocsparse_indextype_i32;
    constexpr rocsparse_index_base BASE = rocsparse_index_base_zero;
    constexpr rocsparse_datatype   DT   = rocsparse_datatype_f32_r;

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

    // Configure an spmv_descr for float/none/default and run the two-stage
    // (analysis then compute) v2_spmv pipeline against spmat A and dnvecs x,y.
    // A is always the m x n identity here and x = ones, alpha = 1, beta = 0, so
    // y = alpha*A*x + beta*y = ones(m); the result is read back and checked.
    void run_v2_spmv(rocsparse_handle handle, rocsparse_spmat_descr A, int64_t m, int64_t n)
    {
        device_vector<float> x{std::vector<float>(size_t(n), 1.0f)};
        device_vector<float> y{std::vector<float>(size_t(m), 0.0f)};
        ASSERT_TRUE(x.ptr && y.ptr);

        rocsparse_dnvec_descr dx = nullptr, dy = nullptr;
        ASSERT_EQ(rocsparse_create_dnvec_descr(&dx, n, x.ptr, DT), rocsparse_status_success);
        ASSERT_EQ(rocsparse_create_dnvec_descr(&dy, m, y.ptr, DT), rocsparse_status_success);

        rocsparse_spmv_descr spmv_descr = nullptr;
        ASSERT_EQ(rocsparse_create_spmv_descr(&spmv_descr), rocsparse_status_success);

        const rocsparse_spmv_alg  alg = rocsparse_spmv_alg_default;
        const rocsparse_operation op  = rocsparse_operation_none;
        const rocsparse_datatype  sdt = DT;
        const rocsparse_datatype  cdt = DT;
        ASSERT_EQ(rocsparse_spmv_set_input(
                      handle, spmv_descr, rocsparse_spmv_input_alg, &alg, sizeof(alg), nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spmv_set_input(
                      handle, spmv_descr, rocsparse_spmv_input_operation, &op, sizeof(op), nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spmv_set_input(handle,
                                           spmv_descr,
                                           rocsparse_spmv_input_scalar_datatype,
                                           &sdt,
                                           sizeof(sdt),
                                           nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_spmv_set_input(handle,
                                           spmv_descr,
                                           rocsparse_spmv_input_compute_datatype,
                                           &cdt,
                                           sizeof(cdt),
                                           nullptr),
                  rocsparse_status_success);

        const float alpha = 1.0f, beta = 0.0f;

        size_t bs_analysis = 0;
        ASSERT_EQ(rocsparse_v2_spmv_buffer_size(handle,
                                                spmv_descr,
                                                A,
                                                dx,
                                                dy,
                                                rocsparse_v2_spmv_stage_analysis,
                                                &bs_analysis,
                                                nullptr),
                  rocsparse_status_success);
        device_vector<char> buf_a{bs_analysis ? bs_analysis : size_t(1)};
        ASSERT_TRUE(buf_a.ptr);
        // The API rejects a non-null buffer paired with a zero size, so pass
        // nullptr when the stage needs no buffer.
        ASSERT_EQ(rocsparse_v2_spmv(handle,
                                    spmv_descr,
                                    &alpha,
                                    A,
                                    dx,
                                    &beta,
                                    dy,
                                    rocsparse_v2_spmv_stage_analysis,
                                    bs_analysis,
                                    bs_analysis ? buf_a.ptr : nullptr,
                                    nullptr),
                  rocsparse_status_success);

        size_t bs_compute = 0;
        ASSERT_EQ(rocsparse_v2_spmv_buffer_size(handle,
                                                spmv_descr,
                                                A,
                                                dx,
                                                dy,
                                                rocsparse_v2_spmv_stage_compute,
                                                &bs_compute,
                                                nullptr),
                  rocsparse_status_success);
        device_vector<char> buf_c{bs_compute ? bs_compute : size_t(1)};
        ASSERT_TRUE(buf_c.ptr);
        ASSERT_EQ(rocsparse_v2_spmv(handle,
                                    spmv_descr,
                                    &alpha,
                                    A,
                                    dx,
                                    &beta,
                                    dy,
                                    rocsparse_v2_spmv_stage_compute,
                                    bs_compute,
                                    bs_compute ? buf_c.ptr : nullptr,
                                    nullptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        // A = I, x = ones, beta = 0 => y = ones(m).
        auto hy = to_host(y.ptr, size_t(m));
        for(int64_t i = 0; i < m; ++i)
        {
            EXPECT_FLOAT_EQ(hy[size_t(i)], 1.0f) << "y[" << i << "]";
        }

        EXPECT_EQ(rocsparse_destroy_spmv_descr(spmv_descr), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_dnvec_descr(dx), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_dnvec_descr(dy), rocsparse_status_success);
    }
}

// ===========================================================================
// v2_spmv : new descriptor-based SpMV (y = alpha*A*x + beta*y)
// ===========================================================================
class V2Spmv : public HandleTest
{
};

// v2_spmv on a CSR 3x3 identity: y = A*x = ones(3) (checked in run_v2_spmv).
TEST_F(V2Spmv, csr)
{
    device_vector<int32_t> row_ptr{std::vector<int32_t>{0, 1, 2, 3}};
    device_vector<int32_t> col_ind{std::vector<int32_t>{0, 1, 2}};
    device_vector<float>   val{std::vector<float>{1.0f, 1.0f, 1.0f}};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

    rocsparse_spmat_descr A = nullptr;
    ASSERT_EQ(rocsparse_create_csr_descr(&A, 3, 3, 3, row_ptr, col_ind, val, IT, IT, BASE, DT),
              rocsparse_status_success);
    run_v2_spmv(handle, A, 3, 3);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
}

// v2_spmv on a COO 3x3 identity: y = A*x = ones(3) (checked in run_v2_spmv).
TEST_F(V2Spmv, coo)
{
    device_vector<int32_t> row_ind{std::vector<int32_t>{0, 1, 2}};
    device_vector<int32_t> col_ind{std::vector<int32_t>{0, 1, 2}};
    device_vector<float>   val{std::vector<float>{1.0f, 1.0f, 1.0f}};
    ASSERT_TRUE(row_ind.ptr && col_ind.ptr && val.ptr);

    rocsparse_spmat_descr A = nullptr;
    ASSERT_EQ(rocsparse_create_coo_descr(&A, 3, 3, 3, row_ind, col_ind, val, IT, BASE, DT),
              rocsparse_status_success);
    run_v2_spmv(handle, A, 3, 3);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
}

TEST_F(V2Spmv, bad_args)
{
    device_vector<int32_t> row_ptr{std::vector<int32_t>{0, 1, 2, 3}};
    device_vector<int32_t> col_ind{std::vector<int32_t>{0, 1, 2}};
    device_vector<float>   val{std::vector<float>{1.0f, 1.0f, 1.0f}};
    rocsparse_spmat_descr  A = nullptr;
    ASSERT_EQ(rocsparse_create_csr_descr(&A, 3, 3, 3, row_ptr, col_ind, val, IT, IT, BASE, DT),
              rocsparse_status_success);

    device_vector<float>  x{std::vector<float>(3, 1.0f)};
    rocsparse_dnvec_descr dx = nullptr;
    ASSERT_EQ(rocsparse_create_dnvec_descr(&dx, 3, x.ptr, DT), rocsparse_status_success);

    rocsparse_spmv_descr spmv_descr = nullptr;
    ASSERT_EQ(rocsparse_create_spmv_descr(&spmv_descr), rocsparse_status_success);

    size_t bs = 0;
    EXPECT_EQ(rocsparse_v2_spmv_buffer_size(
                  nullptr, spmv_descr, A, dx, dx, rocsparse_v2_spmv_stage_analysis, &bs, nullptr),
              rocsparse_status_invalid_handle);

    EXPECT_EQ(rocsparse_destroy_spmv_descr(spmv_descr), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_dnvec_descr(dx), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_spmat_descr(A), rocsparse_status_success);
}

// ===========================================================================
// bsrxmv : masked block-sparse mv.  bsr_row_ptr/bsr_end_ptr give per-row start/
// end offsets; bsr_mask_ptr selects which block-rows are processed.
// ===========================================================================
class BsrxMV : public HandleTest
{
};

// bsrxmv with a block-diagonal identity (two 2x2 identity blocks) and both
// block-rows selected by the mask: y = A*x = x = {1,2,3,4}.
TEST_F(BsrxMV, masked_diagonal)
{
    // bsrxmv is only implemented for block_dim >= 2, so use 2x2 blocks.
    const rocsparse_direction dir          = rocsparse_direction_row;
    const rocsparse_int       size_of_mask = 2;
    const rocsparse_int       mb           = 2;
    const rocsparse_int       nb           = 2;
    const rocsparse_int       nnzb         = 2;
    const rocsparse_int       block_dim    = 2;

    device_vector<int32_t> mask{std::vector<int32_t>{0, 1}};
    device_vector<int32_t> row_ptr{std::vector<int32_t>{0, 1}};
    device_vector<int32_t> end_ptr{std::vector<int32_t>{1, 2}};
    device_vector<int32_t> col_ind{std::vector<int32_t>{0, 1}};
    // Two 2x2 identity blocks (row-major) -> block-diagonal identity.
    device_vector<float> val{std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f}};
    device_vector<float> x{std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}};
    device_vector<float> y{std::vector<float>(4, 0.0f)};
    ASSERT_TRUE(mask.ptr && row_ptr.ptr && end_ptr.ptr && col_ind.ptr && val.ptr && x.ptr && y.ptr);

    MatDescr    descr;
    const float alpha = 1.0f, beta = 0.0f;
    ASSERT_EQ(rocsparse_sbsrxmv(handle,
                                dir,
                                rocsparse_operation_none,
                                size_of_mask,
                                mb,
                                nb,
                                nnzb,
                                &alpha,
                                descr.d,
                                val,
                                mask,
                                row_ptr,
                                end_ptr,
                                col_ind,
                                block_dim,
                                x,
                                &beta,
                                y),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // Block-diagonal identity => y = x.
    EXPECT_EQ(to_host(y.ptr, 4), (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));
}

TEST_F(BsrxMV, bad_args)
{
    MatDescr               descr;
    device_vector<int32_t> mask{std::vector<int32_t>{0, 1}};
    device_vector<int32_t> row_ptr{std::vector<int32_t>{0, 1}};
    device_vector<int32_t> end_ptr{std::vector<int32_t>{1, 2}};
    device_vector<int32_t> col_ind{std::vector<int32_t>{0, 1}};
    device_vector<float>   val{std::vector<float>{2.0f, 3.0f}};
    device_vector<float>   x{std::vector<float>{1.0f, 1.0f}};
    device_vector<float>   y{std::vector<float>{0.0f, 0.0f}};
    const float            alpha = 1.0f, beta = 0.0f;

    EXPECT_EQ(rocsparse_sbsrxmv(nullptr,
                                rocsparse_direction_row,
                                rocsparse_operation_none,
                                2,
                                2,
                                2,
                                2,
                                &alpha,
                                descr.d,
                                val,
                                mask,
                                row_ptr,
                                end_ptr,
                                col_ind,
                                1,
                                x,
                                &beta,
                                y),
              rocsparse_status_invalid_handle);
}

// ===========================================================================
// gemvi : y = alpha * A * x_sparse + beta * y   (A dense, x sparse)
// ===========================================================================
class Gemvi : public HandleTest
{
};

// gemvi y = alpha*A*x_sparse + beta*y with dense A = [[1,3],[2,4]] and sparse x
// = {index 0 -> 5}. Only column 0 of A contributes: y = 5*{1,2} = {5,10}.
TEST_F(Gemvi, dense_times_sparse_vector)
{
    const rocsparse_int m = 2, n = 2, nnz = 1;

    size_t buffer_size = 0;
    ASSERT_EQ(
        rocsparse_sgemvi_buffer_size(handle, rocsparse_operation_none, m, n, nnz, &buffer_size),
        rocsparse_status_success);
    device_vector<char> tmp{buffer_size ? buffer_size : size_t(1)};
    ASSERT_TRUE(tmp.ptr);

    // A column-major 2x2 = [[1,3],[2,4]] -> {1,2,3,4}, lda=2.
    device_vector<float>   A{std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}};
    device_vector<float>   x_val{std::vector<float>{5.0f}};
    device_vector<int32_t> x_ind{std::vector<int32_t>{0}};
    device_vector<float>   y{std::vector<float>{0.0f, 0.0f}};
    ASSERT_TRUE(A.ptr && x_val.ptr && x_ind.ptr && y.ptr);

    const float alpha = 1.0f, beta = 0.0f;
    ASSERT_EQ(rocsparse_sgemvi(handle,
                               rocsparse_operation_none,
                               m,
                               n,
                               &alpha,
                               A,
                               2,
                               nnz,
                               x_val,
                               x_ind,
                               &beta,
                               y,
                               BASE,
                               tmp.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // y = 5 * column 0 of A = {5, 10}.
    auto hy = to_host(y.ptr, 2);
    EXPECT_FLOAT_EQ(hy[0], 5.0f);
    EXPECT_FLOAT_EQ(hy[1], 10.0f);
}

TEST_F(Gemvi, bad_args)
{
    // gemvi validates the handle first in the compute entry point.
    device_vector<float>   A{std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}};
    device_vector<float>   x_val{std::vector<float>{5.0f}};
    device_vector<int32_t> x_ind{std::vector<int32_t>{0}};
    device_vector<float>   y{std::vector<float>{0.0f, 0.0f}};
    device_vector<char>    tmp{size_t(16)};
    const float            alpha = 1.0f, beta = 0.0f;
    EXPECT_EQ(rocsparse_sgemvi(nullptr,
                               rocsparse_operation_none,
                               2,
                               2,
                               &alpha,
                               A,
                               2,
                               1,
                               x_val,
                               x_ind,
                               &beta,
                               y,
                               BASE,
                               tmp.ptr),
              rocsparse_status_invalid_handle);
}

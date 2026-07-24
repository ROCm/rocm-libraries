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

#include <type_traits>

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

// ======================================================================
// csric0 : buffer_size -> analysis -> compute -> clear on a tiny SPD
// diagonal matrix diag(2,3,4). IC0 of a diagonal is trivially the sqrt of
// the diagonal (no zero pivot), so the pipeline is well-conditioned for all
// four precisions.
// ======================================================================
class PrecondCsric0 : public HandleTest
{
};

template <typename T>
static void run_csric0_pipeline(rocsparse_handle handle)
{
    const rocsparse_int          m = 3, nnz = 3;
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<T> val{std::vector<T>{scalar<T>(2), scalar<T>(3), scalar<T>(4)}};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    rocsparse_mat_info info = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

    size_t           buffer_size = 0;
    rocsparse_status st;
    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_scsric0_buffer_size(
            handle, m, nnz, descr, val, row_ptr, col_ind, info, &buffer_size);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dcsric0_buffer_size(
            handle, m, nnz, descr, val, row_ptr, col_ind, info, &buffer_size);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_ccsric0_buffer_size(
            handle, m, nnz, descr, val, row_ptr, col_ind, info, &buffer_size);
    else
        st = rocsparse_zcsric0_buffer_size(
            handle, m, nnz, descr, val, row_ptr, col_ind, info, &buffer_size);

    if(st == rocsparse_status_not_implemented)
    {
        EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
        return;
    }
    ASSERT_EQ(st, rocsparse_status_success);
    EXPECT_GT(buffer_size, 0u);

    device_vector<char> buffer{buffer_size};
    ASSERT_TRUE(buffer.ptr);

    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_scsric0_analysis(handle,
                                        m,
                                        nnz,
                                        descr,
                                        val,
                                        row_ptr,
                                        col_ind,
                                        info,
                                        rocsparse_analysis_policy_reuse,
                                        rocsparse_solve_policy_auto,
                                        buffer.ptr);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dcsric0_analysis(handle,
                                        m,
                                        nnz,
                                        descr,
                                        val,
                                        row_ptr,
                                        col_ind,
                                        info,
                                        rocsparse_analysis_policy_reuse,
                                        rocsparse_solve_policy_auto,
                                        buffer.ptr);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_ccsric0_analysis(handle,
                                        m,
                                        nnz,
                                        descr,
                                        val,
                                        row_ptr,
                                        col_ind,
                                        info,
                                        rocsparse_analysis_policy_reuse,
                                        rocsparse_solve_policy_auto,
                                        buffer.ptr);
    else
        st = rocsparse_zcsric0_analysis(handle,
                                        m,
                                        nnz,
                                        descr,
                                        val,
                                        row_ptr,
                                        col_ind,
                                        info,
                                        rocsparse_analysis_policy_reuse,
                                        rocsparse_solve_policy_auto,
                                        buffer.ptr);
    ASSERT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_scsric0(
            handle, m, nnz, descr, val, row_ptr, col_ind, info, rocsparse_solve_policy_auto, buffer.ptr);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dcsric0(
            handle, m, nnz, descr, val, row_ptr, col_ind, info, rocsparse_solve_policy_auto, buffer.ptr);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_ccsric0(
            handle, m, nnz, descr, val, row_ptr, col_ind, info, rocsparse_solve_policy_auto, buffer.ptr);
    else
        st = rocsparse_zcsric0(
            handle, m, nnz, descr, val, row_ptr, col_ind, info, rocsparse_solve_policy_auto, buffer.ptr);
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // No zero pivot expected on a positive diagonal.
    rocsparse_int position = -2;
    EXPECT_EQ(rocsparse_csric0_zero_pivot(handle, info, &position), rocsparse_status_success);
    EXPECT_EQ(position, -1);

    EXPECT_EQ(rocsparse_csric0_clear(handle, info), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

TEST_F(PrecondCsric0, pipeline_float)
{
    run_csric0_pipeline<float>(handle);
}
TEST_F(PrecondCsric0, pipeline_double)
{
    run_csric0_pipeline<double>(handle);
}
TEST_F(PrecondCsric0, pipeline_float_complex)
{
    run_csric0_pipeline<rocsparse_float_complex>(handle);
}
TEST_F(PrecondCsric0, pipeline_double_complex)
{
    run_csric0_pipeline<rocsparse_double_complex>(handle);
}

TEST_F(PrecondCsric0, bad_args)
{
    const rocsparse_int          m = 3, nnz = 3;
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<float>         val{std::vector<float>{2, 3, 4}};
    rocsparse_mat_descr          descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    rocsparse_mat_info info = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

    size_t buffer_size = 0;
    EXPECT_EQ(rocsparse_scsric0_buffer_size(
                  nullptr, m, nnz, descr, val, row_ptr, col_ind, info, &buffer_size),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_scsric0_buffer_size(
                  handle, m, nnz, nullptr, val, row_ptr, col_ind, info, &buffer_size),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_scsric0_buffer_size(
                  handle, -1, nnz, descr, val, row_ptr, col_ind, info, &buffer_size),
              rocsparse_status_invalid_size);
    EXPECT_EQ(rocsparse_scsric0_buffer_size(
                  handle, m, -1, descr, val, row_ptr, col_ind, info, &buffer_size),
              rocsparse_status_invalid_size);

    rocsparse_int position = -2;
    EXPECT_EQ(rocsparse_csric0_zero_pivot(nullptr, info, &position),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_csric0_clear(nullptr, info), rocsparse_status_invalid_handle);

    EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

// ======================================================================
// bsric0 : block-diagonal BSR (mb=2, nnzb=2, block_dim=2) with SPD 2x2
// diagonal blocks [[4,1],[1,4]] (eigenvalues 3,5). IC0 well-conditioned.
// ======================================================================
class PrecondBsric0 : public HandleTest
{
};

template <typename T>
static void run_bsric0_pipeline(rocsparse_handle handle)
{
    const rocsparse_direction dir = rocsparse_direction_row;
    const rocsparse_int       mb = 2, nnzb = 2, block_dim = 2;
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1}};
    // Two SPD blocks [[4,1],[1,4]] stored row-major.
    device_vector<T>             val{std::vector<T>{scalar<T>(4),
                                                   scalar<T>(1),
                                                   scalar<T>(1),
                                                   scalar<T>(4),
                                                   scalar<T>(4),
                                                   scalar<T>(1),
                                                   scalar<T>(1),
                                                   scalar<T>(4)}};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    rocsparse_mat_info info = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

    size_t           buffer_size = 0;
    rocsparse_status st;
    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_sbsric0_buffer_size(
            handle, dir, mb, nnzb, descr, val, row_ptr, col_ind, block_dim, info, &buffer_size);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dbsric0_buffer_size(
            handle, dir, mb, nnzb, descr, val, row_ptr, col_ind, block_dim, info, &buffer_size);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_cbsric0_buffer_size(
            handle, dir, mb, nnzb, descr, val, row_ptr, col_ind, block_dim, info, &buffer_size);
    else
        st = rocsparse_zbsric0_buffer_size(
            handle, dir, mb, nnzb, descr, val, row_ptr, col_ind, block_dim, info, &buffer_size);

    if(st == rocsparse_status_not_implemented)
    {
        EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
        return;
    }
    ASSERT_EQ(st, rocsparse_status_success);
    EXPECT_GT(buffer_size, 0u);

    device_vector<char> buffer{buffer_size};
    ASSERT_TRUE(buffer.ptr);

    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_sbsric0_analysis(handle,
                                        dir,
                                        mb,
                                        nnzb,
                                        descr,
                                        val,
                                        row_ptr,
                                        col_ind,
                                        block_dim,
                                        info,
                                        rocsparse_analysis_policy_reuse,
                                        rocsparse_solve_policy_auto,
                                        buffer.ptr);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dbsric0_analysis(handle,
                                        dir,
                                        mb,
                                        nnzb,
                                        descr,
                                        val,
                                        row_ptr,
                                        col_ind,
                                        block_dim,
                                        info,
                                        rocsparse_analysis_policy_reuse,
                                        rocsparse_solve_policy_auto,
                                        buffer.ptr);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_cbsric0_analysis(handle,
                                        dir,
                                        mb,
                                        nnzb,
                                        descr,
                                        val,
                                        row_ptr,
                                        col_ind,
                                        block_dim,
                                        info,
                                        rocsparse_analysis_policy_reuse,
                                        rocsparse_solve_policy_auto,
                                        buffer.ptr);
    else
        st = rocsparse_zbsric0_analysis(handle,
                                        dir,
                                        mb,
                                        nnzb,
                                        descr,
                                        val,
                                        row_ptr,
                                        col_ind,
                                        block_dim,
                                        info,
                                        rocsparse_analysis_policy_reuse,
                                        rocsparse_solve_policy_auto,
                                        buffer.ptr);
    ASSERT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_sbsric0(handle,
                               dir,
                               mb,
                               nnzb,
                               descr,
                               val,
                               row_ptr,
                               col_ind,
                               block_dim,
                               info,
                               rocsparse_solve_policy_auto,
                               buffer.ptr);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dbsric0(handle,
                               dir,
                               mb,
                               nnzb,
                               descr,
                               val,
                               row_ptr,
                               col_ind,
                               block_dim,
                               info,
                               rocsparse_solve_policy_auto,
                               buffer.ptr);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_cbsric0(handle,
                               dir,
                               mb,
                               nnzb,
                               descr,
                               val,
                               row_ptr,
                               col_ind,
                               block_dim,
                               info,
                               rocsparse_solve_policy_auto,
                               buffer.ptr);
    else
        st = rocsparse_zbsric0(handle,
                               dir,
                               mb,
                               nnzb,
                               descr,
                               val,
                               row_ptr,
                               col_ind,
                               block_dim,
                               info,
                               rocsparse_solve_policy_auto,
                               buffer.ptr);
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    rocsparse_int position = -2;
    EXPECT_EQ(rocsparse_bsric0_zero_pivot(handle, info, &position), rocsparse_status_success);
    EXPECT_EQ(position, -1);

    EXPECT_EQ(rocsparse_bsric0_clear(handle, info), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

TEST_F(PrecondBsric0, pipeline_float)
{
    run_bsric0_pipeline<float>(handle);
}
TEST_F(PrecondBsric0, pipeline_double)
{
    run_bsric0_pipeline<double>(handle);
}
TEST_F(PrecondBsric0, pipeline_float_complex)
{
    run_bsric0_pipeline<rocsparse_float_complex>(handle);
}
TEST_F(PrecondBsric0, pipeline_double_complex)
{
    run_bsric0_pipeline<rocsparse_double_complex>(handle);
}

TEST_F(PrecondBsric0, bad_args)
{
    const rocsparse_direction    dir = rocsparse_direction_row;
    const rocsparse_int          mb = 2, nnzb = 2, block_dim = 2;
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1}};
    device_vector<float>         val{std::vector<float>{4, 1, 1, 4, 4, 1, 1, 4}};
    rocsparse_mat_descr          descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    rocsparse_mat_info info = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

    size_t buffer_size = 0;
    EXPECT_EQ(rocsparse_sbsric0_buffer_size(
                  nullptr, dir, mb, nnzb, descr, val, row_ptr, col_ind, block_dim, info, &buffer_size),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_sbsric0_buffer_size(
                  handle, dir, mb, nnzb, nullptr, val, row_ptr, col_ind, block_dim, info, &buffer_size),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_sbsric0_buffer_size(
                  handle, dir, -1, nnzb, descr, val, row_ptr, col_ind, block_dim, info, &buffer_size),
              rocsparse_status_invalid_size);
    EXPECT_EQ(rocsparse_sbsric0_buffer_size(
                  handle, dir, mb, nnzb, descr, val, row_ptr, col_ind, -1, info, &buffer_size),
              rocsparse_status_invalid_size);

    EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

// ======================================================================
// bsrilu0 : block-diagonal BSR (mb=2, nnzb=2, block_dim=2) with invertible
// diagonally-dominant 2x2 blocks [[4,1],[1,4]]. ILU0 well-conditioned.
// ======================================================================
class PrecondBsrilu0 : public HandleTest
{
};

template <typename T>
static void run_bsrilu0_pipeline(rocsparse_handle handle)
{
    const rocsparse_direction dir = rocsparse_direction_row;
    const rocsparse_int       mb = 2, nnzb = 2, block_dim = 2;
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1}};
    device_vector<T>             val{std::vector<T>{scalar<T>(4),
                                                   scalar<T>(1),
                                                   scalar<T>(1),
                                                   scalar<T>(4),
                                                   scalar<T>(4),
                                                   scalar<T>(1),
                                                   scalar<T>(1),
                                                   scalar<T>(4)}};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    rocsparse_mat_info info = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

    size_t           buffer_size = 0;
    rocsparse_status st;
    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_sbsrilu0_buffer_size(
            handle, dir, mb, nnzb, descr, val, row_ptr, col_ind, block_dim, info, &buffer_size);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dbsrilu0_buffer_size(
            handle, dir, mb, nnzb, descr, val, row_ptr, col_ind, block_dim, info, &buffer_size);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_cbsrilu0_buffer_size(
            handle, dir, mb, nnzb, descr, val, row_ptr, col_ind, block_dim, info, &buffer_size);
    else
        st = rocsparse_zbsrilu0_buffer_size(
            handle, dir, mb, nnzb, descr, val, row_ptr, col_ind, block_dim, info, &buffer_size);

    if(st == rocsparse_status_not_implemented)
    {
        EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
        return;
    }
    ASSERT_EQ(st, rocsparse_status_success);
    EXPECT_GT(buffer_size, 0u);

    device_vector<char> buffer{buffer_size};
    ASSERT_TRUE(buffer.ptr);

    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_sbsrilu0_analysis(handle,
                                         dir,
                                         mb,
                                         nnzb,
                                         descr,
                                         val,
                                         row_ptr,
                                         col_ind,
                                         block_dim,
                                         info,
                                         rocsparse_analysis_policy_reuse,
                                         rocsparse_solve_policy_auto,
                                         buffer.ptr);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dbsrilu0_analysis(handle,
                                         dir,
                                         mb,
                                         nnzb,
                                         descr,
                                         val,
                                         row_ptr,
                                         col_ind,
                                         block_dim,
                                         info,
                                         rocsparse_analysis_policy_reuse,
                                         rocsparse_solve_policy_auto,
                                         buffer.ptr);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_cbsrilu0_analysis(handle,
                                         dir,
                                         mb,
                                         nnzb,
                                         descr,
                                         val,
                                         row_ptr,
                                         col_ind,
                                         block_dim,
                                         info,
                                         rocsparse_analysis_policy_reuse,
                                         rocsparse_solve_policy_auto,
                                         buffer.ptr);
    else
        st = rocsparse_zbsrilu0_analysis(handle,
                                         dir,
                                         mb,
                                         nnzb,
                                         descr,
                                         val,
                                         row_ptr,
                                         col_ind,
                                         block_dim,
                                         info,
                                         rocsparse_analysis_policy_reuse,
                                         rocsparse_solve_policy_auto,
                                         buffer.ptr);
    ASSERT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_sbsrilu0(handle,
                                dir,
                                mb,
                                nnzb,
                                descr,
                                val,
                                row_ptr,
                                col_ind,
                                block_dim,
                                info,
                                rocsparse_solve_policy_auto,
                                buffer.ptr);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dbsrilu0(handle,
                                dir,
                                mb,
                                nnzb,
                                descr,
                                val,
                                row_ptr,
                                col_ind,
                                block_dim,
                                info,
                                rocsparse_solve_policy_auto,
                                buffer.ptr);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_cbsrilu0(handle,
                                dir,
                                mb,
                                nnzb,
                                descr,
                                val,
                                row_ptr,
                                col_ind,
                                block_dim,
                                info,
                                rocsparse_solve_policy_auto,
                                buffer.ptr);
    else
        st = rocsparse_zbsrilu0(handle,
                                dir,
                                mb,
                                nnzb,
                                descr,
                                val,
                                row_ptr,
                                col_ind,
                                block_dim,
                                info,
                                rocsparse_solve_policy_auto,
                                buffer.ptr);
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    rocsparse_int position = -2;
    EXPECT_EQ(rocsparse_bsrilu0_zero_pivot(handle, info, &position), rocsparse_status_success);
    EXPECT_EQ(position, -1);

    EXPECT_EQ(rocsparse_bsrilu0_clear(handle, info), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

TEST_F(PrecondBsrilu0, pipeline_float)
{
    run_bsrilu0_pipeline<float>(handle);
}
TEST_F(PrecondBsrilu0, pipeline_double)
{
    run_bsrilu0_pipeline<double>(handle);
}
TEST_F(PrecondBsrilu0, pipeline_float_complex)
{
    run_bsrilu0_pipeline<rocsparse_float_complex>(handle);
}
TEST_F(PrecondBsrilu0, pipeline_double_complex)
{
    run_bsrilu0_pipeline<rocsparse_double_complex>(handle);
}

TEST_F(PrecondBsrilu0, numeric_boost_and_bad_args)
{
    const rocsparse_direction    dir = rocsparse_direction_row;
    const rocsparse_int          mb = 2, nnzb = 2, block_dim = 2;
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1}};
    device_vector<float>         val{std::vector<float>{4, 1, 1, 4, 4, 1, 1, 4}};
    rocsparse_mat_descr          descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    rocsparse_mat_info info = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

    // numeric_boost: exercise host-side option setter (disabled).
    const float boost_tol = 0.0f, boost_val = 1.0f;
    EXPECT_EQ(rocsparse_sbsrilu0_numeric_boost(handle, info, 0, &boost_tol, &boost_val),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_sbsrilu0_numeric_boost(nullptr, info, 0, &boost_tol, &boost_val),
              rocsparse_status_invalid_handle);

    size_t buffer_size = 0;
    EXPECT_EQ(
        rocsparse_sbsrilu0_buffer_size(
            nullptr, dir, mb, nnzb, descr, val, row_ptr, col_ind, block_dim, info, &buffer_size),
        rocsparse_status_invalid_handle);
    EXPECT_EQ(
        rocsparse_sbsrilu0_buffer_size(
            handle, dir, mb, nnzb, nullptr, val, row_ptr, col_ind, block_dim, info, &buffer_size),
        rocsparse_status_invalid_pointer);
    EXPECT_EQ(
        rocsparse_sbsrilu0_buffer_size(
            handle, dir, -1, nnzb, descr, val, row_ptr, col_ind, block_dim, info, &buffer_size),
        rocsparse_status_invalid_size);

    EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

// ======================================================================
// gtsv family : tiny diagonally-dominant tridiagonal systems.
// ======================================================================
class PrecondGtsv : public HandleTest
{
};

template <typename T>
static void run_gtsv_pipeline(rocsparse_handle handle)
{
    const rocsparse_int m = 4, n = 1, ldb = 4;
    device_vector<T>    dl{std::vector<T>{scalar<T>(0), scalar<T>(1), scalar<T>(1), scalar<T>(1)}};
    device_vector<T>    d{std::vector<T>{scalar<T>(4), scalar<T>(4), scalar<T>(4), scalar<T>(4)}};
    device_vector<T>    du{std::vector<T>{scalar<T>(1), scalar<T>(1), scalar<T>(1), scalar<T>(0)}};
    device_vector<T>    B{std::vector<T>{scalar<T>(1), scalar<T>(2), scalar<T>(3), scalar<T>(4)}};
    ASSERT_TRUE(dl.ptr && d.ptr && du.ptr && B.ptr);

    size_t           buffer_size = 0;
    rocsparse_status st;
    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_sgtsv_buffer_size(handle, m, n, dl, d, du, B, ldb, &buffer_size);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dgtsv_buffer_size(handle, m, n, dl, d, du, B, ldb, &buffer_size);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_cgtsv_buffer_size(handle, m, n, dl, d, du, B, ldb, &buffer_size);
    else
        st = rocsparse_zgtsv_buffer_size(handle, m, n, dl, d, du, B, ldb, &buffer_size);

    if(st == rocsparse_status_not_implemented)
        return;
    ASSERT_EQ(st, rocsparse_status_success);
    EXPECT_GT(buffer_size, 0u);

    device_vector<char> buffer{buffer_size};
    ASSERT_TRUE(buffer.ptr);

    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_sgtsv(handle, m, n, dl, d, du, B, ldb, buffer.ptr);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dgtsv(handle, m, n, dl, d, du, B, ldb, buffer.ptr);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_cgtsv(handle, m, n, dl, d, du, B, ldb, buffer.ptr);
    else
        st = rocsparse_zgtsv(handle, m, n, dl, d, du, B, ldb, buffer.ptr);
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
}

TEST_F(PrecondGtsv, pipeline_float)
{
    run_gtsv_pipeline<float>(handle);
}
TEST_F(PrecondGtsv, pipeline_double)
{
    run_gtsv_pipeline<double>(handle);
}
TEST_F(PrecondGtsv, pipeline_float_complex)
{
    run_gtsv_pipeline<rocsparse_float_complex>(handle);
}
TEST_F(PrecondGtsv, pipeline_double_complex)
{
    run_gtsv_pipeline<rocsparse_double_complex>(handle);
}

template <typename T>
static void run_gtsv_no_pivot_pipeline(rocsparse_handle handle)
{
    const rocsparse_int m = 4, n = 1, ldb = 4;
    device_vector<T>    dl{std::vector<T>{scalar<T>(0), scalar<T>(1), scalar<T>(1), scalar<T>(1)}};
    device_vector<T>    d{std::vector<T>{scalar<T>(4), scalar<T>(4), scalar<T>(4), scalar<T>(4)}};
    device_vector<T>    du{std::vector<T>{scalar<T>(1), scalar<T>(1), scalar<T>(1), scalar<T>(0)}};
    device_vector<T>    B{std::vector<T>{scalar<T>(1), scalar<T>(2), scalar<T>(3), scalar<T>(4)}};
    ASSERT_TRUE(dl.ptr && d.ptr && du.ptr && B.ptr);

    size_t           buffer_size = 0;
    rocsparse_status st;
    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_sgtsv_no_pivot_buffer_size(handle, m, n, dl, d, du, B, ldb, &buffer_size);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dgtsv_no_pivot_buffer_size(handle, m, n, dl, d, du, B, ldb, &buffer_size);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_cgtsv_no_pivot_buffer_size(handle, m, n, dl, d, du, B, ldb, &buffer_size);
    else
        st = rocsparse_zgtsv_no_pivot_buffer_size(handle, m, n, dl, d, du, B, ldb, &buffer_size);

    if(st == rocsparse_status_not_implemented)
        return;
    ASSERT_EQ(st, rocsparse_status_success);
    // A tiny system may legitimately need a zero-size buffer.

    device_vector<char> buffer{buffer_size ? buffer_size : size_t(1)};
    ASSERT_TRUE(buffer.ptr);

    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_sgtsv_no_pivot(handle, m, n, dl, d, du, B, ldb, buffer.ptr);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dgtsv_no_pivot(handle, m, n, dl, d, du, B, ldb, buffer.ptr);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_cgtsv_no_pivot(handle, m, n, dl, d, du, B, ldb, buffer.ptr);
    else
        st = rocsparse_zgtsv_no_pivot(handle, m, n, dl, d, du, B, ldb, buffer.ptr);
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
}

TEST_F(PrecondGtsv, no_pivot_float)
{
    run_gtsv_no_pivot_pipeline<float>(handle);
}
TEST_F(PrecondGtsv, no_pivot_double)
{
    run_gtsv_no_pivot_pipeline<double>(handle);
}
TEST_F(PrecondGtsv, no_pivot_float_complex)
{
    run_gtsv_no_pivot_pipeline<rocsparse_float_complex>(handle);
}
TEST_F(PrecondGtsv, no_pivot_double_complex)
{
    run_gtsv_no_pivot_pipeline<rocsparse_double_complex>(handle);
}

template <typename T>
static void run_gtsv_no_pivot_strided_batch_pipeline(rocsparse_handle handle)
{
    const rocsparse_int m = 4, batch_count = 2, batch_stride = 4;
    device_vector<T>    dl{std::vector<T>{scalar<T>(0),
                                       scalar<T>(1),
                                       scalar<T>(1),
                                       scalar<T>(1),
                                       scalar<T>(0),
                                       scalar<T>(1),
                                       scalar<T>(1),
                                       scalar<T>(1)}};
    device_vector<T>    d{std::vector<T>{scalar<T>(4),
                                      scalar<T>(4),
                                      scalar<T>(4),
                                      scalar<T>(4),
                                      scalar<T>(4),
                                      scalar<T>(4),
                                      scalar<T>(4),
                                      scalar<T>(4)}};
    device_vector<T>    du{std::vector<T>{scalar<T>(1),
                                       scalar<T>(1),
                                       scalar<T>(1),
                                       scalar<T>(0),
                                       scalar<T>(1),
                                       scalar<T>(1),
                                       scalar<T>(1),
                                       scalar<T>(0)}};
    device_vector<T>    x{std::vector<T>{scalar<T>(1),
                                      scalar<T>(2),
                                      scalar<T>(3),
                                      scalar<T>(4),
                                      scalar<T>(1),
                                      scalar<T>(2),
                                      scalar<T>(3),
                                      scalar<T>(4)}};
    ASSERT_TRUE(dl.ptr && d.ptr && du.ptr && x.ptr);

    size_t           buffer_size = 0;
    rocsparse_status st;
    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_sgtsv_no_pivot_strided_batch_buffer_size(
            handle, m, dl, d, du, x, batch_count, batch_stride, &buffer_size);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dgtsv_no_pivot_strided_batch_buffer_size(
            handle, m, dl, d, du, x, batch_count, batch_stride, &buffer_size);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_cgtsv_no_pivot_strided_batch_buffer_size(
            handle, m, dl, d, du, x, batch_count, batch_stride, &buffer_size);
    else
        st = rocsparse_zgtsv_no_pivot_strided_batch_buffer_size(
            handle, m, dl, d, du, x, batch_count, batch_stride, &buffer_size);

    if(st == rocsparse_status_not_implemented)
        return;
    ASSERT_EQ(st, rocsparse_status_success);
    // A tiny system may legitimately need a zero-size buffer.

    device_vector<char> buffer{buffer_size ? buffer_size : size_t(1)};
    ASSERT_TRUE(buffer.ptr);

    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_sgtsv_no_pivot_strided_batch(
            handle, m, dl, d, du, x, batch_count, batch_stride, buffer.ptr);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dgtsv_no_pivot_strided_batch(
            handle, m, dl, d, du, x, batch_count, batch_stride, buffer.ptr);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_cgtsv_no_pivot_strided_batch(
            handle, m, dl, d, du, x, batch_count, batch_stride, buffer.ptr);
    else
        st = rocsparse_zgtsv_no_pivot_strided_batch(
            handle, m, dl, d, du, x, batch_count, batch_stride, buffer.ptr);
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
}

TEST_F(PrecondGtsv, no_pivot_strided_batch_float)
{
    run_gtsv_no_pivot_strided_batch_pipeline<float>(handle);
}
TEST_F(PrecondGtsv, no_pivot_strided_batch_double)
{
    run_gtsv_no_pivot_strided_batch_pipeline<double>(handle);
}
TEST_F(PrecondGtsv, no_pivot_strided_batch_float_complex)
{
    run_gtsv_no_pivot_strided_batch_pipeline<rocsparse_float_complex>(handle);
}
TEST_F(PrecondGtsv, no_pivot_strided_batch_double_complex)
{
    run_gtsv_no_pivot_strided_batch_pipeline<rocsparse_double_complex>(handle);
}

template <typename T>
static void run_gtsv_interleaved_batch_pipeline(rocsparse_handle handle)
{
    const rocsparse_gtsv_interleaved_alg alg = rocsparse_gtsv_interleaved_alg_default;
    const rocsparse_int                  m = 4, batch_count = 2, batch_stride = 2;
    // Interleaved layout: element j of system i at index j*batch_stride + i.
    device_vector<T> dl{std::vector<T>{scalar<T>(0),
                                       scalar<T>(0),
                                       scalar<T>(1),
                                       scalar<T>(1),
                                       scalar<T>(1),
                                       scalar<T>(1),
                                       scalar<T>(1),
                                       scalar<T>(1)}};
    device_vector<T> d{std::vector<T>{scalar<T>(4),
                                      scalar<T>(4),
                                      scalar<T>(4),
                                      scalar<T>(4),
                                      scalar<T>(4),
                                      scalar<T>(4),
                                      scalar<T>(4),
                                      scalar<T>(4)}};
    device_vector<T> du{std::vector<T>{scalar<T>(1),
                                       scalar<T>(1),
                                       scalar<T>(1),
                                       scalar<T>(1),
                                       scalar<T>(1),
                                       scalar<T>(1),
                                       scalar<T>(0),
                                       scalar<T>(0)}};
    device_vector<T> x{std::vector<T>{scalar<T>(1),
                                      scalar<T>(1),
                                      scalar<T>(2),
                                      scalar<T>(2),
                                      scalar<T>(3),
                                      scalar<T>(3),
                                      scalar<T>(4),
                                      scalar<T>(4)}};
    ASSERT_TRUE(dl.ptr && d.ptr && du.ptr && x.ptr);

    size_t           buffer_size = 0;
    rocsparse_status st;
    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_sgtsv_interleaved_batch_buffer_size(
            handle, alg, m, dl, d, du, x, batch_count, batch_stride, &buffer_size);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dgtsv_interleaved_batch_buffer_size(
            handle, alg, m, dl, d, du, x, batch_count, batch_stride, &buffer_size);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_cgtsv_interleaved_batch_buffer_size(
            handle, alg, m, dl, d, du, x, batch_count, batch_stride, &buffer_size);
    else
        st = rocsparse_zgtsv_interleaved_batch_buffer_size(
            handle, alg, m, dl, d, du, x, batch_count, batch_stride, &buffer_size);

    if(st == rocsparse_status_not_implemented)
        return;
    ASSERT_EQ(st, rocsparse_status_success);
    EXPECT_GT(buffer_size, 0u);

    device_vector<char> buffer{buffer_size};
    ASSERT_TRUE(buffer.ptr);

    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_sgtsv_interleaved_batch(
            handle, alg, m, dl, d, du, x, batch_count, batch_stride, buffer.ptr);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dgtsv_interleaved_batch(
            handle, alg, m, dl, d, du, x, batch_count, batch_stride, buffer.ptr);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_cgtsv_interleaved_batch(
            handle, alg, m, dl, d, du, x, batch_count, batch_stride, buffer.ptr);
    else
        st = rocsparse_zgtsv_interleaved_batch(
            handle, alg, m, dl, d, du, x, batch_count, batch_stride, buffer.ptr);
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
}

TEST_F(PrecondGtsv, interleaved_batch_float)
{
    run_gtsv_interleaved_batch_pipeline<float>(handle);
}
TEST_F(PrecondGtsv, interleaved_batch_double)
{
    run_gtsv_interleaved_batch_pipeline<double>(handle);
}
TEST_F(PrecondGtsv, interleaved_batch_float_complex)
{
    run_gtsv_interleaved_batch_pipeline<rocsparse_float_complex>(handle);
}
TEST_F(PrecondGtsv, interleaved_batch_double_complex)
{
    run_gtsv_interleaved_batch_pipeline<rocsparse_double_complex>(handle);
}

TEST_F(PrecondGtsv, bad_args)
{
    const rocsparse_int  m = 4, n = 1, ldb = 4;
    device_vector<float> dl{std::vector<float>{0, 1, 1, 1}};
    device_vector<float> d{std::vector<float>{4, 4, 4, 4}};
    device_vector<float> du{std::vector<float>{1, 1, 1, 0}};
    device_vector<float> B{std::vector<float>{1, 2, 3, 4}};
    ASSERT_TRUE(dl.ptr && d.ptr && du.ptr && B.ptr);

    size_t buffer_size = 0;
    EXPECT_EQ(rocsparse_sgtsv_buffer_size(nullptr, m, n, dl, d, du, B, ldb, &buffer_size),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_sgtsv_buffer_size(handle, m, n, nullptr, d, du, B, ldb, &buffer_size),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_sgtsv_buffer_size(handle, -1, n, dl, d, du, B, ldb, &buffer_size),
              rocsparse_status_invalid_size);

    EXPECT_EQ(
        rocsparse_sgtsv_no_pivot_buffer_size(nullptr, m, n, dl, d, du, B, ldb, &buffer_size),
        rocsparse_status_invalid_handle);
    EXPECT_EQ(
        rocsparse_sgtsv_no_pivot_buffer_size(handle, m, n, nullptr, d, du, B, ldb, &buffer_size),
        rocsparse_status_invalid_pointer);

    EXPECT_EQ(rocsparse_sgtsv_no_pivot_strided_batch_buffer_size(
                  nullptr, m, dl, d, du, B, 1, m, &buffer_size),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_sgtsv_interleaved_batch_buffer_size(
                  nullptr, rocsparse_gtsv_interleaved_alg_default, m, dl, d, du, B, 1, 1, &buffer_size),
              rocsparse_status_invalid_handle);
}

// ======================================================================
// gpsv : batched pentadiagonal solver (interleaved). m=4, batch_count=2.
// Diagonally-dominant main diagonal keeps the QR solve well-conditioned.
// ======================================================================
class PrecondGpsv : public HandleTest
{
};

template <typename T>
static void run_gpsv_interleaved_batch_pipeline(rocsparse_handle handle)
{
    const rocsparse_gpsv_interleaved_alg alg = rocsparse_gpsv_interleaved_alg_default;
    const rocsparse_int                  m = 4, batch_count = 2, batch_stride = 2;
    // Interleaved layout: element j of system i at index j*batch_stride + i.
    device_vector<T> ds{std::vector<T>{scalar<T>(0),
                                       scalar<T>(0),
                                       scalar<T>(0),
                                       scalar<T>(0),
                                       scalar<T>(1),
                                       scalar<T>(1),
                                       scalar<T>(1),
                                       scalar<T>(1)}};
    device_vector<T> dl{std::vector<T>{scalar<T>(0),
                                       scalar<T>(0),
                                       scalar<T>(1),
                                       scalar<T>(1),
                                       scalar<T>(1),
                                       scalar<T>(1),
                                       scalar<T>(1),
                                       scalar<T>(1)}};
    device_vector<T> d{std::vector<T>{scalar<T>(10),
                                      scalar<T>(10),
                                      scalar<T>(10),
                                      scalar<T>(10),
                                      scalar<T>(10),
                                      scalar<T>(10),
                                      scalar<T>(10),
                                      scalar<T>(10)}};
    device_vector<T> du{std::vector<T>{scalar<T>(1),
                                       scalar<T>(1),
                                       scalar<T>(1),
                                       scalar<T>(1),
                                       scalar<T>(1),
                                       scalar<T>(1),
                                       scalar<T>(0),
                                       scalar<T>(0)}};
    device_vector<T> dw{std::vector<T>{scalar<T>(1),
                                       scalar<T>(1),
                                       scalar<T>(1),
                                       scalar<T>(1),
                                       scalar<T>(0),
                                       scalar<T>(0),
                                       scalar<T>(0),
                                       scalar<T>(0)}};
    device_vector<T> x{std::vector<T>{scalar<T>(1),
                                      scalar<T>(1),
                                      scalar<T>(2),
                                      scalar<T>(2),
                                      scalar<T>(3),
                                      scalar<T>(3),
                                      scalar<T>(4),
                                      scalar<T>(4)}};
    ASSERT_TRUE(ds.ptr && dl.ptr && d.ptr && du.ptr && dw.ptr && x.ptr);

    size_t           buffer_size = 0;
    rocsparse_status st;
    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_sgpsv_interleaved_batch_buffer_size(
            handle, alg, m, ds, dl, d, du, dw, x, batch_count, batch_stride, &buffer_size);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dgpsv_interleaved_batch_buffer_size(
            handle, alg, m, ds, dl, d, du, dw, x, batch_count, batch_stride, &buffer_size);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_cgpsv_interleaved_batch_buffer_size(
            handle, alg, m, ds, dl, d, du, dw, x, batch_count, batch_stride, &buffer_size);
    else
        st = rocsparse_zgpsv_interleaved_batch_buffer_size(
            handle, alg, m, ds, dl, d, du, dw, x, batch_count, batch_stride, &buffer_size);

    if(st == rocsparse_status_not_implemented)
        return;
    ASSERT_EQ(st, rocsparse_status_success);
    EXPECT_GT(buffer_size, 0u);

    device_vector<char> buffer{buffer_size};
    ASSERT_TRUE(buffer.ptr);

    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_sgpsv_interleaved_batch(
            handle, alg, m, ds, dl, d, du, dw, x, batch_count, batch_stride, buffer.ptr);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dgpsv_interleaved_batch(
            handle, alg, m, ds, dl, d, du, dw, x, batch_count, batch_stride, buffer.ptr);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_cgpsv_interleaved_batch(
            handle, alg, m, ds, dl, d, du, dw, x, batch_count, batch_stride, buffer.ptr);
    else
        st = rocsparse_zgpsv_interleaved_batch(
            handle, alg, m, ds, dl, d, du, dw, x, batch_count, batch_stride, buffer.ptr);
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
}

TEST_F(PrecondGpsv, interleaved_batch_float)
{
    run_gpsv_interleaved_batch_pipeline<float>(handle);
}
TEST_F(PrecondGpsv, interleaved_batch_double)
{
    run_gpsv_interleaved_batch_pipeline<double>(handle);
}
TEST_F(PrecondGpsv, interleaved_batch_float_complex)
{
    run_gpsv_interleaved_batch_pipeline<rocsparse_float_complex>(handle);
}
TEST_F(PrecondGpsv, interleaved_batch_double_complex)
{
    run_gpsv_interleaved_batch_pipeline<rocsparse_double_complex>(handle);
}

TEST_F(PrecondGpsv, bad_args)
{
    const rocsparse_gpsv_interleaved_alg alg = rocsparse_gpsv_interleaved_alg_default;
    const rocsparse_int                  m = 4, batch_count = 1, batch_stride = 1;
    device_vector<float>                 ds{std::vector<float>{0, 0, 1, 1}};
    device_vector<float>                 dl{std::vector<float>{0, 1, 1, 1}};
    device_vector<float>                 d{std::vector<float>{10, 10, 10, 10}};
    device_vector<float>                 du{std::vector<float>{1, 1, 1, 0}};
    device_vector<float>                 dw{std::vector<float>{1, 1, 0, 0}};
    device_vector<float>                 x{std::vector<float>{1, 2, 3, 4}};
    ASSERT_TRUE(ds.ptr && dl.ptr && d.ptr && du.ptr && dw.ptr && x.ptr);

    size_t buffer_size = 0;
    EXPECT_EQ(rocsparse_sgpsv_interleaved_batch_buffer_size(
                  nullptr, alg, m, ds, dl, d, du, dw, x, batch_count, batch_stride, &buffer_size),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(
        rocsparse_sgpsv_interleaved_batch_buffer_size(
            handle, alg, m, nullptr, dl, d, du, dw, x, batch_count, batch_stride, &buffer_size),
        rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_sgpsv_interleaved_batch_buffer_size(
                  handle, alg, -1, ds, dl, d, du, dw, x, batch_count, batch_stride, &buffer_size),
              rocsparse_status_invalid_size);
}

// ======================================================================
// csritilu0 : iterative ILU0. Full buffer_size -> preprocess -> compute_ex
// pipeline on diag(2,3,4) (converges immediately), plus bad-arg guards.
// ======================================================================
class PrecondCsritilu0 : public HandleTest
{
};

template <typename T>
static void run_csritilu0_pipeline(rocsparse_handle handle)
{
    const rocsparse_itilu0_alg alg      = rocsparse_itilu0_alg_default;
    const rocsparse_int        option   = 0;
    const rocsparse_int        m = 3, nnz = 3;
    const rocsparse_index_base base     = rocsparse_index_base_zero;
    rocsparse_int              nmaxiter = 20;

    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<T> val{std::vector<T>{scalar<T>(2), scalar<T>(3), scalar<T>(4)}};
    device_vector<T> ilu0{(size_t)nnz};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr && ilu0.ptr);

    size_t buffer_size = 0;
    rocsparse_status st = rocsparse_csritilu0_buffer_size(
        handle, alg, option, nmaxiter, m, nnz, row_ptr, col_ind, base, dt_of<T>(), &buffer_size);
    if(st == rocsparse_status_not_implemented)
        return;
    ASSERT_EQ(st, rocsparse_status_success);
    EXPECT_GT(buffer_size, 0u);

    device_vector<char> buffer{buffer_size};
    ASSERT_TRUE(buffer.ptr);

    st = rocsparse_csritilu0_preprocess(handle,
                                        alg,
                                        option,
                                        nmaxiter,
                                        m,
                                        nnz,
                                        row_ptr,
                                        col_ind,
                                        base,
                                        dt_of<T>(),
                                        buffer_size,
                                        buffer.ptr);
    if(st == rocsparse_status_not_implemented)
        return;
    ASSERT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_scsritilu0_compute_ex(handle,
                                             alg,
                                             option,
                                             &nmaxiter,
                                             0,
                                             1.0e-6f,
                                             m,
                                             nnz,
                                             row_ptr,
                                             col_ind,
                                             val,
                                             ilu0,
                                             base,
                                             buffer_size,
                                             buffer.ptr);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dcsritilu0_compute_ex(handle,
                                             alg,
                                             option,
                                             &nmaxiter,
                                             0,
                                             1.0e-10,
                                             m,
                                             nnz,
                                             row_ptr,
                                             col_ind,
                                             val,
                                             ilu0,
                                             base,
                                             buffer_size,
                                             buffer.ptr);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_ccsritilu0_compute_ex(handle,
                                             alg,
                                             option,
                                             &nmaxiter,
                                             0,
                                             1.0e-6f,
                                             m,
                                             nnz,
                                             row_ptr,
                                             col_ind,
                                             val,
                                             ilu0,
                                             base,
                                             buffer_size,
                                             buffer.ptr);
    else
        st = rocsparse_zcsritilu0_compute_ex(handle,
                                             alg,
                                             option,
                                             &nmaxiter,
                                             0,
                                             1.0e-10,
                                             m,
                                             nnz,
                                             row_ptr,
                                             col_ind,
                                             val,
                                             ilu0,
                                             base,
                                             buffer_size,
                                             buffer.ptr);
    if(st == rocsparse_status_not_implemented)
        return;
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
}

TEST_F(PrecondCsritilu0, pipeline_float)
{
    run_csritilu0_pipeline<float>(handle);
}
TEST_F(PrecondCsritilu0, pipeline_double)
{
    run_csritilu0_pipeline<double>(handle);
}
TEST_F(PrecondCsritilu0, pipeline_float_complex)
{
    run_csritilu0_pipeline<rocsparse_float_complex>(handle);
}
TEST_F(PrecondCsritilu0, pipeline_double_complex)
{
    run_csritilu0_pipeline<rocsparse_double_complex>(handle);
}

TEST_F(PrecondCsritilu0, bad_args)
{
    const rocsparse_itilu0_alg   alg    = rocsparse_itilu0_alg_default;
    const rocsparse_int          option = 0;
    const rocsparse_int          m = 3, nnz = 3;
    const rocsparse_index_base   base     = rocsparse_index_base_zero;
    const rocsparse_int          nmaxiter = 10;
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr);

    size_t buffer_size = 0;
    EXPECT_EQ(rocsparse_csritilu0_buffer_size(nullptr,
                                              alg,
                                              option,
                                              nmaxiter,
                                              m,
                                              nnz,
                                              row_ptr,
                                              col_ind,
                                              base,
                                              rocsparse_datatype_f32_r,
                                              &buffer_size),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_csritilu0_buffer_size(handle,
                                              alg,
                                              option,
                                              nmaxiter,
                                              m,
                                              nnz,
                                              nullptr,
                                              col_ind,
                                              base,
                                              rocsparse_datatype_f32_r,
                                              &buffer_size),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_csritilu0_buffer_size(handle,
                                              alg,
                                              option,
                                              nmaxiter,
                                              -1,
                                              nnz,
                                              row_ptr,
                                              col_ind,
                                              base,
                                              rocsparse_datatype_f32_r,
                                              &buffer_size),
              rocsparse_status_invalid_size);
    EXPECT_EQ(rocsparse_csritilu0_buffer_size(handle,
                                              alg,
                                              option,
                                              nmaxiter,
                                              m,
                                              nnz,
                                              row_ptr,
                                              col_ind,
                                              (rocsparse_index_base)-1,
                                              rocsparse_datatype_f32_r,
                                              &buffer_size),
              rocsparse_status_invalid_value);
}

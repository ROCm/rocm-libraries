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
    device_vector<T>             val{std::vector<T>{scalar<T>(2), scalar<T>(3), scalar<T>(4)}};
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
        st = rocsparse_scsric0(handle,
                               m,
                               nnz,
                               descr,
                               val,
                               row_ptr,
                               col_ind,
                               info,
                               rocsparse_solve_policy_auto,
                               buffer.ptr);
    else if constexpr(std::is_same_v<T, double>)
        st = rocsparse_dcsric0(handle,
                               m,
                               nnz,
                               descr,
                               val,
                               row_ptr,
                               col_ind,
                               info,
                               rocsparse_solve_policy_auto,
                               buffer.ptr);
    else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
        st = rocsparse_ccsric0(handle,
                               m,
                               nnz,
                               descr,
                               val,
                               row_ptr,
                               col_ind,
                               info,
                               rocsparse_solve_policy_auto,
                               buffer.ptr);
    else
        st = rocsparse_zcsric0(handle,
                               m,
                               nnz,
                               descr,
                               val,
                               row_ptr,
                               col_ind,
                               info,
                               rocsparse_solve_policy_auto,
                               buffer.ptr);
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
    const rocsparse_direction    dir = rocsparse_direction_row;
    const rocsparse_int          mb = 2, nnzb = 2, block_dim = 2;
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1}};
    // Two SPD blocks [[4,1],[1,4]] stored row-major.
    device_vector<T> val{std::vector<T>{scalar<T>(4),
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
    EXPECT_EQ(
        rocsparse_sbsric0_buffer_size(
            nullptr, dir, mb, nnzb, descr, val, row_ptr, col_ind, block_dim, info, &buffer_size),
        rocsparse_status_invalid_handle);
    EXPECT_EQ(
        rocsparse_sbsric0_buffer_size(
            handle, dir, mb, nnzb, nullptr, val, row_ptr, col_ind, block_dim, info, &buffer_size),
        rocsparse_status_invalid_pointer);
    EXPECT_EQ(
        rocsparse_sbsric0_buffer_size(
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
    const rocsparse_direction    dir = rocsparse_direction_row;
    const rocsparse_int          mb = 2, nnzb = 2, block_dim = 2;
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

    EXPECT_EQ(rocsparse_sgtsv_no_pivot_buffer_size(nullptr, m, n, dl, d, du, B, ldb, &buffer_size),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(
        rocsparse_sgtsv_no_pivot_buffer_size(handle, m, n, nullptr, d, du, B, ldb, &buffer_size),
        rocsparse_status_invalid_pointer);

    EXPECT_EQ(rocsparse_sgtsv_no_pivot_strided_batch_buffer_size(
                  nullptr, m, dl, d, du, B, 1, m, &buffer_size),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(
        rocsparse_sgtsv_interleaved_batch_buffer_size(
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
// gtsv : algorithm-selection coverage. The buffer_size/solve dispatch in
// rocsparse_gtsv.cpp selects an internal block_dim / gridsize (spike solver
// template instantiation) from the system size m, and a right-hand-side
// kernel variant from n (n%8, n%4, n%2, else). Drive a range of m and n on
// tiny diagonally-dominant tridiagonal systems (d=4, dl=du=1) so the solve
// stays well-conditioned across every branch.
// ======================================================================
template <typename T>
static void
    run_gtsv_solve(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, rocsparse_int ldb)
{
    std::vector<T> hdl((size_t)m), hd((size_t)m), hdu((size_t)m);
    for(rocsparse_int i = 0; i < m; ++i)
    {
        hdl[i] = scalar<T>(i == 0 ? 0.0f : 1.0f);
        hd[i]  = scalar<T>(4.0f);
        hdu[i] = scalar<T>(i == m - 1 ? 0.0f : 1.0f);
    }
    std::vector<T>   hB((size_t)ldb * (size_t)n, scalar<T>(1.0f));
    device_vector<T> dl{hdl}, d{hd}, du{hdu}, B{hB};
    ASSERT_TRUE(dl.ptr && d.ptr && du.ptr && B.ptr);

    size_t           buffer_size = 0;
    rocsparse_status st;
    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_sgtsv_buffer_size(handle, m, n, dl, d, du, B, ldb, &buffer_size);
    else
        st = rocsparse_dgtsv_buffer_size(handle, m, n, dl, d, du, B, ldb, &buffer_size);

    if(st == rocsparse_status_not_implemented)
        return;
    ASSERT_EQ(st, rocsparse_status_success);
    EXPECT_GT(buffer_size, 0u);

    device_vector<char> buffer{buffer_size};
    ASSERT_TRUE(buffer.ptr);

    if constexpr(std::is_same_v<T, float>)
        st = rocsparse_sgtsv(handle, m, n, dl, d, du, B, ldb, buffer.ptr);
    else
        st = rocsparse_dgtsv(handle, m, n, dl, d, du, B, ldb, buffer.ptr);
    EXPECT_EQ(st, rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
}

// m >= 513 forces the padded grid to gridsize == 2 (spike grid-level<2>).
TEST_F(PrecondGtsv, alg_gridsize2_float)
{
    run_gtsv_solve<float>(handle, 513, 1, 513);
}
TEST_F(PrecondGtsv, alg_gridsize2_double)
{
    run_gtsv_solve<double>(handle, 513, 1, 513);
}
// m >= 1537 forces gridsize == 4 (spike grid-level<4>).
TEST_F(PrecondGtsv, alg_gridsize4_double)
{
    run_gtsv_solve<double>(handle, 1600, 1, 1600);
}
// Right-hand-side kernel variants selected by n % 8 / 4 / 2 / else.
TEST_F(PrecondGtsv, alg_rhs_n8)
{
    run_gtsv_solve<float>(handle, 16, 8, 16);
}
TEST_F(PrecondGtsv, alg_rhs_n4)
{
    run_gtsv_solve<float>(handle, 16, 4, 16);
}
TEST_F(PrecondGtsv, alg_rhs_n2)
{
    run_gtsv_solve<double>(handle, 16, 2, 16);
}
TEST_F(PrecondGtsv, alg_rhs_n3)
{
    run_gtsv_solve<double>(handle, 16, 3, 16);
}
// A larger ldb than m exercises the strided right-hand-side copy path.
TEST_F(PrecondGtsv, alg_large_ldb)
{
    run_gtsv_solve<float>(handle, 32, 2, 40);
}

// n == 0 quick-return branch (buffer_size==0, solve is a no-op success).
TEST_F(PrecondGtsv, quick_return_n0)
{
    const rocsparse_int  m = 4, n = 0, ldb = 4;
    device_vector<float> dl{std::vector<float>{0, 1, 1, 1}};
    device_vector<float> d{std::vector<float>{4, 4, 4, 4}};
    device_vector<float> du{std::vector<float>{1, 1, 1, 0}};
    device_vector<float> B{std::vector<float>{1, 2, 3, 4}};
    ASSERT_TRUE(dl.ptr && d.ptr && du.ptr && B.ptr);

    size_t buffer_size = 1;
    ASSERT_EQ(rocsparse_sgtsv_buffer_size(handle, m, n, dl, d, du, B, ldb, &buffer_size),
              rocsparse_status_success);
    EXPECT_EQ(buffer_size, 0u);
    EXPECT_EQ(rocsparse_sgtsv(handle, m, n, dl, d, du, B, ldb, nullptr), rocsparse_status_success);
}

// Additional size/pointer guards not covered by PrecondGtsv.bad_args.
TEST_F(PrecondGtsv, more_bad_args)
{
    const rocsparse_int  m = 4, n = 1, ldb = 4;
    device_vector<float> dl{std::vector<float>{0, 1, 1, 1}};
    device_vector<float> d{std::vector<float>{4, 4, 4, 4}};
    device_vector<float> du{std::vector<float>{1, 1, 1, 0}};
    device_vector<float> B{std::vector<float>{1, 2, 3, 4}};
    ASSERT_TRUE(dl.ptr && d.ptr && du.ptr && B.ptr);

    size_t buffer_size = 0;
    // m <= 1 is rejected as an invalid size by gtsv.
    EXPECT_EQ(rocsparse_sgtsv_buffer_size(handle, 1, n, dl, d, du, B, ldb, &buffer_size),
              rocsparse_status_invalid_size);
    // ldb < max(1, m) is an invalid size.
    EXPECT_EQ(rocsparse_sgtsv_buffer_size(handle, m, n, dl, d, du, B, 2, &buffer_size),
              rocsparse_status_invalid_size);
    // null buffer_size pointer.
    EXPECT_EQ(rocsparse_sgtsv_buffer_size(handle, m, n, dl, d, du, B, ldb, nullptr),
              rocsparse_status_invalid_pointer);

    // Solve-side guards.
    device_vector<char> buffer{size_t(1)};
    ASSERT_TRUE(buffer.ptr);
    EXPECT_EQ(rocsparse_sgtsv(nullptr, m, n, dl, d, du, B, ldb, buffer.ptr),
              rocsparse_status_invalid_handle);
    EXPECT_EQ(rocsparse_sgtsv(handle, 1, n, dl, d, du, B, ldb, buffer.ptr),
              rocsparse_status_invalid_size);
    EXPECT_EQ(rocsparse_sgtsv(handle, m, n, dl, d, du, B, 2, buffer.ptr),
              rocsparse_status_invalid_size);
    EXPECT_EQ(rocsparse_sgtsv(handle, m, n, nullptr, d, du, B, ldb, buffer.ptr),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse_sgtsv(handle, m, n, dl, d, du, B, ldb, nullptr),
              rocsparse_status_invalid_pointer);
}

// ======================================================================
// csrilu0 : additional analysis-policy, numeric-boost, descriptor and
// kernel-launch coverage on top of Precond.csrilu0_full_pipeline.
//   * analysis run with force then reuse (both the build and the reuse
//     branch of the shared csrsv analysis).
//   * numeric boost enabled (boost path in the compute kernel launch).
//   * not_implemented / requires_sorted_storage descriptor guards.
//   * a matrix with a wide row (>= 512 nnz) selects the binary-search
//     kernel launch instead of the hash launch.
// ======================================================================
class PrecondCsrilu0 : public HandleTest
{
};

TEST_F(PrecondCsrilu0, analysis_force_then_reuse)
{
    const rocsparse_int          m = 3, nnz = 3;
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
    device_vector<char> buffer{buffer_size};
    ASSERT_TRUE(buffer.ptr);

    // First analysis with the force policy builds the meta data.
    ASSERT_EQ(rocsparse_scsrilu0_analysis(handle,
                                          m,
                                          nnz,
                                          descr,
                                          val,
                                          row_ptr,
                                          col_ind,
                                          info,
                                          rocsparse_analysis_policy_force,
                                          rocsparse_solve_policy_auto,
                                          buffer.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    // Second analysis with reuse hits the "already analyzed" branch.
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

    rocsparse_int position = -2;
    EXPECT_EQ(rocsparse_csrilu0_zero_pivot(handle, info, &position), rocsparse_status_success);
    EXPECT_EQ(position, -1);

    EXPECT_EQ(rocsparse_csrilu0_clear(handle, info), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

TEST_F(PrecondCsrilu0, numeric_boost_enabled)
{
    const rocsparse_int          m = 3, nnz = 3;
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<double>        val{std::vector<double>{2, 3, 4}};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    rocsparse_mat_info info = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

    // Enable numeric boost (host pointers). A well-conditioned diagonal never
    // triggers the actual replacement, but the boost-enabled code path runs.
    const double boost_tol = 1.0e-12, boost_val = 1.0;
    EXPECT_EQ(rocsparse_dcsrilu0_numeric_boost(handle, info, 1, &boost_tol, &boost_val),
              rocsparse_status_success);

    size_t buffer_size = 0;
    ASSERT_EQ(rocsparse_dcsrilu0_buffer_size(
                  handle, m, nnz, descr, val, row_ptr, col_ind, info, &buffer_size),
              rocsparse_status_success);
    device_vector<char> buffer{buffer_size};
    ASSERT_TRUE(buffer.ptr);

    ASSERT_EQ(rocsparse_dcsrilu0_analysis(handle,
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

    EXPECT_EQ(rocsparse_dcsrilu0(handle,
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
    EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

TEST_F(PrecondCsrilu0, descr_not_implemented_and_unsorted)
{
    const rocsparse_int          m = 3, nnz = 3;
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<float>         val{std::vector<float>{2, 3, 4}};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    rocsparse_mat_info info = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

    size_t buffer_size = 0;

    // Non-general matrix type -> not_implemented.
    ASSERT_EQ(rocsparse_set_mat_type(descr, rocsparse_matrix_type_symmetric),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_scsrilu0_buffer_size(
                  handle, m, nnz, descr, val, row_ptr, col_ind, info, &buffer_size),
              rocsparse_status_not_implemented);
    ASSERT_EQ(rocsparse_set_mat_type(descr, rocsparse_matrix_type_general),
              rocsparse_status_success);

    // Unsorted storage -> requires_sorted_storage.
    ASSERT_EQ(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_unsorted),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_scsrilu0_buffer_size(
                  handle, m, nnz, descr, val, row_ptr, col_ind, info, &buffer_size),
              rocsparse_status_requires_sorted_storage);

    EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

// Wide first row (>= 512 nnz) makes max_nnz-per-row >= 512, selecting the
// binary-search kernel launch (rocsparse_csrilu0_kernel_launch.cpp) instead of
// the hash launch that the tiny diagonal pipelines exercise. The matrix is
// upper triangular with a strongly dominant diagonal, so ILU0 has no fill and
// no zero pivot.
TEST_F(PrecondCsrilu0, wide_row_binsearch_launch)
{
    const rocsparse_int wide = 512;
    const rocsparse_int m    = wide;
    const rocsparse_int nnz  = wide + (m - 1); // row 0 dense + diagonal of the rest

    std::vector<rocsparse_int> hptr(m + 1, 0);
    std::vector<rocsparse_int> hcol;
    std::vector<float>         hval;
    hcol.reserve(nnz);
    hval.reserve(nnz);
    // Row 0: columns 0..wide-1.
    for(rocsparse_int j = 0; j < wide; ++j)
    {
        hcol.push_back(j);
        hval.push_back(j == 0 ? 1000.0f : 1.0f);
    }
    hptr[1] = wide;
    // Rows 1..m-1: single diagonal entry.
    for(rocsparse_int i = 1; i < m; ++i)
    {
        hcol.push_back(i);
        hval.push_back(4.0f);
        hptr[i + 1] = hptr[i] + 1;
    }
    ASSERT_EQ((rocsparse_int)hcol.size(), nnz);

    device_vector<rocsparse_int> row_ptr{hptr};
    device_vector<rocsparse_int> col_ind{hcol};
    device_vector<float>         val{hval};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    rocsparse_mat_info info = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

    size_t buffer_size = 0;
    ASSERT_EQ(rocsparse_scsrilu0_buffer_size(
                  handle, m, nnz, descr, val, row_ptr, col_ind, info, &buffer_size),
              rocsparse_status_success);
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
                                          rocsparse_analysis_policy_force,
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

    rocsparse_int position = -2;
    EXPECT_EQ(rocsparse_csrilu0_zero_pivot(handle, info, &position), rocsparse_status_success);
    EXPECT_EQ(position, -1);

    EXPECT_EQ(rocsparse_csrilu0_clear(handle, info), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

// ======================================================================
// csric0 analysis coverage (rocsparse_xcsric0_analysis.cpp):
//   * force then reuse analysis policy.
//   * descriptor / enum / pointer guards (not_implemented,
//     requires_sorted_storage, invalid analysis/solve enum, null info/buffer).
// ======================================================================
TEST_F(PrecondCsric0, analysis_force_then_reuse)
{
    const rocsparse_int          m = 3, nnz = 3;
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<double>        val{std::vector<double>{2, 3, 4}};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    rocsparse_mat_info info = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

    size_t buffer_size = 0;
    ASSERT_EQ(rocsparse_dcsric0_buffer_size(
                  handle, m, nnz, descr, val, row_ptr, col_ind, info, &buffer_size),
              rocsparse_status_success);
    device_vector<char> buffer{buffer_size};
    ASSERT_TRUE(buffer.ptr);

    for(auto policy : {rocsparse_analysis_policy_force, rocsparse_analysis_policy_reuse})
    {
        ASSERT_EQ(rocsparse_dcsric0_analysis(handle,
                                             m,
                                             nnz,
                                             descr,
                                             val,
                                             row_ptr,
                                             col_ind,
                                             info,
                                             policy,
                                             rocsparse_solve_policy_auto,
                                             buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    }

    EXPECT_EQ(rocsparse_dcsric0(handle,
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

    EXPECT_EQ(rocsparse_csric0_clear(handle, info), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

TEST_F(PrecondCsric0, analysis_bad_args)
{
    const rocsparse_int          m = 3, nnz = 3;
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<float>         val{std::vector<float>{2, 3, 4}};
    device_vector<char>          buffer{size_t(256)};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr && buffer.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    rocsparse_mat_info info = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

    // Invalid handle.
    EXPECT_EQ(rocsparse_scsric0_analysis(nullptr,
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
              rocsparse_status_invalid_handle);

    // Non-general matrix type -> not_implemented.
    ASSERT_EQ(rocsparse_set_mat_type(descr, rocsparse_matrix_type_triangular),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_scsric0_analysis(handle,
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
              rocsparse_status_not_implemented);
    ASSERT_EQ(rocsparse_set_mat_type(descr, rocsparse_matrix_type_general),
              rocsparse_status_success);

    // Unsorted storage -> requires_sorted_storage.
    ASSERT_EQ(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_unsorted),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_scsric0_analysis(handle,
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
              rocsparse_status_requires_sorted_storage);
    ASSERT_EQ(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_sorted),
              rocsparse_status_success);

    // Invalid analysis-policy enum.
    EXPECT_EQ(rocsparse_scsric0_analysis(handle,
                                         m,
                                         nnz,
                                         descr,
                                         val,
                                         row_ptr,
                                         col_ind,
                                         info,
                                         (rocsparse_analysis_policy)-1,
                                         rocsparse_solve_policy_auto,
                                         buffer.ptr),
              rocsparse_status_invalid_value);

    // Invalid solve-policy enum.
    EXPECT_EQ(rocsparse_scsric0_analysis(handle,
                                         m,
                                         nnz,
                                         descr,
                                         val,
                                         row_ptr,
                                         col_ind,
                                         info,
                                         rocsparse_analysis_policy_reuse,
                                         (rocsparse_solve_policy)-1,
                                         buffer.ptr),
              rocsparse_status_invalid_value);

    // Null info pointer.
    EXPECT_EQ(rocsparse_scsric0_analysis(handle,
                                         m,
                                         nnz,
                                         descr,
                                         val,
                                         row_ptr,
                                         col_ind,
                                         nullptr,
                                         rocsparse_analysis_policy_reuse,
                                         rocsparse_solve_policy_auto,
                                         buffer.ptr),
              rocsparse_status_invalid_pointer);

    // Null temp buffer.
    EXPECT_EQ(rocsparse_scsric0_analysis(handle,
                                         m,
                                         nnz,
                                         descr,
                                         val,
                                         row_ptr,
                                         col_ind,
                                         info,
                                         rocsparse_analysis_policy_reuse,
                                         rocsparse_solve_policy_auto,
                                         nullptr),
              rocsparse_status_invalid_pointer);

    EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

// ======================================================================
// bsrilu0 : numeric-boost-enabled compute pipeline. The tiny diagonal
// pipelines only exercise numeric boost disabled, so enable it here to run
// the boost branch of the general kernel launch. block_dim=2, SPD blocks.
// ======================================================================
TEST_F(PrecondBsrilu0, numeric_boost_enabled_pipeline)
{
    const rocsparse_direction    dir = rocsparse_direction_row;
    const rocsparse_int          mb = 2, nnzb = 2, block_dim = 2;
    device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2}};
    device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1}};
    device_vector<double>        val{std::vector<double>{4, 1, 1, 4, 4, 1, 1, 4}};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    rocsparse_mat_info info = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

    const double boost_tol = 1.0e-12, boost_val = 1.0;
    EXPECT_EQ(rocsparse_dbsrilu0_numeric_boost(handle, info, 1, &boost_tol, &boost_val),
              rocsparse_status_success);

    size_t buffer_size = 0;
    ASSERT_EQ(
        rocsparse_dbsrilu0_buffer_size(
            handle, dir, mb, nnzb, descr, val, row_ptr, col_ind, block_dim, info, &buffer_size),
        rocsparse_status_success);
    device_vector<char> buffer{buffer_size};
    ASSERT_TRUE(buffer.ptr);

    ASSERT_EQ(rocsparse_dbsrilu0_analysis(handle,
                                          dir,
                                          mb,
                                          nnzb,
                                          descr,
                                          val,
                                          row_ptr,
                                          col_ind,
                                          block_dim,
                                          info,
                                          rocsparse_analysis_policy_force,
                                          rocsparse_solve_policy_auto,
                                          buffer.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    EXPECT_EQ(rocsparse_dbsrilu0(handle,
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
                                 buffer.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    rocsparse_int position = -2;
    EXPECT_EQ(rocsparse_bsrilu0_zero_pivot(handle, info, &position), rocsparse_status_success);
    EXPECT_EQ(position, -1);

    EXPECT_EQ(rocsparse_bsrilu0_clear(handle, info), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

// ======================================================================
// gtsv second wave: cover the internal spike-solver template block_dim
// selection and the moderate-grid grid-level kernel branches in
// rocsparse_gtsv.cpp.
//
// The solve dispatch (gtsv_template) starts at block_dim = 2 and doubles it
// while the padded gridsize exceeds 512 (BLOCKSIZE = 256, so the doubling
// stops once block_dim * 131072 >= m). The resulting block_dim selects a
// distinct gtsv_spike_solver_template<BLOCKSIZE, BD> instantiation:
//   block_dim ==  16 : m in (1048576, 2097152]
//   block_dim ==  32 : m in (2097152, 4194304]
//   block_dim ==  64 : m in (4194304, 8388608]
//   block_dim == 128 : m in (8388608, 16777216]
//   block_dim == 256 : m in (16777216, 33554432]
//   block_dim  > 256 : m > 33554432 -> rocsparse_status_not_implemented
// Each solve uses a well-conditioned diagonally-dominant tridiagonal system
// (d = 4, dl = du = 1) with two right-hand sides so the multi-RHS path in
// the selected instantiation runs too.
//
// Moderate m additionally selects the grid-level spike kernel by the padded
// gridsize rounded up to a power of two: gridsize == 8 for m in (2048, 4096]
// and gridsize == 16 for m in (4096, 8192].
// ======================================================================
TEST_F(PrecondGtsv, grid_level_gridsize8)
{
    run_gtsv_solve<float>(handle, 3000, 2, 3000);
}
TEST_F(PrecondGtsv, grid_level_gridsize16)
{
    run_gtsv_solve<float>(handle, 8000, 2, 8000);
}
TEST_F(PrecondGtsv, spike_block_dim16)
{
    run_gtsv_solve<float>(handle, 1500000, 2, 1500000);
}
TEST_F(PrecondGtsv, spike_block_dim32)
{
    run_gtsv_solve<float>(handle, 3000000, 2, 3000000);
}
TEST_F(PrecondGtsv, spike_block_dim64)
{
    run_gtsv_solve<float>(handle, 6000000, 2, 6000000);
}
TEST_F(PrecondGtsv, spike_block_dim128)
{
    run_gtsv_solve<float>(handle, 12000000, 2, 12000000);
}
TEST_F(PrecondGtsv, spike_block_dim256)
{
    run_gtsv_solve<float>(handle, 25000000, 2, 25000000);
}
// m > 256 * 131072 forces block_dim past 256, which the solve dispatch
// rejects with not_implemented. Only the dispatch runs (it returns before
// touching any array), so dummy single-element inputs are sufficient.
TEST_F(PrecondGtsv, spike_block_dim_not_implemented)
{
    const rocsparse_int  m = 34000000, n = 1, ldb = 34000000;
    device_vector<float> dl{std::vector<float>{1}};
    device_vector<float> d{std::vector<float>{4}};
    device_vector<float> du{std::vector<float>{1}};
    device_vector<float> B{std::vector<float>{1}};
    device_vector<char>  buffer{size_t(1)};
    ASSERT_TRUE(dl.ptr && d.ptr && du.ptr && B.ptr && buffer.ptr);
    EXPECT_EQ(rocsparse_sgtsv(handle, m, n, dl, d, du, B, ldb, buffer.ptr),
              rocsparse_status_not_implemented);
}

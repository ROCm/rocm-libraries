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
// Host-path unit tests for the internal *_info.hpp struct helpers that live in
// library/src/include. These are LIVE library methods (deep-copy helpers,
// singularity accessors, csrmv/bsrmv clear/destroy paths, trm_data_t
// create/recreate/storage_index) that were never exercised by the existing
// test corpus. Every method here is reached purely through the public C API:
//
//   * The per-info ::copy() deep-copy methods run only through
//     rocsparse_copy_mat_info(dest, src), which walks whichever sub-infos were
//     populated in src (see rocsparse::copy_mat_info in
//     library/src/auxiliary/rocsparse_auxiliary.cpp). So each test populates a
//     sub-info via the matching X_analysis / compute routine, then copies the
//     mat_info into a freshly created dest.
//   * The singularity accessors + trm_data_t create/recreate run inside the
//     X_analysis / X_solve / compute routines.
//   * csrmv_info::clear()/dtor + copy_csrmv_info and the bsrmv_info
//     ctor/dtor/get/set run through the ADAPTIVE csrmv/bsrmv analysis path and
//     the mat_info copy/destroy.
//
// All matrices are tiny, well-conditioned diagonal systems (SPD, diagonally
// dominant, trivially triangular) so no zero pivot / breakdown occurs.
//
#include "unit_test_utils.hpp"

using namespace rocsparse_ut;

namespace
{
    // diag(2,3,4) in CSR / BSR(block_dim=1). Triangular, SPD, diag-dominant.
    const std::vector<rocsparse_int> g_row_ptr{0, 1, 2, 3};
    const std::vector<rocsparse_int> g_col_ind{0, 1, 2};
    const std::vector<float>         g_val{2.0f, 3.0f, 4.0f};
    constexpr rocsparse_int          g_m   = 3;
    constexpr rocsparse_int          g_nnz = 3;
}

class Info : public HandleTest
{
};

// ======================================================================
// csrsv : buffer_size -> analysis -> solve -> zero_pivot -> copy_mat_info.
//
// Covers _rocsparse_csrsv_info::copy (via rocsparse_copy_mat_info),
// get_singularity_numeric_exact (via zero_pivot),
// create_singularity_numeric_exact (via solve), the pivot_info_t::
// create_zero_pivot_async(indextype,stream) inline overload (via analysis),
// and rocsparse::trm_data_t storage_index/create/recreate (via analysis).
// ======================================================================
TEST_F(Info, csrsv_copy_and_singularity)
{
    device_vector<rocsparse_int> row_ptr{g_row_ptr};
    device_vector<rocsparse_int> col_ind{g_col_ind};
    device_vector<float>         val{g_val};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    rocsparse_mat_info src = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&src), rocsparse_status_success);

    const rocsparse_operation trans = rocsparse_operation_none;

    size_t buffer_size = 0;
    ASSERT_EQ(rocsparse_scsrsv_buffer_size(
                  handle, trans, g_m, g_nnz, descr, val, row_ptr, col_ind, src, &buffer_size),
              rocsparse_status_success);
    device_vector<char> buffer{buffer_size};
    ASSERT_TRUE(buffer.ptr);

    ASSERT_EQ(rocsparse_scsrsv_analysis(handle,
                                        trans,
                                        g_m,
                                        g_nnz,
                                        descr,
                                        val,
                                        row_ptr,
                                        col_ind,
                                        src,
                                        rocsparse_analysis_policy_reuse,
                                        rocsparse_solve_policy_auto,
                                        buffer.ptr),
              rocsparse_status_success);

    const float          alpha = 1.0f;
    device_vector<float> x{std::vector<float>{1.0f, 1.0f, 1.0f}};
    device_vector<float> y{(size_t)g_m};
    ASSERT_TRUE(x.ptr && y.ptr);

    EXPECT_EQ(rocsparse_scsrsv_solve(handle,
                                     trans,
                                     g_m,
                                     g_nnz,
                                     &alpha,
                                     descr,
                                     val,
                                     row_ptr,
                                     col_ind,
                                     src,
                                     x,
                                     y,
                                     rocsparse_solve_policy_auto,
                                     buffer.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // Well-conditioned system: no zero pivot expected.
    rocsparse_int position = 0;
    EXPECT_EQ(rocsparse_csrsv_zero_pivot(handle, descr, src, &position), rocsparse_status_success);

    // Deep-copy the populated csrsv sub-info into a fresh dest.
    rocsparse_mat_info dest = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&dest), rocsparse_status_success);
    EXPECT_EQ(rocsparse_copy_mat_info(dest, src), rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    EXPECT_EQ(rocsparse_destroy_mat_info(dest), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_info(src), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

// ======================================================================
// csrilu0 : buffer_size -> analysis -> compute -> copy_mat_info.
//
// Covers _rocsparse_csrilu0_info::copy plus its exact/near singularity
// get/create accessors (create runs in rocsparse_scsrilu0).
// ======================================================================
TEST_F(Info, csrilu0_copy_and_singularity)
{
    device_vector<rocsparse_int> row_ptr{g_row_ptr};
    device_vector<rocsparse_int> col_ind{g_col_ind};
    device_vector<float>         val{g_val};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    rocsparse_mat_info src = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&src), rocsparse_status_success);

    size_t buffer_size = 0;
    ASSERT_EQ(rocsparse_scsrilu0_buffer_size(
                  handle, g_m, g_nnz, descr, val, row_ptr, col_ind, src, &buffer_size),
              rocsparse_status_success);
    device_vector<char> buffer{buffer_size};
    ASSERT_TRUE(buffer.ptr);

    ASSERT_EQ(rocsparse_scsrilu0_analysis(handle,
                                          g_m,
                                          g_nnz,
                                          descr,
                                          val,
                                          row_ptr,
                                          col_ind,
                                          src,
                                          rocsparse_analysis_policy_reuse,
                                          rocsparse_solve_policy_auto,
                                          buffer.ptr),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_scsrilu0(handle,
                                 g_m,
                                 g_nnz,
                                 descr,
                                 val,
                                 row_ptr,
                                 col_ind,
                                 src,
                                 rocsparse_solve_policy_auto,
                                 buffer.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    rocsparse_mat_info dest = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&dest), rocsparse_status_success);
    EXPECT_EQ(rocsparse_copy_mat_info(dest, src), rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    EXPECT_EQ(rocsparse_destroy_mat_info(dest), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_info(src), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

// ======================================================================
// csric0 : buffer_size -> analysis -> compute -> copy_mat_info.
//
// Covers _rocsparse_csric0_info::copy plus exact/near singularity accessors.
// ======================================================================
TEST_F(Info, csric0_copy_and_singularity)
{
    device_vector<rocsparse_int> row_ptr{g_row_ptr};
    device_vector<rocsparse_int> col_ind{g_col_ind};
    device_vector<float>         val{g_val};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    rocsparse_mat_info src = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&src), rocsparse_status_success);

    size_t buffer_size = 0;
    ASSERT_EQ(rocsparse_scsric0_buffer_size(
                  handle, g_m, g_nnz, descr, val, row_ptr, col_ind, src, &buffer_size),
              rocsparse_status_success);
    device_vector<char> buffer{buffer_size};
    ASSERT_TRUE(buffer.ptr);

    ASSERT_EQ(rocsparse_scsric0_analysis(handle,
                                         g_m,
                                         g_nnz,
                                         descr,
                                         val,
                                         row_ptr,
                                         col_ind,
                                         src,
                                         rocsparse_analysis_policy_reuse,
                                         rocsparse_solve_policy_auto,
                                         buffer.ptr),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_scsric0(handle,
                                g_m,
                                g_nnz,
                                descr,
                                val,
                                row_ptr,
                                col_ind,
                                src,
                                rocsparse_solve_policy_auto,
                                buffer.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    rocsparse_mat_info dest = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&dest), rocsparse_status_success);
    EXPECT_EQ(rocsparse_copy_mat_info(dest, src), rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    EXPECT_EQ(rocsparse_destroy_mat_info(dest), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_info(src), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

// ======================================================================
// bsrsv (block_dim = 1) : buffer_size -> analysis -> copy_mat_info.
//
// Covers _rocsparse_bsrsv_info::copy (trm_data_t::copy + singularity copy)
// via rocsparse_copy_mat_info.
// ======================================================================
TEST_F(Info, bsrsv_copy)
{
    device_vector<rocsparse_int> row_ptr{g_row_ptr};
    device_vector<rocsparse_int> col_ind{g_col_ind};
    device_vector<float>         val{g_val};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    rocsparse_mat_info src = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&src), rocsparse_status_success);

    const rocsparse_direction dir       = rocsparse_direction_row;
    const rocsparse_operation trans     = rocsparse_operation_none;
    const rocsparse_int       block_dim = 1;

    size_t buffer_size = 0;
    ASSERT_EQ(rocsparse_sbsrsv_buffer_size(handle,
                                           dir,
                                           trans,
                                           g_m,
                                           g_nnz,
                                           descr,
                                           val,
                                           row_ptr,
                                           col_ind,
                                           block_dim,
                                           src,
                                           &buffer_size),
              rocsparse_status_success);
    device_vector<char> buffer{buffer_size};
    ASSERT_TRUE(buffer.ptr);

    ASSERT_EQ(rocsparse_sbsrsv_analysis(handle,
                                        dir,
                                        trans,
                                        g_m,
                                        g_nnz,
                                        descr,
                                        val,
                                        row_ptr,
                                        col_ind,
                                        block_dim,
                                        src,
                                        rocsparse_analysis_policy_reuse,
                                        rocsparse_solve_policy_auto,
                                        buffer.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    rocsparse_mat_info dest = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&dest), rocsparse_status_success);
    EXPECT_EQ(rocsparse_copy_mat_info(dest, src), rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    EXPECT_EQ(rocsparse_destroy_mat_info(dest), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_info(src), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

// ======================================================================
// bsrsm (block_dim = 1) : buffer_size -> analysis -> copy_mat_info.
//
// Covers _rocsparse_bsrsm_info::copy via rocsparse_copy_mat_info.
// ======================================================================
TEST_F(Info, bsrsm_copy)
{
    device_vector<rocsparse_int> row_ptr{g_row_ptr};
    device_vector<rocsparse_int> col_ind{g_col_ind};
    device_vector<float>         val{g_val};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    rocsparse_mat_info src = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&src), rocsparse_status_success);

    const rocsparse_direction dir       = rocsparse_direction_row;
    const rocsparse_operation trans_A   = rocsparse_operation_none;
    const rocsparse_operation trans_X   = rocsparse_operation_none;
    const rocsparse_int       block_dim = 1;
    const rocsparse_int       nrhs      = 1;

    size_t buffer_size = 0;
    ASSERT_EQ(rocsparse_sbsrsm_buffer_size(handle,
                                           dir,
                                           trans_A,
                                           trans_X,
                                           g_m,
                                           nrhs,
                                           g_nnz,
                                           descr,
                                           val,
                                           row_ptr,
                                           col_ind,
                                           block_dim,
                                           src,
                                           &buffer_size),
              rocsparse_status_success);
    device_vector<char> buffer{buffer_size};
    ASSERT_TRUE(buffer.ptr);

    ASSERT_EQ(rocsparse_sbsrsm_analysis(handle,
                                        dir,
                                        trans_A,
                                        trans_X,
                                        g_m,
                                        nrhs,
                                        g_nnz,
                                        descr,
                                        val,
                                        row_ptr,
                                        col_ind,
                                        block_dim,
                                        src,
                                        rocsparse_analysis_policy_reuse,
                                        rocsparse_solve_policy_auto,
                                        buffer.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    rocsparse_mat_info dest = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&dest), rocsparse_status_success);
    EXPECT_EQ(rocsparse_copy_mat_info(dest, src), rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    EXPECT_EQ(rocsparse_destroy_mat_info(dest), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_info(src), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

// ======================================================================
// bsrilu0 (block_dim = 1) : buffer_size -> analysis -> compute -> copy.
//
// Covers _rocsparse_bsrilu0_info::copy plus its singularity get/create
// (create runs in rocsparse_sbsrilu0).
// ======================================================================
TEST_F(Info, bsrilu0_copy_and_singularity)
{
    device_vector<rocsparse_int> row_ptr{g_row_ptr};
    device_vector<rocsparse_int> col_ind{g_col_ind};
    device_vector<float>         val{g_val};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    rocsparse_mat_info src = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&src), rocsparse_status_success);

    const rocsparse_direction dir       = rocsparse_direction_row;
    const rocsparse_int       block_dim = 1;

    size_t buffer_size = 0;
    ASSERT_EQ(
        rocsparse_sbsrilu0_buffer_size(
            handle, dir, g_m, g_nnz, descr, val, row_ptr, col_ind, block_dim, src, &buffer_size),
        rocsparse_status_success);
    device_vector<char> buffer{buffer_size};
    ASSERT_TRUE(buffer.ptr);

    ASSERT_EQ(rocsparse_sbsrilu0_analysis(handle,
                                          dir,
                                          g_m,
                                          g_nnz,
                                          descr,
                                          val,
                                          row_ptr,
                                          col_ind,
                                          block_dim,
                                          src,
                                          rocsparse_analysis_policy_reuse,
                                          rocsparse_solve_policy_auto,
                                          buffer.ptr),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_sbsrilu0(handle,
                                 dir,
                                 g_m,
                                 g_nnz,
                                 descr,
                                 val,
                                 row_ptr,
                                 col_ind,
                                 block_dim,
                                 src,
                                 rocsparse_solve_policy_auto,
                                 buffer.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    rocsparse_mat_info dest = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&dest), rocsparse_status_success);
    EXPECT_EQ(rocsparse_copy_mat_info(dest, src), rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    EXPECT_EQ(rocsparse_destroy_mat_info(dest), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_info(src), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

// ======================================================================
// bsric0 (block_dim = 1) : buffer_size -> analysis -> compute -> copy.
//
// Covers _rocsparse_bsric0_info::copy plus its singularity get/create.
// ======================================================================
TEST_F(Info, bsric0_copy_and_singularity)
{
    device_vector<rocsparse_int> row_ptr{g_row_ptr};
    device_vector<rocsparse_int> col_ind{g_col_ind};
    device_vector<float>         val{g_val};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    rocsparse_mat_info src = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&src), rocsparse_status_success);

    const rocsparse_direction dir       = rocsparse_direction_row;
    const rocsparse_int       block_dim = 1;

    size_t buffer_size = 0;
    ASSERT_EQ(
        rocsparse_sbsric0_buffer_size(
            handle, dir, g_m, g_nnz, descr, val, row_ptr, col_ind, block_dim, src, &buffer_size),
        rocsparse_status_success);
    device_vector<char> buffer{buffer_size};
    ASSERT_TRUE(buffer.ptr);

    ASSERT_EQ(rocsparse_sbsric0_analysis(handle,
                                         dir,
                                         g_m,
                                         g_nnz,
                                         descr,
                                         val,
                                         row_ptr,
                                         col_ind,
                                         block_dim,
                                         src,
                                         rocsparse_analysis_policy_reuse,
                                         rocsparse_solve_policy_auto,
                                         buffer.ptr),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse_sbsric0(handle,
                                dir,
                                g_m,
                                g_nnz,
                                descr,
                                val,
                                row_ptr,
                                col_ind,
                                block_dim,
                                src,
                                rocsparse_solve_policy_auto,
                                buffer.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    rocsparse_mat_info dest = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&dest), rocsparse_status_success);
    EXPECT_EQ(rocsparse_copy_mat_info(dest, src), rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    EXPECT_EQ(rocsparse_destroy_mat_info(dest), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_info(src), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

// ======================================================================
// csrmv adaptive : analysis -> csrmv -> copy_mat_info -> destroy.
//
// Covers _rocsparse_csrmv_info::clear() + destructor (destroy path) and
// rocsparse::copy_csrmv_info (mat_info copy path).
// ======================================================================
TEST_F(Info, csrmv_adaptive_copy_and_clear)
{
    device_vector<rocsparse_int> row_ptr{g_row_ptr};
    device_vector<rocsparse_int> col_ind{g_col_ind};
    device_vector<float>         val{g_val};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    rocsparse_mat_info src = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&src), rocsparse_status_success);

    const rocsparse_operation trans = rocsparse_operation_none;
    const rocsparse_int       n     = g_m;

    ASSERT_EQ(
        rocsparse_scsrmv_analysis(handle, trans, g_m, n, g_nnz, descr, val, row_ptr, col_ind, src),
        rocsparse_status_success);

    const float          alpha = 1.0f, beta = 0.0f;
    device_vector<float> x{std::vector<float>{1.0f, 1.0f, 1.0f}};
    device_vector<float> y{(size_t)g_m};
    ASSERT_TRUE(x.ptr && y.ptr);

    EXPECT_EQ(
        rocsparse_scsrmv(
            handle, trans, g_m, n, g_nnz, &alpha, descr, val, row_ptr, col_ind, src, x, &beta, y),
        rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // copy path -> rocsparse::copy_csrmv_info
    rocsparse_mat_info dest = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&dest), rocsparse_status_success);
    EXPECT_EQ(rocsparse_copy_mat_info(dest, src), rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // destroy path -> ~_rocsparse_csrmv_info -> clear()
    EXPECT_EQ(rocsparse_destroy_mat_info(dest), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_info(src), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

// ======================================================================
// bsrmv (block_dim = 1) : analysis -> bsrmv -> copy_mat_info -> destroy.
//
// Covers _rocsparse_bsrmv_info default ctor / dtor / get_csrmv_info /
// set_csrmv_info and rocsparse::copy_bsrmv_info (mat_info copy path).
// ======================================================================
TEST_F(Info, bsrmv_copy_and_destroy)
{
    device_vector<rocsparse_int> row_ptr{g_row_ptr};
    device_vector<rocsparse_int> col_ind{g_col_ind};
    device_vector<float>         val{g_val};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    rocsparse_mat_info src = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&src), rocsparse_status_success);

    const rocsparse_direction dir       = rocsparse_direction_row;
    const rocsparse_operation trans     = rocsparse_operation_none;
    const rocsparse_int       block_dim = 1;
    const rocsparse_int       nb        = g_m;

    const rocsparse_status st = rocsparse_sbsrmv_analysis(
        handle, dir, trans, g_m, nb, g_nnz, descr, val, row_ptr, col_ind, block_dim, src);
    if(st == rocsparse_status_not_implemented)
    {
        EXPECT_EQ(rocsparse_destroy_mat_info(src), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
        GTEST_SKIP() << "bsrmv analysis not implemented on this arch";
    }
    ASSERT_EQ(st, rocsparse_status_success);

    const float          alpha = 1.0f, beta = 0.0f;
    device_vector<float> x{std::vector<float>{1.0f, 1.0f, 1.0f}};
    device_vector<float> y{(size_t)g_m};
    ASSERT_TRUE(x.ptr && y.ptr);

    EXPECT_EQ(rocsparse_sbsrmv(handle,
                               dir,
                               trans,
                               g_m,
                               nb,
                               g_nnz,
                               &alpha,
                               descr,
                               val,
                               row_ptr,
                               col_ind,
                               block_dim,
                               src,
                               x,
                               &beta,
                               y),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    rocsparse_mat_info dest = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&dest), rocsparse_status_success);
    EXPECT_EQ(rocsparse_copy_mat_info(dest, src), rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    EXPECT_EQ(rocsparse_destroy_mat_info(dest), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_info(src), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

// ======================================================================
// csritsv : buffer_size -> analysis (contiguous triangular matrix).
//
// A contiguous (non-submatrix) triangular matrix drives the internal
// csrmv analysis, exercising _rocsparse_csritsv_info::get_csrmv_info /
// set_csrmv_info accessors.
// ======================================================================
TEST_F(Info, csritsv_analysis_csrmv_accessors)
{
    device_vector<rocsparse_int> row_ptr{g_row_ptr};
    device_vector<rocsparse_int> col_ind{g_col_ind};
    device_vector<float>         val{g_val};
    ASSERT_TRUE(row_ptr.ptr && col_ind.ptr && val.ptr);

    rocsparse_mat_descr descr = nullptr;
    ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
    rocsparse_mat_info src = nullptr;
    ASSERT_EQ(rocsparse_create_mat_info(&src), rocsparse_status_success);

    const rocsparse_operation trans = rocsparse_operation_none;

    size_t                 buffer_size = 0;
    const rocsparse_status stbs        = rocsparse_scsritsv_buffer_size(
        handle, trans, g_m, g_nnz, descr, val, row_ptr, col_ind, src, &buffer_size);
    if(stbs == rocsparse_status_not_implemented)
    {
        EXPECT_EQ(rocsparse_destroy_mat_info(src), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
        GTEST_SKIP() << "csritsv not implemented on this arch";
    }
    ASSERT_EQ(stbs, rocsparse_status_success);
    device_vector<char> buffer{buffer_size};
    ASSERT_TRUE(buffer.ptr);

    EXPECT_EQ(rocsparse_scsritsv_analysis(handle,
                                          trans,
                                          g_m,
                                          g_nnz,
                                          descr,
                                          val,
                                          row_ptr,
                                          col_ind,
                                          src,
                                          rocsparse_analysis_policy_force,
                                          rocsparse_solve_policy_auto,
                                          buffer.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    EXPECT_EQ(rocsparse_destroy_mat_info(src), rocsparse_status_success);
    EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
}

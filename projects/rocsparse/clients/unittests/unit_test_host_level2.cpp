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
// Host-path unit tests for the level-2 (per-precision) C API:
//   csrmv (+analysis/clear), coomv, ellmv, hybmv, bsrmv (+analysis/clear),
//   gebsrmv, gemvi (+buffer_size), csrsv (buffer_size/analysis/solve/clear/
//   zero_pivot) and bsrsv (buffer_size/analysis/solve/clear/zero_pivot).
//
// These drive the public C entry points so they exercise the host dispatch,
// argument-validation and analysis code in library/src/level2 across all four
// precisions (s/d/c/z) plus bad-arg and quick-return branches. Inputs are kept
// tiny (<= 3x3, nnz <= 3) so the suite never trips the "Insufficient memory"
// auto-skip on the 15 GB card. Triangular-solve routines use a well-conditioned
// diagonal matrix so there is never a zero pivot.
//
#include "unit_test_utils.hpp"

#include "rocsparse.h"

using namespace rocsparse_ut;

namespace
{
    // ---------------------------------------------------------------------
    // Tiny 3x3 identity in CSR/COO/ELL/BSR(1x1), plus dense x and y vectors.
    // ---------------------------------------------------------------------
    template <typename T>
    struct Id3
    {
        device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
        device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
        device_vector<rocsparse_int> row_ind{
            std::vector<rocsparse_int>{0, 1, 2}}; // COO row indices
        device_vector<T> val{std::vector<T>{scalar<T>(1), scalar<T>(1), scalar<T>(1)}};
        device_vector<T> x{std::vector<T>{scalar<T>(1), scalar<T>(2), scalar<T>(3)}};
        device_vector<T> y{std::vector<T>(3, scalar<T>(1))};
        bool             ok() const
        {
            return row_ptr.ptr && col_ind.ptr && row_ind.ptr && val.ptr && x.ptr && y.ptr;
        }
    };

    // Diagonal diag(2,3,4) in CSR (well-conditioned lower/upper triangular).
    template <typename T>
    struct Diag3
    {
        device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
        device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
        device_vector<T>             val{std::vector<T>{scalar<T>(2), scalar<T>(3), scalar<T>(4)}};
        device_vector<T>             x{std::vector<T>{scalar<T>(2), scalar<T>(3), scalar<T>(4)}};
        device_vector<T>             y{std::vector<T>(3, scalar<T>(0))};
        bool                         ok() const
        {
            return row_ptr.ptr && col_ind.ptr && val.ptr && x.ptr && y.ptr;
        }
    };

    // ===== per-precision overload sets =====================================
#define UT_OVL_CSRMV(T, PFX)                                                             \
    inline rocsparse_status ut_csrmv(rocsparse_handle          h,                        \
                                     rocsparse_operation       trans,                    \
                                     rocsparse_int             m,                        \
                                     rocsparse_int             n,                        \
                                     rocsparse_int             nnz,                      \
                                     const T*                  alpha,                    \
                                     const rocsparse_mat_descr descr,                    \
                                     const T*                  val,                      \
                                     const rocsparse_int*      row_ptr,                  \
                                     const rocsparse_int*      col_ind,                  \
                                     rocsparse_mat_info        info,                     \
                                     const T*                  x,                        \
                                     const T*                  beta,                     \
                                     T*                        y)                        \
    {                                                                                    \
        return rocsparse_##PFX##csrmv(                                                   \
            h, trans, m, n, nnz, alpha, descr, val, row_ptr, col_ind, info, x, beta, y); \
    }                                                                                    \
    inline rocsparse_status ut_csrmv_analysis(rocsparse_handle          h,               \
                                              rocsparse_operation       trans,           \
                                              rocsparse_int             m,               \
                                              rocsparse_int             n,               \
                                              rocsparse_int             nnz,             \
                                              const rocsparse_mat_descr descr,           \
                                              const T*                  val,             \
                                              const rocsparse_int*      row_ptr,         \
                                              const rocsparse_int*      col_ind,         \
                                              rocsparse_mat_info        info)            \
    {                                                                                    \
        return rocsparse_##PFX##csrmv_analysis(                                          \
            h, trans, m, n, nnz, descr, val, row_ptr, col_ind, info);                    \
    }
    UT_OVL_CSRMV(float, s)
    UT_OVL_CSRMV(double, d)
    UT_OVL_CSRMV(rocsparse_float_complex, c)
    UT_OVL_CSRMV(rocsparse_double_complex, z)

#define UT_OVL_COOMV(T, PFX)                                                       \
    inline rocsparse_status ut_coomv(rocsparse_handle          h,                  \
                                     rocsparse_operation       trans,              \
                                     rocsparse_int             m,                  \
                                     rocsparse_int             n,                  \
                                     rocsparse_int             nnz,                \
                                     const T*                  alpha,              \
                                     const rocsparse_mat_descr descr,              \
                                     const T*                  val,                \
                                     const rocsparse_int*      row_ind,            \
                                     const rocsparse_int*      col_ind,            \
                                     const T*                  x,                  \
                                     const T*                  beta,               \
                                     T*                        y)                  \
    {                                                                              \
        return rocsparse_##PFX##coomv(                                             \
            h, trans, m, n, nnz, alpha, descr, val, row_ind, col_ind, x, beta, y); \
    }
    UT_OVL_COOMV(float, s)
    UT_OVL_COOMV(double, d)
    UT_OVL_COOMV(rocsparse_float_complex, c)
    UT_OVL_COOMV(rocsparse_double_complex, z)

#define UT_OVL_ELLMV(T, PFX)                                                    \
    inline rocsparse_status ut_ellmv(rocsparse_handle          h,               \
                                     rocsparse_operation       trans,           \
                                     rocsparse_int             m,               \
                                     rocsparse_int             n,               \
                                     const T*                  alpha,           \
                                     const rocsparse_mat_descr descr,           \
                                     const T*                  val,             \
                                     const rocsparse_int*      col_ind,         \
                                     rocsparse_int             ell_width,       \
                                     const T*                  x,               \
                                     const T*                  beta,            \
                                     T*                        y)               \
    {                                                                           \
        return rocsparse_##PFX##ellmv(                                          \
            h, trans, m, n, alpha, descr, val, col_ind, ell_width, x, beta, y); \
    }
    UT_OVL_ELLMV(float, s)
    UT_OVL_ELLMV(double, d)
    UT_OVL_ELLMV(rocsparse_float_complex, c)
    UT_OVL_ELLMV(rocsparse_double_complex, z)

#define UT_OVL_BSRMV(T, PFX)                                                             \
    inline rocsparse_status ut_bsrmv(rocsparse_handle          h,                        \
                                     rocsparse_direction       dir,                      \
                                     rocsparse_operation       trans,                    \
                                     rocsparse_int             mb,                       \
                                     rocsparse_int             nb,                       \
                                     rocsparse_int             nnzb,                     \
                                     const T*                  alpha,                    \
                                     const rocsparse_mat_descr descr,                    \
                                     const T*                  val,                      \
                                     const rocsparse_int*      row_ptr,                  \
                                     const rocsparse_int*      col_ind,                  \
                                     rocsparse_int             block_dim,                \
                                     rocsparse_mat_info        info,                     \
                                     const T*                  x,                        \
                                     const T*                  beta,                     \
                                     T*                        y)                        \
    {                                                                                    \
        return rocsparse_##PFX##bsrmv(h,                                                 \
                                      dir,                                               \
                                      trans,                                             \
                                      mb,                                                \
                                      nb,                                                \
                                      nnzb,                                              \
                                      alpha,                                             \
                                      descr,                                             \
                                      val,                                               \
                                      row_ptr,                                           \
                                      col_ind,                                           \
                                      block_dim,                                         \
                                      info,                                              \
                                      x,                                                 \
                                      beta,                                              \
                                      y);                                                \
    }                                                                                    \
    inline rocsparse_status ut_bsrmv_analysis(rocsparse_handle          h,               \
                                              rocsparse_direction       dir,             \
                                              rocsparse_operation       trans,           \
                                              rocsparse_int             mb,              \
                                              rocsparse_int             nb,              \
                                              rocsparse_int             nnzb,            \
                                              const rocsparse_mat_descr descr,           \
                                              const T*                  val,             \
                                              const rocsparse_int*      row_ptr,         \
                                              const rocsparse_int*      col_ind,         \
                                              rocsparse_int             block_dim,       \
                                              rocsparse_mat_info        info)            \
    {                                                                                    \
        return rocsparse_##PFX##bsrmv_analysis(                                          \
            h, dir, trans, mb, nb, nnzb, descr, val, row_ptr, col_ind, block_dim, info); \
    }
    UT_OVL_BSRMV(float, s)
    UT_OVL_BSRMV(double, d)
    UT_OVL_BSRMV(rocsparse_float_complex, c)
    UT_OVL_BSRMV(rocsparse_double_complex, z)

#define UT_OVL_GEBSRMV(T, PFX)                                                  \
    inline rocsparse_status ut_gebsrmv(rocsparse_handle          h,             \
                                       rocsparse_direction       dir,           \
                                       rocsparse_operation       trans,         \
                                       rocsparse_int             mb,            \
                                       rocsparse_int             nb,            \
                                       rocsparse_int             nnzb,          \
                                       const T*                  alpha,         \
                                       const rocsparse_mat_descr descr,         \
                                       const T*                  val,           \
                                       const rocsparse_int*      row_ptr,       \
                                       const rocsparse_int*      col_ind,       \
                                       rocsparse_int             row_block_dim, \
                                       rocsparse_int             col_block_dim, \
                                       const T*                  x,             \
                                       const T*                  beta,          \
                                       T*                        y)             \
    {                                                                           \
        return rocsparse_##PFX##gebsrmv(h,                                      \
                                        dir,                                    \
                                        trans,                                  \
                                        mb,                                     \
                                        nb,                                     \
                                        nnzb,                                   \
                                        alpha,                                  \
                                        descr,                                  \
                                        val,                                    \
                                        row_ptr,                                \
                                        col_ind,                                \
                                        row_block_dim,                          \
                                        col_block_dim,                          \
                                        x,                                      \
                                        beta,                                   \
                                        y);                                     \
    }
    UT_OVL_GEBSRMV(float, s)
    UT_OVL_GEBSRMV(double, d)
    UT_OVL_GEBSRMV(rocsparse_float_complex, c)
    UT_OVL_GEBSRMV(rocsparse_double_complex, z)

#define UT_OVL_HYBMV(T, PFX)                                                              \
    inline rocsparse_status ut_hybmv(rocsparse_handle          h,                         \
                                     rocsparse_operation       trans,                     \
                                     const T*                  alpha,                     \
                                     const rocsparse_mat_descr descr,                     \
                                     const rocsparse_hyb_mat   hyb,                       \
                                     const T*                  x,                         \
                                     const T*                  beta,                      \
                                     T*                        y)                         \
    {                                                                                     \
        return rocsparse_##PFX##hybmv(h, trans, alpha, descr, hyb, x, beta, y);           \
    }                                                                                     \
    inline rocsparse_status ut_csr2hyb(rocsparse_handle          h,                       \
                                       rocsparse_int             m,                       \
                                       rocsparse_int             n,                       \
                                       const rocsparse_mat_descr descr,                   \
                                       const T*                  val,                     \
                                       const rocsparse_int*      row_ptr,                 \
                                       const rocsparse_int*      col_ind,                 \
                                       rocsparse_hyb_mat         hyb)                     \
    {                                                                                     \
        return rocsparse_##PFX##csr2hyb(                                                  \
            h, m, n, descr, val, row_ptr, col_ind, hyb, 0, rocsparse_hyb_partition_auto); \
    }
    UT_OVL_HYBMV(float, s)
    UT_OVL_HYBMV(double, d)
    UT_OVL_HYBMV(rocsparse_float_complex, c)
    UT_OVL_HYBMV(rocsparse_double_complex, z)

#define UT_OVL_GEMVI(T, PFX)                                                          \
    inline rocsparse_status ut_gemvi_buffer_size(rocsparse_handle    h,               \
                                                 rocsparse_operation trans,           \
                                                 rocsparse_int       m,               \
                                                 rocsparse_int       n,               \
                                                 rocsparse_int       nnz,             \
                                                 size_t*             buffer_size,     \
                                                 T                   dummy)           \
    {                                                                                 \
        (void)dummy;                                                                  \
        return rocsparse_##PFX##gemvi_buffer_size(h, trans, m, n, nnz, buffer_size);  \
    }                                                                                 \
    inline rocsparse_status ut_gemvi(rocsparse_handle     h,                          \
                                     rocsparse_operation  trans,                      \
                                     rocsparse_int        m,                          \
                                     rocsparse_int        n,                          \
                                     const T*             alpha,                      \
                                     const T*             A,                          \
                                     rocsparse_int        lda,                        \
                                     rocsparse_int        nnz,                        \
                                     const T*             x_val,                      \
                                     const rocsparse_int* x_ind,                      \
                                     const T*             beta,                       \
                                     T*                   y,                          \
                                     rocsparse_index_base base,                       \
                                     void*                buffer)                     \
    {                                                                                 \
        return rocsparse_##PFX##gemvi(                                                \
            h, trans, m, n, alpha, A, lda, nnz, x_val, x_ind, beta, y, base, buffer); \
    }
    UT_OVL_GEMVI(float, s)
    UT_OVL_GEMVI(double, d)
    UT_OVL_GEMVI(rocsparse_float_complex, c)
    UT_OVL_GEMVI(rocsparse_double_complex, z)

#define UT_OVL_CSRSV(T, PFX)                                                            \
    inline rocsparse_status ut_csrsv_buffer_size(rocsparse_handle          h,           \
                                                 rocsparse_operation       trans,       \
                                                 rocsparse_int             m,           \
                                                 rocsparse_int             nnz,         \
                                                 const rocsparse_mat_descr descr,       \
                                                 const T*                  val,         \
                                                 const rocsparse_int*      row_ptr,     \
                                                 const rocsparse_int*      col_ind,     \
                                                 rocsparse_mat_info        info,        \
                                                 size_t*                   buffer_size) \
    {                                                                                   \
        return rocsparse_##PFX##csrsv_buffer_size(                                      \
            h, trans, m, nnz, descr, val, row_ptr, col_ind, info, buffer_size);         \
    }                                                                                   \
    inline rocsparse_status ut_csrsv_analysis(rocsparse_handle          h,              \
                                              rocsparse_operation       trans,          \
                                              rocsparse_int             m,              \
                                              rocsparse_int             nnz,            \
                                              const rocsparse_mat_descr descr,          \
                                              const T*                  val,            \
                                              const rocsparse_int*      row_ptr,        \
                                              const rocsparse_int*      col_ind,        \
                                              rocsparse_mat_info        info,           \
                                              void*                     buffer)         \
    {                                                                                   \
        return rocsparse_##PFX##csrsv_analysis(h,                                       \
                                               trans,                                   \
                                               m,                                       \
                                               nnz,                                     \
                                               descr,                                   \
                                               val,                                     \
                                               row_ptr,                                 \
                                               col_ind,                                 \
                                               info,                                    \
                                               rocsparse_analysis_policy_reuse,         \
                                               rocsparse_solve_policy_auto,             \
                                               buffer);                                 \
    }                                                                                   \
    inline rocsparse_status ut_csrsv_solve(rocsparse_handle          h,                 \
                                           rocsparse_operation       trans,             \
                                           rocsparse_int             m,                 \
                                           rocsparse_int             nnz,               \
                                           const T*                  alpha,             \
                                           const rocsparse_mat_descr descr,             \
                                           const T*                  val,               \
                                           const rocsparse_int*      row_ptr,           \
                                           const rocsparse_int*      col_ind,           \
                                           rocsparse_mat_info        info,              \
                                           const T*                  x,                 \
                                           T*                        y,                 \
                                           void*                     buffer)            \
    {                                                                                   \
        return rocsparse_##PFX##csrsv_solve(h,                                          \
                                            trans,                                      \
                                            m,                                          \
                                            nnz,                                        \
                                            alpha,                                      \
                                            descr,                                      \
                                            val,                                        \
                                            row_ptr,                                    \
                                            col_ind,                                    \
                                            info,                                       \
                                            x,                                          \
                                            y,                                          \
                                            rocsparse_solve_policy_auto,                \
                                            buffer);                                    \
    }
    UT_OVL_CSRSV(float, s)
    UT_OVL_CSRSV(double, d)
    UT_OVL_CSRSV(rocsparse_float_complex, c)
    UT_OVL_CSRSV(rocsparse_double_complex, z)

#define UT_OVL_BSRSV(T, PFX)                                                                      \
    inline rocsparse_status ut_bsrsv_buffer_size(rocsparse_handle          h,                     \
                                                 rocsparse_direction       dir,                   \
                                                 rocsparse_operation       trans,                 \
                                                 rocsparse_int             mb,                    \
                                                 rocsparse_int             nnzb,                  \
                                                 const rocsparse_mat_descr descr,                 \
                                                 const T*                  val,                   \
                                                 const rocsparse_int*      row_ptr,               \
                                                 const rocsparse_int*      col_ind,               \
                                                 rocsparse_int             block_dim,             \
                                                 rocsparse_mat_info        info,                  \
                                                 size_t*                   buffer_size)           \
    {                                                                                             \
        return rocsparse_##PFX##bsrsv_buffer_size(                                                \
            h, dir, trans, mb, nnzb, descr, val, row_ptr, col_ind, block_dim, info, buffer_size); \
    }                                                                                             \
    inline rocsparse_status ut_bsrsv_analysis(rocsparse_handle          h,                        \
                                              rocsparse_direction       dir,                      \
                                              rocsparse_operation       trans,                    \
                                              rocsparse_int             mb,                       \
                                              rocsparse_int             nnzb,                     \
                                              const rocsparse_mat_descr descr,                    \
                                              const T*                  val,                      \
                                              const rocsparse_int*      row_ptr,                  \
                                              const rocsparse_int*      col_ind,                  \
                                              rocsparse_int             block_dim,                \
                                              rocsparse_mat_info        info,                     \
                                              void*                     buffer)                   \
    {                                                                                             \
        return rocsparse_##PFX##bsrsv_analysis(h,                                                 \
                                               dir,                                               \
                                               trans,                                             \
                                               mb,                                                \
                                               nnzb,                                              \
                                               descr,                                             \
                                               val,                                               \
                                               row_ptr,                                           \
                                               col_ind,                                           \
                                               block_dim,                                         \
                                               info,                                              \
                                               rocsparse_analysis_policy_reuse,                   \
                                               rocsparse_solve_policy_auto,                       \
                                               buffer);                                           \
    }                                                                                             \
    inline rocsparse_status ut_bsrsv_solve(rocsparse_handle          h,                           \
                                           rocsparse_direction       dir,                         \
                                           rocsparse_operation       trans,                       \
                                           rocsparse_int             mb,                          \
                                           rocsparse_int             nnzb,                        \
                                           const T*                  alpha,                       \
                                           const rocsparse_mat_descr descr,                       \
                                           const T*                  val,                         \
                                           const rocsparse_int*      row_ptr,                     \
                                           const rocsparse_int*      col_ind,                     \
                                           rocsparse_int             block_dim,                   \
                                           rocsparse_mat_info        info,                        \
                                           const T*                  x,                           \
                                           T*                        y,                           \
                                           void*                     buffer)                      \
    {                                                                                             \
        return rocsparse_##PFX##bsrsv_solve(h,                                                    \
                                            dir,                                                  \
                                            trans,                                                \
                                            mb,                                                   \
                                            nnzb,                                                 \
                                            alpha,                                                \
                                            descr,                                                \
                                            val,                                                  \
                                            row_ptr,                                              \
                                            col_ind,                                              \
                                            block_dim,                                            \
                                            info,                                                 \
                                            x,                                                    \
                                            y,                                                    \
                                            rocsparse_solve_policy_auto,                          \
                                            buffer);                                              \
    }
    UT_OVL_BSRSV(float, s)
    UT_OVL_BSRSV(double, d)
    UT_OVL_BSRSV(rocsparse_float_complex, c)
    UT_OVL_BSRSV(rocsparse_double_complex, z)

    // ===== templated per-precision runners =================================
    template <typename T>
    void run_csrmv(rocsparse_handle handle)
    {
        Id3<T> A;
        ASSERT_TRUE(A.ok());
        const T             alpha = scalar<T>(2), beta = scalar<T>(1);
        rocsparse_mat_descr descr = nullptr;
        ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);

        // compute without analysis info
        UT_EXPECT_ROC(ut_csrmv(handle,
                               rocsparse_operation_none,
                               3,
                               3,
                               3,
                               &alpha,
                               descr,
                               A.val,
                               A.row_ptr,
                               A.col_ind,
                               nullptr,
                               A.x,
                               &beta,
                               A.y),
                      rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        // analysis -> compute with info -> clear
        rocsparse_mat_info info = nullptr;
        ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);
        UT_EXPECT_ROC(ut_csrmv_analysis(handle,
                                        rocsparse_operation_none,
                                        3,
                                        3,
                                        3,
                                        descr,
                                        A.val,
                                        A.row_ptr,
                                        A.col_ind,
                                        info),
                      rocsparse_status_success);
        UT_EXPECT_ROC(ut_csrmv(handle,
                               rocsparse_operation_none,
                               3,
                               3,
                               3,
                               &alpha,
                               descr,
                               A.val,
                               A.row_ptr,
                               A.col_ind,
                               info,
                               A.x,
                               &beta,
                               A.y),
                      rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_EQ(rocsparse_csrmv_clear(handle, info), rocsparse_status_success);

        // bad args
        UT_EXPECT_ROC(ut_csrmv(nullptr,
                               rocsparse_operation_none,
                               3,
                               3,
                               3,
                               &alpha,
                               descr,
                               A.val,
                               A.row_ptr,
                               A.col_ind,
                               nullptr,
                               A.x,
                               &beta,
                               A.y),
                      rocsparse_status_invalid_handle);
        UT_EXPECT_ROC(ut_csrmv(handle,
                               rocsparse_operation_none,
                               -1,
                               3,
                               3,
                               &alpha,
                               descr,
                               A.val,
                               A.row_ptr,
                               A.col_ind,
                               nullptr,
                               A.x,
                               &beta,
                               A.y),
                      rocsparse_status_invalid_size);
        UT_EXPECT_ROC(ut_csrmv(handle,
                               rocsparse_operation_none,
                               3,
                               3,
                               3,
                               nullptr,
                               descr,
                               A.val,
                               A.row_ptr,
                               A.col_ind,
                               nullptr,
                               A.x,
                               &beta,
                               A.y),
                      rocsparse_status_invalid_pointer);

        EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
    }

    template <typename T>
    void run_coomv(rocsparse_handle handle)
    {
        Id3<T> A;
        ASSERT_TRUE(A.ok());
        const T             alpha = scalar<T>(2), beta = scalar<T>(1);
        rocsparse_mat_descr descr = nullptr;
        ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);

        UT_EXPECT_ROC(ut_coomv(handle,
                               rocsparse_operation_none,
                               3,
                               3,
                               3,
                               &alpha,
                               descr,
                               A.val,
                               A.row_ind,
                               A.col_ind,
                               A.x,
                               &beta,
                               A.y),
                      rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        // quick return: m == 0
        UT_EXPECT_ROC(ut_coomv(handle,
                               rocsparse_operation_none,
                               0,
                               0,
                               0,
                               &alpha,
                               descr,
                               A.val,
                               A.row_ind,
                               A.col_ind,
                               A.x,
                               &beta,
                               A.y),
                      rocsparse_status_success);

        UT_EXPECT_ROC(ut_coomv(nullptr,
                               rocsparse_operation_none,
                               3,
                               3,
                               3,
                               &alpha,
                               descr,
                               A.val,
                               A.row_ind,
                               A.col_ind,
                               A.x,
                               &beta,
                               A.y),
                      rocsparse_status_invalid_handle);
        UT_EXPECT_ROC(ut_coomv(handle,
                               rocsparse_operation_none,
                               -1,
                               3,
                               3,
                               &alpha,
                               descr,
                               A.val,
                               A.row_ind,
                               A.col_ind,
                               A.x,
                               &beta,
                               A.y),
                      rocsparse_status_invalid_size);
        UT_EXPECT_ROC(ut_coomv(handle,
                               rocsparse_operation_none,
                               3,
                               3,
                               3,
                               nullptr,
                               descr,
                               A.val,
                               A.row_ind,
                               A.col_ind,
                               A.x,
                               &beta,
                               A.y),
                      rocsparse_status_invalid_pointer);

        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
    }

    template <typename T>
    void run_ellmv(rocsparse_handle handle)
    {
        // identity 3x3 in ELL: width 1, one column index per row.
        device_vector<rocsparse_int> ell_col{std::vector<rocsparse_int>{0, 1, 2}};
        device_vector<T> ell_val{std::vector<T>{scalar<T>(1), scalar<T>(1), scalar<T>(1)}};
        device_vector<T> x{std::vector<T>{scalar<T>(1), scalar<T>(2), scalar<T>(3)}};
        device_vector<T> y{std::vector<T>(3, scalar<T>(1))};
        ASSERT_TRUE(ell_col.ptr && ell_val.ptr && x.ptr && y.ptr);
        const T             alpha = scalar<T>(2), beta = scalar<T>(1);
        rocsparse_mat_descr descr = nullptr;
        ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);

        UT_EXPECT_ROC(ut_ellmv(handle,
                               rocsparse_operation_none,
                               3,
                               3,
                               &alpha,
                               descr,
                               ell_val,
                               ell_col,
                               1,
                               x,
                               &beta,
                               y),
                      rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        UT_EXPECT_ROC(ut_ellmv(nullptr,
                               rocsparse_operation_none,
                               3,
                               3,
                               &alpha,
                               descr,
                               ell_val,
                               ell_col,
                               1,
                               x,
                               &beta,
                               y),
                      rocsparse_status_invalid_handle);
        UT_EXPECT_ROC(ut_ellmv(handle,
                               rocsparse_operation_none,
                               -1,
                               3,
                               &alpha,
                               descr,
                               ell_val,
                               ell_col,
                               1,
                               x,
                               &beta,
                               y),
                      rocsparse_status_invalid_size);
        UT_EXPECT_ROC(ut_ellmv(handle,
                               rocsparse_operation_none,
                               3,
                               3,
                               &alpha,
                               descr,
                               ell_val,
                               ell_col,
                               -1,
                               x,
                               &beta,
                               y),
                      rocsparse_status_invalid_size);
        UT_EXPECT_ROC(ut_ellmv(handle,
                               rocsparse_operation_none,
                               3,
                               3,
                               nullptr,
                               descr,
                               ell_val,
                               ell_col,
                               1,
                               x,
                               &beta,
                               y),
                      rocsparse_status_invalid_pointer);

        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
    }

    template <typename T>
    void run_bsrmv(rocsparse_handle handle)
    {
        // block_dim = 1 -> BSR identity is a 3x3 CSR identity.
        Id3<T> A;
        ASSERT_TRUE(A.ok());
        const T             alpha = scalar<T>(2), beta = scalar<T>(1);
        rocsparse_mat_descr descr = nullptr;
        ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
        rocsparse_mat_info info = nullptr;
        ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

        // analysis -> compute (with info) -> clear
        UT_EXPECT_ROC(ut_bsrmv_analysis(handle,
                                        rocsparse_direction_row,
                                        rocsparse_operation_none,
                                        3,
                                        3,
                                        3,
                                        descr,
                                        A.val,
                                        A.row_ptr,
                                        A.col_ind,
                                        1,
                                        info),
                      rocsparse_status_success);
        UT_EXPECT_ROC(ut_bsrmv(handle,
                               rocsparse_direction_row,
                               rocsparse_operation_none,
                               3,
                               3,
                               3,
                               &alpha,
                               descr,
                               A.val,
                               A.row_ptr,
                               A.col_ind,
                               1,
                               info,
                               A.x,
                               &beta,
                               A.y),
                      rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_EQ(rocsparse_bsrmv_clear(handle, info), rocsparse_status_success);

        // compute without info
        UT_EXPECT_ROC(ut_bsrmv(handle,
                               rocsparse_direction_row,
                               rocsparse_operation_none,
                               3,
                               3,
                               3,
                               &alpha,
                               descr,
                               A.val,
                               A.row_ptr,
                               A.col_ind,
                               1,
                               nullptr,
                               A.x,
                               &beta,
                               A.y),
                      rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        // bad args
        UT_EXPECT_ROC(ut_bsrmv(nullptr,
                               rocsparse_direction_row,
                               rocsparse_operation_none,
                               3,
                               3,
                               3,
                               &alpha,
                               descr,
                               A.val,
                               A.row_ptr,
                               A.col_ind,
                               1,
                               nullptr,
                               A.x,
                               &beta,
                               A.y),
                      rocsparse_status_invalid_handle);
        UT_EXPECT_ROC(ut_bsrmv(handle,
                               rocsparse_direction_row,
                               rocsparse_operation_none,
                               3,
                               3,
                               3,
                               &alpha,
                               descr,
                               A.val,
                               A.row_ptr,
                               A.col_ind,
                               -1,
                               nullptr,
                               A.x,
                               &beta,
                               A.y),
                      rocsparse_status_invalid_size);
        UT_EXPECT_ROC(ut_bsrmv(handle,
                               rocsparse_direction_row,
                               rocsparse_operation_none,
                               3,
                               3,
                               3,
                               nullptr,
                               descr,
                               A.val,
                               A.row_ptr,
                               A.col_ind,
                               1,
                               nullptr,
                               A.x,
                               &beta,
                               A.y),
                      rocsparse_status_invalid_pointer);

        EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
    }

    template <typename T>
    void run_gebsrmv(rocsparse_handle handle)
    {
        // row_block_dim = col_block_dim = 1 -> a 3x3 CSR identity.
        Id3<T> A;
        ASSERT_TRUE(A.ok());
        const T             alpha = scalar<T>(2), beta = scalar<T>(1);
        rocsparse_mat_descr descr = nullptr;
        ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);

        UT_EXPECT_ROC(ut_gebsrmv(handle,
                                 rocsparse_direction_row,
                                 rocsparse_operation_none,
                                 3,
                                 3,
                                 3,
                                 &alpha,
                                 descr,
                                 A.val,
                                 A.row_ptr,
                                 A.col_ind,
                                 1,
                                 1,
                                 A.x,
                                 &beta,
                                 A.y),
                      rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        UT_EXPECT_ROC(ut_gebsrmv(nullptr,
                                 rocsparse_direction_row,
                                 rocsparse_operation_none,
                                 3,
                                 3,
                                 3,
                                 &alpha,
                                 descr,
                                 A.val,
                                 A.row_ptr,
                                 A.col_ind,
                                 1,
                                 1,
                                 A.x,
                                 &beta,
                                 A.y),
                      rocsparse_status_invalid_handle);
        UT_EXPECT_ROC(ut_gebsrmv(handle,
                                 rocsparse_direction_row,
                                 rocsparse_operation_none,
                                 3,
                                 3,
                                 3,
                                 &alpha,
                                 descr,
                                 A.val,
                                 A.row_ptr,
                                 A.col_ind,
                                 -1,
                                 1,
                                 A.x,
                                 &beta,
                                 A.y),
                      rocsparse_status_invalid_size);
        UT_EXPECT_ROC(ut_gebsrmv(handle,
                                 rocsparse_direction_row,
                                 rocsparse_operation_none,
                                 3,
                                 3,
                                 3,
                                 nullptr,
                                 descr,
                                 A.val,
                                 A.row_ptr,
                                 A.col_ind,
                                 1,
                                 1,
                                 A.x,
                                 &beta,
                                 A.y),
                      rocsparse_status_invalid_pointer);

        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
    }

    template <typename T>
    void run_gemvi(rocsparse_handle handle)
    {
        // dense 3x3 identity (column-major), sparse x with nnz=2.
        std::vector<T> hA(9, scalar<T>(0));
        hA[0] = scalar<T>(1);
        hA[4] = scalar<T>(1);
        hA[8] = scalar<T>(1);
        device_vector<T>             A{hA};
        device_vector<T>             x_val{std::vector<T>{scalar<T>(1), scalar<T>(1)}};
        device_vector<rocsparse_int> x_ind{std::vector<rocsparse_int>{0, 2}};
        device_vector<T>             y{std::vector<T>(3, scalar<T>(0))};
        ASSERT_TRUE(A.ptr && x_val.ptr && x_ind.ptr && y.ptr);
        const T alpha = scalar<T>(1), beta = scalar<T>(0);

        size_t buffer_size = 0;
        UT_EXPECT_ROC(ut_gemvi_buffer_size(
                          handle, rocsparse_operation_none, 3, 3, 2, &buffer_size, scalar<T>(0)),
                      rocsparse_status_success);
        device_vector<char> buffer{buffer_size ? buffer_size : 1};
        ASSERT_TRUE(buffer.ptr);

        UT_EXPECT_ROC(ut_gemvi(handle,
                               rocsparse_operation_none,
                               3,
                               3,
                               &alpha,
                               A,
                               3,
                               2,
                               x_val,
                               x_ind,
                               &beta,
                               y,
                               rocsparse_index_base_zero,
                               buffer.ptr),
                      rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        // bad args. Note: Xgemvi_buffer_size unconditionally reports success
        // (buffer size is a fixed 0), so it has no handle/argument guards.
        UT_EXPECT_ROC(ut_gemvi_buffer_size(
                          nullptr, rocsparse_operation_none, 3, 3, 2, &buffer_size, scalar<T>(0)),
                      rocsparse_status_success);
        UT_EXPECT_ROC(ut_gemvi(nullptr,
                               rocsparse_operation_none,
                               3,
                               3,
                               &alpha,
                               A,
                               3,
                               2,
                               x_val,
                               x_ind,
                               &beta,
                               y,
                               rocsparse_index_base_zero,
                               buffer.ptr),
                      rocsparse_status_invalid_handle);
        UT_EXPECT_ROC(ut_gemvi(handle,
                               rocsparse_operation_none,
                               -1,
                               3,
                               &alpha,
                               A,
                               3,
                               2,
                               x_val,
                               x_ind,
                               &beta,
                               y,
                               rocsparse_index_base_zero,
                               buffer.ptr),
                      rocsparse_status_invalid_size);
        UT_EXPECT_ROC(ut_gemvi(handle,
                               rocsparse_operation_none,
                               3,
                               3,
                               nullptr,
                               A,
                               3,
                               2,
                               x_val,
                               x_ind,
                               &beta,
                               y,
                               rocsparse_index_base_zero,
                               buffer.ptr),
                      rocsparse_status_invalid_pointer);
    }

    template <typename T>
    void run_csrsv(rocsparse_handle handle)
    {
        Diag3<T> A;
        ASSERT_TRUE(A.ok());
        const T             alpha = scalar<T>(1);
        rocsparse_mat_descr descr = nullptr;
        ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
        // triangular fill mode required for a solve.
        ASSERT_EQ(rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_lower),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_diag_type(descr, rocsparse_diag_type_non_unit),
                  rocsparse_status_success);
        rocsparse_mat_info info = nullptr;
        ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

        size_t buffer_size = 0;
        ASSERT_EQ(ut_csrsv_buffer_size(handle,
                                       rocsparse_operation_none,
                                       3,
                                       3,
                                       descr,
                                       A.val,
                                       A.row_ptr,
                                       A.col_ind,
                                       info,
                                       &buffer_size),
                  rocsparse_status_success);
        device_vector<char> buffer{buffer_size ? buffer_size : 1};
        ASSERT_TRUE(buffer.ptr);

        ASSERT_EQ(ut_csrsv_analysis(handle,
                                    rocsparse_operation_none,
                                    3,
                                    3,
                                    descr,
                                    A.val,
                                    A.row_ptr,
                                    A.col_ind,
                                    info,
                                    buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        UT_EXPECT_ROC(ut_csrsv_solve(handle,
                                     rocsparse_operation_none,
                                     3,
                                     3,
                                     &alpha,
                                     descr,
                                     A.val,
                                     A.row_ptr,
                                     A.col_ind,
                                     info,
                                     A.x,
                                     A.y,
                                     buffer.ptr),
                      rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        rocsparse_int position = -1;
        EXPECT_EQ(rocsparse_csrsv_zero_pivot(handle, descr, info, &position),
                  rocsparse_status_success);
        EXPECT_EQ(rocsparse_csrsv_clear(handle, descr, info), rocsparse_status_success);

        // bad args
        UT_EXPECT_ROC(ut_csrsv_buffer_size(nullptr,
                                           rocsparse_operation_none,
                                           3,
                                           3,
                                           descr,
                                           A.val,
                                           A.row_ptr,
                                           A.col_ind,
                                           info,
                                           &buffer_size),
                      rocsparse_status_invalid_handle);
        UT_EXPECT_ROC(ut_csrsv_buffer_size(handle,
                                           rocsparse_operation_none,
                                           -1,
                                           3,
                                           descr,
                                           A.val,
                                           A.row_ptr,
                                           A.col_ind,
                                           info,
                                           &buffer_size),
                      rocsparse_status_invalid_size);

        EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
    }

    template <typename T>
    void run_bsrsv(rocsparse_handle handle)
    {
        // block_dim = 1 -> diagonal 3x3 solve.
        Diag3<T> A;
        ASSERT_TRUE(A.ok());
        const T             alpha = scalar<T>(1);
        rocsparse_mat_descr descr = nullptr;
        ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_lower),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_diag_type(descr, rocsparse_diag_type_non_unit),
                  rocsparse_status_success);
        rocsparse_mat_info info = nullptr;
        ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

        size_t buffer_size = 0;
        ASSERT_EQ(ut_bsrsv_buffer_size(handle,
                                       rocsparse_direction_row,
                                       rocsparse_operation_none,
                                       3,
                                       3,
                                       descr,
                                       A.val,
                                       A.row_ptr,
                                       A.col_ind,
                                       1,
                                       info,
                                       &buffer_size),
                  rocsparse_status_success);
        device_vector<char> buffer{buffer_size ? buffer_size : 1};
        ASSERT_TRUE(buffer.ptr);

        ASSERT_EQ(ut_bsrsv_analysis(handle,
                                    rocsparse_direction_row,
                                    rocsparse_operation_none,
                                    3,
                                    3,
                                    descr,
                                    A.val,
                                    A.row_ptr,
                                    A.col_ind,
                                    1,
                                    info,
                                    buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        UT_EXPECT_ROC(ut_bsrsv_solve(handle,
                                     rocsparse_direction_row,
                                     rocsparse_operation_none,
                                     3,
                                     3,
                                     &alpha,
                                     descr,
                                     A.val,
                                     A.row_ptr,
                                     A.col_ind,
                                     1,
                                     info,
                                     A.x,
                                     A.y,
                                     buffer.ptr),
                      rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        rocsparse_int position = -1;
        EXPECT_EQ(rocsparse_bsrsv_zero_pivot(handle, info, &position), rocsparse_status_success);
        EXPECT_EQ(rocsparse_bsrsv_clear(handle, info), rocsparse_status_success);

        // bad args
        UT_EXPECT_ROC(ut_bsrsv_buffer_size(nullptr,
                                           rocsparse_direction_row,
                                           rocsparse_operation_none,
                                           3,
                                           3,
                                           descr,
                                           A.val,
                                           A.row_ptr,
                                           A.col_ind,
                                           1,
                                           info,
                                           &buffer_size),
                      rocsparse_status_invalid_handle);
        UT_EXPECT_ROC(ut_bsrsv_buffer_size(handle,
                                           rocsparse_direction_row,
                                           rocsparse_operation_none,
                                           -1,
                                           3,
                                           descr,
                                           A.val,
                                           A.row_ptr,
                                           A.col_ind,
                                           1,
                                           info,
                                           &buffer_size),
                      rocsparse_status_invalid_size);

        EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
    }

    template <typename T>
    void run_hybmv(rocsparse_handle handle)
    {
        Id3<T> A;
        ASSERT_TRUE(A.ok());
        const T             alpha = scalar<T>(2), beta = scalar<T>(1);
        rocsparse_mat_descr descr = nullptr;
        ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
        rocsparse_hyb_mat hyb = nullptr;
        ASSERT_EQ(rocsparse_create_hyb_mat(&hyb), rocsparse_status_success);

        ASSERT_EQ(ut_csr2hyb(handle, 3, 3, descr, A.val, A.row_ptr, A.col_ind, hyb),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        UT_EXPECT_ROC(
            ut_hybmv(handle, rocsparse_operation_none, &alpha, descr, hyb, A.x, &beta, A.y),
            rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        UT_EXPECT_ROC(
            ut_hybmv(nullptr, rocsparse_operation_none, &alpha, descr, hyb, A.x, &beta, A.y),
            rocsparse_status_invalid_handle);
        UT_EXPECT_ROC(
            ut_hybmv(handle, rocsparse_operation_none, nullptr, descr, hyb, A.x, &beta, A.y),
            rocsparse_status_invalid_pointer);

        EXPECT_EQ(rocsparse_destroy_hyb_mat(hyb), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
    }

    // -----------------------------------------------------------------------
    // gebsrmv with row_block_dim in {9..16} and col_block_dim in
    // {1..8, 10, 16, 17}. row_block_dim != col_block_dim routes through the
    // gebsrmv_template_row_block_dim_9_12 / _13_16 dispatch cascades, so this
    // fires every col_block_dim==N branch, the "col_block_dim <= 16" branch
    // and the ">16" general-kernel else branch for each row_block_dim. Uses a
    // minimal one-block GEBSR matrix (mb=nb=nnzb=1). Both storage directions
    // are exercised by the caller.
    // -----------------------------------------------------------------------
    template <typename T>
    void run_gebsrmv_block_dim(rocsparse_handle handle, rocsparse_direction dir)
    {
        rocsparse_mat_descr descr = nullptr;
        ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
        const T alpha = scalar<T>(2), beta = scalar<T>(1);

        // One block column / one block row.
        device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1}};
        device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0}};
        ASSERT_TRUE(row_ptr.ptr && col_ind.ptr);

        const rocsparse_int row_dims[] = {9, 10, 11, 12, 13, 14, 15, 16};
        const rocsparse_int col_dims[] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 16, 17};

        for(rocsparse_int R : row_dims)
        {
            for(rocsparse_int C : col_dims)
            {
                // A single R x C block, dense x (nb*C) and y (mb*R).
                device_vector<T> val{std::vector<T>(static_cast<size_t>(R) * C, scalar<T>(1))};
                device_vector<T> x{std::vector<T>(static_cast<size_t>(C), scalar<T>(1))};
                device_vector<T> y{std::vector<T>(static_cast<size_t>(R), scalar<T>(0))};
                ASSERT_TRUE(val.ptr && x.ptr && y.ptr);

                UT_EXPECT_ROC(ut_gebsrmv(handle,
                                         dir,
                                         rocsparse_operation_none,
                                         1,
                                         1,
                                         1,
                                         &alpha,
                                         descr,
                                         val,
                                         row_ptr,
                                         col_ind,
                                         R,
                                         C,
                                         x,
                                         &beta,
                                         y),
                              rocsparse_status_success);
                ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            }
        }

        // gebsrmv only implements rocsparse_operation_none; the transpose /
        // conjugate-transpose operations must report not_implemented.
        {
            const rocsparse_int R = 9, C = 1;
            device_vector<T>    val{std::vector<T>(static_cast<size_t>(R) * C, scalar<T>(1))};
            device_vector<T>    x{std::vector<T>(static_cast<size_t>(C), scalar<T>(1))};
            device_vector<T>    y{std::vector<T>(static_cast<size_t>(R), scalar<T>(0))};
            ASSERT_TRUE(val.ptr && x.ptr && y.ptr);

            UT_EXPECT_ROC(ut_gebsrmv(handle,
                                     dir,
                                     rocsparse_operation_transpose,
                                     1,
                                     1,
                                     1,
                                     &alpha,
                                     descr,
                                     val,
                                     row_ptr,
                                     col_ind,
                                     R,
                                     C,
                                     x,
                                     &beta,
                                     y),
                          rocsparse_status_not_implemented);
            UT_EXPECT_ROC(ut_gebsrmv(handle,
                                     dir,
                                     rocsparse_operation_conjugate_transpose,
                                     1,
                                     1,
                                     1,
                                     &alpha,
                                     descr,
                                     val,
                                     row_ptr,
                                     col_ind,
                                     R,
                                     C,
                                     x,
                                     &beta,
                                     y),
                          rocsparse_status_not_implemented);
        }

        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
    }

    // -----------------------------------------------------------------------
    // bsrsv_buffer_size across directions / operations plus its argument and
    // descriptor validation branches (invalid enums/sizes, non-general type,
    // unsorted storage, null arrays/pointers). The transpose operation takes a
    // distinct buffer-size adjustment path.
    // -----------------------------------------------------------------------
    template <typename T>
    void run_bsrsv_buffer_size(rocsparse_handle handle)
    {
        Diag3<T> A;
        ASSERT_TRUE(A.ok());
        rocsparse_mat_descr descr = nullptr;
        ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
        rocsparse_mat_info info = nullptr;
        ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

        size_t buffer_size = 0;

        // Valid buffer-size queries: both directions, several operations and
        // fill/diag descr settings, and block_dim 1 and 2.
        const rocsparse_direction dirs[]  = {rocsparse_direction_row, rocsparse_direction_column};
        const rocsparse_operation ops[]   = {rocsparse_operation_none,
                                           rocsparse_operation_transpose,
                                           rocsparse_operation_conjugate_transpose};
        const rocsparse_fill_mode fills[] = {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper};
        const rocsparse_diag_type diags[]
            = {rocsparse_diag_type_non_unit, rocsparse_diag_type_unit};

        for(rocsparse_direction d : dirs)
        {
            for(rocsparse_operation op : ops)
            {
                for(rocsparse_fill_mode fm : fills)
                {
                    for(rocsparse_diag_type dt : diags)
                    {
                        ASSERT_EQ(rocsparse_set_mat_fill_mode(descr, fm), rocsparse_status_success);
                        ASSERT_EQ(rocsparse_set_mat_diag_type(descr, dt), rocsparse_status_success);
                        for(rocsparse_int block_dim : {1, 2})
                        {
                            buffer_size = 0;
                            UT_EXPECT_ROC(ut_bsrsv_buffer_size(handle,
                                                               d,
                                                               op,
                                                               3,
                                                               3,
                                                               descr,
                                                               A.val,
                                                               A.row_ptr,
                                                               A.col_ind,
                                                               block_dim,
                                                               info,
                                                               &buffer_size),
                                          rocsparse_status_success);
                        }
                    }
                }
            }
        }

        // Reset to a plain lower/non-unit descriptor for the bad-arg cases.
        ASSERT_EQ(rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_lower),
                  rocsparse_status_success);
        ASSERT_EQ(rocsparse_set_mat_diag_type(descr, rocsparse_diag_type_non_unit),
                  rocsparse_status_success);

        // Invalid handle.
        UT_EXPECT_ROC(ut_bsrsv_buffer_size(nullptr,
                                           rocsparse_direction_row,
                                           rocsparse_operation_none,
                                           3,
                                           3,
                                           descr,
                                           A.val,
                                           A.row_ptr,
                                           A.col_ind,
                                           1,
                                           info,
                                           &buffer_size),
                      rocsparse_status_invalid_handle);

        // Invalid enum: direction.
        UT_EXPECT_ROC(ut_bsrsv_buffer_size(handle,
                                           static_cast<rocsparse_direction>(99),
                                           rocsparse_operation_none,
                                           3,
                                           3,
                                           descr,
                                           A.val,
                                           A.row_ptr,
                                           A.col_ind,
                                           1,
                                           info,
                                           &buffer_size),
                      rocsparse_status_invalid_value);

        // Invalid enum: operation.
        UT_EXPECT_ROC(ut_bsrsv_buffer_size(handle,
                                           rocsparse_direction_row,
                                           static_cast<rocsparse_operation>(99),
                                           3,
                                           3,
                                           descr,
                                           A.val,
                                           A.row_ptr,
                                           A.col_ind,
                                           1,
                                           info,
                                           &buffer_size),
                      rocsparse_status_invalid_value);

        // Invalid size: mb.
        UT_EXPECT_ROC(ut_bsrsv_buffer_size(handle,
                                           rocsparse_direction_row,
                                           rocsparse_operation_none,
                                           -1,
                                           3,
                                           descr,
                                           A.val,
                                           A.row_ptr,
                                           A.col_ind,
                                           1,
                                           info,
                                           &buffer_size),
                      rocsparse_status_invalid_size);

        // Invalid size: nnzb.
        UT_EXPECT_ROC(ut_bsrsv_buffer_size(handle,
                                           rocsparse_direction_row,
                                           rocsparse_operation_none,
                                           3,
                                           -1,
                                           descr,
                                           A.val,
                                           A.row_ptr,
                                           A.col_ind,
                                           1,
                                           info,
                                           &buffer_size),
                      rocsparse_status_invalid_size);

        // Null descriptor.
        UT_EXPECT_ROC(ut_bsrsv_buffer_size(handle,
                                           rocsparse_direction_row,
                                           rocsparse_operation_none,
                                           3,
                                           3,
                                           nullptr,
                                           A.val,
                                           A.row_ptr,
                                           A.col_ind,
                                           1,
                                           info,
                                           &buffer_size),
                      rocsparse_status_invalid_pointer);

        // Non-general matrix type -> not_implemented.
        {
            rocsparse_mat_descr descr_sym = nullptr;
            ASSERT_EQ(rocsparse_create_mat_descr(&descr_sym), rocsparse_status_success);
            ASSERT_EQ(rocsparse_set_mat_type(descr_sym, rocsparse_matrix_type_symmetric),
                      rocsparse_status_success);
            UT_EXPECT_ROC(ut_bsrsv_buffer_size(handle,
                                               rocsparse_direction_row,
                                               rocsparse_operation_none,
                                               3,
                                               3,
                                               descr_sym,
                                               A.val,
                                               A.row_ptr,
                                               A.col_ind,
                                               1,
                                               info,
                                               &buffer_size),
                          rocsparse_status_not_implemented);
            EXPECT_EQ(rocsparse_destroy_mat_descr(descr_sym), rocsparse_status_success);
        }

        // Unsorted storage mode -> requires_sorted_storage.
        {
            rocsparse_mat_descr descr_uns = nullptr;
            ASSERT_EQ(rocsparse_create_mat_descr(&descr_uns), rocsparse_status_success);
            ASSERT_EQ(rocsparse_set_mat_storage_mode(descr_uns, rocsparse_storage_mode_unsorted),
                      rocsparse_status_success);
            UT_EXPECT_ROC(ut_bsrsv_buffer_size(handle,
                                               rocsparse_direction_row,
                                               rocsparse_operation_none,
                                               3,
                                               3,
                                               descr_uns,
                                               A.val,
                                               A.row_ptr,
                                               A.col_ind,
                                               1,
                                               info,
                                               &buffer_size),
                          rocsparse_status_requires_sorted_storage);
            EXPECT_EQ(rocsparse_destroy_mat_descr(descr_uns), rocsparse_status_success);
        }

        // Null bsr_val with nnzb > 0.
        UT_EXPECT_ROC(ut_bsrsv_buffer_size(handle,
                                           rocsparse_direction_row,
                                           rocsparse_operation_none,
                                           3,
                                           3,
                                           descr,
                                           static_cast<const T*>(nullptr),
                                           A.row_ptr,
                                           A.col_ind,
                                           1,
                                           info,
                                           &buffer_size),
                      rocsparse_status_invalid_pointer);

        // Null bsr_row_ptr with mb > 0.
        UT_EXPECT_ROC(ut_bsrsv_buffer_size(handle,
                                           rocsparse_direction_row,
                                           rocsparse_operation_none,
                                           3,
                                           3,
                                           descr,
                                           A.val,
                                           static_cast<const rocsparse_int*>(nullptr),
                                           A.col_ind,
                                           1,
                                           info,
                                           &buffer_size),
                      rocsparse_status_invalid_pointer);

        // Null bsr_col_ind with nnzb > 0.
        UT_EXPECT_ROC(ut_bsrsv_buffer_size(handle,
                                           rocsparse_direction_row,
                                           rocsparse_operation_none,
                                           3,
                                           3,
                                           descr,
                                           A.val,
                                           A.row_ptr,
                                           static_cast<const rocsparse_int*>(nullptr),
                                           1,
                                           info,
                                           &buffer_size),
                      rocsparse_status_invalid_pointer);

        // Invalid block_dim.
        UT_EXPECT_ROC(ut_bsrsv_buffer_size(handle,
                                           rocsparse_direction_row,
                                           rocsparse_operation_none,
                                           3,
                                           3,
                                           descr,
                                           A.val,
                                           A.row_ptr,
                                           A.col_ind,
                                           0,
                                           info,
                                           &buffer_size),
                      rocsparse_status_invalid_size);

        // Null info.
        UT_EXPECT_ROC(ut_bsrsv_buffer_size(handle,
                                           rocsparse_direction_row,
                                           rocsparse_operation_none,
                                           3,
                                           3,
                                           descr,
                                           A.val,
                                           A.row_ptr,
                                           A.col_ind,
                                           1,
                                           nullptr,
                                           &buffer_size),
                      rocsparse_status_invalid_pointer);

        // Null buffer_size.
        UT_EXPECT_ROC(ut_bsrsv_buffer_size(handle,
                                           rocsparse_direction_row,
                                           rocsparse_operation_none,
                                           3,
                                           3,
                                           descr,
                                           A.val,
                                           A.row_ptr,
                                           A.col_ind,
                                           1,
                                           info,
                                           static_cast<size_t*>(nullptr)),
                      rocsparse_status_invalid_pointer);

        EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
    }
} // namespace

// ===========================================================================
// gtest suites (one fixture per level-2 routine, prefix "Level2").
// ===========================================================================
class Level2Csrmv : public HandleTest
{
};
class Level2Coomv : public HandleTest
{
};
class Level2Ellmv : public HandleTest
{
};
class Level2Bsrmv : public HandleTest
{
};
class Level2Gebsrmv : public HandleTest
{
};
class Level2Gemvi : public HandleTest
{
};
class Level2Csrsv : public HandleTest
{
};
class Level2Bsrsv : public HandleTest
{
};
class Level2Hybmv : public HandleTest
{
};
class Level2GebsrmvBlockDim : public HandleTest
{
};
class Level2BsrsvBufferSize : public HandleTest
{
};

#define UT_L2_ALL_PRECISIONS(FIXTURE, FN)     \
    TEST_F(FIXTURE, all_precisions)           \
    {                                         \
        FN<float>(handle);                    \
        FN<double>(handle);                   \
        FN<rocsparse_float_complex>(handle);  \
        FN<rocsparse_double_complex>(handle); \
    }

UT_L2_ALL_PRECISIONS(Level2Csrmv, run_csrmv)
UT_L2_ALL_PRECISIONS(Level2Coomv, run_coomv)
UT_L2_ALL_PRECISIONS(Level2Ellmv, run_ellmv)
UT_L2_ALL_PRECISIONS(Level2Bsrmv, run_bsrmv)
UT_L2_ALL_PRECISIONS(Level2Gebsrmv, run_gebsrmv)
UT_L2_ALL_PRECISIONS(Level2Gemvi, run_gemvi)
UT_L2_ALL_PRECISIONS(Level2Csrsv, run_csrsv)
UT_L2_ALL_PRECISIONS(Level2Bsrsv, run_bsrsv)
UT_L2_ALL_PRECISIONS(Level2Hybmv, run_hybmv)

#define UT_L2_GEBSRMV_BLOCK_DIM(FIXTURE, FN)                                                 \
    TEST_F(FIXTURE, all_precisions)                                                          \
    {                                                                                        \
        for(rocsparse_direction dir : {rocsparse_direction_row, rocsparse_direction_column}) \
        {                                                                                    \
            FN<float>(handle, dir);                                                          \
            FN<double>(handle, dir);                                                         \
            FN<rocsparse_float_complex>(handle, dir);                                        \
            FN<rocsparse_double_complex>(handle, dir);                                       \
        }                                                                                    \
    }

UT_L2_GEBSRMV_BLOCK_DIM(Level2GebsrmvBlockDim, run_gebsrmv_block_dim)
UT_L2_ALL_PRECISIONS(Level2BsrsvBufferSize, run_bsrsv_buffer_size)

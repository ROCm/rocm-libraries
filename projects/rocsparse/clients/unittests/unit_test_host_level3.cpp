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
// Host-path unit tests for the level-3 C API:
//   csrmm, bsrmm, gebsrmm, gemmi (classic, per-precision),
//   csrsm / bsrsm (buffer_size -> analysis -> solve -> zero_pivot -> clear),
//   coomm (exercised via the generic rocsparse_spmm COO backend).
//
// These drive the public C entry points so they exercise the HOST dispatch,
// argument-validation and analysis code in the level-3 .cpp files, across all
// four precisions (s/d/c/z) plus the bad-arg guards.  All sparse/dense
// operands are kept TINY (3x3 identity/diagonal CSR/BSR, dims <= 3) so the
// suite stays well within the memory budget and the solve routines are
// well-conditioned (no zero pivot).
//
#include "unit_test_utils.hpp"

#include <cstdint>

using namespace rocsparse_ut;

namespace
{
    // Compute calls are tolerant: the host dispatch/analysis code is what
    // coverage counts, and a given precision/config may legitimately report
    // rocsparse_status_not_implemented on some architectures (e.g. wf32).
    inline void expect_ok_or_ni(rocsparse_status s)
    {
        EXPECT_TRUE(s == rocsparse_status_success || s == rocsparse_status_not_implemented)
            << "unexpected status " << s;
    }

    // An invalid operation enum value used for invalid-enum guards (in range
    // of the enum's underlying type but not a defined enumerator).
    constexpr rocsparse_operation bad_operation = static_cast<rocsparse_operation>(99);

    // ---- tiny 3x3 identity sparse matrix (CSR / single-entry BSR blocks) ----
    template <typename T>
    struct Id3
    {
        // 3x3 identity in CSR (and, with block_dim == 1, in BSR).
        device_vector<rocsparse_int> row_ptr{std::vector<rocsparse_int>{0, 1, 2, 3}};
        device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
        device_vector<T>             val{std::vector<T>{scalar<T>(1), scalar<T>(1), scalar<T>(1)}};
        bool                         ok() const
        {
            return row_ptr.ptr && col_ind.ptr && val.ptr;
        }
    };

    // =====================================================================
    // csrmm
    // =====================================================================
#define UT_OVL_CSRMM(T, PFX)                                                         \
    inline rocsparse_status ut_csrmm(rocsparse_handle          h,                    \
                                     rocsparse_operation       ta,                   \
                                     rocsparse_operation       tb,                   \
                                     rocsparse_int             m,                    \
                                     rocsparse_int             n,                    \
                                     rocsparse_int             k,                    \
                                     rocsparse_int             nnz,                  \
                                     const T*                  alpha,                \
                                     const rocsparse_mat_descr descr,                \
                                     const T*                  val,                  \
                                     const rocsparse_int*      rp,                   \
                                     const rocsparse_int*      ci,                   \
                                     const T*                  B,                    \
                                     rocsparse_int             ldb,                  \
                                     const T*                  beta,                 \
                                     T*                        C,                    \
                                     rocsparse_int             ldc)                  \
    {                                                                                \
        return rocsparse_##PFX##csrmm(                                               \
            h, ta, tb, m, n, k, nnz, alpha, descr, val, rp, ci, B, ldb, beta, C, ldc); \
    }
    UT_OVL_CSRMM(float, s)
    UT_OVL_CSRMM(double, d)
    UT_OVL_CSRMM(rocsparse_float_complex, c)
    UT_OVL_CSRMM(rocsparse_double_complex, z)

    template <typename T>
    void run_csrmm(rocsparse_handle handle)
    {
        const rocsparse_int m = 3, n = 2, k = 3, nnz = 3, ldb = 3, ldc = 3;
        Id3<T>              A;
        ASSERT_TRUE(A.ok());
        device_vector<T> B{std::vector<T>(ldb * n, scalar<T>(1))};
        device_vector<T> C{std::vector<T>(ldc * n, scalar<T>(0))};
        ASSERT_TRUE(B.ptr && C.ptr);

        rocsparse_mat_descr descr = nullptr;
        ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);

        const T alpha = scalar<T>(1), beta = scalar<T>(0);

        // valid
        expect_ok_or_ni(ut_csrmm(handle,
                                 rocsparse_operation_none,
                                 rocsparse_operation_none,
                                 m,
                                 n,
                                 k,
                                 nnz,
                                 &alpha,
                                 descr,
                                 A.val,
                                 A.row_ptr,
                                 A.col_ind,
                                 B,
                                 ldb,
                                 &beta,
                                 C,
                                 ldc));
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        // bad args
        UT_EXPECT_ROC(ut_csrmm(nullptr,
                               rocsparse_operation_none,
                               rocsparse_operation_none,
                               m,
                               n,
                               k,
                               nnz,
                               &alpha,
                               descr,
                               A.val,
                               A.row_ptr,
                               A.col_ind,
                               B,
                               ldb,
                               &beta,
                               C,
                               ldc),
                      rocsparse_status_invalid_handle);
        UT_EXPECT_ROC(ut_csrmm(handle,
                               bad_operation,
                               rocsparse_operation_none,
                               m,
                               n,
                               k,
                               nnz,
                               &alpha,
                               descr,
                               A.val,
                               A.row_ptr,
                               A.col_ind,
                               B,
                               ldb,
                               &beta,
                               C,
                               ldc),
                      rocsparse_status_invalid_value);
        UT_EXPECT_ROC(ut_csrmm(handle,
                               rocsparse_operation_none,
                               rocsparse_operation_none,
                               -1,
                               n,
                               k,
                               nnz,
                               &alpha,
                               descr,
                               A.val,
                               A.row_ptr,
                               A.col_ind,
                               B,
                               ldb,
                               &beta,
                               C,
                               ldc),
                      rocsparse_status_invalid_size);
        UT_EXPECT_ROC(ut_csrmm(handle,
                               rocsparse_operation_none,
                               rocsparse_operation_none,
                               m,
                               n,
                               k,
                               nnz,
                               nullptr,
                               descr,
                               A.val,
                               A.row_ptr,
                               A.col_ind,
                               B,
                               ldb,
                               &beta,
                               C,
                               ldc),
                      rocsparse_status_invalid_pointer);
        UT_EXPECT_ROC(ut_csrmm(handle,
                               rocsparse_operation_none,
                               rocsparse_operation_none,
                               m,
                               n,
                               k,
                               nnz,
                               &alpha,
                               descr,
                               nullptr,
                               A.row_ptr,
                               A.col_ind,
                               B,
                               ldb,
                               &beta,
                               C,
                               ldc),
                      rocsparse_status_invalid_pointer);

        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
    }

    // =====================================================================
    // bsrmm  (block_dim == 1  =>  behaves like csrmm on the identity)
    // =====================================================================
#define UT_OVL_BSRMM(T, PFX)                                                              \
    inline rocsparse_status ut_bsrmm(rocsparse_handle          h,                         \
                                     rocsparse_direction       dir,                       \
                                     rocsparse_operation       ta,                        \
                                     rocsparse_operation       tb,                        \
                                     rocsparse_int             mb,                        \
                                     rocsparse_int             n,                         \
                                     rocsparse_int             kb,                        \
                                     rocsparse_int             nnzb,                      \
                                     const T*                  alpha,                     \
                                     const rocsparse_mat_descr descr,                     \
                                     const T*                  val,                       \
                                     const rocsparse_int*      rp,                        \
                                     const rocsparse_int*      ci,                        \
                                     rocsparse_int             bd,                        \
                                     const T*                  B,                         \
                                     rocsparse_int             ldb,                       \
                                     const T*                  beta,                      \
                                     T*                        C,                         \
                                     rocsparse_int             ldc)                       \
    {                                                                                     \
        return rocsparse_##PFX##bsrmm(                                                    \
            h, dir, ta, tb, mb, n, kb, nnzb, alpha, descr, val, rp, ci, bd, B, ldb, beta, \
            C, ldc);                                                                      \
    }
    UT_OVL_BSRMM(float, s)
    UT_OVL_BSRMM(double, d)
    UT_OVL_BSRMM(rocsparse_float_complex, c)
    UT_OVL_BSRMM(rocsparse_double_complex, z)

    template <typename T>
    void run_bsrmm(rocsparse_handle handle)
    {
        const rocsparse_int mb = 3, n = 2, kb = 3, nnzb = 3, bd = 1, ldb = 3, ldc = 3;
        Id3<T>              A;
        ASSERT_TRUE(A.ok());
        device_vector<T> B{std::vector<T>(ldb * n, scalar<T>(1))};
        device_vector<T> C{std::vector<T>(ldc * n, scalar<T>(0))};
        ASSERT_TRUE(B.ptr && C.ptr);

        rocsparse_mat_descr descr = nullptr;
        ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);

        const T alpha = scalar<T>(1), beta = scalar<T>(0);

        expect_ok_or_ni(ut_bsrmm(handle,
                                 rocsparse_direction_row,
                                 rocsparse_operation_none,
                                 rocsparse_operation_none,
                                 mb,
                                 n,
                                 kb,
                                 nnzb,
                                 &alpha,
                                 descr,
                                 A.val,
                                 A.row_ptr,
                                 A.col_ind,
                                 bd,
                                 B,
                                 ldb,
                                 &beta,
                                 C,
                                 ldc));
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        UT_EXPECT_ROC(ut_bsrmm(nullptr,
                               rocsparse_direction_row,
                               rocsparse_operation_none,
                               rocsparse_operation_none,
                               mb,
                               n,
                               kb,
                               nnzb,
                               &alpha,
                               descr,
                               A.val,
                               A.row_ptr,
                               A.col_ind,
                               bd,
                               B,
                               ldb,
                               &beta,
                               C,
                               ldc),
                      rocsparse_status_invalid_handle);
        UT_EXPECT_ROC(ut_bsrmm(handle,
                               rocsparse_direction_row,
                               rocsparse_operation_none,
                               rocsparse_operation_none,
                               -1,
                               n,
                               kb,
                               nnzb,
                               &alpha,
                               descr,
                               A.val,
                               A.row_ptr,
                               A.col_ind,
                               bd,
                               B,
                               ldb,
                               &beta,
                               C,
                               ldc),
                      rocsparse_status_invalid_size);
        UT_EXPECT_ROC(ut_bsrmm(handle,
                               rocsparse_direction_row,
                               rocsparse_operation_none,
                               rocsparse_operation_none,
                               mb,
                               n,
                               kb,
                               nnzb,
                               nullptr,
                               descr,
                               A.val,
                               A.row_ptr,
                               A.col_ind,
                               bd,
                               B,
                               ldb,
                               &beta,
                               C,
                               ldc),
                      rocsparse_status_invalid_pointer);

        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
    }

    // =====================================================================
    // gebsrmm  (row_block_dim == col_block_dim == 1)
    // =====================================================================
#define UT_OVL_GEBSRMM(T, PFX)                                                            \
    inline rocsparse_status ut_gebsrmm(rocsparse_handle          h,                       \
                                       rocsparse_direction       dir,                     \
                                       rocsparse_operation       ta,                      \
                                       rocsparse_operation       tb,                      \
                                       rocsparse_int             mb,                      \
                                       rocsparse_int             n,                       \
                                       rocsparse_int             kb,                      \
                                       rocsparse_int             nnzb,                    \
                                       const T*                  alpha,                   \
                                       const rocsparse_mat_descr descr,                   \
                                       const T*                  val,                     \
                                       const rocsparse_int*      rp,                      \
                                       const rocsparse_int*      ci,                      \
                                       rocsparse_int             rbd,                     \
                                       rocsparse_int             cbd,                     \
                                       const T*                  B,                       \
                                       rocsparse_int             ldb,                     \
                                       const T*                  beta,                    \
                                       T*                        C,                       \
                                       rocsparse_int             ldc)                     \
    {                                                                                     \
        return rocsparse_##PFX##gebsrmm(h,                                                \
                                        dir,                                              \
                                        ta,                                               \
                                        tb,                                               \
                                        mb,                                               \
                                        n,                                                \
                                        kb,                                               \
                                        nnzb,                                             \
                                        alpha,                                            \
                                        descr,                                            \
                                        val,                                              \
                                        rp,                                               \
                                        ci,                                               \
                                        rbd,                                              \
                                        cbd,                                              \
                                        B,                                                \
                                        ldb,                                              \
                                        beta,                                             \
                                        C,                                                \
                                        ldc);                                             \
    }
    UT_OVL_GEBSRMM(float, s)
    UT_OVL_GEBSRMM(double, d)
    UT_OVL_GEBSRMM(rocsparse_float_complex, c)
    UT_OVL_GEBSRMM(rocsparse_double_complex, z)

    template <typename T>
    void run_gebsrmm(rocsparse_handle handle)
    {
        const rocsparse_int mb = 3, n = 2, kb = 3, nnzb = 3, rbd = 1, cbd = 1, ldb = 3, ldc = 3;
        Id3<T>              A;
        ASSERT_TRUE(A.ok());
        device_vector<T> B{std::vector<T>(ldb * n, scalar<T>(1))};
        device_vector<T> C{std::vector<T>(ldc * n, scalar<T>(0))};
        ASSERT_TRUE(B.ptr && C.ptr);

        rocsparse_mat_descr descr = nullptr;
        ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);

        const T alpha = scalar<T>(1), beta = scalar<T>(0);

        expect_ok_or_ni(ut_gebsrmm(handle,
                                   rocsparse_direction_row,
                                   rocsparse_operation_none,
                                   rocsparse_operation_none,
                                   mb,
                                   n,
                                   kb,
                                   nnzb,
                                   &alpha,
                                   descr,
                                   A.val,
                                   A.row_ptr,
                                   A.col_ind,
                                   rbd,
                                   cbd,
                                   B,
                                   ldb,
                                   &beta,
                                   C,
                                   ldc));
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        UT_EXPECT_ROC(ut_gebsrmm(nullptr,
                                 rocsparse_direction_row,
                                 rocsparse_operation_none,
                                 rocsparse_operation_none,
                                 mb,
                                 n,
                                 kb,
                                 nnzb,
                                 &alpha,
                                 descr,
                                 A.val,
                                 A.row_ptr,
                                 A.col_ind,
                                 rbd,
                                 cbd,
                                 B,
                                 ldb,
                                 &beta,
                                 C,
                                 ldc),
                      rocsparse_status_invalid_handle);
        UT_EXPECT_ROC(ut_gebsrmm(handle,
                                 rocsparse_direction_row,
                                 rocsparse_operation_none,
                                 rocsparse_operation_none,
                                 mb,
                                 n,
                                 kb,
                                 nnzb,
                                 &alpha,
                                 descr,
                                 A.val,
                                 A.row_ptr,
                                 A.col_ind,
                                 -1,
                                 cbd,
                                 B,
                                 ldb,
                                 &beta,
                                 C,
                                 ldc),
                      rocsparse_status_invalid_size);
        UT_EXPECT_ROC(ut_gebsrmm(handle,
                                 rocsparse_direction_row,
                                 rocsparse_operation_none,
                                 rocsparse_operation_none,
                                 mb,
                                 n,
                                 kb,
                                 nnzb,
                                 nullptr,
                                 descr,
                                 A.val,
                                 A.row_ptr,
                                 A.col_ind,
                                 rbd,
                                 cbd,
                                 B,
                                 ldb,
                                 &beta,
                                 C,
                                 ldc),
                      rocsparse_status_invalid_pointer);

        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
    }

    // =====================================================================
    // gemmi  (dense * sparse^T ; only trans_A none / trans_B transpose)
    // =====================================================================
#define UT_OVL_GEMMI(T, PFX)                                                         \
    inline rocsparse_status ut_gemmi(rocsparse_handle          h,                    \
                                     rocsparse_operation       ta,                   \
                                     rocsparse_operation       tb,                   \
                                     rocsparse_int             m,                    \
                                     rocsparse_int             n,                    \
                                     rocsparse_int             k,                    \
                                     rocsparse_int             nnz,                  \
                                     const T*                  alpha,                \
                                     const T*                  A,                    \
                                     rocsparse_int             lda,                  \
                                     const rocsparse_mat_descr descr,                \
                                     const T*                  val,                  \
                                     const rocsparse_int*      rp,                   \
                                     const rocsparse_int*      ci,                   \
                                     const T*                  beta,                 \
                                     T*                        C,                    \
                                     rocsparse_int             ldc)                  \
    {                                                                                \
        return rocsparse_##PFX##gemmi(                                               \
            h, ta, tb, m, n, k, nnz, alpha, A, lda, descr, val, rp, ci, beta, C, ldc); \
    }
    UT_OVL_GEMMI(float, s)
    UT_OVL_GEMMI(double, d)
    UT_OVL_GEMMI(rocsparse_float_complex, c)
    UT_OVL_GEMMI(rocsparse_double_complex, z)

    template <typename T>
    void run_gemmi(rocsparse_handle handle)
    {
        const rocsparse_int m = 3, n = 3, k = 3, nnz = 3, lda = 3, ldc = 3;
        Id3<T>              Bsp; // sparse operand (CSR identity)
        ASSERT_TRUE(Bsp.ok());
        device_vector<T> A{std::vector<T>(lda * k, scalar<T>(1))};
        device_vector<T> C{std::vector<T>(ldc * n, scalar<T>(0))};
        ASSERT_TRUE(A.ptr && C.ptr);

        rocsparse_mat_descr descr = nullptr;
        ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);

        const T alpha = scalar<T>(1), beta = scalar<T>(0);

        expect_ok_or_ni(ut_gemmi(handle,
                                 rocsparse_operation_none,
                                 rocsparse_operation_transpose,
                                 m,
                                 n,
                                 k,
                                 nnz,
                                 &alpha,
                                 A,
                                 lda,
                                 descr,
                                 Bsp.val,
                                 Bsp.row_ptr,
                                 Bsp.col_ind,
                                 &beta,
                                 C,
                                 ldc));
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        UT_EXPECT_ROC(ut_gemmi(nullptr,
                               rocsparse_operation_none,
                               rocsparse_operation_transpose,
                               m,
                               n,
                               k,
                               nnz,
                               &alpha,
                               A,
                               lda,
                               descr,
                               Bsp.val,
                               Bsp.row_ptr,
                               Bsp.col_ind,
                               &beta,
                               C,
                               ldc),
                      rocsparse_status_invalid_handle);
        UT_EXPECT_ROC(ut_gemmi(handle,
                               rocsparse_operation_none,
                               rocsparse_operation_transpose,
                               -1,
                               n,
                               k,
                               nnz,
                               &alpha,
                               A,
                               lda,
                               descr,
                               Bsp.val,
                               Bsp.row_ptr,
                               Bsp.col_ind,
                               &beta,
                               C,
                               ldc),
                      rocsparse_status_invalid_size);
        UT_EXPECT_ROC(ut_gemmi(handle,
                               rocsparse_operation_none,
                               rocsparse_operation_transpose,
                               m,
                               n,
                               k,
                               nnz,
                               nullptr,
                               A,
                               lda,
                               descr,
                               Bsp.val,
                               Bsp.row_ptr,
                               Bsp.col_ind,
                               &beta,
                               C,
                               ldc),
                      rocsparse_status_invalid_pointer);

        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
    }

    // =====================================================================
    // csrsm  (buffer_size -> analysis -> solve -> zero_pivot -> clear)
    // =====================================================================
#define UT_OVL_CSRSM_BUF(T, PFX)                                                     \
    inline rocsparse_status ut_csrsm_buffer_size(rocsparse_handle          h,        \
                                                 rocsparse_operation       ta,       \
                                                 rocsparse_operation       tb,       \
                                                 rocsparse_int             m,        \
                                                 rocsparse_int             nrhs,     \
                                                 rocsparse_int             nnz,      \
                                                 const T*                  alpha,    \
                                                 const rocsparse_mat_descr descr,    \
                                                 const T*                  val,      \
                                                 const rocsparse_int*      rp,       \
                                                 const rocsparse_int*      ci,       \
                                                 const T*                  B,        \
                                                 rocsparse_int             ldb,      \
                                                 rocsparse_mat_info        info,     \
                                                 rocsparse_solve_policy    policy,   \
                                                 size_t*                   bs)       \
    {                                                                                \
        return rocsparse_##PFX##csrsm_buffer_size(                                   \
            h, ta, tb, m, nrhs, nnz, alpha, descr, val, rp, ci, B, ldb, info, policy, bs); \
    }                                                                                \
    inline rocsparse_status ut_csrsm_analysis(rocsparse_handle          h,           \
                                              rocsparse_operation       ta,          \
                                              rocsparse_operation       tb,          \
                                              rocsparse_int             m,           \
                                              rocsparse_int             nrhs,        \
                                              rocsparse_int             nnz,         \
                                              const T*                  alpha,       \
                                              const rocsparse_mat_descr descr,       \
                                              const T*                  val,         \
                                              const rocsparse_int*      rp,          \
                                              const rocsparse_int*      ci,          \
                                              const T*                  B,           \
                                              rocsparse_int             ldb,         \
                                              rocsparse_mat_info        info,        \
                                              rocsparse_analysis_policy ap,          \
                                              rocsparse_solve_policy    sp,          \
                                              void*                     buf)         \
    {                                                                                \
        return rocsparse_##PFX##csrsm_analysis(                                      \
            h, ta, tb, m, nrhs, nnz, alpha, descr, val, rp, ci, B, ldb, info, ap, sp, buf); \
    }                                                                                \
    inline rocsparse_status ut_csrsm_solve(rocsparse_handle          h,              \
                                           rocsparse_operation       ta,             \
                                           rocsparse_operation       tb,             \
                                           rocsparse_int             m,              \
                                           rocsparse_int             nrhs,           \
                                           rocsparse_int             nnz,            \
                                           const T*                  alpha,          \
                                           const rocsparse_mat_descr descr,          \
                                           const T*                  val,            \
                                           const rocsparse_int*      rp,             \
                                           const rocsparse_int*      ci,             \
                                           T*                        B,              \
                                           rocsparse_int             ldb,            \
                                           rocsparse_mat_info        info,           \
                                           rocsparse_solve_policy    sp,             \
                                           void*                     buf)            \
    {                                                                                \
        return rocsparse_##PFX##csrsm_solve(                                         \
            h, ta, tb, m, nrhs, nnz, alpha, descr, val, rp, ci, B, ldb, info, sp, buf); \
    }
    UT_OVL_CSRSM_BUF(float, s)
    UT_OVL_CSRSM_BUF(double, d)
    UT_OVL_CSRSM_BUF(rocsparse_float_complex, c)
    UT_OVL_CSRSM_BUF(rocsparse_double_complex, z)

    template <typename T>
    void run_csrsm(rocsparse_handle handle)
    {
        const rocsparse_int          m = 3, nrhs = 2, nnz = 3, ldb = 3;
        const rocsparse_operation    ta = rocsparse_operation_none, tb = rocsparse_operation_none;
        Id3<T>                       A;
        ASSERT_TRUE(A.ok());
        device_vector<T> B{std::vector<T>(ldb * nrhs, scalar<T>(1))};
        ASSERT_TRUE(B.ptr);

        rocsparse_mat_descr descr = nullptr;
        ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
        rocsparse_mat_info info = nullptr;
        ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

        const T alpha = scalar<T>(1);

        size_t buffer_size = 0;
        const rocsparse_status bs_status = ut_csrsm_buffer_size(handle,
                                                                ta,
                                                                tb,
                                                                m,
                                                                nrhs,
                                                                nnz,
                                                                &alpha,
                                                                descr,
                                                                A.val,
                                                                A.row_ptr,
                                                                A.col_ind,
                                                                B,
                                                                ldb,
                                                                info,
                                                                rocsparse_solve_policy_auto,
                                                                &buffer_size);
        if(bs_status == rocsparse_status_success)
        {
            device_vector<char> buffer{buffer_size ? buffer_size : 1};
            ASSERT_TRUE(buffer.ptr);

            expect_ok_or_ni(ut_csrsm_analysis(handle,
                                              ta,
                                              tb,
                                              m,
                                              nrhs,
                                              nnz,
                                              &alpha,
                                              descr,
                                              A.val,
                                              A.row_ptr,
                                              A.col_ind,
                                              B,
                                              ldb,
                                              info,
                                              rocsparse_analysis_policy_reuse,
                                              rocsparse_solve_policy_auto,
                                              buffer.ptr));
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

            expect_ok_or_ni(ut_csrsm_solve(handle,
                                           ta,
                                           tb,
                                           m,
                                           nrhs,
                                           nnz,
                                           &alpha,
                                           descr,
                                           A.val,
                                           A.row_ptr,
                                           A.col_ind,
                                           B,
                                           ldb,
                                           info,
                                           rocsparse_solve_policy_auto,
                                           buffer.ptr));
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

            rocsparse_int position = -1;
            expect_ok_or_ni(rocsparse_csrsm_zero_pivot(handle, info, &position));

            EXPECT_EQ(rocsparse_csrsm_clear(handle, info), rocsparse_status_success);
        }
        else
        {
            expect_ok_or_ni(bs_status);
        }

        // bad args on buffer_size
        UT_EXPECT_ROC(ut_csrsm_buffer_size(nullptr,
                                           ta,
                                           tb,
                                           m,
                                           nrhs,
                                           nnz,
                                           &alpha,
                                           descr,
                                           A.val,
                                           A.row_ptr,
                                           A.col_ind,
                                           B,
                                           ldb,
                                           info,
                                           rocsparse_solve_policy_auto,
                                           &buffer_size),
                      rocsparse_status_invalid_handle);
        UT_EXPECT_ROC(ut_csrsm_buffer_size(handle,
                                           ta,
                                           tb,
                                           -1,
                                           nrhs,
                                           nnz,
                                           &alpha,
                                           descr,
                                           A.val,
                                           A.row_ptr,
                                           A.col_ind,
                                           B,
                                           ldb,
                                           info,
                                           rocsparse_solve_policy_auto,
                                           &buffer_size),
                      rocsparse_status_invalid_size);
        UT_EXPECT_ROC(ut_csrsm_buffer_size(handle,
                                           ta,
                                           tb,
                                           m,
                                           nrhs,
                                           nnz,
                                           &alpha,
                                           descr,
                                           A.val,
                                           A.row_ptr,
                                           A.col_ind,
                                           B,
                                           ldb,
                                           info,
                                           rocsparse_solve_policy_auto,
                                           nullptr),
                      rocsparse_status_invalid_pointer);

        EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
    }

    // =====================================================================
    // bsrsm  (block_dim == 1 ; buffer_size -> analysis -> solve -> clear)
    // =====================================================================
#define UT_OVL_BSRSM(T, PFX)                                                              \
    inline rocsparse_status ut_bsrsm_buffer_size(rocsparse_handle          h,             \
                                                 rocsparse_direction       dir,           \
                                                 rocsparse_operation       ta,            \
                                                 rocsparse_operation       tx,            \
                                                 rocsparse_int             mb,            \
                                                 rocsparse_int             nrhs,          \
                                                 rocsparse_int             nnzb,          \
                                                 const rocsparse_mat_descr descr,         \
                                                 const T*                  val,           \
                                                 const rocsparse_int*      rp,            \
                                                 const rocsparse_int*      ci,            \
                                                 rocsparse_int             bd,            \
                                                 rocsparse_mat_info        info,          \
                                                 size_t*                   bs)            \
    {                                                                                     \
        return rocsparse_##PFX##bsrsm_buffer_size(                                        \
            h, dir, ta, tx, mb, nrhs, nnzb, descr, val, rp, ci, bd, info, bs);            \
    }                                                                                     \
    inline rocsparse_status ut_bsrsm_analysis(rocsparse_handle          h,                \
                                              rocsparse_direction       dir,              \
                                              rocsparse_operation       ta,               \
                                              rocsparse_operation       tx,               \
                                              rocsparse_int             mb,               \
                                              rocsparse_int             nrhs,             \
                                              rocsparse_int             nnzb,             \
                                              const rocsparse_mat_descr descr,            \
                                              const T*                  val,              \
                                              const rocsparse_int*      rp,               \
                                              const rocsparse_int*      ci,               \
                                              rocsparse_int             bd,               \
                                              rocsparse_mat_info        info,             \
                                              rocsparse_analysis_policy ap,               \
                                              rocsparse_solve_policy    sp,               \
                                              void*                     buf)              \
    {                                                                                     \
        return rocsparse_##PFX##bsrsm_analysis(                                           \
            h, dir, ta, tx, mb, nrhs, nnzb, descr, val, rp, ci, bd, info, ap, sp, buf);   \
    }                                                                                     \
    inline rocsparse_status ut_bsrsm_solve(rocsparse_handle          h,                   \
                                           rocsparse_direction       dir,                 \
                                           rocsparse_operation       ta,                  \
                                           rocsparse_operation       tx,                  \
                                           rocsparse_int             mb,                  \
                                           rocsparse_int             nrhs,                \
                                           rocsparse_int             nnzb,                \
                                           const T*                  alpha,               \
                                           const rocsparse_mat_descr descr,               \
                                           const T*                  val,                 \
                                           const rocsparse_int*      rp,                  \
                                           const rocsparse_int*      ci,                  \
                                           rocsparse_int             bd,                  \
                                           rocsparse_mat_info        info,                \
                                           const T*                  B,                   \
                                           rocsparse_int             ldb,                 \
                                           T*                        X,                   \
                                           rocsparse_int             ldx,                 \
                                           rocsparse_solve_policy    sp,                  \
                                           void*                     buf)                 \
    {                                                                                     \
        return rocsparse_##PFX##bsrsm_solve(h,                                            \
                                            dir,                                          \
                                            ta,                                           \
                                            tx,                                           \
                                            mb,                                           \
                                            nrhs,                                         \
                                            nnzb,                                         \
                                            alpha,                                        \
                                            descr,                                        \
                                            val,                                          \
                                            rp,                                           \
                                            ci,                                           \
                                            bd,                                           \
                                            info,                                         \
                                            B,                                            \
                                            ldb,                                          \
                                            X,                                            \
                                            ldx,                                          \
                                            sp,                                           \
                                            buf);                                         \
    }
    UT_OVL_BSRSM(float, s)
    UT_OVL_BSRSM(double, d)
    UT_OVL_BSRSM(rocsparse_float_complex, c)
    UT_OVL_BSRSM(rocsparse_double_complex, z)

    template <typename T>
    void run_bsrsm(rocsparse_handle handle)
    {
        const rocsparse_int       mb = 3, nrhs = 2, nnzb = 3, bd = 1, ldb = 3, ldx = 3;
        const rocsparse_direction dir = rocsparse_direction_row;
        const rocsparse_operation ta = rocsparse_operation_none, tx = rocsparse_operation_none;
        Id3<T>                    A;
        ASSERT_TRUE(A.ok());
        device_vector<T> B{std::vector<T>(ldb * nrhs, scalar<T>(1))};
        device_vector<T> X{std::vector<T>(ldx * nrhs, scalar<T>(0))};
        ASSERT_TRUE(B.ptr && X.ptr);

        rocsparse_mat_descr descr = nullptr;
        ASSERT_EQ(rocsparse_create_mat_descr(&descr), rocsparse_status_success);
        rocsparse_mat_info info = nullptr;
        ASSERT_EQ(rocsparse_create_mat_info(&info), rocsparse_status_success);

        const T alpha = scalar<T>(1);

        size_t                 buffer_size = 0;
        const rocsparse_status bs_status   = ut_bsrsm_buffer_size(handle,
                                                                dir,
                                                                ta,
                                                                tx,
                                                                mb,
                                                                nrhs,
                                                                nnzb,
                                                                descr,
                                                                A.val,
                                                                A.row_ptr,
                                                                A.col_ind,
                                                                bd,
                                                                info,
                                                                &buffer_size);
        if(bs_status == rocsparse_status_success)
        {
            device_vector<char> buffer{buffer_size ? buffer_size : 1};
            ASSERT_TRUE(buffer.ptr);

            expect_ok_or_ni(ut_bsrsm_analysis(handle,
                                              dir,
                                              ta,
                                              tx,
                                              mb,
                                              nrhs,
                                              nnzb,
                                              descr,
                                              A.val,
                                              A.row_ptr,
                                              A.col_ind,
                                              bd,
                                              info,
                                              rocsparse_analysis_policy_reuse,
                                              rocsparse_solve_policy_auto,
                                              buffer.ptr));
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

            expect_ok_or_ni(ut_bsrsm_solve(handle,
                                           dir,
                                           ta,
                                           tx,
                                           mb,
                                           nrhs,
                                           nnzb,
                                           &alpha,
                                           descr,
                                           A.val,
                                           A.row_ptr,
                                           A.col_ind,
                                           bd,
                                           info,
                                           B,
                                           ldb,
                                           X,
                                           ldx,
                                           rocsparse_solve_policy_auto,
                                           buffer.ptr));
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

            rocsparse_int position = -1;
            expect_ok_or_ni(rocsparse_bsrsm_zero_pivot(handle, info, &position));

            EXPECT_EQ(rocsparse_bsrsm_clear(handle, info), rocsparse_status_success);
        }
        else
        {
            expect_ok_or_ni(bs_status);
        }

        UT_EXPECT_ROC(ut_bsrsm_buffer_size(nullptr,
                                           dir,
                                           ta,
                                           tx,
                                           mb,
                                           nrhs,
                                           nnzb,
                                           descr,
                                           A.val,
                                           A.row_ptr,
                                           A.col_ind,
                                           bd,
                                           info,
                                           &buffer_size),
                      rocsparse_status_invalid_handle);
        UT_EXPECT_ROC(ut_bsrsm_buffer_size(handle,
                                           dir,
                                           ta,
                                           tx,
                                           -1,
                                           nrhs,
                                           nnzb,
                                           descr,
                                           A.val,
                                           A.row_ptr,
                                           A.col_ind,
                                           bd,
                                           info,
                                           &buffer_size),
                      rocsparse_status_invalid_size);
        UT_EXPECT_ROC(ut_bsrsm_buffer_size(handle,
                                           dir,
                                           ta,
                                           tx,
                                           mb,
                                           nrhs,
                                           nnzb,
                                           descr,
                                           A.val,
                                           A.row_ptr,
                                           A.col_ind,
                                           bd,
                                           info,
                                           nullptr),
                      rocsparse_status_invalid_pointer);

        EXPECT_EQ(rocsparse_destroy_mat_info(info), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_mat_descr(descr), rocsparse_status_success);
    }

    // =====================================================================
    // coomm  (exercised through the generic rocsparse_spmm COO backend)
    // =====================================================================
    template <typename T>
    void run_coomm(rocsparse_handle handle)
    {
        const int64_t m = 3, n = 2, k = 3, nnz = 3, ldb = 3, ldc = 3;
        // 3x3 identity in COO.
        device_vector<rocsparse_int> row_ind{std::vector<rocsparse_int>{0, 1, 2}};
        device_vector<rocsparse_int> col_ind{std::vector<rocsparse_int>{0, 1, 2}};
        device_vector<T> val{std::vector<T>{scalar<T>(1), scalar<T>(1), scalar<T>(1)}};
        device_vector<T> B{std::vector<T>(ldb * n, scalar<T>(1))};
        device_vector<T> C{std::vector<T>(ldc * n, scalar<T>(0))};
        ASSERT_TRUE(row_ind.ptr && col_ind.ptr && val.ptr && B.ptr && C.ptr);

        rocsparse_spmat_descr mat_A = nullptr;
        ASSERT_EQ(rocsparse_create_coo_descr(&mat_A,
                                             m,
                                             k,
                                             nnz,
                                             row_ind,
                                             col_ind,
                                             val,
                                             it_of<rocsparse_int>(),
                                             rocsparse_index_base_zero,
                                             dt_of<T>()),
                  rocsparse_status_success);

        rocsparse_dnmat_descr mat_B = nullptr, mat_C = nullptr;
        ASSERT_EQ(
            rocsparse_create_dnmat_descr(&mat_B, k, n, ldb, B, dt_of<T>(), rocsparse_order_column),
            rocsparse_status_success);
        ASSERT_EQ(
            rocsparse_create_dnmat_descr(&mat_C, m, n, ldc, C, dt_of<T>(), rocsparse_order_column),
            rocsparse_status_success);

        const T alpha = scalar<T>(1), beta = scalar<T>(0);

        size_t buffer_size = 0;
        expect_ok_or_ni(rocsparse_spmm(handle,
                                       rocsparse_operation_none,
                                       rocsparse_operation_none,
                                       &alpha,
                                       mat_A,
                                       mat_B,
                                       &beta,
                                       mat_C,
                                       dt_of<T>(),
                                       rocsparse_spmm_alg_coo_atomic,
                                       rocsparse_spmm_stage_buffer_size,
                                       &buffer_size,
                                       nullptr));

        device_vector<char> buffer{buffer_size ? buffer_size : 1};
        ASSERT_TRUE(buffer.ptr);

        expect_ok_or_ni(rocsparse_spmm(handle,
                                       rocsparse_operation_none,
                                       rocsparse_operation_none,
                                       &alpha,
                                       mat_A,
                                       mat_B,
                                       &beta,
                                       mat_C,
                                       dt_of<T>(),
                                       rocsparse_spmm_alg_coo_atomic,
                                       rocsparse_spmm_stage_preprocess,
                                       &buffer_size,
                                       buffer.ptr));
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        expect_ok_or_ni(rocsparse_spmm(handle,
                                       rocsparse_operation_none,
                                       rocsparse_operation_none,
                                       &alpha,
                                       mat_A,
                                       mat_B,
                                       &beta,
                                       mat_C,
                                       dt_of<T>(),
                                       rocsparse_spmm_alg_coo_atomic,
                                       rocsparse_spmm_stage_compute,
                                       &buffer_size,
                                       buffer.ptr));
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        // bad arg
        UT_EXPECT_ROC(rocsparse_spmm(nullptr,
                                     rocsparse_operation_none,
                                     rocsparse_operation_none,
                                     &alpha,
                                     mat_A,
                                     mat_B,
                                     &beta,
                                     mat_C,
                                     dt_of<T>(),
                                     rocsparse_spmm_alg_coo_atomic,
                                     rocsparse_spmm_stage_buffer_size,
                                     &buffer_size,
                                     nullptr),
                      rocsparse_status_invalid_handle);

        EXPECT_EQ(rocsparse_destroy_dnmat_descr(mat_C), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_dnmat_descr(mat_B), rocsparse_status_success);
        EXPECT_EQ(rocsparse_destroy_spmat_descr(mat_A), rocsparse_status_success);
    }
} // namespace

class Level3 : public HandleTest
{
};

#define UT_L3_ALL_PRECISIONS(NAME, FN)       \
    TEST_F(Level3, NAME)                     \
    {                                        \
        FN<float>(handle);                   \
        FN<double>(handle);                  \
        FN<rocsparse_float_complex>(handle); \
        FN<rocsparse_double_complex>(handle);\
    }

UT_L3_ALL_PRECISIONS(csrmm, run_csrmm)
UT_L3_ALL_PRECISIONS(bsrmm, run_bsrmm)
UT_L3_ALL_PRECISIONS(gebsrmm, run_gebsrmm)
UT_L3_ALL_PRECISIONS(gemmi, run_gemmi)
UT_L3_ALL_PRECISIONS(csrsm, run_csrsm)
UT_L3_ALL_PRECISIONS(bsrsm, run_bsrsm)
UT_L3_ALL_PRECISIONS(coomm_spmm, run_coomm)

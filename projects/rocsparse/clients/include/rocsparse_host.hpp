/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#pragma once
#ifndef ROCSPARSE_HOST_HPP
#define ROCSPARSE_HOST_HPP

#include "rocsparse_test.hpp"

#include <hip/hip_runtime_api.h>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

template <typename T, typename I, typename J, typename A, typename B, typename C>
struct rocsparse_host
{

    static void csrddmm(rocsparse_operation  trans_A,
                        rocsparse_operation  trans_B,
                        rocsparse_order      order_A,
                        rocsparse_order      order_B,
                        J                    M,
                        J                    N,
                        J                    K,
                        I                    nnz,
                        const T*             alpha,
                        const A*             dense_A,
                        int64_t              lda,
                        const B*             dense_B,
                        int64_t              ldb,
                        const T*             beta,
                        const I*             csr_row_ptr_C,
                        const J*             csr_col_ind_C,
                        C*                   csr_val_C,
                        rocsparse_index_base base_C);

    static void cscddmm(rocsparse_operation  trans_A,
                        rocsparse_operation  trans_B,
                        rocsparse_order      order_A,
                        rocsparse_order      order_B,
                        J                    M,
                        J                    N,
                        J                    K,
                        I                    nnz,
                        const T*             alpha,
                        const A*             dense_A,
                        int64_t              lda,
                        const B*             dense_B,
                        int64_t              ldb,
                        const T*             beta,
                        const I*             csr_row_ptr_C,
                        const J*             csr_col_ind_C,
                        C*                   csr_val_C,
                        rocsparse_index_base base_C);

    static void cooddmm(rocsparse_operation  trans_A,
                        rocsparse_operation  trans_B,
                        rocsparse_order      order_A,
                        rocsparse_order      order_B,
                        J                    M,
                        J                    N,
                        J                    K,
                        I                    nnz,
                        const T*             alpha,
                        const A*             dense_A,
                        int64_t              lda,
                        const B*             dense_B,
                        int64_t              ldb,
                        const T*             beta,
                        const I*             coo_row_ind_C,
                        const I*             coo_col_ind_C,
                        C*                   coo_val_C,
                        rocsparse_index_base base_C);

    static void cooaosddmm(rocsparse_operation  trans_A,
                           rocsparse_operation  trans_B,
                           rocsparse_order      order_A,
                           rocsparse_order      order_B,
                           J                    M,
                           J                    N,
                           J                    K,
                           I                    nnz,
                           const T*             alpha,
                           const A*             dense_A,
                           int64_t              lda,
                           const B*             dense_B,
                           int64_t              ldb,
                           const T*             beta,
                           const I*             coo_row_ind_C,
                           const I*             coo_col_ind_C,
                           C*                   coo_val_C,
                           rocsparse_index_base base_C);

    static void ellddmm(rocsparse_operation  trans_A,
                        rocsparse_operation  trans_B,
                        rocsparse_order      order_A,
                        rocsparse_order      order_B,
                        J                    M,
                        J                    N,
                        J                    K,
                        I                    nnz,
                        const T*             alpha,
                        const A*             dense_A,
                        int64_t              lda,
                        const B*             dense_B,
                        int64_t              ldb,
                        const T*             beta,
                        const J              ell_width,
                        const I*             ell_ind_C,
                        C*                   ell_val_C,
                        rocsparse_index_base base_C);
};

// BSR indexing macros
#define BSR_IND(j, bi, bj, dir) \
    ((dir == rocsparse_direction_row) ? BSR_IND_R(j, bi, bj) : BSR_IND_C(j, bi, bj))
#define BSR_IND_R(j, bi, bj) (bsr_dim * bsr_dim * (j) + (bi)*bsr_dim + (bj))
#define BSR_IND_C(j, bi, bj) (bsr_dim * bsr_dim * (j) + (bi) + (bj)*bsr_dim)

/*
 * ===========================================================================
 *    level 1 SPARSE
 * ===========================================================================
 */
template <typename T, typename I, typename X, typename Y>
void host_axpby(I                    size,
                I                    nnz,
                T                    alpha,
                const X*             x_val,
                const I*             x_ind,
                T                    beta,
                Y*                   y,
                rocsparse_index_base base);

template <typename I, typename X, typename Y, typename T>
void host_doti(
    I nnz, const X* x_val, const I* x_ind, const Y* y, T* result, rocsparse_index_base base);
template <typename I, typename X, typename Y, typename T>
void host_dotci(
    I nnz, const X* x_val, const I* x_ind, const Y* y, T* result, rocsparse_index_base base);
template <typename I, typename T>
void host_gthr(I nnz, const T* y, T* x_val, const I* x_ind, rocsparse_index_base base);
template <typename T>
void host_gthrz(
    rocsparse_int nnz, T* y, T* x_val, const rocsparse_int* x_ind, rocsparse_index_base base);
template <typename I, typename T>
void host_roti(
    I nnz, T* x_val, const I* x_ind, T* y, const T* c, const T* s, rocsparse_index_base base);

template <typename I, typename T>
void host_sctr(I nnz, const T* x_val, const I* x_ind, T* y, rocsparse_index_base base);

/*
 * ===========================================================================
 *    level 2 SPARSE
 * ===========================================================================
 */
template <typename T, typename I, typename J, typename A, typename X, typename Y>
void host_bsrmv(rocsparse_direction  dir,
                rocsparse_operation  trans,
                J                    mb,
                J                    nb,
                I                    nnzb,
                T                    alpha,
                const I*             bsr_row_ptr,
                const I*             bsr_end_ptr,
                const J*             bsr_col_ind,
                const A*             bsr_val,
                J                    bsr_dim,
                const X*             x,
                T                    beta,
                Y*                   y,
                rocsparse_index_base base);

template <typename T, typename I, typename J, typename A, typename X, typename Y>
void host_bsrmv(rocsparse_direction  dir,
                rocsparse_operation  trans,
                J                    mb,
                J                    nb,
                I                    nnzb,
                T                    alpha,
                const I*             bsr_row_ptr,
                const J*             bsr_col_ind,
                const A*             bsr_val,
                J                    bsr_dim,
                const X*             x,
                T                    beta,
                Y*                   y,
                rocsparse_index_base base);

template <typename T>
void host_bsrxmv(rocsparse_direction  dir,
                 rocsparse_operation  trans,
                 rocsparse_int        size_of_mask,
                 rocsparse_int        mb,
                 rocsparse_int        nb,
                 rocsparse_int        nnzb,
                 T                    alpha,
                 const rocsparse_int* bsr_mask_ptr,
                 const rocsparse_int* bsr_row_ptr,
                 const rocsparse_int* bsr_end_ptr,
                 const rocsparse_int* bsr_col_ind,
                 const T*             bsr_val,
                 rocsparse_int        bsr_dim,
                 const T*             x,
                 T                    beta,
                 T*                   y,
                 rocsparse_index_base base);

template <typename T>
void host_bsrsv(rocsparse_operation  trans,
                rocsparse_direction  dir,
                rocsparse_int        mb,
                rocsparse_int        nnzb,
                T                    alpha,
                const rocsparse_int* bsr_row_ptr,
                const rocsparse_int* bsr_col_ind,
                const T*             bsr_val,
                rocsparse_int        bsr_dim,
                const T*             x,
                T*                   y,
                rocsparse_diag_type  diag_type,
                rocsparse_fill_mode  fill_mode,
                rocsparse_index_base base,
                rocsparse_int*       struct_pivot,
                rocsparse_int*       numeric_pivot);

template <typename T, typename I, typename A, typename X, typename Y>
void host_coomv(rocsparse_operation  trans,
                I                    M,
                I                    N,
                int64_t              nnz,
                T                    alpha,
                const I*             coo_row_ind,
                const I*             coo_col_ind,
                const A*             coo_val,
                const X*             x,
                T                    beta,
                Y*                   y,
                rocsparse_index_base base);

template <typename T, typename I, typename A, typename X, typename Y>
void host_coomv_aos(rocsparse_operation  trans,
                    I                    M,
                    I                    N,
                    int64_t              nnz,
                    T                    alpha,
                    const I*             coo_ind,
                    const A*             coo_val,
                    const X*             x,
                    T                    beta,
                    Y*                   y,
                    rocsparse_index_base base);

template <typename T, typename I, typename J, typename A, typename X, typename Y>
void host_csrmv(rocsparse_operation   trans,
                J                     M,
                J                     N,
                I                     nnz,
                T                     alpha,
                const I*              csr_row_ptr,
                const J*              csr_col_ind,
                const A*              csr_val,
                const X*              x,
                T                     beta,
                Y*                    y,
                rocsparse_index_base  base,
                rocsparse_matrix_type matrix_type,
                rocsparse_spmv_alg    algo,
                bool                  force_conj);

template <typename T, typename I, typename J, typename A, typename X, typename Y>
void host_cscmv(rocsparse_operation trans,
                J                   M,
                J                   N,
                I                   nnz,
                T                   alpha,
                const I* __restrict csc_col_ptr,
                const J* __restrict csc_row_ind,
                const A* __restrict csc_val,
                const X* __restrict x,
                T beta,
                Y* __restrict y,
                rocsparse_index_base  base,
                rocsparse_matrix_type matrix_type,
                rocsparse_spmv_alg    algo);

template <typename I, typename J, typename T>
void host_csrsv(rocsparse_operation  trans,
                J                    M,
                I                    nnz,
                T                    alpha,
                const I*             csr_row_ptr,
                const J*             csr_col_ind,
                const T*             csr_val,
                const T*             x,
                int64_t              x_inc,
                T*                   y,
                rocsparse_diag_type  diag_type,
                rocsparse_fill_mode  fill_mode,
                rocsparse_index_base base,
                J*                   struct_pivot,
                J*                   numeric_pivot);

template <typename I, typename T>
void host_coosv(rocsparse_operation  trans,
                I                    M,
                int64_t              nnz,
                T                    alpha,
                const I*             coo_row_ind,
                const I*             coo_col_ind,
                const T*             coo_val,
                const T*             x,
                T*                   y,
                rocsparse_diag_type  diag_type,
                rocsparse_fill_mode  fill_mode,
                rocsparse_index_base base,
                I*                   struct_pivot,
                I*                   numeric_pivot);

template <typename T, typename I, typename A, typename X, typename Y>
void host_ellmv(rocsparse_operation  trans,
                I                    M,
                I                    N,
                T                    alpha,
                const I*             ell_col_ind,
                const A*             ell_val,
                I                    ell_width,
                const X*             x,
                T                    beta,
                Y*                   y,
                rocsparse_index_base base);

template <typename T>
void host_hybmv(rocsparse_operation  trans,
                rocsparse_int        M,
                rocsparse_int        N,
                T                    alpha,
                rocsparse_int        ell_nnz,
                const rocsparse_int* ell_col_ind,
                const T*             ell_val,
                rocsparse_int        ell_width,
                rocsparse_int        coo_nnz,
                const rocsparse_int* coo_row_ind,
                const rocsparse_int* coo_col_ind,
                const T*             coo_val,
                const T*             x,
                T                    beta,
                T*                   y,
                rocsparse_index_base base);

template <typename T>
void host_gebsrmv(rocsparse_direction  dir,
                  rocsparse_operation  trans,
                  rocsparse_int        mb,
                  rocsparse_int        nb,
                  rocsparse_int        nnzb,
                  T                    alpha,
                  const rocsparse_int* bsr_row_ptr,
                  const rocsparse_int* bsr_col_ind,
                  const T*             bsr_val,
                  rocsparse_int        row_block_dim,
                  rocsparse_int        col_block_dim,
                  const T*             x,
                  T                    beta,
                  T*                   y,
                  rocsparse_index_base base);

template <typename I, typename T>
void host_gemvi(I                    M,
                I                    N,
                T                    alpha,
                const T*             A,
                int64_t              lda,
                I                    nnz,
                const T*             x_val,
                const I*             x_ind,
                T                    beta,
                T*                   y,
                rocsparse_index_base base);
/*
 * ===========================================================================
 *    level 3 SPARSE
 * ===========================================================================
 */
template <typename T, typename I, typename J, typename A, typename B, typename C>
void host_bsrmm(rocsparse_handle     handle,
                rocsparse_direction  dir,
                rocsparse_operation  trans_A,
                rocsparse_operation  trans_B,
                J                    mb,
                J                    n,
                J                    kb,
                I                    nnzb,
                T                    alpha,
                const A*             bsr_val,
                const I*             bsr_row_ptr,
                const J*             bsr_col_ind,
                J                    block_dim,
                const B*             dense_B,
                int64_t              ldb,
                rocsparse_order      order_B,
                T                    beta,
                C*                   dense_C,
                int64_t              ldc,
                rocsparse_order      order_C,
                rocsparse_index_base base);

template <typename T>
void host_gebsrmm(rocsparse_handle          handle,
                  rocsparse_direction       dir,
                  rocsparse_operation       trans_A,
                  rocsparse_operation       trans_B,
                  rocsparse_int             mb,
                  rocsparse_int             n,
                  rocsparse_int             kb,
                  rocsparse_int             nnzb,
                  const T*                  alpha,
                  const rocsparse_mat_descr descr,
                  const T*                  bsr_val,
                  const rocsparse_int*      bsr_row_ptr,
                  const rocsparse_int*      bsr_col_ind,
                  rocsparse_int             row_block_dim,
                  rocsparse_int             col_block_dim,
                  const T*                  B,
                  int64_t                   ldb,
                  const T*                  beta,
                  T*                        C,
                  int64_t                   ldc);

template <typename T, typename I, typename J, typename A, typename B, typename C>
void host_csrmm(J                    M,
                J                    N,
                J                    K,
                rocsparse_operation  transA,
                rocsparse_operation  transB,
                T                    alpha,
                const I*             csr_row_ptr_A,
                const J*             csr_col_ind_A,
                const A*             csr_val_A,
                const B*             dense_B,
                int64_t              ldb,
                rocsparse_order      order_B,
                T                    beta,
                C*                   dense_C,
                int64_t              ldc,
                rocsparse_order      order_C,
                rocsparse_index_base base,
                bool                 force_conj_A);

template <typename T, typename I, typename J, typename A, typename B, typename C>
void host_csrmm_batched(J                    M,
                        J                    N,
                        J                    K,
                        J                    batch_count_A,
                        int64_t              offsets_batch_stride_A,
                        int64_t              columns_values_batch_stride_A,
                        rocsparse_operation  transA,
                        rocsparse_operation  transB,
                        T                    alpha,
                        const I*             csr_row_ptr_A,
                        const J*             csr_col_ind_A,
                        const A*             csr_val_A,
                        const B*             dense_B,
                        int64_t              ldb,
                        J                    batch_count_B,
                        int64_t              batch_stride_B,
                        rocsparse_order      order_B,
                        T                    beta,
                        C*                   dense_C,
                        int64_t              ldc,
                        J                    batch_count_C,
                        int64_t              batch_stride_C,
                        rocsparse_order      order_C,
                        rocsparse_index_base base,
                        bool                 force_conj_A);

template <typename T, typename I, typename A, typename B, typename C>
void host_coomm(I                    M,
                I                    N,
                I                    K,
                int64_t              nnz,
                rocsparse_operation  transA,
                rocsparse_operation  transB,
                T                    alpha,
                const I*             coo_row_ind_A,
                const I*             coo_col_ind_A,
                const A*             coo_val_A,
                const B*             dense_B,
                int64_t              ldb,
                rocsparse_order      order_B,
                T                    beta,
                C*                   dense_C,
                int64_t              ldc,
                rocsparse_order      order_C,
                rocsparse_index_base base);

template <typename T, typename I, typename A, typename B, typename C>
void host_coomm_batched(I                    M,
                        I                    N,
                        I                    K,
                        int64_t              nnz,
                        I                    batch_count_A,
                        int64_t              batch_stride_A,
                        rocsparse_operation  transA,
                        rocsparse_operation  transB,
                        T                    alpha,
                        const I*             coo_row_ind_A,
                        const I*             coo_col_ind_A,
                        const A*             coo_val_A,
                        const B*             dense_B,
                        int64_t              ldb,
                        I                    batch_count_B,
                        int64_t              batch_stride_B,
                        rocsparse_order      order_B,
                        T                    beta,
                        C*                   dense_C,
                        int64_t              ldc,
                        I                    batch_count_C,
                        int64_t              batch_stride_C,
                        rocsparse_order      order_C,
                        rocsparse_index_base base);

template <typename T, typename I, typename J, typename A, typename B, typename C>
void host_cscmm(J                   M,
                J                   N,
                J                   K,
                rocsparse_operation transA,
                rocsparse_operation transB,
                T                   alpha,
                const I* __restrict csc_col_ptr_A,
                const J* __restrict csc_row_ind_A,
                const A* __restrict csc_val_A,
                const B* __restrict dense_B,
                int64_t         ldb,
                rocsparse_order order_B,
                T               beta,
                C* __restrict dense_C,
                int64_t              ldc,
                rocsparse_order      order_C,
                rocsparse_index_base base);

template <typename T, typename I, typename J, typename A, typename B, typename C>
void host_cscmm_batched(J                    M,
                        J                    N,
                        J                    K,
                        J                    batch_count_A,
                        int64_t              offsets_batch_stride_A,
                        int64_t              rows_values_batch_stride_A,
                        rocsparse_operation  transA,
                        rocsparse_operation  transB,
                        T                    alpha,
                        const I*             csc_col_ptr_A,
                        const J*             csc_row_ind_A,
                        const A*             csc_val_A,
                        const B*             dense_B,
                        int64_t              ldb,
                        J                    batch_count_B,
                        int64_t              batch_stride_B,
                        rocsparse_order      order_B,
                        T                    beta,
                        C*                   dense_C,
                        int64_t              ldc,
                        J                    batch_count_C,
                        int64_t              batch_stride_C,
                        rocsparse_order      order_C,
                        rocsparse_index_base base);

template <typename I, typename J, typename T>
void host_csrsm(J                    M,
                J                    nrhs,
                I                    nnz,
                rocsparse_operation  transA,
                rocsparse_operation  transB,
                T                    alpha,
                const I*             csr_row_ptr,
                const J*             csr_col_ind,
                const T*             csr_val,
                T*                   B,
                int64_t              ldb,
                rocsparse_order      order_B,
                rocsparse_diag_type  diag_type,
                rocsparse_fill_mode  fill_mode,
                rocsparse_index_base base,
                J*                   struct_pivot,
                J*                   numeric_pivot);

template <typename I, typename T>
void host_coosm(I                    M,
                I                    nrhs,
                int64_t              nnz,
                rocsparse_operation  transA,
                rocsparse_operation  transB,
                T                    alpha,
                const I*             coo_row_ind,
                const I*             coo_col_ind,
                const T*             coo_val,
                T*                   B,
                int64_t              ldb,
                rocsparse_order      order_B,
                rocsparse_diag_type  diag_type,
                rocsparse_fill_mode  fill_mode,
                rocsparse_index_base base,
                I*                   struct_pivot,
                I*                   numeric_pivot);

template <typename T>
void host_bsrsm(rocsparse_int       mb,
                rocsparse_int       nrhs,
                rocsparse_int       nnzb,
                rocsparse_direction dir,
                rocsparse_operation transA,
                rocsparse_operation transX,
                T                   alpha,
                const rocsparse_int* __restrict bsr_row_ptr,
                const rocsparse_int* __restrict bsr_col_ind,
                const T* __restrict bsr_val,
                rocsparse_int        bsr_dim,
                const T*             B,
                int64_t              ldb,
                T*                   X,
                int64_t              ldx,
                rocsparse_diag_type  diag_type,
                rocsparse_fill_mode  fill_mode,
                rocsparse_index_base base,
                rocsparse_int*       struct_pivot,
                rocsparse_int*       numeric_pivot);

template <typename T>
void host_gemmi(rocsparse_int        M,
                rocsparse_int        N,
                rocsparse_operation  transA,
                rocsparse_operation  transB,
                T                    alpha,
                const T*             A,
                int64_t              lda,
                const rocsparse_int* csr_row_ptr,
                const rocsparse_int* csr_col_ind,
                const T*             csr_val,
                T                    beta,
                T*                   C,
                int64_t              ldc,
                rocsparse_index_base base);

/*
 * ===========================================================================
 *    extra SPARSE
 * ===========================================================================
 */
template <typename T>
void host_bsrgeam_nnzb(rocsparse_direction  dir,
                       rocsparse_int        Mb,
                       rocsparse_int        Nb,
                       rocsparse_int        block_dim,
                       const T*             alpha,
                       const rocsparse_int* bsr_row_ptr_A,
                       const rocsparse_int* bsr_col_ind_A,
                       const T*             beta,
                       const rocsparse_int* bsr_row_ptr_B,
                       const rocsparse_int* bsr_col_ind_B,
                       rocsparse_int*       bsr_row_ptr_C,
                       rocsparse_int*       nnzb_C,
                       rocsparse_index_base base_A,
                       rocsparse_index_base base_B,
                       rocsparse_index_base base_C);

template <typename T>
void host_bsrgeam(rocsparse_direction  dir,
                  rocsparse_int        Mb,
                  rocsparse_int        Nb,
                  rocsparse_int        block_dim,
                  const T*             alpha,
                  const rocsparse_int* bsr_row_ptr_A,
                  const rocsparse_int* bsr_col_ind_A,
                  const T*             bsr_val_A,
                  const T*             beta,
                  const rocsparse_int* bsr_row_ptr_B,
                  const rocsparse_int* bsr_col_ind_B,
                  const T*             bsr_val_B,
                  const rocsparse_int* bsr_row_ptr_C,
                  rocsparse_int*       bsr_col_ind_C,
                  T*                   bsr_val_C,
                  rocsparse_index_base base_A,
                  rocsparse_index_base base_B,
                  rocsparse_index_base base_C);

template <typename T, typename I, typename J>
void host_csrgeam_nnz(J                    M,
                      J                    N,
                      const T*             alpha,
                      const I*             csr_row_ptr_A,
                      const J*             csr_col_ind_A,
                      const T*             beta,
                      const I*             csr_row_ptr_B,
                      const J*             csr_col_ind_B,
                      I*                   csr_row_ptr_C,
                      I*                   nnz_C,
                      rocsparse_index_base base_A,
                      rocsparse_index_base base_B,
                      rocsparse_index_base base_C);

template <typename T, typename I, typename J>
void host_csrgeam(J                    M,
                  J                    N,
                  const T*             alpha,
                  const I*             csr_row_ptr_A,
                  const J*             csr_col_ind_A,
                  const T*             csr_val_A,
                  const T*             beta,
                  const I*             csr_row_ptr_B,
                  const J*             csr_col_ind_B,
                  const T*             csr_val_B,
                  const I*             csr_row_ptr_C,
                  J*                   csr_col_ind_C,
                  T*                   csr_val_C,
                  rocsparse_index_base base_A,
                  rocsparse_index_base base_B,
                  rocsparse_index_base base_C);

template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
void host_bsrgemm_nnzb(J                    Mb,
                       J                    Nb,
                       J                    Kb,
                       J                    block_dim,
                       const T*             alpha,
                       const I*             bsr_row_ptr_A,
                       const J*             bsr_col_ind_A,
                       const I*             bsr_row_ptr_B,
                       const J*             bsr_col_ind_B,
                       const T*             beta,
                       const I*             bsr_row_ptr_D,
                       const J*             bsr_col_ind_D,
                       I*                   bsr_row_ptr_C,
                       I*                   nnzb_C,
                       rocsparse_index_base base_A,
                       rocsparse_index_base base_B,
                       rocsparse_index_base base_C,
                       rocsparse_index_base base_D);

template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
void host_bsrgemm(rocsparse_direction  dir,
                  J                    Mb,
                  J                    Nb,
                  J                    Kb,
                  J                    block_dim,
                  const T*             alpha,
                  const I*             bsr_row_ptr_A,
                  const J*             bsr_col_ind_A,
                  const T*             bsr_val_A,
                  const I*             bsr_row_ptr_B,
                  const J*             bsr_col_ind_B,
                  const T*             bsr_val_B,
                  const T*             beta,
                  const I*             bsr_row_ptr_D,
                  const J*             bsr_col_ind_D,
                  const T*             bsr_val_D,
                  const I*             bsr_row_ptr_C,
                  J*                   bsr_col_ind_C,
                  T*                   bsr_val_C,
                  rocsparse_index_base base_A,
                  rocsparse_index_base base_B,
                  rocsparse_index_base base_C,
                  rocsparse_index_base base_D);

template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
void host_csrgemm_nnz(J                    M,
                      J                    N,
                      J                    K,
                      const T*             alpha,
                      const I*             csr_row_ptr_A,
                      const J*             csr_col_ind_A,
                      const I*             csr_row_ptr_B,
                      const J*             csr_col_ind_B,
                      const T*             beta,
                      const I*             csr_row_ptr_D,
                      const J*             csr_col_ind_D,
                      I*                   csr_row_ptr_C,
                      I*                   nnz_C,
                      rocsparse_index_base base_A,
                      rocsparse_index_base base_B,
                      rocsparse_index_base base_C,
                      rocsparse_index_base base_D);

template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
void host_csrgemm(J                    M,
                  J                    N,
                  J                    L,
                  const T*             alpha,
                  const I*             csr_row_ptr_A,
                  const J*             csr_col_ind_A,
                  const T*             csr_val_A,
                  const I*             csr_row_ptr_B,
                  const J*             csr_col_ind_B,
                  const T*             csr_val_B,
                  const T*             beta,
                  const I*             csr_row_ptr_D,
                  const J*             csr_col_ind_D,
                  const T*             csr_val_D,
                  const I*             csr_row_ptr_C,
                  J*                   csr_col_ind_C,
                  T*                   csr_val_C,
                  rocsparse_index_base base_A,
                  rocsparse_index_base base_B,
                  rocsparse_index_base base_C,
                  rocsparse_index_base base_D);

/*
 * ===========================================================================
 *    precond SPARSE
 * ===========================================================================
 */
template <typename T>
void host_bsric0(rocsparse_direction               direction,
                 rocsparse_int                     Mb,
                 rocsparse_int                     block_dim,
                 const std::vector<rocsparse_int>& bsr_row_ptr,
                 const std::vector<rocsparse_int>& bsr_col_ind,
                 std::vector<T>&                   bsr_val,
                 rocsparse_index_base              base,
                 rocsparse_int*                    struct_pivot,
                 rocsparse_int*                    numeric_pivot);

template <typename T, typename U>
void host_bsrilu0(rocsparse_direction               dir,
                  rocsparse_int                     mb,
                  const std::vector<rocsparse_int>& bsr_row_ptr,
                  const std::vector<rocsparse_int>& bsr_col_ind,
                  std::vector<T>&                   bsr_val,
                  rocsparse_int                     bsr_dim,
                  rocsparse_index_base              base,
                  rocsparse_int*                    struct_pivot,
                  rocsparse_int*                    numeric_pivot,
                  bool                              boost,
                  U                                 boost_tol,
                  T                                 boost_val);

template <typename T>
void host_csric0(rocsparse_int                     M,
                 const std::vector<rocsparse_int>& csr_row_ptr,
                 const std::vector<rocsparse_int>& csr_col_ind,
                 std::vector<T>&                   csr_val,
                 rocsparse_index_base              base,
                 rocsparse_int*                    struct_pivot,
                 rocsparse_int*                    numeric_pivot,
                 rocsparse_int*                    singular_pivot,
                 double                            tol);

template <typename T, typename U>
void host_csrilu0(rocsparse_int                     M,
                  const std::vector<rocsparse_int>& csr_row_ptr,
                  const std::vector<rocsparse_int>& csr_col_ind,
                  std::vector<T>&                   csr_val,
                  rocsparse_index_base              base,
                  rocsparse_int*                    struct_pivot,
                  rocsparse_int*                    numeric_pivot,
                  rocsparse_int*                    singular_pivot,
                  double                            tol,
                  bool                              boost,
                  U                                 boost_tol,
                  T                                 boost_val);

template <typename T>
void host_gtsv_no_pivot(rocsparse_int         m,
                        rocsparse_int         n,
                        const std::vector<T>& dl,
                        const std::vector<T>& d,
                        const std::vector<T>& du,
                        std::vector<T>&       B,
                        rocsparse_int         ldb);

template <typename T>
void host_gtsv_no_pivot_strided_batch(rocsparse_int         m,
                                      const std::vector<T>& dl,
                                      const std::vector<T>& d,
                                      const std::vector<T>& du,
                                      std::vector<T>&       x,
                                      rocsparse_int         batch_count,
                                      rocsparse_int         batch_stride);

template <typename T>
void host_gtsv_interleaved_batch(rocsparse_gtsv_interleaved_alg algo,
                                 rocsparse_int                  m,
                                 const T*                       dl,
                                 const T*                       d,
                                 const T*                       du,
                                 T*                             x,
                                 rocsparse_int                  batch_count,
                                 rocsparse_int                  batch_stride);

template <typename T>
void host_gpsv_interleaved_batch(rocsparse_gpsv_interleaved_alg algo,
                                 rocsparse_int                  m,
                                 T*                             ds,
                                 T*                             dl,
                                 T*                             d,
                                 T*                             du,
                                 T*                             dw,
                                 T*                             x,
                                 rocsparse_int                  batch_count,
                                 rocsparse_int                  batch_stride);

/*
 * ===========================================================================
 *    conversion SPARSE
 * ===========================================================================
 */
template <typename T>
rocsparse_status host_nnz(rocsparse_direction dirA,
                          rocsparse_int       m,
                          rocsparse_int       n,
                          const T*            A,
                          int64_t             lda,
                          rocsparse_int*      nnz_per_row_columns,
                          rocsparse_int*      nnz_total_dev_host_ptr);

template <typename T>
void host_prune_dense2csr(rocsparse_int               m,
                          rocsparse_int               n,
                          const std::vector<T>&       A,
                          int64_t                     lda,
                          rocsparse_index_base        base,
                          T                           threshold,
                          rocsparse_int&              nnz,
                          std::vector<T>&             csr_val,
                          std::vector<rocsparse_int>& csr_row_ptr,
                          std::vector<rocsparse_int>& csr_col_ind);

template <typename T>
void host_prune_dense2csr_by_percentage(rocsparse_int               m,
                                        rocsparse_int               n,
                                        const std::vector<T>&       A,
                                        int64_t                     lda,
                                        rocsparse_index_base        base,
                                        T                           percentage,
                                        rocsparse_int&              nnz,
                                        std::vector<T>&             csr_val,
                                        std::vector<rocsparse_int>& csr_row_ptr,
                                        std::vector<rocsparse_int>& csr_col_ind);

template <rocsparse_direction DIRA, typename T, typename I, typename J>
void host_dense2csx(J                    m,
                    J                    n,
                    rocsparse_index_base base,
                    const T*             A,
                    int64_t              ld,
                    rocsparse_order      order,
                    const I*             nnz_per_row_columns,
                    T*                   csx_val,
                    I*                   csx_row_col_ptr,
                    J*                   csx_col_row_ind);

template <rocsparse_direction DIRA, typename T, typename I, typename J>
void host_csx2dense(J                    m,
                    J                    n,
                    rocsparse_index_base base,
                    rocsparse_order      order,
                    const T*             csx_val,
                    const I*             csx_row_col_ptr,
                    const J*             csx_col_row_ind,
                    T*                   A,
                    int64_t              ld);

template <typename I, typename T>
void host_dense_to_coo(I                     m,
                       I                     n,
                       rocsparse_index_base  base,
                       const std::vector<T>& A,
                       int64_t               ld,
                       rocsparse_order       order,
                       const std::vector<I>& nnz_per_row,
                       std::vector<T>&       coo_val,
                       std::vector<I>&       coo_row_ind,
                       std::vector<I>&       coo_col_ind);

template <typename I, typename T>
void host_coo_to_dense(I                     m,
                       I                     n,
                       int64_t               nnz,
                       rocsparse_index_base  base,
                       const std::vector<T>& coo_val,
                       const std::vector<I>& coo_row_ind,
                       const std::vector<I>& coo_col_ind,
                       std::vector<T>&       A,
                       int64_t               ld,
                       rocsparse_order       order);

template <typename I, typename J>
void host_csr_to_coo(J                     M,
                     I                     nnz,
                     const std::vector<I>& csr_row_ptr,
                     std::vector<J>&       coo_row_ind,
                     rocsparse_index_base  base);

template <typename I, typename J>
void host_csr_to_coo_aos(J                     M,
                         I                     nnz,
                         const std::vector<I>& csr_row_ptr,
                         const std::vector<J>& csr_col_ind,
                         std::vector<I>&       coo_ind,
                         rocsparse_index_base  base);

template <typename I, typename J, typename T>
void host_csr_to_csc(J                    M,
                     J                    N,
                     I                    nnz,
                     const I*             csr_row_ptr,
                     const J*             csr_col_ind,
                     const T*             csr_val,
                     std::vector<J>&      csc_row_ind,
                     std::vector<I>&      csc_col_ptr,
                     std::vector<T>&      csc_val,
                     rocsparse_action     action,
                     rocsparse_index_base base);

template <typename T>
void host_bsr_to_csr(rocsparse_direction               direction,
                     rocsparse_int                     mb,
                     rocsparse_int                     nb,
                     rocsparse_int                     nnzb,
                     const std::vector<T>&             bsr_val,
                     const std::vector<rocsparse_int>& bsr_row_ptr,
                     const std::vector<rocsparse_int>& bsr_col_ind,
                     rocsparse_int                     block_dim,
                     rocsparse_index_base              bsr_base,
                     std::vector<T>&                   csr_val,
                     std::vector<rocsparse_int>&       csr_row_ptr,
                     std::vector<rocsparse_int>&       csr_col_ind,
                     rocsparse_index_base              csr_base);

template <typename T>
void host_csr_to_bsr(rocsparse_direction               direction,
                     rocsparse_int                     m,
                     rocsparse_int                     n,
                     rocsparse_int                     nnz,
                     const std::vector<T>&             csr_val,
                     const std::vector<rocsparse_int>& csr_row_ptr,
                     const std::vector<rocsparse_int>& csr_col_ind,
                     rocsparse_int                     block_dim,
                     rocsparse_index_base              csr_base,
                     std::vector<T>&                   bsr_val,
                     std::vector<rocsparse_int>&       bsr_row_ptr,
                     std::vector<rocsparse_int>&       bsr_col_ind,
                     rocsparse_index_base              bsr_base);

template <typename T>
void host_gebsr_to_gebsc(rocsparse_int                     Mb,
                         rocsparse_int                     Nb,
                         rocsparse_int                     nnzb,
                         const std::vector<rocsparse_int>& bsr_row_ptr,
                         const std::vector<rocsparse_int>& bsr_col_ind,
                         const std::vector<T>&             bsr_val,
                         rocsparse_int                     row_block_dim,
                         rocsparse_int                     col_block_dim,
                         std::vector<rocsparse_int>&       bsc_row_ind,
                         std::vector<rocsparse_int>&       bsc_col_ptr,
                         std::vector<T>&                   bsc_val,
                         rocsparse_action                  action,
                         rocsparse_index_base              base);

template <typename T>
void host_gebsr_to_csr(rocsparse_direction               direction,
                       rocsparse_int                     mb,
                       rocsparse_int                     nb,
                       rocsparse_int                     nnzb,
                       const std::vector<T>&             bsr_val,
                       const std::vector<rocsparse_int>& bsr_row_ptr,
                       const std::vector<rocsparse_int>& bsr_col_ind,
                       rocsparse_int                     row_block_dim,
                       rocsparse_int                     col_block_dim,
                       rocsparse_index_base              bsr_base,
                       std::vector<T>&                   csr_val,
                       std::vector<rocsparse_int>&       csr_row_ptr,
                       std::vector<rocsparse_int>&       csr_col_ind,
                       rocsparse_index_base              csr_base);

template <typename T>
void host_csr_to_gebsr(rocsparse_direction               direction,
                       rocsparse_int                     m,
                       rocsparse_int                     n,
                       rocsparse_int                     nnz,
                       const std::vector<T>&             csr_val,
                       const std::vector<rocsparse_int>& csr_row_ptr,
                       const std::vector<rocsparse_int>& csr_col_ind,
                       rocsparse_int                     row_block_dim,
                       rocsparse_int                     col_block_dim,
                       rocsparse_index_base              csr_base,
                       std::vector<T>&                   bsr_val,
                       std::vector<rocsparse_int>&       bsr_row_ptr,
                       std::vector<rocsparse_int>&       bsr_col_ind,
                       rocsparse_index_base              bsr_base);

template <typename T>
void host_bsrpad_value(rocsparse_int m,
                       rocsparse_int mb,
                       rocsparse_int nnzb,
                       rocsparse_int block_dim,
                       T             value,
                       T* __restrict bsr_val,
                       const rocsparse_int* __restrict bsr_row_ptr,
                       const rocsparse_int* __restrict bsr_col_ind,
                       rocsparse_index_base bsr_base);

template <typename T>
void host_gebsr_to_gebsr(rocsparse_direction               direction,
                         rocsparse_int                     mb,
                         rocsparse_int                     nb,
                         rocsparse_int                     nnzb,
                         const std::vector<T>&             bsr_val_A,
                         const std::vector<rocsparse_int>& bsr_row_ptr_A,
                         const std::vector<rocsparse_int>& bsr_col_ind_A,
                         rocsparse_int                     row_block_dim_A,
                         rocsparse_int                     col_block_dim_A,
                         rocsparse_index_base              base_A,
                         std::vector<T>&                   bsr_val_C,
                         std::vector<rocsparse_int>&       bsr_row_ptr_C,
                         std::vector<rocsparse_int>&       bsr_col_ind_C,
                         rocsparse_int                     row_block_dim_C,
                         rocsparse_int                     col_block_dim_C,
                         rocsparse_index_base              base_C);

template <typename T>
void host_bsr_to_bsc(rocsparse_int               mb,
                     rocsparse_int               nb,
                     rocsparse_int               nnzb,
                     rocsparse_int               bsr_dim,
                     const rocsparse_int*        bsr_row_ptr,
                     const rocsparse_int*        bsr_col_ind,
                     const T*                    bsr_val,
                     std::vector<rocsparse_int>& bsc_row_ind,
                     std::vector<rocsparse_int>& bsc_col_ptr,
                     std::vector<T>&             bsc_val,
                     rocsparse_index_base        bsr_base,
                     rocsparse_index_base        bsc_base);

template <typename I, typename J, typename T>
void host_csr_to_ell(J                     M,
                     const std::vector<I>& csr_row_ptr,
                     const std::vector<J>& csr_col_ind,
                     const std::vector<T>& csr_val,
                     std::vector<J>&       ell_col_ind,
                     std::vector<T>&       ell_val,
                     J&                    ell_width,
                     rocsparse_index_base  csr_base,
                     rocsparse_index_base  ell_base);

template <typename T>
void host_csr_to_hyb(rocsparse_int                     M,
                     rocsparse_int                     N,
                     rocsparse_int                     nnz,
                     const std::vector<rocsparse_int>& csr_row_ptr,
                     const std::vector<rocsparse_int>& csr_col_ind,
                     const std::vector<T>&             csr_val,
                     std::vector<rocsparse_int>&       ell_col_ind,
                     std::vector<T>&                   ell_val,
                     rocsparse_int&                    ell_width,
                     rocsparse_int&                    ell_nnz,
                     std::vector<rocsparse_int>&       coo_row_ind,
                     std::vector<rocsparse_int>&       coo_col_ind,
                     std::vector<T>&                   coo_val,
                     rocsparse_int&                    coo_nnz,
                     rocsparse_hyb_partition           part,
                     rocsparse_index_base              base);

template <typename T>
void host_csr_to_csr_compress(rocsparse_int                     M,
                              rocsparse_int                     N,
                              rocsparse_int                     nnz,
                              const std::vector<rocsparse_int>& csr_row_ptr_A,
                              const std::vector<rocsparse_int>& csr_col_ind_A,
                              const std::vector<T>&             csr_val_A,
                              std::vector<rocsparse_int>&       csr_row_ptr_C,
                              std::vector<rocsparse_int>&       csr_col_ind_C,
                              std::vector<T>&                   csr_val_C,
                              rocsparse_index_base              base,
                              T                                 tol);
template <typename T>
void host_prune_csr_to_csr(rocsparse_int                     M,
                           rocsparse_int                     N,
                           rocsparse_int                     nnz_A,
                           const std::vector<rocsparse_int>& csr_row_ptr_A,
                           const std::vector<rocsparse_int>& csr_col_ind_A,
                           const std::vector<T>&             csr_val_A,
                           rocsparse_int&                    nnz_C,
                           std::vector<rocsparse_int>&       csr_row_ptr_C,
                           std::vector<rocsparse_int>&       csr_col_ind_C,
                           std::vector<T>&                   csr_val_C,
                           rocsparse_index_base              csr_base_A,
                           rocsparse_index_base              csr_base_C,
                           T                                 threshold);

template <typename T>
void host_prune_csr_to_csr_by_percentage(rocsparse_int                     M,
                                         rocsparse_int                     N,
                                         rocsparse_int                     nnz_A,
                                         const std::vector<rocsparse_int>& csr_row_ptr_A,
                                         const std::vector<rocsparse_int>& csr_col_ind_A,
                                         const std::vector<T>&             csr_val_A,
                                         rocsparse_int&                    nnz_C,
                                         std::vector<rocsparse_int>&       csr_row_ptr_C,
                                         std::vector<rocsparse_int>&       csr_col_ind_C,
                                         std::vector<T>&                   csr_val_C,
                                         rocsparse_index_base              csr_base_A,
                                         rocsparse_index_base              csr_base_C,
                                         T                                 percentage);

template <typename I, typename J>
void host_coo_to_csr(J M, I nnz, const J* coo_row_ind, I* csr_row_ptr, rocsparse_index_base base);

template <typename T>
void host_ell_to_csr(rocsparse_int                     M,
                     rocsparse_int                     N,
                     const std::vector<rocsparse_int>& ell_col_ind,
                     const std::vector<T>&             ell_val,
                     rocsparse_int                     ell_width,
                     std::vector<rocsparse_int>&       csr_row_ptr,
                     std::vector<rocsparse_int>&       csr_col_ind,
                     std::vector<T>&                   csr_val,
                     rocsparse_int&                    csr_nnz,
                     rocsparse_index_base              ell_base,
                     rocsparse_index_base              csr_base);

template <typename T>
void host_coosort_by_column(rocsparse_int               M,
                            rocsparse_int               nnz,
                            std::vector<rocsparse_int>& coo_row_ind,
                            std::vector<rocsparse_int>& coo_col_ind,
                            std::vector<T>&             coo_val);

#endif // ROCSPARSE_HOST_HPP

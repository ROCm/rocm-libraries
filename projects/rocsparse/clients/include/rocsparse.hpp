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

/*! \file
 *  \brief rocsparse.hpp exposes C++ templated Sparse Linear Algebra interface
 *  with only the precision templated.
 */

#pragma once
#ifndef ROCSPARSE_HPP
#define ROCSPARSE_HPP

#include "rocsparse.h"
#include "rocsparse_traits.hpp"

#define REAL_TEMPLATE(NAME_, ...)                              \
    template <typename T>                                      \
    inline rocsparse_status (*rocsparse_##NAME_)(__VA_ARGS__); \
    template <>                                                \
    inline auto rocsparse_##NAME_<float> = rocsparse_s##NAME_; \
    template <>                                                \
    inline auto rocsparse_##NAME_<double> = rocsparse_d##NAME_

#define COMPLEX_TEMPLATE(NAME_, ...)                                             \
    template <typename T>                                                        \
    inline rocsparse_status (*rocsparse_##NAME_)(__VA_ARGS__);                   \
    template <>                                                                  \
    inline auto rocsparse_##NAME_<rocsparse_float_complex> = rocsparse_c##NAME_; \
    template <>                                                                  \
    inline auto rocsparse_##NAME_<rocsparse_double_complex> = rocsparse_z##NAME_

#define REAL_COMPLEX_TEMPLATE(NAME_, ...)                                        \
    template <typename T>                                                        \
    inline rocsparse_status (*rocsparse_##NAME_)(__VA_ARGS__);                   \
    template <>                                                                  \
    inline auto rocsparse_##NAME_<float> = rocsparse_s##NAME_;                   \
    template <>                                                                  \
    inline auto rocsparse_##NAME_<double> = rocsparse_d##NAME_;                  \
    template <>                                                                  \
    inline auto rocsparse_##NAME_<rocsparse_float_complex> = rocsparse_c##NAME_; \
    template <>                                                                  \
    inline auto rocsparse_##NAME_<rocsparse_double_complex> = rocsparse_z##NAME_

/*
 * ===========================================================================
 *    utility SPARSE
 * ===========================================================================
 */
// check_matrix_csr
REAL_COMPLEX_TEMPLATE(check_matrix_csr_buffer_size,
                      rocsparse_handle       handle,
                      rocsparse_int          m,
                      rocsparse_int          n,
                      rocsparse_int          nnz,
                      const T*               csr_val,
                      const rocsparse_int*   csr_row_ptr,
                      const rocsparse_int*   csr_col_ind,
                      rocsparse_index_base   idx_base,
                      rocsparse_matrix_type  matrix_type,
                      rocsparse_fill_mode    uplo,
                      rocsparse_storage_mode storage,
                      size_t*                buffer_size);

REAL_COMPLEX_TEMPLATE(check_matrix_csr,
                      rocsparse_handle       handle,
                      rocsparse_int          m,
                      rocsparse_int          n,
                      rocsparse_int          nnz,
                      const T*               csr_val,
                      const rocsparse_int*   csr_row_ptr,
                      const rocsparse_int*   csr_col_ind,
                      rocsparse_index_base   idx_base,
                      rocsparse_matrix_type  matrix_type,
                      rocsparse_fill_mode    uplo,
                      rocsparse_storage_mode storage,
                      rocsparse_data_status* data_status,
                      void*                  temp_buffer);

// check_matrix_coo
REAL_COMPLEX_TEMPLATE(check_matrix_coo_buffer_size,
                      rocsparse_handle       handle,
                      rocsparse_int          m,
                      rocsparse_int          n,
                      rocsparse_int          nnz,
                      const T*               coo_val,
                      const rocsparse_int*   coo_row_ind,
                      const rocsparse_int*   coo_col_ind,
                      rocsparse_index_base   idx_base,
                      rocsparse_matrix_type  matrix_type,
                      rocsparse_fill_mode    uplo,
                      rocsparse_storage_mode storage,
                      size_t*                buffer_size);

REAL_COMPLEX_TEMPLATE(check_matrix_coo,
                      rocsparse_handle       handle,
                      rocsparse_int          m,
                      rocsparse_int          n,
                      rocsparse_int          nnz,
                      const T*               coo_val,
                      const rocsparse_int*   coo_row_ind,
                      const rocsparse_int*   coo_col_ind,
                      rocsparse_index_base   idx_base,
                      rocsparse_matrix_type  matrix_type,
                      rocsparse_fill_mode    uplo,
                      rocsparse_storage_mode storage,
                      rocsparse_data_status* data_status,
                      void*                  temp_buffer);

// check_matrix_gebsr
REAL_COMPLEX_TEMPLATE(check_matrix_gebsr_buffer_size,
                      rocsparse_handle       handle,
                      rocsparse_direction    dir,
                      rocsparse_int          mb,
                      rocsparse_int          nb,
                      rocsparse_int          nnzb,
                      rocsparse_int          row_block_dim,
                      rocsparse_int          col_block_dim,
                      const T*               bsr_val,
                      const rocsparse_int*   bsr_row_ptr,
                      const rocsparse_int*   bsr_col_ind,
                      rocsparse_index_base   idx_base,
                      rocsparse_matrix_type  matrix_type,
                      rocsparse_fill_mode    uplo,
                      rocsparse_storage_mode storage,
                      size_t*                buffer_size);

REAL_COMPLEX_TEMPLATE(check_matrix_gebsr,
                      rocsparse_handle       handle,
                      rocsparse_direction    dir,
                      rocsparse_int          mb,
                      rocsparse_int          nb,
                      rocsparse_int          nnzb,
                      rocsparse_int          row_block_dim,
                      rocsparse_int          col_block_dim,
                      const T*               bsr_val,
                      const rocsparse_int*   bsr_row_ptr,
                      const rocsparse_int*   bsr_col_ind,
                      rocsparse_index_base   idx_base,
                      rocsparse_matrix_type  matrix_type,
                      rocsparse_fill_mode    uplo,
                      rocsparse_storage_mode storage,
                      rocsparse_data_status* data_status,
                      void*                  temp_buffer);

// check_matrix_gebsc
REAL_COMPLEX_TEMPLATE(check_matrix_gebsc_buffer_size,
                      rocsparse_handle       handle,
                      rocsparse_direction    dir,
                      rocsparse_int          mb,
                      rocsparse_int          nb,
                      rocsparse_int          nnzb,
                      rocsparse_int          row_block_dim,
                      rocsparse_int          col_block_dim,
                      const T*               bsc_val,
                      const rocsparse_int*   bsc_col_ptr,
                      const rocsparse_int*   bsc_row_ind,
                      rocsparse_index_base   idx_base,
                      rocsparse_matrix_type  matrix_type,
                      rocsparse_fill_mode    uplo,
                      rocsparse_storage_mode storage,
                      size_t*                buffer_size);

REAL_COMPLEX_TEMPLATE(check_matrix_gebsc,
                      rocsparse_handle       handle,
                      rocsparse_direction    dir,
                      rocsparse_int          mb,
                      rocsparse_int          nb,
                      rocsparse_int          nnzb,
                      rocsparse_int          row_block_dim,
                      rocsparse_int          col_block_dim,
                      const T*               bsc_val,
                      const rocsparse_int*   bsc_col_ptr,
                      const rocsparse_int*   bsc_row_ind,
                      rocsparse_index_base   idx_base,
                      rocsparse_matrix_type  matrix_type,
                      rocsparse_fill_mode    uplo,
                      rocsparse_storage_mode storage,
                      rocsparse_data_status* data_status,
                      void*                  temp_buffer);

// check_matrix_csc
REAL_COMPLEX_TEMPLATE(check_matrix_csc_buffer_size,
                      rocsparse_handle       handle,
                      rocsparse_int          m,
                      rocsparse_int          n,
                      rocsparse_int          nnz,
                      const T*               csc_val,
                      const rocsparse_int*   csc_col_ptr,
                      const rocsparse_int*   csc_row_ind,
                      rocsparse_index_base   idx_base,
                      rocsparse_matrix_type  matrix_type,
                      rocsparse_fill_mode    uplo,
                      rocsparse_storage_mode storage,
                      size_t*                buffer_size);

REAL_COMPLEX_TEMPLATE(check_matrix_csc,
                      rocsparse_handle       handle,
                      rocsparse_int          m,
                      rocsparse_int          n,
                      rocsparse_int          nnz,
                      const T*               csc_val,
                      const rocsparse_int*   csc_col_ptr,
                      const rocsparse_int*   csc_row_ind,
                      rocsparse_index_base   idx_base,
                      rocsparse_matrix_type  matrix_type,
                      rocsparse_fill_mode    uplo,
                      rocsparse_storage_mode storage,
                      rocsparse_data_status* data_status,
                      void*                  temp_buffer);

// check_matrix_ell
REAL_COMPLEX_TEMPLATE(check_matrix_ell_buffer_size,
                      rocsparse_handle       handle,
                      rocsparse_int          m,
                      rocsparse_int          n,
                      rocsparse_int          ell_width,
                      const T*               ell_val,
                      const rocsparse_int*   ell_col_ind,
                      rocsparse_index_base   idx_base,
                      rocsparse_matrix_type  matrix_type,
                      rocsparse_fill_mode    uplo,
                      rocsparse_storage_mode storage,
                      size_t*                buffer_size);

REAL_COMPLEX_TEMPLATE(check_matrix_ell,
                      rocsparse_handle       handle,
                      rocsparse_int          m,
                      rocsparse_int          n,
                      rocsparse_int          ell_width,
                      const T*               ell_val,
                      const rocsparse_int*   ell_col_ind,
                      rocsparse_index_base   idx_base,
                      rocsparse_matrix_type  matrix_type,
                      rocsparse_fill_mode    uplo,
                      rocsparse_storage_mode storage,
                      rocsparse_data_status* data_status,
                      void*                  temp_buffer);

/*
 * ===========================================================================
 *    level 1 SPARSE
 * ===========================================================================
 */
// axpyi
REAL_COMPLEX_TEMPLATE(axpyi,
                      rocsparse_handle     handle,
                      rocsparse_int        nnz,
                      const T*             alpha,
                      const T*             x_vec,
                      const rocsparse_int* x_ind,
                      T*                   y,
                      rocsparse_index_base idx_base);

// doti
REAL_COMPLEX_TEMPLATE(doti,
                      rocsparse_handle     handle,
                      rocsparse_int        nnz,
                      const T*             x_val,
                      const rocsparse_int* x_ind,
                      const T*             y,
                      T*                   result,
                      rocsparse_index_base idx_base);

// dotci
COMPLEX_TEMPLATE(dotci,
                 rocsparse_handle     handle,
                 rocsparse_int        nnz,
                 const T*             x_val,
                 const rocsparse_int* x_ind,
                 const T*             y,
                 T*                   result,
                 rocsparse_index_base idx_base);

// gthr
REAL_COMPLEX_TEMPLATE(gthr,
                      rocsparse_handle     handle,
                      rocsparse_int        nnz,
                      const T*             y,
                      T*                   x_val,
                      const rocsparse_int* x_ind,
                      rocsparse_index_base idx_base);

// gthrz
REAL_COMPLEX_TEMPLATE(gthrz,
                      rocsparse_handle     handle,
                      rocsparse_int        nnz,
                      T*                   y,
                      T*                   x_val,
                      const rocsparse_int* x_ind,
                      rocsparse_index_base idx_base);

// roti
REAL_TEMPLATE(roti,
              rocsparse_handle     handle,
              rocsparse_int        nnz,
              T*                   x_val,
              const rocsparse_int* x_ind,
              T*                   y,
              const T*             c,
              const T*             s,
              rocsparse_index_base idx_base);

// sctr
REAL_COMPLEX_TEMPLATE(sctr,
                      rocsparse_handle     handle,
                      rocsparse_int        nnz,
                      const T*             x_val,
                      const rocsparse_int* x_ind,
                      T*                   y,
                      rocsparse_index_base idx_base);

/*
 * ===========================================================================
 *    level 2 SPARSE
 * ===========================================================================
 */
// bsrmv_analysis
REAL_COMPLEX_TEMPLATE(bsrmv_analysis,
                      rocsparse_handle          handle,
                      rocsparse_direction       dir,
                      rocsparse_operation       trans,
                      rocsparse_int             mb,
                      rocsparse_int             nb,
                      rocsparse_int             nnzb,
                      const rocsparse_mat_descr descr,
                      const T*                  bsr_val,
                      const rocsparse_int*      bsr_row_ptr,
                      const rocsparse_int*      bsr_col_ind,
                      rocsparse_int             bsr_dim,
                      rocsparse_mat_info        info);

// bsrmv
REAL_COMPLEX_TEMPLATE(bsrmv,
                      rocsparse_handle          handle,
                      rocsparse_direction       dir,
                      rocsparse_operation       trans,
                      rocsparse_int             mb,
                      rocsparse_int             nb,
                      rocsparse_int             nnzb,
                      const T*                  alpha,
                      const rocsparse_mat_descr descr,
                      const T*                  bsr_val,
                      const rocsparse_int*      bsr_row_ptr,
                      const rocsparse_int*      bsr_col_ind,
                      rocsparse_int             bsr_dim,
                      rocsparse_mat_info        info,
                      const T*                  x,
                      const T*                  beta,
                      T*                        y);

// bsrxmv
REAL_COMPLEX_TEMPLATE(bsrxmv,
                      rocsparse_handle          handle,
                      rocsparse_direction       dir,
                      rocsparse_operation       trans,
                      rocsparse_int             size_of_mask,
                      rocsparse_int             mb,
                      rocsparse_int             nb,
                      rocsparse_int             nnzb,
                      const T*                  alpha,
                      const rocsparse_mat_descr descr,
                      const T*                  bsr_val,
                      const rocsparse_int*      bsr_mask_ptr,
                      const rocsparse_int*      bsr_row_ptr,
                      const rocsparse_int*      bsr_end_ptr,
                      const rocsparse_int*      bsr_col_ind,
                      rocsparse_int             bsr_dim,
                      const T*                  x,
                      const T*                  beta,
                      T*                        y);

// bsrsv
REAL_COMPLEX_TEMPLATE(bsrsv_buffer_size,
                      rocsparse_handle          handle,
                      rocsparse_direction       dir,
                      rocsparse_operation       trans,
                      rocsparse_int             mb,
                      rocsparse_int             nnzb,
                      const rocsparse_mat_descr descr,
                      const T*                  bsr_val,
                      const rocsparse_int*      bsr_row_ptr,
                      const rocsparse_int*      bsr_col_ind,
                      rocsparse_int             bsr_dim,
                      rocsparse_mat_info        info,
                      size_t*                   buffer_size);

REAL_COMPLEX_TEMPLATE(bsrsv_analysis,
                      rocsparse_handle          handle,
                      rocsparse_direction       dir,
                      rocsparse_operation       trans,
                      rocsparse_int             mb,
                      rocsparse_int             nnzb,
                      const rocsparse_mat_descr descr,
                      const T*                  bsr_val,
                      const rocsparse_int*      bsr_row_ptr,
                      const rocsparse_int*      bsr_col_ind,
                      rocsparse_int             bsr_dim,
                      rocsparse_mat_info        info,
                      rocsparse_analysis_policy analysis,
                      rocsparse_solve_policy    solve,
                      void*                     temp_buffer);

REAL_COMPLEX_TEMPLATE(bsrsv_solve,
                      rocsparse_handle          handle,
                      rocsparse_direction       dir,
                      rocsparse_operation       trans,
                      rocsparse_int             mb,
                      rocsparse_int             nnzb,
                      const T*                  alpha,
                      const rocsparse_mat_descr descr,
                      const T*                  bsr_val,
                      const rocsparse_int*      bsr_row_ptr,
                      const rocsparse_int*      bsr_col_ind,
                      rocsparse_int             bsr_dim,
                      rocsparse_mat_info        info,
                      const T*                  x,
                      T*                        y,
                      rocsparse_solve_policy    policy,
                      void*                     temp_buffer);

// coomv
REAL_COMPLEX_TEMPLATE(coomv,
                      rocsparse_handle          handle,
                      rocsparse_operation       trans,
                      rocsparse_int             m,
                      rocsparse_int             n,
                      rocsparse_int             nnz,
                      const T*                  alpha,
                      const rocsparse_mat_descr descr,
                      const T*                  coo_val,
                      const rocsparse_int*      coo_row_ind,
                      const rocsparse_int*      coo_col_ind,
                      const T*                  x,
                      const T*                  beta,
                      T*                        y);

// csrmv
REAL_COMPLEX_TEMPLATE(csrmv_analysis,
                      rocsparse_handle          handle,
                      rocsparse_operation       trans,
                      rocsparse_int             m,
                      rocsparse_int             n,
                      rocsparse_int             nnz,
                      const rocsparse_mat_descr descr,
                      const T*                  csr_val,
                      const rocsparse_int*      csr_row_ptr,
                      const rocsparse_int*      csr_col_ind,
                      rocsparse_mat_info        info);

REAL_COMPLEX_TEMPLATE(csrmv,
                      rocsparse_handle          handle,
                      rocsparse_operation       trans,
                      rocsparse_int             m,
                      rocsparse_int             n,
                      rocsparse_int             nnz,
                      const T*                  alpha,
                      const rocsparse_mat_descr descr,
                      const T*                  csr_val,
                      const rocsparse_int*      csr_row_ptr,
                      const rocsparse_int*      csr_col_ind,
                      rocsparse_mat_info        info,
                      const T*                  x,
                      const T*                  beta,
                      T*                        y);

// csrsv
REAL_COMPLEX_TEMPLATE(csrsv_buffer_size,
                      rocsparse_handle          handle,
                      rocsparse_operation       trans,
                      rocsparse_int             m,
                      rocsparse_int             nnz,
                      const rocsparse_mat_descr descr,
                      const T*                  csr_val,
                      const rocsparse_int*      csr_row_ptr,
                      const rocsparse_int*      csr_col_ind,
                      rocsparse_mat_info        info,
                      size_t*                   buffer_size);

REAL_COMPLEX_TEMPLATE(csrsv_analysis,
                      rocsparse_handle          handle,
                      rocsparse_operation       trans,
                      rocsparse_int             m,
                      rocsparse_int             nnz,
                      const rocsparse_mat_descr descr,
                      const T*                  csr_val,
                      const rocsparse_int*      csr_row_ptr,
                      const rocsparse_int*      csr_col_ind,
                      rocsparse_mat_info        info,
                      rocsparse_analysis_policy analysis,
                      rocsparse_solve_policy    solve,
                      void*                     temp_buffer);

REAL_COMPLEX_TEMPLATE(csrsv_solve,
                      rocsparse_handle          handle,
                      rocsparse_operation       trans,
                      rocsparse_int             m,
                      rocsparse_int             nnz,
                      const T*                  alpha,
                      const rocsparse_mat_descr descr,
                      const T*                  csr_val,
                      const rocsparse_int*      csr_row_ptr,
                      const rocsparse_int*      csr_col_ind,
                      rocsparse_mat_info        info,
                      const T*                  x,
                      T*                        y,
                      rocsparse_solve_policy    policy,
                      void*                     temp_buffer);

// csritsv
REAL_COMPLEX_TEMPLATE(csritsv_buffer_size,
                      rocsparse_handle          handle,
                      rocsparse_operation       trans,
                      rocsparse_int             m,
                      rocsparse_int             nnz,
                      const rocsparse_mat_descr descr,
                      const T*                  csr_val,
                      const rocsparse_int*      csr_row_ptr,
                      const rocsparse_int*      csr_col_ind,
                      rocsparse_mat_info        info,
                      size_t*                   buffer_size);

REAL_COMPLEX_TEMPLATE(csritsv_analysis,
                      rocsparse_handle          handle,
                      rocsparse_operation       trans,
                      rocsparse_int             m,
                      rocsparse_int             nnz,
                      const rocsparse_mat_descr descr,
                      const T*                  csr_val,
                      const rocsparse_int*      csr_row_ptr,
                      const rocsparse_int*      csr_col_ind,
                      rocsparse_mat_info        info,
                      rocsparse_analysis_policy analysis,
                      rocsparse_solve_policy    solve,
                      void*                     temp_buffer);

REAL_COMPLEX_TEMPLATE(csritsv_solve,
                      rocsparse_handle          handle,
                      rocsparse_int*            nmaxiter,
                      const floating_data_t<T>* host_tol,
                      floating_data_t<T>*       host_history,
                      rocsparse_operation       trans,
                      rocsparse_int             m,
                      rocsparse_int             nnz,
                      const T*                  alpha,
                      const rocsparse_mat_descr descr,
                      const T*                  csr_val,
                      const rocsparse_int*      csr_row_ptr,
                      const rocsparse_int*      csr_col_ind,
                      rocsparse_mat_info        info,
                      const T*                  x,
                      T*                        y,
                      rocsparse_solve_policy    policy,
                      void*                     temp_buffer);

REAL_COMPLEX_TEMPLATE(csritsv_solve_ex,
                      rocsparse_handle          handle,
                      rocsparse_int*            nmaxiter,
                      rocsparse_int             nfreeiter,
                      const floating_data_t<T>* host_tol,
                      floating_data_t<T>*       host_history,
                      rocsparse_operation       trans,
                      rocsparse_int             m,
                      rocsparse_int             nnz,
                      const T*                  alpha,
                      const rocsparse_mat_descr descr,
                      const T*                  csr_val,
                      const rocsparse_int*      csr_row_ptr,
                      const rocsparse_int*      csr_col_ind,
                      rocsparse_mat_info        info,
                      const T*                  x,
                      T*                        y,
                      rocsparse_solve_policy    policy,
                      void*                     temp_buffer);

// ellmv
REAL_COMPLEX_TEMPLATE(ellmv,
                      rocsparse_handle          handle,
                      rocsparse_operation       trans,
                      rocsparse_int             m,
                      rocsparse_int             n,
                      const T*                  alpha,
                      const rocsparse_mat_descr descr,
                      const T*                  ell_val,
                      const rocsparse_int*      ell_col_ind,
                      rocsparse_int             ell_width,
                      const T*                  x,
                      const T*                  beta,
                      T*                        y);

// hybmv
REAL_COMPLEX_TEMPLATE(hybmv,
                      rocsparse_handle          handle,
                      rocsparse_operation       trans,
                      const T*                  alpha,
                      const rocsparse_mat_descr descr,
                      const rocsparse_hyb_mat   hyb,
                      const T*                  x,
                      const T*                  beta,
                      T*                        y);

// gebsrmv
REAL_COMPLEX_TEMPLATE(gebsrmv,
                      rocsparse_handle          handle,
                      rocsparse_direction       dir,
                      rocsparse_operation       trans,
                      rocsparse_int             mb,
                      rocsparse_int             nb,
                      rocsparse_int             nnzb,
                      const T*                  alpha,
                      const rocsparse_mat_descr descr,
                      const T*                  bsr_val,
                      const rocsparse_int*      bsr_row_ptr,
                      const rocsparse_int*      bsr_col_ind,
                      rocsparse_int             row_block_dim,
                      rocsparse_int             col_block_dim,
                      const T*                  x,
                      const T*                  beta,
                      T*                        y);

// gemvi
REAL_COMPLEX_TEMPLATE(gemvi_buffer_size,
                      rocsparse_handle    handle,
                      rocsparse_operation trans,
                      rocsparse_int       m,
                      rocsparse_int       n,
                      rocsparse_int       nnz,
                      size_t*             buffer_size);

REAL_COMPLEX_TEMPLATE(gemvi,
                      rocsparse_handle     handle,
                      rocsparse_operation  trans,
                      rocsparse_int        m,
                      rocsparse_int        n,
                      const T*             alpha,
                      const T*             A,
                      rocsparse_int        lda,
                      rocsparse_int        nnz,
                      const T*             x_val,
                      const rocsparse_int* x_ind,
                      const T*             beta,
                      T*                   y,
                      rocsparse_index_base idx_base,
                      void*                temp_buffer);

/*
 * ===========================================================================
 *    level 3 SPARSE
 * ===========================================================================
 */
// bsrmm
REAL_COMPLEX_TEMPLATE(bsrmm,
                      rocsparse_handle          handle,
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
                      rocsparse_int             block_dim,
                      const T*                  B,
                      rocsparse_int             ldb,
                      const T*                  beta,
                      T*                        C,
                      rocsparse_int             ldc);

// gebsrmm
REAL_COMPLEX_TEMPLATE(gebsrmm,
                      rocsparse_handle          handle,
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
                      rocsparse_int             ldb,
                      const T*                  beta,
                      T*                        C,
                      rocsparse_int             ldc);

// csrmm
REAL_COMPLEX_TEMPLATE(csrmm,
                      rocsparse_handle          handle,
                      rocsparse_operation       trans_A,
                      rocsparse_operation       trans_B,
                      rocsparse_int             m,
                      rocsparse_int             n,
                      rocsparse_int             k,
                      rocsparse_int             nnz,
                      const T*                  alpha,
                      const rocsparse_mat_descr descr,
                      const T*                  csr_val,
                      const rocsparse_int*      csr_row_ptr,
                      const rocsparse_int*      csr_col_ind,
                      const T*                  B,
                      rocsparse_int             ldb,
                      const T*                  beta,
                      T*                        C,
                      rocsparse_int             ldc);

// csrsm
REAL_COMPLEX_TEMPLATE(csrsm_buffer_size,
                      rocsparse_handle          handle,
                      rocsparse_operation       trans_A,
                      rocsparse_operation       trans_B,
                      rocsparse_int             m,
                      rocsparse_int             nrhs,
                      rocsparse_int             nnz,
                      const T*                  alpha,
                      const rocsparse_mat_descr descr,
                      const T*                  csr_val,
                      const rocsparse_int*      csr_row_ptr,
                      const rocsparse_int*      csr_col_ind,
                      const T*                  B,
                      rocsparse_int             ldb,
                      rocsparse_mat_info        info,
                      rocsparse_solve_policy    policy,
                      size_t*                   buffer_size);

REAL_COMPLEX_TEMPLATE(csrsm_analysis,
                      rocsparse_handle          handle,
                      rocsparse_operation       trans_A,
                      rocsparse_operation       trans_B,
                      rocsparse_int             m,
                      rocsparse_int             nrhs,
                      rocsparse_int             nnz,
                      const T*                  alpha,
                      const rocsparse_mat_descr descr,
                      const T*                  csr_val,
                      const rocsparse_int*      csr_row_ptr,
                      const rocsparse_int*      csr_col_ind,
                      const T*                  B,
                      rocsparse_int             ldb,
                      rocsparse_mat_info        info,
                      rocsparse_analysis_policy analysis,
                      rocsparse_solve_policy    solve,
                      void*                     temp_buffer);

REAL_COMPLEX_TEMPLATE(csrsm_solve,
                      rocsparse_handle          handle,
                      rocsparse_operation       trans_A,
                      rocsparse_operation       trans_B,
                      rocsparse_int             m,
                      rocsparse_int             nrhs,
                      rocsparse_int             nnz,
                      const T*                  alpha,
                      const rocsparse_mat_descr descr,
                      const T*                  csr_val,
                      const rocsparse_int*      csr_row_ptr,
                      const rocsparse_int*      csr_col_ind,
                      T*                        B,
                      rocsparse_int             ldb,
                      rocsparse_mat_info        info,
                      rocsparse_solve_policy    policy,
                      void*                     temp_buffer);

// bsrsm
REAL_COMPLEX_TEMPLATE(bsrsm_buffer_size,
                      rocsparse_handle          handle,
                      rocsparse_direction       dir,
                      rocsparse_operation       trans_A,
                      rocsparse_operation       trans_X,
                      rocsparse_int             mb,
                      rocsparse_int             nrhs,
                      rocsparse_int             nnzb,
                      const rocsparse_mat_descr descr,
                      const T*                  bsr_val,
                      const rocsparse_int*      bsr_row_ptr,
                      const rocsparse_int*      bsr_col_ind,
                      rocsparse_int             bsr_dim,
                      rocsparse_mat_info        info,
                      size_t*                   buffer_size);

REAL_COMPLEX_TEMPLATE(bsrsm_analysis,
                      rocsparse_handle          handle,
                      rocsparse_direction       dir,
                      rocsparse_operation       trans_A,
                      rocsparse_operation       trans_X,
                      rocsparse_int             mb,
                      rocsparse_int             nrhs,
                      rocsparse_int             nnzb,
                      const rocsparse_mat_descr descr,
                      const T*                  bsr_val,
                      const rocsparse_int*      bsr_row_ptr,
                      const rocsparse_int*      bsr_col_ind,
                      rocsparse_int             bsr_dim,
                      rocsparse_mat_info        info,
                      rocsparse_analysis_policy analysis,
                      rocsparse_solve_policy    solve,
                      void*                     temp_buffer);

REAL_COMPLEX_TEMPLATE(bsrsm_solve,
                      rocsparse_handle          handle,
                      rocsparse_direction       dir,
                      rocsparse_operation       trans_A,
                      rocsparse_operation       trans_X,
                      rocsparse_int             mb,
                      rocsparse_int             nrhs,
                      rocsparse_int             nnzb,
                      const T*                  alpha,
                      const rocsparse_mat_descr descr,
                      const T*                  bsr_val,
                      const rocsparse_int*      bsr_row_ptr,
                      const rocsparse_int*      bsr_col_ind,
                      rocsparse_int             bsr_dim,
                      rocsparse_mat_info        info,
                      const T*                  B,
                      rocsparse_int             ldb,
                      T*                        X,
                      rocsparse_int             ldx,
                      rocsparse_solve_policy    policy,
                      void*                     temp_buffer);

// gemmi
REAL_COMPLEX_TEMPLATE(gemmi,
                      rocsparse_handle          handle,
                      rocsparse_operation       trans_A,
                      rocsparse_operation       trans_B,
                      rocsparse_int             m,
                      rocsparse_int             n,
                      rocsparse_int             k,
                      rocsparse_int             nnz,
                      const T*                  alpha,
                      const T*                  A,
                      rocsparse_int             lda,
                      const rocsparse_mat_descr descr,
                      const T*                  csr_val,
                      const rocsparse_int*      csr_row_ptr,
                      const rocsparse_int*      csr_col_ind,
                      const T*                  beta,
                      T*                        C,
                      rocsparse_int             ldc);

/*
 * ===========================================================================
 *    extra SPARSE
 * ===========================================================================
 */
// bsrgeam
REAL_COMPLEX_TEMPLATE(bsrgeam,
                      rocsparse_handle          handle,
                      rocsparse_direction       dir,
                      rocsparse_int             mb,
                      rocsparse_int             nb,
                      rocsparse_int             block_dim,
                      const T*                  alpha,
                      const rocsparse_mat_descr descr_A,
                      rocsparse_int             nnzb_A,
                      const T*                  bsr_val_A,
                      const rocsparse_int*      bsr_row_ptr_A,
                      const rocsparse_int*      bsr_col_ind_A,
                      const T*                  beta,
                      const rocsparse_mat_descr descr_B,
                      rocsparse_int             nnzb_B,
                      const T*                  bsr_val_B,
                      const rocsparse_int*      bsr_row_ptr_B,
                      const rocsparse_int*      bsr_col_ind_B,
                      const rocsparse_mat_descr descr_C,
                      T*                        bsr_val_C,
                      const rocsparse_int*      bsr_row_ptr_C,
                      rocsparse_int*            bsr_col_ind_C);

// bsrgemm
REAL_COMPLEX_TEMPLATE(bsrgemm_buffer_size,
                      rocsparse_handle          handle,
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
                      size_t*                   buffer_size);

REAL_COMPLEX_TEMPLATE(bsrgemm,
                      rocsparse_handle          handle,
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
                      void*                     temp_buffer);

// csrgeam
REAL_COMPLEX_TEMPLATE(csrgeam,
                      rocsparse_handle          handle,
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
                      rocsparse_int*            csr_col_ind_C);

// csrgemm
REAL_COMPLEX_TEMPLATE(csrgemm_buffer_size,
                      rocsparse_handle          handle,
                      rocsparse_operation       trans_A,
                      rocsparse_operation       trans_B,
                      rocsparse_int             m,
                      rocsparse_int             n,
                      rocsparse_int             k,
                      const T*                  alpha,
                      const rocsparse_mat_descr descr_A,
                      rocsparse_int             nnz_A,
                      const rocsparse_int*      csr_row_ptr_A,
                      const rocsparse_int*      csr_col_ind_A,
                      const rocsparse_mat_descr descr_B,
                      rocsparse_int             nnz_B,
                      const rocsparse_int*      csr_row_ptr_B,
                      const rocsparse_int*      csr_col_ind_B,
                      const T*                  beta,
                      const rocsparse_mat_descr descr_D,
                      rocsparse_int             nnz_D,
                      const rocsparse_int*      csr_row_ptr_D,
                      const rocsparse_int*      csr_col_ind_D,
                      rocsparse_mat_info        info_C,
                      size_t*                   buffer_size);

REAL_COMPLEX_TEMPLATE(csrgemm,
                      rocsparse_handle          handle,
                      rocsparse_operation       trans_A,
                      rocsparse_operation       trans_B,
                      rocsparse_int             m,
                      rocsparse_int             n,
                      rocsparse_int             k,
                      const T*                  alpha,
                      const rocsparse_mat_descr descr_A,
                      rocsparse_int             nnz_A,
                      const T*                  csr_val_A,
                      const rocsparse_int*      csr_row_ptr_A,
                      const rocsparse_int*      csr_col_ind_A,
                      const rocsparse_mat_descr descr_B,
                      rocsparse_int             nnz_B,
                      const T*                  csr_val_B,
                      const rocsparse_int*      csr_row_ptr_B,
                      const rocsparse_int*      csr_col_ind_B,
                      const T*                  beta,
                      const rocsparse_mat_descr descr_D,
                      rocsparse_int             nnz_D,
                      const T*                  csr_val_D,
                      const rocsparse_int*      csr_row_ptr_D,
                      const rocsparse_int*      csr_col_ind_D,
                      const rocsparse_mat_descr descr_C,
                      T*                        csr_val_C,
                      const rocsparse_int*      csr_row_ptr_C,
                      rocsparse_int*            csr_col_ind_C,
                      const rocsparse_mat_info  info_C,
                      void*                     temp_buffer);

REAL_COMPLEX_TEMPLATE(csrgemm_numeric,
                      rocsparse_handle          handle,
                      rocsparse_operation       trans_A,
                      rocsparse_operation       trans_B,
                      rocsparse_int             m,
                      rocsparse_int             n,
                      rocsparse_int             k,
                      const T*                  alpha,
                      const rocsparse_mat_descr descr_A,
                      rocsparse_int             nnz_A,
                      const T*                  csr_val_A,
                      const rocsparse_int*      csr_row_ptr_A,
                      const rocsparse_int*      csr_col_ind_A,
                      const rocsparse_mat_descr descr_B,
                      rocsparse_int             nnz_B,
                      const T*                  csr_val_B,
                      const rocsparse_int*      csr_row_ptr_B,
                      const rocsparse_int*      csr_col_ind_B,
                      const T*                  beta,
                      const rocsparse_mat_descr descr_D,
                      rocsparse_int             nnz_D,
                      const T*                  csr_val_D,
                      const rocsparse_int*      csr_row_ptr_D,
                      const rocsparse_int*      csr_col_ind_D,
                      const rocsparse_mat_descr descr_C,
                      rocsparse_int             nnz_C,
                      T*                        csr_val_C,
                      const rocsparse_int*      csr_row_ptr_C,
                      const rocsparse_int*      csr_col_ind_C,
                      const rocsparse_mat_info  info_C,
                      void*                     temp_buffer);

/*
 * ===========================================================================
 *    precond SPARSE
 * ===========================================================================
 */

// bsric0
REAL_COMPLEX_TEMPLATE(bsric0_buffer_size,
                      rocsparse_handle          handle,
                      rocsparse_direction       dir,
                      rocsparse_int             mb,
                      rocsparse_int             nnzb,
                      const rocsparse_mat_descr descr,
                      const T*                  bsr_val,
                      const rocsparse_int*      bsr_row_ptr,
                      const rocsparse_int*      bsr_col_ind,
                      rocsparse_int             block_dim,
                      rocsparse_mat_info        info,
                      size_t*                   buffer_size);

REAL_COMPLEX_TEMPLATE(bsric0_analysis,
                      rocsparse_handle          handle,
                      rocsparse_direction       dir,
                      rocsparse_int             mb,
                      rocsparse_int             nnzb,
                      const rocsparse_mat_descr descr,
                      const T*                  bsr_val,
                      const rocsparse_int*      bsr_row_ptr,
                      const rocsparse_int*      bsr_col_ind,
                      rocsparse_int             block_dim,
                      rocsparse_mat_info        info,
                      rocsparse_analysis_policy analysis,
                      rocsparse_solve_policy    solve,
                      void*                     temp_buffer);

REAL_COMPLEX_TEMPLATE(bsric0,
                      rocsparse_handle          handle,
                      rocsparse_direction       dir,
                      rocsparse_int             mb,
                      rocsparse_int             nnzb,
                      const rocsparse_mat_descr descr,
                      T*                        bsr_val,
                      const rocsparse_int*      bsr_row_ptr,
                      const rocsparse_int*      bsr_col_ind,
                      rocsparse_int             block_dim,
                      rocsparse_mat_info        info,
                      rocsparse_solve_policy    policy,
                      void*                     temp_buffer);

// bsrilu0
REAL_COMPLEX_TEMPLATE(bsrilu0_buffer_size,
                      rocsparse_handle          handle,
                      rocsparse_direction       dir,
                      rocsparse_int             mb,
                      rocsparse_int             nnzb,
                      const rocsparse_mat_descr descr,
                      const T*                  bsr_val,
                      const rocsparse_int*      bsr_row_ptr,
                      const rocsparse_int*      bsr_col_ind,
                      rocsparse_int             block_dim,
                      rocsparse_mat_info        info,
                      size_t*                   buffer_size);

REAL_COMPLEX_TEMPLATE(bsrilu0_numeric_boost,
                      rocsparse_handle          handle,
                      rocsparse_mat_info        info,
                      int                       enable_boost,
                      const floating_data_t<T>* boost_tol,
                      const T*                  boost_val);

REAL_COMPLEX_TEMPLATE(bsrilu0_analysis,
                      rocsparse_handle          handle,
                      rocsparse_direction       dir,
                      rocsparse_int             mb,
                      rocsparse_int             nnzb,
                      const rocsparse_mat_descr descr,
                      const T*                  bsr_val,
                      const rocsparse_int*      bsr_row_ptr,
                      const rocsparse_int*      bsr_col_ind,
                      rocsparse_int             block_dim,
                      rocsparse_mat_info        info,
                      rocsparse_analysis_policy analysis,
                      rocsparse_solve_policy    solve,
                      void*                     temp_buffer);

REAL_COMPLEX_TEMPLATE(bsrilu0,
                      rocsparse_handle          handle,
                      rocsparse_direction       dir,
                      rocsparse_int             mb,
                      rocsparse_int             nnzb,
                      const rocsparse_mat_descr descr,
                      T*                        bsr_val,
                      const rocsparse_int*      bsr_row_ptr,
                      const rocsparse_int*      bsr_col_ind,
                      rocsparse_int             block_dim,
                      rocsparse_mat_info        info,
                      rocsparse_solve_policy    policy,
                      void*                     temp_buffer);

// csric0
REAL_COMPLEX_TEMPLATE(csric0_buffer_size,
                      rocsparse_handle          handle,
                      rocsparse_int             m,
                      rocsparse_int             nnz,
                      const rocsparse_mat_descr descr,
                      const T*                  csr_val,
                      const rocsparse_int*      csr_row_ptr,
                      const rocsparse_int*      csr_col_ind,
                      rocsparse_mat_info        info,
                      size_t*                   buffer_size);

REAL_COMPLEX_TEMPLATE(csric0_analysis,
                      rocsparse_handle          handle,
                      rocsparse_int             m,
                      rocsparse_int             nnz,
                      const rocsparse_mat_descr descr,
                      const T*                  csr_val,
                      const rocsparse_int*      csr_row_ptr,
                      const rocsparse_int*      csr_col_ind,
                      rocsparse_mat_info        info,
                      rocsparse_analysis_policy analysis,
                      rocsparse_solve_policy    solve,
                      void*                     temp_buffer);

REAL_COMPLEX_TEMPLATE(csric0,
                      rocsparse_handle          handle,
                      rocsparse_int             m,
                      rocsparse_int             nnz,
                      const rocsparse_mat_descr descr,
                      T*                        csr_val,
                      const rocsparse_int*      csr_row_ptr,
                      const rocsparse_int*      csr_col_ind,
                      rocsparse_mat_info        info,
                      rocsparse_solve_policy    policy,
                      void*                     temp_buffer);

// csritilu0_compute_ex
REAL_COMPLEX_TEMPLATE(csritilu0_compute_ex,
                      rocsparse_handle     handle,
                      rocsparse_itilu0_alg alg,
                      rocsparse_int        options,
                      rocsparse_int*       nmaxiter,
                      rocsparse_int        nfreeiter,
                      floating_data_t<T>   tol_correction,
                      rocsparse_int        m,
                      rocsparse_int        nnz,
                      const rocsparse_int* ptr,
                      const rocsparse_int* ind,
                      const T*             val,
                      T*                   ilu0,
                      rocsparse_index_base base,
                      size_t               buffer_size_,
                      void*                buffer);

// csritilu0_compute
REAL_COMPLEX_TEMPLATE(csritilu0_compute,
                      rocsparse_handle     handle,
                      rocsparse_itilu0_alg alg,
                      rocsparse_int        options,
                      rocsparse_int*       nmaxiter,
                      floating_data_t<T>   tol_correction,
                      rocsparse_int        m,
                      rocsparse_int        nnz,
                      const rocsparse_int* ptr,
                      const rocsparse_int* ind,
                      const T*             val,
                      T*                   ilu0,
                      rocsparse_index_base base,
                      size_t               buffer_size_,
                      void*                buffer);

// csritilu0_update
REAL_COMPLEX_TEMPLATE(csritilu0_history,
                      rocsparse_handle     handle,
                      rocsparse_itilu0_alg alg,
                      rocsparse_int*       niter,
                      floating_data_t<T>*  data,
                      size_t               buffer_size_,
                      void*                buffer);

// csrilu0
REAL_COMPLEX_TEMPLATE(csrilu0_buffer_size,
                      rocsparse_handle          handle,
                      rocsparse_int             m,
                      rocsparse_int             nnz,
                      const rocsparse_mat_descr descr,
                      const T*                  csr_val,
                      const rocsparse_int*      csr_row_ptr,
                      const rocsparse_int*      csr_col_ind,
                      rocsparse_mat_info        info,
                      size_t*                   buffer_size);

REAL_COMPLEX_TEMPLATE(csrilu0_numeric_boost,
                      rocsparse_handle          handle,
                      rocsparse_mat_info        info,
                      int                       enable_boost,
                      const floating_data_t<T>* boost_tol,
                      const T*                  boost_val);

REAL_COMPLEX_TEMPLATE(csrilu0_analysis,
                      rocsparse_handle          handle,
                      rocsparse_int             m,
                      rocsparse_int             nnz,
                      const rocsparse_mat_descr descr,
                      const T*                  csr_val,
                      const rocsparse_int*      csr_row_ptr,
                      const rocsparse_int*      csr_col_ind,
                      rocsparse_mat_info        info,
                      rocsparse_analysis_policy analysis,
                      rocsparse_solve_policy    solve,
                      void*                     temp_buffer);

REAL_COMPLEX_TEMPLATE(csrilu0,
                      rocsparse_handle          handle,
                      rocsparse_int             m,
                      rocsparse_int             nnz,
                      const rocsparse_mat_descr descr,
                      T*                        csr_val,
                      const rocsparse_int*      csr_row_ptr,
                      const rocsparse_int*      csr_col_ind,
                      rocsparse_mat_info        info,
                      rocsparse_solve_policy    policy,
                      void*                     temp_buffer);

REAL_COMPLEX_TEMPLATE(gtsv_buffer_size,
                      rocsparse_handle handle,
                      rocsparse_int    m,
                      rocsparse_int    n,
                      const T*         dl,
                      const T*         d,
                      const T*         du,
                      const T*         B,
                      rocsparse_int    ldb,
                      size_t*          buffer_size);

REAL_COMPLEX_TEMPLATE(gtsv,
                      rocsparse_handle handle,
                      rocsparse_int    m,
                      rocsparse_int    n,
                      const T*         dl,
                      const T*         d,
                      const T*         du,
                      T*               B,
                      rocsparse_int    ldb,
                      void*            temp_buffer);

// gtsv_no_pivot
REAL_COMPLEX_TEMPLATE(gtsv_no_pivot_buffer_size,
                      rocsparse_handle handle,
                      rocsparse_int    m,
                      rocsparse_int    n,
                      const T*         dl,
                      const T*         d,
                      const T*         du,
                      const T*         B,
                      rocsparse_int    ldb,
                      size_t*          buffer_size);

REAL_COMPLEX_TEMPLATE(gtsv_no_pivot,
                      rocsparse_handle handle,
                      rocsparse_int    m,
                      rocsparse_int    n,
                      const T*         dl,
                      const T*         d,
                      const T*         du,
                      T*               B,
                      rocsparse_int    ldb,
                      void*            temp_buffer);

// gtsv_no_pivot_strided_batch
REAL_COMPLEX_TEMPLATE(gtsv_no_pivot_strided_batch_buffer_size,
                      rocsparse_handle handle,
                      rocsparse_int    m,
                      const T*         dl,
                      const T*         d,
                      const T*         du,
                      const T*         x,
                      rocsparse_int    batch_count,
                      rocsparse_int    batch_stride,
                      size_t*          buffer_size);

REAL_COMPLEX_TEMPLATE(gtsv_no_pivot_strided_batch,
                      rocsparse_handle handle,
                      rocsparse_int    m,
                      const T*         dl,
                      const T*         d,
                      const T*         du,
                      T*               x,
                      rocsparse_int    batch_count,
                      rocsparse_int    batch_stride,
                      void*            temp_buffer);

// gtsv_interleaved_batch
REAL_COMPLEX_TEMPLATE(gtsv_interleaved_batch_buffer_size,
                      rocsparse_handle               handle,
                      rocsparse_gtsv_interleaved_alg alg,
                      rocsparse_int                  m,
                      const T*                       dl,
                      const T*                       d,
                      const T*                       du,
                      const T*                       x,
                      rocsparse_int                  batch_count,
                      rocsparse_int                  batch_stride,
                      size_t*                        buffer_size);

REAL_COMPLEX_TEMPLATE(gtsv_interleaved_batch,
                      rocsparse_handle               handle,
                      rocsparse_gtsv_interleaved_alg alg,
                      rocsparse_int                  m,
                      const T*                       dl,
                      const T*                       d,
                      const T*                       du,
                      T*                             x,
                      rocsparse_int                  batch_count,
                      rocsparse_int                  batch_stride,
                      void*                          temp_buffer);

// gpsv_interleaved_batch
REAL_COMPLEX_TEMPLATE(gpsv_interleaved_batch_buffer_size,
                      rocsparse_handle               handle,
                      rocsparse_gpsv_interleaved_alg alg,
                      rocsparse_int                  m,
                      const T*                       ds,
                      const T*                       dl,
                      const T*                       d,
                      const T*                       du,
                      const T*                       dw,
                      const T*                       x,
                      rocsparse_int                  batch_count,
                      rocsparse_int                  batch_stride,
                      size_t*                        buffer_size);

REAL_COMPLEX_TEMPLATE(gpsv_interleaved_batch,
                      rocsparse_handle               handle,
                      rocsparse_gpsv_interleaved_alg alg,
                      rocsparse_int                  m,
                      const T*                       ds,
                      const T*                       dl,
                      const T*                       d,
                      const T*                       du,
                      const T*                       dw,
                      T*                             x,
                      rocsparse_int                  batch_count,
                      rocsparse_int                  batch_stride,
                      void*                          temp_buffer);

/*
 * ===========================================================================
 *    conversion SPARSE
 * ===========================================================================
 */

// nnz
REAL_COMPLEX_TEMPLATE(nnz,
                      rocsparse_handle          handle,
                      rocsparse_direction       dir,
                      rocsparse_int             m,
                      rocsparse_int             n,
                      const rocsparse_mat_descr descr,
                      const T*                  A,
                      rocsparse_int             lda,
                      rocsparse_int*            nnz_per_row_columns,
                      rocsparse_int*            nnz_total_dev_host_ptr);

// nnz_compress
REAL_COMPLEX_TEMPLATE(nnz_compress,
                      rocsparse_handle          handle,
                      rocsparse_int             m,
                      const rocsparse_mat_descr descr_A,
                      const T*                  csr_val_A,
                      const rocsparse_int*      csr_row_ptr_A,
                      rocsparse_int*            nnz_per_row,
                      rocsparse_int*            nnz_C,
                      T                         tol);

// dense2csr
REAL_COMPLEX_TEMPLATE(dense2csr,
                      rocsparse_handle          handle,
                      rocsparse_int             m,
                      rocsparse_int             n,
                      const rocsparse_mat_descr descr,
                      const T*                  A,
                      rocsparse_int             lda,
                      const rocsparse_int*      nnz_per_rows,
                      T*                        csr_val,
                      rocsparse_int*            csr_row_ptr,
                      rocsparse_int*            csr_col_ind);

// dense2coo
REAL_COMPLEX_TEMPLATE(dense2coo,
                      rocsparse_handle          handle,
                      rocsparse_int             m,
                      rocsparse_int             n,
                      const rocsparse_mat_descr descr,
                      const T*                  A,
                      rocsparse_int             lda,
                      const rocsparse_int*      nnz_per_rows,
                      T*                        coo_val,
                      rocsparse_int*            coo_row_ind,
                      rocsparse_int*            coo_col_ind);

// prune_dense2csr_buffer_size
REAL_TEMPLATE(prune_dense2csr_buffer_size,
              rocsparse_handle          handle,
              rocsparse_int             m,
              rocsparse_int             n,
              const T*                  A,
              rocsparse_int             lda,
              const T*                  threshold,
              const rocsparse_mat_descr descr,
              const T*                  csr_val,
              const rocsparse_int*      csr_row_ptr,
              const rocsparse_int*      csr_col_ind,
              size_t*                   buffer_size);

// prune_dense2csr_nnz
REAL_TEMPLATE(prune_dense2csr_nnz,
              rocsparse_handle          handle,
              rocsparse_int             m,
              rocsparse_int             n,
              const T*                  A,
              rocsparse_int             lda,
              const T*                  threshold,
              const rocsparse_mat_descr descr,
              rocsparse_int*            csr_row_ptr,
              rocsparse_int*            nnz_total_dev_host_ptr,
              void*                     temp_buffer);

// prune_dense2csr
REAL_TEMPLATE(prune_dense2csr,
              rocsparse_handle          handle,
              rocsparse_int             m,
              rocsparse_int             n,
              const T*                  A,
              rocsparse_int             lda,
              const T*                  threshold,
              const rocsparse_mat_descr descr,
              T*                        csr_val,
              const rocsparse_int*      csr_row_ptr,
              rocsparse_int*            csr_col_ind,
              void*                     temp_buffer);

// prune_dense2csr_by_percentage_buffer_size
REAL_TEMPLATE(prune_dense2csr_by_percentage_buffer_size,
              rocsparse_handle          handle,
              rocsparse_int             m,
              rocsparse_int             n,
              const T*                  A,
              rocsparse_int             lda,
              T                         percentage,
              const rocsparse_mat_descr descr,
              const T*                  csr_val,
              const rocsparse_int*      csr_row_ptr,
              const rocsparse_int*      csr_col_ind,
              rocsparse_mat_info        info,
              size_t*                   buffer_size);

// prune_dense2csr_nnz_by_percentage
REAL_TEMPLATE(prune_dense2csr_nnz_by_percentage,
              rocsparse_handle          handle,
              rocsparse_int             m,
              rocsparse_int             n,
              const T*                  A,
              rocsparse_int             lda,
              T                         percentage,
              const rocsparse_mat_descr descr,
              rocsparse_int*            csr_row_ptr,
              rocsparse_int*            nnz_total_dev_host_ptr,
              rocsparse_mat_info        info,
              void*                     temp_buffer);

// prune_dense2csr_by_percentage
REAL_TEMPLATE(prune_dense2csr_by_percentage,
              rocsparse_handle          handle,
              rocsparse_int             m,
              rocsparse_int             n,
              const T*                  A,
              rocsparse_int             lda,
              T                         percentage,
              const rocsparse_mat_descr descr,
              T*                        csr_val,
              const rocsparse_int*      csr_row_ptr,
              rocsparse_int*            csr_col_ind,
              rocsparse_mat_info        info,
              void*                     temp_buffer);

// dense2csc
REAL_COMPLEX_TEMPLATE(dense2csc,
                      rocsparse_handle          handle,
                      rocsparse_int             m,
                      rocsparse_int             n,
                      const rocsparse_mat_descr descr,
                      const T*                  A,
                      rocsparse_int             lda,
                      const rocsparse_int*      nnz_per_columns,
                      T*                        csc_val,
                      rocsparse_int*            csc_col_ptr,
                      rocsparse_int*            csc_row_ind);

// csr2dense
REAL_COMPLEX_TEMPLATE(csr2dense,
                      rocsparse_handle          handle,
                      rocsparse_int             m,
                      rocsparse_int             n,
                      const rocsparse_mat_descr descr,
                      const T*                  csr_val,
                      const rocsparse_int*      csr_row_ptr,
                      const rocsparse_int*      csr_col_ind,
                      T*                        A,
                      rocsparse_int             lda);

// csc2dense
REAL_COMPLEX_TEMPLATE(csc2dense,
                      rocsparse_handle          handle,
                      rocsparse_int             m,
                      rocsparse_int             n,
                      const rocsparse_mat_descr descr,
                      const T*                  csc_val,
                      const rocsparse_int*      csc_col_ptr,
                      const rocsparse_int*      csc_row_ind,
                      T*                        A,
                      rocsparse_int             lda);

// coo2dense
REAL_COMPLEX_TEMPLATE(coo2dense,
                      rocsparse_handle          handle,
                      rocsparse_int             m,
                      rocsparse_int             n,
                      rocsparse_int             nnz,
                      const rocsparse_mat_descr descr,
                      const T*                  coo_val,
                      const rocsparse_int*      coo_row_ind,
                      const rocsparse_int*      coo_col_ind,
                      T*                        A,
                      rocsparse_int             lda);

// csr2csc
REAL_COMPLEX_TEMPLATE(csr2csc,
                      rocsparse_handle     handle,
                      rocsparse_int        m,
                      rocsparse_int        n,
                      rocsparse_int        nnz,
                      const T*             csr_val,
                      const rocsparse_int* csr_row_ptr,
                      const rocsparse_int* csr_col_ind,
                      T*                   csc_val,
                      rocsparse_int*       csc_row_ind,
                      rocsparse_int*       csc_col_ptr,
                      rocsparse_action     copy_values,
                      rocsparse_index_base idx_base,
                      void*                temp_buffer);
// gebsr2gebsc
REAL_COMPLEX_TEMPLATE(gebsr2gebsc_buffer_size,
                      rocsparse_handle     handle,
                      rocsparse_int        mb,
                      rocsparse_int        nb,
                      rocsparse_int        nnzb,
                      const T*             bsr_val,
                      const rocsparse_int* bsr_row_ptr,
                      const rocsparse_int* bsr_col_ind,
                      rocsparse_int        row_block_dim,
                      rocsparse_int        col_block_dim,
                      size_t*              p_buffer_size);

REAL_COMPLEX_TEMPLATE(gebsr2gebsc,
                      rocsparse_handle     handle,
                      rocsparse_int        mb,
                      rocsparse_int        nb,
                      rocsparse_int        nnzb,
                      const T*             bsr_val,
                      const rocsparse_int* bsr_row_ptr,
                      const rocsparse_int* bsr_col_ind,
                      rocsparse_int        row_block_dim,
                      rocsparse_int        col_block_dim,
                      T*                   bsc_val,
                      rocsparse_int*       bsc_row_ind,
                      rocsparse_int*       bsc_col_ptr,
                      rocsparse_action     copy_values,
                      rocsparse_index_base idx_base,
                      void*                temp_buffer);

// csr2ell
REAL_COMPLEX_TEMPLATE(csr2ell,
                      rocsparse_handle          handle,
                      rocsparse_int             m,
                      const rocsparse_mat_descr csr_descr,
                      const T*                  csr_val,
                      const rocsparse_int*      csr_row_ptr,
                      const rocsparse_int*      csr_col_ind,
                      const rocsparse_mat_descr ell_descr,
                      rocsparse_int             ell_width,
                      T*                        ell_val,
                      rocsparse_int*            ell_col_ind);

// csr2hyb
REAL_COMPLEX_TEMPLATE(csr2hyb,
                      rocsparse_handle          handle,
                      rocsparse_int             m,
                      rocsparse_int             n,
                      const rocsparse_mat_descr descr,
                      const T*                  csr_val,
                      const rocsparse_int*      csr_row_ptr,
                      const rocsparse_int*      csr_col_ind,
                      rocsparse_hyb_mat         hyb,
                      rocsparse_int             user_ell_width,
                      rocsparse_hyb_partition   partition_type);

// csr2bsr
REAL_COMPLEX_TEMPLATE(csr2bsr,
                      rocsparse_handle          handle,
                      rocsparse_direction       direction,
                      rocsparse_int             m,
                      rocsparse_int             n,
                      const rocsparse_mat_descr csr_descr,
                      const T*                  csr_val,
                      const rocsparse_int*      csr_row_ptr,
                      const rocsparse_int*      csr_col_ind,
                      rocsparse_int             block_dim,
                      const rocsparse_mat_descr bsr_descr,
                      T*                        bsr_val,
                      rocsparse_int*            bsr_row_ptr,
                      rocsparse_int*            bsr_col_ind);

// bsrpad_value
REAL_COMPLEX_TEMPLATE(bsrpad_value,
                      rocsparse_handle          handle,
                      rocsparse_int             m,
                      rocsparse_int             mb,
                      rocsparse_int             nnzb,
                      rocsparse_int             block_dim,
                      T                         value,
                      const rocsparse_mat_descr bsr_descr,
                      T*                        bsr_val,
                      rocsparse_int*            bsr_row_ptr,
                      rocsparse_int*            bsr_col_ind);

// csr2gebsr_buffer_size
REAL_COMPLEX_TEMPLATE(csr2gebsr_buffer_size,
                      rocsparse_handle          handle,
                      rocsparse_direction       direction,
                      rocsparse_int             m,
                      rocsparse_int             n,
                      const rocsparse_mat_descr csr_descr,
                      const T*                  csr_val,
                      const rocsparse_int*      csr_row_ptr,
                      const rocsparse_int*      csr_col_ind,
                      rocsparse_int             row_block_dim,
                      rocsparse_int             col_block_dim,
                      size_t*                   p_buffer_size);

// csr2gebsr
REAL_COMPLEX_TEMPLATE(csr2gebsr,
                      rocsparse_handle          handle,
                      rocsparse_direction       direction,
                      rocsparse_int             m,
                      rocsparse_int             n,
                      const rocsparse_mat_descr csr_descr,
                      const T*                  csr_val,
                      const rocsparse_int*      csr_row_ptr,
                      const rocsparse_int*      csr_col_ind,
                      const rocsparse_mat_descr bsr_descr,
                      T*                        bsr_val,
                      rocsparse_int*            bsr_row_ptr,
                      rocsparse_int*            bsr_col_ind,
                      rocsparse_int             row_block_dim,
                      rocsparse_int             col_block_dim,
                      void*                     p_buffer);

// ell2csr
REAL_COMPLEX_TEMPLATE(ell2csr,
                      rocsparse_handle          handle,
                      rocsparse_int             m,
                      rocsparse_int             n,
                      const rocsparse_mat_descr ell_descr,
                      rocsparse_int             ell_width,
                      const T*                  ell_val,
                      const rocsparse_int*      ell_col_ind,
                      const rocsparse_mat_descr csr_descr,
                      T*                        csr_val,
                      const rocsparse_int*      csr_row_ptr,
                      rocsparse_int*            csr_col_ind);

// hyb2csr
REAL_COMPLEX_TEMPLATE(hyb2csr,
                      rocsparse_handle          handle,
                      const rocsparse_mat_descr descr,
                      const rocsparse_hyb_mat   hyb,
                      T*                        csr_val,
                      rocsparse_int*            csr_row_ptr,
                      rocsparse_int*            csr_col_ind,
                      void*                     temp_buffer);

// bsr2csr
REAL_COMPLEX_TEMPLATE(bsr2csr,
                      rocsparse_handle          handle,
                      rocsparse_direction       dir,
                      rocsparse_int             mb,
                      rocsparse_int             nb,
                      const rocsparse_mat_descr bsr_descr,
                      const T*                  bsr_val,
                      const rocsparse_int*      bsr_row_ptr,
                      const rocsparse_int*      bsr_col_ind,
                      rocsparse_int             block_dim,
                      const rocsparse_mat_descr csr_descr,
                      T*                        csr_val,
                      rocsparse_int*            csr_row_ptr,
                      rocsparse_int*            csr_col_ind);

// gebsr2csr
REAL_COMPLEX_TEMPLATE(gebsr2csr,
                      rocsparse_handle          handle,
                      rocsparse_direction       dir,
                      rocsparse_int             mb,
                      rocsparse_int             nb,
                      const rocsparse_mat_descr bsr_descr,
                      const T*                  bsr_val,
                      const rocsparse_int*      bsr_row_ptr,
                      const rocsparse_int*      bsr_col_ind,
                      rocsparse_int             row_block_dim,
                      rocsparse_int             col_block_dim,
                      const rocsparse_mat_descr csr_descr,
                      T*                        csr_val,
                      rocsparse_int*            csr_row_ptr,
                      rocsparse_int*            csr_col_ind);

// gebsr2csr_buffer_size
REAL_COMPLEX_TEMPLATE(gebsr2gebsr_buffer_size,
                      rocsparse_handle          handle,
                      rocsparse_direction       dir,
                      rocsparse_int             mb,
                      rocsparse_int             nb,
                      rocsparse_int             nnzb,
                      const rocsparse_mat_descr descr_A,
                      const T*                  bsr_val_A,
                      const rocsparse_int*      bsr_row_ptr_A,
                      const rocsparse_int*      bsr_col_ind_A,
                      rocsparse_int             row_block_dim_A,
                      rocsparse_int             col_block_dim_A,
                      rocsparse_int             row_block_dim_C,
                      rocsparse_int             col_block_dim_C,
                      size_t*                   buffer_size);

// gebsr2gebsr
REAL_COMPLEX_TEMPLATE(gebsr2gebsr,
                      rocsparse_handle          handle,
                      rocsparse_direction       dir,
                      rocsparse_int             mb,
                      rocsparse_int             nb,
                      rocsparse_int             nnzb,
                      const rocsparse_mat_descr descr_A,
                      const T*                  bsr_val_A,
                      const rocsparse_int*      bsr_row_ptr_A,
                      const rocsparse_int*      bsr_col_ind_A,
                      rocsparse_int             row_block_dim_A,
                      rocsparse_int             col_block_dim_A,
                      const rocsparse_mat_descr descr_C,
                      T*                        bsr_val_C,
                      rocsparse_int*            bsr_row_ptr_C,
                      rocsparse_int*            bsr_col_ind_C,
                      rocsparse_int             row_block_dim_C,
                      rocsparse_int             col_block_dim_C,
                      void*                     temp_buffer);

// csr2csr_compress
REAL_COMPLEX_TEMPLATE(csr2csr_compress,
                      rocsparse_handle          handle,
                      rocsparse_int             m,
                      rocsparse_int             n,
                      const rocsparse_mat_descr descr_A,
                      const T*                  csr_val_A,
                      const rocsparse_int*      csr_row_ptr_A,
                      const rocsparse_int*      csr_col_ind_A,
                      rocsparse_int             nnz_A,
                      const rocsparse_int*      nnz_per_row,
                      T*                        csr_val_C,
                      rocsparse_int*            csr_row_ptr_C,
                      rocsparse_int*            csr_col_ind_C,
                      T                         tol);

// prune_csr2csr_buffer_size
REAL_TEMPLATE(prune_csr2csr_buffer_size,
              rocsparse_handle          handle,
              rocsparse_int             m,
              rocsparse_int             n,
              rocsparse_int             nnz_A,
              const rocsparse_mat_descr csr_descr_A,
              const T*                  csr_val_A,
              const rocsparse_int*      csr_row_ptr_A,
              const rocsparse_int*      csr_col_ind_A,
              const T*                  threshold,
              const rocsparse_mat_descr csr_descr_C,
              const T*                  csr_val_C,
              const rocsparse_int*      csr_row_ptr_C,
              const rocsparse_int*      csr_col_ind_C,
              size_t*                   buffer_size);

// prune_csr2csr_nnz
REAL_TEMPLATE(prune_csr2csr_nnz,
              rocsparse_handle          handle,
              rocsparse_int             m,
              rocsparse_int             n,
              rocsparse_int             nnz_A,
              const rocsparse_mat_descr csr_descr_A,
              const T*                  csr_val_A,
              const rocsparse_int*      csr_row_ptr_A,
              const rocsparse_int*      csr_col_ind_A,
              const T*                  threshold,
              const rocsparse_mat_descr csr_descr_C,
              rocsparse_int*            csr_row_ptr_C,
              rocsparse_int*            nnz_total_dev_host_ptr,
              void*                     buffer);

// prune_csr2csr
REAL_TEMPLATE(prune_csr2csr,
              rocsparse_handle          handle,
              rocsparse_int             m,
              rocsparse_int             n,
              rocsparse_int             nnz_A,
              const rocsparse_mat_descr csr_descr_A,
              const T*                  csr_val_A,
              const rocsparse_int*      csr_row_ptr_A,
              const rocsparse_int*      csr_col_ind_A,
              const T*                  threshold,
              const rocsparse_mat_descr csr_descr_C,
              T*                        csr_val_C,
              const rocsparse_int*      csr_row_ptr_C,
              rocsparse_int*            csr_col_ind_C,
              void*                     buffer);

// prune_csr2csr_by_percentage_buffer_size
REAL_TEMPLATE(prune_csr2csr_by_percentage_buffer_size,
              rocsparse_handle          handle,
              rocsparse_int             m,
              rocsparse_int             n,
              rocsparse_int             nnz_A,
              const rocsparse_mat_descr csr_descr_A,
              const T*                  csr_val_A,
              const rocsparse_int*      csr_row_ptr_A,
              const rocsparse_int*      csr_col_ind_A,
              T                         percentage,
              const rocsparse_mat_descr csr_descr_C,
              const T*                  csr_val_C,
              const rocsparse_int*      csr_row_ptr_C,
              const rocsparse_int*      csr_col_ind_C,
              rocsparse_mat_info        info,
              size_t*                   buffer_size);

// prune_csr2csr_nnz_by_percentage
REAL_TEMPLATE(prune_csr2csr_nnz_by_percentage,
              rocsparse_handle          handle,
              rocsparse_int             m,
              rocsparse_int             n,
              rocsparse_int             nnz_A,
              const rocsparse_mat_descr csr_descr_A,
              const T*                  csr_val_A,
              const rocsparse_int*      csr_row_ptr_A,
              const rocsparse_int*      csr_col_ind_A,
              T                         percentage,
              const rocsparse_mat_descr csr_descr_C,
              rocsparse_int*            csr_row_ptr_C,
              rocsparse_int*            nnz_total_dev_host_ptr,
              rocsparse_mat_info        info,
              void*                     buffer);

// prune_csr2csr_by_percentage
REAL_TEMPLATE(prune_csr2csr_by_percentage,
              rocsparse_handle          handle,
              rocsparse_int             m,
              rocsparse_int             n,
              rocsparse_int             nnz_A,
              const rocsparse_mat_descr csr_descr_A,
              const T*                  csr_val_A,
              const rocsparse_int*      csr_row_ptr_A,
              const rocsparse_int*      csr_col_ind_A,
              T                         percentage,
              const rocsparse_mat_descr csr_descr_C,
              T*                        csr_val_C,
              const rocsparse_int*      csr_row_ptr_C,
              rocsparse_int*            csr_col_ind_C,
              rocsparse_mat_info        info,
              void*                     buffer);

/*
 * ===========================================================================
 *    reordering SPARSE
 * ===========================================================================
 */

// csrcolor
REAL_COMPLEX_TEMPLATE(csrcolor,
                      rocsparse_handle          handle,
                      rocsparse_int             m,
                      rocsparse_int             nnz,
                      const rocsparse_mat_descr descr,
                      const T*                  csr_val,
                      const rocsparse_int*      csr_row_ptr,
                      const rocsparse_int*      csr_col_ind,
                      const floating_data_t<T>* fraction_to_color,
                      rocsparse_int*            ncolors,
                      rocsparse_int*            coloring,
                      rocsparse_int*            reordering,
                      rocsparse_mat_info        info);

#endif // ROCSPARSE_HPP

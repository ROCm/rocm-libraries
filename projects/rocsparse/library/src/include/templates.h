/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse-types.h"

#include "internal/level2/rocsparse_bsrmv.h"
#include "internal/level2/rocsparse_csrsv.h"
#include "internal/level2/rocsparse_gebsrmv.h"
#include "internal/level3/rocsparse_csrmm.h"

//
// For reusing without recompiling.
// e.g. call rocsparse_bsrmv rather than rocsparse_bsrmv_template.
//

//
// csrsv_buffer_size.
//
template <typename T>
inline rocsparse_status rocsparse_csrsv_buffer_size(rocsparse_handle          handle,
                                                    rocsparse_operation       trans,
                                                    rocsparse_int             m,
                                                    rocsparse_int             nnz,
                                                    const rocsparse_mat_descr descr,
                                                    const T*                  csr_val,
                                                    const rocsparse_int*      csr_row_ptr,
                                                    const rocsparse_int*      csr_col_ind,
                                                    rocsparse_mat_info        info,
                                                    size_t*                   buffer_size);

#define SPZL(NAME, TYPE)                                                                         \
    template <>                                                                                  \
    inline rocsparse_status rocsparse_csrsv_buffer_size(rocsparse_handle          handle,        \
                                                        rocsparse_operation       trans,         \
                                                        rocsparse_int             m,             \
                                                        rocsparse_int             nnz,           \
                                                        const rocsparse_mat_descr descr,         \
                                                        const TYPE*               csr_val,       \
                                                        const rocsparse_int*      csr_row_ptr,   \
                                                        const rocsparse_int*      csr_col_ind,   \
                                                        rocsparse_mat_info        info,          \
                                                        size_t*                   buffer_size)   \
    {                                                                                            \
        return NAME(                                                                             \
            handle, trans, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size); \
    }

SPZL(rocsparse_scsrsv_buffer_size, float);
SPZL(rocsparse_dcsrsv_buffer_size, double);
SPZL(rocsparse_ccsrsv_buffer_size, rocsparse_float_complex);
SPZL(rocsparse_zcsrsv_buffer_size, rocsparse_double_complex);
#undef SPZL

// bsrmv
template <typename T>
inline rocsparse_status rocsparse_bsrmv(rocsparse_handle          handle,
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

template <>
inline rocsparse_status rocsparse_bsrmv(rocsparse_handle          handle,
                                        rocsparse_direction       dir,
                                        rocsparse_operation       trans,
                                        rocsparse_int             mb,
                                        rocsparse_int             nb,
                                        rocsparse_int             nnzb,
                                        const float*              alpha,
                                        const rocsparse_mat_descr descr,
                                        const float*              bsr_val,
                                        const rocsparse_int*      bsr_row_ptr,
                                        const rocsparse_int*      bsr_col_ind,
                                        rocsparse_int             bsr_dim,
                                        rocsparse_mat_info        info,
                                        const float*              x,
                                        const float*              beta,
                                        float*                    y)
{
    return rocsparse_sbsrmv(handle,
                            dir,
                            trans,
                            mb,
                            nb,
                            nnzb,
                            alpha,
                            descr,
                            bsr_val,
                            bsr_row_ptr,
                            bsr_col_ind,
                            bsr_dim,
                            info,
                            x,
                            beta,
                            y);
}

template <>
inline rocsparse_status rocsparse_bsrmv(rocsparse_handle          handle,
                                        rocsparse_direction       dir,
                                        rocsparse_operation       trans,
                                        rocsparse_int             mb,
                                        rocsparse_int             nb,
                                        rocsparse_int             nnzb,
                                        const double*             alpha,
                                        const rocsparse_mat_descr descr,
                                        const double*             bsr_val,
                                        const rocsparse_int*      bsr_row_ptr,
                                        const rocsparse_int*      bsr_col_ind,
                                        rocsparse_int             bsr_dim,
                                        rocsparse_mat_info        info,
                                        const double*             x,
                                        const double*             beta,
                                        double*                   y)
{
    return rocsparse_dbsrmv(handle,
                            dir,
                            trans,
                            mb,
                            nb,
                            nnzb,
                            alpha,
                            descr,
                            bsr_val,
                            bsr_row_ptr,
                            bsr_col_ind,
                            bsr_dim,
                            info,
                            x,
                            beta,
                            y);
}

template <>
inline rocsparse_status rocsparse_bsrmv(rocsparse_handle               handle,
                                        rocsparse_direction            dir,
                                        rocsparse_operation            trans,
                                        rocsparse_int                  mb,
                                        rocsparse_int                  nb,
                                        rocsparse_int                  nnzb,
                                        const rocsparse_float_complex* alpha,
                                        const rocsparse_mat_descr      descr,
                                        const rocsparse_float_complex* bsr_val,
                                        const rocsparse_int*           bsr_row_ptr,
                                        const rocsparse_int*           bsr_col_ind,
                                        rocsparse_int                  bsr_dim,
                                        rocsparse_mat_info             info,
                                        const rocsparse_float_complex* x,
                                        const rocsparse_float_complex* beta,
                                        rocsparse_float_complex*       y)
{
    return rocsparse_cbsrmv(handle,
                            dir,
                            trans,
                            mb,
                            nb,
                            nnzb,
                            alpha,
                            descr,
                            bsr_val,
                            bsr_row_ptr,
                            bsr_col_ind,
                            bsr_dim,
                            info,
                            x,
                            beta,
                            y);
}

template <>
inline rocsparse_status rocsparse_bsrmv(rocsparse_handle                handle,
                                        rocsparse_direction             dir,
                                        rocsparse_operation             trans,
                                        rocsparse_int                   mb,
                                        rocsparse_int                   nb,
                                        rocsparse_int                   nnzb,
                                        const rocsparse_double_complex* alpha,
                                        const rocsparse_mat_descr       descr,
                                        const rocsparse_double_complex* bsr_val,
                                        const rocsparse_int*            bsr_row_ptr,
                                        const rocsparse_int*            bsr_col_ind,
                                        rocsparse_int                   bsr_dim,
                                        rocsparse_mat_info              info,
                                        const rocsparse_double_complex* x,
                                        const rocsparse_double_complex* beta,
                                        rocsparse_double_complex*       y)
{
    return rocsparse_zbsrmv(handle,
                            dir,
                            trans,
                            mb,
                            nb,
                            nnzb,
                            alpha,
                            descr,
                            bsr_val,
                            bsr_row_ptr,
                            bsr_col_ind,
                            bsr_dim,
                            info,
                            x,
                            beta,
                            y);
}

// gebsrmv
template <typename T>
inline rocsparse_status rocsparse_gebsrmv(rocsparse_handle          handle,
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

template <>
inline rocsparse_status rocsparse_gebsrmv(rocsparse_handle          handle,
                                          rocsparse_direction       dir,
                                          rocsparse_operation       trans,
                                          rocsparse_int             mb,
                                          rocsparse_int             nb,
                                          rocsparse_int             nnzb,
                                          const float*              alpha,
                                          const rocsparse_mat_descr descr,
                                          const float*              bsr_val,
                                          const rocsparse_int*      bsr_row_ptr,
                                          const rocsparse_int*      bsr_col_ind,
                                          rocsparse_int             row_block_dim,
                                          rocsparse_int             col_block_dim,
                                          const float*              x,
                                          const float*              beta,
                                          float*                    y)
{
    return rocsparse_sgebsrmv(handle,
                              dir,
                              trans,
                              mb,
                              nb,
                              nnzb,
                              alpha,
                              descr,
                              bsr_val,
                              bsr_row_ptr,
                              bsr_col_ind,
                              row_block_dim,
                              col_block_dim,
                              x,
                              beta,
                              y);
}

template <>
inline rocsparse_status rocsparse_gebsrmv(rocsparse_handle          handle,
                                          rocsparse_direction       dir,
                                          rocsparse_operation       trans,
                                          rocsparse_int             mb,
                                          rocsparse_int             nb,
                                          rocsparse_int             nnzb,
                                          const double*             alpha,
                                          const rocsparse_mat_descr descr,
                                          const double*             bsr_val,
                                          const rocsparse_int*      bsr_row_ptr,
                                          const rocsparse_int*      bsr_col_ind,
                                          rocsparse_int             row_block_dim,
                                          rocsparse_int             col_block_dim,
                                          const double*             x,
                                          const double*             beta,
                                          double*                   y)
{
    return rocsparse_dgebsrmv(handle,
                              dir,
                              trans,
                              mb,
                              nb,
                              nnzb,
                              alpha,
                              descr,
                              bsr_val,
                              bsr_row_ptr,
                              bsr_col_ind,
                              row_block_dim,
                              col_block_dim,
                              x,
                              beta,
                              y);
}

template <>
inline rocsparse_status rocsparse_gebsrmv(rocsparse_handle               handle,
                                          rocsparse_direction            dir,
                                          rocsparse_operation            trans,
                                          rocsparse_int                  mb,
                                          rocsparse_int                  nb,
                                          rocsparse_int                  nnzb,
                                          const rocsparse_float_complex* alpha,
                                          const rocsparse_mat_descr      descr,
                                          const rocsparse_float_complex* bsr_val,
                                          const rocsparse_int*           bsr_row_ptr,
                                          const rocsparse_int*           bsr_col_ind,
                                          rocsparse_int                  row_block_dim,
                                          rocsparse_int                  col_block_dim,
                                          const rocsparse_float_complex* x,
                                          const rocsparse_float_complex* beta,
                                          rocsparse_float_complex*       y)
{
    return rocsparse_cgebsrmv(handle,
                              dir,
                              trans,
                              mb,
                              nb,
                              nnzb,
                              alpha,
                              descr,
                              bsr_val,
                              bsr_row_ptr,
                              bsr_col_ind,
                              row_block_dim,
                              col_block_dim,
                              x,
                              beta,
                              y);
}

template <>
inline rocsparse_status rocsparse_gebsrmv(rocsparse_handle                handle,
                                          rocsparse_direction             dir,
                                          rocsparse_operation             trans,
                                          rocsparse_int                   mb,
                                          rocsparse_int                   nb,
                                          rocsparse_int                   nnzb,
                                          const rocsparse_double_complex* alpha,
                                          const rocsparse_mat_descr       descr,
                                          const rocsparse_double_complex* bsr_val,
                                          const rocsparse_int*            bsr_row_ptr,
                                          const rocsparse_int*            bsr_col_ind,
                                          rocsparse_int                   row_block_dim,
                                          rocsparse_int                   col_block_dim,
                                          const rocsparse_double_complex* x,
                                          const rocsparse_double_complex* beta,
                                          rocsparse_double_complex*       y)
{
    return rocsparse_zgebsrmv(handle,
                              dir,
                              trans,
                              mb,
                              nb,
                              nnzb,
                              alpha,
                              descr,
                              bsr_val,
                              bsr_row_ptr,
                              bsr_col_ind,
                              row_block_dim,
                              col_block_dim,
                              x,
                              beta,
                              y);
}

// csrmm
template <typename T>
inline rocsparse_status rocsparse_csrmm(rocsparse_handle          handle,
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
                                        int64_t                   ldb,
                                        const T*                  beta,
                                        T*                        C,
                                        int64_t                   ldc);

template <>
inline rocsparse_status rocsparse_csrmm(rocsparse_handle          handle,
                                        rocsparse_operation       trans_A,
                                        rocsparse_operation       trans_B,
                                        rocsparse_int             m,
                                        rocsparse_int             n,
                                        rocsparse_int             k,
                                        rocsparse_int             nnz,
                                        const float*              alpha,
                                        const rocsparse_mat_descr descr,
                                        const float*              csr_val,
                                        const rocsparse_int*      csr_row_ptr,
                                        const rocsparse_int*      csr_col_ind,
                                        const float*              B,
                                        int64_t                   ldb,
                                        const float*              beta,
                                        float*                    C,
                                        int64_t                   ldc)
{
    return rocsparse_scsrmm(handle,
                            trans_A,
                            trans_B,
                            m,
                            n,
                            k,
                            nnz,
                            alpha,
                            descr,
                            csr_val,
                            csr_row_ptr,
                            csr_col_ind,
                            B,
                            ldb,
                            beta,
                            C,
                            ldc);
}

template <>
inline rocsparse_status rocsparse_csrmm(rocsparse_handle          handle,
                                        rocsparse_operation       trans_A,
                                        rocsparse_operation       trans_B,
                                        rocsparse_int             m,
                                        rocsparse_int             n,
                                        rocsparse_int             k,
                                        rocsparse_int             nnz,
                                        const double*             alpha,
                                        const rocsparse_mat_descr descr,
                                        const double*             csr_val,
                                        const rocsparse_int*      csr_row_ptr,
                                        const rocsparse_int*      csr_col_ind,
                                        const double*             B,
                                        int64_t                   ldb,
                                        const double*             beta,
                                        double*                   C,
                                        int64_t                   ldc)
{
    return rocsparse_dcsrmm(handle,
                            trans_A,
                            trans_B,
                            m,
                            n,
                            k,
                            nnz,
                            alpha,
                            descr,
                            csr_val,
                            csr_row_ptr,
                            csr_col_ind,
                            B,
                            ldb,
                            beta,
                            C,
                            ldc);
}

template <>
inline rocsparse_status rocsparse_csrmm(rocsparse_handle               handle,
                                        rocsparse_operation            trans_A,
                                        rocsparse_operation            trans_B,
                                        rocsparse_int                  m,
                                        rocsparse_int                  n,
                                        rocsparse_int                  k,
                                        rocsparse_int                  nnz,
                                        const rocsparse_float_complex* alpha,
                                        const rocsparse_mat_descr      descr,
                                        const rocsparse_float_complex* csr_val,
                                        const rocsparse_int*           csr_row_ptr,
                                        const rocsparse_int*           csr_col_ind,
                                        const rocsparse_float_complex* B,
                                        int64_t                        ldb,
                                        const rocsparse_float_complex* beta,
                                        rocsparse_float_complex*       C,
                                        int64_t                        ldc)
{
    return rocsparse_ccsrmm(handle,
                            trans_A,
                            trans_B,
                            m,
                            n,
                            k,
                            nnz,
                            alpha,
                            descr,
                            csr_val,
                            csr_row_ptr,
                            csr_col_ind,
                            B,
                            ldb,
                            beta,
                            C,
                            ldc);
}

template <>
inline rocsparse_status rocsparse_csrmm(rocsparse_handle                handle,
                                        rocsparse_operation             trans_A,
                                        rocsparse_operation             trans_B,
                                        rocsparse_int                   m,
                                        rocsparse_int                   n,
                                        rocsparse_int                   k,
                                        rocsparse_int                   nnz,
                                        const rocsparse_double_complex* alpha,
                                        const rocsparse_mat_descr       descr,
                                        const rocsparse_double_complex* csr_val,
                                        const rocsparse_int*            csr_row_ptr,
                                        const rocsparse_int*            csr_col_ind,
                                        const rocsparse_double_complex* B,
                                        int64_t                         ldb,
                                        const rocsparse_double_complex* beta,
                                        rocsparse_double_complex*       C,
                                        int64_t                         ldc)
{
    return rocsparse_zcsrmm(handle,
                            trans_A,
                            trans_B,
                            m,
                            n,
                            k,
                            nnz,
                            alpha,
                            descr,
                            csr_val,
                            csr_row_ptr,
                            csr_col_ind,
                            B,
                            ldb,
                            beta,
                            C,
                            ldc);
}

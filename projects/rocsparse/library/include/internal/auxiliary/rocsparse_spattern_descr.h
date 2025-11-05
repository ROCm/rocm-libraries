/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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
 *  \brief rocsparse_spattern_descr.h provides auxilary functions in rocsparse
 */

#ifndef ROCSPARSE_SPATTERN_DESCR_H
#define ROCSPARSE_SPATTERN_DESCR_H

#include "rocsparse-types.h"
#include "rocsparse/rocsparse-export.h"

#ifdef __cplusplus
extern "C" {
#endif

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spattern_create_csr(rocsparse_handle          handle,
                                               rocsparse_spattern_descr* p_descr,
                                               int64_t                   rows,
                                               int64_t                   cols,
                                               int64_t                   nnz,
                                               rocsparse_idvec_descr     row_data,
                                               rocsparse_idvec_descr     col_data,
                                               rocsparse_error*          p_error);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spattern_create_bsr(rocsparse_handle          handle,
                                               rocsparse_spattern_descr* p_descr,
                                               int64_t                   rowsb,
                                               int64_t                   colsb,
                                               int64_t                   nnzb,
                                               rocsparse_idvec_descr     row_data,
                                               rocsparse_idvec_descr     col_data,
                                               rocsparse_error*          p_error);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spattern_create_csc(rocsparse_handle          handle,
                                               rocsparse_spattern_descr* p_descr,
                                               int64_t                   rows,
                                               int64_t                   cols,
                                               int64_t                   nnz,
                                               rocsparse_idvec_descr     row_data,
                                               rocsparse_idvec_descr     col_data,
                                               rocsparse_error*          p_error);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spattern_create_coo(rocsparse_handle          handle,
                                               rocsparse_spattern_descr* p_descr,
                                               int64_t                   rows,
                                               int64_t                   cols,
                                               int64_t                   nnz,
                                               rocsparse_idvec_descr     row_data,
                                               rocsparse_idvec_descr     col_data,
                                               rocsparse_error*          p_error);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spattern_create_coo_aos(rocsparse_handle          handle,
                                                   rocsparse_spattern_descr* p_descr,
                                                   int64_t                   rows,
                                                   int64_t                   cols,
                                                   int64_t                   nnz,
                                                   rocsparse_idvec_descr     row_data,
                                                   rocsparse_idvec_descr     col_data,
                                                   rocsparse_error*          p_error);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spattern_create_ell(rocsparse_handle          handle,
                                               rocsparse_spattern_descr* p_descr,
                                               int64_t                   rows,
                                               int64_t                   cols,
                                               int64_t                   width,
                                               rocsparse_idvec_descr     col_data,
                                               rocsparse_error*          p_error);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spattern_create_bell(rocsparse_handle          handle,
                                                rocsparse_spattern_descr* p_descr,
                                                int64_t                   rowsb,
                                                int64_t                   colsb,
                                                int64_t                   width,
                                                rocsparse_idvec_descr     col_data,
                                                rocsparse_error*          p_error);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spattern_get_prop(rocsparse_handle               handle,
                                             rocsparse_const_spattern_descr descr,
                                             rocsparse_spattern_prop        prop,
                                             void*                          value,
                                             size_t                         value_size_in_bytes,
                                             rocsparse_error*               p_error);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spattern_set_prop(rocsparse_handle         handle,
                                             rocsparse_spattern_descr descr,
                                             rocsparse_spattern_prop  prop,
                                             const void*              value,
                                             size_t                   value_size_in_bytes,
                                             rocsparse_error*         p_error);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spattern_get_row_data(rocsparse_handle               handle,
                                                 rocsparse_const_spattern_descr descr,
                                                 rocsparse_idvec_descr*         p_data,
                                                 rocsparse_error*               p_error);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spattern_get_col_data(rocsparse_handle               handle,
                                                 rocsparse_const_spattern_descr descr,
                                                 rocsparse_idvec_descr*         p_data,
                                                 rocsparse_error*               p_error);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spattern_set_row_data(rocsparse_handle         handle,
                                                 rocsparse_spattern_descr descr,
                                                 rocsparse_idvec_descr    data,
                                                 rocsparse_error*         p_error);

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spattern_set_col_data(rocsparse_handle         handle,
                                                 rocsparse_spattern_descr descr,
                                                 rocsparse_idvec_descr    data,
                                                 rocsparse_error*         p_error);

#ifdef __cplusplus
}
#endif

#endif

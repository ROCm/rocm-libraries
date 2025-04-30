/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
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

#include "handle.h"

namespace rocsparse
{
    template <typename T, typename I, typename J>
    rocsparse_status csxsldu_buffer_size_template(rocsparse_handle     handle_,
                                                  rocsparse_direction  dir_,
                                                  J                    m_,
                                                  J                    n_,
                                                  I                    nnz_,
                                                  const I*             ptr_,
                                                  const J*             ind_,
                                                  const T*             val_,
                                                  rocsparse_index_base base_,
                                                  rocsparse_diag_type  ldiag_,
                                                  rocsparse_direction  ldir_,
                                                  rocsparse_diag_type  udiag_,
                                                  rocsparse_direction  udir_,
                                                  size_t*              buffer_size_);

    template <typename T, typename I, typename J>
    rocsparse_status csxsldu_preprocess_template(rocsparse_handle     handle_,
                                                 rocsparse_direction  dir_,
                                                 J                    m_,
                                                 J                    n_,
                                                 I                    nnz_,
                                                 const I*             ptr_,
                                                 const J*             ind_,
                                                 const T*             val_,
                                                 rocsparse_index_base base_,
                                                 rocsparse_diag_type  ldiag_,
                                                 rocsparse_direction  ldir_,
                                                 I*                   lnnz_,
                                                 I*                   lptr_,
                                                 rocsparse_index_base lbase_,
                                                 rocsparse_diag_type  udiag_,
                                                 rocsparse_direction  udir_,
                                                 I*                   unnz_,
                                                 I*                   uptr_,
                                                 rocsparse_index_base ubase_,
                                                 void*                buffer_);

    template <typename T, typename I, typename J>
    rocsparse_status csxsldu_compute_template(rocsparse_handle handle_,
                                              //
                                              rocsparse_direction  dir_,
                                              J                    m_,
                                              J                    n_,
                                              I                    nnz_,
                                              const I*             ptr_,
                                              const J*             ind_,
                                              T*                   val_,
                                              rocsparse_index_base base_,
                                              //
                                              rocsparse_diag_type  ldiag_,
                                              rocsparse_direction  ldir_,
                                              I                    lnnz_,
                                              I*                   lptr_,
                                              J*                   lind_,
                                              T*                   lval_,
                                              rocsparse_index_base lbase_,
                                              //
                                              rocsparse_diag_type  udiag_,
                                              rocsparse_direction  udir_,
                                              I                    unnz_,
                                              I*                   uptr_,
                                              J*                   uind_,
                                              T*                   uval_,
                                              rocsparse_index_base ubase_,
                                              //
                                              T* diag_,
                                              //
                                              void* buffer_);
}

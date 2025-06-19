/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
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

#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <typename T, typename I, typename J>
    rocsparse_status csritilu0x_compute_template(rocsparse_handle     handle_,
                                                 rocsparse_itilu0_alg alg_,
                                                 rocsparse_int        options_,
                                                 J* __restrict__ nsweeps_,
                                                 J                  nfreeiter_,
                                                 floating_data_t<T> tol_,
                                                 J                  m_,
                                                 I                  nnz_,
                                                 const I* __restrict__ ptr_begin_,
                                                 const I* __restrict__ ptr_end_,
                                                 const J* __restrict__ ind_,
                                                 const T* __restrict__ val_,
                                                 rocsparse_index_base base_,
                                                 rocsparse_diag_type  ldiag_type_,
                                                 rocsparse_direction  ldir_,
                                                 I                    lnnz_,
                                                 const I* __restrict__ lptr_begin_,
                                                 const I* __restrict__ lptr_end_,
                                                 const J* __restrict__ lind_,
                                                 T* __restrict__ lval_,
                                                 rocsparse_index_base lbase_,
                                                 rocsparse_diag_type  udiag_type_,
                                                 rocsparse_direction  udir_,
                                                 I                    unnz_,
                                                 const I* __restrict__ uptr_begin_,
                                                 const I* __restrict__ uptr_end_,
                                                 const J* __restrict__ uind_,
                                                 T* __restrict__ uval_,
                                                 rocsparse_index_base ubase_,
                                                 T* __restrict__ dval_,
                                                 size_t buffer_size_,
                                                 void* __restrict__ buffer_);

    template <typename T, typename I, typename J>
    rocsparse_status csritilu0x_compute_impl(rocsparse_handle     handle_,
                                             rocsparse_itilu0_alg alg_,
                                             rocsparse_int        options_,
                                             J* __restrict__ nsweeps_,
                                             J                  nfreeiter_,
                                             floating_data_t<T> tol_,
                                             J                  m_,
                                             I                  nnz_,
                                             const I* __restrict__ ptr_begin_,
                                             const I* __restrict__ ptr_end_,
                                             const J* __restrict__ ind_,
                                             const T* __restrict__ val_,
                                             rocsparse_index_base base_,
                                             rocsparse_diag_type  ldiag_type_,
                                             rocsparse_direction  ldir_,
                                             I                    lnnz_,
                                             const I* __restrict__ lptr_begin_,
                                             const I* __restrict__ lptr_end_,
                                             const J* __restrict__ lind_,
                                             T* __restrict__ lval_,
                                             rocsparse_index_base lbase_,
                                             rocsparse_diag_type  udiag_type_,
                                             rocsparse_direction  udir_,
                                             I                    unnz_,
                                             const I* __restrict__ uptr_begin_,
                                             const I* __restrict__ uptr_end_,
                                             const J* __restrict__ uind_,
                                             T* __restrict__ uval_,
                                             rocsparse_index_base ubase_,
                                             T* __restrict__ dval_,
                                             size_t buffer_size_,
                                             void* __restrict__ buffer_);
}

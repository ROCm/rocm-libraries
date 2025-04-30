/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights Reserved.
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
    template <typename T>
    rocsparse_status prune_csr2csr_buffer_size_template(rocsparse_handle          handle,
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

    template <typename T>
    rocsparse_status prune_csr2csr_nnz_template(rocsparse_handle          handle,
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
                                                void*                     temp_buffer);
}

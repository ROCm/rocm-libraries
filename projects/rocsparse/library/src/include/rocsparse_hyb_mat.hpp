/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

/********************************************************************************
 * \brief rocsparse_hyb_mat is a structure holding the rocsparse HYB matrix.
 * It must be initialized using rocsparse_create_hyb_mat() and the returned
 * handle must be passed to all subsequent library function calls that involve
 * the HYB matrix.
 * It should be destroyed at the end using rocsparse_destroy_hyb_mat().
 *******************************************************************************/
struct _rocsparse_hyb_mat
{
    // num rows
    rocsparse_int m{};
    // num cols
    rocsparse_int n{};

    // partition type
    rocsparse_hyb_partition partition = rocsparse_hyb_partition_auto;

    // ELL matrix part
    rocsparse_int  ell_nnz{};
    rocsparse_int  ell_width{};
    rocsparse_int* ell_col_ind{};
    void*          ell_val{};

    // COO matrix part
    rocsparse_int  coo_nnz{};
    rocsparse_int* coo_row_ind{};
    rocsparse_int* coo_col_ind{};
    void*          coo_val{};

    rocsparse_datatype data_type_T = rocsparse_datatype_f32_r;
};

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

#include "rocsparse_cscsv.hpp"
#include "rocsparse_cscsv_common.hpp"

size_t rocsparse::cscsv_column_analysis_buffer_size()
{
    // Longest column only
    return 256;
}

size_t rocsparse::cscsv_column_solve_buffer_size(rocsparse_const_spmat_descr A, int64_t batch_count)
{
    const size_t m        = A->cols;
    const size_t sizeof_J = rocsparse::indextype_sizeof(A->row_type);
    const size_t sizeof_T = rocsparse::datatype_sizeof(A->data_type);
    // ticket, remaining dependencies and partial sums
    return rocsparse::cscsv_aligned(sizeof(int32_t) * batch_count)
           + rocsparse::cscsv_aligned(sizeof_J * m * batch_count)
           + rocsparse::cscsv_aligned(sizeof_T * m * batch_count);
}

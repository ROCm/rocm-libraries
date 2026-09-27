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

#pragma once

#include "rocsparse_handle.hpp"

namespace rocsparse
{
    typedef enum rocsparse_csrsort_alg_
    {
        rocsparse_csrsort_alg_default = 0
    } rocsparse_csrsort_alg;

    // Shared by the CSR and CSC sorts: dir == rocsparse_direction_row sorts the column indices
    // within each row of a CSR matrix, and dir == rocsparse_direction_column sorts the row
    // indices within each column of a CSC matrix.
    rocsparse_status csxsort_buffer_size(rocsparse_handle            handle,
                                         rocsparse_direction         dir,
                                         rocsparse_const_spmat_descr source,
                                         rocsparse_const_spmat_descr target,
                                         size_t*                     buffer_size_in_bytes);

    rocsparse_status csxsort(rocsparse_handle            handle,
                             rocsparse_direction         dir,
                             rocsparse_const_spmat_descr source,
                             rocsparse_spmat_descr       target,
                             size_t                      buffer_size_in_bytes,
                             void*                       buffer);

    rocsparse_status csrsort_buffer_size(rocsparse_handle            handle,
                                         rocsparse_csrsort_alg       alg,
                                         rocsparse_const_spmat_descr source,
                                         rocsparse_const_spmat_descr target,
                                         size_t*                     buffer_size_in_bytes);

    // Sorts the column indices and values within each row of the CSR matrix source into the
    // CSR matrix target, which must have the same sizes, types, index base and batch layout as
    // source, and whose row pointer receives a copy of the row pointer of source. Each array of
    // target may either alias the matching array of source, in which case it is sorted in
    // place, or not overlap it at all. Each batch is sorted independently.
    rocsparse_status csrsort(rocsparse_handle            handle,
                             rocsparse_csrsort_alg       alg,
                             rocsparse_const_spmat_descr source,
                             rocsparse_spmat_descr       target,
                             size_t                      buffer_size_in_bytes,
                             void*                       buffer);
}

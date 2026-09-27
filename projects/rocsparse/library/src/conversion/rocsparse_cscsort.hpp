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
    typedef enum rocsparse_cscsort_alg_
    {
        rocsparse_cscsort_alg_default = 0
    } rocsparse_cscsort_alg;

    rocsparse_status cscsort_buffer_size(rocsparse_handle            handle,
                                         rocsparse_cscsort_alg       alg,
                                         rocsparse_const_spmat_descr source,
                                         rocsparse_const_spmat_descr target,
                                         size_t*                     buffer_size_in_bytes);

    // Sorts the row indices and values within each column of the CSC matrix source into the
    // CSC matrix target, which must have the same sizes, types, index base and batch layout as
    // source, and whose column pointer receives a copy of the column pointer of source. Each
    // array of target may either alias the matching array of source, in which case it is
    // sorted in place, or not overlap it at all. Each batch is sorted independently.
    rocsparse_status cscsort(rocsparse_handle            handle,
                             rocsparse_cscsort_alg       alg,
                             rocsparse_const_spmat_descr source,
                             rocsparse_spmat_descr       target,
                             size_t                      buffer_size_in_bytes,
                             void*                       buffer);
}

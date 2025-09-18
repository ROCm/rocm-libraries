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
#include "rocsparse_handle.hpp"

namespace rocsparse
{
    template <typename I, typename T>
    rocsparse_status gthr_template(rocsparse_handle     handle,
                                   int64_t              nnz,
                                   const void*          y,
                                   void*                x_val,
                                   const void*          x_ind,
                                   rocsparse_index_base idx_base);

    rocsparse_status gthr(rocsparse_handle     handle,
                          int64_t              nnz,
                          rocsparse_datatype   y_datatype,
                          const void*          y,
                          rocsparse_datatype   x_datatype,
                          void*                x_val,
                          rocsparse_indextype  x_indextype,
                          const void*          x_ind,
                          rocsparse_index_base idx_base);
}

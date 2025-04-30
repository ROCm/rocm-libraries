/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

namespace rocsparse
{
    rocsparse_status gcoosort_buffer_size(rocsparse_handle    handle_,
                                          int64_t             m,
                                          int64_t             n,
                                          int64_t             nnz,
                                          rocsparse_indextype idx_type,
                                          const void*         row_data,
                                          const void*         col_data,
                                          size_t*             buffer_size);

    rocsparse_status gcoosort_by_row(rocsparse_handle    handle_,
                                     int64_t             m,
                                     int64_t             n,
                                     int64_t             nnz,
                                     rocsparse_indextype idx_type,
                                     void*               row_data,
                                     void*               col_data,
                                     void*               perm,
                                     void*               buffer);

    rocsparse_status gcoosort_by_column(rocsparse_handle    handle_,
                                        int64_t             m,
                                        int64_t             n,
                                        int64_t             nnz,
                                        rocsparse_indextype idx_type,
                                        void*               row_data,
                                        void*               col_data,
                                        void*               perm,
                                        void*               buffer);
}

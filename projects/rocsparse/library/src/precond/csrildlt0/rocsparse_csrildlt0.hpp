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

#include "rocsparse_utility.hpp"

namespace rocsparse
{
    rocsparse_status csrildlt0_analysis_buffer_size(rocsparse_handle            handle,
                                                    rocsparse_const_spmat_descr A,
                                                    size_t*                     buffer_size);

    rocsparse_status csrildlt0_solve_buffer_size(rocsparse_handle            handle,
                                                 rocsparse_const_spmat_descr A,
                                                 size_t*                     buffer_size);

    rocsparse_status csrildlt0_analysis(rocsparse_handle          handle,
                                        rocsparse_spmat_descr     A,
                                        rocsparse_analysis_policy analysis,
                                        rocsparse_csrildlt0_info* p_csrildlt0_info,
                                        void*                     temp_buffer);

    rocsparse_status csrildlt0(rocsparse_handle         handle,
                               rocsparse_csrildlt0_info csrildlt0_info,
                               rocsparse_spmat_descr    A,
                               void*                    diag,
                               size_t                   buffer_size,
                               void*                    buffer);
}

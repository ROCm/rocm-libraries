/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "../level2/rocsparse_csrsv.hpp"
#include "rocsparse_bsrilu0.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::bsrilu0_analysis_buffer_size(rocsparse_handle            handle,
                                                         rocsparse_const_spmat_descr A,
                                                         size_t* p_buffer_size_in_bytes)
{
    ROCSPARSE_ROUTINE_TRACE;
    p_buffer_size_in_bytes[0] = std::numeric_limits<size_t>::max();
    RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrsv_analysis_buffer_size(
        handle, rocsparse_operation_none, A, p_buffer_size_in_bytes)));
    return rocsparse_status_success;
}

rocsparse_status rocsparse::bsrilu0_solve_buffer_size(rocsparse_handle            handle,
                                                      rocsparse_const_spmat_descr A,
                                                      size_t* p_buffer_size_in_bytes)
{
    ROCSPARSE_ROUTINE_TRACE;
    p_buffer_size_in_bytes[0] = std::numeric_limits<size_t>::max();
    RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrsv_solve_buffer_size(
        handle, rocsparse_operation_none, A, p_buffer_size_in_bytes)));
    return rocsparse_status_success;
}

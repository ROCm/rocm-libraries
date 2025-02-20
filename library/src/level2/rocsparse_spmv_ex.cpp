/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "control.h"
#include "handle.h"
#include "internal/generic/rocsparse_spmv.h"
#include "utility.h"

extern "C" rocsparse_status rocsparse_spmv_ex(rocsparse_handle            handle,
                                              rocsparse_operation         trans,
                                              const void*                 alpha,
                                              const rocsparse_spmat_descr mat,
                                              const rocsparse_dnvec_descr x,
                                              const void*                 beta,
                                              const rocsparse_dnvec_descr y,
                                              rocsparse_datatype          compute_type,
                                              rocsparse_spmv_alg          alg,
                                              rocsparse_spmv_stage        stage,
                                              size_t*                     buffer_size,
                                              void*                       temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check for invalid handle
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_spmv_ex",
                         trans,
                         (const void*&)alpha,
                         (const void*&)mat,
                         (const void*&)x,
                         (const void*&)beta,
                         (const void*&)y,
                         compute_type,
                         alg,
                         (const void*&)buffer_size,
                         (const void*&)temp_buffer);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_spmv(
        handle, trans, alpha, mat, x, beta, y, compute_type, alg, stage, buffer_size, temp_buffer));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

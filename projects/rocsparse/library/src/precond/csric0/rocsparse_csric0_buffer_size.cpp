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
#include "rocsparse_csric0.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    rocsparse_status csric0_analysis_buffer_size(rocsparse_handle            handle,
                                                 rocsparse_const_spmat_descr A,
                                                 size_t*                     buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);

        ROCSPARSE_CHECKARG_POINTER(1, A);

        ROCSPARSE_CHECKARG(1, A, A->descr == nullptr, rocsparse_status_invalid_pointer);

        ROCSPARSE_CHECKARG(1,
                           A,
                           (A->descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);

        ROCSPARSE_CHECKARG(1,
                           A,
                           (A->descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);

        RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrsv_analysis_buffer_size(
            handle, rocsparse_operation_none, A, buffer_size)));
        return rocsparse_status_success;
    }

    rocsparse_status csric0_solve_buffer_size(rocsparse_handle            handle,
                                              rocsparse_const_spmat_descr A,
                                              size_t*                     buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);

        ROCSPARSE_CHECKARG_POINTER(1, A);

        ROCSPARSE_CHECKARG(1, A, A->descr == nullptr, rocsparse_status_invalid_pointer);

        ROCSPARSE_CHECKARG(1,
                           A,
                           (A->descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);

        ROCSPARSE_CHECKARG(1,
                           A,
                           (A->descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);

        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::csrsv_solve_buffer_size(handle, rocsparse_operation_none, A, buffer_size)));

        return rocsparse_status_success;
    }

}

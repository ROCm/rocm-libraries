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

#include "rocsparse_utility.hpp"

#include "../precond/bsrilu0/rocsparse_bsrilu0.hpp"
#include "../precond/csrilu0/rocsparse_csrilu0.hpp"
#include "rocsparse_spilu0_descr.hpp"

#include "internal/generic/rocsparse_spilu0.h"

namespace rocsparse
{
    static rocsparse_status spilu0_buffer_size(rocsparse_handle            handle,
                                               rocsparse_spilu0_descr      spilu0_descr,
                                               rocsparse_const_spmat_descr A,
                                               rocsparse_spilu0_stage      spilu0_stage,
                                               size_t*                     buffer_size_in_bytes)
    {
        ROCSPARSE_ROUTINE_TRACE;
        const rocsparse_format format = A->format;
        switch(format)
        {
        case rocsparse_format_csr:
        {
            switch(spilu0_stage)
            {
            case rocsparse_spilu0_stage_analysis:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::csrilu0_analysis_buffer_size(handle, A, buffer_size_in_bytes));
                return rocsparse_status_success;
            }
            case rocsparse_spilu0_stage_compute:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::csrilu0_solve_buffer_size(handle, A, buffer_size_in_bytes));
                return rocsparse_status_success;
            }
            }

            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            // LCOV_EXCL_STOP
        }

        case rocsparse_format_bsr:
        {
            switch(spilu0_stage)
            {
            case rocsparse_spilu0_stage_analysis:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::bsrilu0_analysis_buffer_size(handle, A, buffer_size_in_bytes));
                return rocsparse_status_success;
            }
            case rocsparse_spilu0_stage_compute:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::bsrilu0_solve_buffer_size(handle, A, buffer_size_in_bytes));
                return rocsparse_status_success;
            }
            }
            return rocsparse_status_success;
        }

        case rocsparse_format_coo:
        case rocsparse_format_csc:
        case rocsparse_format_ell:
        case rocsparse_format_sell:
        case rocsparse_format_bell:
        case rocsparse_format_coo_aos:
        {
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            // LCOV_EXCL_STOP
        }
        }

        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
extern "C" rocsparse_status rocsparse_spilu0_buffer_size(rocsparse_handle            handle,
                                                         rocsparse_spilu0_descr      spilu0_descr,
                                                         rocsparse_const_spmat_descr A,
                                                         rocsparse_const_spmat_descr P,
                                                         rocsparse_spilu0_stage      spilu0_stage,
                                                         size_t*          p_buffer_size_in_bytes,
                                                         rocsparse_error* p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, spilu0_descr);
    //
    ROCSPARSE_CHECKARG_POINTER(2, A);
    //
    ROCSPARSE_CHECKARG_POINTER(3, P);
    //
    ROCSPARSE_CHECKARG(3, P, (A != P), rocsparse_status_not_implemented);
    //
    ROCSPARSE_CHECKARG_ENUM(4, spilu0_stage);

    ROCSPARSE_CHECKARG_POINTER(5, p_buffer_size_in_bytes);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::spilu0_buffer_size(
        handle, spilu0_descr, A, spilu0_stage, p_buffer_size_in_bytes));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

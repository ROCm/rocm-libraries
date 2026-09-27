/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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

#include "rocsparse_control.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_coosort.hpp"
#include "rocsparse_cscsort.hpp"
#include "rocsparse_csrsort.hpp"
#include "rocsparse_spsort.hpp"

namespace rocsparse
{
    static rocsparse_status spsort_buffer_size(rocsparse_handle            handle,
                                               rocsparse_spsort_descr      descr,
                                               rocsparse_const_spmat_descr source,
                                               rocsparse_const_spmat_descr target,
                                               rocsparse_spsort_stage      stage,
                                               size_t*                     buffer_size_in_bytes)
    {
        ROCSPARSE_ROUTINE_TRACE;

        const rocsparse_format format = source->format;

        const rocsparse_direction dir = descr->get_dir();

        switch(stage)
        {
        case rocsparse_spsort_stage_analysis:
        {
            switch(format)
            {
            case rocsparse_format_coo:
            case rocsparse_format_csr:
            case rocsparse_format_csc:
            {
                *buffer_size_in_bytes = 0;
                return rocsparse_status_success;
            }
            case rocsparse_format_coo_aos:
            case rocsparse_format_bsr:
            case rocsparse_format_ell:
            case rocsparse_format_bell:
            case rocsparse_format_sell:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }
        }
        case rocsparse_spsort_stage_compute:
        {
            switch(format)
            {
            case rocsparse_format_coo:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::coosort_buffer_size(handle,
                                                   rocsparse_coosort_alg_default,
                                                   dir,
                                                   source,
                                                   target,
                                                   buffer_size_in_bytes));
                return rocsparse_status_success;
            }
            case rocsparse_format_csr:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsort_buffer_size(
                    handle, rocsparse_csrsort_alg_default, source, target, buffer_size_in_bytes));
                return rocsparse_status_success;
            }
            case rocsparse_format_csc:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::cscsort_buffer_size(
                    handle, rocsparse_cscsort_alg_default, source, target, buffer_size_in_bytes));
                return rocsparse_status_success;
            }
            case rocsparse_format_coo_aos:
            case rocsparse_format_bsr:
            case rocsparse_format_ell:
            case rocsparse_format_bell:
            case rocsparse_format_sell:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }
        }
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    }
}

extern "C" rocsparse_status rocsparse_spsort_buffer_size(rocsparse_handle            handle,
                                                         rocsparse_spsort_descr      descr,
                                                         rocsparse_const_spmat_descr source,
                                                         rocsparse_spmat_descr       target,
                                                         rocsparse_spsort_stage      stage,
                                                         size_t*          buffer_size_in_bytes,
                                                         rocsparse_error* error)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    ROCSPARSE_CHECKARG_POINTER(2, source);
    ROCSPARSE_CHECKARG_POINTER(3, target);
    ROCSPARSE_CHECKARG_ENUM(4, stage);
    ROCSPARSE_CHECKARG_POINTER(5, buffer_size_in_bytes);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::spsort_check_arguments(descr, source, target));

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::spsort_buffer_size(handle, descr, source, target, stage, buffer_size_in_bytes));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/* ************************************************************************
 * Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/generic/rocsparse_gather.h"
#include "rocsparse_control.hpp"
#include "rocsparse_gthr.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_utility.hpp"

extern "C" rocsparse_status rocsparse_gather(rocsparse_handle            handle,
                                             rocsparse_const_dnvec_descr y,
                                             rocsparse_spvec_descr       x)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check for invalid handle
    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    // Logging
    rocsparse::log_trace(handle, "rocsparse_gather", (const void*&)y, (const void*&)x);

    // Check for invalid descriptors
    ROCSPARSE_CHECKARG_POINTER(1, y);
    ROCSPARSE_CHECKARG_POINTER(2, x);

    // Check if descriptors are initialized
    ROCSPARSE_CHECKARG(1, y, y->get_init() == false, rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(2, x, x->get_init() == false, rocsparse_status_not_initialized);

    ROCSPARSE_CHECKARG(1, y, (y->get_batch_count() != 1), rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(2, x, (x->get_batch_count() != 1), rocsparse_status_not_implemented);

    // Check for matching types while we do not support mixed precision computation
    ROCSPARSE_CHECKARG(
        2, x, (x->get_data_type() != y->get_data_type()), rocsparse_status_not_implemented);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gthr(handle,
                                              x->get_nnz(),
                                              y->get_data_type(),
                                              y->get_const_values(),
                                              x->get_data_type(),
                                              x->get_val_data(),
                                              x->get_idx_type(),
                                              x->get_const_idx_data(),
                                              x->get_idx_base()));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

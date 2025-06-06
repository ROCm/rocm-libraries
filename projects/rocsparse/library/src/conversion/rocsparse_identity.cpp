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
#include "internal/conversion/rocsparse_inverse_permutation.h"
#include "rocsparse_utility.hpp"

#include "rocsparse_gcreate_identity_permutation.hpp"
#include "rocsparse_identity.hpp"

#include "identity_device.h"

template <typename I>
rocsparse_status rocsparse::create_identity_permutation_core(rocsparse_handle handle, I n, I* p)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Stream
    hipStream_t stream = handle->stream;

#define IDENTITY_DIM 512
    dim3 identity_blocks((n - 1) / IDENTITY_DIM + 1);
    dim3 identity_threads(IDENTITY_DIM);

    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::identity_kernel<IDENTITY_DIM>),
                                       identity_blocks,
                                       identity_threads,
                                       0,
                                       stream,
                                       n,
                                       p);
#undef IDENTITY_DIM

    return rocsparse_status_success;
}

template <typename I>
rocsparse_status rocsparse::create_identity_permutation_template(rocsparse_handle handle, I n, I* p)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Quick return if possible
    if(n == 0)
    {
        return rocsparse_status_success;
    }

    return rocsparse::create_identity_permutation_core(handle, n, p);
}

template <typename I>
rocsparse_status rocsparse::create_identity_permutation_impl(rocsparse_handle handle, I n, I* p)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle, "rocsparse_create_identity_permutation", n, (const void*&)p);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, n);
    ROCSPARSE_CHECKARG_ARRAY(2, n, p);
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::create_identity_permutation_template(handle, n, p));
    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE)                                                        \
    template rocsparse_status rocsparse::create_identity_permutation_core(        \
        rocsparse_handle handle, ITYPE n, ITYPE* p);                              \
    template rocsparse_status rocsparse::create_identity_permutation_template(    \
        rocsparse_handle handle, ITYPE n, ITYPE* p);                              \
    template rocsparse_status rocsparse::create_identity_permutation_impl<ITYPE>( \
        rocsparse_handle handle, ITYPE n, ITYPE * p)

INSTANTIATE(int32_t);
INSTANTIATE(int64_t);
#undef INSTANTIATE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_create_identity_permutation(rocsparse_handle handle,
                                                                  rocsparse_int    n,
                                                                  rocsparse_int*   p)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::create_identity_permutation_impl(handle, n, p));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_set_identity_permutation(rocsparse_handle    handle,
                                                               int64_t             n,
                                                               void*               p,
                                                               rocsparse_indextype indextype)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gcreate_identity_permutation(handle, n, indextype, p));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

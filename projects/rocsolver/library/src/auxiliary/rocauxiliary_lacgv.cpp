/* **************************************************************************
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#include "rocauxiliary_lacgv.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I>
rocblas_status rocsolver_lacgv_impl(rocblas_handle handle, const I n, T* x, const I incx)
{
    ROCSOLVER_ENTER_TOP("lacgv", "-n", n, "--incx", incx);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_lacgv_argCheck(handle, n, incx, x);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftx = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride stridex = 0;
    I batch_count = 1;

    // this function does not require memory work space
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_size_unchanged;

    // execution
    return rocsolver_lacgv_template<T>(handle, n, x, shiftx, incx, stridex, batch_count);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_clacgv(rocblas_handle handle,
                                const rocblas_int n,
                                rocblas_float_complex* x,
                                const rocblas_int incx)
{
    return rocsolver::rocsolver_lacgv_impl<rocblas_float_complex>(handle, n, x, incx);
}

rocblas_status rocsolver_zlacgv(rocblas_handle handle,
                                const rocblas_int n,
                                rocblas_double_complex* x,
                                const rocblas_int incx)
{
    return rocsolver::rocsolver_lacgv_impl<rocblas_double_complex>(handle, n, x, incx);
}

rocblas_status rocsolver_clacgv_64(rocblas_handle handle,
                                   const int64_t n,
                                   rocblas_float_complex* x,
                                   const int64_t incx)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_lacgv_impl<rocblas_float_complex>(handle, n, x, incx);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zlacgv_64(rocblas_handle handle,
                                   const int64_t n,
                                   rocblas_double_complex* x,
                                   const int64_t incx)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_lacgv_impl<rocblas_double_complex>(handle, n, x, incx);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C

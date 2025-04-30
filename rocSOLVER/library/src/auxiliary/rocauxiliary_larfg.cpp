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

#include "rocauxiliary_larfg.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I>
rocblas_status
    rocsolver_larfg_impl(rocblas_handle handle, const I n, T* alpha, T* x, const I incx, T* tau)
{
    // TODO: How to get alpha for bench logging
    ROCSOLVER_ENTER_TOP("larfg", "-n", n, "--incx", incx);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_larfg_argCheck(handle, n, incx, alpha, x, tau);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shifta = 0;
    rocblas_stride shiftx = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride stridex = 0;
    rocblas_stride strideP = 0;
    I batch_count = 1;

    // memory workspace sizes:
    // size of re-usable workspace
    size_t size_work;
    // size to store the norms
    size_t size_norms;
    ROCBLAS_CHECK(rocsolver_larfg_getMemorySize<T>(n, batch_count, &size_work, &size_norms));

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work, size_norms);

    // memory workspace allocation
    void *work, *norms;
    rocblas_device_malloc mem(handle, size_work, size_norms);
    if(!mem)
        return rocblas_status_memory_error;

    work = mem[0];
    norms = mem[1];

    // execution
    return rocsolver_larfg_template<T>(handle, n, alpha, shifta, x, shiftx, incx, stridex, tau,
                                       strideP, batch_count, (T*)work, (T*)norms);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_slarfg(rocblas_handle handle,
                                const rocblas_int n,
                                float* alpha,
                                float* x,
                                const rocblas_int incx,
                                float* tau)
{
    return rocsolver::rocsolver_larfg_impl<float>(handle, n, alpha, x, incx, tau);
}

rocblas_status rocsolver_dlarfg(rocblas_handle handle,
                                const rocblas_int n,
                                double* alpha,
                                double* x,
                                const rocblas_int incx,
                                double* tau)
{
    return rocsolver::rocsolver_larfg_impl<double>(handle, n, alpha, x, incx, tau);
}

rocblas_status rocsolver_clarfg(rocblas_handle handle,
                                const rocblas_int n,
                                rocblas_float_complex* alpha,
                                rocblas_float_complex* x,
                                const rocblas_int incx,
                                rocblas_float_complex* tau)
{
    return rocsolver::rocsolver_larfg_impl<rocblas_float_complex>(handle, n, alpha, x, incx, tau);
}

rocblas_status rocsolver_zlarfg(rocblas_handle handle,
                                const rocblas_int n,
                                rocblas_double_complex* alpha,
                                rocblas_double_complex* x,
                                const rocblas_int incx,
                                rocblas_double_complex* tau)
{
    return rocsolver::rocsolver_larfg_impl<rocblas_double_complex>(handle, n, alpha, x, incx, tau);
}

rocblas_status rocsolver_slarfg_64(rocblas_handle handle,
                                   const int64_t n,
                                   float* alpha,
                                   float* x,
                                   const int64_t incx,
                                   float* tau)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_larfg_impl<float>(handle, n, alpha, x, incx, tau);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_dlarfg_64(rocblas_handle handle,
                                   const int64_t n,
                                   double* alpha,
                                   double* x,
                                   const int64_t incx,
                                   double* tau)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_larfg_impl<double>(handle, n, alpha, x, incx, tau);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_clarfg_64(rocblas_handle handle,
                                   const int64_t n,
                                   rocblas_float_complex* alpha,
                                   rocblas_float_complex* x,
                                   const int64_t incx,
                                   rocblas_float_complex* tau)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_larfg_impl<rocblas_float_complex>(handle, n, alpha, x, incx, tau);
#else
    return rocblas_status_not_implemented;
#endif
}

rocblas_status rocsolver_zlarfg_64(rocblas_handle handle,
                                   const int64_t n,
                                   rocblas_double_complex* alpha,
                                   rocblas_double_complex* x,
                                   const int64_t incx,
                                   rocblas_double_complex* tau)
{
#ifdef HAVE_ROCBLAS_64
    return rocsolver::rocsolver_larfg_impl<rocblas_double_complex>(handle, n, alpha, x, incx, tau);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C

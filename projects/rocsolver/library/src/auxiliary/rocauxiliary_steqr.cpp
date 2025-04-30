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

#include "rocauxiliary_steqr.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename S>
rocblas_status rocsolver_steqr_impl(rocblas_handle handle,
                                    const rocblas_evect evect,
                                    const rocblas_int n,
                                    S* D,
                                    S* E,
                                    T* C,
                                    const rocblas_int ldc,
                                    rocblas_int* info)
{
    ROCSOLVER_ENTER_TOP("steqr", "--evect", evect, "-n", n, "--ldc", ldc);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_steqr_argCheck(handle, evect, n, D, E, C, ldc, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftD = 0;
    rocblas_int shiftE = 0;
    rocblas_int shiftC = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideD = 0;
    rocblas_stride strideE = 0;
    rocblas_stride strideC = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for lasrt stack/steqr workspace
    size_t size_work_stack;
    rocsolver_steqr_getMemorySize<T, S>(evect, n, batch_count, &size_work_stack);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work_stack);

    // memory workspace allocation
    void* work_stack;
    rocblas_device_malloc mem(handle, size_work_stack);
    if(!mem)
        return rocblas_status_memory_error;

    work_stack = mem[0];

    // execution
    return rocsolver_steqr_template<T>(handle, evect, n, D, shiftD, strideD, E, shiftE, strideE, C,
                                       shiftC, ldc, strideC, info, batch_count, work_stack);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_ssteqr(rocblas_handle handle,
                                const rocblas_evect evect,
                                const rocblas_int n,
                                float* D,
                                float* E,
                                float* C,
                                const rocblas_int ldc,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_steqr_impl<float>(handle, evect, n, D, E, C, ldc, info);
}

rocblas_status rocsolver_dsteqr(rocblas_handle handle,
                                const rocblas_evect evect,
                                const rocblas_int n,
                                double* D,
                                double* E,
                                double* C,
                                const rocblas_int ldc,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_steqr_impl<double>(handle, evect, n, D, E, C, ldc, info);
}

rocblas_status rocsolver_csteqr(rocblas_handle handle,
                                const rocblas_evect evect,
                                const rocblas_int n,
                                float* D,
                                float* E,
                                rocblas_float_complex* C,
                                const rocblas_int ldc,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_steqr_impl<rocblas_float_complex>(handle, evect, n, D, E, C, ldc,
                                                                  info);
}

rocblas_status rocsolver_zsteqr(rocblas_handle handle,
                                const rocblas_evect evect,
                                const rocblas_int n,
                                double* D,
                                double* E,
                                rocblas_double_complex* C,
                                const rocblas_int ldc,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_steqr_impl<rocblas_double_complex>(handle, evect, n, D, E, C, ldc,
                                                                   info);
}

} // extern C

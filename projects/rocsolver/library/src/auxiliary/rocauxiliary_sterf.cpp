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

#include "rocauxiliary_sterf.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T>
rocblas_status
    rocsolver_sterf_impl(rocblas_handle handle, const rocblas_int n, T* D, T* E, rocblas_int* info)
{
    ROCSOLVER_ENTER_TOP("sterf", "-n", n);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_sterf_argCheck(handle, n, D, E, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftD = 0;
    rocblas_int shiftE = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideD = 0;
    rocblas_stride strideE = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for lasrt stack
    size_t size_stack;
    rocsolver_sterf_getMemorySize<T>(n, batch_count, &size_stack);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_stack);

    // memory workspace allocation
    void* stack;
    rocblas_device_malloc mem(handle, size_stack);
    if(!mem)
        return rocblas_status_memory_error;

    stack = mem[0];

    // execution
    return rocsolver_sterf_template<T>(handle, n, D, shiftD, strideD, E, shiftE, strideE, info,
                                       batch_count, (rocblas_int*)stack);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status
    rocsolver_ssterf(rocblas_handle handle, const rocblas_int n, float* D, float* E, rocblas_int* info)
{
    return rocsolver::rocsolver_sterf_impl<float>(handle, n, D, E, info);
}

rocblas_status rocsolver_dsterf(rocblas_handle handle,
                                const rocblas_int n,
                                double* D,
                                double* E,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_sterf_impl<double>(handle, n, D, E, info);
}

} // extern C

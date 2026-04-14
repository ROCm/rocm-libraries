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

#include "rocauxiliary_stein.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename S>
rocblas_status rocsolver_stein_impl(rocblas_handle handle,
                                    const rocblas_int n,
                                    S* D,
                                    S* E,
                                    rocblas_int* nev,
                                    S* W,
                                    rocblas_int* iblock,
                                    rocblas_int* isplit,
                                    T* Z,
                                    const rocblas_int ldz,
                                    rocblas_int* ifail,
                                    rocblas_int* info)
{
    ROCSOLVER_ENTER_TOP("stein", "-n", n, "--ldz", ldz);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_stein_argCheck(handle, n, D, E, nev, W, iblock, isplit, Z, ldz, ifail, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftD = 0;
    rocblas_int shiftE = 0;
    rocblas_int shiftW = 0;
    rocblas_int shiftZ = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideD = 0;
    rocblas_stride strideE = 0;
    rocblas_stride strideW = 0;
    rocblas_stride strideIblock = 0;
    rocblas_stride strideIsplit = 0;
    rocblas_stride strideZ = 0;
    rocblas_stride strideIfail = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for lagtf/stein workspace
    size_t size_work, size_iwork;
    rocsolver_stein_getMemorySize<T, S>(n, batch_count, &size_work, &size_iwork);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work, size_iwork);

    // memory workspace allocation
    void *work, *iwork;
    rocblas_device_malloc mem(handle, size_work, size_iwork);
    if(!mem)
        return rocblas_status_memory_error;

    work = mem[0];
    iwork = mem[1];

    // execution
    return rocsolver_stein_template<T>(handle, n, D, shiftD, strideD, E, shiftE, strideE, nev, W,
                                       shiftW, strideW, iblock, strideIblock, isplit, strideIsplit,
                                       Z, shiftZ, ldz, strideZ, ifail, strideIfail, info,
                                       batch_count, (S*)work, (rocblas_int*)iwork);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sstein(rocblas_handle handle,
                                const rocblas_int n,
                                float* D,
                                float* E,
                                rocblas_int* nev,
                                float* W,
                                rocblas_int* iblock,
                                rocblas_int* isplit,
                                float* Z,
                                const rocblas_int ldz,
                                rocblas_int* ifail,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_stein_impl<float, float>(handle, n, D, E, nev, W, iblock, isplit, Z,
                                                         ldz, ifail, info);
}

rocblas_status rocsolver_dstein(rocblas_handle handle,
                                const rocblas_int n,
                                double* D,
                                double* E,
                                rocblas_int* nev,
                                double* W,
                                rocblas_int* iblock,
                                rocblas_int* isplit,
                                double* Z,
                                const rocblas_int ldz,
                                rocblas_int* ifail,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_stein_impl<double, double>(handle, n, D, E, nev, W, iblock, isplit,
                                                           Z, ldz, ifail, info);
}

rocblas_status rocsolver_cstein(rocblas_handle handle,
                                const rocblas_int n,
                                float* D,
                                float* E,
                                rocblas_int* nev,
                                float* W,
                                rocblas_int* iblock,
                                rocblas_int* isplit,
                                rocblas_float_complex* Z,
                                const rocblas_int ldz,
                                rocblas_int* ifail,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_stein_impl<rocblas_float_complex, float>(
        handle, n, D, E, nev, W, iblock, isplit, Z, ldz, ifail, info);
}

rocblas_status rocsolver_zstein(rocblas_handle handle,
                                const rocblas_int n,
                                double* D,
                                double* E,
                                rocblas_int* nev,
                                double* W,
                                rocblas_int* iblock,
                                rocblas_int* isplit,
                                rocblas_double_complex* Z,
                                const rocblas_int ldz,
                                rocblas_int* ifail,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_stein_impl<rocblas_double_complex, double>(
        handle, n, D, E, nev, W, iblock, isplit, Z, ldz, ifail, info);
}

} // extern C

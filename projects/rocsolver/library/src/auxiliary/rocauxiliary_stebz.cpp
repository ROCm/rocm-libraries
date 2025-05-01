/* **************************************************************************
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocauxiliary_stebz.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T>
rocblas_status rocsolver_stebz_impl(rocblas_handle handle,
                                    const rocblas_erange erange,
                                    const rocblas_eorder eorder,
                                    const rocblas_int n,
                                    const T vl,
                                    const T vu,
                                    const rocblas_int il,
                                    const rocblas_int iu,
                                    const T abstol,
                                    T* D,
                                    T* E,
                                    rocblas_int* nev,
                                    rocblas_int* nsplit,
                                    T* W,
                                    rocblas_int* iblock,
                                    rocblas_int* isplit,
                                    rocblas_int* info)
{
    ROCSOLVER_ENTER_TOP("stebz", "--erange", erange, "--eorder", eorder, "-n", n, "--vl", vl,
                        "--vu", vu, "--il", il, "--iu", iu, "--abstol", abstol);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_stebz_argCheck(handle, erange, eorder, n, vl, vu, il, iu, D, E,
                                                 nev, nsplit, W, iblock, isplit, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftD = 0;
    rocblas_int shiftE = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideD = 0;
    rocblas_stride strideE = 0;
    rocblas_stride strideW = 0;
    rocblas_stride strideIblock = 0;
    rocblas_stride strideIsplit = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    size_t size_work, size_pivmin, size_Esqr, size_bounds, size_inter, size_ninter;
    rocsolver_stebz_getMemorySize<T>(n, batch_count, &size_work, &size_pivmin, &size_Esqr,
                                     &size_bounds, &size_inter, &size_ninter);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work, size_pivmin, size_Esqr,
                                                      size_bounds, size_inter, size_ninter);

    // memory workspace allocation
    void *work, *pivmin, *Esqr, *bounds, *inter, *ninter;
    rocblas_device_malloc mem(handle, size_work, size_pivmin, size_Esqr, size_bounds, size_inter,
                              size_ninter);
    if(!mem)
        return rocblas_status_memory_error;

    work = mem[0];
    pivmin = mem[1];
    Esqr = mem[2];
    bounds = mem[3];
    inter = mem[4];
    ninter = mem[5];

    // execution
    return rocsolver_stebz_template<T>(
        handle, erange, eorder, n, vl, vu, il, iu, abstol, D, shiftD, strideD, E, shiftE, strideE,
        nev, nsplit, W, strideW, iblock, strideIblock, isplit, strideIsplit, info, batch_count,
        (rocblas_int*)work, (T*)pivmin, (T*)Esqr, (T*)bounds, (T*)inter, (rocblas_int*)ninter);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sstebz(rocblas_handle handle,
                                const rocblas_erange erange,
                                const rocblas_eorder eorder,
                                const rocblas_int n,
                                const float vl,
                                const float vu,
                                const rocblas_int il,
                                const rocblas_int iu,
                                const float abstol,
                                float* D,
                                float* E,
                                rocblas_int* nev,
                                rocblas_int* nsplit,
                                float* W,
                                rocblas_int* iblock,
                                rocblas_int* isplit,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_stebz_impl<float>(handle, erange, eorder, n, vl, vu, il, iu, abstol,
                                                  D, E, nev, nsplit, W, iblock, isplit, info);
}

rocblas_status rocsolver_dstebz(rocblas_handle handle,
                                const rocblas_erange erange,
                                const rocblas_eorder eorder,
                                const rocblas_int n,
                                const double vl,
                                const double vu,
                                const rocblas_int il,
                                const rocblas_int iu,
                                const double abstol,
                                double* D,
                                double* E,
                                rocblas_int* nev,
                                rocblas_int* nsplit,
                                double* W,
                                rocblas_int* iblock,
                                rocblas_int* isplit,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_stebz_impl<double>(handle, erange, eorder, n, vl, vu, il, iu, abstol,
                                                   D, E, nev, nsplit, W, iblock, isplit, info);
}

} // extern C

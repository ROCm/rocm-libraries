/* **************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocauxiliary_stedc.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename S>
rocblas_status rocsolver_stedc_impl(rocblas_handle handle,
                                    const rocblas_evect evect,
                                    const rocblas_int n,
                                    S* D,
                                    S* E,
                                    T* C,
                                    const rocblas_int ldc,
                                    rocblas_int* info)
{
    ROCSOLVER_ENTER_TOP("stedc", "--evect", evect, "-n", n, "--ldc", ldc);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_stedc_argCheck(handle, evect, n, D, E, C, ldc, info);
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
    // size for lasrt stack/stedc workspace
    size_t size_work_stack;
    // size for temporary computations
    size_t size_tempvect, size_tempgemm;
    // size for pointers to workspace (batched case)
    size_t size_workArr;
    // size for vector with positions of split blocks
    size_t size_splits_map;
    // size for temporary diagonal and z vectors.
    size_t size_tmpz;
    rocsolver_stedc_getMemorySize<false, T, S>(evect, n, batch_count, &size_work_stack,
                                               &size_tempvect, &size_tempgemm, &size_tmpz,
                                               &size_splits_map, &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work_stack, size_tempvect,
                                                      size_tempgemm, size_tmpz, size_splits_map,
                                                      size_workArr);

    // memory workspace allocation
    void *work_stack, *tempvect, *tempgemm, *tmpz, *splits_map, *workArr;
    rocblas_device_malloc mem(handle, size_work_stack, size_tempvect, size_tempgemm, size_tmpz,
                              size_splits_map, size_workArr);
    if(!mem)
        return rocblas_status_memory_error;

    work_stack = mem[0];
    tempvect = mem[1];
    tempgemm = mem[2];
    tmpz = mem[3];
    splits_map = mem[4];
    workArr = mem[5];

    // execution
    return rocsolver_stedc_template<false, false, T>(
        handle, evect, n, D, shiftD, strideD, E, shiftE, strideE, C, shiftC, ldc, strideC, info,
        batch_count, work_stack, (S*)tempvect, (S*)tempgemm, (S*)tmpz, (rocblas_int*)splits_map,
        (S**)workArr);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_sstedc(rocblas_handle handle,
                                const rocblas_evect evect,
                                const rocblas_int n,
                                float* D,
                                float* E,
                                float* C,
                                const rocblas_int ldc,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_stedc_impl<float>(handle, evect, n, D, E, C, ldc, info);
}

rocblas_status rocsolver_dstedc(rocblas_handle handle,
                                const rocblas_evect evect,
                                const rocblas_int n,
                                double* D,
                                double* E,
                                double* C,
                                const rocblas_int ldc,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_stedc_impl<double>(handle, evect, n, D, E, C, ldc, info);
}

rocblas_status rocsolver_cstedc(rocblas_handle handle,
                                const rocblas_evect evect,
                                const rocblas_int n,
                                float* D,
                                float* E,
                                rocblas_float_complex* C,
                                const rocblas_int ldc,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_stedc_impl<rocblas_float_complex>(handle, evect, n, D, E, C, ldc,
                                                                  info);
}

rocblas_status rocsolver_zstedc(rocblas_handle handle,
                                const rocblas_evect evect,
                                const rocblas_int n,
                                double* D,
                                double* E,
                                rocblas_double_complex* C,
                                const rocblas_int ldc,
                                rocblas_int* info)
{
    return rocsolver::rocsolver_stedc_impl<rocblas_double_complex>(handle, evect, n, D, E, C, ldc,
                                                                   info);
}

} // extern C

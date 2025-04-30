/* **************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocrefact_csrrf_refactchol.hpp"

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename U>
rocblas_status rocsolver_csrrf_refactchol_impl(rocblas_handle handle,
                                               const rocblas_int n,
                                               const rocblas_int nnzA,
                                               rocblas_int* ptrA,
                                               rocblas_int* indA,
                                               U valA,
                                               const rocblas_int nnzT,
                                               rocblas_int* ptrT,
                                               rocblas_int* indT,
                                               U valT,
                                               rocblas_int* pivQ,
                                               rocsolver_rfinfo rfinfo)
{
    ROCSOLVER_ENTER_TOP("csrrf_refactchol", "-n", n, "--nnzA", nnzA, "--nnzT", nnzT);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_csrrf_refactchol_argCheck(handle, n, nnzA, ptrA, indA, valA, nnzT,
                                                            ptrT, indT, valT, pivQ, rfinfo);
    if(st != rocblas_status_continue)
        return st;

    // TODO: add batched versions
    // working with unshifted arrays
    // normal (non-batched non-strided) execution

    // memory workspace sizes:
    // size for temp buffer in refactlu calls
    size_t size_work = 0;

    rocsolver_csrrf_refactchol_getMemorySize<T>(n, nnzT, ptrT, indT, valT, rfinfo, &size_work);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work);

    // memory workspace allocation
    void* work = nullptr;
    rocblas_device_malloc mem(handle, size_work);

    if(!mem)
        return rocblas_status_memory_error;

    work = mem[0];

    // execution
    return rocsolver_csrrf_refactchol_template<T, U>(handle, n, nnzA, ptrA, indA, valA, nnzT, ptrT,
                                                     indT, valT, pivQ, rfinfo, work);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_scsrrf_refactchol(rocblas_handle handle,
                                           const rocblas_int n,
                                           const rocblas_int nnzA,
                                           rocblas_int* ptrA,
                                           rocblas_int* indA,
                                           float* valA,
                                           const rocblas_int nnzT,
                                           rocblas_int* ptrT,
                                           rocblas_int* indT,
                                           float* valT,
                                           rocblas_int* pivQ,
                                           rocsolver_rfinfo rfinfo)
{
    return rocsolver::rocsolver_csrrf_refactchol_impl<float>(handle, n, nnzA, ptrA, indA, valA,
                                                             nnzT, ptrT, indT, valT, pivQ, rfinfo);
}

rocblas_status rocsolver_dcsrrf_refactchol(rocblas_handle handle,
                                           const rocblas_int n,
                                           const rocblas_int nnzA,
                                           rocblas_int* ptrA,
                                           rocblas_int* indA,
                                           double* valA,
                                           const rocblas_int nnzT,
                                           rocblas_int* ptrT,
                                           rocblas_int* indT,
                                           double* valT,
                                           rocblas_int* pivQ,
                                           rocsolver_rfinfo rfinfo)
{
    return rocsolver::rocsolver_csrrf_refactchol_impl<double>(handle, n, nnzA, ptrA, indA, valA,
                                                              nnzT, ptrT, indT, valT, pivQ, rfinfo);
}

} // extern C

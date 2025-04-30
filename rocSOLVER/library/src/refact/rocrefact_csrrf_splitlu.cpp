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

#include "rocrefact_csrrf_splitlu.hpp"

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename U>
rocblas_status rocsolver_csrrf_splitlu_impl(rocblas_handle handle,
                                            const rocblas_int n,
                                            const rocblas_int nnzT,
                                            rocblas_int* ptrT,
                                            rocblas_int* indT,
                                            U valT,
                                            rocblas_int* ptrL,
                                            rocblas_int* indL,
                                            U valL,
                                            rocblas_int* ptrU,
                                            rocblas_int* indU,
                                            U valU)
{
    ROCSOLVER_ENTER_TOP("csrrf_splitlu", "-n", n, "--nnzT", nnzT);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_csrrf_splitlu_argCheck(handle, n, nnzT, ptrT, indT, valT, ptrL,
                                                         indL, valL, ptrU, indU, valU);
    if(st != rocblas_status_continue)
        return st;

    // TODO: add batched versions
    // working with unshifted arrays
    // normal (non-batched non-strided) execution

    // memory workspace sizes:
    // size to store number of non-zeros per row
    size_t size_work = 0;

    ROCBLAS_CHECK(rocsolver_csrrf_splitlu_getMemorySize<T>(n, nnzT, ptrT, &size_work));

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work);

    // memory workspace allocation
    void* work = nullptr;
    rocblas_device_malloc mem(handle, size_work);

    if(!mem)
        return rocblas_status_memory_error;

    work = mem[0];

    // execution
    return rocsolver_csrrf_splitlu_template<T>(handle, n, nnzT, ptrT, indT, valT, ptrL, indL, valL,
                                               ptrU, indU, valU, static_cast<rocblas_int*>(work),
                                               size_work);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_scsrrf_splitlu(rocblas_handle handle,
                                        const rocblas_int n,
                                        const rocblas_int nnzT,
                                        rocblas_int* ptrT,
                                        rocblas_int* indT,
                                        float* valT,
                                        rocblas_int* ptrL,
                                        rocblas_int* indL,
                                        float* valL,
                                        rocblas_int* ptrU,
                                        rocblas_int* indU,
                                        float* valU)
{
    return rocsolver::rocsolver_csrrf_splitlu_impl<float>(handle, n, nnzT, ptrT, indT, valT, ptrL,
                                                          indL, valL, ptrU, indU, valU);
}

rocblas_status rocsolver_dcsrrf_splitlu(rocblas_handle handle,
                                        const rocblas_int n,
                                        const rocblas_int nnzT,
                                        rocblas_int* ptrT,
                                        rocblas_int* indT,
                                        double* valT,
                                        rocblas_int* ptrL,
                                        rocblas_int* indL,
                                        double* valL,
                                        rocblas_int* ptrU,
                                        rocblas_int* indU,
                                        double* valU)
{
    return rocsolver::rocsolver_csrrf_splitlu_impl<double>(handle, n, nnzT, ptrT, indT, valT, ptrL,
                                                           indL, valL, ptrU, indU, valU);
}

} // extern C

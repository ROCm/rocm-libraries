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

#include "rocrefact_csrrf_sumlu.hpp"

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename U>
rocblas_status rocsolver_csrrf_sumlu_impl(rocblas_handle handle,
                                          const rocblas_int n,
                                          const rocblas_int nnzL,
                                          rocblas_int* ptrL,
                                          rocblas_int* indL,
                                          U valL,
                                          const rocblas_int nnzU,
                                          rocblas_int* ptrU,
                                          rocblas_int* indU,
                                          U valU,
                                          rocblas_int* ptrT,
                                          rocblas_int* indT,
                                          U valT)
{
    ROCSOLVER_ENTER_TOP("csrrf_sumlu", "-n", n, "--nnzL", nnzL, "--nnzU", nnzU);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_csrrf_sumlu_argCheck(handle, n, nnzL, ptrL, indL, valL, nnzU,
                                                       ptrU, indU, valU, ptrT, indT, valT);
    if(st != rocblas_status_continue)
        return st;

    // TODO: add batched versions
    // working with unshifted arrays
    // normal (non-batched non-strided) execution

    // this function does not require memory work space
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_size_unchanged;

    // execution
    return rocsolver_csrrf_sumlu_template<T>(handle, n, nnzL, ptrL, indL, valL, nnzU, ptrU, indU,
                                             valU, ptrT, indT, valT);
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocsolver_scsrrf_sumlu(rocblas_handle handle,
                                      const rocblas_int n,
                                      const rocblas_int nnzL,
                                      rocblas_int* ptrL,
                                      rocblas_int* indL,
                                      float* valL,
                                      const rocblas_int nnzU,
                                      rocblas_int* ptrU,
                                      rocblas_int* indU,
                                      float* valU,
                                      rocblas_int* ptrT,
                                      rocblas_int* indT,
                                      float* valT)
{
    return rocsolver::rocsolver_csrrf_sumlu_impl<float>(handle, n, nnzL, ptrL, indL, valL, nnzU,
                                                        ptrU, indU, valU, ptrT, indT, valT);
}

rocblas_status rocsolver_dcsrrf_sumlu(rocblas_handle handle,
                                      const rocblas_int n,
                                      const rocblas_int nnzL,
                                      rocblas_int* ptrL,
                                      rocblas_int* indL,
                                      double* valL,
                                      const rocblas_int nnzU,
                                      rocblas_int* ptrU,
                                      rocblas_int* indU,
                                      double* valU,
                                      rocblas_int* ptrT,
                                      rocblas_int* indT,
                                      double* valT)
{
    return rocsolver::rocsolver_csrrf_sumlu_impl<double>(handle, n, nnzL, ptrL, indL, valL, nnzU,
                                                         ptrU, indU, valU, ptrT, indT, valT);
}

} // extern C

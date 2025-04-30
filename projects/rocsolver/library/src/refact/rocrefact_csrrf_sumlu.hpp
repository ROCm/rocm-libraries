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

#pragma once

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T>
ROCSOLVER_KERNEL void rf_sumLU_kernel(const rocblas_int n,
                                      const rocblas_int nnzL,
                                      rocblas_int* Lp,
                                      rocblas_int* Li,
                                      T* Lx,
                                      const rocblas_int nnzU,
                                      rocblas_int* Up,
                                      rocblas_int* Ui,
                                      T* Ux,
                                      rocblas_int* LUp,
                                      rocblas_int* LUi,
                                      T* LUx)
{
    rocblas_int tid = hipThreadIdx_x;

    // -----------------------
    // 1st pass to set up LUp
    // -----------------------
    rocblas_int i, j;
    rocblas_int irow, icol, istart, iend;

    for(irow = tid; irow <= n; irow += hipBlockDim_x)
        LUp[irow] = (Lp[irow] - irow) + Up[irow];
    __syncthreads();

    // ------------------------------
    // 2nd pass to populate LUi, LUx
    // ------------------------------
    for(irow = tid; irow < n; irow += hipBlockDim_x)
    {
        istart = Lp[irow];
        iend = Lp[irow + 1];
        for(i = istart; i < iend - 1; i++)
        {
            j = LUp[irow] + (i - istart);

            LUi[j] = Li[i];
            LUx[j] = Lx[i];
        }

        istart = Up[irow];
        iend = Up[irow + 1];
        for(i = istart; i < iend; i++)
        {
            j = LUp[irow + 1] - (iend - i);

            LUi[j] = Ui[i];
            LUx[j] = Ux[i];
        }
    }
}

template <typename T>
rocblas_status rocsolver_csrrf_sumlu_argCheck(rocblas_handle handle,
                                              const rocblas_int n,
                                              const rocblas_int nnzL,
                                              rocblas_int* ptrL,
                                              rocblas_int* indL,
                                              T valL,
                                              const rocblas_int nnzU,
                                              rocblas_int* ptrU,
                                              rocblas_int* indU,
                                              T valU,
                                              rocblas_int* ptrT,
                                              rocblas_int* indT,
                                              T valT)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A
    if(handle == nullptr)
    {
        return rocblas_status_invalid_handle;
    };

    // 2. invalid size
    if(n < 0 || nnzL < n || nnzU < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if(!ptrL || !ptrU || !ptrT || (nnzL && (!indL || !valL)) || (nnzU && (!indU || !valU))
       || ((nnzL > n || nnzU > 0) && (!indT || !valT)))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U>
rocblas_status rocsolver_csrrf_sumlu_template(rocblas_handle handle,
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
    ROCSOLVER_ENTER("csrrf_sumlu", "n:", n, "nnzL:", nnzL, "nnzU:", nnzU);

    // quick return
    if(n == 0)
        return rocblas_status_success;

    hipStream_t stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &stream));

    // quick return with matrix zero
    if(nnzL == n && nnzU == 0)
    {
        // set ptrT = 0
        rocblas_int blocks = n / BS1 + 1;
        dim3 grid(blocks, 1, 1);
        dim3 threads(BS1, 1, 1);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, grid, threads, 0, stream, ptrT, n + 1, 0);

        return rocblas_status_success;
    }

    rocblas_int nthreads = BS1;
    rocblas_int nblocks = 1;
    ROCSOLVER_LAUNCH_KERNEL(rf_sumLU_kernel<T>, dim3(nblocks), dim3(nthreads), 0, stream, n, nnzL,
                            ptrL, indL, valL, nnzU, ptrU, indU, valU, ptrT, indT, valT);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

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
#include "rocsolver_rfinfo.hpp"
#include "rocsparse.hpp"

#include "refact_helpers.hpp"

ROCSOLVER_BEGIN_NAMESPACE

// -------------------------------------------
// Compute B = beta * B + alpha * (P * A * Q') as
// (1) B = beta * B
// (2) B += alpha * (P * A * Q')
// where sparsity pattern of reordered A is a proper subset
// of sparsity pattern of B.
// Further assume for each row, the column indices are
// in increasing sorted order
// -------------------------------------------
template <typename T>
ROCSOLVER_KERNEL void rf_add_PAQ_kernel(const rocblas_int n,
                                        rocblas_int* pivP,
                                        rocblas_int* inv_pivQ,
                                        const T alpha,
                                        rocblas_int* Ap,
                                        rocblas_int* Ai,
                                        T* Ax,
                                        rocblas_int* LUp,
                                        rocblas_int* LUi,
                                        T* LUx)
{
    rocblas_int tix = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    rocblas_int tiy = hipThreadIdx_y;

    // -------------------------------------------
    // If P or Q is NULL, then treat as identity permutation
    // -------------------------------------------
    if(tix < n)
    {
        rocblas_int irow = tix;
        rocblas_int istart = LUp[irow];
        rocblas_int iend = LUp[irow + 1];
        rocblas_int i, icol;

        rocblas_int irow_old = (pivP ? pivP[irow] : irow);
        rocblas_int istart_old = Ap[irow_old];
        rocblas_int iend_old = Ap[irow_old + 1];
        rocblas_int i_old, icol_old;

        // ------------------------------
        // scale A by alpha and add to B
        // ------------------------------
        for(i_old = istart_old + tiy; i_old < iend_old; i_old += hipBlockDim_y)
        {
            icol_old = Ai[i_old];
            icol = (inv_pivQ ? inv_pivQ[icol_old] : icol_old);

            i = rf_search<T>(LUi, istart, iend, icol);
            if(i != -1)
            {
                LUx[i] += alpha * Ax[i_old];
            }
        }
    }
}

template <typename T>
rocblas_status rocsolver_csrrf_refactlu_argCheck(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 const rocblas_int nnzA,
                                                 rocblas_int* ptrA,
                                                 rocblas_int* indA,
                                                 T valA,
                                                 const rocblas_int nnzT,
                                                 rocblas_int* ptrT,
                                                 rocblas_int* indT,
                                                 T valT,
                                                 rocblas_int* pivP,
                                                 rocblas_int* pivQ,
                                                 rocsolver_rfinfo rfinfo)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(n < 0 || nnzA < 0 || nnzT < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if(!rfinfo || !ptrA || !ptrT || (n && (!pivP || !pivQ)) || (nnzA && (!indA || !valA))
       || (nnzT && (!indT || !valT)))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U>
void rocsolver_csrrf_refactlu_getMemorySize(const rocblas_int n,
                                            const rocblas_int nnzT,
                                            rocblas_int* ptrT,
                                            rocblas_int* indT,
                                            U valT,
                                            rocsolver_rfinfo rfinfo,
                                            size_t* size_work)
{
    // if quick return, no need of workspace
    if(n == 0)
    {
        *size_work = 0;
        return;
    }

    // requirements for incomplete factorization
    rocsparseCall_csrilu0_buffer_size(rfinfo->sphandle, n, nnzT, rfinfo->descrT, valT, ptrT, indT,
                                      rfinfo->infoT, size_work);

    // need at least size n integers to generate inverse permutation inv_pivQ
    *size_work = std::max(*size_work, sizeof(rocblas_int) * n);
}

template <typename T, typename U>
rocblas_status rocsolver_csrrf_refactlu_template(rocblas_handle handle,
                                                 const rocblas_int n,
                                                 const rocblas_int nnzA,
                                                 rocblas_int* ptrA,
                                                 rocblas_int* indA,
                                                 U valA,
                                                 const rocblas_int nnzT,
                                                 rocblas_int* ptrT,
                                                 rocblas_int* indT,
                                                 U valT,
                                                 rocblas_int* pivP,
                                                 rocblas_int* pivQ,
                                                 rocsolver_rfinfo rfinfo,
                                                 void* work)
{
    ROCSOLVER_ENTER("csrrf_refactlu", "n:", n, "nnzA:", nnzA, "nnzT:", nnzT);

    // quick return
    if(n == 0)
        return rocblas_status_success;

    // check state of rfinfo
    if(!rfinfo->analyzed || rfinfo->analyzed_mode != rocsolver_rfinfo_mode_lu
       || rfinfo->mode != rocsolver_rfinfo_mode_lu)
        return rocblas_status_internal_error;

    hipStream_t stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &stream));

    rocblas_int nblocks = (n - 1) / BS2 + 1;
    ROCSOLVER_LAUNCH_KERNEL(rf_ipvec_kernel<T>, dim3(nblocks), dim3(BS2), 0, stream, n, pivQ,
                            (rocblas_int*)work);

    // set T to zero
    HIP_CHECK(hipMemsetAsync((void*)valT, 0, sizeof(T) * nnzT, stream));

    // ---------------------------------------------------------------------
    // copy P*A*Q into T
    // Note: the sparsity pattern of A is a subset of T, and since the re-orderings
    // P and Q are applied, the incomplete factorization of P*A*Q (factorization without fill-in),
    // yields the complete factorization of A.
    // ---------------------------------------------------------------------
    T const alpha = static_cast<T>(1);
    ROCSOLVER_LAUNCH_KERNEL(rf_add_PAQ_kernel<T>, dim3(nblocks, 1), dim3(BS2, BS2), 0, stream, n,
                            pivP, (rocblas_int*)work, alpha, ptrA, indA, valA, ptrT, indT, valT);

    // perform incomplete factorization of T
    ROCSPARSE_CHECK(rocsparseCall_csrilu0(rfinfo->sphandle, n, nnzT, rfinfo->descrT, valT, ptrT,
                                          indT, rfinfo->infoT, rocsparse_solve_policy_auto, work));

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

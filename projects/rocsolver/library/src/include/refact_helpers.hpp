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

ROCSOLVER_BEGIN_NAMESPACE

// -------------------------------------------------
// function to perform search in array
// -------------------------------------------------
// search array ind[istart], ..., ind[iend-1]
// for matching value "key"
//
// return the index value of matching position
// ---------------------------------------
template <typename T>
__device__ rocblas_int rf_search(rocblas_int* ind, rocblas_int istart, rocblas_int iend, rocblas_int key)
{
    // -----------------
    // use binary search
    // -----------------
    rocblas_int const small_size = 8;
    while(iend - istart > small_size)
    {
        rocblas_int imid = istart + (iend - istart) / 2;
        rocblas_int curr = ind[imid];

        if(curr == key)
            return imid;
        else if(curr > key)
            iend = imid;
        else
            istart = imid + 1;
    }

    // ------------------------
    // use simple linear search
    // ------------------------
    for(rocblas_int imid = istart; imid < iend; imid++)
    {
        if(ind[imid] == key)
            return imid;
    }

    return -1;
}

// ------------------------------------------------------------
// Compute the inverse permutation inv_pivQ[] from pivQ
// ------------------------------------------------------------
template <typename T>
ROCSOLVER_KERNEL void rf_ipvec_kernel(rocblas_int n, rocblas_int* pivQ, rocblas_int* inv_pivQ)
{
    rocblas_int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(tid < n)
    {
        rocblas_int iold = pivQ[tid];
        inv_pivQ[iold] = tid;
    }
}

ROCSOLVER_END_NAMESPACE

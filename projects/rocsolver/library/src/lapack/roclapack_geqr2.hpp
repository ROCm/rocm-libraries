/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.9.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2019
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "auxiliary/rocauxiliary_lacgv.hpp"
#include "auxiliary/rocauxiliary_larf.hpp"
#include "auxiliary/rocauxiliary_larfg.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

template <int MAX_THDS, typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(MAX_THDS) geqr2_kernel_small(const I m,
                                                                     const I n,
                                                                     U AA,
                                                                     const rocblas_stride shiftA,
                                                                     const I lda,
                                                                     const rocblas_stride strideA,
                                                                     S* diagA,
                                                                     const rocblas_stride strideD,
                                                                     T* tauA,
                                                                     const rocblas_stride strideP)
{
    I bid = blockIdx.z;
    I tid = threadIdx.x;

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    S* diag = load_ptr_batch<S>(diagA, bid, 0, strideD);
    T* tau = load_ptr_batch<T>(tauA, bid, 0, strideP);

    // shared variables
    extern __shared__ double lmem[];
    T* a = reinterpret_cast<T*>(lmem);
    T* w = reinterpret_cast<T*>(a + m * n);
    T* tmptau = reinterpret_cast<T*>(w + n);
    T* sval = reinterpret_cast<T*>(tmptau + 1);

    T* x;

    I dim = std::min(m, n); // total number of pivots

    // load A to lds
    for(I i = tid % (MAX_THDS / 2); i < m; i += (MAX_THDS / 2))
    {
        const auto tidy = tid / (MAX_THDS / 2);
        for(I j = tidy; j < n; j += 2)
        {
            a[i + j * m] = A[i + j * lda];
        }
    }

    __syncthreads();

    // main loop running forwards (for each column)
    for(rocblas_int j = 0; j < dim; ++j)
    {
        I mm = m - j;
        I nn = n - j;

        // ----- 1. generate Householder reflector to annihilate A(j+1:m-1,j) -----
        // load A(j:m-1,j) into x
        x = a + j + j * m;

        // larfg
        T norm2 = 0;
        for(I i = tid; i < mm - 1; i += MAX_THDS)
            norm2 += x[i + 1] * conj(x[i + 1]);

        // reduce squared entries to find squared norm of x
        norm2 += shift_left(norm2, 1);
        norm2 += shift_left(norm2, 2);
        norm2 += shift_left(norm2, 4);
        norm2 += shift_left(norm2, 8);
        norm2 += shift_left(norm2, 16);
        if(warpSize > 32)
            norm2 += shift_left(norm2, 32);
        if(tid % warpSize == 0)
            sval[tid / warpSize] = norm2;
        __syncthreads();
        if(tid == 0)
        {
            for(I k = 1; k < MAX_THDS / warpSize; k++)
                norm2 += sval[k];

            // set tau, beta, and put scaling factor into sval[0]
            run_set_taubeta<T>(tmptau, &norm2, x, diag + j);

            tau[j] = tmptau[0];
            sval[0] = norm2;

            tmptau[0] = conj(tmptau[0]);
        }
        __syncthreads();

        // scale x by scaling factor
        for(I i = tid; i < mm - 1; i += MAX_THDS)
            x[i + 1] *= sval[0];
        __syncthreads();

        // ----- 2. compute w = tau'*v'*A -----
        // gemv
        for(I i = tid; i < nn - 1; i += MAX_THDS)
        {
            T temp = 0;
            T* Atmp = a + j + j * m;
            for(I jj = 0; jj < mm; jj++)
                temp += Atmp[jj + (i + 1) * m] * conj(x[jj]);
            w[i] = tmptau[0] * temp;
        }
        __syncthreads();

        // ----- 3. apply the Householder reflector to A as a rank-1 update: A = A - v*w -----
        // ger
        for(I i = tid; i < mm; i += MAX_THDS)
        {
            T* Atmp = a + j + j * m;
            for(I jj = 0; jj < nn - 1; jj++)
            {
                Atmp[i + (jj + 1) * m] = Atmp[i + (jj + 1) * m] - x[i] * w[jj];
            }
        }
        __syncthreads();
    }

    // write lds back to A
    for(I i = tid % (MAX_THDS / 2); i < m; i += (MAX_THDS / 2))
    {
        const auto tidy = tid / (MAX_THDS / 2);
        for(I j = tidy; j < n; j += 2)
        {
            A[i + j * lda] = a[i + j * m];
        }
    }
}

template <bool BATCHED, typename T, typename I>
void rocsolver_geqr2_getMemorySize(const I m,
                                   const I n,
                                   const I batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work_workArr,
                                   size_t* size_Abyx_norms,
                                   size_t* size_diag)
{
    using S = decltype(std::real(T{}));

    // if quick return no workspace needed
    if(m == 0 || n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work_workArr = 0;
        *size_Abyx_norms = 0;
        *size_diag = 0;
        return;
    }

    // size of Abyx_norms is maximum of what is needed by larf and larfg
    // size_work_workArr is maximum of re-usable work space and array of pointers to workspace
    size_t s1, s2, w1, w2;
    rocsolver_larf_getMemorySize<BATCHED, T>(rocblas_side_left, m, n, batch_count, size_scalars,
                                             &s1, &w1);
    rocsolver_larfg_getMemorySize<T>(m, batch_count, &w2, &s2);
    *size_work_workArr = std::max(w1, w2);
    *size_Abyx_norms = std::max(s1, s2);

    // size of array to store temporary diagonal values
    *size_diag = sizeof(S) * std::min(m, n) * batch_count;
}

template <typename T, typename I, typename U>
rocblas_status rocsolver_geqr2_geqrf_argCheck(rocblas_handle handle,
                                              const I m,
                                              const I n,
                                              const I lda,
                                              T A,
                                              U ipiv,
                                              const I batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(m < 0 || n < 0 || lda < m || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((m && n && !A) || (m && n && !ipiv))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename I, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_geqr2_template(rocblas_handle handle,
                                        const I m,
                                        const I n,
                                        U A,
                                        const rocblas_stride shiftA,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        T* ipiv,
                                        const rocblas_stride strideP,
                                        const I batch_count,
                                        T* scalars,
                                        void* work_workArr,
                                        T* Abyx_norms,
                                        void* diag)
{
    ROCSOLVER_ENTER("geqr2", "m:", m, "n:", n, "shiftA:", shiftA, "lda:", lda, "bc:", batch_count);
    using S = decltype(std::real(T{}));

    // quick return
    if(m == 0 || n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // get device prop
    const hipDeviceProp_t* props = rocblas_internal_get_device_prop(handle);

    I dim = std::min(m, n); // total number of pivots
    for(I j = 0; j < dim; ++j)
    {
        I mm = m - j;
        I nn = n - j;

        const size_t lmemsize = ((256 / props->warpSize) + mm + nn + 1 + mm * nn) * sizeof(T);
        if(lmemsize <= props->sharedMemPerBlock && nn == mm)
        {
            ROCSOLVER_LAUNCH_KERNEL((geqr2_kernel_small<256, T>), dim3(1, 1, batch_count), dim3(256),
                                    lmemsize, stream, mm, nn, A, shiftA + idx2D(j, j, lda), lda,
                                    strideA, (S*)diag + j, dim, ipiv + j, strideP);
            break;
        }

        // generate Householder reflector to work on column j
        rocsolver_larfg_template<T>(handle, m - j, A, shiftA + idx2D(j, j, lda), (S*)diag, j, dim,
                                    A, shiftA + idx2D(std::min(j + 1, m - 1), j, lda), (I)1, strideA,
                                    (ipiv + j), strideP, batch_count, (T*)work_workArr, Abyx_norms);

        // Apply Householder reflector to the rest of matrix from the left
        if(j < n - 1)
        {
            // conjugate tau
            if(COMPLEX)
                rocsolver_lacgv_template<T>(handle, (I)1, ipiv, j, (I)1, strideP, batch_count);

            rocsolver_larf_template(handle, rocblas_side_left, m - j, n - j - 1, A,
                                    shiftA + idx2D(j, j, lda), (I)1, strideA, (ipiv + j), strideP,
                                    A, shiftA + idx2D(j, j + 1, lda), lda, strideA, batch_count,
                                    scalars, Abyx_norms, (T**)work_workArr);

            // restore tau
            if(COMPLEX)
                rocsolver_lacgv_template<T>(handle, (I)1, ipiv, j, (I)1, strideP, batch_count);
        }
    }

    // restore diagonal values of A
    constexpr int DIAG_NTHREADS = 64;
    I blocks = (dim - 1) / DIAG_NTHREADS + 1;
    ROCSOLVER_LAUNCH_KERNEL((restore_diag<T, I>), dim3(batch_count, blocks, 1),
                            dim3(1, DIAG_NTHREADS, 1), 0, stream, (S*)diag, 0, dim, A, shiftA, lda,
                            strideA, dim);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "auxiliary/rocauxiliary_larfg.hpp"
#include "auxiliary/rocauxiliary_latrd.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

template <int MAX_THDS, typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(MAX_THDS)
    sytd2_lower_kernel_small(const I n,
                             U AA,
                             const rocblas_stride shiftA,
                             const I lda,
                             const rocblas_stride strideA,
                             S* DD,
                             const rocblas_stride strideD,
                             S* EE,
                             const rocblas_stride strideE,
                             T* tauA,
                             const rocblas_stride strideP)
{
    I bid = blockIdx.z;
    I tid = threadIdx.x;

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    S* D = load_ptr_batch<S>(DD, bid, 0, strideD);
    S* E = load_ptr_batch<S>(EE, bid, 0, strideE);
    T* tau = load_ptr_batch<T>(tauA, bid, 0, strideP);

    // shared variables
    extern __shared__ double lmem[];
    T* tmptau = reinterpret_cast<T*>(lmem);
    T* a = reinterpret_cast<T*>(tmptau + 1);
    T* x = reinterpret_cast<T*>(a + n * n);
    T* w = reinterpret_cast<T*>(x + n);
    T* sval = reinterpret_cast<T*>(w + n);

    // load A to lds
    for(I i = tid % (MAX_THDS / 2); i < n; i += (MAX_THDS / 2))
    {
        const auto tidy = tid / (MAX_THDS / 2);
        for(I j = tidy; j < n; j += 2)
        {
            a[i + j * n] = A[i + j * lda];
        }
    }

    __syncthreads();

    for(I i = tid % (MAX_THDS / 2); i < n; i += (MAX_THDS / 2))
    {
        const auto tidy = tid / (MAX_THDS / 2);
        for(I j = tidy; j < n; j += 2)
        {
            // ignore imaginary part of the diagonal
            if(i == j)
                a[i + j * n] = std::real(a[i + j * n]);
            // copy lower triangle to upper triangle
            if(i < j)
                a[i + j * n] = conj(a[j + i * n]);
        }
    }

    __syncthreads();

    // reduce the lower part of A
    // main loop running forwards (for each column)
    for(rocblas_int j = 0; j < n - 1; ++j)
    {
        I nn = n - j - 1;

        // ----- 1. generate Householder reflector to annihilate A(j+2:n-1,j) and copy off-diagonal element to E[j] -----
        // load A(j+1:n-1,j) into x
        for(I i = tid; i < nn; i += MAX_THDS)
            x[i] = a[(i + j + 1) + j * n];
        __syncthreads();

        // larfg
        T norm2 = 0;
        for(I i = tid; i < nn - 1; i += MAX_THDS)
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
            run_set_taubeta<T>(tmptau, &norm2, x, E + j);

            tau[j] = tmptau[0];
            sval[0] = norm2;
        }
        __syncthreads();

        // scale x by scaling factor
        for(I i = tid; i < nn - 1; i += MAX_THDS)
            x[i + 1] *= sval[0];
        __syncthreads();

        // ----- 2. compute w = tau*A*v - 1/2*tau*tau*(v'*A*v)*v -----
        // symv
        for(I i = tid; i < nn; i += MAX_THDS)
        {
            T temp = 0;
            T* Atmp = a + (j + 1) + (j + 1) * n;
            for(I jj = 0; jj < nn; jj++)
                temp += Atmp[i + jj * n] * x[jj];
            w[i] = tmptau[0] * temp;
        }

        // copy x back to A(j+1:n-1,j)
        for(I i = tid; i < nn; i += MAX_THDS)
            a[(i + j + 1) + j * n] = x[i];
        __syncthreads();

        // dot
        norm2 = 0;
        for(I i = tid; i < nn; i += MAX_THDS)
            norm2 += x[i] * conj(w[i]);

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
            sval[0] = -0.5 * tmptau[0] * norm2;
        }
        __syncthreads();

        // axpy
        for(I i = tid; i < nn; i += MAX_THDS)
            w[i] += sval[0] * x[i];
        __syncthreads();

        // ----- 3. apply the Householder reflector to A as a rank-2 update: A = A - v*w' - w*v' -----
        // syr2
        for(I i = tid; i < nn; i += MAX_THDS)
        {
            for(I jj = 0; jj < nn; jj++)
            {
                T* Atmp = a + (j + 1) + (j + 1) * n;
                Atmp[i + jj * n] = Atmp[i + jj * n] - x[i] * conj(w[jj]) - w[i] * conj(x[jj]);
            }
        }
        __syncthreads();
    }

    // write lds back to A
    for(I i = tid % (MAX_THDS / 2); i < n; i += (MAX_THDS / 2))
    {
        const auto tidy = tid / (MAX_THDS / 2);
        for(I j = tidy; j < n; j += 2)
        {
            if(i >= j)
                A[i + j * lda] = a[i + j * n];
        }
    }
}

template <int MAX_THDS, typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(MAX_THDS)
    sytd2_upper_kernel_small(const I n,
                             U AA,
                             const rocblas_stride shiftA,
                             const I lda,
                             const rocblas_stride strideA,
                             S* DD,
                             const rocblas_stride strideD,
                             S* EE,
                             const rocblas_stride strideE,
                             T* tauA,
                             const rocblas_stride strideP)
{
    I bid = blockIdx.z;
    I tid = threadIdx.x;

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    S* D = load_ptr_batch<S>(DD, bid, 0, strideD);
    S* E = load_ptr_batch<S>(EE, bid, 0, strideE);
    T* tau = load_ptr_batch<T>(tauA, bid, 0, strideP);

    // shared variables
    extern __shared__ double lmem[];
    T* tmptau = reinterpret_cast<T*>(lmem);
    T* a = reinterpret_cast<T*>(tmptau + 1);
    T* x = reinterpret_cast<T*>(a + n * n);
    T* w = reinterpret_cast<T*>(x + n);
    T* sval = reinterpret_cast<T*>(w + n);

    // load A to lds
    for(I i = tid % (MAX_THDS / 2); i < n; i += (MAX_THDS / 2))
    {
        const auto tidy = tid / (MAX_THDS / 2);
        for(I j = tidy; j < n; j += 2)
        {
            a[i + j * n] = A[i + j * lda];
        }
    }

    __syncthreads();

    for(I i = tid % (MAX_THDS / 2); i < n; i += (MAX_THDS / 2))
    {
        const auto tidy = tid / (MAX_THDS / 2);
        for(I j = tidy; j < n; j += 2)
        {
            // ignore imaginary part of the diagonal
            if(i == j)
                a[i + j * n] = std::real(a[i + j * n]);
            // copy upper triangle to lower triangle
            if(i > j)
                a[i + j * n] = conj(a[j + i * n]);
        }
    }

    __syncthreads();

    // reduce the upper part of A
    // main loop running backwards (for each column)
    for(rocblas_int j = n - 1; j > 0; --j)
    {
        I nn = j;

        // ----- 1. generate Householder reflector to annihilate A(0:j-2,j) and copy off-diagonal element to E[j-1] -----
        // load A(0:j-1,j) into x
        for(I i = tid; i < nn; i += MAX_THDS)
            x[i] = a[i + j * n];
        __syncthreads();

        // larfg
        T norm2 = 0;
        for(I i = tid; i < nn - 1; i += MAX_THDS)
            norm2 += x[i] * conj(x[i]);

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
            run_set_taubeta<T>(tmptau, &norm2, x + (nn - 1), E + (j - 1));

            tau[j - 1] = tmptau[0];
            sval[0] = norm2;
        }
        __syncthreads();

        // scale x by scaling factor
        for(I i = tid; i < nn - 1; i += MAX_THDS)
            x[i] *= sval[0];
        __syncthreads();

        // ----- 2. compute w = tau*A*v - 1/2*tau*tau*(v'*A*v*)v -----
        // symv
        for(I i = tid; i < nn; i += MAX_THDS)
        {
            T temp = 0;
            for(I jj = 0; jj < nn; jj++)
                temp += a[i + jj * n] * x[jj];
            w[i] = tmptau[0] * temp;
        }

        // copy x back to A(0:j-1,j)
        for(I i = tid; i < nn; i += MAX_THDS)
            a[i + j * n] = x[i];
        __syncthreads();

        // dot
        norm2 = 0;
        for(I i = tid; i < nn; i += MAX_THDS)
            norm2 += x[i] * conj(w[i]);

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
            sval[0] = -0.5 * tmptau[0] * norm2;
        }
        __syncthreads();

        // axpy
        for(I i = tid; i < nn; i += MAX_THDS)
            w[i] += sval[0] * x[i];
        __syncthreads();

        // ----- 3. apply the Householder reflector to A as a rank-2 update: A = A - v*w' - w*v' -----
        // syr2
        for(I i = tid; i < nn; i += MAX_THDS)
        {
            for(I jj = 0; jj < nn; jj++)
                a[i + jj * n] = a[i + jj * n] - x[i] * conj(w[jj]) - w[i] * conj(x[jj]);
        }
        __syncthreads();
    }

    // write lds back to A
    for(I i = tid % (MAX_THDS / 2); i < n; i += (MAX_THDS / 2))
    {
        const auto tidy = tid / (MAX_THDS / 2);
        for(I j = tidy; j < n; j += 2)
        {
            if(i <= j)
                A[i + j * lda] = a[i + j * n];
        }
    }
}

/** set_tau kernel copies to tau the corresponding Householder scalars **/
template <typename T>
ROCSOLVER_KERNEL void
    set_tau(const rocblas_int batch_count, T* tmptau, T* tau, const rocblas_stride strideP)
{
    rocblas_int b = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(b < batch_count)
    {
        T* t = tau + b * strideP;

        t[0] = tmptau[b];
    }
}

/** set_tridiag kernel copies results to set tridiagonal form in A, diagonal elements in D
    and off-diagonal elements in E **/
template <typename T, typename S, typename U, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
ROCSOLVER_KERNEL void set_tridiag(const rocblas_fill uplo,
                                  const rocblas_int n,
                                  U A,
                                  const rocblas_int shiftA,
                                  const rocblas_int lda,
                                  const rocblas_stride strideA,
                                  S* D,
                                  const rocblas_stride strideD,
                                  S* E,
                                  const rocblas_stride strideE)
{
    rocblas_int b = hipBlockIdx_y;
    rocblas_int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    bool lower = (uplo == rocblas_fill_lower);

    if(i < n)
    {
        T* a = load_ptr_batch<T>(A, b, shiftA, strideA);
        S* d = D + b * strideD;
        S* e = E + b * strideE;

        // diagonal
        d[i] = a[i + i * lda];

        // off-diagonal
        if(i < n - 1)
        {
            if(lower)
                a[(i + 1) + i * lda] = T(e[i]);
            else
                a[i + (i + 1) * lda] = T(e[i]);
        }
    }
}

template <typename T, typename S, typename U, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
ROCSOLVER_KERNEL void set_tridiag(const rocblas_fill uplo,
                                  const rocblas_int n,
                                  U A,
                                  const rocblas_int shiftA,
                                  const rocblas_int lda,
                                  const rocblas_stride strideA,
                                  S* D,
                                  const rocblas_stride strideD,
                                  S* E,
                                  const rocblas_stride strideE)
{
    rocblas_int b = hipBlockIdx_y;
    rocblas_int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    bool lower = (uplo == rocblas_fill_lower);
    S tmp;

    if(i < n)
    {
        T* a = load_ptr_batch<T>(A, b, shiftA, strideA);
        S* d = D + b * strideD;
        S* e = E + b * strideE;

        // diagonal
        /* (Hermitian matrices have real terms in the diagonal. In the case where the input
            matrix is not Hermitian, we are simply ignoring the imaginary part so that outputs
            in A and E coincide with the outputs generated by other similar libraries.
            -- Note that the resulting tridiagonal form is not really similar to the original A;
            they will not have the same eigenvalues) */
        tmp = a[i + i * lda].real();
        d[i] = tmp;
        a[i + i * lda] = T(tmp);

        // off-diagonal
        if(i < n - 1)
        {
            if(lower)
                a[(i + 1) + i * lda] = T(e[i]);
            else
                a[i + (i + 1) * lda] = T(e[i]);
        }
    }
}

template <bool BATCHED, typename T>
void rocsolver_sytd2_hetd2_getMemorySize(const rocblas_int n,
                                         const rocblas_int batch_count,
                                         size_t* size_scalars,
                                         size_t* size_work,
                                         size_t* size_norms,
                                         size_t* size_tmptau,
                                         size_t* size_workArr)
{
    // if quick return no workspace needed
    if(n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work = 0;
        *size_norms = 0;
        *size_tmptau = 0;
        *size_workArr = 0;
        return;
    }

    size_t n1 = 0, n2 = 0;
    size_t w1 = 0, w2 = 0, w3 = 0;

    // size of scalars (constants)
    *size_scalars = sizeof(T) * 3;

    // size of array to store temporary householder scalars
    *size_tmptau = sizeof(T) * batch_count;

    // size of array of pointers to workspace
    if(BATCHED)
        *size_workArr = sizeof(T*) * batch_count;
    else
        *size_workArr = 0;

    // extra requirements to call larfg
    rocsolver_larfg_getMemorySize<T>(n, batch_count, &w1, &n1);

    // extra requirements for calling symv/hemv
    rocblasCall_symv_hemv_mem<BATCHED, T>(n, batch_count, &w2);

    // size of re-usable workspace
    // TODO: replace with rocBLAS call
    constexpr int ROCBLAS_DOT_NB = 512;
    w3 = n > 2 ? (n - 2) / ROCBLAS_DOT_NB + 2 : 1;
    w3 *= sizeof(T) * batch_count;
    n2 = sizeof(T) * batch_count;

    *size_norms = std::max(n1, n2);
    *size_work = std::max({w1, w2, w3});
}

template <typename T, typename S, typename U>
rocblas_status rocsolver_sytd2_hetd2_argCheck(rocblas_handle handle,
                                              const rocblas_fill uplo,
                                              const rocblas_int n,
                                              const rocblas_int lda,
                                              T A,
                                              S D,
                                              S E,
                                              U tau,
                                              const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !D) || (n > 1 && !E) || (n > 1 && !tau))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename S, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_sytd2_hetd2_template(rocblas_handle handle,
                                              const rocblas_fill uplo,
                                              const rocblas_int n,
                                              U A,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              S* D,
                                              const rocblas_stride strideD,
                                              S* E,
                                              const rocblas_stride strideE,
                                              T* tau,
                                              const rocblas_stride strideP,
                                              const rocblas_int batch_count,
                                              T* scalars,
                                              T* work,
                                              T* norms,
                                              T* tmptau,
                                              T** workArr)
{
    ROCSOLVER_ENTER("sytd2_hetd2", "uplo:", uplo, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    // quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // everything must be executed with scalars on the device
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

    // configure kernels
    rocblas_int blocks = (n - 1) / BS1 + 1;
    dim3 grid_n(blocks, batch_count);
    dim3 threads(BS1, 1, 1);
    blocks = (batch_count - 1) / BS1 + 1;
    dim3 grid_b(blocks, 1);

    rocblas_stride stridet = 1; //stride for tmptau

    // get device prop
    const hipDeviceProp_t* props = rocblas_internal_get_device_prop(handle);

    if(uplo == rocblas_fill_lower)
    {
        // reduce the lower part of A
        // main loop running forwards (for each column)
        for(rocblas_int j = 0; j < n - 1; ++j)
        {
            const rocblas_int nn = n - j;
            const size_t lmemsize = ((256 / props->warpSize) + 2 * nn + 1 + nn * nn) * sizeof(T);
            if(lmemsize <= props->sharedMemPerBlock && nn <= xxTD2_SSKER_MAX_N)
            {
                ROCSOLVER_LAUNCH_KERNEL((sytd2_lower_kernel_small<256, T>), dim3(1, 1, batch_count),
                                        dim3(256), lmemsize, stream, nn, A,
                                        shiftA + idx2D(j, j, lda), lda, strideA, D + j, strideD,
                                        E + j, strideE, tau + j, strideP);

                break;
            }

            // 1. generate Householder reflector to annihilate A(j+2:n-1,j) and copy off-diagonal element to E[j]
            rocsolver_larfg_template<T>(handle, n - 1 - j, A, shiftA + idx2D(j + 1, j, lda), E, j,
                                        strideE, A, shiftA + idx2D(std::min(j + 2, n - 1), j, lda),
                                        1, strideA, tmptau, stridet, batch_count, work, norms);

            // 2. overwrite tau with w = tmptau*A*v - 1/2*tmptau*(tmptau*v'*A*v)*v
            rocblasCall_symv_hemv<T>(handle, uplo, n - 1 - j, tmptau, stridet, A,
                                     shiftA + idx2D(j + 1, j + 1, lda), lda, strideA, A,
                                     shiftA + idx2D(j + 1, j, lda), 1, strideA, scalars + 1, 0, tau,
                                     j, 1, strideP, batch_count, work, workArr);

            ROCSOLVER_LAUNCH_KERNEL((latrd_dot_scale_axpy<64, T>), dim3(1, 1, batch_count),
                                    dim3(64, 1, 1), 0, stream, n - 1 - j, A,
                                    shiftA + idx2D(j + 1, j, lda), strideA, tau, j, strideP, tmptau,
                                    stridet);

            // 3. apply the Householder reflector to A as a rank-2 update:
            // A = A - v*w' - w*v'
            rocblasCall_syr2_her2<T>(handle, uplo, n - 1 - j, scalars, A,
                                     shiftA + idx2D(j + 1, j, lda), 1, strideA, tau, j, 1, strideP,
                                     A, shiftA + idx2D(j + 1, j + 1, lda), lda, strideA,
                                     batch_count, workArr);

            // 4. Save the used householder scalar
            ROCSOLVER_LAUNCH_KERNEL(set_tau<T>, grid_b, threads, 0, stream, batch_count, tmptau,
                                    tau + j, strideP);
        }
    }

    else
    {
        // reduce the upper part of A
        // main loop running backwards (for each column)
        for(rocblas_int j = n - 1; j > 0; --j)
        {
            const rocblas_int nn = j + 1;
            const size_t lmemsize = ((256 / props->warpSize) + 2 * nn + 1 + nn * nn) * sizeof(T);
            if(lmemsize <= props->sharedMemPerBlock && nn <= xxTD2_SSKER_MAX_N)
            {
                ROCSOLVER_LAUNCH_KERNEL((sytd2_upper_kernel_small<256, T>), dim3(1, 1, batch_count),
                                        dim3(256), lmemsize, stream, nn, A, shiftA, lda, strideA, D,
                                        strideD, E, strideE, tau, strideP);
                break;
            }

            // 1. generate Householder reflector to annihilate A(0:j-2,j) and copy off-diagonal element to E[j-1]
            rocsolver_larfg_template<T>(handle, j, A, shiftA + idx2D(j - 1, j, lda), E, j - 1,
                                        strideE, A, shiftA + idx2D(0, j, lda), 1, strideA, tmptau,
                                        1, batch_count, work, norms);

            // 2. overwrite tau with w = tmptau*A*v - 1/2*tmptau*tmptau*(v'*A*v*)v
            rocblasCall_symv_hemv<T>(handle, uplo, j, tmptau, stridet, A, shiftA, lda, strideA, A,
                                     shiftA + idx2D(0, j, lda), 1, strideA, scalars + 1, 0, tau, 0,
                                     1, strideP, batch_count, work, workArr);

            ROCSOLVER_LAUNCH_KERNEL((latrd_dot_scale_axpy<64, T>), dim3(1, 1, batch_count),
                                    dim3(64, 1, 1), 0, stream, j, A, shiftA + idx2D(0, j, lda),
                                    strideA, tau, 0, strideP, tmptau, stridet);

            // 3. apply the Householder reflector to A as a rank-2 update:
            // A = A - v*w' - w*v'
            rocblasCall_syr2_her2<T>(handle, uplo, j, scalars, A, shiftA + idx2D(0, j, lda), 1,
                                     strideA, tau, 0, 1, strideP, A, shiftA, lda, strideA,
                                     batch_count, workArr);

            // 4. Save the used householder scalar
            ROCSOLVER_LAUNCH_KERNEL(set_tau<T>, grid_b, threads, 0, stream, batch_count, tmptau,
                                    tau + j - 1, strideP);
        }
    }

    // Copy results (set tridiagonal form in A)
    ROCSOLVER_LAUNCH_KERNEL(set_tridiag<T>, grid_n, threads, 0, stream, uplo, n, A, shiftA, lda,
                            strideA, D, strideD, E, strideE);

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

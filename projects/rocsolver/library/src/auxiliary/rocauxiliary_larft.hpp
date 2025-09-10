/****************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * and
 *     Joffrain, Low, Quintana-Orti, et al. (2006). Accumulating householder
 *     transformations, revisited.
 *     ACM Transactions on Mathematical Software 32(2), p. 169-179.
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

#include "rocauxiliary_lacgv.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/*************** Main kernels *********************************************************/
/**************************************************************************************/

template <typename T, typename U, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
ROCSOLVER_KERNEL void set_triangular(const rocblas_int n,
                                     const rocblas_int k,
                                     U V,
                                     const rocblas_int shiftV,
                                     const rocblas_int ldv,
                                     const rocblas_stride strideV,
                                     T* tau,
                                     const rocblas_stride strideT,
                                     T* F,
                                     const rocblas_int ldf,
                                     const rocblas_stride strideF,
                                     const rocblas_direct direct,
                                     const rocblas_storev storev,
                                     const bool add_fp)
{
    const auto b = hipBlockIdx_z;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(i < k && j < k)
    {
        T *tp, *Vp, *Fp;
        tp = tau + b * strideT;
        Vp = load_ptr_batch<T>(V, b, shiftV, strideV);
        Fp = F + b * strideF;

        if(j == i)
            Fp[idx2D(j, i, ldf)] = tp[i];
        else if(direct == rocblas_forward_direction)
        {
            if(j < i)
            {
                if(storev == rocblas_column_wise)
                {
                    if(!add_fp)
                    {
                        Fp[idx2D(j, i, ldf)] = -tp[i] * Vp[idx2D(i, j, ldv)];
                    }
                    else
                    {
                        Fp[idx2D(j, i, ldf)] = -tp[i] * (Fp[idx2D(j, i, ldf)] + Vp[idx2D(i, j, ldv)]);
                    }
                }
                else
                {
                    if(!add_fp)
                    {
                        Fp[idx2D(j, i, ldf)] = -tp[i] * Vp[idx2D(j, i, ldv)];
                    }
                    else
                    {
                        Fp[idx2D(j, i, ldf)] = -tp[i] * (Fp[idx2D(j, i, ldf)] + Vp[idx2D(j, i, ldv)]);
                    }
                }
            }
            else
                Fp[idx2D(j, i, ldf)] = 0;
        }
        else
        {
            if(j > i)
            {
                if(storev == rocblas_column_wise)
                {
                    if(!add_fp)
                    {
                        Fp[idx2D(j, i, ldf)] = -tp[i] * Vp[idx2D((n - k + i), j, ldv)];
                    }
                    else
                    {
                        Fp[idx2D(j, i, ldf)]
                            = -tp[i] * (Fp[idx2D(j, i, ldf)] + Vp[idx2D((n - k + i), j, ldv)]);
                    }
                }
                else
                {
                    if(!add_fp)
                    {
                        Fp[idx2D(j, i, ldf)] = -tp[i] * Vp[idx2D(j, (n - k + i), ldv)];
                    }
                    else
                    {
                        Fp[idx2D(j, i, ldf)]
                            = -tp[i] * (Fp[idx2D(j, i, ldf)] + Vp[idx2D(j, (n - k + i), ldv)]);
                    }
                }
            }
            else
                Fp[idx2D(j, i, ldf)] = 0;
        }
    }
}

template <typename T, typename U, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
ROCSOLVER_KERNEL void set_triangular(const rocblas_int n,
                                     const rocblas_int k,
                                     U V,
                                     const rocblas_int shiftV,
                                     const rocblas_int ldv,
                                     const rocblas_stride strideV,
                                     T* tau,
                                     const rocblas_stride strideT,
                                     T* F,
                                     const rocblas_int ldf,
                                     const rocblas_stride strideF,
                                     const rocblas_direct direct,
                                     const rocblas_storev storev,
                                     const bool add_fp)
{
    const auto b = hipBlockIdx_z;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(i < k && j < k)
    {
        T *tp, *Vp, *Fp;
        tp = tau + b * strideT;
        Vp = load_ptr_batch<T>(V, b, shiftV, strideV);
        Fp = F + b * strideF;

        if(j == i)
            Fp[idx2D(j, i, ldf)] = tp[i];
        else if(direct == rocblas_forward_direction)
        {
            if(j < i)
            {
                if(storev == rocblas_column_wise)
                {
                    if(!add_fp)
                    {
                        Fp[idx2D(j, i, ldf)] = -tp[i] * conj(Vp[idx2D(i, j, ldv)]);
                    }
                    else
                    {
                        Fp[idx2D(j, i, ldf)]
                            = -tp[i] * (Fp[idx2D(j, i, ldf)] + conj(Vp[idx2D(i, j, ldv)]));
                    }
                }
                else
                {
                    if(!add_fp)
                    {
                        Fp[idx2D(j, i, ldf)] = -tp[i] * Vp[idx2D(j, i, ldv)];
                    }
                    else
                    {
                        Fp[idx2D(j, i, ldf)] = -tp[i] * (Fp[idx2D(j, i, ldf)] + Vp[idx2D(j, i, ldv)]);
                    }
                }
            }
            else
                Fp[idx2D(j, i, ldf)] = 0;
        }
        else
        {
            if(j > i)
            {
                if(storev == rocblas_column_wise)
                {
                    if(!add_fp)
                    {
                        Fp[idx2D(j, i, ldf)] = -tp[i] * conj(Vp[idx2D((n - k + i), j, ldv)]);
                    }
                    else
                    {
                        Fp[idx2D(j, i, ldf)]
                            = -tp[i] * (Fp[idx2D(j, i, ldf)] + conj(Vp[idx2D((n - k + i), j, ldv)]));
                    }
                }
                else
                {
                    if(!add_fp)
                    {
                        Fp[idx2D(j, i, ldf)] = -tp[i] * Vp[idx2D(j, (n - k + i), ldv)];
                    }
                    else
                    {
                        Fp[idx2D(j, i, ldf)]
                            = -tp[i] * (Fp[idx2D(j, i, ldf)] + Vp[idx2D(j, (n - k + i), ldv)]);
                    }
                }
            }
            else
                Fp[idx2D(j, i, ldf)] = 0;
        }
    }
}

template <typename T>
ROCSOLVER_KERNEL void set_tau(const rocblas_int k, T* tau, const rocblas_stride strideT)
{
    const auto b = hipBlockIdx_y;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(i < k)
    {
        T* tp = tau + b * strideT;
        tp[i] = -tp[i];
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void larft_kernel_forward(const rocblas_storev storev,
                                           const rocblas_int n,
                                           const rocblas_int k,
                                           U VA,
                                           const rocblas_int shiftV,
                                           const rocblas_int ldv,
                                           const rocblas_stride strideV,
                                           T* tauA,
                                           const rocblas_stride strideT,
                                           T* FA,
                                           const rocblas_int ldfA,
                                           const rocblas_stride strideF)
{
    const rocblas_int bid = hipBlockIdx_y;
    const rocblas_int tid = hipThreadIdx_x;
    const rocblas_int tid_inc = hipBlockDim_x;

    // select batch instance
    T* V = load_ptr_batch<T>(VA, bid, shiftV, strideV);
    T* tau = tauA + bid * strideT;
    T* Ftemp = FA + bid * strideF;

    // shared memory setup
    extern __shared__ double lmem[];
    T* work = reinterpret_cast<T*>(lmem);
    T* F = work + k;
    rocblas_int ldf = k;

    // copy F to shared memory
    for(rocblas_int i = tid; i < k; i += tid_inc)
        for(rocblas_int j = i; j < k; j++)
            F[i + j * ldf] = Ftemp[i + j * ldfA];
    __syncthreads();

    // --------- MAIN BODY ---------
    for(rocblas_int kk = 1; kk < k; kk++)
    {
        const rocblas_int mm = kk;
        const rocblas_int nn = n - 1 - kk;

        T* Fx = F + kk * ldf;

        // compute the matrix vector product, using the householder vectors
        if(storev == rocblas_column_wise)
        {
            T* Vm = V + (kk + 1);
            T* Vx = V + (kk + 1) + kk * ldv;

            // gemv (conjugate transpose)
            for(rocblas_int i = tid; i < mm; i += tid_inc)
            {
                T temp = 0;
                for(rocblas_int j = 0; j < nn; j++)
                    temp += conj(Vm[j + i * ldv]) * Vx[j];
                work[i] = tau[kk] * temp + Fx[i];
            }
        }
        else
        {
            T* Vm = V + (kk + 1) * ldv;
            T* Vx = V + kk + (kk + 1) * ldv;

            // gemv (no transpose)
            for(rocblas_int i = tid; i < mm; i += tid_inc)
            {
                T temp = 0;
                for(rocblas_int j = 0; j < nn; j++)
                    temp += Vm[i + j * ldv] * conj(Vx[j * ldv]);
                work[i] = tau[kk] * temp + Fx[i];
            }
        }

        __syncthreads();

        // multiply by previous triangular factor
        // trmv (no transpose)
        for(rocblas_int i = tid; i < mm; i += tid_inc)
        {
            T temp = 0;
            for(rocblas_int j = i; j < mm; j++)
                temp += F[i + j * ldf] * work[j];
            Fx[i] = temp;
        }

        __syncthreads();
    }

    // copy shared memory back to F
    for(rocblas_int i = tid; i < k; i += tid_inc)
        for(rocblas_int j = i; j < k; j++)
            Ftemp[i + j * ldfA] = F[i + j * ldf];
}

template <typename T, typename U>
ROCSOLVER_KERNEL void larft_kernel_backward(const rocblas_storev storev,
                                            const rocblas_int n,
                                            const rocblas_int k,
                                            U VA,
                                            const rocblas_int shiftV,
                                            const rocblas_int ldv,
                                            const rocblas_stride strideV,
                                            T* tauA,
                                            const rocblas_stride strideT,
                                            T* FA,
                                            const rocblas_int ldfA,
                                            const rocblas_stride strideF)
{
    const rocblas_int bid = hipBlockIdx_y;
    const rocblas_int tid = hipThreadIdx_x;
    const rocblas_int tid_inc = hipBlockDim_x;

    // select batch instance
    T* V = load_ptr_batch<T>(VA, bid, shiftV, strideV);
    T* tau = tauA + bid * strideT;
    T* Ftemp = FA + bid * strideF;

    // shared memory setup
    extern __shared__ double lmem[];
    T* work = reinterpret_cast<T*>(lmem);
    T* F = work + k;
    rocblas_int ldf = k;

    // copy F to shared memory
    for(rocblas_int i = tid; i < k; i += tid_inc)
        for(rocblas_int j = 0; j <= i; j++)
            F[i + j * ldf] = Ftemp[i + j * ldfA];
    __syncthreads();

    // --------- MAIN BODY ---------
    for(rocblas_int kk = k - 2; kk >= 0; kk--)
    {
        const rocblas_int mm = k - kk - 1;
        const rocblas_int nn = n - k + kk;

        T* Fm = F + (kk + 1) + (kk + 1) * ldf;
        T* Fx = F + (kk + 1) + kk * ldf;

        // compute the matrix vector product, using the householder vectors
        if(storev == rocblas_column_wise)
        {
            T* Vm = V + (kk + 1) * ldv;
            T* Vx = V + kk * ldv;

            // gemv (conjugate transpose)
            for(rocblas_int i = tid; i < mm; i += tid_inc)
            {
                T temp = 0;
                for(rocblas_int j = 0; j < nn; j++)
                    temp += conj(Vm[j + i * ldv]) * Vx[j];
                work[i] = tau[kk] * temp + Fx[i];
            }
        }
        else
        {
            T* Vm = V + (kk + 1);
            T* Vx = V + kk;

            // gemv (no transpose)
            for(rocblas_int i = tid; i < mm; i += tid_inc)
            {
                T temp = 0;
                for(rocblas_int j = 0; j < nn; j++)
                    temp += Vm[i + j * ldv] * conj(Vx[j * ldv]);
                work[i] = tau[kk] * temp + Fx[i];
            }
        }

        __syncthreads();

        // multiply by previous triangular factor
        // trmv (no transpose)
        for(rocblas_int i = tid; i < mm; i += tid_inc)
        {
            T temp = 0;
            for(rocblas_int j = 0; j <= i; j++)
                temp += Fm[i + j * ldf] * work[j];
            Fx[i] = temp;
        }

        __syncthreads();
    }

    // copy shared memory back to F
    for(rocblas_int i = tid; i < k; i += tid_inc)
        for(rocblas_int j = 0; j <= i; j++)
            Ftemp[i + j * ldfA] = F[i + j * ldf];
}

/******************* Host functions *********************************************/
/*******************************************************************************/

template <bool BATCHED, typename T>
void rocsolver_larft_getMemorySize(const rocblas_int n,
                                   const rocblas_int k,
                                   const rocblas_int batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work,
                                   size_t* size_workArr)
{
    // if quick return, no workspace is needed
    if(n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work = 0;
        *size_workArr = 0;
        return;
    }

    // size of scalars (constants)
    *size_scalars = sizeof(T) * 3;

    // size of re-usable workspace
    *size_work = sizeof(T) * k * batch_count;

    // size of array of pointers to workspace
    if(BATCHED)
        *size_workArr = sizeof(T*) * batch_count;
    else
        *size_workArr = 0;
}

template <typename T, typename U>
rocblas_status rocsolver_larft_argCheck(rocblas_handle handle,
                                        const rocblas_direct direct,
                                        const rocblas_storev storev,
                                        const rocblas_int n,
                                        const rocblas_int k,
                                        const rocblas_int ldv,
                                        const rocblas_int ldf,
                                        T V,
                                        U tau,
                                        U F)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(direct != rocblas_backward_direction && direct != rocblas_forward_direction)
        return rocblas_status_invalid_value;
    if(storev != rocblas_column_wise && storev != rocblas_row_wise)
        return rocblas_status_invalid_value;
    bool row = (storev == rocblas_row_wise);

    // 2. invalid size
    if(n < 0 || k < 1 || ldf < k)
        return rocblas_status_invalid_size;
    if((row && ldv < k) || (!row && ldv < n))
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !V) || !tau || !F)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_larft_template(rocblas_handle handle,
                                        const rocblas_direct direct,
                                        const rocblas_storev storev,
                                        const rocblas_int n,
                                        const rocblas_int k,
                                        U V,
                                        const rocblas_int shiftV,
                                        const rocblas_int ldv,
                                        const rocblas_stride strideV,
                                        T* tau,
                                        const rocblas_stride strideT,
                                        T* F,
                                        const rocblas_int ldf,
                                        const rocblas_stride strideF,
                                        const rocblas_int batch_count,
                                        T* scalars,
                                        T* work,
                                        T** workArr)
{
    ROCSOLVER_ENTER("larft", "direct:", direct, "storev:", storev, "n:", n, "k:", k,
                    "shiftV:", shiftV, "ldv:", ldv, "ldf:", ldf, "bc:", batch_count);

    // quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // save old pointer mode
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);

    rocblas_stride stridew = rocblas_stride(k);
    rocblas_diagonal diag = rocblas_diagonal_non_unit;
    rocblas_fill uplo;
    rocblas_operation trans;

    const bool use_gemm = n > k;
    const T zero = T(0);
    const T one = T(1);

    const rocblas_int u1_n = use_gemm ? k : n;
    const rocblas_int u2_n = use_gemm ? n - k : 0;

    // Compute T=V2'*V2 or V2*V2' (V'=[V1' V2'] where V1 is triangular and V is trapezoidal)
    // SYRK/HERK can be used alternatively, but GEMM is currently more performant.
    if(use_gemm)
    {
        rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        if(direct == rocblas_forward_direction && storev == rocblas_column_wise)
        {
            rocsolver_gemm(handle, rocblas_operation_conjugate_transpose, rocblas_operation_none, k,
                           k, u2_n, &one, V, shiftV + idx2D(u1_n, 0, ldv), ldv, strideV, V,
                           shiftV + idx2D(u1_n, 0, ldv), ldv, strideV, &zero, F, 0, ldf, strideF,
                           batch_count, workArr);
        }
        else if(direct == rocblas_backward_direction && storev == rocblas_column_wise)
        {
            rocsolver_gemm(handle, rocblas_operation_conjugate_transpose, rocblas_operation_none, k,
                           k, u2_n, &one, V, shiftV, ldv, strideV, V, shiftV, ldv, strideV, &zero,
                           F, 0, ldf, strideF, batch_count, workArr);
        }
        else if(direct == rocblas_forward_direction && storev == rocblas_row_wise)
        {
            rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose, k,
                           k, u2_n, &one, V, shiftV + idx2D(0, u1_n, ldv), ldv, strideV, V,
                           shiftV + idx2D(0, u1_n, ldv), ldv, strideV, &zero, F, 0, ldf, strideF,
                           batch_count, workArr);
        }
        else if(direct == rocblas_backward_direction && storev == rocblas_row_wise)
        {
            rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_conjugate_transpose, k,
                           k, u2_n, &one, V, shiftV, ldv, strideV, V, shiftV, ldv, strideV, &zero,
                           F, 0, ldf, strideF, batch_count, workArr);
        }
    }

    // Fix diagonal of T, make zero the not used triangular part,
    // setup tau (changing signs) and account for the non-stored 1's on the
    // householder vectors
    rocblas_int blocks = (k - 1) / 32 + 1;
    ROCSOLVER_LAUNCH_KERNEL(set_triangular, dim3(blocks, blocks, batch_count), dim3(32, 32), 0,
                            stream, n, k, V, shiftV, ldv, strideV, tau, strideT, F, ldf, strideF,
                            direct, storev, use_gemm);
    ROCSOLVER_LAUNCH_KERNEL(set_tau, dim3(blocks, batch_count), dim3(32, 1), 0, stream, k, tau,
                            strideT);

    const hipDeviceProp_t* props = rocblas_internal_get_device_prop(handle);
    size_t lmemsize = sizeof(T) * (k + 1) * k;

    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

    if(direct == rocblas_forward_direction)
    {
        uplo = rocblas_fill_upper;

        // **** FOR NOW, IT DOES NOT LOOK FOR TRAILING ZEROS
        //      AS THIS WOULD REQUIRE SYNCHRONIZATION WITH GPU.
        //      IT WILL WORK ON THE ENTIRE MATRIX/VECTOR REGARDLESS OF
        //      ZERO ENTRIES ****

        if(k <= LARFT_SWITCHSIZE && lmemsize <= props->sharedMemPerBlock)
        {
            ROCSOLVER_LAUNCH_KERNEL(larft_kernel_forward, dim3(1, batch_count), dim3(BS1, 1),
                                    lmemsize, stream, storev, u1_n, k, V, shiftV, ldv, strideV, tau,
                                    strideT, F, ldf, strideF);
        }
        else
        {
            for(rocblas_int i = 1; i < k; ++i)
            {
                // compute the matrix vector product, using the householder vectors
                if(storev == rocblas_column_wise)
                {
                    trans = rocblas_operation_conjugate_transpose;
                    rocblasCall_gemv<T>(handle, trans, u1_n - 1 - i, i, tau + i, strideT, V,
                                        shiftV + idx2D(i + 1, 0, ldv), ldv, strideV, V,
                                        shiftV + idx2D(i + 1, i, ldv), 1, strideV, scalars + 2, 0,
                                        F, idx2D(0, i, ldf), 1, strideF, batch_count, workArr);
                }
                else
                {
                    if(COMPLEX)
                        rocsolver_lacgv_template<T>(handle, n - i - 1, V,
                                                    shiftV + idx2D(i, i + 1, ldv), ldv, strideV,
                                                    batch_count);

                    trans = rocblas_operation_none;
                    rocblasCall_gemv<T>(handle, trans, i, u1_n - 1 - i, tau + i, strideT, V,
                                        shiftV + idx2D(0, i + 1, ldv), ldv, strideV, V,
                                        shiftV + idx2D(i, i + 1, ldv), ldv, strideV, scalars + 2, 0,
                                        F, idx2D(0, i, ldf), 1, strideF, batch_count, workArr);

                    if(COMPLEX)
                        rocsolver_lacgv_template<T>(handle, n - i - 1, V,
                                                    shiftV + idx2D(i, i + 1, ldv), ldv, strideV,
                                                    batch_count);
                }

                // multiply by the previous triangular factor
                trans = rocblas_operation_none;
                rocblasCall_trmv<T>(handle, uplo, trans, diag, i, F, 0, ldf, strideF, F,
                                    idx2D(0, i, ldf), 1, strideF, work, stridew, batch_count);
            }
        }
    }
    else
    {
        uplo = rocblas_fill_lower;

        // **** FOR NOW, IT DOES NOT LOOK FOR TRAILING ZEROS
        //      AS THIS WOULD REQUIRE SYNCHRONIZATION WITH GPU.
        //      IT WILL WORK ON THE ENTIRE MATRIX/VECTOR REGARDLESS OF
        //      ZERO ENTRIES ****

        if(k <= LARFT_SWITCHSIZE && lmemsize <= props->sharedMemPerBlock)
        {
            auto shiftU2 = shiftV
                + ((storev == rocblas_column_wise) ? idx2D(u2_n, 0, ldv) : idx2D(0, u2_n, ldv));
            ROCSOLVER_LAUNCH_KERNEL(larft_kernel_backward, dim3(1, batch_count), dim3(BS1, 1),
                                    lmemsize, stream, storev, u1_n, k, V, shiftU2, ldv, strideV,
                                    tau, strideT, F, ldf, strideF);
        }
        else
        {
            for(rocblas_int i = k - 2; i >= 0; --i)
            {
                // compute the matrix vector product, using the householder vectors
                if(storev == rocblas_column_wise)
                {
                    trans = rocblas_operation_conjugate_transpose;
                    rocblasCall_gemv<T>(handle, trans, u1_n - k + i, k - i - 1, tau + i, strideT, V,
                                        shiftV + idx2D(u2_n, i + 1, ldv), ldv, strideV, V,
                                        shiftV + idx2D(u2_n, i, ldv), 1, strideV, scalars + 2, 0, F,
                                        idx2D(i + 1, i, ldf), 1, strideF, batch_count, workArr);
                }
                else
                {
                    if(COMPLEX)
                        rocsolver_lacgv_template<T>(handle, n - k + i, V, shiftV + idx2D(i, 0, ldv),
                                                    ldv, strideV, batch_count);

                    trans = rocblas_operation_none;
                    rocblasCall_gemv<T>(handle, trans, k - i - 1, u1_n - k + i, tau + i, strideT, V,
                                        shiftV + idx2D(i + 1, u2_n, ldv), ldv, strideV, V,
                                        shiftV + idx2D(i, u2_n, ldv), ldv, strideV, scalars + 2, 0,
                                        F, idx2D(i + 1, i, ldf), 1, strideF, batch_count, workArr);

                    if(COMPLEX)
                        rocsolver_lacgv_template<T>(handle, n - k + i, V, shiftV + idx2D(i, 0, ldv),
                                                    ldv, strideV, batch_count);
                }

                // multiply by the previous triangular factor
                trans = rocblas_operation_none;
                rocblasCall_trmv<T>(handle, uplo, trans, diag, k - i - 1, F,
                                    idx2D(i + 1, i + 1, ldf), ldf, strideF, F, idx2D(i + 1, i, ldf),
                                    1, strideF, work, stridew, batch_count);
            }
        }
    }

    // restore tau
    ROCSOLVER_LAUNCH_KERNEL(set_tau, dim3(blocks, batch_count), dim3(32, 1), 0, stream, k, tau,
                            strideT);

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

template <typename T, typename U>
ROCSOLVER_KERNEL void larft_set_tri(const rocblas_fill uplo,
                                    const rocblas_int k,
                                    U A,
                                    const rocblas_int shiftA,
                                    const rocblas_int lda,
                                    const rocblas_stride strideA,
                                    T* buffer)
{
    const auto b = hipBlockIdx_z;
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    const rocblas_int ldb = k;
    const rocblas_stride strideB = rocblas_stride(ldb) * k;

    const bool upper = (uplo == rocblas_fill_upper);
    const bool lower = (uplo == rocblas_fill_lower);

    if(i < k && j < k)
    {
        if((upper && j >= i) || (lower && i >= j))
        {
            T* Ap = load_ptr_batch<T>(A, b, shiftA, strideA);
            T* Bp = &buffer[b * strideB];

            // copy A to buffer
            Bp[i + j * ldb] = Ap[i + j * lda];

            // set A to unit triangular
            Ap[i + j * lda] = (i == j) ? 1 : 0;
        }
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void larft_restore_tri(const rocblas_fill uplo,
                                        const rocblas_int k,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        T* buffer)
{
    const auto b = hipBlockIdx_z;
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    const rocblas_int ldb = k;
    const rocblas_stride strideB = rocblas_stride(ldb) * k;

    const bool upper = (uplo == rocblas_fill_upper);
    const bool lower = (uplo == rocblas_fill_lower);

    if(i < k && j < k)
    {
        if((upper && j >= i) || (lower && i >= j))
        {
            T* Ap = load_ptr_batch<T>(A, b, shiftA, strideA);
            T* Bp = &buffer[b * strideB];

            // copy buffer to A
            Ap[i + j * lda] = Bp[i + j * ldb];
        }
    }
}

template <typename T>
ROCSOLVER_KERNEL void larft_set_diag(rocblas_int k,
                                     T* tau,
                                     const rocblas_stride strideT,
                                     T* F,
                                     const rocblas_int ldf,
                                     const rocblas_stride strideF)
{
    const auto b = hipBlockIdx_z;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(i < k)
    {
        T *tp, *Fp;
        tp = tau + b * strideT;
        Fp = F + b * strideF;

        Fp[i + i * ldf] = 1 / tp[i];
    }
}

template <bool BATCHED, typename T>
void rocsolver_larft_inverse_getMemorySize(const rocblas_int n,
                                           const rocblas_int k,
                                           const rocblas_int batch_count,
                                           size_t* size_work,
                                           size_t* size_workArr)
{
    // if quick return, no workspace is needed
    if(n == 0 || batch_count == 0)
    {
        *size_work = 0;
        *size_workArr = 0;
        return;
    }

    // size of re-usable workspace
    *size_work = sizeof(T) * k * k * batch_count;

    // size of array of pointers to workspace
    if(BATCHED)
        *size_workArr = sizeof(T*) * batch_count;
    else
        *size_workArr = 0;
}

template <typename T, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_larft_inverse_template(rocblas_handle handle,
                                                const rocblas_direct direct,
                                                const rocblas_storev storev,
                                                const rocblas_int n,
                                                const rocblas_int k,
                                                U V,
                                                const rocblas_int shiftV,
                                                const rocblas_int ldv,
                                                const rocblas_stride strideV,
                                                T* tau,
                                                const rocblas_stride strideT,
                                                T* F,
                                                const rocblas_int ldf,
                                                const rocblas_stride strideF,
                                                const rocblas_int batch_count,
                                                T* work,
                                                T** workArr)
{
    ROCSOLVER_ENTER("larft_inverse", "direct:", direct, "storev:", storev, "n:", n, "k:", k,
                    "shiftV:", shiftV, "ldv:", ldv, "ldf:", ldf, "bc:", batch_count);

    // quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // everything must be executed with scalars on the device
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    T one = 1;
    T zero = 0;

    const bool colwise = (storev == rocblas_column_wise);
    const bool forward = (direct == rocblas_forward_direction);

    rocblas_operation transA
        = colwise ? rocblas_operation_conjugate_transpose : rocblas_operation_none;
    rocblas_operation transB
        = colwise ? rocblas_operation_none : rocblas_operation_conjugate_transpose;

    rocblas_int tri_offset;
    rocblas_fill tri_uplo;

    if(colwise)
    {
        tri_uplo = forward ? rocblas_fill_upper : rocblas_fill_lower;
        tri_offset = (!forward && n > k) ? idx2D(n - k, 0, ldv) : 0;
    }
    else
    {
        tri_uplo = forward ? rocblas_fill_lower : rocblas_fill_upper;
        tri_offset = (!forward && n > k) ? idx2D(0, n - k, ldv) : 0;
    }

    rocblas_int blocks = (k - 1) / 32 + 1;
    dim3 gridTri(blocks, blocks, batch_count);
    dim3 blockTri(32, 32);

    // set V to unit triangular/trapezoidal
    ROCSOLVER_LAUNCH_KERNEL((larft_set_tri), gridTri, blockTri, 0, stream, tri_uplo, k, V,
                            shiftV + tri_offset, ldv, strideV, work);

    // compute: V' * V or V * V'
    rocsolver_gemm(handle, transA, transB, k, k, n, &one, V, shiftV, ldv, strideV, V, shiftV, ldv,
                   strideV, &zero, F, 0, ldf, strideF, batch_count, workArr);

    // set F diag to 1 / tau
    ROCSOLVER_LAUNCH_KERNEL(larft_set_diag, dim3(blocks, 1, batch_count), dim3(32, 1), 0, stream, k,
                            tau, strideT, F, ldf, strideF);

    // restore original V
    ROCSOLVER_LAUNCH_KERNEL((larft_restore_tri), gridTri, blockTri, 0, stream, tri_uplo, k, V,
                            shiftV + tri_offset, ldv, strideV, work);

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

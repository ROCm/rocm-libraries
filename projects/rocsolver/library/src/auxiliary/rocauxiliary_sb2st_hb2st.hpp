/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lapack_device_functions.hpp"
#include "lib_device_helpers.hpp"
#include "rocsolver_hybrid_storage.hpp"

ROCSOLVER_BEGIN_NAMESPACE

#define SB2ST_HB2ST_MAX_THDS 128

template <int MAX_THDS, typename T, typename S>
__device__ void sb2st_hb2st_sweep_step(const rocblas_int tid,
                                       rocblas_int n,
                                       rocblas_int nb,
                                       rocblas_int s,
                                       rocblas_int sm_i,
                                       T* A,
                                       rocblas_int lda,
                                       S* D,
                                       S* E,
                                       T* sval,
                                       T* work)
{
    __shared__ T tau;

    // first step of the sweep
    if(sm_i == s + 1)
    {
        rocblas_int sm_e = std::min(sm_i + nb, n);
        rocblas_int su_i = sm_e;
        rocblas_int su_e = std::min(su_i + nb, n);

        // generate Householder reflector
        rocblas_int mm = sm_e - sm_i;
        larfg<MAX_THDS>(tid, mm, A[sm_i + s * lda], A + (sm_i + 1) + s * lda, 1, tau, sval);
        __syncthreads();

        if(tid == 0)
        {
            E[s] = std::real(A[sm_i + s * lda]);
            A[sm_i + s * lda] = 1;
        }

        __syncthreads();

        // apply Householder reflector
        rocblas_int nn = su_e - sm_i;
        larf(tid, MAX_THDS, rocblas_side_left, mm, nn, A + sm_i + s * lda, 1, conj(tau),
             A + sm_i + sm_i * lda, lda, work);
        __syncthreads();
        larf(tid, MAX_THDS, rocblas_side_right, mm, mm, A + sm_i + s * lda, 1, tau,
             A + sm_i + sm_i * lda, lda, work);

        // copy transpose blocks
        nn = su_e - su_i;
        for(rocblas_int idx1d = tid; idx1d < nn * mm; idx1d += MAX_THDS)
        {
            rocblas_int i = su_i + idx1d % nn;
            rocblas_int j = sm_i + idx1d / nn;
            A[i + j * lda] = conj(A[j + i * lda]);
        }

        // save tau
        if(tid == 0)
            A[sm_i + s * lda] = tau;
    }

    // bulge chasing
    else
    {
        rocblas_int sm_e = std::min(sm_i + nb, n);
        rocblas_int su_i = sm_e;
        rocblas_int su_e = std::min(su_i + nb, n);
        rocblas_int sd_i = sm_i - nb;
        rocblas_int sd_e = sm_i;

        // generate Householder reflector
        rocblas_int mm = sm_e - sm_i;
        larfg<MAX_THDS>(tid, mm, A[sm_i + sd_i * lda], A + (sm_i + 1) + sd_i * lda, 1, tau, sval);
        __syncthreads();

        // copy Householder vector to column s
        if(tid == 0)
            A[sm_i + s * lda] = 1;
        for(rocblas_int i = sm_i + 1 + tid; i < sm_e; i += MAX_THDS)
        {
            A[i + s * lda] = A[i + sd_i * lda];
            A[i + sd_i * lda] = 0;
        }
        __syncthreads();

        // apply Householder reflector
        rocblas_int nn = su_e - sd_i - 1;
        larf(tid, MAX_THDS, rocblas_side_left, mm, nn, A + sm_i + s * lda, 1, conj(tau),
             A + sm_i + (sd_i + 1) * lda, lda, work);
        __syncthreads();
        larf(tid, MAX_THDS, rocblas_side_right, mm, mm, A + sm_i + s * lda, 1, tau,
             A + sm_i + sm_i * lda, lda, work);

        // copy transpose blocks
        nn = su_e - su_i;
        for(rocblas_int idx1d = tid; idx1d < nn * mm; idx1d += MAX_THDS)
        {
            rocblas_int i = su_i + idx1d % nn;
            rocblas_int j = sm_i + idx1d / nn;
            A[i + j * lda] = conj(A[j + i * lda]);
        }
        nn = sd_e - sd_i;
        for(rocblas_int idx1d = tid; idx1d < nn * mm; idx1d += MAX_THDS)
        {
            rocblas_int i = sd_i + idx1d % nn;
            rocblas_int j = sm_i + idx1d / nn;
            A[i + j * lda] = conj(A[j + i * lda]);
        }

        // save tau
        if(tid == 0)
            A[sm_i + s * lda] = tau;
    }
}

/* SB2ST_HB2ST_KERNEL runs all sweeps on a single thread block per batch instance. Run with
   batch_count thread blocks in z. */
template <typename T, typename S>
ROCSOLVER_KERNEL void __launch_bounds__(SB2ST_HB2ST_MAX_THDS)
    sb2st_hb2st_kernel(rocblas_int n,
                       rocblas_int nb,
                       T* AA,
                       rocblas_stride shiftA,
                       rocblas_int lda,
                       rocblas_stride strideA,
                       S* DD,
                       rocblas_stride strideD,
                       S* EE,
                       rocblas_stride strideE,
                       T* workA,
                       rocblas_stride strideW)
{
    const rocblas_int tid = threadIdx.x;
    const rocblas_int bid = blockIdx.z;

    assert(blockDim.x == SB2ST_HB2ST_MAX_THDS);

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    S* D = load_ptr_batch<S>(DD, bid, 0, strideD);
    S* E = load_ptr_batch<S>(EE, bid, 0, strideE);
    T* work;

    // shared mem for temporary values
    extern __shared__ double lmem[];
    T* sval = reinterpret_cast<T*>(lmem);

    if(workA)
        work = workA + bid * strideW;
    else
        work = reinterpret_cast<T*>(lmem);

    // execute sweeps
    for(rocblas_int s = 0; s < n - 1; s++)
    {
        for(rocblas_int sm_i = s + 1; sm_i < n; sm_i += nb)
        {
            sb2st_hb2st_sweep_step<SB2ST_HB2ST_MAX_THDS, T, S>(tid, n, nb, s, sm_i, A, lda, D, E,
                                                               sval, work);
        }
    }
}

/* SB2ST_HB2ST_STEP_KERNEL runs a single step from multiple sweeps in parallel. Run with
   sweeps_in_parallel thread blocks in y and batch_count thread blocks in z.

   Sweep i can begin execution when sweep i-1 has completed 3 steps. That is,
   - Sweep 0 can start at step 0
   - Sweep 1 can start at step 3
   ...
   - Sweep i can start at step 3*i
   ...
   - Sweep n-1 can start at step 3*(n-1)

   Sweep n-1 is complete after 1 step, therefore the total number of steps is 3*(n-1)+1 */
template <typename T, typename S>
ROCSOLVER_KERNEL void __launch_bounds__(SB2ST_HB2ST_MAX_THDS)
    sb2st_hb2st_step_kernel(rocblas_int n,
                            rocblas_int nb,
                            rocblas_int step,
                            T* AA,
                            rocblas_stride shiftA,
                            rocblas_int lda,
                            rocblas_stride strideA,
                            S* DD,
                            rocblas_stride strideD,
                            S* EE,
                            rocblas_stride strideE,
                            T* workA,
                            rocblas_stride strideW)
{
    const rocblas_int tid = threadIdx.x;
    const rocblas_int sid = blockIdx.y;
    const rocblas_int bid = blockIdx.z;

    assert(blockDim.x == SB2ST_HB2ST_MAX_THDS);

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    S* D = load_ptr_batch<S>(DD, bid, 0, strideD);
    S* E = load_ptr_batch<S>(EE, bid, 0, strideE);
    T* work;

    // shared mem for temporary values
    extern __shared__ double lmem[];
    T* sval = reinterpret_cast<T*>(lmem);

    if(workA)
        work = workA + bid * strideW;
    else
        work = reinterpret_cast<T*>(lmem);

    // get sweep parameters
    rocblas_int last_started = step / 3;
    rocblas_int s = last_started - sid;
    rocblas_int step_in_sweep = step - (3 * s);
    rocblas_int sm_i = s + 1 + step_in_sweep * nb;

    if(s < 0 || sm_i >= n)
        return;

    // if(tid == 0)
    //     printf("Global step %i, sweep %i, local step %i\n", step, s, step_in_sweep);

    // execute sweep step
    sb2st_hb2st_sweep_step<SB2ST_HB2ST_MAX_THDS, T, S>(tid, n, nb, s, sm_i, A, lda, D, E, sval, work);
}

template <typename T, typename S>
ROCSOLVER_KERNEL void sb2st_hb2st_copy_diag(rocblas_int n,
                                            T* AA,
                                            rocblas_stride shiftA,
                                            rocblas_int lda,
                                            rocblas_stride strideA,
                                            S* DD,
                                            rocblas_stride strideD)
{
    const rocblas_int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const rocblas_int bid = blockIdx.z;

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    S* D = load_ptr_batch<S>(DD, bid, 0, strideD);

    // copy diag
    if(tid < n)
        D[tid] = std::real(A[tid + tid * lda]);
}

template <bool BATCHED, typename T, typename S>
void rocsolver_sb2st_hb2st_getMemorySize(const rocblas_int n,
                                         const rocblas_int nb,
                                         const rocblas_int batch_count,
                                         size_t* size_work)
{
    if(n <= 1)
    {
        *size_work = 0;
        return;
    }

    *size_work = sizeof(T) * (3 * nb) * batch_count;
}

template <typename T, typename S>
rocblas_status rocsolver_sb2st_hb2st_argCheck(rocblas_handle handle,
                                              const rocblas_int n,
                                              const rocblas_int nb,
                                              const rocblas_int lda,
                                              T A,
                                              S* D,
                                              S* E,
                                              const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(n < 0 || nb < 0 || lda < n)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n > 0 && !A) || (n > 0 && !D) || (n > 1 && !E))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
rocblas_status rocsolver_sb2st_hb2st_template(rocblas_handle handle,
                                              const rocblas_int n,
                                              const rocblas_int nb,
                                              U A,
                                              const rocblas_stride shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              S* D,
                                              const rocblas_stride strideD,
                                              S* E,
                                              const rocblas_stride strideE,
                                              const rocblas_int batch_count,
                                              T* W)
{
    ROCSOLVER_ENTER("sb2st_hb2st", "n:", n, "nb:", nb, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    // quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // quick return for n = 1 (scalar case)
    if(n == 1)
    {
        rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
        dim3 gridReset(blocksReset, 1, 1);
        dim3 threadsReset(BS1, 1, 1);

        ROCSOLVER_LAUNCH_KERNEL(scalar_case<T>, gridReset, threadsReset, 0, stream,
                                rocblas_evect_none, A, strideA, D, strideD, batch_count);
        return rocblas_status_success;
    }

    // rocblas_pointer_mode old_mode;
    // rocblas_get_pointer_mode(handle, &old_mode);
    // rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    int device;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device));

    T* work = nullptr;
    rocblas_int strideW = 3 * nb;

    size_t lmemsize_larfg = sizeof(T) * SB2ST_HB2ST_MAX_THDS;
    size_t lmemsize_larf = sizeof(T) * strideW;
    size_t lmemsize = std::max(lmemsize_larfg, lmemsize_larf);

    if(lmemsize > props.sharedMemPerBlock)
    {
        lmemsize = lmemsize_larfg;
        work = W;
    }

    const rocblas_int steps_per_sweep = (n - 2) / nb + 1;
    const rocblas_int sweeps_in_parallel = (steps_per_sweep - 1) / 3 + 1;
    const rocblas_int num_steps = 3 * (n - 1) + 1;

    // execute sweeps
    if(sweeps_in_parallel < 2)
    {
        ROCSOLVER_LAUNCH_KERNEL(sb2st_hb2st_kernel<T>, dim3(1, 1, batch_count),
                                dim3(SB2ST_HB2ST_MAX_THDS, 1, 1), lmemsize, stream, n, nb, A,
                                shiftA, lda, strideA, D, strideD, E, strideE, work, strideW);
    }
    else
    {
        for(rocblas_int step = 0; step < num_steps; step++)
        {
            ROCSOLVER_LAUNCH_KERNEL(sb2st_hb2st_step_kernel<T>,
                                    dim3(1, sweeps_in_parallel, batch_count),
                                    dim3(SB2ST_HB2ST_MAX_THDS, 1, 1), lmemsize, stream, n, nb, step,
                                    A, shiftA, lda, strideA, D, strideD, E, strideE, work, strideW);
        }
    }

    // copy diagonal
    const rocblas_int copyblocks = (n - 1) / BS1 + 1;
    ROCSOLVER_LAUNCH_KERNEL((sb2st_hb2st_copy_diag<T>), dim3(copyblocks, 1, batch_count), dim3(BS1),
                            0, stream, n, A, shiftA, lda, strideA, D, strideD);

    // rocblas_set_pointer_mode(handle, old_mode);

    return rocblas_status_success;
}

#undef SB2ST_HB2ST_MAX_THDS

ROCSOLVER_END_NAMESPACE

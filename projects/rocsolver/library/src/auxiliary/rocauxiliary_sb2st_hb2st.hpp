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

// Number of threads in x and y
// Reductions in larfg and larf must be updated if DIMX is changed
#define DIMX 32
#define DIMY 32

template <typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
__device__ __inline__ T shift_left(T& value, int lane_delta)
{
    T r = value;
    r = __shfl_down(r, lane_delta);
    return r;
}

template <typename T, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
__device__ __inline__ T shift_left(T& value, int lane_delta)
{
    using S = decltype(std::real(T{}));
    S r = value.real();
    S i = value.imag();
    r = __shfl_down(r, lane_delta);
    i = __shfl_down(i, lane_delta);
    return rocblas_complex_num<S>(r, i);
}

template <typename T, typename I, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
__device__ void sb2st_larfg(const I xid, I n, T& alpha, T* x, T& tau, T* reduct)
{
    // dot reduction
    T norm2 = 0;
    for(I i = xid; i < n - 1; i += DIMX)
        norm2 += x[i] * x[i];
    norm2 += shift_left(norm2, 16);
    norm2 += shift_left(norm2, 8);
    norm2 += shift_left(norm2, 4);
    norm2 += shift_left(norm2, 2);
    norm2 += shift_left(norm2, 1);
    if(xid == 0)
        reduct[0] = norm2;
    __threadfence();
    norm2 = reduct[0] + alpha * conj(alpha);

    __shared__ T s;

    if(norm2 > 0)
    {
        if(xid == 0)
        {
            T norm = alpha >= 0 ? -std::sqrt(norm2) : std::sqrt(norm2);

            s = (T)(1.0 / (alpha - norm));
            tau = (norm - alpha) / norm;
            alpha = norm;
        }

        // scal
        for(I i = xid; i < n - 1; i += DIMX)
            x[i] *= s;
    }
    else
    {
        tau = 0;
    }
}

template <typename T, typename I, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
__device__ void sb2st_larfg(const I xid, I n, T& alpha, T* x, T& tau, T* reduct)
{
    using S = decltype(std::real(T{}));

    // dot reduction
    T norm2 = 0;
    for(I i = xid; i < n - 1; i += DIMX)
        norm2 += x[i] * conj(x[i]);
    norm2 += shift_left(norm2, 16);
    norm2 += shift_left(norm2, 8);
    norm2 += shift_left(norm2, 4);
    norm2 += shift_left(norm2, 2);
    norm2 += shift_left(norm2, 1);
    if(xid == 0)
        reduct[0] = norm2;
    __threadfence();
    norm2 = reduct[0] + alpha * conj(alpha);

    S ar = alpha.real();
    S ai = alpha.imag();
    __shared__ T s;

    if(norm2.real() > 0 || ai > 0)
    {
        if(xid == 0)
        {
            S norm = ar >= 0 ? -std::sqrt(norm2.real()) : std::sqrt(norm2.real());

            // scaling factor
            S r = (ar - norm) * (ar - norm) + ai * ai;
            S rr = (ar - norm) / r;
            S ri = -ai / r;
            s = rocblas_complex_num<S>(rr, ri);

            // tau
            rr = (norm - ar) / norm;
            ri = -ai / norm;
            tau = rocblas_complex_num<S>(rr, ri);

            // alpha
            alpha = norm;
        }

        // scal
        for(I i = xid; i < n - 1; i += DIMX)
            x[i] *= s;
    }
    else
    {
        tau = 0;
    }
}

template <typename T, typename I>
__device__ void
    sb2st_larf(const I xid, const I yid, rocblas_side side, I m, I n, T* v, T tau, T* C, I ldc, T* reduct)
{
    if(side == rocblas_side_left)
    {
        for(I j = yid; j < n; j += DIMY)
        {
            // gemv reduction
            T value = 0;
            for(I i = xid; i < m; i += DIMX)
                value += conj(C[i + j * ldc]) * v[i];
            value += shift_left(value, 16);
            value += shift_left(value, 8);
            value += shift_left(value, 4);
            value += shift_left(value, 2);
            value += shift_left(value, 1);
            if(xid == 0)
                reduct[yid] = value;
            __threadfence();

            // ger
            for(I i = xid; i < m; i += DIMX)
                C[i + j * ldc] -= tau * v[i] * conj(reduct[yid]);
        }
    }
    else
    {
        for(I i = yid; i < m; i += DIMY)
        {
            // gemv reduction
            T value = 0;
            for(I j = xid; j < n; j += DIMX)
                value += C[i + j * ldc] * v[j];
            value += shift_left(value, 16);
            value += shift_left(value, 8);
            value += shift_left(value, 4);
            value += shift_left(value, 2);
            value += shift_left(value, 1);
            if(xid == 0)
                reduct[yid] = value;
            __threadfence();

            // ger
            for(I j = xid; j < n; j += DIMX)
                C[i + j * ldc] -= tau * conj(v[j]) * reduct[yid];
        }
    }
}

template <typename T, typename S>
__device__ void sb2st_hb2st_sweep_step(const rocblas_int xid,
                                       const rocblas_int yid,
                                       rocblas_int n,
                                       rocblas_int nb,
                                       rocblas_int s,
                                       rocblas_int sm_i,
                                       T* A,
                                       rocblas_int lda,
                                       S* D,
                                       S* E,
                                       T* housev,
                                       T* reduct)
{
    const rocblas_int tid = xid + yid * DIMX;

    __shared__ T tau;

    // first step of the sweep
    if(sm_i == s + 1)
    {
        rocblas_int sm_e = std::min(sm_i + nb, n);
        rocblas_int su_i = sm_e;
        rocblas_int su_e = std::min(su_i + nb, n);

        rocblas_int mm = sm_e - sm_i;
        if(yid == 0)
        {
            // copy column s to shared memory
            for(rocblas_int i = xid; i < mm; i += DIMX)
                housev[i] = A[(sm_i + i) + s * lda];

            // generate Householder reflector
            sb2st_larfg(xid, mm, housev[0], housev + 1, tau, reduct);

            // copy Householder vector to column s of A
            if(xid == 0)
            {
                A[sm_i + s * lda] = tau;
                E[s] = std::real(housev[0]);
                housev[0] = T(1);
            }
            for(rocblas_int i = 1 + xid; i < mm; i += DIMX)
                A[(sm_i + i) + s * lda] = housev[i];
        }
        __syncthreads();

        // apply Householder reflector
        if(tau != 0)
        {
            rocblas_int nn = su_e - sm_i;
            sb2st_larf(xid, yid, rocblas_side_left, mm, nn, housev, conj(tau),
                       A + sm_i + sm_i * lda, lda, reduct);
            __syncthreads();
            sb2st_larf(xid, yid, rocblas_side_right, mm, mm, housev, tau, A + sm_i + sm_i * lda,
                       lda, reduct);

            // copy transpose blocks
            nn = su_e - su_i;
            for(rocblas_int idx1d = tid; idx1d < nn * mm; idx1d += DIMX * DIMY)
            {
                rocblas_int i = su_i + idx1d % nn;
                rocblas_int j = sm_i + idx1d / nn;
                A[i + j * lda] = conj(A[j + i * lda]);
            }
        }
    }

    // bulge chasing
    else
    {
        rocblas_int sm_e = std::min(sm_i + nb, n);
        rocblas_int su_i = sm_e;
        rocblas_int su_e = std::min(su_i + nb, n);
        rocblas_int sd_i = sm_i - nb;
        rocblas_int sd_e = sm_i;

        rocblas_int mm = sm_e - sm_i;
        if(yid == 0)
        {
            // copy column s to shared memory
            for(rocblas_int i = xid; i < mm; i += DIMX)
                housev[i] = A[(sm_i + i) + sd_i * lda];

            // generate Householder reflector
            sb2st_larfg(xid, mm, housev[0], housev + 1, tau, reduct);

            // copy Householder vector to column s of A
            if(xid == 0)
            {
                A[sm_i + s * lda] = tau;
                A[sm_i + sd_i * lda] = housev[0];
                housev[0] = T(1);
            }
            for(rocblas_int i = 1 + xid; i < mm; i += DIMX)
            {
                A[(sm_i + i) + s * lda] = housev[i];
                A[(sm_i + i) + sd_i * lda] = 0;
            }
        }
        __syncthreads();

        // apply Householder reflector
        if(tau != 0)
        {
            rocblas_int nn = su_e - sd_i - 1;
            sb2st_larf(xid, yid, rocblas_side_left, mm, nn, housev, conj(tau),
                       A + sm_i + (sd_i + 1) * lda, lda, reduct);
            __syncthreads();
            sb2st_larf(xid, yid, rocblas_side_right, mm, mm, housev, tau, A + sm_i + sm_i * lda,
                       lda, reduct);

            // copy transpose blocks
            nn = su_e - su_i;
            for(rocblas_int idx1d = tid; idx1d < nn * mm; idx1d += DIMX * DIMY)
            {
                rocblas_int i = su_i + idx1d % nn;
                rocblas_int j = sm_i + idx1d / nn;
                A[i + j * lda] = conj(A[j + i * lda]);
            }
            nn = sd_e - sd_i;
            for(rocblas_int idx1d = tid; idx1d < nn * mm; idx1d += DIMX * DIMY)
            {
                rocblas_int i = sd_i + idx1d % nn;
                rocblas_int j = sm_i + idx1d / nn;
                A[i + j * lda] = conj(A[j + i * lda]);
            }
        }
    }
}

/* SB2ST_HB2ST_KERNEL runs all sweeps on a single thread block per batch instance. Run with
   batch_count thread blocks in z. */
template <typename T, typename S>
ROCSOLVER_KERNEL void sb2st_hb2st_kernel(rocblas_int n,
                                         rocblas_int nb,
                                         T* AA,
                                         rocblas_stride shiftA,
                                         rocblas_int lda,
                                         rocblas_stride strideA,
                                         S* DD,
                                         rocblas_stride strideD,
                                         S* EE,
                                         rocblas_stride strideE)
{
    const rocblas_int xid = threadIdx.x;
    const rocblas_int yid = threadIdx.y;
    const rocblas_int bid = blockIdx.z;

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    S* D = load_ptr_batch<S>(DD, bid, 0, strideD);
    S* E = load_ptr_batch<S>(EE, bid, 0, strideE);

    // shared memory setup
    extern __shared__ double lmem[];
    T* housev = reinterpret_cast<T*>(lmem);
    T* reduct = reinterpret_cast<T*>(housev + nb);

    // execute sweeps
    for(rocblas_int s = 0; s < n - 1; s++)
    {
        for(rocblas_int sm_i = s + 1; sm_i < n; sm_i += nb)
        {
            sb2st_hb2st_sweep_step<T, S>(xid, yid, n, nb, s, sm_i, A, lda, D, E, housev, reduct);
            __syncthreads();
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
ROCSOLVER_KERNEL void sb2st_hb2st_step_kernel(rocblas_int n,
                                              rocblas_int nb,
                                              rocblas_int step,
                                              T* AA,
                                              rocblas_stride shiftA,
                                              rocblas_int lda,
                                              rocblas_stride strideA,
                                              S* DD,
                                              rocblas_stride strideD,
                                              S* EE,
                                              rocblas_stride strideE)
{
    const rocblas_int xid = threadIdx.x;
    const rocblas_int yid = threadIdx.y;
    const rocblas_int sid = blockIdx.y;
    const rocblas_int bid = blockIdx.z;

    assert(blockDim.x == SB2ST_HB2ST_MAX_THDS);

    // select batch instance
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    S* D = load_ptr_batch<S>(DD, bid, 0, strideD);
    S* E = load_ptr_batch<S>(EE, bid, 0, strideE);

    // shared memory setup
    extern __shared__ double lmem[];
    T* housev = reinterpret_cast<T*>(lmem);
    T* reduct = reinterpret_cast<T*>(housev + nb);

    // get sweep parameters
    rocblas_int last_started = step / 3;
    rocblas_int s = last_started - sid;
    rocblas_int step_in_sweep = step - (3 * s);
    rocblas_int sm_i = s + 1 + step_in_sweep * nb;

    if(s < 0 || sm_i >= n)
        return;

    // execute sweep step
    sb2st_hb2st_sweep_step<T, S>(xid, yid, n, nb, s, sm_i, A, lda, D, E, housev, reduct);
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

    size_t lmemsize_housev = sizeof(T) * nb;
    size_t lmemsize_reduction = sizeof(T) * DIMY;
    size_t lmemsize = lmemsize_housev + lmemsize_reduction;

    if(lmemsize > props.sharedMemPerBlock)
    {
        return rocblas_status_internal_error;
    }

    const rocblas_int steps_per_sweep = (n - 2) / nb + 1;
    const rocblas_int sweeps_in_parallel = (steps_per_sweep - 1) / 3 + 1;
    const rocblas_int num_steps = 3 * (n - 1) + 1;

    // execute sweeps
    if(sweeps_in_parallel < 3)
    {
        ROCSOLVER_LAUNCH_KERNEL(sb2st_hb2st_kernel<T>, dim3(1, 1, batch_count), dim3(DIMX, DIMY, 1),
                                lmemsize, stream, n, nb, A, shiftA, lda, strideA, D, strideD, E,
                                strideE);
    }
    else
    {
        for(rocblas_int step = 0; step < num_steps; step++)
        {
            ROCSOLVER_LAUNCH_KERNEL(sb2st_hb2st_step_kernel<T>,
                                    dim3(1, sweeps_in_parallel, batch_count), dim3(DIMX, DIMY, 1),
                                    lmemsize, stream, n, nb, step, A, shiftA, lda, strideA, D,
                                    strideD, E, strideE);
        }
    }

    // copy diagonal
    const rocblas_int copyblocks = (n - 1) / BS1 + 1;
    ROCSOLVER_LAUNCH_KERNEL((sb2st_hb2st_copy_diag<T>), dim3(copyblocks, 1, batch_count), dim3(BS1),
                            0, stream, n, A, shiftA, lda, strideA, D, strideD);

    // rocblas_set_pointer_mode(handle, old_mode);

    return rocblas_status_success;
}

#undef DIMX
#undef DIMY

ROCSOLVER_END_NAMESPACE

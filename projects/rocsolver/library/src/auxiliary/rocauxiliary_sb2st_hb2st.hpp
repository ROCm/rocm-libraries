/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

//------------------------------------------------------------------------------
// Hermitianize a band matrix stored in LAPACK band format (column-major).
//
// Layout: Aband has ldab rows and n columns.
//   ku super-diagonals  (band rows 0 .. ku-1)
//   main diagonal       (band row  ku)
//   ku+1 sub-diagonals  (band rows ku+1 .. 2*ku+1)
//
// Element A(i,j) (0-indexed) is stored at Aband[b*strideAb + j*ldab + (ku+i-j)].
//
// For each d = 1..ku and valid column j, copies sub-diagonal d to
// super-diagonal d with conjugation:
//   A(j, j+d) := conj( A(j+d, j) )
//
// Sub-diagonal ku+1 is left untouched.
template <typename T, typename I>
ROCSOLVER_KERNEL void hermitianize_band_kernel(I n, I ku, T* Aband, I ldab, rocblas_stride strideAb)
{
    I j = blockIdx.x * blockDim.x + threadIdx.x; // matrix column, 0-indexed
    I d = blockIdx.y * blockDim.y + threadIdx.y + 1; // sub-diagonal index, 1..ku
    I b = blockIdx.z; // batch index

    if(d > ku || j >= n || j + d >= n)
        return;

    T* AB = Aband + b * strideAb;

    // Source: A(j+d, j)  band row = ku+d, col = j
    // Dest:   A(j, j+d)  band row = ku-d, col = j+d
    AB[(j + d) * ldab + (ku - d)] = conj(AB[j * ldab + (ku + d)]);
}

//------------------------------------------------------------------------------
template <typename T, typename I>
void hermitianize_band(rocblas_handle handle,
                       I n,
                       I ku,
                       T* Aband,
                       I ldab,
                       rocblas_stride strideAb,
                       I batch_count)
{
    if(n <= 0 || ku <= 0 || batch_count <= 0)
        return;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    dim3 block(BS2, BS2 / 4, 1);
    dim3 grid(ceildiv(n, I(block.x)), ceildiv(ku, I(block.y)), batch_count);

    ROCSOLVER_LAUNCH_KERNEL((hermitianize_band_kernel<T, I>), grid, block, 0, stream, n, ku, Aband,
                            ldab, strideAb);
}

//------------------------------------------------------------------------------
// Generates a Householder reflector.
// Must be called with one wave; this assumes threads are synchronized and does
// not do __syncthreads. It does __threadfence_block where __syncthreads would
// normally be required.
// Compared to usual larfg, this passes x = [alpha; xhat] as single argument
// to make reduction simpler.
// If norm( xhat ) == 0, LAPACK sets tau = 0 and H = I,
// whereas this will set tau = 2 and H = [ -1 0 ].
//                                       [  0 I ]
//
//  xid         Thread x index.
//  n           Vector dimension.
//  x           On input, vector to reduce.
//              On output, x[0] = plusminus norm(x),
//              x[1:] is Householder vector with implicit 1 in x[0].
//  tau         On output, Householder tau value.
//  s_work      Shared memory workspace of size >= 1.
//
template <typename T, typename I, typename S = decltype(std::real(T{}))>
__device__ void sb2st_larfg(const I xid, I n, T* x, T& tau, S* s_work)
{
    // Reduction assumes this DIMX.
    static_assert(DIMX == 32);

    const S one = 1;

    // norm reduction
    S norm2 = 0;
    for(I i = xid; i < n; i += DIMX)
    {
        norm2 += std::norm(x[i]); // i.e., abs(xi)^2
    }
    norm2 += shift_left(norm2, 16);
    norm2 += shift_left(norm2, 8);
    norm2 += shift_left(norm2, 4);
    norm2 += shift_left(norm2, 2);
    norm2 += shift_left(norm2, 1);
    if(xid == 0)
    {
        s_work[0] = norm2;
    }
    __threadfence_block();
    norm2 = s_work[0];

    S alpha_r = std::real(x[0]);
    S alpha_i = std::imag(x[0]); // In real, alpha_i = 0. Compiler can eliminate it.

    __shared__ T s_scale;

    // The way we do norm2 above, it already includes alpha.
    // In LAPACK, at this point norm is just x (== x[1:n]), excluding alpha (== x[0]).
    if(norm2 > 0 || alpha_i > 0)
    {
        if(xid == 0)
        {
            S sqrt_norm2 = std::sqrt(norm2);
            S norm = alpha_r >= 0 ? -sqrt_norm2 : sqrt_norm2;

            if constexpr(rocblas_is_complex<T>)
            {
                // scaling factor
                S r = (alpha_r - norm) * (alpha_r - norm) + alpha_i * alpha_i;
                S rr = (alpha_r - norm) / r;
                S ri = -alpha_i / r;
                s_scale = rocblas_complex_num<S>(rr, ri);

                // tau
                rr = (norm - alpha_r) / norm;
                ri = -alpha_i / norm;
                tau = rocblas_complex_num<S>(rr, ri);
            }
            else
            {
                s_scale = one / (alpha_r - norm);
                tau = (norm - alpha_r) / norm;
            }

            x[0] = norm;
        }
        __threadfence_block(); // for s_scale

        // scal x[1:n]
        for(I i = xid + 1; i < n; i += DIMX)
        {
            x[i] *= s_scale;
        }
    }
    else
    {
        tau = 0;
    }
}

//------------------------------------------------------------------------------
// Applies H on left or right of C:
// C := H C if on left,
// C := C H if on right.
// To apply H^H, pass in conj( tau ).
// Called with DIMX x DIMY thread block, such that threads in the x dimension
// are in a wavefront and are implicitly synchronized.
//
//  xid, yid    Thread x and y indices.
//  side        Side to apply H, on left or right.
//  m, n        Dimensions of block being updated.
//  v, tau      Householder vector and tau; assumes v[0] == 1 explicitly.
//  C           M-by-n block to update.
//  s_work      Shared memory workspace of size = block y dimension.
//
template <typename T, typename I>
__device__ void
    sb2st_larf(const I xid, const I yid, rocblas_side side, I m, I n, T* v, T tau, T* C, I ldc, T* s_work)
{
    // Reductions assume this DIMX.
    static_assert(DIMX == 32);

    if(side == rocblas_side_left)
    {
        for(I j = yid; j < n; j += DIMY)
        {
            // gemv reduction
            // C = (I - tau v v^H) C = C - tau v (v^H C)
            // w = C^H v
            T value = 0;
            for(I i = xid; i < m; i += DIMX)
            {
                value += conj(C[i + j * ldc]) * v[i];
            }
            value += shift_left(value, 16);
            value += shift_left(value, 8);
            value += shift_left(value, 4);
            value += shift_left(value, 2);
            value += shift_left(value, 1);
            if(xid == 0)
            {
                // todo: what about multiplying conj(tau) here instead of in loop below?
                s_work[yid] = value;
            }
            __threadfence_block(); // threads are sync'd in one wavefront

            // ger
            // Cj = Cj - tau v conj( wj ) = C[:,j] - tau v (v^H Cj)
            for(I i = xid; i < m; i += DIMX)
            {
                C[i + j * ldc] -= tau * v[i] * conj(s_work[yid]);
            }
        }
    }
    else
    {
        for(I i = yid; i < m; i += DIMY)
        {
            // gemv reduction
            // C = C (I - tau v v^H) = C - tau (C v) v^H
            // w = C v
            T value = 0;
            for(I j = xid; j < n; j += DIMX)
            {
                value += C[i + j * ldc] * v[j];
            }
            value += shift_left(value, 16);
            value += shift_left(value, 8);
            value += shift_left(value, 4);
            value += shift_left(value, 2);
            value += shift_left(value, 1);
            if(xid == 0)
            {
                // todo: what about multiplying tau here instead of in loop below?
                s_work[yid] = value;
            }
            __threadfence_block(); // threads are sync'd in one wavefront

            // ger
            // Cj = Cj - tau wj v^H = Cj - tau (Cj v) v^H
            for(I j = xid; j < n; j += DIMX)
            {
                C[i + j * ldc] -= tau * conj(v[j]) * s_work[yid];
            }
        }
    }
}

//------------------------------------------------------------------------------
// Applies H on left and right of Hermitian block:
// C := H^H C H
// Sets the whole Hermitian block, both upper and lower.
// (In LAPACK, this is larfy, which doesn't fit well into LAPACK's naming
// conventions. I guess y comes from sy.)
// Called with DIMX x DIMY thread block, such that threads in the x dimension
// are in a wavefront and are implicitly synchronized.
//
//  xid, yid    Thread x and y indices.
//  n           Block dimension.
//  v, tau      Householder vector; assumes v[0] == 1 explicitly.
//  C           n-by-n Hermitian block to update.
//  s_work      Shared memory workspace of size = block y dimension.
//
template <typename T, typename I>
__device__ void sb2st_helarf(const I xid, const I yid, I n, T* v, T tau, T* C, I ldc, T* s_work)
{
    // Reductions assume this DIMX.
    static_assert(DIMX == 32);

    // gemv/hemv: w = C * v
    for(I j = yid; j < n; j += DIMY)
    {
        // gemv reduction
        // C = (I - tau v v^H) C = C - tau v (v^H C)
        // wj = v^T * conj( Cj )
        // todo: why not do wj = v^H * Cj, i.e., conj( v )?
        T value = 0;
        for(I i = xid; i < n; i += DIMX)
        {
            value += C[i + j * ldc] * v[i];
        }
        value += shift_left(value, 16);
        value += shift_left(value, 8);
        value += shift_left(value, 4);
        value += shift_left(value, 2);
        value += shift_left(value, 1);
        if(xid == 0)
        {
            // todo: what about multiplying tau here?
            s_work[yid] = value;
        }
    }
    __syncthreads();

    // w = w - (0.5 tau w^H v) v
    // dot reduction: alpha = 0.5 tau w^H v
    __shared__ T s_alpha;
    if(yid == 0)
    {
        T value = 0;
        for(I i = xid; i < n; i += DIMX)
        {
            value += conj(s_work[i]) * v[i];
        }
        value += shift_left(value, 16);
        value += shift_left(value, 8);
        value += shift_left(value, 4);
        value += shift_left(value, 2);
        value += shift_left(value, 1);
        if(xid == 0)
        {
            s_alpha = 0.5 * tau * value;
        }
    }
    __syncthreads();

    // axpy: w = w - alpha v
    if(yid == 0)
    {
        for(I i = xid; i < n; i += DIMX)
        {
            s_work[i] -= s_alpha * v[i];
        }
    }
    __syncthreads();

    // ger2/her2: C := C - tau v w^H - conj(tau) w v^H
    for(I j = yid; j < n; j += DIMY)
    {
        for(I i = xid; i < n; i += DIMX)
        {
            C[i + j * ldc] -= tau * v[i] * conj(s_work[yid]) + conj(tau) * s_work[yid] * conj(v[i]);
        }
    }
}

//------------------------------------------------------------------------------
// Does a single task in the hb2st task-graph.
//
//  n           Matrix dimension.
//  kd          Matrix bandwidth.
//  sweep       Column being reduced to tridiagonal and its bulges chased.
//  task        Step in current sweep.
//  Aband       Band matrix. On output:
//              If task = 0: reduces column sweep to tridiagonal and updates
//              diagonal block at sweep + 1.
//              If task > 0: updates off-diagonal block (bulge) at column
//              sweep + 1 + (task - 1)*kd, reduces first column of bulge back
//              to bandwidth kd, and updates diagonal block.
//  E           On output with task = 0, sets sub-diagonal element E[sweep].
//  V           Array of Householder vectors.
//  tau         Householder tau values.
//  s_housev    Shared memory workspace for Householder vectors of size = kd.
//  s_work      Shared memory workspace of size = block y dimension.
//
template <typename T, typename I, typename S>
__device__ void sb2st_hb2st_task(const I xid,
                                 const I yid,
                                 I n,
                                 I kd,
                                 I sweep,
                                 I task,
                                 T* Aband,
                                 I ldab,
                                 S* E,
                                 T* V,
                                 I ldv,
                                 T* tau,
                                 T* s_housev,
                                 T* s_work)
{
    // gemv implementation is faster than hemv,
    // which is provided for comparison.
    bool const use_hemv = false;

    const I tid = xid + yid * DIMX;

    I idiag = kd - 1;

    // row, col index for current Householder vector vc within V.
    I vi, vj;
    get_v_index(n, kd, sweep, task, vi, vj);

    __shared__ T s_tau;

    // `vp` is Householder vector generated in previous task.
    // `vc` is Householder vector generated in current  task.
    //
    // `jp` is left col of previous diagonal tile and current off-diagonal tile
    //      (defined later).
    // `jc` is left col of current  diagonal tile.
    // `jn` is left col of next     diagonal tile; end of update.
    //         (I.e., `jn` is right + 1 col of current diagonal tile.)
    I jc = sweep + 1 + task * kd;
    I jn = std::min(jc + kd, n);
    I nc = jn - jc;

    if(task == 0)
    {
        // First task of the sweep brings column sweep to tridiagonal,
        // and applies reflector to diagonal block A{jc, jc}.
        if(yid == 0)
        {
            // Copy column sweep to shared memory, A[j+1:j+1+nc, s].
            for(I i = xid; i < nc; i += DIMX)
            {
                s_housev[i] = Aband[(idiag + 1 + i) + sweep * ldab];
            }

            // Generate Householder reflector.
            sb2st_larfg(xid, nc, s_housev, s_tau, (S*)s_work);

            // Copy Householder vector and tau to V,
            // and copy subdiagonal element to E.
            if(xid == 0)
            {
                Aband[idiag + 1 + sweep * ldab] = s_housev[0];
                assert(std::imag(s_housev[0]) == 0);
                E[sweep] = std::real(s_housev[0]);
                s_housev[0] = T(1);
                tau[vj] = s_tau;
            }
            // if V is initialized to Identity, don't need to store i=0.
            for(I i = xid; i < nc; i += DIMX)
            {
                V[vi + i + vj * ldv] = s_housev[i];
                if(xid > 0)
                {
                    Aband[idiag + 1 + i + sweep * ldab] = 0; // todo: only for clarity
                }
            }
        }
        __syncthreads();

        if(s_tau != 0)
        {
            // Apply H on both sides to diagonal block, A{i,i} := H^H A{i,i} H.
            // Using ldab-1 adjusts for band format.
            if constexpr(use_hemv)
            {
                sb2st_helarf(xid, yid, nc, s_housev, s_tau, Aband + idiag + (sweep + 1) * ldab,
                             ldab - 1, s_work);
            }
            else
            {
                sb2st_larf(xid, yid, rocblas_side_left, nc, nc, s_housev, conj(s_tau),
                           Aband + idiag + (sweep + 1) * ldab, ldab - 1, s_work);
                __syncthreads();
                sb2st_larf(xid, yid, rocblas_side_right, nc, nc, s_housev, s_tau,
                           Aband + idiag + (sweep + 1) * ldab, ldab - 1, s_work);
            }
        }
    }
    else
    {
        // Bulge chasing applies reflector from previous step to off-diagonal
        // block A{jc, jp}, creating a bulge, then creates reflector to bring
        // 1st column of bulge back to bandwidth kd, and applies reflector to
        // off-diagonal block A{jc, jp} and diagonal block A{jc, jc}.
        I jp = jc - kd;

        if(yid == 0)
        {
            // Copy previous task's Householder vector, vp, to shared memory.
            I vpi, vpj;
            get_v_index(n, kd, sweep, task - 1, vpi, vpj);
            for(I i = xid; i < kd; i += DIMX)
            {
                s_housev[i] = V[vpi + i + vpj * ldv];
            }
            if(xid == 0)
            {
                s_tau = tau[vpj];
            }
        }
        __syncthreads();

        // Apply vp on right to lower off-diagonal tile,
        // A{jc, jp} := A{jc, jp} H.
        if(s_tau != 0)
        {
            sb2st_larf(xid, yid, rocblas_side_right, nc, kd, s_housev, s_tau,
                       Aband + idiag + kd + jp * ldab, ldab - 1, s_work);
            __syncthreads();
        }

        if(nc > 1)
        {
            if(yid == 0)
            {
                // Copy 1st column of bulge to shared memory.
                for(I i = xid; i < nc; i += DIMX)
                {
                    s_housev[i] = Aband[idiag + kd + i + jp * ldab];
                }

                // Generate current Householder reflector, vc.
                sb2st_larfg(xid, nc, s_housev, s_tau, (S*)s_work);

                // Copy Householder vector and tau to column V,
                // and copy 1st element of larfg back to A.
                if(xid == 0)
                {
                    Aband[idiag + kd + jp * ldab] = s_housev[0];
                    s_housev[0] = T(1);
                    V[vi + vj * ldv] = s_housev[0];
                    tau[vj] = s_tau;
                }
                for(I i = xid + 1; i < nc; i += DIMX)
                {
                    V[vi + i + vj * ldv] = s_housev[i];
                    Aband[idiag + kd + i + jp * ldab] = T(0);
                }
            }
            __syncthreads();

            if(s_tau != 0)
            {
                // Apply vc on left of lower off-diagonal block, A{jc, jp+1} := H^H A{jc, jp+1}.
                // Skip 1st column that was eliminated above.
                sb2st_larf(xid, yid, rocblas_side_left, nc, kd - 1, s_housev, conj(s_tau),
                           Aband + idiag + kd - 1 + (jp + 1) * ldab, ldab - 1, s_work);
                __syncthreads();

                // Apply vc on left and right of diagonal, A{jc, jc} := H^H A{jc, jc} H.
                if constexpr(use_hemv)
                {
                    sb2st_helarf(xid, yid, nc, s_housev, s_tau, Aband + idiag + jc * ldab, ldab - 1,
                                 s_work);
                }
                else
                {
                    sb2st_larf(xid, yid, rocblas_side_left, nc, nc, s_housev, conj(s_tau),
                               Aband + idiag + jc * ldab, ldab - 1, s_work);
                    __syncthreads();

                    sb2st_larf(xid, yid, rocblas_side_right, nc, nc, s_housev, s_tau,
                               Aband + idiag + jc * ldab, ldab - 1, s_work);
                }
            }
        }

        // Copy conj of top row of A{jc, jp} to 1st col of A{jp, jc} to maintain
        // symmetry for next task.
        if(yid == 0)
        {
            for(I i = xid; i < kd - 1; i += DIMX)
            {
                Aband[idiag - (kd - 1) + i + jc * ldab]
                    = conj(Aband[idiag + (kd - 1) - i + (jp + 1 + i) * ldab]);
            }
        }
    }
}

//------------------------------------------------------------------------------
// SB2ST_HB2ST_STEP_KERNEL runs a single round with multiple sweeps in parallel.
// Run with 1 block in x, parallel_sweeps blocks in y, and batch_count blocks in z.
// Each thread block is DIMX x DIMY.
// (Batch is unused and untested.)
//
// Sweep i can begin execution when sweep i-1 has completed 2 rounds. That is,
// - Sweep 0 can start at round 0
// - Sweep 1 can start at round 2
//   ...
// - Sweep i can start at round 2*i
//   ...
// - Sweep n-1 can start at round 2*(n-1)
//
// Sweep n-1 is complete after 1 round, therefore the total number of rounds is
// 2*(n-1)+1.
//
//  n           Matrix dimension.
//  kd          Matrix bandwidth.
//  round       Index of round.
//  AAband      Band matrix. See sb2st_hb2st_task.
//  EE          Sub-diagonal.
//  VV          Array of Householder vectors.
//  TTau        Householder tau values.
//
// Requires shared memory of type T, size = kd + DIMY.
//
template <typename T, typename I, typename S>
ROCSOLVER_KERNEL void sb2st_hb2st_kernel(I n,
                                         I kd,
                                         I round,
                                         T* AAband,
                                         rocblas_stride shiftA,
                                         I ldab,
                                         rocblas_stride strideA,
                                         S* EE,
                                         rocblas_stride strideE,
                                         T* VV,
                                         I ldv,
                                         rocblas_stride strideV,
                                         T* TTau,
                                         rocblas_stride strideTau)
{
    const I xid = threadIdx.x;
    const I yid = threadIdx.y;
    const I sid = blockIdx.y;
    const I bid = blockIdx.z;

    // select batch instance
    T* Aband = load_ptr_batch<T>(AAband, bid, shiftA, strideA);
    T* V = load_ptr_batch<T>(VV, bid, 0, strideV);
    S* E = load_ptr_batch<S>(EE, bid, 0, strideE);
    T* tau = load_ptr_batch<T>(TTau, bid, 0, strideTau);

    // shared memory setup
    extern __shared__ char s_mem[];
    T* s_housev = reinterpret_cast<T*>(s_mem);
    T* s_work = s_housev + kd;

    // get sweep parameters
    I last_sweep = round / 2;
    I sweep = last_sweep - sid;
    I task = round - (2 * sweep);

    // execute sweep task
    sb2st_hb2st_task<T, I, S>(xid, yid, n, kd, sweep, task, Aband, ldab, E, V, ldv, tau, s_housev,
                              s_work);
}

//------------------------------------------------------------------------------
// Copies the final diagonal from the band matrix A to the vector D.
// Called with grid ( ceil( n / BS1 ), 1, batch_count ).
// Thread blocks are size BS1.
// (Batch is unused and untested.)
//
//  n           Matrix dimension.
//  AAband      n-by-n band matrix, shifted so diagonal is in row 0.
//  DD          On output, diagonal vector of length n.
//
template <typename T, typename I, typename S>
ROCSOLVER_KERNEL void sb2st_hb2st_copy_diag(I n,
                                            T* AAband,
                                            rocblas_stride shiftA,
                                            I ldab,
                                            rocblas_stride strideA,
                                            S* DD,
                                            rocblas_stride strideD)
{
    const I tid = blockIdx.x * blockDim.x + threadIdx.x;
    const I bid = blockIdx.z;

    if(tid < n)
    {
        // select batch instance
        T* Aband = load_ptr_batch<T>(AAband, bid, shiftA, strideA);
        S* D = load_ptr_batch<S>(DD, bid, 0, strideD);

        // copy diag
        D[tid] = std::real(Aband[tid * ldab]);
    }
}

//------------------------------------------------------------------------------
// hb2st does not require main memory workspace.
//
template <bool BATCHED, typename T, typename I, typename S>
void rocsolver_sb2st_hb2st_getMemorySize(const I n, const I kd, const I batch_count, size_t* size_work)
{
    *size_work = 0;
}

//------------------------------------------------------------------------------
// Checks hb2st arguments. See rocsolver_sb2st_hb2st_impl.
//
template <typename T, typename I, typename S>
rocblas_status rocsolver_sb2st_hb2st_argCheck(rocblas_handle handle,
                                              rocblas_fill uplo,
                                              const I n,
                                              const I kd,
                                              const I ldab,
                                              const I ldv,
                                              T Aband,
                                              S D,
                                              S E,
                                              T V,
                                              T tau,
                                              const I batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo == rocblas_fill_upper)
        return rocblas_status_not_implemented;

    // 2. invalid size
    if(n < 0 || kd < 1 || ldab < 3 * kd - 1 || ldv < 2 * kd - 1 || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // skip pointer check if quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_continue;

    // 3. invalid pointers
    if(!Aband || !D || !E || !V || !tau)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

//------------------------------------------------------------------------------
// Implements hb2st. See rocsolver_sb2st_hb2st_impl.
//
template <bool BATCHED, bool STRIDED, typename T, typename I, typename S, typename U>
rocblas_status rocsolver_sb2st_hb2st_template(rocblas_handle handle,
                                              rocblas_fill uplo,
                                              const I n,
                                              const I kd,
                                              U Aband,
                                              const rocblas_stride shiftA,
                                              const I ldab,
                                              const rocblas_stride strideA,
                                              S* D,
                                              const rocblas_stride strideD,
                                              S* E,
                                              const rocblas_stride strideE,
                                              U V,
                                              const I ldv,
                                              const rocblas_stride strideV,
                                              T* tau,
                                              const rocblas_stride strideTau,
                                              const I batch_count)
{
    ROCSOLVER_ENTER("sb2st_hb2st", "n:", n, "kd:", kd, "shiftA:", shiftA, "ldab:", ldab,
                    "ldv:", ldv, "bc:", batch_count);

    // quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    const T zero = 0;

    // Clear diagonals below sub-diagonal kd, where bulges will go.
    rocsolver_laset(handle, rocblas_fill_full, kd - 1, n, zero, zero, // opts
                    Aband, shiftA + 2 * kd, ldab, strideA, // Aband
                    batch_count);

    // Copy lower band to upper band.
    hermitianize_band(handle, n, kd - 1, Aband + shiftA, ldab, strideA, batch_count);

    // Set V = 0.
    // Ideally, set each Vk = I, but need to iterate over Vk.
    I nt = ceildiv(n - 1, kd);
    I nv_blocks = nt * (nt + 1) / 2;
    I nv = nv_blocks * kd;
    rocsolver_laset(handle, rocblas_fill_full, ldv, nv, zero, zero, // opts
                    V, 0, ldv, strideV, // V
                    batch_count);

    const hipDeviceProp_t* props = rocblas_internal_get_device_prop(handle);

    size_t s_mem_size_housev = sizeof(T) * kd;
    size_t s_mem_size_reduct = sizeof(T) * DIMY;
    size_t s_mem_size = s_mem_size_housev + s_mem_size_reduct;

    if(s_mem_size > props->sharedMemPerBlock)
    {
        return rocblas_status_internal_error;
    }

    // Sweep s starts in round r/2 and has ceil( (n - s - 1) / kd ) - 1 tasks,
    // so it finishes after round 2*s + ceil( (n - s - 1) / kd ) - 1.
    I sweep_begin = 0;
    I sweep_begin_finishes = 2 * sweep_begin + ceildiv(n - sweep_begin - 1, kd) - 1;
    I num_rounds = 2 * (n - 2) + 1;

    // execute sweeps
    for(I round = 0; round < num_rounds; ++round)
    {
        // Run sweeps in half-open interval [begin, ..., end).
        // Near the end, there are kd - 1 empty rounds, where a sweep has
        // finished but the next sweep hasn't started per these formulas;
        // skip those rounds.
        I sweep_end = I(round / 2) + 1;

        I parallel_sweeps = sweep_end - sweep_begin;
        if(parallel_sweeps > 0)
        {
            ROCSOLVER_LAUNCH_KERNEL((sb2st_hb2st_kernel<T, I, S>),
                                    dim3(1, parallel_sweeps, batch_count), dim3(DIMX, DIMY, 1),
                                    s_mem_size, stream, n, kd, round, Aband, shiftA, ldab, strideA,
                                    E, strideE, V, ldv, strideV, tau, strideTau);
        }
        if(round == sweep_begin_finishes)
        {
            sweep_begin += 1;
            sweep_begin_finishes = 2 * sweep_begin + ceildiv(n - sweep_begin - 1, kd) - 1;
        }
    }

    // copy diagonal
    I idiag = kd - 1;
    I copyblocks = ceildiv(n, BS1);
    ROCSOLVER_LAUNCH_KERNEL((sb2st_hb2st_copy_diag<T, I, S>), dim3(copyblocks, 1, batch_count),
                            dim3(BS1), 0, stream, n, Aband, shiftA + idiag, ldab, strideA, D,
                            strideD);

    return rocblas_status_success;
}

#undef DIMX
#undef DIMY

ROCSOLVER_END_NAMESPACE

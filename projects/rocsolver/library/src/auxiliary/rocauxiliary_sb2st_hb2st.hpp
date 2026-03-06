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

//------------------------------------------------------------------------------
// todo:should these be static to avoid one definition rule violations?
// or namespaced? or in some general utility header?
template <typename T, std::enable_if_t<! rocblas_is_complex<T>, int> = 0>
__device__ __inline__ T shift_left( T& value, int lane_delta )
{
    T r = value;
    r = __shfl_down( r, lane_delta );
    return r;
}

//------------------------------------------------------------------------------
template <typename T, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
__device__ __inline__ T shift_left( T& value, int lane_delta )
{
    using S = decltype( std::real( T{} ) );
    S r = value.real();
    S i = value.imag();
    r = __shfl_down( r, lane_delta );
    i = __shfl_down( i, lane_delta );
    return rocblas_complex_num<S>( r, i );
}

//------------------------------------------------------------------------------
// Generate Householder reflector.
// Must be called with one wave
// Compared to usual larfg, this passes x = [alpha; xhat] as single argument
// to make reduction simpler.
// If norm( xhat ) == 0, LAPACK sets tau = 0 and H = I,
// whereas this will set tau = 2 and H = [ -1 0 ].
//                                       [  0 I ]
//, std::enable_if_t<! rocblas_is_complex<T>, int> = 0>
template <typename T, typename I
__device__ void sb2st_larfg(
    const I xid, I n, T* x, T& tau, S* s_reduct )
{
    using S = decltype( std::real( T{} ) );

    const S one = 1;

    // dot reduction
    // was T, but should be real.
    S norm2 = 0;
    for (I i = xid; i < n; i += DIMX)
        norm2 += std::norm( x[i] );  // i.e., abs(xi)^2
    norm2 += shift_left( norm2, 16 );
    norm2 += shift_left( norm2, 8 );
    norm2 += shift_left( norm2, 4 );
    norm2 += shift_left( norm2, 2 );
    norm2 += shift_left( norm2, 1 );
    if (xid == 0)
        s_reduct[0] = norm2;
    // larfg is called with one wave, so __syncthreads is impossible & not needed.
    __threadfence_block();
    norm2 = s_reduct[0];

    S ar = real( alpha );
    S ai = imag( alpha );  // In real, ai = 0. Compiler can eliminate it.

    __shared__ T s_scale;

    // The way we do norm2 above, it already includes alpha.
    // In LAPACK, at this point norm is just x (== x[1:n]), excluding alpha (== x[0]).
    if (norm2 > 0 || ai > 0)
    {
        if (xid == 0)
        {
            S norm = alpha >= 0 ? -std::sqrt( norm2 ) : std::sqrt( norm2 );

            if constexpr (rocblas_is_complex<T>)
            {
                // scaling factor
                S r = (ar - norm) * (ar - norm) + ai * ai;
                S rr = (ar - norm) / r;
                S ri = -ai / r;
                s_scale = rocblas_complex_num<S>( rr, ri );

                // tau
                rr = (norm - ar) / norm;
                ri = -ai / norm;
                tau = rocblas_complex_num<S>( rr, ri );
            }
            else
            {
                s_scale = one / (alpha - norm);
                tau = (norm - alpha) / norm;
                alpha = norm;
            }
        }
        __syncthreads();  // for s_scale; was missing

        // scal x[1:n]
        for (I i = xid+1; i < n; i += DIMX)
            x[i] *= s_scale;
    }
    else
    {
        tau = 0;
    }
}

//------------------------------------------------------------------------------
/*
template <typename T, typename I, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
__device__ void sb2st_larfg(
    const I xid, I n, T* x, T& tau, S* s_reduct )
{
    using S = decltype( std::real( T{} ) );

    const S one = 1;

    // dot reduction
    // was T, but should be real.
    S norm2 = 0;
    for (I i = xid; i < n - 1; i += DIMX)
        norm2 += std::norm( x[i] );  // i.e., abs(xi)^2.
    norm2 += shift_left( norm2, 16 );
    norm2 += shift_left( norm2, 8 );
    norm2 += shift_left( norm2, 4 );
    norm2 += shift_left( norm2, 2 );
    norm2 += shift_left( norm2, 1 );
    if (xid == 0)
        s_reduct[0] = norm2;
    //__threadfence();  // not strong enough
    __syncthreads();  // was missing
    norm2 = s_reduct[0];
    // was: ... + std::norm( alpha );

    /// If we use `S` for real_t, and `s` for scale, can we rename one?
    /// Greatly dislike having variables differ only by case,
    /// esp. S vs. s. Hard to read.
    S ar = alpha.real();
    S ai = alpha.imag();
    __shared__ T s_scale;

    /// Why need to check ai, isn't it already incorporated into norm2?
    /// Ah, LAPACK doesn't add (ar, ai) into norm2 until after this statement.
    if (norm2 > 0 || ai > 0)
    {
        if (xid == 0)
        {
            S norm = ar >= 0 ? -std::sqrt( norm2 ) : std::sqrt( norm2 );

            // scaling factor
            /// If norm is T, this is doing complex arithmetic. norm should be S.
            S r = (ar - norm) * (ar - norm) + ai * ai;
            S rr = (ar - norm) / r;
            S ri = -ai / r;
            s_scale = rocblas_complex_num<S>( rr, ri );

            // tau
            rr = (norm - ar) / norm;
            ri = -ai / norm;
            tau = rocblas_complex_num<S>( rr, ri );

            // alpha
            alpha = norm;
        }
        __syncthreads();  // for s_scale; was missing

        // scal x[1:n]
        for (I i = xid+1; i < n; i += DIMX)
            x[i] *= s_scale;
    }
    else
    {
        tau = 0;
    }
}
*/

//------------------------------------------------------------------------------
// Apply H on left or right
// C := H C on left
// C := C H on right
// To apply H^H, pass in conj( tau ).
template <typename T, typename I>
__device__ void sb2st_larf(
    const I xid, const I yid, rocblas_side side, I m, I n,
    T* v, T tau, T* C, I ldc, T* s_reduct )
{
    if (side == rocblas_side_left)
    {
        for (I j = yid; j < n; j += DIMY)
        {
            // gemv reduction
            /// C = (I - tau v v^H) C = C - tau v (v^H C)
            /// wj = v^T * conj( Cj )
            /// Why not do wj = v^H * Cj, i.e., conj( v )?
            T value = 0;
            for (I i = xid; i < m; i += DIMX)
                value += conj( C[i + j * ldc] ) * v[i];
            value += shift_left( value, 16 );
            value += shift_left( value, 8 );
            value += shift_left( value, 4 );
            value += shift_left( value, 2 );
            value += shift_left( value, 1 );
            if (xid == 0)
                /// What about multiplying tau here?
                s_reduct[yid] = value;
            __threadfence();  // todo: need __syncthreads? This gets called by whole block.

            // ger
            /// Cj = Cj - tau v conj( wj ) = C[:,j] - tau v (v^H Cj)
            for (I i = xid; i < m; i += DIMX)
                C[i + j * ldc] -= tau * v[i] * conj( s_reduct[yid] );
        }
    }
    else
    {
        for (I i = yid; i < m; i += DIMY)
        {
            // gemv reduction
            /// C = C (I - tau v v^H) = C - tau C v v^H
            T value = 0;
            for (I j = xid; j < n; j += DIMX)
                value += C[i + j * ldc] * v[j];
            value += shift_left( value, 16 );
            value += shift_left( value, 8 );
            value += shift_left( value, 4 );
            value += shift_left( value, 2 );
            value += shift_left( value, 1 );
            if (xid == 0)
                /// What about multiplying tau here?
                s_reduct[yid] = value;
            __threadfence();

            // ger
            // Cj = Cj - tau wj v^H = Cj - tau (Cj v) v^H
            for (I j = xid; j < n; j += DIMX)
                C[i + j * ldc] -= tau * conj( v[j] ) * s_reduct[yid];
        }
    }
}

//------------------------------------------------------------------------------
__device__ inline void get_vindex(
    rocblas_int n,
    rocblas_int kd,
    rocblas_int sweep,
    rocblas_int task,
    rocblas_int& vi,
    rocblas_int& vj )
{
    rocblas_int k, kindex;
    rocblas_int nt = ceildiv( n, kd );  // todo: compute once & pass?
    vi = sweep % band;              // row within V trapezoid
    k  = sweep / band;              // block k
    kindex = k*nt - k*(k-1)/2;      // index of diagonal V{k,k} block
    vj = vi + (kindex + task)*kd;   // col within V
}

//------------------------------------------------------------------------------
template <typename T, typename S>
__device__ void sb2st_hb2st_sweep_step(
    const rocblas_int xid,
    const rocblas_int yid,
    rocblas_int n,
    rocblas_int kd,
    rocblas_int sweep,
    rocblas_int task,
    //rocblas_int sm_i,
    T* A,
    rocblas_int ldab,
    S* D,
    S* E,
    T* s_housev,
    T* s_reduct)
{
    const rocblas_int tid = xid + yid * DIMX;

    // row, col index for current Householder vector vc within V.
    rocblas_int vi, vj;
    get_vindex( n, kd, sweep, task, &vi, &vj );

    __shared__ T tau;

    //if (sm_i == s + 1)
    if (task == 0)
    {
        // first step of the sweep
        // `vp` is Householder vector generated in previous task.
        // `vc` is Householder vector generated in current  task.
        //
        // `jp` is left col of previous diagonal tile and current off-diagonal tile.
        // `jc` is left col of current  diagonal tile.
        // `jn` is left col of next     diagonal tile.
        rocblas_int jc = sweep + 1;
        rocblas_int jn = std::min( jc + kd, n );

        //rocblas_int sm_e = std::min( sm_i + kd, n );
        //rocblas_int su_i = sm_e;
        //rocblas_int su_e = std::min( su_i + kd, n );
        //
        //rocblas_int mm = sm_e - sm_i;

        rocblas_int nc = jn - jc;
        if (yid == 0)
        {
            // copy column s to shared memory, A[j+1:j+1+nc, s]
            for (rocblas_int i = xid; i < nc; i += DIMX)
                s_housev[i] = Aband[(idiag + 1 + i) + s * ldab];

            // generate Householder reflector
            sb2st_larfg( xid, nc, s_housev, tau, s_reduct );

            // Copy Householder vector and tau to V, and subdiagonal element to E.
            // todo: copy diagonal to D? A[sweep+1, sweep+1] is computed below in larfy.
            if (xid == 0)
            {
                // Bottom row of V stores tau.
                V[ldab - 1 + vj*ldv] = tau;  // was A[sm_i + s * ldab]
                E[s] = std::real( s_housev[0] );
                s_housev[0] = T( 1 );
            }
            // was starting from 1, for (i = 1 + xid; ...
            for (rocblas_int i = xid; i < nc; i += DIMX)
                V[vi + i + vj*ldv] = s_housev[i];  // A[(sm_i + i) + s * ldab]
        }
        __syncthreads();

        // apply Householder reflector
        if (tau != 0)
        {
            // todo: larfy:
            // hemv( C, v, work )
            // alpha = -0.5 tau dotc( work, v )
            // work -= alpha*v
            // C := C - tau*v*work^H - conj(tau)*work*v^H  // her2

            // Apply H on both sides to diagonal block, A{i,i} := H^H A{i,i} H.
            // Using ldab-1 adjusts for band format.
            sb2st_larf( xid, yid, rocblas_side_left, nc, nc, s_housev, conj( tau ),
                        A + idiag + (sweep + 1)*ldab, ldab-1, s_reduct );
            __syncthreads();
            sb2st_larf( xid, yid, rocblas_side_right, nc, nc, s_housev, tau,
                        A + idiag + (sweep + 1)*ldab, ldab-1, s_reduct);

            // todo: copy A[ idiag + (sweep + 1)*ldab ] to D[s+1]?

            /*
            // Apply H^H on left to diagonal block, A{i,i} := H^H A{i,i},
            // and upper off-diagonal block, A{i,i+1} = H^H A{i,i+1}.
            rocblas_int nn = su_e - sm_i;
            sb2st_larf( xid, yid, rocblas_side_left, mm, m, s_housev, conj( tau ),
                        A + sm_i + sm_i * ldab, ldab, s_reduct );
            __syncthreads();

            // Apply H^H on right to diagonal block, A{i,i} := A{i,i} H.
            sb2st_larf( xid, yid, rocblas_side_right, mm, mm, s_housev, tau,
                        A + sm_i + sm_i * ldab, ldab, s_reduct);

            // Copy & transpose upper off-diagonal block A{i,i+1}
            // to lower off-diagonal block A{i+1,i}.
            // Don't need to do this.
            nn = su_e - su_i;
            for (rocblas_int idx1d = tid; idx1d < nn * mm; idx1d += DIMX * DIMY)
            {
                rocblas_int i = su_i + idx1d % nn;
                rocblas_int j = sm_i + idx1d / nn;
                A[i + j * ldab] = conj( A[j + i * ldab] );
            }
            */
        }
    }
    else
    {
        // bulge chasing
        rocblas_int jc = sweep + 1 + task*kd;
        rocblas_int jp = jc - kd;
        rocblas_int jn = std::min( jc + kd, n );
        rocblas_int nc = jn -jc;

        /*
        rocblas_int sm_e = std::min( sm_i + kd, n );    // diag end
        rocblas_int su_i = sm_e;                        // upper begin
        rocblas_int su_e = std::min( su_i + kd, n );    // upper end
        rocblas_int sd_i = sm_i - kd;                   // lower begin
        rocblas_int sd_e = sm_i;                        // lower end

        rocblas_int mm = sm_e - sm_i;
        */

        if (yid == 0)
        {
            // Copy previous Householder vector, vp, to shared memory.
            for (rocblas_int i = xid; i < mm; i += DIMX)
                s_housev[i] = V[vi + i + (vj - kd)*ldab];
            if (xid == 0)
                tau = V[ldv - 1 + (vj - kd)*ldab];
        }
        __syncthreads();

        // Apply vp on right to lower off-diagonal tile,
        // A{jc, jp} := A{jc, jp} H.
        if (tau != 0)
        {
            sb2st_larf( xid, yid, rocblas_side_right, nc, kd, s_housev, conj( tau ),
                        A + sm_i + (sd_i + 1) * ldab, ldab, s_reduct );
            __syncthreads();
        }

        if (nc > 1)
        {
            if (yid == 0)
            {
                // copy column s to shared memory
                for (rocblas_int i = xid; i < mm; i += DIMX)
                    s_housev[i] = Aband[idiag + kd + i + jp*ldab];

                // generate Householder reflector
                sb2st_larfg( xid, mm, s_housev, tau, s_reduct );

                /// TODO: copy to V storage. The larfg kernel could do this during scaling.
                // copy Householder vector to column s of A,
                if (xid == 0)
                {
                    V[ldab - 1 + vc*ldv] = tau;  // was A[sm_i + s * ldab]
                    A[idiag + kd + jp*ldab] = s_housev[0];
                    s_housev[0] = T( 1 );
                    V[vi + vj*ldv] = s_housev[0];
                }
                // hmm... if I did i = xid; then it copies the whole s_housev,
                // but 0's whole column of A; need to preserve the top element
                // assigned above, A[idiag+kd, jp].
                // todo: use 2 loops?
                // V needs to be 0'd out, so maybe set Vk = I and don't set V[vi,vj] = 1 above?
                for (rocblas_int i = xid+1; i < nc; i += DIMX)
                {
                    V[vi + i + vj*ldv] = s_housev[i];
                    A[idiag + kd + i + jp*ldab] = T( 0 );
                }
            }
            __syncthreads();

            // apply Householder reflector
            if (tau != 0)
            {
                // Apply on left of lower off-diagonal, A{jc, jp} := H^H A{jc, jp}.
                sb2st_larf( xid, yid, rocblas_side_left, nc, kd, s_housev, conj( tau ),
                            A + idiag + kd + jp*ldab, ldab-1, s_reduct );
                __syncthreads();

                // Apply on left and right of diagonal, A{jc, jc} := H^H A{jc, jc} H.
                // todo: larfy
                sb2st_larf( xid, yid, rocblas_side_left, nc, nc, s_housev, conj( tau ),
                            A + idiag + jc*ldab, ldab-1, s_reduct );
                __syncthreads();

                sb2st_larf( xid, yid, rocblas_side_left, nc, nc, s_housev, tau,
                            A + idiag + jc*ldab, ldab-1, s_reduct );

                /*
                // Apply on left of (prev) off-diagonal and diagonal block, A{i, i-1:i} = H^H A{i, i-1:i}.
                rocblas_int nn = su_e - sd_i - 1;
                sb2st_larf( xid, yid, rocblas_side_left, mm, nn, s_housev, conj( tau ),
                            A + sm_i + (sd_i + 1) * ldab, ldab, s_reduct );
                __syncthreads();
                // Apply on right of diagonal and (next) off-diagonal block, A{i:i+1, i} = A{i:i+1, i} H.
                /// Isn't this a race condition with the thread block updating the
                /// left off-diagonal of next block? Or are they not packing the
                /// tasks as tightly, skipping every other task to avoid races?
                sb2st_larf( xid, yid, rocblas_side_right, mm, mm, s_housev, tau,
                            A + sm_i + sm_i * ldab, ldab, s_reduct);

                /// I think there's no reason to copy transpose blocks.
                /// Are they ever used for anything?
                // copy transpose blocks
                nn = su_e - su_i;
                for (rocblas_int idx1d = tid; idx1d < nn * mm; idx1d += DIMX * DIMY)
                {
                    rocblas_int i = su_i + idx1d % nn;
                    rocblas_int j = sm_i + idx1d / nn;
                    A[i + j * ldab] = conj( A[j + i * ldab] );
                }
                nn = sd_e - sd_i;
                for (rocblas_int idx1d = tid; idx1d < nn * mm; idx1d += DIMX * DIMY)
                {
                    rocblas_int i = sd_i + idx1d % nn;
                    rocblas_int j = sm_i + idx1d / nn;
                    A[i + j * ldab] = conj( A[j + i * ldab] );
                }
                */
            }
        }
    }
}

//------------------------------------------------------------------------------
/* SB2ST_HB2ST_STEP_KERNEL runs a single round from multiple sweeps in parallel. Run with
   sweeps_in_parallel thread blocks in y and batch_count thread blocks in z.

   Sweep i can begin execution when sweep i-1 has completed 3 rounds. That is,
   - Sweep 0 can start at round 0
   - Sweep 1 can start at round 3
   ...
   - Sweep i can start at round 3*i
   ...
   - Sweep n-1 can start at round 3*(n-1)

   Sweep n-1 is complete after 1 round, therefore the total number of rounds is 3*(n-1)+1 */
template <typename T, typename S>
ROCSOLVER_KERNEL void sb2st_hb2st_round_kernel(
    rocblas_int n,
    rocblas_int kd,
    rocblas_int round,
    T* AAband,
    rocblas_stride shiftA,
    rocblas_int ldab,
    rocblas_stride strideA,
    T* VV,
    rocblas_int ldv,
    rocblas_stride strideV,
    S* EE,
    rocblas_stride strideE)
{
    const rocblas_int xid = threadIdx.x;
    const rocblas_int yid = threadIdx.y;
    const rocblas_int sid = blockIdx.y;
    const rocblas_int bid = blockIdx.z;

    /// hmmm... SB2ST_HB2ST_MAX_THDS isn't defined anywhere.
    /// I guess it should be == 1.
    assert( blockDim.x == SB2ST_HB2ST_MAX_THDS );

    // select batch instance
    T* Aband = load_ptr_batch<T>( AAband, bid, shiftA, strideA );
    T* V = load_ptr_batch<S>( VV, bid, 0, strideV );
    S* E = load_ptr_batch<S>( EE, bid, 0, strideE );

    // shared memory setup
    extern __shared__ double s_mem[];
    T* s_housev = reinterpret_cast<T*>(s_mem);
    T* s_reduct = reinterpret_cast<T*>(s_housev + kd);

    // get sweep parameters
    rocblas_int last_sweep = round / 3;
    rocblas_int sweep = last_sweep - sid;
    rocblas_int step = round - (2 * sweep);
    rocblas_int sm_i = sweep + 1 + step_in_sweep * kd;

    if (sweep < 0 || sm_i >= n)
        return;

    // execute sweep step
    sb2st_hb2st_sweep_step<T, S>(
        xid, yid, n, kd, sweep, sm_i, Aband, ldab, V, ldv, E,
        s_housev, s_reduct );
}

//------------------------------------------------------------------------------
/*
template <typename T, typename S>
ROCSOLVER_KERNEL void sb2st_hb2st_copy_diag(
    rocblas_int n,
    T* AAband,
    rocblas_stride shiftA,
    rocblas_int ldab,
    rocblas_stride strideA,
    S* DD,
    rocblas_stride strideD)
{
    const rocblas_int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const rocblas_int bid = blockIdx.z;

    // select batch instance
    T* Aband = load_ptr_batch<T>( AAband, bid, shiftA, strideA );
    S* D = load_ptr_batch<S>( DD, bid, 0, strideD );

    // copy diag
    if (tid < n)
        D[tid] = std::real( A[tid + tid * ldab] );
}
*/

//------------------------------------------------------------------------------
template <bool BATCHED, typename T, typename S>
void rocsolver_sb2st_hb2st_getMemorySize(
    const rocblas_int n,
    const rocblas_int kd,
    const rocblas_int batch_count,
    size_t* size_work)
{
    if (n <= 1)
    {
        *size_work = 0;
        return;
    }

    *size_work = sizeof( T ) * (3 * kd) * batch_count;
}

//------------------------------------------------------------------------------
template <typename T, typename S>
rocblas_status rocsolver_sb2st_hb2st_argCheck(
    rocblas_handle handle,
    const rocblas_int n,
    const rocblas_int kd,
    const rocblas_int ldab,
    const rocblas_int ldv,
    T Aband,
    S D,
    S E,
    T V,
    const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if (n < 0 || kd < 0 || ldab < 3*kd + 1 || ldv < 3*kd)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if (rocblas_is_device_memory_size_query( handle ))
        return rocblas_status_continue;

    // 3. invalid pointers
    if ((n > 0 && ! A) || (n > 0 && ! D) || (n > 0 && ! E) || (n > 0 && ! V))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

//------------------------------------------------------------------------------
template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
rocblas_status rocsolver_sb2st_hb2st_template(
    rocblas_handle handle,
    const rocblas_int n,
    const rocblas_int kd,
    U Aband,
    const rocblas_stride shiftA,
    const rocblas_int ldab,
    const rocblas_stride strideA,
    U V,
    const rocblas_int ldv,
    const rocblas_stride strideV,
    S* D,
    const rocblas_stride strideD,
    S* E,
    const rocblas_stride strideE,
    const rocblas_int batch_count)
{
    ROCSOLVER_ENTER( "sb2st_hb2st", "n:", n, "kd:", kd, "shiftA:", shiftA,
                     "ldab:", ldab, "ldv:", ldv, "bc:", batch_count );

    // quick return
    if (n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream( handle, &stream );

    // quick return for n = 1 (scalar case)
    /// Nothing to do: already real (tri)diagonal. Don't need to solve eig, just
    /// trd. Maybe copy A => D, set E = 0, set tau = 0 (identity).
    // Or just let the algorithm work. Who solves n=1 case? Does it need to be fast?
    /*
    if (n == 1)
    {
        rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
        dim3 gridReset( blocksReset, 1, 1 );
        dim3 threadsReset( BS1, 1, 1 );

        /// TODO: rename `scalar_case`, perhaps `stev_scalar_case`?
        ROCSOLVER_LAUNCH_KERNEL(
            scalar_case<T>,
            gridReset, threadsReset, 0, stream,
            rocblas_evect_none, A, strideA, D, strideD, batch_count);
        return rocblas_status_success;
    }
    */

    // rocblas_pointer_mode old_mode;
    // rocblas_get_pointer_mode( handle, &old_mode );
    // rocblas_set_pointer_mode( handle, rocblas_pointer_mode_host );

    // Is this slow?
    int device;
    HIP_CHECK( hipGetDevice( &device ) );
    hipDeviceProp_t props;
    HIP_CHECK( hipGetDeviceProperties( &props, device ) );

    size_t s_mem_size_housev = sizeof( T ) * kd;
    size_t s_mem_size_reduct = sizeof( T ) * DIMY;
    size_t s_mem_size = s_mem_size_housev + s_mem_size_reduct;

    // What about just launching kernel and checking error for sharedMem exceeded?
    if (s_mem_size > props.sharedMemPerBlock)
    {
        return rocblas_status_internal_error;
    }

    /// Must be max. steps per sweep, and max. sweeps in parallel?
    const rocblas_int max_steps_per_sweep = ceildiv( n - 1, kd ) - 1;
    const rocblas_int max_parallel_sweeps = ceildiv( max_steps_per_sweep, 2 );  // ???
    const rocblas_int num_rounds = 2*(n - 2) + 1;

    // execute sweeps
    for (rocblas_int round = 0; round < num_rounds; round++)
    {
        ROCSOLVER_LAUNCH_KERNEL(
            sb2st_hb2st_round_kernel<T>,
            dim3( 1, sweeps_in_parallel, batch_count ),
            dim3( DIMX, DIMY, 1 ), s_mem_size, stream,
            n, kd, round,
            Aband, shiftA, ldab, strideA,
            D, strideD, E, strideE,
            V, ldv, strideV );
    }

    /// Why copy diagonal? Need both D and E. Ah, E set during kernel.
    /// Why not set D during kernel, too?
    // copy diagonal
    // can we call BLAS copy( Aband[idiag, 0], ldab, D, 1 )?
    const rocblas_int copyblocks = ceildiv( n, BS1 );
    ROCSOLVER_LAUNCH_KERNEL(
        (sb2st_hb2st_copy_diag<T>),
        dim3( copyblocks, 1, batch_count ), dim3( BS1 ), 0, stream,
        n, idiag, Aband, shiftA, ldab, strideA, D, strideD );

    // rocblas_set_pointer_mode( handle, old_mode );

    return rocblas_status_success;
}

#undef DIMX
#undef DIMY

ROCSOLVER_END_NAMESPACE

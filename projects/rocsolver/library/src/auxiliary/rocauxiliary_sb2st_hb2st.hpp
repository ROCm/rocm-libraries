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
#include "laset.hpp"
#include "print_matrix.hpp"

ROCSOLVER_BEGIN_NAMESPACE

// Number of threads in x and y
// Reductions in larfg and larf must be updated if DIMX is changed
#define DIMX 32
#define DIMY 32

//------------------------------------------------------------------------------
// Generate Householder reflector.
// Must be called with one wave; this assumes threads are synchronized and does
// not do __syncthreads. It does __threadfence_block where __syncthreads would
// normally be required.
// Compared to usual larfg, this passes x = [alpha; xhat] as single argument
// to make reduction simpler.
// If norm( xhat ) == 0, LAPACK sets tau = 0 and H = I,
// whereas this will set tau = 2 and H = [ -1 0 ].
//                                       [  0 I ]
//, std::enable_if_t<! rocblas_is_complex<T>, int> = 0>
template <typename T, typename I, typename S=decltype(std::real(T{}))>
__device__ void sb2st_larfg(
    const I xid, I n, T* x, T& tau, S* s_work )
{
    // Reduction assumes this DIMX.
    static_assert( DIMX == 32 );

    const S one = 1;

    // norm reduction
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
        s_work[0] = norm2;
    __threadfence_block();
    norm2 = s_work[0];

    S alpha_r = std::real( x[0] );
    S alpha_i = std::imag( x[0] );  // In real, alpha_i = 0. Compiler can eliminate it.

    __shared__ T s_scale;

    // The way we do norm2 above, it already includes alpha.
    // In LAPACK, at this point norm is just x (== x[1:n]), excluding alpha (== x[0]).
    if (norm2 > 0 || alpha_i > 0)
    {
        if (xid == 0)
        {
            S norm = alpha_r >= 0 ? -std::sqrt( norm2 ) : std::sqrt( norm2 );

            if constexpr (rocblas_is_complex<T>)
            {
                // scaling factor
                S r = (alpha_r - norm) * (alpha_r - norm) + alpha_i * alpha_i;
                S rr = (alpha_r - norm) / r;
                S ri = -alpha_i / r;
                s_scale = rocblas_complex_num<S>( rr, ri );

                // tau
                rr = (norm - alpha_r) / norm;
                ri = -alpha_i / norm;
                tau = rocblas_complex_num<S>( rr, ri );
            }
            else
            {
                s_scale = one / (alpha_r - norm);
                tau = (norm - alpha_r) / norm;
            }

            x[0] = norm;
        }
        __threadfence_block();  // for s_scale; was missing

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
// Apply H on left or right of C:
// C := H C if on left,
// C := C H if on right.
// To apply H^H, pass in conj( tau ).
template <typename T, typename I>
__device__ void sb2st_larf(
    const I xid, const I yid, rocblas_side side, I m, I n,
    T* v, T tau, T* C, I ldc, T* s_work )
{
    // Reductions assume this DIMX.
    static_assert( DIMX == 32 );

    if (side == rocblas_side_left)
    {
        for (I j = yid; j < n; j += DIMY)
        {
            // gemv reduction
            // C = (I - tau v v^H) C = C - tau v (v^H C)
            // w = C^H v
            T value = 0;
            for (I i = xid; i < m; i += DIMX)
                value += conj( C[i + j * ldc] ) * v[i];
            value += shift_left( value, 16 );
            value += shift_left( value, 8 );
            value += shift_left( value, 4 );
            value += shift_left( value, 2 );
            value += shift_left( value, 1 );
            if (xid == 0)
                /// What about multiplying conj(tau) here?
                s_work[yid] = value;
            __threadfence_block();  // threads are sync'd in one wavefront

            // ger
            // Cj = Cj - tau v conj( wj ) = C[:,j] - tau v (v^H Cj)
            for (I i = xid; i < m; i += DIMX)
                C[i + j * ldc] -= tau * v[i] * conj( s_work[yid] );
        }
    }
    else
    {
        for (I i = yid; i < m; i += DIMY)
        {
            // gemv reduction
            // C = C (I - tau v v^H) = C - tau (C v) v^H
            // w = C v
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
                s_work[yid] = value;
            __threadfence_block();  // threads are sync'd in one wavefront

            // ger
            // Cj = Cj - tau wj v^H = Cj - tau (Cj v) v^H
            for (I j = xid; j < n; j += DIMX)
                C[i + j * ldc] -= tau * conj( v[j] ) * s_work[yid];
        }
    }
}

//------------------------------------------------------------------------------
// Apply H on left and right of Hermitian block:
// C := H^H C H
// Assumes the whole Hermitian block is set, both upper and lower.
// (In LAPACK, this is larfy, which doesn't fit well into LAPACK's naming
// conventions. I guess y comes from sy.)
template <typename T, typename I>
__device__ void sb2st_helarf(
    const I xid, const I yid, rocblas_side side, I n,
    T* v, T tau, T* C, I ldc, T* s_work )
{
    // Reductions assume this DIMX.
    static_assert( DIMX == 32 );

    // gemv/hemv: w = C * v
    for (I j = yid; j < n; j += DIMY)
    {
        // gemv reduction
        /// C = (I - tau v v^H) C = C - tau v (v^H C)
        /// wj = v^T * conj( Cj )
        /// Why not do wj = v^H * Cj, i.e., conj( v )?
        T value = 0;
        for (I i = xid; i < n; i += DIMX)
            value += C[i + j*ldc] * v[i];
        value += shift_left( value, 16 );
        value += shift_left( value, 8 );
        value += shift_left( value, 4 );
        value += shift_left( value, 2 );
        value += shift_left( value, 1 );
        if (xid == 0)
            /// What about multiplying tau here?
            s_work[yid] = value;
    }
    __syncthreads();

    // w = w - (0.5 tau w^H v) v
    // dot reduction: alpha = 0.5 tau w^H v
    __shared__ T s_alpha;
    if (yid == 0)
    {
        T value = 0;
        for (I i = xid; i < n; i += DIMX)
            value += conj( s_work[xid] ) * v[ xid ];
        value += shift_left( value, 16 );
        value += shift_left( value, 8 );
        value += shift_left( value, 4 );
        value += shift_left( value, 2 );
        value += shift_left( value, 1 );
        if (xid == 0)
            s_alpha = 0.5 * tau * value;
    }
    __syncthreads();

    // axpy: w = w - alpha v
    if (yid == 0)
    {
        for (I i = xid; i < n; i += DIMX)
        {
            s_work[xid] -= s_alpha*v[xid];
        }
    }
    __syncthreads();

    // ger2/her2: C := C - tau v w^H - conj(tau) w v^H
    for (I j = yid; j < n; j += DIMY)
    {
        for (I i = xid; i < n; i += DIMX)
        {
            C[i + j*ldc] -= tau * v[i] * conj( s_work[yid] )
                + conj( tau ) * s_work[yid] * conj( v[i] );
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
    vi = sweep % kd;               // row within V trapezoid
    k  = sweep / kd;               // block k
    kindex = k*nt - k*(k-1)/2;     // index of diagonal V{k,k} block
    vj = vi + (kindex + task)*kd;  // col within V
}

//------------------------------------------------------------------------------
template <typename T, typename S>
__device__ void sb2st_hb2st_task(
    const rocblas_int xid,
    const rocblas_int yid,
    rocblas_int n,
    rocblas_int kd,
    rocblas_int sweep,
    rocblas_int task,
    T* Aband,
    rocblas_int ldab,
    S* E,
    T* V,
    rocblas_int ldv,
    T* s_housev,
    T* s_work)
{
    const rocblas_int tid = xid + yid * DIMX;

    rocblas_int idiag = kd - 1;

    // row, col index for current Householder vector vc within V.
    rocblas_int vi, vj;
    get_vindex( n, kd, sweep, task, vi, vj );

    __shared__ T s_tau;

    // `vp` is Householder vector generated in previous task.
    // `vc` is Householder vector generated in current  task.
    //
    // `jp` is left col of previous diagonal tile and current off-diagonal tile.
    // `jc` is left col of current  diagonal tile.
    // `jn` is left col of next     diagonal tile; end of update.
    //         (I.e., `jn` is right + 1 col of current diagonal tile.)
    rocblas_int jc = sweep + 1 + task*kd;
    rocblas_int jn = std::min( jc + kd, n );
    rocblas_int nc = jn - jc;
    assert( nc > 0 );

    if (task == 0)
    {
        // First task of the sweep brings column sweep to tridiagonal,
        // and applies reflector to diagonal block A{jc, jc}.
        if (yid == 0)
        {
            // Copy column sweep to shared memory, A[j+1:j+1+nc, s].
            for (rocblas_int i = xid; i < nc; i += DIMX)
                s_housev[i] = Aband[(idiag + 1 + i) + sweep*ldab];

            // Generate Householder reflector.
            sb2st_larfg( xid, nc, s_housev, s_tau, (S*) s_work );

            // Copy Householder vector and tau to V,
            // and copy subdiagonal element to E.
            if (xid == 0)
            {
                // Bottom row of V stores tau.
                // todo: if desired, save s_housev[0] back to Aband as well.
                Aband[idiag + 1 + sweep*ldab] = s_housev[0];
                assert( std::imag( s_housev[0] ) == 0 );
                E[sweep] = std::real( s_housev[0] );
                s_housev[0] = T( 1 );
                V[ldv - 1 + vj*ldv] = s_tau;
            }
            // was: starting from 1, for (i = 1 + xid; ...
            // if V is initialized to Identity, don't need to store i=0.
            for (rocblas_int i = xid; i < nc; i += DIMX)
            {
                V[vi + i + vj*ldv] = s_housev[i];
                if (xid > 0)
                    Aband[idiag + 1 + i + sweep*ldab] = 0;  // todo: only for clarity
            }
        }
        __syncthreads();

        #if 1
            if (s_tau != 0)
            {
                // Apply H on both sides to diagonal block, A{i,i} := H^H A{i,i} H.
                // Using ldab-1 adjusts for band format.
                #if 0
                    sb2st_helarf( xid, yid, rocblas_side_left, nc, s_housev, s_tau,
                                Aband + idiag + (sweep + 1)*ldab, ldab-1, s_work );
                #else
                    sb2st_larf( xid, yid, rocblas_side_left, nc, nc, s_housev, conj( s_tau ),
                                Aband + idiag + (sweep + 1)*ldab, ldab-1, s_work );
                    __syncthreads();
                    sb2st_larf( xid, yid, rocblas_side_right, nc, nc, s_housev, s_tau,
                                Aband + idiag + (sweep + 1)*ldab, ldab-1, s_work );
                #endif

                // todo: copy A[ idiag + (sweep + 1)*ldab ] to D[s+1]?
            }
        #endif
    }
    else
    {
        // Bulge chasing applies reflector from previous step to off-diagonal
        // block A{jc, jp}, creating a bulge, then creates reflector to bring
        // 1st column of bulge back to bandwidth kd, and applies reflector to
        // off-diagonal block A{jc, jp} and diagonal block A{jc, jc}.
        rocblas_int jp = jc - kd;

        if (yid == 0)
        {
            // Copy previous Householder vector, vp, to shared memory.
            for (rocblas_int i = xid; i < kd; i += DIMX)
                s_housev[i] = V[vi + i + (vj - kd)*ldv];
            if (xid == 0)
                s_tau = V[ldv - 1 + (vj - kd)*ldv];
        }
        __syncthreads();

        // Apply vp on right to lower off-diagonal tile,
        // A{jc, jp} := A{jc, jp} H.
        if (s_tau != 0)
        {
            sb2st_larf( xid, yid, rocblas_side_right, nc, kd, s_housev, s_tau,
                        Aband + idiag + kd + jp*ldab, ldab-1, s_work );
            __syncthreads();
        }

        #if 1
            if (nc > 1)
            {
                if (yid == 0)
                {
                    // Copy 1st column of bulge to shared memory.
                    for (rocblas_int i = xid; i < nc; i += DIMX)
                        s_housev[i] = Aband[idiag + kd + i + jp*ldab];

                    // Generate current Householder reflector, vc.
                    sb2st_larfg( xid, nc, s_housev, s_tau, (S*) s_work );

                    // Copy Householder vector and tau to column V,
                    // and copy 1st element of larfg back to A.
                    if (xid == 0)
                    {
                        Aband[idiag + kd + jp*ldab] = s_housev[0];
                        s_housev[0] = T( 1 );
                        V[vi + vj*ldv] = s_housev[0];
                        V[ldv - 1 + vj*ldv] = s_tau;
                    }
                    // hmm... if I did i = xid; then it copies the whole s_housev,
                    // but 0's whole column of A; need to preserve the top element
                    // assigned above, A[idiag+kd, jp].
                    // todo: use 2 loops?
                    // V needs to be 0'd out, so maybe set Vk = I and don't set V[vi,vj] = 1 above?
                    for (rocblas_int i = xid+1; i < nc; i += DIMX)
                    {
                        V[vi + i + vj*ldv] = s_housev[i];
                        Aband[idiag + kd + i + jp*ldab] = T( 0 );
                    }
                }
                __syncthreads();

                if (s_tau != 0)
                {
                    // Apply vc on left of lower off-diagonal block, A{jc, jp+1} := H^H A{jc, jp+1}.
                    // Skip 1st column that was eliminated above.
                    sb2st_larf( xid, yid, rocblas_side_left, nc, kd-1, s_housev, conj( s_tau ),
                                Aband + idiag + kd - 1 + (jp + 1)*ldab, ldab-1, s_work );
                    __syncthreads();

                    // Apply vc on left and right of diagonal, A{jc, jc} := H^H A{jc, jc} H.
                    #if 0
                        sb2st_helarf( xid, yid, rocblas_side_left, nc, s_housev, s_tau,
                                    Aband + idiag + jc*ldab, ldab-1, s_work );
                    #else
                        sb2st_larf( xid, yid, rocblas_side_left, nc, nc, s_housev, conj( s_tau ),
                                    Aband + idiag + jc*ldab, ldab-1, s_work );
                        __syncthreads();

                        sb2st_larf( xid, yid, rocblas_side_right, nc, nc, s_housev, s_tau,
                                    Aband + idiag + jc*ldab, ldab-1, s_work );
                    #endif
                }
            }
        #endif

        // Copy conj of top row of A{jc, jp} to 1st col of A{jp, jc} to maintain
        // symmetry for next task.
        if (yid == 0)
        {
            for (rocblas_int i = xid; i < kd-1; i += DIMX)
            {
                Aband[ idiag - (kd - 1) + i + jc*ldab ]
                    = conj( Aband[ idiag + (kd - 1) - i + (jp + 1 + i)*ldab ] );
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
    S* EE,
    rocblas_stride strideE,
    T* VV,
    rocblas_int ldv,
    rocblas_stride strideV )
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
    T* V = load_ptr_batch<T>( VV, bid, 0, strideV );
    S* E = load_ptr_batch<S>( EE, bid, 0, strideE );

    // shared memory setup
    extern __shared__ double s_mem[];
    T* s_housev = reinterpret_cast<T*>(s_mem);
    T* s_work   = reinterpret_cast<T*>(s_housev + kd);

    // get sweep parameters
    rocblas_int last_sweep = round / 2;
    rocblas_int sweep = last_sweep - sid;
    rocblas_int task = round - (2 * sweep);
    assert( task >= 0 );
    assert( sweep >= 0 );

    //#define ONLY_SWEEP_0 1  // passes
    #if ONLY_SWEEP_0
        if (sweep > 0)
            return;
    #endif

    // execute sweep task
    sb2st_hb2st_task<T, S>(
        xid, yid, n, kd, sweep, task, Aband, ldab, E, V, ldv,
        s_housev, s_work );
}

//------------------------------------------------------------------------------
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

    if (tid < n)
    {
        // select batch instance
        T* Aband = load_ptr_batch<T>( AAband, bid, shiftA, strideA );
        S* D = load_ptr_batch<S>( DD, bid, 0, strideD );

        // copy diag
        D[tid] = std::real( Aband[tid*ldab] );
    }
}

//------------------------------------------------------------------------------
template <bool BATCHED, typename T, typename S>
void rocsolver_sb2st_hb2st_getMemorySize(
    const rocblas_int n,
    const rocblas_int kd,
    const rocblas_int batch_count,
    size_t* size_work)
{
    *size_work = 0;
}

//------------------------------------------------------------------------------
template <typename T, typename S>
rocblas_status rocsolver_sb2st_hb2st_argCheck(
    rocblas_handle handle,
    rocblas_fill uplo,
    const rocblas_int n,
    const rocblas_int kd,
    const rocblas_int ldab,
    const rocblas_int ldv,
    // why are these T and S instead of T* and S* ?
    T Aband,
    S D,
    S E,
    T V,
    const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if (uplo == rocblas_fill_upper)
        return rocblas_status_not_implemented;

    // 2. invalid size
//printf( "%s: n %d, kd %d, ldab %d, ldv %d\n", __func__, n, kd, ldab, ldv );
    if (n < 0 || kd < 0 || ldab < 3*kd - 1 || ldv < 3*kd)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if (rocblas_is_device_memory_size_query( handle ))
        return rocblas_status_continue;

    // 3. invalid pointers
    // why not: if (n > 0 && (! Aband || ! D || ! E || ! V))
    if ((n > 0 && ! Aband) || (n > 0 && ! D) || (n > 0 && ! E) || (n > 0 && ! V))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

//------------------------------------------------------------------------------
template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
rocblas_status rocsolver_sb2st_hb2st_template(
    rocblas_handle handle,
    rocblas_fill uplo,
    const rocblas_int n,
    const rocblas_int kd,
    // where is this U instead of T* ?
    U Aband,
    const rocblas_stride shiftA,
    const rocblas_int ldab,
    const rocblas_stride strideA,
    S* D,
    const rocblas_stride strideD,
    S* E,
    const rocblas_stride strideE,
    U V,
    const rocblas_int ldv,
    const rocblas_stride strideV,
    const rocblas_int batch_count)
{
    ROCSOLVER_ENTER( "sb2st_hb2st", "n:", n, "kd:", kd, "shiftA:", shiftA,
                     "ldab:", ldab, "ldv:", ldv, "bc:", batch_count );

    // quick return
    if (n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream( handle, &stream );

    const T zero = 0;

    // Set V = 0.
    // Ideally, set each Vk = I, but need to iterate over Vk.
    // Strided batch laset, with stride = kd*ldv?
    rocblas_int nt = ceildiv( n, kd );
    rocblas_int nv_blocks = nt*(nt + 1)/2;
    rocblas_int nv = nv_blocks*kd;
    rocblas_stride shiftV = 0;
    laset( handle, 'g', 3*kd, nv, zero, zero, V, shiftV, ldv, strideV,
           batch_count );

    print_matrix( "dA_in", ldab, n, Aband, ldab, 6 );
    print_matrix( "dV_init", ldv, nv, V, ldv, 6 );

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

    // Sweep s starts in round r/2 and has ceil( (n - s - 1) / kd ) - 1 tasks,
    // so it finishes after round 2*s + ceil( (n - s - 1) / kd ) - 1.
    rocblas_int sweep_begin = 0;
    rocblas_int sweep_begin_finishes
        = 2*sweep_begin + ceildiv( n - sweep_begin - 1, kd ) - 1;
    rocblas_int num_rounds = 2*(n - 2) + 1;

    // execute sweeps
    for (rocblas_int round = 0; round < num_rounds; ++round)
    {
        // Run sweeps in half-open interval [begin, ..., end).
        // Near the end, there are kd - 1 empty rounds, where a sweep has
        // finished but the next sweep hasn't started per these formulas;
        // skip those rounds.
        rocblas_int sweep_end = rocblas_int( round / 2 ) + 1;

        //#define ONLY_SWEEP_0 1
        //#define ONLY_TASK_0 1     passes larfg, passes larf2x
        //#define ONLY_TASK_01 1
        #if ONLY_TASK_0
            // Run all sweeps, task 0. See also below.
            sweep_begin = sweep_end - 1;
        #elif ONLY_TASK_01
            // Run all sweeps, task 0. See also below.
            sweep_begin_finishes = 2*sweep_begin + std::min( 2, ceildiv( n - sweep_begin - 1, kd ) - 1 );
        #endif
        rocblas_int parallel_sweeps = sweep_end - sweep_begin;
        printf( "# round %d, sweeps %d : %d, parallel sweeps %d\n",
                round, sweep_begin, sweep_end, parallel_sweeps );
        if (parallel_sweeps > 0)
        {
            ROCSOLVER_LAUNCH_KERNEL(
                sb2st_hb2st_round_kernel<T>,
                dim3( 1, parallel_sweeps, batch_count ),
                dim3( DIMX, DIMY, 1 ), s_mem_size, stream,
                n, kd, round,
                Aband, shiftA, ldab, strideA,
                E, strideE,
                V, ldv, strideV );

            std::string lbl = "dA" + std::to_string( round );
            print_matrix( lbl.c_str(), ldab, n, Aband, ldab, 6 );

            lbl = "dV" + std::to_string( round );
            print_matrix( lbl.c_str(), ldv, nv, V, ldv, 6 );
        }
        if (round == sweep_begin_finishes)
        {
            sweep_begin += 1;
            sweep_begin_finishes
                = 2*sweep_begin + ceildiv( n - sweep_begin - 1, kd ) - 1;
        }

        #if ONLY_TASK_0
            // Run all sweeps, task 0. See also above.
            // Skip odd rounds.
            ++round;
        #endif
    }

    // copy diagonal
    // todo: can we call BLAS copy( Aband[idiag, 0], ldab, D, 1 )?
    // todo: should this be done in sb2st_hb2st_task when E is set?
    rocblas_int idiag = kd - 1;
    rocblas_int copyblocks = ceildiv( n, BS1 );
    ROCSOLVER_LAUNCH_KERNEL(
        (sb2st_hb2st_copy_diag<T>),
        dim3( copyblocks, 1, batch_count ), dim3( BS1 ), 0, stream,
        n, Aband, shiftA + idiag, ldab, strideA,
        D, strideD );

    // rocblas_set_pointer_mode( handle, old_mode );

    return rocblas_status_success;
}

#undef DIMX
#undef DIMY

ROCSOLVER_END_NAMESPACE

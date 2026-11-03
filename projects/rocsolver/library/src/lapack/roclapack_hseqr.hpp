/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.12.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2023
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "auxiliary/rocauxiliary_lahqr.hpp"
#include "auxiliary/rocauxiliary_laqr.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

#include <cmath>
#include <vector>

ROCSOLVER_BEGIN_NAMESPACE

/** HSEQR_NONFINITE_BLOCK returns whether the Hessenberg part of the diagonal block
    H(ilo:ihi, ilo:ihi) of H contains a NaN or an infinite entry (all the threads of the
    thread-block must call it). **/
template <int BS, typename T, typename I>
__device__ __forceinline__ bool hseqr_nonfinite_block(const I ilo, const I ihi, const T* H, const I ldh)
{
    const I nh = ihi - ilo + 1;
    int bad = 0;
    for(I j = 0; j < nh && !bad; j++)
        for(I i = hipThreadIdx_x; i <= std::min(j + 1, nh - 1); i += BS)
        {
            const T h = H[idx2D(ilo - 1 + i, ilo - 1 + j, ldh)];
            if(!std::isfinite(h.real()) || !std::isfinite(h.imag()))
                bad = 1;
        }
    return __syncthreads_or(bad) != 0;
}

/** HSEQR_NONFINITE_BOTTOM splits the active block ilo:ihi of the upper Hessenberg matrix H
    into irreducible diagonal blocks at the exactly zero subdiagonal entries H(k+1,k), and
    returns the last row (1-based) of the lowest block of size > 1 whose Hessenberg part
    contains a NaN or an infinite entry, or 0 if there is none (all the threads of the
    thread-block must call it). The QR algorithm cannot converge on such a block (LAPACK
    would iterate until its iteration limit); the blocks below it do not depend on it, and
    the entries outside the diagonal blocks do not affect the convergence. A 1x1 block
    deflates at once (as in ZLAHQR, or in the deflation window of ZLAQR0): its diagonal
    entry is its eigenvalue, even if it is a NaN or infinite. **/
template <int BS, typename T, typename I>
__device__ I hseqr_nonfinite_bottom(const I ilo, const I ihi, const T* H, const I ldh)
{
    // blocks from the bottom (the subdiagonal entries are read by all the threads)
    I bot = ihi;
    while(bot >= ilo)
    {
        I top = bot;
        while(top > ilo && !(H[idx2D(top - 1, top - 2, ldh)] == T(0)))
            top--;
        // (top and bot are the same for all the threads)
        if(top < bot && hseqr_nonfinite_block<BS>(top, bot, H, ldh))
            return bot;
        bot = top - 1;
    }
    return 0;
}

/** HSEQR_SET_NONFINITE sets the eigenvalues ilo:ihi to NaN (the rows of the diagonal
    blocks with NaN or infinite entries and those above them). **/
template <int BS, typename T, typename I>
__device__ void hseqr_set_nonfinite(const I ilo, const I ihi, T* W)
{
    using S = decltype(std::real(T{}));
    const S nan = std::numeric_limits<S>::quiet_NaN();
    for(I j = ilo - 1 + hipThreadIdx_x; j < ihi; j += BS)
        W[j] = T(nan, nan);
}

/** HSEQR_ANY_NONFINITE_KERNEL sets flag[b] to 1 if the Hessenberg part of the active block
    ilo:ihi of matrix b contains a NaN or an infinite entry (flag must be 0 on entry). It
    reads the matrix with many thread-blocks (grid = (columns, batch_count)), so that
    hseqr_check_kernel, whose exact search runs on one thread-block per matrix, only needs to
    run when there is such an entry. **/
template <int BS, typename T, typename I, typename UH>
ROCSOLVER_KERNEL void __launch_bounds__(BS) hseqr_any_nonfinite_kernel(const I n,
                                                                       const I* iloA,
                                                                       const I* ihiA,
                                                                       UH HH,
                                                                       const rocblas_stride shiftH,
                                                                       const I ldh,
                                                                       const rocblas_stride strideH,
                                                                       I* flag)
{
    const I bid = hipBlockIdx_y;
    const I ilo = std::min(std::max(iloA[bid], I(1)), n);
    const I ihi = std::min(std::max(ihiA[bid], ilo), n);
    const T* H = load_ptr_batch<T>(HH, bid, shiftH, strideH);
    int bad = 0;
    for(I j = ilo + hipBlockIdx_x; j <= ihi; j += hipGridDim_x)
        for(I i = ilo + hipThreadIdx_x; i <= std::min(j + 1, ihi); i += BS)
        {
            const T h = H[idx2D(i - 1, j - 1, ldh)];
            if(!std::isfinite(h.real()) || !std::isfinite(h.imag()))
                bad = 1;
        }
    if(__syncthreads_or(bad) && hipThreadIdx_x == 0)
        flag[bid] = 1;
}

/** HSEQR_CHECK_KERNEL finds, for each matrix of the batch (multishift path), the last row
    ibad of the lowest diagonal block of size > 1 of the active block with a NaN or an
    infinite entry (see hseqr_nonfinite_bottom); if there is one, the eigenvalues W(ilo:ibad)
    are set to NaN (and W(ihi) = H(ihi,ihi) if ibad = ihi-1: the 1x1 block ihi is left, and
    there is nothing to iterate on).
    flag is set to ibad, or to 0 if there is no such block. On entry, flag must be 0 if the
    active block has no NaN or infinite entry (see hseqr_any_nonfinite_kernel), and then
    there is nothing to do. **/
template <int BS, typename T, typename I, typename UH>
ROCSOLVER_KERNEL void __launch_bounds__(BS) hseqr_check_kernel(const I n,
                                                               const I* iloA,
                                                               const I* ihiA,
                                                               UH HH,
                                                               const rocblas_stride shiftH,
                                                               const I ldh,
                                                               const rocblas_stride strideH,
                                                               T* WW,
                                                               const rocblas_stride strideW,
                                                               I* flag)
{
    const I bid = hipBlockIdx_x;
    if(flag[bid] == 0)
        return;
    const I ilo = std::min(std::max(iloA[bid], I(1)), n);
    const I ihi = std::min(std::max(ihiA[bid], ilo), n);
    const T* H = load_ptr_batch<T>(HH, bid, shiftH, strideH);
    T* W = WW + rocblas_stride(bid) * strideW;
    // (with ilo = ihi there is nothing to iterate on, as in the small-size kernel)
    const I ibad = (ilo < ihi) ? hseqr_nonfinite_bottom<BS>(ilo, ihi, H, ldh) : I(0);
    if(ibad > 0)
    {
        hseqr_set_nonfinite<BS>(ilo, ibad, W);
        if(ibad == ihi - 1 && hipThreadIdx_x == 0)
            W[ihi - 1] = H[idx2D(ihi - 1, ihi - 1, ldh)];
    }
    if(hipThreadIdx_x == 0)
        flag[bid] = ibad;
}

/** HSEQR_KERNEL computes the eigenvalues and, optionally, the Schur form and
    Schur vectors of each Hessenberg matrix in the batch, following LAPACK ZHSEQR.
    Each matrix is processed by a single thread-block of BS threads with the
    single-shift algorithm of ZLAHQR. It is used for n <= HSEQR_NMIN (75, as in
    LAPACK); larger matrices use the multishift algorithm (hseqr_multishift). With LDS,
    the matrices with ilo = 1 and ihi = n (and n <= HQR_LDS_NMAX, wantt and wantz) are
    processed in shared memory (lahqr_lds_block). The first row of the rows iterated on
    (see below) is stored in ilo1A, for the retry of the failures (see
    rocsolver_hseqr_template). **/
template <int BS, bool LDS, typename T, typename I, typename UH, typename UZ>
ROCSOLVER_KERNEL void __launch_bounds__(BS) hseqr_kernel(const rocsolver_schur_job job,
                                                         const rocsolver_schur_vectors compz,
                                                         const I n,
                                                         const I* iloA,
                                                         const I* ihiA,
                                                         UH HH,
                                                         const rocblas_stride shiftH,
                                                         const I ldh,
                                                         const rocblas_stride strideH,
                                                         T* WW,
                                                         const rocblas_stride strideW,
                                                         UZ ZZ,
                                                         const rocblas_stride shiftZ,
                                                         const I ldz,
                                                         const rocblas_stride strideZ,
                                                         I* infoA,
                                                         I* ilo1A)
{
    const I bid = hipBlockIdx_x;
    const I tid = hipThreadIdx_x;

    // quick return
    if(n == 0)
    {
        if(tid == 0)
            infoA[bid] = 0;
        return;
    }

    const bool wantt = (job == rocsolver_schur_form);
    const bool initz = (compz == rocsolver_schur_vectors_initialize);
    const bool wantz = initz || (compz == rocsolver_schur_vectors_update);

    T* H = load_ptr_batch<T>(HH, bid, shiftH, strideH);
    T* W = WW + rocblas_stride(bid) * strideW;
    T* Z = wantz ? load_ptr_batch<T>(ZZ, bid, shiftZ, strideZ) : nullptr;
    auto h = [&](const I i, const I j) -> T& { return H[idx2D(i - 1, j - 1, ldh)]; };

    // ilo and ihi are read from device memory and cannot be validated on the host;
    // they are clamped to a valid range (1 <= ilo <= ihi <= n)
    const I ilo = std::min(std::max(iloA[bid], I(1)), n);
    const I ihi = std::min(std::max(ihiA[bid], ilo), n);

    __shared__ I s_red[2][HQR_RED(BS)];
    int buf = 0;

    // copy eigenvalues isolated by GEBAL
    for(I j = 1 + tid; j <= n; j += BS)
        if(j < ilo || j > ihi)
            W[j - 1] = h(j, j);

    // initialize Z, if requested
    if(initz)
    {
        for(I j = 0; j < n; j++)
            for(I r = tid; r < n; r += BS)
                Z[idx2D(r, j, ldz)] = (r == j) ? T(1) : T(0);
    }
    __syncthreads();

    // quick return if possible (as in LAPACK, H is not cleaned up in this case)
    if(ilo == ihi)
    {
        if(tid == 0)
        {
            W[ilo - 1] = h(ilo, ilo);
            infoA[bid] = 0;
            if(ilo1A)
                ilo1A[bid] = ilo;
        }
        return;
    }

    // a NaN or an infinite entry in a diagonal block of size > 1 of the active block (split
    // at the exactly zero subdiagonal entries): the rows ibad+1:ihi below the lowest such block
    // are processed as usual (H(ibad+1, ibad) = 0), and info = ibad if they converge,
    // with W(ilo:ibad) = NaN (LAPACK would iterate on that block until its iteration limit)
    const I ibad = hseqr_nonfinite_bottom<BS>(ilo, ihi, H, ldh);
    I info = 0;
    if(ibad > 0)
        hseqr_set_nonfinite<BS>(ilo, ibad, W);
    if constexpr(LDS)
    {
        __shared__ T lds_ws[HQR_LDS_WS_SIZE];
        if(ibad == 0 && ilo == 1 && ihi == n && wantt && wantz && n <= HQR_LDS_NMAX)
            info = lahqr_lds_block<BS>(n, H, ldh, W, Z, ldz, lds_ws);
        else if(ibad < ihi)
            info = lahqr_block<BS>(wantt, wantz, n, ibad > 0 ? ibad + 1 : ilo, ihi, H, ldh, W, ilo,
                                   ihi, Z, ldz, s_red, buf);
    }
    else if(ibad < ihi)
        info = lahqr_block<BS>(wantt, wantz, n, ibad > 0 ? ibad + 1 : ilo, ihi, H, ldh, W, ilo, ihi,
                               Z, ldz, s_red, buf);
    if(info == 0)
        info = ibad;
    if(tid == 0 && ilo1A)
        ilo1A[bid] = ibad > 0 ? ibad + 1 : ilo;
    __syncthreads();

    // clear out the trash, if necessary
    if((wantt || info != 0) && n > 2)
    {
        for(I j = 1; j <= n - 2; j++)
            for(I r = j + 2 + tid; r <= n; r += BS)
                h(r, j) = T(0);
    }

    if(tid == 0)
        infoA[bid] = info;
}

/** HSEQR_PREPARE_KERNEL performs the first steps of ZHSEQR for the multishift path:
    it copies the eigenvalues isolated by GEBAL into W, initializes Z if requested,
    and handles the case ilo = ihi. **/
template <typename T, typename I, typename UH, typename UZ>
ROCSOLVER_KERNEL void hseqr_prepare_kernel(const rocsolver_schur_vectors compz,
                                           const I n,
                                           const I* iloA,
                                           const I* ihiA,
                                           UH HH,
                                           const rocblas_stride shiftH,
                                           const I ldh,
                                           const rocblas_stride strideH,
                                           T* WW,
                                           const rocblas_stride strideW,
                                           UZ ZZ,
                                           const rocblas_stride shiftZ,
                                           const I ldz,
                                           const rocblas_stride strideZ)
{
    const I bid = hipBlockIdx_y;
    const I tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const I nthreads = hipGridDim_x * hipBlockDim_x;
    const bool initz = (compz == rocsolver_schur_vectors_initialize);

    T* H = load_ptr_batch<T>(HH, bid, shiftH, strideH);
    T* W = WW + rocblas_stride(bid) * strideW;
    const I ilo = std::min(std::max(iloA[bid], I(1)), n);
    const I ihi = std::min(std::max(ihiA[bid], ilo), n);

    for(I j = 1 + tid; j <= n; j += nthreads)
        if(j < ilo || j > ihi || ilo == ihi)
            W[j - 1] = H[idx2D(j - 1, j - 1, ldh)];

    if(initz)
    {
        T* Z = load_ptr_batch<T>(ZZ, bid, shiftZ, strideZ);
        for(I j = 0; j < n; j++)
            for(I r = tid; r < n; r += nthreads)
                Z[idx2D(r, j, ldz)] = (r == j) ? T(1) : T(0);
    }
}

/** HSEQR_FINISH_KERNEL clears out the trash below the first subdiagonal of H (if
    needed, as in ZHSEQR) and sets info for one matrix. **/
template <typename T, typename I>
ROCSOLVER_KERNEL void
    hseqr_finish_kernel(const bool clean, const I n, T* H, const I ldh, I* info, const I infoval)
{
    const I tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const I nthreads = hipGridDim_x * hipBlockDim_x;
    if(clean && n > 2)
        for(I j = 1; j <= n - 2; j++)
            for(I r = j + 2 + tid; r <= n; r += nthreads)
                H[idx2D(r - 1, j - 1, ldh)] = T(0);
    if(tid == 0)
        *info = infoval;
}

template <int BS, typename T, typename I>
ROCSOLVER_KERNEL void __launch_bounds__(BS) laqr0_iteration_kernel(const I n,
                                                                   const I ilo,
                                                                   const I kbot,
                                                                   const I ndfl,
                                                                   const I nw,
                                                                   const I ndec,
                                                                   const I nwr,
                                                                   const I nwmax,
                                                                   const I nsr,
                                                                   const I nsmax,
                                                                   T* H,
                                                                   const I ldh,
                                                                   T* W,
                                                                   I* status,
                                                                   T* statusT)
{
    laqr0_iteration_block<BS>(n, ilo, kbot, ndfl, nw, ndec, nwr, nwmax, nsr, nsmax, H, ldh, W,
                              status, statusT);
}

/** HSEQR_COPY_BAND_KERNEL copies the entries (i, j) of the m-by-m matrix A with i <= j + d
    to B (grid = dim3(ceil(m / BS2), ceil(m / BS2)), block = dim3(BS2, BS2)). **/
template <typename T, typename I>
ROCSOLVER_KERNEL void
    hseqr_copy_band_kernel(const I m, const I d, const T* A, const I lda, T* B, const I ldb)
{
    const I i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const I j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    if(i < m && j < m && i <= j + d)
        B[idx2D(i, j, ldb)] = A[idx2D(i, j, lda)];
}

/** LAQR0_CORE4_KERNEL completes an iteration deferred by laqr0_iteration_kernel because
    of a large deflation window (see laqr0_core4_block). **/
template <int BS, typename T, typename I>
ROCSOLVER_KERNEL void __launch_bounds__(BS) laqr0_core4_kernel(const I n,
                                                               const I kbot,
                                                               const I ndfl,
                                                               const I nwmax,
                                                               const I nsr,
                                                               const I nsmax,
                                                               T* H,
                                                               const I ldh,
                                                               T* W,
                                                               I* status,
                                                               T* statusT)
{
    laqr0_core4_block<BS>(n, kbot, ndfl, nwmax, nsr, nsmax, H, ldh, W, status, statusT);
}

/** LAQR0_PART1_KERNEL and LAQR0_PART2_KERNEL run the device parts of one iteration
    of the multishift QR algorithm in hybrid mode (the core of the aggressive early
    deflation runs on the host in between). **/
template <int BS, typename T, typename I>
ROCSOLVER_KERNEL void __launch_bounds__(BS) laqr0_part1_kernel(const I n,
                                                               const I ilo,
                                                               const I kbot,
                                                               const I ndfl,
                                                               const I nw,
                                                               const I ndec,
                                                               const I nwr,
                                                               const I nwmax,
                                                               T* H,
                                                               const I ldh,
                                                               T* W,
                                                               I* status,
                                                               T* statusT)
{
    __shared__ I s_ired[2][HQR_RED(BS)];
    int ibuf = 0;
    laqr0_part1_block<BS>(n, ilo, kbot, ndfl, nw, ndec, nwr, nwmax, H, ldh, W, status, statusT,
                          s_ired, ibuf);
}

template <int BS, typename T, typename I>
ROCSOLVER_KERNEL void __launch_bounds__(BS) laqr0_part2_kernel(const I n,
                                                               const I kbot,
                                                               const I ndfl,
                                                               const I nwmax,
                                                               const I nsr,
                                                               const I nsmax,
                                                               T* H,
                                                               const I ldh,
                                                               T* W,
                                                               I* status,
                                                               T* statusT)
{
    laqr0_part2_block<BS>(n, kbot, ndfl, nwmax, nsr, nsmax, H, ldh, W, status, statusT);
}

template <int BS, typename T, typename I>
ROCSOLVER_KERNEL void __launch_bounds__(BS)
    laqr0_toofew_kernel(const I n, T* H, const I ldh, T* W, I* status)
{
    __shared__ I s_ired[2][HQR_RED(BS)];
    int ibuf = 0;
    laqr0_toofew_block<BS>(n, H, ldh, W, status, s_ired, ibuf);
}

template <int BS, typename T, typename I>
ROCSOLVER_KERNEL void __launch_bounds__(BS) laqr5_chunk_kernel(const bool wantt,
                                                               const bool wantz,
                                                               const bool accum,
                                                               const I n,
                                                               const I ktop,
                                                               const I kbot,
                                                               const I nbmps,
                                                               const I incol,
                                                               const T* sh,
                                                               T* H,
                                                               const I ldh,
                                                               const I iloz,
                                                               const I ihiz,
                                                               T* Z,
                                                               const I ldz,
                                                               T* Vbuf,
                                                               unsigned* bar)
{
    __shared__ T sV[HSEQR_MAX_SHIFTS / 2 + 2][3];
    I G = I(hipGridDim_x), blk = I(hipBlockIdx_x);
    if(G > 1)
    {
        // (all the thread-blocks must be resident for the grid barriers; if they are not, the
        // first one does the chunk alone, see laqr5_coresident)
        __shared__ int s_role;
        if(hipThreadIdx_x == 0)
            s_role = laqr5_coresident(bar + 2, unsigned(G));
        __syncthreads();
        if(s_role < 0)
            return;
        if(s_role == 0)
        {
            G = 1;
            blk = 0;
        }
    }
    laqr5_chunk_block<BS>(wantt, wantz, accum, n, ktop, kbot, nbmps, incol, sh, H, ldh, iloz, ihiz,
                          Z, ldz, Vbuf, sV, G, blk, bar);
}

/** LAQR5_BUILD_U_KERNEL forms U from the stored reflections (laqr5_build_u_block), each
    thread-block computing HSEQR_BUILD_U_ROWS(kdu) rows. **/
template <int BS, typename T, typename I>
ROCSOLVER_KERNEL void __launch_bounds__(BS) laqr5_build_u_kernel(const I ktop,
                                                                 const I kbot,
                                                                 const I nbmps,
                                                                 const I incol,
                                                                 const T* Vbuf,
                                                                 const I nr,
                                                                 T* U,
                                                                 const I ldu)
{
    extern __shared__ double lmem[];
    const I kdu = 6 * nbmps - 3;
    const I r0 = 1 + hipBlockIdx_x * nr;
    const I nrb = std::min(nr, kdu - r0 + 1);
    laqr5_build_u_block<BS>(ktop, kbot, nbmps, incol, Vbuf, r0, nrb, reinterpret_cast<T*>(lmem), U,
                            ldu);
}

/** LAQR5_LEFT_APPLY_KERNEL applies the reflections of a chunk from the left to the
    columns j0:j0+gridDim.x-1 of H (laqr5_left_apply_block), one thread-block per column. **/
template <int BS, typename T, typename I>
ROCSOLVER_KERNEL void __launch_bounds__(BS) laqr5_left_apply_kernel(const I ktop,
                                                                    const I kbot,
                                                                    const I nbmps,
                                                                    const I incol,
                                                                    const T* Vbuf,
                                                                    T* H,
                                                                    const I ldh,
                                                                    const I j0)
{
    extern __shared__ double lmem[];
    laqr5_left_apply_block<BS>(ktop, kbot, nbmps, incol, Vbuf, H, ldh, j0 + I(hipBlockIdx_x),
                               reinterpret_cast<T*>(lmem));
}

/** HSEQR_SIDE_STREAM holds the second stream (and its events) on which the formation of U
    and the far-from-diagonal updates of each chunk of the sweep run, concurrently with
    the next chunk. It is created when first needed and destroyed with the object, after
    the work of both streams is complete: the main stream waits for the side stream at the
    end of each sweep, but that wait may still be pending on the device when the host
    returns, and it must not refer to a destroyed event (the work of the side stream would
    then outlive the call). **/
struct hseqr_side_stream
{
    hipStream_t s0 = nullptr; // main stream
    hipStream_t s1 = nullptr;
    hipEvent_t chased[2] = {nullptr, nullptr}; // chunk kernel done (Vbuf and H ready)
    hipEvent_t ubuilt[2] = {nullptr, nullptr}; // U formed (Vbuf slot free again)
    hipEvent_t far[2] = {nullptr, nullptr}; // far column updates of the chunk done
    hipEvent_t done = nullptr; // all the work of the sweep on s1 done

    rocblas_status init()
    {
        if(s1)
            return rocblas_status_success;
        HIP_CHECK(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking));
        for(hipEvent_t* e : {&chased[0], &chased[1], &ubuilt[0], &ubuilt[1], &far[0], &far[1], &done})
            HIP_CHECK(hipEventCreateWithFlags(e, hipEventDisableTiming));
        return rocblas_status_success;
    }
    ~hseqr_side_stream()
    {
        if(!s1)
            return;
        (void)hipStreamSynchronize(s1);
        (void)hipStreamSynchronize(s0);
        for(hipEvent_t e : {chased[0], chased[1], ubuilt[0], ubuilt[1], far[0], far[1], done})
            (void)hipEventDestroy(e);
        (void)hipStreamDestroy(s1);
    }
};

/** HSEQR_CHASE_GROUPS returns the number of thread-blocks that chase the bulges of a
    chunk (with accum). They synchronize with grid barriers (laqr5_grid_barrier), so they
    must all become resident: at most a quarter of the compute units are requested, each of
    which can hold one of them (checked with the occupancy of the kernel; otherwise a single
    thread-block is used). Kernels running concurrently on the side stream may delay some of
    them, but not indefinitely, as those kernels do not wait for the chase; if they do not
    all become resident (for example with other processes on the device), the first one
    chases the chunk alone (see laqr5_coresident). **/
template <typename T, typename I>
rocblas_status hseqr_chase_groups(rocblas_handle handle, I& maxgroups)
{
    const int ncu = rocblas_internal_get_device_prop(handle)->multiProcessorCount;
    int occupancy = 0;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, laqr5_chunk_kernel<HSEQR_CHASE_BLOCKSIZE, T, I>, HSEQR_CHASE_BLOCKSIZE, 0));
    maxgroups = (occupancy >= 1) ? std::max(1, std::min(int(HSEQR_CHASE_GROUPS), ncu / 4)) : 1;
    return rocblas_status_success;
}

/** HSEQR_WORKT_LAYOUT is the layout of the scalar workspace workT of HSEQR (in entries
    of type T) for matrices of order n: the status scalars, the reflections of two chunks
    of the sweep with the largest number of shifts (see hseqr_multishift), and the
    compact copy of the band of a chunk, with leading dimension ldw. **/
struct hseqr_workT_layout
{
    size_t vbuf; // offset of the two slots of reflections
    size_t win; // offset of the compact copy of the band of a chunk
    size_t ldw; // its leading dimension (at least kdu + 8 for every sweep)
    size_t size; // total number of entries

    template <typename I>
    explicit hseqr_workT_layout(const I n)
    {
        I nsmax = std::min((n + 6) / 9, I(HSEQR_MAX_SHIFTS));
        nsmax = std::max(I(2), nsmax - nsmax % 2);
        const size_t nbmps = nsmax / 2;
        vbuf = LAQR0_STATUS_SCALAR_SIZE;
        win = vbuf + 2 * (3 * nbmps) * 3 * (nbmps + 1);
        ldw = 6 * nbmps + 5;
        size = win + ldw * ldw;
    }
};

/** HSEQR_HOST_OPS returns the host versions of the routines of the aggressive early
    deflation of the hybrid mode (see hqr_host_ops) for the processor. The library is built
    for the baseline x86-64 instruction set; on x86-64 hosts, the routines are also compiled
    for x86-64-v3 (AVX2, FMA) and x86-64-v4 (AVX-512), and the best versions that the
    processor supports are used. (flatten inlines the generic routine into each version, so
    that it is compiled for that instruction set; the versions are distinct functions, so
    the code of different instruction sets is never mixed.) **/
#define HSEQR_HOST_VERSION(suffix, attrs)                                                         \
    template <typename T, typename I>                                                             \
    attrs I hseqr_host_schur_##suffix(I n, T* Tw, I ldt, T* W, T* V, I ldv)                       \
    {                                                                                             \
        I s_ired[2][1];                                                                           \
        int ibuf = 0;                                                                             \
        return lahqr_block<1>(true, true, n, I(1), n, Tw, ldt, W, I(1), n, V, ldv, s_ired, ibuf); \
    }                                                                                             \
    template <typename T, typename I>                                                             \
    attrs void hseqr_host_trexc_##suffix(I n, T* A, I ldt, T* Q, I ldq, I ifst, I ilst)           \
    {                                                                                             \
        trexc_block<1>(true, n, A, ldt, Q, ldq, ifst, ilst);                                      \
    }                                                                                             \
    template <typename T, typename I>                                                             \
    attrs void hseqr_host_larf_left_##suffix(I m, I n, const T* v, T tau, T* C, I ldc)            \
    {                                                                                             \
        hqr_block_larf_left<1>(m, n, v, tau, C, ldc);                                             \
    }                                                                                             \
    template <typename T, typename I>                                                             \
    attrs void hseqr_host_larf_right_##suffix(I m, I n, const T* v, T tau, T* C, I ldc)           \
    {                                                                                             \
        hqr_block_larf_right<1>(m, n, v, tau, C, ldc);                                            \
    }

HSEQR_HOST_VERSION(base, __attribute__((flatten)))
#if defined(__x86_64__) && !defined(__HIP_DEVICE_COMPILE__)
HSEQR_HOST_VERSION(v3, __attribute__((target("arch=x86-64-v3"), flatten)))
HSEQR_HOST_VERSION(v4, __attribute__((target("arch=x86-64-v4"), flatten)))
#endif
#undef HSEQR_HOST_VERSION

template <typename T, typename I>
const hqr_host_ops<T, I>* hseqr_host_ops()
{
#define HSEQR_HOST_OPS(suffix)                                                          \
    {                                                                                   \
        &hseqr_host_schur_##suffix<T, I>, &hseqr_host_trexc_##suffix<T, I>,             \
            &hseqr_host_larf_left_##suffix<T, I>, &hseqr_host_larf_right_##suffix<T, I> \
    }
    static const hqr_host_ops<T, I> base = HSEQR_HOST_OPS(base);
#if defined(__x86_64__) && !defined(__HIP_DEVICE_COMPILE__)
    static const hqr_host_ops<T, I> v3 = HSEQR_HOST_OPS(v3);
    static const hqr_host_ops<T, I> v4 = HSEQR_HOST_OPS(v4);
    static const hqr_host_ops<T, I>* ops = __builtin_cpu_supports("x86-64-v4") ? &v4
        : __builtin_cpu_supports("x86-64-v3")                                  ? &v3
                                                                               : &base;
    return ops;
#else
    return &base;
#endif
#undef HSEQR_HOST_OPS
}

/** HSEQR_AED_WINDOW_CAP returns the cap of the (initial) deflation window for an active
    block of order nh: HSEQR_AED_WINDOW_MAX (0: no cap), raised for large blocks. The
    number of shifts per sweep is capped likewise, and the number of steps of the chase
    grows as nh^2 / (number of shifts), so that larger windows pay off as nh grows,
    sooner in hybrid mode, where the Schur form of the window is computed on the host.
    The thresholds were chosen from timings of zhseqr on random matrices. **/
template <typename I>
I hseqr_aed_window_cap(const I nh, const bool hybrid)
{
    I cap = HSEQR_AED_WINDOW_MAX;
    if(cap <= 0)
        return 0;
    if(hybrid)
    {
        if(nh >= 7500)
            cap = std::max(cap, I(96));
        if(nh >= 15000)
            cap = std::max(cap, I(160));
        if(nh >= 30000)
            cap = std::max(cap, I(192));
    }
    else
    {
        if(nh >= 15000)
            cap = std::max(cap, I(96));
    }
    return cap;
}

/** HSEQR_RESET_BARRIER_KERNEL sets the arrivals counter of laqr5_grid_barrier to 0 before a
    launch of the chunk kernel (with an atomic store, through the L2 cache: a memset may be done
    by another engine and leave a stale copy of the counter in the cache). **/
ROCSOLVER_KERNEL void hseqr_reset_barrier_kernel(unsigned* bar)
{
    __hip_atomic_store(bar, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    // (and the co-residency check, see laqr5_coresident)
    __hip_atomic_store(bar + 2, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __hip_atomic_store(bar + 3, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

/** HSEQR_MULTISHIFT computes the Schur form of one Hessenberg matrix with the
    multishift QR algorithm with aggressive early deflation of LAPACK ZLAQR0. The
    control flow runs on the host; each iteration launches the iteration kernel (active
    block, deflation window, AED and shifts; with a deflation window larger than
    LAQR4_NMIN, the AED core and the shift selection are completed by
    laqr0_core4_kernel), reads back a small status array, and then launches the
    off-window updates of the AED, the computation of extra shifts if the AED provides
    too few (laqr0_toofew_kernel), and the chunks of the sweep, with their
    matrix-matrix products. As in LAPACK, the workspace of these steps is the part of
    H below its first subdiagonal. On success, info is 0, or kbot > 0 if the iterations
    did not converge (as in LAPACK); HIP and rocBLAS errors are returned as the status.

    In hybrid mode, the core of the aggressive early deflation (the Schur form of the
    deflation window, the deflation tests and the return to Hessenberg form) runs on
    the host, with the same code as on the device (except that the Schur form of the
    window always uses ZLAHQR, and in a different order of operations, so that the
    results are equally valid but not bitwise identical): the window is copied to the
    host and back in each iteration, except for windows larger than HSEQR_HYBRID_WMAX, whose
    core runs on the device (laqr0_core4_kernel), as LAPACK uses ZLAQR4 for large windows.

    The side stream (side, with side.s0 the stream of the handle) and the number of
    thread-blocks of the chase (maxgroups, see hseqr_chase_groups) are set up once by the
    caller for all the matrices of the batch. **/
template <typename T, typename I>
rocblas_status hseqr_multishift(rocblas_handle handle,
                                const bool wantt,
                                const bool wantz,
                                const I n,
                                const I ilo,
                                const I ihi,
                                T* H,
                                const I ldh,
                                T* W,
                                const I iloz,
                                const I ihiz,
                                T* Z,
                                const I ldz,
                                I* dstatus,
                                T* dstatusT,
                                unsigned* dbar,
                                const bool hybrid,
                                hseqr_side_stream& side,
                                const I maxgroups,
                                I& info)
{
    using S = decltype(std::real(T{}));
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    rocblas_pointer_mode_saver saver(handle, rocblas_pointer_mode_host);

    const T one = T(1);
    const T zero = T(0);

    // C (m-by-k) = op(A) * B through the scratch W (ldw), then copied back into C, on the
    // stream gstream (the stream of the handle must be the same)
    hipStream_t gstream = stream;
    auto gemm_copy = [&](rocblas_operation transA, const I m, const I nn, const I k, T* A,
                         const I lda, T* B, const I ldb, T* C, const I ldc, T* Ws, const I ldw) {
        if(m <= 0 || nn <= 0)
            return rocblas_status_success;
        ROCBLAS_CHECK(rocblasCall_gemm(handle, transA, rocblas_operation_none, m, nn, k, &one, A, 0,
                                       lda, 0, B, 0, ldb, 0, &zero, Ws, 0, ldw, 0, I(1),
                                       (T**)nullptr));
        const I blocksx = (m - 1) / BS2 + 1;
        const I blocksy = (nn - 1) / BS2 + 1;
        ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, T*, T*>), dim3(blocksx, blocksy, 1), dim3(BS2, BS2), 0,
                                gstream, m, nn, Ws, 0, ldw, 0, C, 0, ldc, 0);
        return rocblas_status_success;
    };
    auto h = [&](const I i, const I j) -> T* { return H + idx2D(i - 1, j - 1, ldh); };
    auto z = [&](const I i, const I j) -> T* { return Z + idx2D(i - 1, j - 1, ldz); };

    // tuning parameters (LAPACK IPARMQ and ZLAQR0 3.9.0, with an unlimited LWORK)
    const I nhfull = ihi - ilo + 1;
    I ns_ip, nw_ip;
    hseqr_iparmq(nhfull, ns_ip, nw_ip);
    const bool accum = (ns_ip >= 14); // KACC22 = 2 (same as 1 here) when NS >= 14, else 0
    const I nwr = std::min(nhfull, std::min((n - 1) / 3, std::max(I(2), nw_ip)));
    I nsr = std::min(ns_ip, std::min((n + 6) / 9, ihi - ilo));
    nsr = std::max(I(2), nsr - nsr % 2);
    nsr = std::min(nsr, I(HSEQR_MAX_SHIFTS));
    const I nwmax = (n - 1) / 3;
    I nsmax = (n + 6) / 9;
    nsmax = std::min(nsmax - nsmax % 2, I(HSEQR_MAX_SHIFTS));
    const I itmax = 30 * std::max(I(10), nhfull);

    // deflation window size: the value recommended by IPARMQ, capped by
    // hseqr_aed_window_cap (see HSEQR_AED_WINDOW_MAX in ideal_sizes.hpp). The number of
    // shifts is capped likewise: with more shifts than the window can provide, every
    // sweep would get the missing ones from ZLAHQR on an ns-by-ns trailing submatrix (a
    // slow, sequential step on the device)
    I nwr_t = nwr;
    I nsr_t = nsr;
    const I wcap = hseqr_aed_window_cap(nhfull, hybrid);
    if(wcap > 0)
    {
        // (the window size selection of ZLAQR0 may add 1 to nwr, so that the window, of at
        // most wcap entries, still fits the shared memory kernels when wcap is HQR_LDS_NMAX)
        nwr_t = std::min(nwr_t, std::max(I(2), wcap - 1));
        nsr_t = std::min(nsr_t, wcap);
        nsr_t = std::max(I(2), nsr_t - nsr_t % 2);
    }

    I nw = nwmax;
    I ndec = -1;
    I ndfl = 1;
    I kbot = ihi;
    I st[LAQR0_STATUS_SIZE];
    T stT[LAQR0_STATUS_SCALAR_SIZE];
    // host workspace of the hybrid mode (pageable, as the other hybrid algorithms)
    std::vector<T> hT, hV, hWsh, hwork, hB, hWs;

    // Workspace. As in LAPACK, all the scratch arrays of an iteration live in H below
    // its first subdiagonal (entries (i,j) with i >= j + 2), which is not part of the
    // Hessenberg matrix; nw <= (n-1)/3 and ns <= (n+6)/9 keep them there. In stream
    // order, with the lifetime of each region:
    //  1. part1 (AED setup) -> off-window updates of the AED (step 3):
    //     T  = H(kv:kv+jw-1, kt:kt+jw-1), the window (dead after part2 writes it back),
    //     V  = H(kv:kv+jw-1, 1:jw),       its Schur vectors (read in step 3),
    //     with kv = n-nw+1, kt = nw+1 (in hybrid mode T and V are also copied to and
    //     from the host around the AED core).
    //  2. AED core (device or host):
    //     work = H(kwv:kwv+jw-1, 1), kwv = nw+2 (rows nw+2:2nw+1 < kv, disjoint from V).
    //  3. off-window updates of the AED (host-ordered GEMMs, after part2):
    //     reads V; scratch WV = H(kwv:kv-1, 1:jw) (rows < kv) and WH = H(kv:n, kt:kt+nho-1)
    //     (the T region, columns > nw), both disjoint from V.
    //  4. extra shifts, when the AED provides too few (laqr0_toofew_kernel, or the host in
    //     hybrid mode; after step 3, which updates the rows of the trailing submatrix
    //     above the window; V is dead): H(n-ns+1:n, 1:ns), as in LAPACK.
    //  5. sweep chunks (after steps 3 and 4):
    //     U  = H(n-kdu+1:n, 1:kdu), kdu = 3ns-3 (with accum: formed by laqr5_build_u_kernel
    //     on the side stream from the reflections of the chunk, and read by the GEMMs that
    //     follow it), WH = H(n-kdu+1:n, kdu+1:kdu+nho') and WV = H(kdu+4:n-kdu, 1:kdu),
    //     both disjoint from U; all of them lie at least 4 rows below the diagonal, outside
    //     the band that the chunk kernels use. The reflections of the chunks (Vbuf) and the
    //     compact copy of the band of a chunk (see below) are in workT.
    // The shifts and eigenvalues are in W (not in H); the status arrays are separate.
    for(I it = 1; it <= itmax; it++)
    {
        // done when kbot falls below ilo
        if(kbot < ilo)
        {
            // (copies from the host workspace may still be pending)
            if(hybrid)
                HIP_CHECK(hipStreamSynchronize(stream));
            info = 0;
            return rocblas_status_success;
        }

        // active block, deflation window, AED and shifts
        if(!hybrid)
        {
            ROCSOLVER_LAUNCH_KERNEL((laqr0_iteration_kernel<HSEQR_BLOCKSIZE, T>), dim3(1),
                                    dim3(HSEQR_BLOCKSIZE), 0, stream, n, ilo, kbot, ndfl, nw, ndec,
                                    nwr_t, nwmax, nsr_t, nsmax, H, ldh, W, dstatus, dstatusT);
        }
        else
        {
            ROCSOLVER_LAUNCH_KERNEL((laqr0_part1_kernel<HSEQR_BLOCKSIZE, T>), dim3(1),
                                    dim3(HSEQR_BLOCKSIZE), 0, stream, n, ilo, kbot, ndfl, nw, ndec,
                                    nwr_t, nwmax, H, ldh, W, dstatus, dstatusT);
            HIP_CHECK(hipMemcpyAsync(st, dstatus, sizeof(I) * LAQR0_STATUS_SIZE,
                                     hipMemcpyDeviceToHost, stream));
            HIP_CHECK(hipMemcpyAsync(stT, dstatusT, sizeof(T) * LAQR0_STATUS_SCALAR_SIZE,
                                     hipMemcpyDeviceToHost, stream));
            HIP_CHECK(hipStreamSynchronize(stream));

            // (large windows: the core on the device with the multishift QR algorithm, as
            // the host only has the single-shift one; laqr0_core4_kernel also does part 2)
            const bool core4 = st[LAQR0_CORE] && st[LAQR0_JW] > I(HSEQR_HYBRID_WMAX);
            if(core4)
                ROCSOLVER_LAUNCH_KERNEL((laqr0_core4_kernel<HSEQR_BLOCKSIZE, T>), dim3(1),
                                        dim3(HSEQR_BLOCKSIZE), 0, stream, n, kbot, ndfl, nwmax,
                                        nsr_t, nsmax, H, ldh, W, dstatus, dstatusT);
            else if(st[LAQR0_CORE])
            {
                // core of the aggressive early deflation on the host, with the window
                // T = H(kv,kt) (and V = I) copied from the device
                const I nwc = st[LAQR0_NW];
                const I jw = st[LAQR0_JW];
                const I kwtop = st[LAQR0_KWTOP];
                const I kv = n - nwc + 1;
                const I kt = nwc + 1;
                // (the host copies have leading dimension jw + 1: with a power of two, the
                // accesses along the rows would map to few cache sets)
                const I ldj = jw + 1;
                hT.resize(size_t(ldj) * jw);
                hV.assign(size_t(ldj) * jw, T(0));
                hWsh.resize(jw);
                hwork.resize(jw);
                for(I i = 0; i < jw; i++)
                    hV[i + size_t(i) * ldj] = T(1);
                HIP_CHECK(hipMemcpy2DAsync(hT.data(), sizeof(T) * ldj, h(kv, kt), sizeof(T) * ldh,
                                           sizeof(T) * jw, jw, hipMemcpyDeviceToHost, stream));
                HIP_CHECK(hipStreamSynchronize(stream));

                I s_ired[2][1];
                S s_sred[2][1];
                int ibuf = 0, sbuf = 0;
                I ns, nd;
                bool update;
                T spike;
                aed_core_block<1>(n, jw, stT[LAQR0_SPIKE_IN], hT.data(), ldj, hV.data(), ldj,
                                  hwork.data(), hWsh.data(), ns, nd, update, spike, s_ired, s_sred,
                                  ibuf, sbuf, (T*)nullptr, (I*)nullptr, (T*)nullptr,
                                  (T(*)[3]) nullptr, hseqr_host_ops<T, I>());
                st[LAQR0_LS] = ns;
                st[LAQR0_LD] = nd;
                st[LAQR0_UPDATE] = update ? 1 : 0;
                stT[LAQR0_SPIKE_OUT] = spike;

                HIP_CHECK(hipMemcpy2DAsync(h(kv, kt), sizeof(T) * ldh, hT.data(), sizeof(T) * ldj,
                                           sizeof(T) * jw, jw, hipMemcpyHostToDevice, stream));
                HIP_CHECK(hipMemcpy2DAsync(h(kv, 1), sizeof(T) * ldh, hV.data(), sizeof(T) * ldj,
                                           sizeof(T) * jw, jw, hipMemcpyHostToDevice, stream));
                HIP_CHECK(hipMemcpyAsync(W + (kwtop - 1), hWsh.data(), sizeof(T) * jw,
                                         hipMemcpyHostToDevice, stream));
                HIP_CHECK(hipMemcpyAsync(dstatus, st, sizeof(I) * LAQR0_STATUS_SIZE,
                                         hipMemcpyHostToDevice, stream));
                HIP_CHECK(hipMemcpyAsync(dstatusT, stT, sizeof(T) * LAQR0_STATUS_SCALAR_SIZE,
                                         hipMemcpyHostToDevice, stream));
            }

            if(!core4)
                ROCSOLVER_LAUNCH_KERNEL((laqr0_part2_kernel<HSEQR_BLOCKSIZE, T>), dim3(1),
                                        dim3(HSEQR_BLOCKSIZE), 0, stream, n, kbot, ndfl, nwmax,
                                        nsr_t, nsmax, H, ldh, W, dstatus, dstatusT);
        }
        HIP_CHECK(hipMemcpyAsync(st, dstatus, sizeof(I) * LAQR0_STATUS_SIZE, hipMemcpyDeviceToHost,
                                 stream));
        HIP_CHECK(hipStreamSynchronize(stream));

        // with a deflation window larger than LAQR4_NMIN, the iteration kernel leaves the AED
        // core (multishift QR) and part 2 to laqr0_core4_kernel
        if(!hybrid && st[LAQR0_DEFER])
        {
            ROCSOLVER_LAUNCH_KERNEL((laqr0_core4_kernel<HSEQR_BLOCKSIZE, T>), dim3(1),
                                    dim3(HSEQR_BLOCKSIZE), 0, stream, n, kbot, ndfl, nwmax, nsr_t,
                                    nsmax, H, ldh, W, dstatus, dstatusT);
            HIP_CHECK(hipMemcpyAsync(st, dstatus, sizeof(I) * LAQR0_STATUS_SIZE,
                                     hipMemcpyDeviceToHost, stream));
            HIP_CHECK(hipStreamSynchronize(stream));
        }

        const I ktop = st[LAQR0_KTOP];
        nw = st[LAQR0_NW];
        ndec = st[LAQR0_NDEC];
        const I ld = st[LAQR0_LD];

        // off-window updates of the AED (ZLAQR2), through the workspace below the
        // subdiagonal: V = H(kv,1), T = H(kv,kt) (nw-by-nho), WV = H(kwv,1) (nve-by-nw)
        if(st[LAQR0_UPDATE])
        {
            const I jw = st[LAQR0_JW];
            const I kwtop = st[LAQR0_KWTOP];
            const I kv = n - nw + 1;
            const I kt = nw + 1;
            const I nho = (n - nw - 1) - kt + 1;
            const I kwv = nw + 2;
            const I nve = (n - nw) - kwv + 1;
            T* V = h(kv, 1);

            const I ltop = wantt ? I(1) : ktop;
            for(I krow = ltop; krow <= kwtop - 1; krow += nve)
            {
                const I kln = std::min(nve, kwtop - krow);
                ROCBLAS_CHECK(gemm_copy(rocblas_operation_none, kln, jw, jw, h(krow, kwtop), ldh, V,
                                        ldh, h(krow, kwtop), ldh, h(kwv, 1), ldh));
            }
            if(wantt)
            {
                for(I kcol = kbot + 1; kcol <= n; kcol += nho)
                {
                    const I kln = std::min(nho, n - kcol + 1);
                    ROCBLAS_CHECK(gemm_copy(rocblas_operation_conjugate_transpose, jw, kln, jw, V,
                                            ldh, h(kwtop, kcol), ldh, h(kwtop, kcol), ldh,
                                            h(kv, kt), ldh));
                }
            }
            if(wantz)
            {
                for(I krow = iloz; krow <= ihiz; krow += nve)
                {
                    const I kln = std::min(nve, ihiz - krow + 1);
                    ROCBLAS_CHECK(gemm_copy(rocblas_operation_none, kln, jw, jw, z(krow, kwtop),
                                            ldz, V, ldh, z(krow, kwtop), ldz, h(kwv, 1), ldh));
                }
            }
        }

        // too few shifts from the AED: they are the eigenvalues of the trailing submatrix
        // H(ks:kbot2, ks:kbot2) (ZLAHQR), computed now that the off-window updates of the
        // AED have updated its rows above the window
        if(st[LAQR0_TOOFEW])
        {
            if(hybrid)
            {
                // on the host, as in laqr0_toofew_block
                const I nsb = st[LAQR0_NS];
                const I ksb = st[LAQR0_KS];
                const I kbot2 = st[LAQR0_KBOT];
                hB.resize(size_t(nsb) * nsb);
                hWs.resize(nsb);
                HIP_CHECK(hipMemcpy2DAsync(hB.data(), sizeof(T) * nsb, h(ksb, ksb), sizeof(T) * ldh,
                                           sizeof(T) * nsb, nsb, hipMemcpyDeviceToHost, stream));
                HIP_CHECK(hipStreamSynchronize(stream));

                auto b = [&](const I i, const I j) -> T { return hB[(i - 1) + size_t(j - 1) * nsb]; };
                const T hb[4] = {b(nsb - 1, nsb - 1), b(nsb, nsb - 1), b(nsb - 1, nsb), b(nsb, nsb)};
                I s_ired[2][1];
                int ibuf = 0;
                const I inf = lahqr_block<1>(false, false, nsb, I(1), nsb, hB.data(), nsb,
                                             hWs.data(), I(1), I(1), (T*)nullptr, I(1), s_ired, ibuf);
                I ns = nsb;
                I ks = ksb + inf;
                laqr0_shifts_tail<1>(true, ks >= kbot2, ns, ks, kbot2, hWs.data(), ksb, hb);
                if(ns > 0)
                    HIP_CHECK(hipMemcpyAsync(W + (ks - 1), hWs.data() + (ks - ksb), sizeof(T) * ns,
                                             hipMemcpyHostToDevice, stream));
                st[LAQR0_NS] = ns;
                st[LAQR0_KS] = ks;
                st[LAQR0_TOOFEW] = 0;
            }
            else
            {
                ROCSOLVER_LAUNCH_KERNEL((laqr0_toofew_kernel<HSEQR_BLOCKSIZE, T>), dim3(1),
                                        dim3(HSEQR_BLOCKSIZE), 0, stream, n, H, ldh, W, dstatus);
                HIP_CHECK(hipMemcpyAsync(st, dstatus, sizeof(I) * LAQR0_STATUS_SIZE,
                                         hipMemcpyDeviceToHost, stream));
                HIP_CHECK(hipStreamSynchronize(stream));
            }
        }

        kbot = st[LAQR0_KBOT];

        // small-bulge multishift QR sweep (ZLAQR5, LAPACK 3.9.0) with the workspace
        // below the subdiagonal: U = H(ku,1) (kdu-by-kdu), WH = H(ku,kwh) (kdu-by-nho),
        // WV = H(kwv,1) (nve-by-kdu)
        const I ns = st[LAQR0_NS];
        if(st[LAQR0_SWEEP] && ns >= 2 && ktop < kbot)
        {
            const I ks = st[LAQR0_KS];
            const I nbmps = ns / 2;
            const I kdu = 3 * ns - 3;
            const I ngroups = std::max(I(1), std::min(maxgroups, nbmps / 4));
            const I ku = n - kdu + 1;
            const I kwh = kdu + 1;
            const I nho = (n - kdu + 1 - 4) - (kdu + 1) + 1;
            const I kwv = kdu + 4;
            const I nve = n - kdu - kwv + 1;
            T* U = h(ku, 1);
            const hseqr_workT_layout layout(n);
            T* Vbuf = dstatusT + layout.vbuf;
            T* Wwin = dstatusT + layout.win;

            // clear trash
            if(ktop + 2 <= kbot)
                HIP_CHECK(hipMemsetAsync(h(ktop + 2, ktop), 0, sizeof(T), stream));

            // With accum, the chunks are pipelined over two streams: on the main stream, the
            // chunk kernel and the left update of the columns that the next chunk uses
            // (laqr5_left_apply_kernel); on the side stream, the formation of U and the
            // matrix-matrix products that update the rest of H and Z, concurrently with the
            // next chunk (they act on entries that the next chunk kernel does not use). The
            // reflections of the chunks alternate between two slots of Vbuf.
            const size_t vslot = size_t(3 * nbmps) * 3 * (nbmps + 1);
            const I jbot = wantt ? n : kbot;
            I chunk = 0;
            if(accum)
            {
                ROCBLAS_CHECK(side.init());
                HIP_CHECK(hipEventRecord(side.done, stream));
                HIP_CHECK(hipStreamWaitEvent(side.s1, side.done, 0));
            }
            for(I incol = 3 * (1 - nbmps) + ktop - 1; incol <= kbot - 2;
                incol += 3 * nbmps - 2, chunk++)
            {
                const int slot = chunk % 2;
                T* Vb = Vbuf + slot * vslot;
                const I ndcol = incol + kdu;

                // the side stream must be done with this slot of Vbuf (chunk-2)
                if(accum && chunk >= 2)
                    HIP_CHECK(hipStreamWaitEvent(stream, side.ubuilt[slot], 0));
                // (without accum, a single thread-block chases the chunk: the multi-block chase
                // passes the reflections through Vbuf, which is only used with accum)
                if(!accum)
                {
                    ROCSOLVER_LAUNCH_KERNEL((laqr5_chunk_kernel<HSEQR_CHASE_BLOCKSIZE, T>), dim3(1),
                                            dim3(HSEQR_CHASE_BLOCKSIZE), 0, stream, wantt, wantz,
                                            accum, n, ktop, kbot, nbmps, incol, W + (ks - 1), H,
                                            ldh, iloz, ihiz, Z, ldz, Vb, dbar);
                    continue;
                }

                // With accum, the chunk kernel works on a compact copy of the part of H that
                // it reads, H(r0:r1, r0:r1) (with leading dimension layout.ldw): with the
                // leading dimension of a large H, the columns of the window span so much
                // memory that most of its accesses miss the address translation caches (the
                // time per step grows several-fold). It only reads and writes
                // entries (i, j) with i <= j + 3 (the Hessenberg part and the bulges), and
                // writes only within H(lo:r1, lo:r1), so only that band is copied in and back:
                // the entries further below are the workspace of the off-window products,
                // which the side stream may be using for the previous chunk (the entries of
                // the band that the previous chunk updates are finished before; see below).
                const I lo = std::max(ktop, incol);
                const I r0 = std::max(I(1), lo - 3);
                const I r1 = std::min(n, incol + kdu + 1);
                const I ldw = I(layout.ldw);
                {
                    const I mw = r1 - r0 + 1;
                    ROCSOLVER_LAUNCH_KERNEL(
                        (hseqr_copy_band_kernel<T>), dim3((mw - 1) / BS2 + 1, (mw - 1) / BS2 + 1),
                        dim3(BS2, BS2), 0, stream, mw, I(3), (const T*)h(r0, r0), ldh, Wwin, ldw);
                }
                T* Hc = Wwin - (r0 - 1) - size_t(r0 - 1) * ldw;
                // (the counter of the grid barriers starts at 0 in each launch)
                if(ngroups > 1)
                    ROCSOLVER_LAUNCH_KERNEL(hseqr_reset_barrier_kernel, dim3(1), dim3(1), 0, stream,
                                            dbar);
                ROCSOLVER_LAUNCH_KERNEL((laqr5_chunk_kernel<HSEQR_CHASE_BLOCKSIZE, T>),
                                        dim3(ngroups > 1 ? ngroups : 1), dim3(HSEQR_CHASE_BLOCKSIZE),
                                        0, stream, wantt, wantz, accum, n, ktop, kbot, nbmps, incol,
                                        W + (ks - 1), Hc, ldw, iloz, ihiz, Z, ldz, Vb, dbar);
                {
                    const I mw = r1 - lo + 1;
                    ROCSOLVER_LAUNCH_KERNEL(
                        (hseqr_copy_band_kernel<T>), dim3((mw - 1) / BS2 + 1, (mw - 1) / BS2 + 1),
                        dim3(BS2, BS2), 0, stream, mw, I(3),
                        (const T*)(Wwin + idx2D(lo - r0, lo - r0, ldw)), ldw, h(lo, lo), ldh);
                }
                HIP_CHECK(hipEventRecord(side.chased[slot], stream));

                // left update of the columns of the next chunk's window (after the far
                // column updates of the previous chunk, which act on some of them)
                const I jl0 = std::min(ndcol, kbot) + 1;
                const I jl1 = std::min(ndcol + 3 * nbmps - 1, jbot);
                if(chunk >= 1)
                    HIP_CHECK(hipStreamWaitEvent(stream, side.far[1 - slot], 0));
                if(jl0 <= jl1)
                    ROCSOLVER_LAUNCH_KERNEL((laqr5_left_apply_kernel<64, T>), dim3(jl1 - jl0 + 1),
                                            dim3(64), sizeof(T) * kdu, stream, ktop, kbot, nbmps,
                                            incol, (const T*)Vb, H, ldh, jl0);

                // side stream: form U, then update the far-from-diagonal entries of H and,
                // if required, Z
                HIP_CHECK(hipStreamWaitEvent(side.s1, side.chased[slot], 0));
                {
                    // (the handle and gemm_copy use the side stream in this block; the main
                    // stream is restored however the block is left. rocblas_set_stream does not
                    // release the device memory of the handle, which holds the workspace, even
                    // with stream-ordered allocation; its release at the end of the call is
                    // ordered on the main stream, which waits for the side stream first)
                    struct stream_restore
                    {
                        rocblas_handle handle;
                        hipStream_t s0;
                        hipStream_t& gs;
                        ~stream_restore()
                        {
                            rocblas_set_stream(handle, s0);
                            gs = s0;
                        }
                    } restore{handle, stream, gstream};
                    ROCBLAS_CHECK(rocblas_set_stream(handle, side.s1));
                    gstream = side.s1;

                    // (rows of U per thread-block, with at most 48 KB of shared memory)
                    constexpr size_t lds_bytes = 49152;
                    const I nr = std::max(I(1), std::min(kdu, I(lds_bytes / (sizeof(T) * kdu))));
                    const I nblk = (kdu - 1) / nr + 1;
                    ROCSOLVER_LAUNCH_KERNEL((laqr5_build_u_kernel<256, T>), dim3(nblk), dim3(256),
                                            sizeof(T) * nr * kdu, side.s1, ktop, kbot, nbmps, incol,
                                            (const T*)Vb, nr, U, ldh);
                    HIP_CHECK(hipEventRecord(side.ubuilt[slot], side.s1));

                    const I jtop = wantt ? I(1) : ktop;
                    const I k1 = std::max(I(1), ktop - incol);
                    const I nu = (kdu - std::max(I(0), ndcol - kbot)) - k1 + 1;
                    T* Uk = U + idx2D(k1 - 1, k1 - 1, ldh);

                    for(I jcol = std::max(jl0, jl1 + 1); jcol <= jbot; jcol += nho)
                    {
                        const I jlen = std::min(nho, jbot - jcol + 1);
                        ROCBLAS_CHECK(gemm_copy(rocblas_operation_conjugate_transpose, nu, jlen, nu,
                                                Uk, ldh, h(incol + k1, jcol), ldh,
                                                h(incol + k1, jcol), ldh, h(ku, kwh), ldh));
                    }
                    HIP_CHECK(hipEventRecord(side.far[slot], side.s1));
                    for(I jrow = jtop; jrow <= std::max(ktop, incol) - 1; jrow += nve)
                    {
                        const I jlen = std::min(nve, std::max(ktop, incol) - jrow);
                        ROCBLAS_CHECK(gemm_copy(rocblas_operation_none, jlen, nu, nu,
                                                h(jrow, incol + k1), ldh, Uk, ldh,
                                                h(jrow, incol + k1), ldh, h(kwv, 1), ldh));
                    }
                    if(wantz)
                    {
                        for(I jrow = iloz; jrow <= ihiz; jrow += nve)
                        {
                            const I jlen = std::min(nve, ihiz - jrow + 1);
                            ROCBLAS_CHECK(gemm_copy(rocblas_operation_none, jlen, nu, nu,
                                                    z(jrow, incol + k1), ldz, Uk, ldh,
                                                    z(jrow, incol + k1), ldz, h(kwv, 1), ldh));
                        }
                    }
                }
            }
            if(accum)
            {
                // the main stream waits for the side stream at the end of the sweep
                HIP_CHECK(hipEventRecord(side.done, side.s1));
                HIP_CHECK(hipStreamWaitEvent(stream, side.done, 0));
            }
        }

        // note progress (or the lack of it)
        ndfl = (ld > 0) ? 1 : ndfl + 1;
    }

    // iteration limit exceeded
    if(hybrid)
        HIP_CHECK(hipStreamSynchronize(stream));
    info = kbot;
    return rocblas_status_success;
}

template <typename T, typename I>
void rocsolver_hseqr_getMemorySize(const I n, const I batch_count, size_t* size_work, size_t* size_workT)
{
    // status arrays of the multishift path
    if(n == 0 || batch_count == 0)
    {
        *size_work = 0;
        *size_workT = 0;
    }
    else
    {
        // (and the flags of the matrices with NaN or infinite entries, or the first rows
        // iterated on by the single-shift path, and the counters of the grid barrier of the
        // sweep)
        *size_work = sizeof(I) * (LAQR0_STATUS_SIZE + batch_count + 4);
        // (and the reflections of two chunks of the sweep and the compact copy of the band
        // of a chunk, see hseqr_multishift; the single-shift path retries the failures with
        // the multishift one, on a copy of order HSEQR_NL when n < HSEQR_NL, as in LAPACK)
        const I nl = std::max(n, I(HSEQR_NL));
        *size_workT = sizeof(T) * hseqr_workT_layout(nl).size;
        if(n < HSEQR_NL)
            *size_workT += sizeof(T) * size_t(nl) * nl;
    }
}

template <typename T, typename I, typename U>
rocblas_status rocsolver_hseqr_argCheck(rocblas_handle handle,
                                        const rocsolver_schur_job job,
                                        const rocsolver_schur_vectors compz,
                                        const I n,
                                        const I* ilo,
                                        const I* ihi,
                                        const I ldh,
                                        const I ldz,
                                        U H,
                                        T* W,
                                        U Z,
                                        I* info,
                                        const I batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(job != rocsolver_schur_eigenvalues && job != rocsolver_schur_form)
        return rocblas_status_invalid_value;
    if(compz != rocsolver_schur_vectors_none && compz != rocsolver_schur_vectors_initialize
       && compz != rocsolver_schur_vectors_update)
        return rocblas_status_invalid_value;
    const bool wantz = (compz != rocsolver_schur_vectors_none);

    // 2. invalid size
    if(n < 0 || ldh < n || ldh < 1 || ldz < 1 || (wantz && ldz < n) || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !H) || (n && !W) || (n && wantz && !Z) || (n && batch_count && !ilo)
       || (n && batch_count && !ihi) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename U>
rocblas_status rocsolver_hseqr_template(rocblas_handle handle,
                                        const rocsolver_schur_job job,
                                        const rocsolver_schur_vectors compz,
                                        const I n,
                                        const I* ilo,
                                        const I* ihi,
                                        U H,
                                        const rocblas_stride shiftH,
                                        const I ldh,
                                        const rocblas_stride strideH,
                                        T* W,
                                        const rocblas_stride strideW,
                                        U Z,
                                        const rocblas_stride shiftZ,
                                        const I ldz,
                                        const rocblas_stride strideZ,
                                        I* info,
                                        const I batch_count,
                                        I* work,
                                        T* workT)
{
    ROCSOLVER_ENTER("hseqr", "job:", job, "compz:", compz, "n:", n, "shiftH:", shiftH, "ldh:", ldh,
                    "shiftZ:", shiftZ, "ldz:", ldz, "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    const bool wantt = (job == rocsolver_schur_form);
    const bool wantz = (compz != rocsolver_schur_vectors_none);

    // the core of the aggressive early deflation runs on the host in hybrid mode
    rocsolver_alg_mode alg_mode;
    ROCBLAS_CHECK(rocsolver_get_alg_mode(handle, rocsolver_function_hseqr, &alg_mode));
    const bool hybrid = (alg_mode == rocsolver_alg_mode_hybrid);

    // side stream of the sweeps and number of thread-blocks of the chase, for all the
    // matrices (see hseqr_multishift)
    hseqr_side_stream side;
    side.s0 = stream;
    I maxgroups = 1;

    // small matrices, and batches of medium-size matrices (see HSEQR_BATCH_NMAX):
    // single-shift QR (ZLAHQR), all the matrices of the batch in parallel (when n == 0 the
    // kernel only sets info = 0)
    const I batch_min = std::max(I(HSEQR_BATCH_MIN), (n + HSEQR_BATCH_NPER - 1) / HSEQR_BATCH_NPER);
    if(n <= HSEQR_NMIN || (n <= HSEQR_BATCH_NMAX && batch_count >= batch_min))
    {
        I* dilo1 = (n > 0) ? work + LAQR0_STATUS_SIZE : nullptr;
        auto launch = [&](auto bs, auto lds) {
            ROCSOLVER_LAUNCH_KERNEL((hseqr_kernel<decltype(bs)::value, decltype(lds)::value, T>),
                                    dim3(batch_count), dim3(decltype(bs)::value), 0, stream, job,
                                    compz, n, ilo, ihi, H, shiftH, ldh, strideH, W, strideW, Z,
                                    shiftZ, ldz, strideZ, info, dilo1);
        };
        using std::integral_constant;
        // Configuration of the kernel for n <= HSEQR_NMIN, from timings of zhseqr on gfx942
        // (wave64) and gfx1150 (wave32) with batches of 1 to 4000 matrices: the Schur form in
        // shared memory (n <= HQR_LDS_NMAX, with the Schur form and vectors) is about 1.7 times
        // faster per matrix, but it allows few thread-blocks per compute unit; on wave64
        // targets, plain thread-blocks of 64 to 256 threads (by n) are faster once the batch
        // needs more than one wave of the shared memory kernel. On wave32 targets the shared
        // memory kernel is faster for all the batches, and otherwise 64 threads are best.
        const hipDeviceProp_t* props = rocblas_internal_get_device_prop(handle);
        bool lds = wantt && wantz && n <= HQR_LDS_NMAX;
        const bool wave64 = (props->warpSize == 64);
        if(lds && wave64)
        {
            int occupancy = 0;
            HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                &occupancy, hseqr_kernel<64, true, T, I, U, U>, 64, 0));
            lds = (int64_t(batch_count) <= int64_t(occupancy) * props->multiProcessorCount);
        }
        if(n > HSEQR_NMIN)
        {
            if(wave64 && n > 128)
                launch(integral_constant<int, HSEQR_MEDIUM_BLOCKSIZE_LARGE>{}, std::false_type{});
            else
                launch(integral_constant<int, HSEQR_MEDIUM_BLOCKSIZE>{}, std::false_type{});
        }
        else if(lds)
            launch(integral_constant<int, 64>{}, std::true_type{});
        else if(!wave64 || n <= 32)
            launch(integral_constant<int, 64>{}, std::false_type{});
        else if(n <= 64)
            launch(integral_constant<int, 128>{}, std::false_type{});
        else
            launch(integral_constant<int, 256>{}, std::false_type{});
        if(n == 0)
            return rocblas_status_success;

        // As in LAPACK ZHSEQR, the rows ilo1:info of a matrix where ZLAHQR fails (info > 0,
        // which is rare) are processed again with the multishift algorithm (ZLAQR0), which
        // sometimes succeeds; for n < HSEQR_NL, on a copy of order HSEQR_NL padded with
        // zeros. This needs the values of info on the host, so that it is skipped when the
        // stream is being captured in a HIP graph (info > 0 is then returned).
        hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
        HIP_CHECK(hipStreamIsCapturing(stream, &capture));
        if(capture != hipStreamCaptureStatusNone)
            return rocblas_status_success;
        std::vector<I> hinfo(batch_count), hilo1(batch_count), hilo(batch_count), hihi(batch_count);
        HIP_CHECK(hipMemcpyAsync(hinfo.data(), info, sizeof(I) * batch_count, hipMemcpyDeviceToHost,
                                 stream));
        HIP_CHECK(hipMemcpyAsync(hilo1.data(), dilo1, sizeof(I) * batch_count,
                                 hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        bool any = false;
        for(I b = 0; b < batch_count; b++)
            any = any || (hinfo[b] > 0 && hinfo[b] >= hilo1[b]);
        if(!any)
            return rocblas_status_success;

        HIP_CHECK(hipMemcpyAsync(hilo.data(), ilo, sizeof(I) * batch_count, hipMemcpyDeviceToHost,
                                 stream));
        HIP_CHECK(hipMemcpyAsync(hihi.data(), ihi, sizeof(I) * batch_count, hipMemcpyDeviceToHost,
                                 stream));
        std::vector<T*> hH(batch_count), hZ(batch_count, nullptr);
        if constexpr(BATCHED)
        {
            HIP_CHECK(hipMemcpyAsync(hH.data(), H, sizeof(T*) * batch_count, hipMemcpyDeviceToHost,
                                     stream));
            if(wantz)
                HIP_CHECK(hipMemcpyAsync(hZ.data(), Z, sizeof(T*) * batch_count,
                                         hipMemcpyDeviceToHost, stream));
        }
        HIP_CHECK(hipStreamSynchronize(stream));
        ROCBLAS_CHECK(hseqr_chase_groups<T>(handle, maxgroups));

        const I nl = std::max(n, I(HSEQR_NL));
        T* Hl = workT + hseqr_workT_layout(nl).size;
        unsigned* dbar = reinterpret_cast<unsigned*>(work + LAQR0_STATUS_SIZE + batch_count);
        const I blocks = (nl - 1) / BS1 + 1;
        const I blocks2 = (nl - 1) / BS2 + 1;
        for(I b = 0; b < batch_count; b++)
        {
            if(!(hinfo[b] > 0 && hinfo[b] >= hilo1[b]))
                continue;
            T* Hb;
            T* Zb = nullptr;
            if constexpr(BATCHED)
            {
                Hb = hH[b] + shiftH;
                if(wantz)
                    Zb = hZ[b] + shiftZ;
            }
            else
            {
                Hb = H + shiftH + b * strideH;
                if(wantz)
                    Zb = Z + shiftZ + b * strideZ;
            }
            T* Wb = W + b * strideW;
            const I ilob = std::min(std::max(hilo[b], I(1)), n);
            const I ihib = std::min(std::max(hihi[b], ilob), n);
            const I ilo1 = hilo1[b];
            const I kbot = hinfo[b];

            HIP_CHECK(hipMemsetAsync(dbar, 0, 4 * sizeof(unsigned), stream));
            I infob = 0;
            if(n >= HSEQR_NL)
            {
                ROCBLAS_CHECK(hseqr_multishift<T>(handle, wantt, wantz, n, ilo1, kbot, Hb, ldh, Wb,
                                                  ilob, ihib, Zb, ldz, work, workT, dbar, hybrid,
                                                  side, maxgroups, infob));
            }
            else
            {
                // Hl = [H(1:n,1:n) 0; 0 0], of order nl
                ROCSOLVER_LAUNCH_KERNEL(set_zero<T>, dim3(blocks2, blocks2, 1), dim3(BS2, BS2), 0,
                                        stream, nl, nl, Hl, 0, nl, 0);
                ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, T*, T*>), dim3(blocks2, blocks2, 1),
                                        dim3(BS2, BS2), 0, stream, n, n, Hb, 0, ldh, 0, Hl, 0, nl, 0);
                ROCBLAS_CHECK(hseqr_multishift<T>(handle, wantt, wantz, nl, ilo1, kbot, Hl, nl, Wb,
                                                  ilob, ihib, Zb, ldz, work, workT, dbar, hybrid,
                                                  side, maxgroups, infob));
                if(wantt || infob != 0)
                    ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, T*, T*>), dim3(blocks2, blocks2, 1),
                                            dim3(BS2, BS2), 0, stream, n, n, Hl, 0, nl, 0, Hb, 0,
                                            ldh, 0);
            }
            // (the rows above ilo1, if any, have a NaN or an infinite entry)
            if(infob == 0)
                infob = (ilo1 > ilob) ? ilo1 - 1 : 0;
            // (clear out the trash, if necessary, as the multishift algorithm uses the part of
            // H below its first subdiagonal as workspace)
            ROCSOLVER_LAUNCH_KERNEL((hseqr_finish_kernel<T>), dim3(blocks), dim3(BS1), 0, stream,
                                    wantt || infob != 0, n, Hb, ldh, info + b, infob);
        }
        return rocblas_status_success;
    }

    // larger matrices: multishift QR with aggressive early deflation (ZLAQR0), one matrix
    // of the batch at a time (the control flow depends on the data)
    ROCBLAS_CHECK(hseqr_chase_groups<T>(handle, maxgroups));

    const I blocks = (n - 1) / BS1 + 1;
    ROCSOLVER_LAUNCH_KERNEL((hseqr_prepare_kernel<T>), dim3(blocks, batch_count), dim3(BS1), 0,
                            stream, compz, n, ilo, ihi, H, shiftH, ldh, strideH, W, strideW, Z,
                            shiftZ, ldz, strideZ);

    // diagonal blocks with a NaN or an infinite entry (and those above them) are not
    // iterated on
    I* dflag = work + LAQR0_STATUS_SIZE;
    unsigned* dbar = reinterpret_cast<unsigned*>(dflag + batch_count);
    HIP_CHECK(hipMemsetAsync(dbar, 0, 4 * sizeof(unsigned), stream));
    HIP_CHECK(hipMemsetAsync(dflag, 0, sizeof(I) * batch_count, stream));
    ROCSOLVER_LAUNCH_KERNEL((hseqr_any_nonfinite_kernel<HSEQR_BLOCKSIZE, T>),
                            dim3(std::min(n, I(1024)), batch_count), dim3(HSEQR_BLOCKSIZE), 0,
                            stream, n, ilo, ihi, H, shiftH, ldh, strideH, dflag);
    ROCSOLVER_LAUNCH_KERNEL((hseqr_check_kernel<HSEQR_BLOCKSIZE, T>), dim3(batch_count),
                            dim3(HSEQR_BLOCKSIZE), 0, stream, n, ilo, ihi, H, shiftH, ldh, strideH,
                            W, strideW, dflag);

    std::vector<I> hilo(batch_count), hihi(batch_count), hflag(batch_count);
    std::vector<T*> hH(batch_count), hZ(batch_count, nullptr);
    HIP_CHECK(hipMemcpyAsync(hflag.data(), dflag, sizeof(I) * batch_count, hipMemcpyDeviceToHost,
                             stream));
    HIP_CHECK(
        hipMemcpyAsync(hilo.data(), ilo, sizeof(I) * batch_count, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(
        hipMemcpyAsync(hihi.data(), ihi, sizeof(I) * batch_count, hipMemcpyDeviceToHost, stream));
    if constexpr(BATCHED)
    {
        HIP_CHECK(
            hipMemcpyAsync(hH.data(), H, sizeof(T*) * batch_count, hipMemcpyDeviceToHost, stream));
        if(wantz)
            HIP_CHECK(hipMemcpyAsync(hZ.data(), Z, sizeof(T*) * batch_count, hipMemcpyDeviceToHost,
                                     stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    for(I b = 0; b < batch_count; b++)
    {
        T* Hb;
        T* Zb = nullptr;
        if constexpr(BATCHED)
        {
            Hb = hH[b] + shiftH;
            if(wantz)
                Zb = hZ[b] + shiftZ;
        }
        else
        {
            Hb = H + shiftH + b * strideH;
            if(wantz)
                Zb = Z + shiftZ + b * strideZ;
        }
        T* Wb = W + b * strideW;

        // (ilo and ihi are clamped to a valid range, as in the small-size kernel)
        const I ilob = std::min(std::max(hilo[b], I(1)), n);
        const I ihib = std::min(std::max(hihi[b], ilob), n);

        // with a NaN or an infinite entry in a diagonal block of size > 1, only the
        // rows below the lowest such block (ibad = hflag[b]) are iterated on, and info = ibad
        // if they converge (see hseqr_check_kernel; the Z rows are still ilo:ihi)
        const I ibad = hflag[b];
        const I ilo1 = ibad > 0 ? ibad + 1 : ilob;
        I infob = 0;
        if(ilo1 < ihib)
            ROCBLAS_CHECK(hseqr_multishift<T>(handle, wantt, wantz, n, ilo1, ihib, Hb, ldh, Wb,
                                              ilob, ihib, Zb, ldz, work, workT, dbar, hybrid, side,
                                              maxgroups, infob));
        if(infob == 0)
            infob = ibad;

        // clear out the trash, if necessary (not when ilo == ihi, as in LAPACK)
        const bool clean = (wantt || infob != 0) && ilob < ihib;
        ROCSOLVER_LAUNCH_KERNEL((hseqr_finish_kernel<T>), dim3(blocks), dim3(BS1), 0, stream, clean,
                                n, Hb, ldh, info + b, infob);
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

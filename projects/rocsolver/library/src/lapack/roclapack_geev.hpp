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

#include "auxiliary/rocauxiliary_lange.hpp"
#include "auxiliary/rocauxiliary_orghr_unghr.hpp"
#include "init_scalars.hpp"
#include "rocblas.hpp"
#include "roclapack_gebak.hpp"
#include "roclapack_gebal.hpp"
#include "roclapack_gehrd.hpp"
#include "roclapack_hseqr.hpp"
#include "roclapack_trevc3.hpp"
#include "rocsolver/rocsolver.h"

#include <algorithm>
#include <numeric>
#include <vector>

ROCSOLVER_BEGIN_NAMESPACE

/*
 * ===========================================================================
 *    GEEV follows LAPACK ZGEEV: scaling of A if its entries are too large or
 *    too small, balancing (GEBAL), Hessenberg reduction (GEHRD) and, if
 *    eigenvectors are wanted, generation of the unitary factor (UNGHR), Schur
 *    factorization (HSEQR), eigenvectors of the Schur form back-transformed with
 *    the Schur vectors (TREVC3), back-transformation of the balancing (GEBAK),
 *    normalization, and unscaling of the eigenvalues. All the matrices of a
 *    batch go through each stage together; the scaling decisions are taken on the
 *    device, but GEHRD and UNGHR take their range ilo:ihi as a host argument, so
 *    the ranges computed by GEBAL are read back to the host (a synchronization).
 *    When all the matrices of a batch have the same range, they are reduced together; otherwise they are reduced in groups with the same range
 *    (through arrays of pointers), each group with exactly its own range. A common
 *    larger range would be exact in exact arithmetic (the reflectors outside a
 *    matrix's own range are identities), but 0 * NaN = NaN would carry a NaN or an
 *    infinite entry outside the active block of a matrix (in A12 or A23) into it.
 * ===========================================================================
 */

/** GEEV_LASCL_FACTORS computes the factors whose product is cto/cfrom, as in
    LAPACK xLASCL (multiplying by them in order never overflows or underflows
    unless the final result does). Returns their number (at most 8). **/
template <typename S>
__device__ int geev_lascl_factors(S cfrom, S cto, S* mul)
{
    const S smlnum = std::numeric_limits<S>::min();
    const S bignum = S(1) / smlnum;
    int nmul = 0;
    bool done = false;
    // (the scalings of GEEV, between anrm and smlnum or bignum, need at most two or three
    // factors; 8 bounds the loop)
    while(!done && nmul < 8)
    {
        const S cfrom1 = cfrom * smlnum;
        if(cfrom1 == cfrom)
        {
            // cfrom is infinite: the result is cto/cfrom (a signed zero or NaN)
            mul[nmul++] = cto / cfrom;
            done = true;
        }
        else
        {
            const S cto1 = cto / bignum;
            if(cto1 == cto)
            {
                // cto is zero or infinite
                mul[nmul++] = cto;
                done = true;
            }
            else if(std::abs(cfrom1) > std::abs(cto) && cto != 0)
            {
                mul[nmul++] = smlnum;
                cfrom = cfrom1;
            }
            else if(std::abs(cto1) > std::abs(cfrom))
            {
                mul[nmul++] = bignum;
                cto = cto1;
            }
            else
            {
                mul[nmul++] = cto / cfrom;
                done = true;
            }
        }
    }
    return nmul;
}

/** GEEV_CSCALE returns the value to which the matrix of norm anrm is scaled
    (as in ZGEEV), or 0 if it is not scaled. **/
template <typename S>
__device__ S geev_cscale(const S anrm)
{
    const S eps = std::numeric_limits<S>::epsilon();
    const S smlnum = std::sqrt(std::numeric_limits<S>::min()) / eps;
    const S bignum = S(1) / smlnum;
    if(anrm > 0 && anrm < smlnum)
        return smlnum;
    if(anrm > bignum)
        return bignum;
    return 0;
}

/** GEEV_SCALE_KERNEL scales each matrix A (n-by-n) from anrm to cscale, if needed. **/
template <typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void geev_scale_kernel(const I n,
                                        U AA,
                                        const rocblas_stride shiftA,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        const S* anrmA)
{
    const I bid = hipBlockIdx_z;
    const S anrm = anrmA[bid];
    const S cscale = geev_cscale(anrm);
    if(cscale == 0)
        return;

    S mul[8];
    const int nmul = geev_lascl_factors(anrm, cscale, mul);
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    const I i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const I j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    if(i < n && j < n)
    {
        T a = A[i + j * size_t(lda)];
        for(int k = 0; k < nmul; k++)
            a = a * mul[k];
        A[i + j * size_t(lda)] = a;
    }
}

/** GEEV_UNSCALE_KERNEL undoes the scaling on the computed eigenvalues: W(info+1:n)
    and, if info > 0, W(1:ilo-1) (1-based), as in ZGEEV. **/
template <typename T, typename I, typename S>
ROCSOLVER_KERNEL void geev_unscale_kernel(const I n,
                                          T* WW,
                                          const rocblas_stride strideW,
                                          const S* anrmA,
                                          const I* iloA,
                                          const I* infoA)
{
    const I bid = hipBlockIdx_y;
    const S anrm = anrmA[bid];
    const S cscale = geev_cscale(anrm);
    if(cscale == 0)
        return;

    S mul[8];
    const int nmul = geev_lascl_factors(cscale, anrm, mul);
    T* W = WW + bid * strideW;
    const I info = infoA[bid];
    const I ilo = iloA[bid];
    const I i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(i < n && (i >= info || (info > 0 && i < ilo - 1)))
    {
        T w = W[i];
        for(int k = 0; k < nmul; k++)
            w = w * mul[k];
        W[i] = w;
    }
}

/** GEEV_COPY_KERNEL copies the n-by-n matrices A into V. If zero_A, the entries of A
    below its first subdiagonal are then set to zero. **/
template <typename T, typename I, typename U>
ROCSOLVER_KERNEL void geev_copy_kernel(const I n,
                                       U AA,
                                       const rocblas_stride shiftA,
                                       const I lda,
                                       const rocblas_stride strideA,
                                       U VV,
                                       const rocblas_stride shiftV,
                                       const I ldv,
                                       const rocblas_stride strideV,
                                       const bool copy,
                                       const bool zero_A)
{
    const I bid = hipBlockIdx_z;
    const I i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const I j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    if(i < n && j < n)
    {
        T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
        if(copy)
        {
            T* V = load_ptr_batch<T>(VV, bid, shiftV, strideV);
            V[i + j * size_t(ldv)] = A[i + j * size_t(lda)];
        }
        if(zero_A && i > j + 1)
            A[i + j * size_t(lda)] = T(0);
    }
}

/** GEEV_NORMALIZE_KERNEL normalizes each column of V as in ZGEEV: to Euclidean norm 1,
    and then multiplied by the complex sign that makes its entry of largest
    |Re|^2 + |Im|^2 (the first one, if there are several) real. The matrices with
    info > 0 are skipped. **/
template <int BS, typename T, typename I, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(BS) geev_normalize_kernel(const I n,
                                                                  U VV,
                                                                  const rocblas_stride shiftV,
                                                                  const I ldv,
                                                                  const rocblas_stride strideV,
                                                                  const I* infoA)
{
    using S = decltype(std::real(T{}));

    const I j = hipBlockIdx_x;
    const I bid = hipBlockIdx_y;
    const I tid = hipThreadIdx_x;
    if(infoA[bid] > 0)
        return;
    T* v = load_ptr_batch<T>(VV, bid, shiftV, strideV) + j * size_t(ldv);
    __shared__ S sval[BS];
    __shared__ I sidx[BS];

    // Euclidean norm, scaled by the largest component to avoid overflow and underflow
    S amax = 0;
    for(I i = tid; i < n; i += BS)
        amax = std::max(amax, std::max(std::abs(v[i].real()), std::abs(v[i].imag())));
    amax = trevc3_block_max<BS>(amax, sval);
    if(!(amax > 0) || !std::isfinite(amax))
        return;
    S ssq = 0;
    for(I i = tid; i < n; i += BS)
    {
        const S re = v[i].real() / amax;
        const S im = v[i].imag() / amax;
        ssq += re * re + im * im;
    }
    sval[tid] = ssq;
    __syncthreads();
    for(int s = BS / 2; s > 0; s /= 2)
    {
        if(tid < s)
            sval[tid] += sval[tid + s];
        __syncthreads();
    }
    // (the scaling is v / amax / sqrt(ssq), in two steps: amax * sqrt(ssq) may overflow)
    const S scl = S(1) / std::sqrt(sval[0]);
    __syncthreads();

    // scale, and find the first entry of largest |Re|^2 + |Im|^2
    S rmax = -1;
    I imax = n;
    for(I i = tid; i < n; i += BS)
    {
        const T x = T(v[i].real() / amax, v[i].imag() / amax) * scl;
        v[i] = x;
        const S r = x.real() * x.real() + x.imag() * x.imag();
        if(r > rmax)
        {
            rmax = r;
            imax = i;
        }
    }
    sval[tid] = rmax;
    sidx[tid] = imax;
    __syncthreads();
    for(int s = BS / 2; s > 0; s /= 2)
    {
        if(tid < s)
        {
            const S r2 = sval[tid + s];
            const I i2 = sidx[tid + s];
            if(r2 > sval[tid] || (r2 == sval[tid] && i2 < sidx[tid]))
            {
                sval[tid] = r2;
                sidx[tid] = i2;
            }
        }
        __syncthreads();
    }
    const I k = sidx[0];
    if(k >= n)
        return;
    const T vk = v[k];
    const T tmp = hqr_cdivr(conj(vk), std::sqrt(sval[0]));
    __syncthreads();

    for(I i = tid; i < n; i += BS)
    {
        const T x = v[i] * tmp;
        v[i] = (i == k) ? T(x.real(), 0) : x;
    }
}

template <typename T, typename I, typename U>
rocblas_status rocsolver_geev_argCheck(rocblas_handle handle,
                                       const rocblas_evect jobvl,
                                       const rocblas_evect jobvr,
                                       const I n,
                                       const I lda,
                                       const I ldvl,
                                       const I ldvr,
                                       U A,
                                       T* W,
                                       U VL,
                                       U VR,
                                       I* info,
                                       const I batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(jobvl != rocblas_evect_original && jobvl != rocblas_evect_none)
        return rocblas_status_invalid_value;
    if(jobvr != rocblas_evect_original && jobvr != rocblas_evect_none)
        return rocblas_status_invalid_value;
    const bool leftv = (jobvl == rocblas_evect_original);
    const bool rightv = (jobvr == rocblas_evect_original);

    // 2. invalid size
    if(n < 0 || lda < n || lda < 1 || ldvl < 1 || ldvr < 1 || (leftv && ldvl < n)
       || (rightv && ldvr < n) || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !W) || (n && leftv && !VL) || (n && rightv && !VR)
       || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

/** Workspace of GEEV: the buffers work1 to work6 are shared by the stages (each has
    the largest size that any stage needs), and tau, scale, iloihi and anrm persist
    across the stages. **/
template <bool BATCHED, typename T, typename I>
void rocsolver_geev_getMemorySize(rocblas_handle handle,
                                  const rocblas_evect jobvl,
                                  const rocblas_evect jobvr,
                                  const I n,
                                  const I batch_count,
                                  size_t* size_scalars,
                                  size_t* size_work1,
                                  size_t* size_work2,
                                  size_t* size_work3,
                                  size_t* size_work4,
                                  size_t* size_work5,
                                  size_t* size_work6,
                                  size_t* size_tau,
                                  size_t* size_scale,
                                  size_t* size_iloihi,
                                  size_t* size_anrm,
                                  size_t* size_ptrs)
{
    using S = decltype(std::real(T{}));

    // quick return
    if(n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_work5 = 0;
        *size_work6 = 0;
        *size_tau = 0;
        *size_scale = 0;
        *size_iloihi = 0;
        *size_anrm = 0;
        *size_ptrs = 0;
        return;
    }

    const bool leftv = (jobvl == rocblas_evect_original);
    const bool rightv = (jobvr == rocblas_evect_original);
    const bool wantv = leftv || rightv;

    // norm of A
    size_t w_ln;
    rocsolver_lange_getMemorySize<T, I, S>(handle, rocsolver_norm_type_max, n, n, batch_count, &w_ln);

    // balancing
    size_t w_bl;
    rocsolver_gebal_getMemorySize<T, I>(rocsolver_balance_both, n, batch_count, &w_bl);

    // Hessenberg reduction (and unitary factor below), for the largest range ilo:ihi.
    // When the matrices of a batch have different ranges, they are reduced in groups
    // through arrays of pointers (the batched versions), whatever the layout of A
    size_t s_hr, w1_hr, w2_hr, w3_hr, w4_hr, w5_hr, w6_hr;
    rocsolver_gehrd_getMemorySize<BATCHED, T>(n, I(1), n, batch_count, &s_hr, &w1_hr, &w2_hr,
                                              &w3_hr, &w4_hr, &w5_hr, &w6_hr);
    if(!BATCHED && batch_count > 1)
    {
        size_t s_b, w1_b, w2_b, w3_b, w4_b, w5_b, w6_b;
        rocsolver_gehrd_getMemorySize<true, T>(n, I(1), n, batch_count, &s_b, &w1_b, &w2_b, &w3_b,
                                               &w4_b, &w5_b, &w6_b);
        s_hr = std::max(s_hr, s_b);
        w1_hr = std::max(w1_hr, w1_b);
        w2_hr = std::max(w2_hr, w2_b);
        w3_hr = std::max(w3_hr, w3_b);
        w4_hr = std::max(w4_hr, w4_b);
        w5_hr = std::max(w5_hr, w5_b);
        w6_hr = std::max(w6_hr, w6_b);
    }

    // unitary factor
    size_t s_gh = 0, w1_gh = 0, w2_gh = 0, w3_gh = 0, w4_gh = 0;
    if(wantv)
    {
        rocsolver_orghr_unghr_getMemorySize<BATCHED, T>(n, 1, n, batch_count, &s_gh, &w1_gh, &w2_gh,
                                                        &w3_gh, &w4_gh);
        if(!BATCHED && batch_count > 1)
        {
            size_t s_b, w1_b, w2_b, w3_b, w4_b;
            rocsolver_orghr_unghr_getMemorySize<true, T>(n, 1, n, batch_count, &s_b, &w1_b, &w2_b,
                                                         &w3_b, &w4_b);
            s_gh = std::max(s_gh, s_b);
            w1_gh = std::max(w1_gh, w1_b);
            w2_gh = std::max(w2_gh, w2_b);
            w3_gh = std::max(w3_gh, w3_b);
            w4_gh = std::max(w4_gh, w4_b);
        }
    }

    // Schur factorization
    size_t w1_hs, w2_hs;
    rocsolver_hseqr_getMemorySize<T, I>(n, batch_count, &w1_hs, &w2_hs);

    // eigenvectors
    size_t w1_tv = 0, w2_tv = 0, w3_tv = 0, w4_tv = 0, w5_tv = 0;
    if(wantv)
    {
        const rocblas_side side = leftv && rightv ? rocblas_side_both
            : leftv                               ? rocblas_side_left
                                                  : rocblas_side_right;
        rocsolver_trevc3_getMemorySize<BATCHED, T>(side, rocsolver_eigenvectors_backtransform, n,
                                                   batch_count, &w1_tv, &w2_tv, &w3_tv, &w4_tv,
                                                   &w5_tv);
    }

    *size_scalars = std::max(s_hr, s_gh);
    *size_work1 = std::max({w_ln, w_bl, w1_hr, w1_gh, w1_hs, w1_tv});
    *size_work2 = std::max({w2_hr, w2_gh, w2_hs, w2_tv});
    *size_work3 = std::max({w3_hr, w3_gh, w3_tv});
    *size_work4 = std::max({w4_hr, w4_gh, w4_tv});
    *size_work5 = std::max(w5_hr, w5_tv);
    *size_work6 = w6_hr;
    *size_tau = sizeof(T) * n * batch_count;
    *size_scale = sizeof(S) * n * batch_count;
    *size_iloihi = sizeof(I) * 2 * batch_count;
    *size_anrm = sizeof(S) * batch_count;

    // arrays of pointers to the matrices A and Q, grouped by range ilo:ihi
    *size_ptrs = batch_count > 1 ? sizeof(T*) * 2 * batch_count : 0;
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename U>
rocblas_status rocsolver_geev_template(rocblas_handle handle,
                                       const rocblas_evect jobvl,
                                       const rocblas_evect jobvr,
                                       const I n,
                                       U A,
                                       const rocblas_stride shiftA,
                                       const I lda,
                                       const rocblas_stride strideA,
                                       T* W,
                                       const rocblas_stride strideW,
                                       U VL,
                                       const rocblas_stride shiftVL,
                                       const I ldvl,
                                       const rocblas_stride strideVL,
                                       U VR,
                                       const rocblas_stride shiftVR,
                                       const I ldvr,
                                       const rocblas_stride strideVR,
                                       I* info,
                                       const I batch_count,
                                       T* scalars,
                                       void* work1,
                                       void* work2,
                                       void* work3,
                                       void* work4,
                                       void* work5,
                                       void* work6,
                                       T* tau,
                                       void* scale,
                                       I* iloihi,
                                       void* anrm,
                                       T** ptrs)
{
    ROCSOLVER_ENTER("geev", "jobvl:", jobvl, "jobvr:", jobvr, "n:", n, "shiftA:", shiftA,
                    "lda:", lda, "shiftVL:", shiftVL, "ldvl:", ldvl, "shiftVR:", shiftVR,
                    "ldvr:", ldvr, "bc:", batch_count);

    using S = decltype(std::real(T{}));

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // quick return (no eigenvalues; info = 0)
    if(n == 0)
    {
        ROCSOLVER_LAUNCH_KERNEL(reset_info, dim3((batch_count - 1) / BS1 + 1), dim3(BS1), 0, stream,
                                info, batch_count, 0);
        return rocblas_status_success;
    }

    const bool leftv = (jobvl == rocblas_evect_original);
    const bool rightv = (jobvr == rocblas_evect_original);
    const bool wantv = leftv || rightv;
    S* scaleS = reinterpret_cast<S*>(scale);
    S* anrmS = reinterpret_cast<S*>(anrm);
    I* ilo = iloihi;
    I* ihi = iloihi + batch_count;
    const rocblas_stride strideP = n;
    const rocblas_stride strideS = n;
    const I blocks = (n - 1) / BS2 + 1;
    const dim3 grid2(blocks, blocks, batch_count);
    const dim3 threads2(BS2, BS2, 1);

    // scale A if its largest entry is outside [smlnum, bignum]
    ROCBLAS_CHECK(rocsolver_lange_template<T>(handle, rocsolver_norm_type_max, n, n, A, shiftA, lda,
                                              strideA, batch_count, anrmS, (S*)work1));
    ROCSOLVER_LAUNCH_KERNEL((geev_scale_kernel<T>), grid2, threads2, 0, stream, n, A, shiftA, lda,
                            strideA, (const S*)anrmS);

    // balance
    ROCBLAS_CHECK(rocsolver_gebal_template<BATCHED, STRIDED, T>(
        handle, rocsolver_balance_both, n, A, shiftA, lda, strideA, ilo, ihi, scaleS, strideS,
        batch_count, (I*)work1));

    // range of the Hessenberg reduction: the active block ilo:ihi of each matrix (read back,
    // as GEHRD and UNGHR take it as a host argument)
    std::vector<I> hilo(batch_count), hihi(batch_count);
    HIP_CHECK(
        hipMemcpyAsync(hilo.data(), ilo, sizeof(I) * batch_count, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(
        hipMemcpyAsync(hihi.data(), ihi, sizeof(I) * batch_count, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    bool uniform = true;
    for(I b = 0; b < batch_count; b++)
    {
        // (clamped to 1 <= ilo <= ihi <= n, as in HSEQR)
        hilo[b] = std::min(std::max(hilo[b], I(1)), n);
        hihi[b] = std::min(std::max(hihi[b], hilo[b]), n);
        // (with ilo = ihi, the reduction does nothing and Q = I, whatever ilo is: these
        // matrices are reduced together, with the range 1:1)
        if(hilo[b] == hihi[b])
            hilo[b] = hihi[b] = 1;
        uniform = uniform && hilo[b] == hilo[0] && hihi[b] == hihi[0];
    }

    // unitary factor Q of the Hessenberg reduction, in VL (or VR if only the right
    // eigenvectors are wanted)
    U Q = leftv ? VL : VR;
    const rocblas_stride shiftQ = leftv ? shiftVL : shiftVR;
    const I ldq = leftv ? ldvl : ldvr;
    const rocblas_stride strideQ = leftv ? strideVL : strideVR;

    if(uniform)
    {
        // reduce to upper Hessenberg form
        ROCBLAS_CHECK(rocsolver_gehrd_template<BATCHED, STRIDED, T>(
            handle, n, hilo[0], hihi[0], A, shiftA, lda, strideA, tau, strideP, batch_count,
            scalars, work1, (T*)work2, work3, (T*)work4, (T*)work5, (T*)work6));

        // Q (the reflectors are then cleared from A)
        ROCSOLVER_LAUNCH_KERNEL((geev_copy_kernel<T>), grid2, threads2, 0, stream, n, A, shiftA,
                                lda, strideA, Q, shiftQ, ldq, strideQ, wantv, true);
        if(wantv)
            ROCBLAS_CHECK(rocsolver_orghr_unghr_template<BATCHED, STRIDED, T>(
                handle, n, hilo[0], hihi[0], Q, shiftQ, ldq, strideQ, tau, strideP, batch_count,
                scalars, (T*)work1, (T*)work2, (T*)work3, (T**)work4));
    }
    else
    {
        // The matrices have different ranges: they are reduced in groups with the same
        // range, each group with exactly its range (with a larger range, the reflectors
        // outside a matrix's own range would be identities, but 0 * NaN = NaN would carry
        // a NaN or an infinite entry outside its active block into it). The groups use
        // arrays of pointers to their matrices (and to their Q), in group order; the
        // Householder scalars of the k-th matrix in that order are in tau + k * strideP,
        // for GEHRD and UNGHR alike.
        std::vector<I> order(batch_count);
        std::iota(order.begin(), order.end(), I(0));
        std::stable_sort(order.begin(), order.end(), [&](const I a, const I b) {
            return hilo[a] < hilo[b] || (hilo[a] == hilo[b] && hihi[a] < hihi[b]);
        });

        std::vector<T*> hA(batch_count), hQ(batch_count, nullptr), hptrs(2 * size_t(batch_count));
        if constexpr(BATCHED)
        {
            HIP_CHECK(hipMemcpyAsync(hA.data(), A, sizeof(T*) * batch_count, hipMemcpyDeviceToHost,
                                     stream));
            if(wantv)
                HIP_CHECK(hipMemcpyAsync(hQ.data(), Q, sizeof(T*) * batch_count,
                                         hipMemcpyDeviceToHost, stream));
            HIP_CHECK(hipStreamSynchronize(stream));
        }
        else
        {
            for(I b = 0; b < batch_count; b++)
            {
                hA[b] = A + b * strideA;
                if(wantv)
                    hQ[b] = Q + b * strideQ;
            }
        }
        for(I k = 0; k < batch_count; k++)
        {
            hptrs[k] = hA[order[k]];
            hptrs[batch_count + k] = hQ[order[k]];
        }
        HIP_CHECK(hipMemcpyAsync(ptrs, hptrs.data(), sizeof(T*) * 2 * batch_count,
                                 hipMemcpyHostToDevice, stream));
        // (the host array must outlive the copy)
        HIP_CHECK(hipStreamSynchronize(stream));
        T* const* gA = ptrs;
        T* const* gQ = ptrs + batch_count;

        // calls fn (which returns a rocblas_status) for each group [g0, g1) of the sorted
        // order, with its range
        auto for_each_group = [&](auto&& fn) -> rocblas_status {
            for(I g0 = 0; g0 < batch_count;)
            {
                I g1 = g0 + 1;
                while(g1 < batch_count && hilo[order[g1]] == hilo[order[g0]]
                      && hihi[order[g1]] == hihi[order[g0]])
                    g1++;
                ROCBLAS_CHECK(fn(g0, g1, hilo[order[g0]], hihi[order[g0]]));
                g0 = g1;
            }
            return rocblas_status_success;
        };

        // reduce to upper Hessenberg form
        ROCBLAS_CHECK(for_each_group([&](const I g0, const I g1, const I ilog,
                                         const I ihig) -> rocblas_status {
            return rocsolver_gehrd_template<true, false, T>(
                handle, n, ilog, ihig, gA + g0, shiftA, lda, rocblas_stride(0), tau + g0 * strideP,
                strideP, g1 - g0, scalars, work1, (T*)work2, work3, (T*)work4, (T*)work5, (T*)work6);
        }));

        // Q (the reflectors are then cleared from A)
        ROCSOLVER_LAUNCH_KERNEL((geev_copy_kernel<T>), grid2, threads2, 0, stream, n, A, shiftA,
                                lda, strideA, Q, shiftQ, ldq, strideQ, wantv, true);
        if(wantv)
            ROCBLAS_CHECK(for_each_group(
                [&](const I g0, const I g1, const I ilog, const I ihig) -> rocblas_status {
                    return rocsolver_orghr_unghr_template<true, false, T>(
                        handle, n, ilog, ihig, gQ + g0, shiftQ, ldq, rocblas_stride(0),
                        tau + g0 * strideP, strideP, g1 - g0, scalars, (T*)work1, (T*)work2,
                        (T*)work3, (T**)work4);
                }));
    }

    // Schur factorization (and Schur vectors Q*Z)
    ROCBLAS_CHECK(rocsolver_hseqr_template<BATCHED, STRIDED, T>(
        handle, wantv ? rocsolver_schur_form : rocsolver_schur_eigenvalues,
        wantv ? rocsolver_schur_vectors_update : rocsolver_schur_vectors_none, n, (const I*)ilo,
        (const I*)ihi, A, shiftA, lda, strideA, W, strideW, Q, shiftQ, ldq, strideQ, info,
        batch_count, (I*)work1, (T*)work2));

    if(wantv)
    {
        // Schur vectors of both sides
        if(leftv && rightv)
            ROCSOLVER_LAUNCH_KERNEL((geev_copy_kernel<T>), grid2, threads2, 0, stream, n, VL, shiftVL,
                                    ldvl, strideVL, VR, shiftVR, ldvr, strideVR, true, false);

        // As in ZGEEV, no eigenvectors are computed if HSEQR failed (info > 0). For a single
        // matrix, info is read back (a synchronization) and the remaining stages are skipped:
        // VL and VR are left with the Schur vectors computed by HSEQR, as in LAPACK. In a batch,
        // the kernels of TREVC3, GEBAK and the normalization skip the matrices with info > 0,
        // so that their VL and VR are also left with the Schur vectors (the documented contents
        // are undefined).
        bool failed = false;
        if(batch_count == 1)
        {
            I hinfo = 0;
            HIP_CHECK(hipMemcpyAsync(&hinfo, info, sizeof(I), hipMemcpyDeviceToHost, stream));
            HIP_CHECK(hipStreamSynchronize(stream));
            failed = hinfo > 0;
        }

        if(!failed)
        {
            // eigenvectors of the Schur form, back-transformed
            const rocblas_side side = leftv && rightv ? rocblas_side_both
                : leftv                               ? rocblas_side_left
                                                      : rocblas_side_right;
            ROCBLAS_CHECK(rocsolver_trevc3_template<BATCHED, STRIDED, T>(
                handle, side, rocsolver_eigenvectors_backtransform, n, A, shiftA, lda, strideA, VL,
                shiftVL, ldvl, strideVL, VR, shiftVR, ldvr, strideVR, batch_count, (T*)work1,
                (T*)work2, (T*)work3, work4, (T**)work5, (const I*)info));

            // undo the balancing, and normalize
            if(leftv)
            {
                ROCBLAS_CHECK(rocsolver_gebak_template<BATCHED, STRIDED, T>(
                    handle, rocsolver_balance_both, rocblas_side_left, n, (const I*)ilo,
                    (const I*)ihi, (const S*)scaleS, strideS, n, VL, shiftVL, ldvl, strideVL,
                    batch_count, (const I*)info));
                ROCSOLVER_LAUNCH_KERNEL((geev_normalize_kernel<BS1, T>), dim3(n, batch_count),
                                        dim3(BS1), 0, stream, n, VL, shiftVL, ldvl, strideVL,
                                        (const I*)info);
            }
            if(rightv)
            {
                ROCBLAS_CHECK(rocsolver_gebak_template<BATCHED, STRIDED, T>(
                    handle, rocsolver_balance_both, rocblas_side_right, n, (const I*)ilo,
                    (const I*)ihi, (const S*)scaleS, strideS, n, VR, shiftVR, ldvr, strideVR,
                    batch_count, (const I*)info));
                ROCSOLVER_LAUNCH_KERNEL((geev_normalize_kernel<BS1, T>), dim3(n, batch_count),
                                        dim3(BS1), 0, stream, n, VR, shiftVR, ldvr, strideVR,
                                        (const I*)info);
            }
        }
    }

    // undo the scaling of the eigenvalues
    ROCSOLVER_LAUNCH_KERNEL((geev_unscale_kernel<T>), dim3((n - 1) / BS1 + 1, batch_count),
                            dim3(BS1), 0, stream, n, W, strideW, (const S*)anrmS, (const I*)ilo,
                            (const I*)info);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

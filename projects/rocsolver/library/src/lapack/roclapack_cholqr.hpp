
/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.9.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2019
 * Copyright (C) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lapack_device_functions.hpp"
#include "rocblas.hpp"
#include "roclapack_potrf.hpp"

ROCSOLVER_BEGIN_NAMESPACE

bool constexpr use_syrk = true;

// kernel to compute the square of g-norm
// which is the max 2-norm square of the columns
//
// max_j  norm( A(:,j),2)^2
//
//
// launch as dim3(1,nby,nbz), dim3(nx,ny,1)
//
// all threads in x-direction in thread block work on
// computing the 2-norm square of a single column
//
// assume nx <= warpsize
// to use DPP instructions
template <typename T, typename I, typename U, typename S = decltype(std::real(T{}))>
static __global__ void cal_gnorm_sq_kernel(const I m,
                                           const I n,
                                           U AA,
                                           const rocblas_stride shiftA,
                                           const I lda,
                                           const rocblas_stride strideA,
                                           S* gnorm_array,
                                           const I batch_count)
{
    I const nx = blockDim.x;
    I const ny = blockDim.y;
    I const nz = blockDim.z;

    I const nbx = gridDim.x;
    I const nby = gridDim.y;
    I const nbz = gridDim.z;

    I const ibx = blockIdx.x;
    I const iby = blockIdx.y;
    I const ibz = blockIdx.z;

    I const tx = threadIdx.x;
    I const ty = threadIdx.y;
    I const tz = threadIdx.z;

    I const i_start = tx;
    I const i_inc = nx;

    I const j_start = ty + iby * ny;
    I const j_inc = ny * nby;

    I const bid_start = ibz;
    I const bid_inc = nbz;

    extern __shared__ double lmem[];
    double* const gnorm_block = (double*)lmem;

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        T const* const A = load_ptr_batch(AA, bid, shiftA, strideA);
        S* const gnorm_bid = gnorm_array + bid;

        bool const use_simple = false;
        if(use_simple)
        {
            // -------------------------
            // use one thread per column
            // -------------------------
            I const txyz = tx + ty * nx + tz * (nx * ny);
            I const nxyz = (nx * ny) * nz;
            I const jcol_start = txyz + iby * nxyz;
            I const jcol_inc = nxyz * nby;

            if(txyz == 0)
            {
                gnorm_block[0] = 0;
            }
            __syncthreads();

            double gnorm_j = 0;
            for(I jcol = jcol_start; jcol < n; jcol += jcol_inc)
            {
                double norm_j = 0;
                for(I i = 0; i < m; i++)
                {
                    auto const ij = idx2D(i, jcol, lda);
                    norm_j += std::norm(A[ij]);
                }
                gnorm_j = rocblas_max_nan(gnorm_j, norm_j);
            }
            atomicMax(gnorm_block, gnorm_j);
            __syncthreads();

            if(txyz == 0)
            {
                atomicMax(gnorm_bid, static_cast<S>(gnorm_block[0]));
            }
        }
        else
        {
            // ----------------------------------------
            // all threads in x-dimension of the block work together
            // to compute the norm of j-th column, which is sum(  A(:,j).^2 )
            // ----------------------------------------

            // -----------------------------------------
            // (1) compute max column norm in warp as  gnorm_j
            // (2) compute max column norm in block as gnorm_block
            // (3) compute max column norm of matrix[bid] in batch as gnorm_bid
            // -----------------------------------------
            if((tx == 0) && (ty == 0) && (tz == 0))
            {
                gnorm_block[0] = 0;
            }
            __syncthreads();

            double gnorm_j = 0;
            for(I j = j_start; j < n; j += j_inc)
            {
                double norm_j = 0;
                for(I i = i_start; i < m; i += i_inc)
                {
                    auto const ij = idx2D(i, j, lda);
                    norm_j += std::norm(A[ij]);
                }

                // ----------------------------------------
                // note: only tx == 0 has the correct value
                // ----------------------------------------
                norm_j += shift_left(norm_j, 1);
                norm_j += shift_left(norm_j, 2);
                norm_j += shift_left(norm_j, 4);
                norm_j += shift_left(norm_j, 8);
                norm_j += shift_left(norm_j, 16);
                if(warpSize > 32)
                    norm_j += shift_left(norm_j, 32);
                if(tx == 0)
                {
                    gnorm_j = std::max(gnorm_j, norm_j);
                }
            }

            if(tx == 0)
            {
                atomicMax(gnorm_block, gnorm_j);
            }
            __syncthreads();

            if((tx == 0) && (ty == 0) && (tz == 0))
            {
                atomicMax(gnorm_bid, static_cast<S>(gnorm_block[0]));
            }
        }
    }
}

// -------------------------------------
// scale an array
// launch as dim3(nbx,1,1), dim3(nx,1,1)
// -------------------------------------
template <typename S, typename I>
static __global__ void scale_kernel(I const batch_count, S const dscale, S* const gnorm_array)
{
    I const bid_start = threadIdx.x + blockIdx.x * blockDim.x;
    I const bid_inc = blockDim.x * gridDim.x;

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        gnorm_array[bid] *= dscale;
    }
}

// ---------------------------------------
// routine to compute the sigma values
//
// sigma values computed based on paper
// "An improved Shifted CholeskyQR based on columns"
// by Yuwei Fan, Haoran Guan, Zhonghua Qiao
//
// sigma = 11 * n * u (m + (n+1) ) * gnorm(A)^2
//
// where u is machine epsilon
// ---------------------------------------
template <typename T, typename I, typename U, typename S = decltype(std::real(T{}))>
static rocblas_status cal_sigma(rocblas_handle handle,
                                const I m,
                                const I n,
                                U A,
                                const rocblas_stride shiftA,
                                const I lda,
                                const rocblas_stride strideA,
                                S* sigma,
                                const I batch_count)
{
    // note: sigma == nullptr treated as no shift
    if(sigma == nullptr)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    const hipDeviceProp_t* props = rocblas_internal_get_device_prop(handle);

    // compute square of gnorm
    // reuse sigma
    size_t const size_sigma = sizeof(S) * batch_count;
    HIP_CHECK(hipMemsetAsync(sigma, 0, size_sigma, stream));

    I const lds_size = sizeof(S);

    I const max_threads = 1024;
    I const nx = props->warpSize; // note nx == warp_size is necessary for correctness
    I const ny = max_threads / nx;
    I const nz = 1;

    I const max_blocks = props->multiProcessorCount;
    I const nbx = 1; // note nbx == 1 is necessary for correctness
    I const nby = std::min(max_blocks, ceil(n, ny));
    I const nbz = std::min(max_blocks, batch_count);

    ROCSOLVER_LAUNCH_KERNEL((cal_gnorm_sq_kernel<T>), dim3(nbx, nby, nbz), dim3(nx, ny, nz),
                            lds_size, stream, m, n, A, shiftA, lda, strideA, sigma, batch_count);

    // sigma = 11 * n * u (m + (n+1) ) * gnorm(A)^2
    S const eps = std::numeric_limits<S>::epsilon();
    S const dscale = 11.0 * n * eps * (m + (n + 1));

    I const nx_scale = 64;
    I const nbx_scale = ceil(batch_count, nx_scale);
    ROCSOLVER_LAUNCH_KERNEL((scale_kernel<S>), dim3(nbx_scale, 1, 1), dim3(nx_scale, 1, 1), 0,
                            stream, batch_count, dscale, sigma);

    return rocblas_status_success;
}

// ---------------------------------
// kernel to perform B <- B + sigma * identity
//
// launch as dim3(nbx,1,batch_count), dim3(nx,1,1)
// ---------------------------------
template <typename T, typename I, typename U, typename S = decltype(std::real(T{}))>
static __global__ void add_shift_kernel(const I m,
                                        const I n,
                                        U BB,
                                        const rocblas_stride shiftB,
                                        const I ldb,
                                        const rocblas_stride strideB,
                                        S* sigma_array,
                                        const I batch_count)
{
    I const bid_start = blockIdx.z;
    I const bid_inc = gridDim.z;

    I const i_start = threadIdx.x + blockIdx.x * blockDim.x;
    I const i_inc = blockDim.x * gridDim.x;

    I const min_mn = std::min(m, n);

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        auto const B = load_ptr_batch(BB, bid, shiftB, strideB);

        // note: ignore null and negative shifts
        S const sigma = (sigma_array == nullptr) ? 0 : max(sigma_array[bid], 0);

        if(sigma != 0)
        {
            for(I i = i_start; i < min_mn; i += i_inc)
            {
                // diagonal entry
                auto const ij = idx2D(i, i, ldb);
                B[ij] += sigma;
            }
        }
    }
}

// --------------------------------------------
// routine to perform B <- B + sigma * identity
// --------------------------------------------
template <typename T, typename I, typename U, typename S = decltype(std::real(T{}))>
static void add_shift(rocblas_handle handle,
                      I const m,
                      I const n,
                      I const batch_count,
                      S* const sigma,
                      U B,
                      rocblas_stride const shiftB,
                      I const ldb,
                      rocblas_stride const strideB)
{
    // note: sigma == nullptr treated as no shift
    if(sigma == nullptr)
        return;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I const nx = 64;

    const hipDeviceProp_t* props = rocblas_internal_get_device_prop(handle);
    I const max_blocks = props->multiProcessorCount;
    I const min_mn = std::min(m, n);

    I const nbx = std::min(max_blocks, ceil(min_mn, nx));
    I const nby = 1;
    I const nbz = std::min(max_blocks, batch_count);

    ROCSOLVER_LAUNCH_KERNEL((add_shift_kernel<T>), dim3(nbx, nby, nbz), dim3(nx, 1, 1), 0, stream,
                            m, n, B, shiftB, ldb, strideB, sigma, batch_count);
}

// ----------------------------------------------------
// set_triangular sets
//
// the  *strictly* lower triangular part if uplo == 'L'
// similar to tri(A,-1) = alpha
//
// the  *strictly* upper triangular part if uplo == 'U'
// similar to triu(A,1) = alpha
//
// the entire matrix if uplo == 'G'
// similar to A = alpha
//
// laset is used by adjusting the indices by 1 for
// the lower and upper case
// ----------------------------------------------------
template <typename T, typename I, typename U>
static void set_triangular(rocblas_handle handle,
                           const rocblas_fill uplo,
                           const I m,
                           const I n,
                           const T alpha,
                           U A,
                           const rocblas_stride shiftA,
                           const I lda,
                           const rocblas_stride strideA,
                           const I batch_count)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_stride const offset = (uplo == rocblas_fill_lower) ? idx2D(1, 0, lda)
        : (uplo == rocblas_fill_upper)                         ? idx2D(0, 1, lda)
                                                               : 0;

    I const mm = (uplo != rocblas_fill_full) ? m - 1 : m;
    I const nn = (uplo != rocblas_fill_full) ? n - 1 : n;

    I const max_threads = 256;
    I const nx = (m <= 32) ? 32 : 64;
    I const ny = max_threads / nx;

    I const max_blocks = 1024;
    I const nbx = std::min(max_blocks, ceil(m, nx));
    I const nby = std::min(max_blocks, ceil(n, ny));
    I const nbz = std::min(max_blocks, batch_count);

    ROCSOLVER_LAUNCH_KERNEL((laset_kernel<T>), dim3(nbx, nby, nbx), dim3(nx, ny, 1), 0, stream, uplo,
                            mm, nn, alpha, alpha, A, shiftA + offset, lda, strideA, batch_count);
}

template <typename T, typename I, typename U, typename S = decltype(std::real(T{}))>
rocblas_status rocsolver_cholqr_argCheck(rocblas_handle handle,
                                         const I m,
                                         const I n,
                                         U A,
                                         const I lda,
                                         const rocblas_stride strideA,
                                         T* R,
                                         const I ldr,
                                         const rocblas_stride strideR,
                                         S* sigma,
                                         const rocsolver_alg_select algo,
                                         I* info,
                                         const I batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(algo != rocsolver_alg_select1 && algo != rocsolver_alg_select2
       && algo != rocsolver_alg_select3 && algo != rocsolver_alg_select4)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(m < 0 || n < 0 || lda < m || ldr < n || batch_count < 0)
        return rocblas_status_invalid_size;
    // only m >= n is supported
    if(m > 0 && m < n)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((m && n && (!A || !R)) || (batch_count > 0 && !info))
        return rocblas_status_invalid_pointer;
    // sigma is required for cholqr3 algorithms
    if(batch_count > 0 && (algo == rocsolver_alg_select3 || algo == rocsolver_alg_select4) && !sigma)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename I>
static rocblas_status rocsolver_cholqr_getMemorySize(const I m,
                                                     const I n,
                                                     const I lda,
                                                     const I ldr,
                                                     const I batch_count,
                                                     const rocsolver_alg_select algo,
                                                     size_t* size_scalars,
                                                     size_t* size_work1,
                                                     size_t* size_work2,
                                                     size_t* size_work3,
                                                     size_t* size_work4,
                                                     size_t* size_pivots,
                                                     size_t* size_iinfo,
                                                     size_t* size_R1,
                                                     size_t* size_workArr,
                                                     bool* optim_mem)
{
    // if quick return, no workspace is needed
    if(m == 0 || n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_pivots = 0;
        *size_iinfo = 0;
        *size_R1 = 0;
        *size_workArr = 0;
        *optim_mem = true;
        return rocblas_status_success;
    }

    // ---- requirements for CHOLQR1 ----
    // storage for Cholesky factorization R = chol(B)
    rocsolver_potrf_getMemorySize<BATCHED, STRIDED, T>(
        n, rocblas_fill_upper, batch_count, size_scalars, size_work1, size_work2, size_work3,
        size_work4, size_pivots, size_iinfo, optim_mem);

    // storage for computing Q = A / R
    size_t w1 = 0, w2 = 0, w3 = 0, w4 = 0;
    ROCBLAS_CHECK(rocblasCall_trsm_mem<BATCHED, T>(rocblas_side_right, rocblas_operation_none, m, n,
                                                   ldr, lda, batch_count, &w1, &w2, &w3, &w4));
    *size_work1 = std::max(*size_work1, w1);
    *size_work2 = std::max(*size_work2, w2);
    *size_work3 = std::max(*size_work3, w3);
    *size_work4 = std::max(*size_work4, w4);

    if(algo == rocsolver_alg_select1)
    {
        // storage for R1 not needed
        *size_R1 = 0;
    }
    else
    {
        // ---- requirements for CHOLQR2 ----
        // storage for iinfo, intended for 2nd call to cholqr1(A)
        *size_iinfo += sizeof(I) * batch_count;

        // storage for R1, in computing [Q,R1] = cholqr1(A)
        *size_R1 = sizeof(T) * n * n * batch_count;

        if((algo == rocsolver_alg_select3) || (algo == rocsolver_alg_select4))
        {
            // ---- requirements for CHOLQR3 ----
            // extra space for iinfo and second copy of R1
            *size_iinfo += sizeof(I) * batch_count;
            *size_R1 += sizeof(T) * n * n * batch_count;
        }
    }

    // size of array of pointers to workspace
    if(BATCHED)
        *size_workArr = sizeof(T*) * batch_count;
    else
        *size_workArr = 0;

    return rocblas_status_success;
}

// -------------------------------------------------
// compute A = Q * R,  using Cholesky factorization
//
// B = A' * A
//
// R = chol(B)
//
// Q = A / R
//
// Q will over-write A
// -------------------------------------------------
template <bool BATCHED,
          bool STRIDED,
          typename T,
          typename I,
          typename UA,
          typename UR,
          typename INFO = I,
          typename S = decltype(std::real(T{}))>
static rocblas_status rocsolver_cholqr1_template(rocblas_handle handle,
                                                 I const m,
                                                 I const n,
                                                 UA A,
                                                 rocblas_stride const shiftA,
                                                 I const lda,
                                                 rocblas_stride strideA,
                                                 UR R,
                                                 rocblas_stride const shiftR,
                                                 I const ldr,
                                                 rocblas_stride strideR,
                                                 I const batch_count,
                                                 I* const info,
                                                 T* scalars,
                                                 void* work1,
                                                 void* work2,
                                                 void* work3,
                                                 void* work4,
                                                 T* pivots,
                                                 I* iinfo,
                                                 T** workArr,
                                                 bool optim_mem,
                                                 S* const sigma_array = nullptr)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    const T zero = T(0);
    const T one = T(1);
    const S Szero = S(0);
    const S Sone = S(1);

    // compute B = A' * A
    // B is stored in R
    if constexpr(use_syrk)
    {
        // Note output matrix for SYRK is n by n
        ROCBLAS_CHECK(rocblasCall_syrk_herk<BATCHED, T>(
            handle, rocblas_fill_upper, rocblas_operation_conjugate_transpose, n, m, &Sone, A,
            shiftA, lda, strideA, &Szero, R, shiftR, ldr, strideR, batch_count, workArr));
    }
    else
    {
        ROCBLAS_CHECK(rocblasCall_gemm<T>(handle, rocblas_operation_conjugate_transpose,
                                          rocblas_operation_none, n, n, m, &one, A, shiftA, lda,
                                          strideA, A, shiftA, lda, strideA, &zero, R, shiftR, ldr,
                                          strideR, batch_count, workArr));
    }

    // optional, if sigma != 0
    // B <- B + sigma * identity
    if(sigma_array != nullptr)
        add_shift<T>(handle, m, n, batch_count, sigma_array, R, shiftR, ldr, strideR);

    // perform Cholesky factorization
    // B = R' * R,   R is upper triangular
    // R will over-write B
    ROCBLAS_CHECK(rocsolver_potrf_template<false, true, T, I, I, S>(
        handle, rocblas_fill_upper, n, R, shiftR, ldr, strideR, info, batch_count, scalars, work1,
        work2, work3, work4, pivots, iinfo, optim_mem));

    // compute Q = A / R
    // note Q over-writes original matrix A
    ROCBLAS_CHECK(rocblasCall_trsm<T>(handle, rocblas_side_right, rocblas_fill_upper,
                                      rocblas_operation_none, rocblas_diagonal_non_unit, m, n, &one,
                                      R, shiftR, ldr, strideR, A, shiftA, lda, strideA, batch_count,
                                      optim_mem, work1, work2, work3, work4, workArr));

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

// -----------------------------------------------------
// perform QR factorization using cholesky factorization
// (1) [Q,R1] = cholqr1( A )
// (2) [Q,R] = cholqr1( Q )
// (3) R = R * R1
//
//
// "Roundoff error analysis of the CholeskyQR2 algorithm",
// by Yamamoto et al, Electronic Transactions on Numerical Analysis,
// Vol 44, p 306-326, 2015.
// -----------------------------------------------------

template <bool BATCHED, bool STRIDED, typename T, typename I, typename UA, typename UR, typename INFO = I>
static rocblas_status rocsolver_cholqr2_template(rocblas_handle handle,
                                                 I const m,
                                                 I const n,
                                                 UA A,
                                                 rocblas_stride const shiftA,
                                                 I const lda,
                                                 rocblas_stride strideA,
                                                 UR R,
                                                 rocblas_stride const shiftR,
                                                 I const ldr,
                                                 rocblas_stride strideR,
                                                 I const batch_count,
                                                 INFO* const info,
                                                 T* scalars,
                                                 void* work1,
                                                 void* work2,
                                                 void* work3,
                                                 void* work4,
                                                 T* pivots,
                                                 I* iinfo,
                                                 T* R1,
                                                 T** workArr,
                                                 bool optim_mem)
{
    const T zero = T(0);
    const T one = T(1);

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    // (1) [Q,R1] = cholqr1( A )
    ROCBLAS_CHECK(rocsolver_cholqr1_template<BATCHED, STRIDED, T>(
        handle, m, n, A, shiftA, lda, strideA, R1, 0, n, n * n, batch_count, info, scalars, work1,
        work2, work3, work4, pivots, iinfo, workArr, optim_mem));

    // (2) [Q,R] = cholqr1( Q )
    // Note: matrix Q over-writes matrix A
    ROCBLAS_CHECK(rocsolver_cholqr1_template<BATCHED, STRIDED, T>(
        handle, m, n, A, shiftA, lda, strideA, R, shiftR, ldr, strideR, batch_count, iinfo, scalars,
        work1, work2, work3, work4, pivots, iinfo + batch_count, workArr, optim_mem));

    // set strictly lower triangular part of R to be zero
    set_triangular(handle, rocblas_fill_lower, n, n, zero, R, shiftR, ldr, strideR, batch_count);

    // R <- R * R1
    ROCBLAS_CHECK(rocblasCall_trmm<T>(handle, rocblas_side_right, rocblas_fill_upper,
                                      rocblas_operation_none, rocblas_diagonal_non_unit, n, n, &one,
                                      0, R1, 0, n, n * n, R, shiftR, ldr, strideR, batch_count,
                                      workArr));

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

//
// shifted CholeskyQR3
//
// (1)  R1 * R1' = A' * A + s * I, where s is the shift
// (2)  Q1 = A/R1
// (3)  [Q2, R2]  = cholQR2(Q1)
// (4)  R = R2 * R1
//
// "Shifted CholeskyQR for computing QR factorization of
// ill-conditioned matrices", Fukaya et al,
// SIAM J Sci Comp, Vol 42, No 1, pp A477-A503, 2020
//
// "An improved Shifted CholeskyQR based on columns",
// by Fan et al, arXiv:2408.06311v4 [math.NA] 07 Feb 2025
//
template <bool BATCHED,
          bool STRIDED,
          typename T,
          typename I,
          typename UA,
          typename UR,
          typename INFO = I,
          typename S = decltype(std::real(T{}))>
static rocblas_status rocsolver_cholqr3_template(rocblas_handle handle,
                                                 I const m,
                                                 I const n,
                                                 UA A,
                                                 rocblas_stride const shiftA,
                                                 I const lda,
                                                 rocblas_stride strideA,
                                                 UR R,
                                                 rocblas_stride const shiftR,
                                                 I const ldr,
                                                 rocblas_stride strideR,
                                                 bool const compute_sigma,
                                                 S* const sigma_array,
                                                 INFO* const info,
                                                 I const batch_count,
                                                 T* scalars,
                                                 void* work1,
                                                 void* work2,
                                                 void* work3,
                                                 void* work4,
                                                 T* pivots,
                                                 I* iinfo,
                                                 T* R1,
                                                 T** workArr,
                                                 bool optim_mem)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    const T zero = T(0);
    const T one = T(1);

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    // (1)  R1 * R1' = A'*A + sigma * identity
    // Note: paper suggests
    // T const sigma = 11 * (m * n * ueps + (n + 1) * (n * ueps)) * gnorm
    if(compute_sigma)
        ROCBLAS_CHECK(
            cal_sigma<T, I>(handle, m, n, A, shiftA, lda, strideA, sigma_array, batch_count));

    // perform CholeskQR1 with shift
    ROCBLAS_CHECK(rocsolver_cholqr1_template<BATCHED, STRIDED, T>(
        handle, m, n, A, shiftA, lda, strideA, R1, 0, n, n * n, batch_count, info, scalars, work1,
        work2, work3, work4, pivots, iinfo, workArr, optim_mem));

    // (2)   CholQR2(Q)
    // Note: matrix Q is stored in matrix A
    ROCBLAS_CHECK(rocsolver_cholqr2_template<BATCHED, STRIDED, T>(
        handle, m, n, A, shiftA, lda, strideA, R, shiftR, ldr, strideR, batch_count, iinfo, scalars,
        work1, work2, work3, work4, pivots, iinfo + batch_count, R1 + n * n * batch_count, workArr,
        optim_mem));

    // (i) set strictly lower triangular part of R be zero
    set_triangular(handle, rocblas_fill_lower, n, n, zero, R, shiftR, ldr, strideR, batch_count);

    // R <- R * R1
    ROCBLAS_CHECK(rocblasCall_trmm<T>(handle, rocblas_side_right, rocblas_fill_upper,
                                      rocblas_operation_none, rocblas_diagonal_non_unit, n, n, &one,
                                      0, R1, 0, n, n * n, R, shiftR, ldr, strideR, batch_count,
                                      workArr));

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename U, typename S = decltype(std::real(T{}))>
static rocblas_status rocsolver_cholqr_template(rocblas_handle handle,
                                                const I m,
                                                const I n,
                                                U A,
                                                const rocblas_stride shiftA,
                                                const I lda,
                                                const rocblas_stride strideA,
                                                T* R,
                                                const rocblas_stride shiftR,
                                                const I ldr,
                                                const rocblas_stride strideR,
                                                S* sigma,
                                                const rocsolver_alg_select algo,
                                                I* info,
                                                const I batch_count,
                                                T* scalars,
                                                void* work1,
                                                void* work2,
                                                void* work3,
                                                void* work4,
                                                T* pivots,
                                                I* iinfo,
                                                T* R1,
                                                T** workArr,
                                                bool optim_mem)

{
    // quick return
    if(m == 0 || n == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    // set info=0
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

    // quick return if no dimensions
    if(m == 0 || n == 0)
        return rocblas_status_success;

    if(algo == rocsolver_alg_select1)
    {
        return rocsolver_cholqr1_template<BATCHED, STRIDED, T>(
            handle, m, n, A, shiftA, lda, strideA, R, shiftR, ldr, strideR, batch_count, info,
            scalars, work1, work2, work3, work4, pivots, iinfo, workArr, optim_mem);
    }
    else if(algo == rocsolver_alg_select2)
    {
        return rocsolver_cholqr2_template<BATCHED, STRIDED, T>(
            handle, m, n, A, shiftA, lda, strideA, R, shiftR, ldr, strideR, batch_count, info,
            scalars, work1, work2, work3, work4, pivots, iinfo, R1, workArr, optim_mem);
    }
    else
    {
        bool const compute_sigma = (algo == rocsolver_alg_select3);
        return rocsolver_cholqr3_template<BATCHED, STRIDED, T>(
            handle, m, n, A, shiftA, lda, strideA, R, shiftR, ldr, strideR, compute_sigma, sigma,
            info, batch_count, scalars, work1, work2, work3, work4, pivots, iinfo, R1, workArr,
            optim_mem);
    }
}

ROCSOLVER_END_NAMESPACE

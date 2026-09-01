/************************************************************************
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

#include "lapack_device_functions.hpp"
#include "rocblas.hpp"
#include "roclapack_potrf.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/****************************************************************************/
/******************* Kernels and compute functions **************************/
/****************************************************************************/

//---------------------------------------------
// This kernel updates the values in nr after
// when needed during refinement
//---------------------------------------------
template <typename I>
ROCSOLVER_KERNEL void cholqr_updatenr_kernel(const I n, I* nrA, I* infoA, const I batch_count)
{
    I const b_start = threadIdx.x + blockIdx.x * blockDim.x;
    I const b_inc = blockDim.x * gridDim.x;

    for(auto b = b_start; b < batch_count; b += b_inc)
    {
        I nr = infoA[b];
        nr = (nr == 0) ? n : nr - 1;
        infoA[b] = nrA[b];
        nrA[b] = nr;
    }
}

//---------------------------------------------
// Kernel to clean the cholesky factor when
// info > 0
//---------------------------------------------
template <typename I, typename T>
ROCSOLVER_KERNEL void cholqr_cleannr_w_kernel(const I m,
                                              const I n,
                                              const I mn,
                                              I* nrA,
                                              I* infoA,
                                              T* WA,
                                              const rocblas_stride shiftW,
                                              const I ldw,
                                              const rocblas_stride strideW,
                                              const I batch_count,
                                              const bool set_nr,
                                              const bool update_nr)
{
    I const b_start = threadIdx.z + blockIdx.z * blockDim.z;
    I const b_inc = blockDim.z * gridDim.z;
    I const j_start = threadIdx.y + blockIdx.y * blockDim.y;
    I const j_inc = blockDim.y * gridDim.y;
    I const i_start = threadIdx.x + blockIdx.x * blockDim.x;
    I const i_inc = blockDim.x * gridDim.x;
    const bool upper = (mn == n);

    for(auto b = b_start; b < batch_count; b += b_inc)
    {
        I info = infoA[b];
        info = (info == 0) ? mn : info - 1;
        I nr = nrA[b];

        if(set_nr)
        {
            nr = info;
            if(i_start == 0 && j_start == 0)
                nrA[b] = nr;
        }
        else if(update_nr)
            nr = (info < nr) ? info : nr;

        T* W = load_ptr_batch(WA, b, shiftW, strideW);
        for(auto j = j_start; j < mn; j += j_inc)
        {
            for(auto i = i_start; i < mn; i += i_inc)
            {
                if(i < nr && j < nr)
                {
                    if((i > j && upper) || (i < j && !upper))
                        W[i + j * ldw] = 0;
                }
                else
                {
                    T val = (i == j) ? 1 : 0;
                    W[i + j * ldw] = val;
                }
            }
        }
    }
}

//---------------------------------------------
// This kernel restore columns of A in Q and
// cleans the cholesky factor accordingly.
//---------------------------------------------
template <typename I, typename T, typename U>
ROCSOLVER_KERNEL void cholqr_cleannr_q_kernel(const I m,
                                              const I n,
                                              I* nrA,
                                              I* infoA,
                                              U QA,
                                              const rocblas_stride shiftQ,
                                              const I ldq,
                                              rocblas_stride strideQ,
                                              T* WA,
                                              const rocblas_stride shiftW,
                                              const I ldw,
                                              const rocblas_stride strideW,
                                              T* AA,
                                              const I batch_count)
{
    I const b_start = threadIdx.z + blockIdx.z * blockDim.z;
    I const b_inc = blockDim.z * gridDim.z;
    I j0 = threadIdx.y + blockIdx.y * blockDim.y;
    I const j_inc = blockDim.y * gridDim.y;
    I i0 = threadIdx.x + blockIdx.x * blockDim.x;
    I const i_inc = blockDim.x * gridDim.x;
    I j_start, j_end, i_start, i_end;
    I mn = std::min(m, n);

    for(auto b = b_start; b < batch_count; b += b_inc)
    {
        T* W = load_ptr_batch(WA, b, shiftW, strideW);
        T* Q = load_ptr_batch(QA, b, shiftQ, strideQ);
        T* A = AA + b * m * n;
        I end = infoA[b];
        I nr = nrA[b];

        if(mn == n)
        {
            // case m >= n
            j_start = j0 + nr;
            j_end = end;
            i_start = i0;
            i_end = m;
        }
        else
        {
            // case m < n
            i_start = i0 + nr;
            i_end = end;
            j_start = j0;
            j_end = n;
        }

        for(auto j = j_start; j < j_end; j += j_inc)
        {
            for(auto i = i_start; i < i_end; i += i_inc)
            {
                // restore columns of A
                Q[i + j * ldq] = A[i + j * m];

                // clean R
                if(i < mn && j < mn)
                {
                    T val = (i == j) ? 1 : 0;
                    W[i + j * ldw] = val;
                }
            }
        }
    }
}

//---------------------------------------------
// kernel to compute the square of g-norm
// which is the max 2-norm square of the columns
// max_j  norm( A(:,j),2)^2
//
// launch as dim3(1,nby,nbz), dim3(nx,ny,1)
// all threads in x-direction in thread block work on
// computing the 2-norm square of a single column
// assume nx <= warpsize
// to use DPP instructions
//---------------------------------------------
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
    S* const gnorm_block = (S*)lmem;

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
                    gnorm_j = rocblas_max_nan(gnorm_j, norm_j);
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
// this kernel scales an array
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
        S const sigma = (sigma_array == nullptr) ? S(0) : std::max(sigma_array[bid], S(0));

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

/****************************************************************************/
/****************************  Host main functions **************************/
/****************************************************************************/

template <typename T, typename I, typename U, typename S = decltype(std::real(T{}))>
rocblas_status rocsolver_cholqr_argCheck(rocblas_handle handle,
                                         const rocsolver_cholqr_shift cholshift,
                                         const rocblas_int cholnum,
                                         const I m,
                                         const I n,
                                         U A,
                                         const I lda,
                                         const rocblas_stride strideA,
                                         T* W,
                                         const I ldw,
                                         const rocblas_stride strideW,
                                         S* sigma,
                                         I* nr,
                                         const I batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(cholshift != rocsolver_cholqr_shift_none && cholshift != rocsolver_cholqr_shift_computed
       && cholshift != rocsolver_cholqr_shift_provided)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(m < 0 || n < 0 || lda < m || ldw < std::min(m, n) || batch_count < 0)
        return rocblas_status_invalid_size;
    // number of cholesky factorizations must be at least 1
    // or 2 if cholshift != rocsolver_cholqr_shift_none
    if(cholnum < 1)
        return rocblas_status_invalid_size;
    if(cholnum < 2
       && (cholshift == rocsolver_cholqr_shift_computed
           || cholshift == rocsolver_cholqr_shift_provided))
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((m && n && (!A || !W)) || (batch_count > 0 && !nr))
        return rocblas_status_invalid_pointer;
    // sigma is required for shifted cases
    if(batch_count > 0 && !sigma
       && (cholshift == rocsolver_cholqr_shift_computed
           || cholshift == rocsolver_cholqr_shift_provided))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename I>
static rocblas_status rocsolver_cholqr_getMemorySize(const rocsolver_cholqr_shift cholshift,
                                                     const rocblas_int cholnum,
                                                     const I m,
                                                     const I n,
                                                     const I lda,
                                                     const I ldw,
                                                     const I batch_count,
                                                     size_t* size_scalars,
                                                     size_t* size_work1,
                                                     size_t* size_work2,
                                                     size_t* size_work3,
                                                     size_t* size_work4,
                                                     size_t* size_pivots,
                                                     size_t* size_iinfo,
                                                     size_t* size_W1,
                                                     size_t* size_Acpy,
                                                     size_t* size_workArr,
                                                     bool* optim_mem)
{
    *size_scalars = 0;
    *size_work1 = 0;
    *size_work2 = 0;
    *size_work3 = 0;
    *size_work4 = 0;
    *size_pivots = 0;
    *size_iinfo = 0;
    *size_W1 = 0;
    *size_Acpy = 0;
    *size_workArr = 0;
    *optim_mem = true;

    // if quick return, no workspace is needed
    if(m == 0 || n == 0 || batch_count == 0)
        return rocblas_status_success;

    rocblas_side side;
    rocblas_fill uplo;
    I mn = std::min(m, n);
    if(mn == n)
    {
        // case min(m, n) == n
        uplo = rocblas_fill_upper;
        side = rocblas_side_right;
    }
    else
    {
        // case min(m, n) == m
        uplo = rocblas_fill_lower;
        side = rocblas_side_left;
    }

    // storage for Cholesky factorization R = chol(B)
    rocsolver_potrf_getMemorySize<BATCHED, STRIDED, T>(mn, uplo, batch_count, size_scalars,
                                                       size_work1, size_work2, size_work3, size_work4,
                                                       size_pivots, size_iinfo, optim_mem);

    // storage for computing Q = A / R
    size_t w1 = 0, w2 = 0, w3 = 0, w4 = 0;
    ROCBLAS_CHECK(rocblasCall_trsm_mem<BATCHED, T>(side, rocblas_operation_none, m, n, ldw, lda,
                                                   batch_count, &w1, &w2, &w3, &w4));
    *size_work1 = std::max(*size_work1, w1);
    *size_work2 = std::max(*size_work2, w2);
    *size_work3 = std::max(*size_work3, w3);
    *size_work4 = std::max(*size_work4, w4);

    // additional storage for temporary values
    *size_iinfo += sizeof(I) * batch_count;
    if(cholnum > 1)
        *size_W1 = sizeof(T) * mn * mn * batch_count;

    // additional storage for a copy of A when needed
    if(cholshift != rocsolver_cholqr_shift_none)
        *size_Acpy = sizeof(T) * m * n * batch_count;

    // size of array of pointers to workspace
    if(BATCHED)
        *size_workArr = sizeof(T*) * batch_count;

    return rocblas_status_success;
}

// -------------------------------------------------
// CholQR factorization step.
// compute A = Q * W (or A = W * Q) where W is upper
// (lower) triangular and Q has orthonormal columns (rows)
//
// B = A' * A (or B = A * A')
// W = chol(B)
// Q is solution of upper triangular system  A = QW, (or
// lower triangular system A = WQ)
//
// Q will over-write A
// -------------------------------------------------
template <bool BATCHED, bool STRIDED, typename T, typename I, typename U, typename S = decltype(std::real(T{}))>
static rocblas_status rocsolver_cholqr1_template(rocblas_handle handle,
                                                 I const m,
                                                 I const n,
                                                 U A,
                                                 rocblas_stride const shiftA,
                                                 I const lda,
                                                 rocblas_stride strideA,
                                                 T* W,
                                                 rocblas_stride const shiftW,
                                                 I const ldw,
                                                 rocblas_stride strideW,
                                                 S* const sigma_array,
                                                 I* const nr,
                                                 I const batch_count,
                                                 T* scalars,
                                                 void* work1,
                                                 void* work2,
                                                 void* work3,
                                                 void* work4,
                                                 T* pivots,
                                                 I* iinfo,
                                                 T** workArr,
                                                 const bool optim_mem,
                                                 const bool add_sigma,
                                                 const bool set_nr,
                                                 const bool update_nr)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    const T zero = T(0);
    const T one = T(1);
    const S Szero = S(0);
    const S Sone = S(1);
    I mn = std::min(m, n);
    I MN = std::max(m, n);

    rocblas_side side;
    rocblas_operation trans1, trans2;
    rocblas_fill uplo;
    if(mn == n)
    {
        // case m >= n
        side = rocblas_side_right;
        trans1 = rocblas_operation_conjugate_transpose;
        trans2 = rocblas_operation_none;
        uplo = rocblas_fill_upper;
    }
    else
    {
        // case m < n
        side = rocblas_side_left;
        trans1 = rocblas_operation_none;
        trans2 = rocblas_operation_conjugate_transpose;
        uplo = rocblas_fill_lower;
    }

    // compute B = A' * A  (or B = A * A')
    // B is stored in W
    ROCBLAS_CHECK(rocblasCall_gemm<T>(handle, trans1, trans2, mn, mn, MN, &one, A, shiftA, lda,
                                      strideA, A, shiftA, lda, strideA, &zero, W, shiftW, ldw,
                                      strideW, batch_count, workArr));

    // optional, if sigma != 0
    // B <- B + sigma * identity
    if(add_sigma)
        add_shift<T>(handle, m, n, batch_count, sigma_array, W, shiftW, ldw, strideW);

    // perform Cholesky factorization
    // B = W' * W with W upper triangular (or B = W * W' with W lower triangular)
    // W will over-write B
    ROCBLAS_CHECK(rocsolver_potrf_template<false, true, T, I, I, S>(
        handle, uplo, mn, W, shiftW, ldw, strideW, iinfo, batch_count, scalars, work1, work2, work3,
        work4, pivots, iinfo + batch_count, optim_mem));

    // clean cholesky factor W if factorization of all columns (rows) failed
    I max_blocks = 1024;
    I thdx = 16, thdy = 16;
    I blkx = std::min(max_blocks, ceil(mn, thdx));
    I blky = std::min(max_blocks, ceil(mn, thdy));
    I blkz = std::min(max_blocks, batch_count);
    ROCSOLVER_LAUNCH_KERNEL(cholqr_cleannr_w_kernel, dim3(blkx, blky, blkz), dim3(thdx, thdy, 1), 0,
                            stream, m, n, mn, nr, iinfo, W, shiftW, ldw, strideW, batch_count,
                            set_nr, update_nr);

    // compute Q by solving triangular system
    // note Q over-writes original matrix A
    ROCBLAS_CHECK(rocblasCall_trsm<T>(handle, side, uplo, rocblas_operation_none,
                                      rocblas_diagonal_non_unit, m, n, &one, W, shiftW, ldw,
                                      strideW, A, shiftA, lda, strideA, batch_count, optim_mem,
                                      work1, work2, work3, work4, workArr));

    if(update_nr)
    {
        // update values in nr (will happen in first refinement iteration when
        // cholshift is not none)
        blkx = ceil(batch_count, BS1);
        ROCSOLVER_LAUNCH_KERNEL(cholqr_updatenr_kernel, dim3(blkx), dim3(BS1), 0, stream, mn, nr,
                                iinfo, batch_count);
    }

    return rocblas_status_success;
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename U, typename S = decltype(std::real(T{}))>
static rocblas_status rocsolver_cholqr_template(rocblas_handle handle,
                                                const rocsolver_cholqr_shift cholshift,
                                                const rocblas_int cholnum,
                                                const I m,
                                                const I n,
                                                U A,
                                                const rocblas_stride shiftA,
                                                const I lda,
                                                const rocblas_stride strideA,
                                                T* W,
                                                const rocblas_stride shiftW,
                                                const I ldw,
                                                const rocblas_stride strideW,
                                                S* sigma,
                                                I* nr,
                                                const I batch_count,
                                                T* scalars,
                                                void* work1,
                                                void* work2,
                                                void* work3,
                                                void* work4,
                                                T* pivots,
                                                I* iinfo,
                                                T* W1,
                                                T* Acpy,
                                                T** workArr,
                                                bool optim_mem)

{
    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // set nr=0
    I blocks = ceil(batch_count, BS1);
    ROCSOLVER_LAUNCH_KERNEL(reset_info, dim3(blocks), dim3(BS1), 0, stream, nr, batch_count, 0);

    // quick return if no dimensions
    if(m == 0 || n == 0)
        return rocblas_status_success;

    // everything must be executed with scalars on the host
    rocblas_pointer_mode_saver saver(handle, rocblas_pointer_mode_host);

    const T one = T(1);

    I mn = std::min(m, n);
    bool compute_sigma = (cholshift == rocsolver_cholqr_shift_computed);
    bool add_sigma = (cholshift == rocsolver_cholqr_shift_computed
                      || cholshift == rocsolver_cholqr_shift_provided);
    if(compute_sigma)
        ROCBLAS_CHECK(cal_sigma<T, I>(handle, m, n, A, shiftA, lda, strideA, sigma, batch_count));

    // save a copy of A to restore it if necessary when using the shifted algorithm
    I blocksm, blocksn;
    if(add_sigma)
    {
        blocksm = ceil(m, BS2);
        blocksn = ceil(n, BS2);
        ROCSOLVER_LAUNCH_KERNEL((copy_mat<T>), dim3(blocksm, blocksn, batch_count), dim3(BS2, BS2, 1),
                                0, stream, copymat_to_buffer, m, n, A, shiftA, lda, strideA, Acpy);
    }

    // compute initial cholqr step
    bool set_nr = true;
    bool update_nr = false;
    ROCBLAS_CHECK(rocsolver_cholqr1_template<BATCHED, STRIDED, T>(
        handle, m, n, A, shiftA, lda, strideA, W, shiftW, ldw, strideW, sigma, nr, batch_count,
        scalars, work1, work2, work3, work4, pivots, iinfo, workArr, optim_mem, add_sigma, set_nr,
        update_nr));

    // refinement iteration
    // (if the initial cholesky was shifted, the first iteration updates the size nr of the
    // factorization to avoid counting dependent rows/columns in Q)
    rocblas_side side;
    rocblas_fill uplo;
    if(mn == n)
    {
        // case m >= n
        side = rocblas_side_left;
        uplo = rocblas_fill_upper;
    }
    else
    {
        // case m < n
        side = rocblas_side_right;
        uplo = rocblas_fill_lower;
    }
    update_nr = add_sigma;
    add_sigma = false;
    set_nr = false;

    for(auto k = 1; k < cholnum; ++k)
    {
        ROCBLAS_CHECK(rocsolver_cholqr1_template<BATCHED, STRIDED, T>(
            handle, m, n, A, shiftA, lda, strideA, W1, 0, mn, mn * mn, sigma, nr, batch_count,
            scalars, work1, work2, work3, work4, pivots, iinfo, workArr, optim_mem, add_sigma,
            set_nr, update_nr));

        if(update_nr)
        {
            update_nr = false;

            // restore columns of A if necessary during the first refinement iteration
            ROCSOLVER_LAUNCH_KERNEL(cholqr_cleannr_q_kernel, dim3(blocksm, blocksn, batch_count),
                                    dim3(BS2, BS2, 1), 0, stream, m, n, nr, iinfo, A, shiftA, lda,
                                    strideA, W, shiftW, ldw, strideW, Acpy, batch_count);
        }

        ROCBLAS_CHECK(rocblasCall_trmm<T>(handle, side, uplo, rocblas_operation_none,
                                          rocblas_diagonal_non_unit, mn, mn, &one, 0, W1, 0, mn,
                                          mn * mn, W, shiftW, ldw, strideW, batch_count, workArr));
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

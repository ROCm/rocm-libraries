
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

#include "rocblas.hpp"
#include "roclapack_potrf.hpp"

ROCSOLVER_BEGIN_NAMESPACE

bool constexpr use_syrk = true;

static inline void adjust_for_alignment(size_t& size_work)
{
    constexpr size_t ialign = 256;

    size_work = ceildiv(size_work, ialign) * ialign;
}

static inline void adjust_for_alignment(size_t* p_size_work)
{
    size_t size_work = *p_size_work;

    adjust_for_alignment(size_work);

    *p_size_work = size_work;
}

#ifndef IS_POINTER_BATCHED
#define IS_POINTER_BATCHED(A, T) \
    (std::is_pointer_v<std::remove_cv_t<std::remove_cv_t<std::remove_reference_t<decltype((A)[0])>>>>)
#endif

#ifndef MEM_CHECK
#define MEM_CHECK(pfree)                                        \
    {                                                           \
        bool const is_mem_ok_ = (pfree <= (pwork + size_work)); \
        if(!is_mem_ok_)                                         \
        {                                                       \
            return (rocblas_status_memory_error);               \
        }                                                       \
    }
#endif

#ifndef MEM_CHECK_THROW
#define MEM_CHECK_THROW(pfree)                                  \
    {                                                           \
        bool const is_mem_ok_ = (pfree <= (pwork + size_work)); \
        if(!is_mem_ok_)                                         \
        {                                                       \
            istat = rocblas_status_memory_error;                \
            throw(istat);                                       \
        }                                                       \
    }
#endif

static int get_num_cu(int deviceId = 0)
{
    int ival = 0;
    auto const attr = hipDeviceAttributeMultiprocessorCount;
    HIP_CHECK(hipDeviceGetAttribute(&ival, attr, deviceId));
    return (ival);
}

static int get_warp_size(int deviceId = 0)
{
    int ival = 0;
    auto const attr = hipDeviceAttributeWarpSize;
    HIP_CHECK(hipDeviceGetAttribute(&ival, attr, deviceId));
    return (ival);
}

template <typename T, typename I>
__device__ static T reduce_sum_shfl_wsize(I const wsize, T val)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    if(wsize == 64)
    {
        val += __shfl_down(val, 32); // offset = 32
        val += __shfl_down(val, 16); // offset = 16
        val += __shfl_down(val, 8); // offset = 8
        val += __shfl_down(val, 4); // offset = 4
        val += __shfl_down(val, 2); // offset = 2
        val += __shfl_down(val, 1); // offset = 1
    }
    else if(wsize == 32)
    {
        val += __shfl_down(val, 16); // offset = 16
        val += __shfl_down(val, 8); // offset = 8
        val += __shfl_down(val, 4); // offset = 4
        val += __shfl_down(val, 2); // offset = 2
        val += __shfl_down(val, 1); // offset = 1
    }
    else
    {
        for(auto offset = wsize / 2; offset > 0; offset /= 2)
        {
            val += __shfl_down(val, offset);
            // g.sync();
        }
    }
    return val; // note: only thread 0 will return full sum
}

// -----------------------------------------------------
// convert from strided batched storage to pointer batch
//
// launch as <<< dim3( nbx, 1, 1), dim3(nx,1,1), 0, stream >>>
// where nbx = ceildiv( batch_count, nx )
// -----------------------------------------------------
template <typename T, typename I, typename Istride>
__global__ static void copy_array_to_ptrs_kernel(I batch_count,

                                                 T* const B,
                                                 Istride const shiftB,
                                                 I const ldb,
                                                 Istride const strideB,

                                                 T** const B_ptr)
{
    I const bid_start = threadIdx.x + blockIdx.x * blockDim.x;
    I const bid_inc = blockDim.x * gridDim.x;

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        B_ptr[bid] = load_ptr_batch(B, bid, shiftB, strideB);
    }
}

template <typename T, typename I, typename Istride>
static void copy_array_to_ptr(hipStream_t stream,

                              I const batch_count,

                              T* const B,
                              Istride const shiftB,
                              I const ldb,
                              Istride const strideB,

                              T** const B_ptr)
{
    // -----------------------------------------------
    // convert from strided batched to pointer batched
    // -----------------------------------------------
    I const nx = 64;
    I const nbx = ceildiv(batch_count, nx);

    copy_array_to_ptrs_kernel<T, I, Istride><<<dim3(nbx, 1, 1), dim3(nx, 1, 1), 0, stream>>>(
        batch_count, B, shiftB, ldb, strideB, B_ptr);
}

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
template <typename T, typename I, typename Istride, typename UA, typename S = decltype(std::real(T{}))>
static __global__ void cal_gnorm_sq_kernel(I const m,
                                           I const n,
                                           I const batch_count,

                                           UA A,
                                           Istride const shiftA,
                                           I const lda,
                                           Istride const strideA,

                                           S* const gnorm_array)
{
    {
        bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1) && (gnorm_array != nullptr);
        if(!has_work)
        {
            return;
        };
    }

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
    double* const gnorm_block = (double*)&(lmem[0]);

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        T const* const Ap = load_ptr_batch(A, bid, shiftA, strideA);

        S* const gnorm_bid = &(gnorm_array[bid]);

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
            };
            __syncthreads();

            double gnorm_j = 0;
            for(I jcol = jcol_start; jcol < n; jcol += jcol_inc)
            {
                double norm_j = 0;
                for(I i = 0; i < m; i++)
                {
                    auto const ij = idx2D(i, jcol, lda);
                    auto const aij = Ap[ij];
                    norm_j += std::norm(aij);
                }
                gnorm_j = rocblas_max_nan(gnorm_j, norm_j);
            }
            atomicMax(&(gnorm_block[0]), gnorm_j);
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
                    auto const aij = Ap[ij];

                    norm_j += std::norm(aij);
                }

                // ----------------------------------------
                // note: only tx == 0 has the correct value
                // ----------------------------------------
                norm_j = reduce_sum_shfl_wsize(nx, norm_j);
                if(tx == 0)
                {
                    gnorm_j = std::max(gnorm_j, norm_j);
                }
            } // end for j

            if(tx == 0)
            {
                atomicMax(&(gnorm_block[0]), gnorm_j);
            }
            __syncthreads();

            if((tx == 0) && (ty == 0) && (tz == 0))
            {
                atomicMax(gnorm_bid, static_cast<S>(gnorm_block[0]));
            }
        }
    } // end for bid
}

// ----------------------------------------
// compute the square gnorm of matrix,
// which is    max_j  norm( A(:,j), 2 )^2
// ----------------------------------------
template <typename T, typename I, typename Istride, typename UA, typename S = decltype(std::real(T{}))>
static void cal_gnorm_sq(hipStream_t stream,
                         I const m,
                         I const n,
                         I const batch_count,

                         UA A,
                         Istride const shiftA,
                         I const lda,
                         Istride const strideA,

                         S* const gnorm_array)
{
    {
        bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
        if(!has_work)
        {
            return;
        };
    }

    {
        size_t const size_gnorm_array = sizeof(S) * batch_count;
        hipError_t const istat = hipMemsetAsync(gnorm_array, 0, size_gnorm_array, stream);
        if(istat != hipSuccess)
        {
            return;
        };
    }

    auto const num_cu = get_num_cu();
    auto const warp_size = get_warp_size();
    I const lds_size = sizeof(S);

    I const max_threads = 1024;
    I const nx = warp_size; // !!! note nx <= warp_size is necessary
        // for correctness in using DPP instructions
    I const ny = max_threads / nx;
    I const nz = 1;

    I const max_blocks = num_cu;
    I const nbx = 1; // !!! note nbx == 1 is necessary for correctness
    I const nby = std::min(max_blocks, ceildiv(n, ny));
    I const nbz = std::min(max_blocks, batch_count);

    cal_gnorm_sq_kernel<T, I, Istride>
        <<<dim3(nbx, nby, nbz), dim3(nx, ny, nz), lds_size, stream>>>(m, n, batch_count,

                                                                      A, shiftA, lda, strideA,

                                                                      gnorm_array);
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

template <typename T, typename I>
static void cal_sigma_getMemorySize(I const m, I const n, I const batch_count, size_t* p_size_work)
{
    size_t size_work = 0;

    *p_size_work = size_work;
}

template <typename T, typename I, typename Istride, typename UA, typename S = decltype(std::real(T{}))>
static rocblas_status cal_sigma(hipStream_t stream,
                                I const m,
                                I const n,
                                I const batch_count,

                                UA A,
                                Istride const shiftA,
                                I const lda,
                                Istride const strideA,

                                S* const sigma_array,
                                void* work,
                                size_t const size_work)
{
    // -----------------
    // compute square of gnorm
    // reuse sigma_array
    // -----------------
    S* const gnorm_array = sigma_array;
    cal_gnorm_sq<T, I, Istride>(stream, m, n, batch_count,

                                A, shiftA, lda, strideA,

                                gnorm_array);

    // --------------------------------------------
    // sigma = 11 * n * u (m + (n+1) ) * gnorm(A)^2
    // --------------------------------------------
    S const eps = std::numeric_limits<S>::epsilon();
    S const dscale = 11.0 * n * eps * (m + (n + 1));

    {
        I const nx = 64;
        I const nbx = ceildiv(batch_count, nx);
        scale_kernel<S, I><<<dim3(nbx, 1, 1), dim3(nx, 1, 1), 0, stream>>>(

            batch_count, dscale, gnorm_array);
    }

    return (rocblas_status_success);
}

// ---------------------------------
// kernel to perform B <- B + sigma * identity
//
// launch as dim3(nbx,1,batch_count), dim3(nx,1,1)
//
// ---------------------------------
template <typename T, typename I, typename Istride, typename UB, typename S = decltype(std::real(T{}))>
static __global__ void add_shift_kernel(I const m,
                                        I const n,
                                        I const batch_count,
                                        S* const sigma_array,

                                        UB B,
                                        Istride const shiftB,
                                        I const ldb,
                                        Istride const strideB

)
{
    {
        // -------------------------------------------------
        // note: sigma_array == nullptr  treated as no shift
        // -------------------------------------------------
        bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1) && (sigma_array != nullptr);
        if(!has_work)
        {
            return;
        };
    }

    I const bid_start = blockIdx.z;
    I const bid_inc = gridDim.z;

    I const i_start = threadIdx.x + blockIdx.x * blockDim.x;
    I const i_inc = blockDim.x * gridDim.x;

    I const min_mn = std::min(m, n);

    S const zero = 0;
    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        auto const Bp = load_ptr_batch(B, bid, shiftB, strideB);
        S const sigma_bid = (sigma_array == nullptr) ? zero : sigma_array[bid];

        // ----------------------------
        // note: ignore negative shifts
        // ----------------------------
        S const sigma = std::max(sigma_bid, zero);

        if(sigma != zero)
        {
            for(I i = i_start; i < min_mn; i += i_inc)
            {
                // diagonal entry
                I const j = i;

                auto const ij = idx2D(i, j, ldb);
                Bp[ij] += sigma;
            }
        }
    } // end for bid
}

// --------------------------------------------
// routine to perform B <- B + sigma * identity
// --------------------------------------------
template <typename T, typename I, typename Istride, typename UB, typename S = decltype(std::real(T{}))>
static void add_shift(hipStream_t stream,
                      I const m,
                      I const n,
                      I const batch_count,
                      S* const sigma_array,

                      UB B,
                      Istride const shiftB,
                      I const ldb,
                      Istride const strideB

)
{
    {
        bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1) && (sigma_array != nullptr);
        if(!has_work)
        {
            return;
        };
    }

    auto const ceil = [](auto n, auto base) { return ((n - 1) / base + 1); };

    I const nx = 64;
    I const ny = 1;
    I const nz = 1;

    I const max_blocks = get_num_cu();
    I const min_mn = std::min(m, n);

    I const nbx = std::min(max_blocks, ceildiv(min_mn, nx));
    I const nby = 1;
    I const nbz = std::min(max_blocks, batch_count);

    add_shift_kernel<T, I, Istride, decltype(B), S>
        <<<dim3(nbx, nby, nbz), dim3(nx, ny, nz), 0, stream>>>(m, n, batch_count, sigma_array,

                                                               B, shiftB, ldb, strideB);
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
template <typename T, typename I, typename Istride, typename UA>
static void set_triangular(rocblas_handle handle,
                           char const uplo,
                           I const m,
                           I const n,
                           T const alpha,

                           UA A_arg,
                           Istride const shiftA,
                           I const lda,
                           Istride const strideA,
                           I const batch_count)
{
    {
        bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
        if(!has_work)
        {
            return;
        }
    }

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I const max_threads = 1024;
    I const nx = (m <= 32) ? 32 : 64;
    I const ny = max_threads / nx;

    I const max_blocks = get_num_cu();
    I const nbx = std::min(max_blocks, ceildiv(m, nx));
    I const nby = std::min(max_blocks, ceildiv(n, ny));
    I const nbz = std::min(max_blocks, batch_count);

    {
        bool const use_lower = (uplo == 'L') || (uplo == 'l');
        bool const use_upper = (uplo == 'U') || (uplo == 'u');
        bool const use_full = (uplo == 'G') || (uplo == 'g'); // 'GE' general full matrix

        {
            bool const isvalid_uplo = (use_lower || use_upper || use_full);
            assert(isvalid_uplo);
        }

        Istride const offset = (use_lower) ? idx2D(1, 0, lda)
            : (use_upper)                  ? idx2D(0, 1, lda)
            : (use_full)                   ? idx2D(0, 0, lda)
                                           : 0;

        T const lalpha = alpha;
        T const lbeta = alpha;

        I const mm = (use_lower || use_upper) ? m - 1 : m;
        I const nn = (use_lower || use_upper) ? n - 1 : n;

        laset(handle, uplo, mm, nn, lalpha, lbeta,

              A_arg, shiftA + offset, lda, strideA,

              batch_count);
    }
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
                                         const rocsolver_cholqr_algo algo,
                                         I* info,
                                         const I batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(algo != rocsolver_cholqr_cholqr1 && algo != rocsolver_cholqr_cholqr2
       && algo != rocsolver_cholqr_default && algo != rocsolver_cholqr_cholqr3_compute
       && algo != rocsolver_cholqr_cholqr3_user)
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
    if(batch_count > 0
       && (algo == rocsolver_cholqr_cholqr3_compute || algo == rocsolver_cholqr_cholqr3_user)
       && !sigma)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename I>
static rocblas_status rocsolver_cholqr1_getMemorySize(I const m,
                                                      I const n,
                                                      const I lda,
                                                      const I ldr,
                                                      I const batch_count,
                                                      size_t* size_scalars,
                                                      size_t* size_work1,
                                                      size_t* size_work2,
                                                      size_t* size_work3,
                                                      size_t* size_work4,
                                                      size_t* size_pivots,
                                                      size_t* size_iinfo,
                                                      bool* optim_mem)
{
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

    return rocblas_status_success;
}

template <bool BATCHED, bool STRIDED, typename T, typename I>
static rocblas_status rocsolver_cholqr2_getMemorySize(I const m,
                                                      I const n,
                                                      const I lda,
                                                      const I ldr,
                                                      I const batch_count,
                                                      size_t* size_scalars,
                                                      size_t* size_work1,
                                                      size_t* size_work2,
                                                      size_t* size_work3,
                                                      size_t* size_work4,
                                                      size_t* size_pivots,
                                                      size_t* size_iinfo,
                                                      size_t* size_R1,
                                                      bool* optim_mem)
{
    // storage for 1st call to cholqr1
    ROCBLAS_CHECK(rocsolver_cholqr1_getMemorySize<BATCHED, STRIDED, T>(
        m, n, lda, ldr, batch_count, size_scalars, size_work1, size_work2, size_work3, size_work4,
        size_pivots, size_iinfo, optim_mem));

    // storage for iinfo, intended for 2nd call to cholqr1(A)
    *size_iinfo += sizeof(I) * batch_count;

    // storage for R1, in computing [Q,R1] = cholqr1(A)
    *size_R1 = sizeof(T) * n * n * batch_count;

    return rocblas_status_success;
}

template <bool BATCHED, bool STRIDED, typename T, typename I>
static rocblas_status rocsolver_cholqr3_getMemorySize(I const m,
                                                      I const n,
                                                      const I lda,
                                                      const I ldr,
                                                      I const batch_count,
                                                      size_t* size_scalars,
                                                      size_t* size_work1,
                                                      size_t* size_work2,
                                                      size_t* size_work3,
                                                      size_t* size_work4,
                                                      size_t* size_pivots,
                                                      size_t* size_iinfo,
                                                      size_t* size_R1,
                                                      bool* optim_mem,
                                                      size_t* p_size_work)
{
    size_t size_work = 0;
    *p_size_work = 0;
    {
        bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
        if(!has_work)
        {
            return (rocblas_status_success);
        }
    }

    ROCBLAS_CHECK(rocsolver_cholqr1_getMemorySize<BATCHED, STRIDED, T, I>(
        m, n, lda, ldr, batch_count, size_scalars, size_work1, size_work2, size_work3, size_work4,
        size_pivots, size_iinfo, optim_mem));

    ROCBLAS_CHECK(rocsolver_cholqr2_getMemorySize<BATCHED, STRIDED, T, I>(
        m, n, lda, ldr, batch_count, size_scalars, size_work1, size_work2, size_work3, size_work4,
        size_pivots, size_iinfo, size_R1, optim_mem));

    *size_iinfo += sizeof(I) * batch_count;
    *size_R1 += sizeof(T) * n * n * batch_count;

    *p_size_work = size_work;
    return (rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T, typename I>
static rocblas_status rocsolver_cholqr_getMemorySize(I const m,
                                                     I const n,
                                                     const I lda,
                                                     const I ldr,
                                                     I const batch_count,
                                                     rocsolver_cholqr_algo const algo,
                                                     size_t* size_scalars,
                                                     size_t* size_work1,
                                                     size_t* size_work2,
                                                     size_t* size_work3,
                                                     size_t* size_work4,
                                                     size_t* size_pivots,
                                                     size_t* size_iinfo,
                                                     size_t* size_R1,
                                                     size_t* size_workArr,
                                                     bool* optim_mem,
                                                     size_t* p_size_work)
{
    size_t size_work = 0;
    *p_size_work = 0;

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

    *size_R1 = 0;

    rocblas_status istat = rocblas_status_success;
    if(algo == rocsolver_cholqr_cholqr1)
    {
        istat = rocsolver_cholqr1_getMemorySize<BATCHED, STRIDED, T, I>(
            m, n, lda, ldr, batch_count, size_scalars, size_work1, size_work2, size_work3,
            size_work4, size_pivots, size_iinfo, optim_mem);
    }
    else if((algo == rocsolver_cholqr_cholqr2) || (algo == rocsolver_cholqr_default))
    {
        istat = rocsolver_cholqr2_getMemorySize<BATCHED, STRIDED, T, I>(
            m, n, lda, ldr, batch_count, size_scalars, size_work1, size_work2, size_work3,
            size_work4, size_pivots, size_iinfo, size_R1, optim_mem);
    }
    else if((algo == rocsolver_cholqr_cholqr3_compute) || (algo == rocsolver_cholqr_cholqr3_user))
    {
        istat = rocsolver_cholqr3_getMemorySize<BATCHED, STRIDED, T, I>(
            m, n, lda, ldr, batch_count, size_scalars, size_work1, size_work2, size_work3,
            size_work4, size_pivots, size_iinfo, size_R1, optim_mem, &size_work);
    }

    if(BATCHED)
        *size_workArr = sizeof(T*) * batch_count;
    else
        *size_workArr = 0;

    *p_size_work = size_work;
    return (rocblas_status_success);
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
        add_shift<T>(stream, m, n, batch_count, sigma_array, R, shiftR, ldr, strideR);

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
    set_triangular(handle, 'L', n, n, zero, R, shiftR, ldr, strideR, batch_count);

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
          typename Istride,
          typename UA,
          typename UR,
          typename INFO = I,
          typename S = decltype(std::real(T{}))>
static rocblas_status rocsolver_cholqr3_template(rocblas_handle handle,
                                                 I const m,
                                                 I const n,

                                                 UA A,
                                                 Istride const shiftA,
                                                 I const lda,
                                                 Istride strideA,

                                                 UR R,
                                                 Istride const shiftR,
                                                 I const ldr,
                                                 Istride strideR,

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
                                                 bool optim_mem,

                                                 void* work,
                                                 size_t const size_work)
{
    bool constexpr is_pointer_batched_A = IS_POINTER_BATCHED(A, T);
    bool constexpr is_pointer_batched_R = IS_POINTER_BATCHED(R, T);

    {
        bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
        if(!has_work)
        {
            return (rocblas_status_success);
        }
    }

    const T one = T(1);

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    rocblas_status istat = rocblas_status_success;

    hipStream_t stream;
    try
    {
        {
            istat = rocblas_get_stream(handle, &stream);
            bool const isok_get_stream
                = (istat == rocblas_status_success) || (istat == rocblas_status_continue);
            if(!isok_get_stream)
            {
                throw(istat);
            }
        }

        {
            // ----------
            // reset info
            // ----------
            auto const istat_hip = hipMemsetAsync(info, 0, sizeof(INFO) * batch_count, stream);
            if(istat_hip != hipSuccess)
            {
                istat = rocblas_status_internal_error;
                throw(istat);
            }
        }

        std::byte* const pwork = (std::byte*)work;
        std::byte* pfree = pwork;

        // -----------------------------
        // Note: matrix Q over-writes matrix A
        // -----------------------------
        auto const Q = A;
        auto const shiftQ = shiftA;
        auto const ldq = lda;
        auto const strideQ = strideA;

        // ---------------------------------------
        // (1)  R1 * R1' = A'*A + sigma * identity
        // ---------------------------------------

        {
            auto const pfree_saved = pfree;

            size_t const size_remain = (pwork + size_work) - pfree;

            if(compute_sigma)
            {
                istat = cal_sigma<T, I>(stream, m, n, batch_count,

                                        A, shiftA, lda, strideA,

                                        sigma_array, (void*)pfree, size_remain);
                if(istat != rocblas_status_success)
                {
                    throw(istat);
                }
            }

            // -------------------------------------------------------------------
            // Note: paper suggests
            // T const sigma = 11 * (m * n * ueps + (n + 1) * (n * ueps)) * gnorm
            // -------------------------------------------------------------------

            // -----------------------------
            // perform CholeskQR1 with shift
            // -----------------------------
            ROCBLAS_CHECK(rocsolver_cholqr1_template<BATCHED, STRIDED, T>(
                handle, m, n, A, shiftA, lda, strideA, R1, 0, n, n * n, batch_count, info, scalars,
                work1, work2, work3, work4, pivots, iinfo, workArr, optim_mem));

            pfree = pfree_saved;
        }

        // ----------------
        // (2)   CholQR2(Q)
        // ----------------

        {
            auto const pfree_saved = pfree;
            size_t const size_remain = (pwork + size_work) - pfree;

            ROCBLAS_CHECK(rocsolver_cholqr2_template<BATCHED, STRIDED, T>(
                handle, m, n, Q, shiftQ, ldq, strideQ, R, shiftR, ldr, strideR, batch_count, iinfo,
                scalars, work1, work2, work3, work4, pivots, iinfo + batch_count,
                R1 + n * n * batch_count, workArr, optim_mem));

            pfree = pfree_saved;
        }

        // -------------------------------
        // compute R <- R * R1, using TRMM
        // -------------------------------

        {
            // ----------------------------------------------------
            // (i)  set strictly lower triangular part of R be zero
            // ----------------------------------------------------

            char const uplo = 'L';
            T const alpha = 0;
            set_triangular(handle, uplo, n, n, alpha,

                           R, shiftR, ldr, strideR,

                           batch_count);
        }

        // R <- R * R1
        ROCBLAS_CHECK(rocblasCall_trmm<T>(handle, rocblas_side_right, rocblas_fill_upper,
                                          rocblas_operation_none, rocblas_diagonal_non_unit, n, n,
                                          &one, 0, R1, 0, n, n * n, R, shiftR, ldr, strideR,
                                          batch_count, workArr));

        // Finally

        rocblas_set_pointer_mode(handle, old_mode);
        return (istat);
    }
    catch(rocblas_status const istat)
    {
        rocblas_set_pointer_mode(handle, old_mode);
        return (istat);
    }
}

template <bool BATCHED,
          bool STRIDED,
          typename T,
          typename I,
          typename Istride,
          typename UA,
          typename UR,
          typename INFO = I,
          typename S = decltype(std::real(T{}))>
static rocblas_status rocsolver_cholqr_template(rocblas_handle handle,
                                                I const m,
                                                I const n,

                                                UA A,
                                                Istride const shiftA,
                                                I const lda,
                                                Istride strideA,

                                                UR R,
                                                Istride const shiftR,
                                                I const ldr,
                                                Istride const strideR,

                                                S* const sigma_array,
                                                rocsolver_cholqr_algo const algo,

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
                                                bool optim_mem,

                                                void* work,
                                                size_t const size_work)

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

    rocblas_status istat = rocblas_status_success;

    if(algo == rocsolver_cholqr_cholqr1)
    {
        ROCBLAS_CHECK(rocsolver_cholqr1_template<BATCHED, STRIDED, T>(
            handle, m, n, A, shiftA, lda, strideA, R, shiftR, ldr, strideR, batch_count, info,
            scalars, work1, work2, work3, work4, pivots, iinfo, workArr, optim_mem));
    }
    else if((algo == rocsolver_cholqr_cholqr2) || (algo == rocsolver_cholqr_default))
    {
        ROCBLAS_CHECK(rocsolver_cholqr2_template<BATCHED, STRIDED, T>(
            handle, m, n, A, shiftA, lda, strideA, R, shiftR, ldr, strideR, batch_count, info,
            scalars, work1, work2, work3, work4, pivots, iinfo, R1, workArr, optim_mem));
    }
    else if((algo == rocsolver_cholqr_cholqr3_compute) || (algo == rocsolver_cholqr_cholqr3_user))
    {
        bool const compute_sigma = (algo == rocsolver_cholqr_cholqr3_compute);

        istat = rocsolver_cholqr3_template<BATCHED, STRIDED, T>(handle, m, n,

                                                                A, shiftA, lda, strideA,

                                                                R, shiftR, ldr, strideR,

                                                                compute_sigma, sigma_array,

                                                                info, batch_count, scalars, work1,
                                                                work2, work3, work4, pivots, iinfo,
                                                                R1, workArr, optim_mem,

                                                                work, size_work);
    }

    return (istat);
}

#undef IS_POINTER_BATCHED
#undef MEM_CHECK
#undef MEM_CHECK_THROW
ROCSOLVER_END_NAMESPACE

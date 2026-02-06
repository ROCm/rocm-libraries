
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

// trsm memory sizes
template <typename T, typename I>
static rocblas_status rocblasCall_trsm_mem_max(rocblas_side const side,
                                               rocblas_operation const transA,
                                               I const m,
                                               I const n,
                                               I const lda,
                                               I const ldb,
                                               I const batch_count,

                                               size_t* const x_temp,
                                               size_t* const x_temp_arr,
                                               size_t* const invA,
                                               size_t* const invA_arr)
{
    size_t size_x_temp = 0;
    size_t size_x_temp_arr = 0;
    size_t size_invA = 0;
    size_t size_invA_arr = 0;

    *x_temp = size_x_temp;
    *x_temp_arr = size_x_temp_arr;
    *invA = size_invA;
    *invA_arr = size_invA_arr;

    {
        bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
        if(!has_work)
        {
            return (rocblas_status_success);
        }
    }

    rocblas_status istat = rocblas_status_success;

    {
        size_t lsize_x_temp = 0;
        size_t lsize_x_temp_arr = 0;
        size_t lsize_invA = 0;
        size_t lsize_invA_arr = 0;

        bool constexpr LBATCHED = true;
        istat = rocblasCall_trsm_mem<LBATCHED, T, I>(side, transA, m, n, lda, ldb, batch_count,
                                                     &lsize_x_temp, &lsize_x_temp_arr, &lsize_invA,
                                                     &lsize_invA_arr);

        {
            bool const isok = (istat == rocblas_status_continue) || (istat == rocblas_status_success);
            if(!isok)
            {
                return (istat);
            }
        }

        size_x_temp = std::max(size_x_temp, lsize_x_temp);
        size_x_temp_arr = std::max(size_x_temp_arr, lsize_x_temp_arr);
        size_invA = std::max(size_invA, lsize_invA);
        size_invA_arr = std::max(size_invA, lsize_invA_arr);
    }

    {
        size_t lsize_x_temp = 0;
        size_t lsize_x_temp_arr = 0;
        size_t lsize_invA = 0;
        size_t lsize_invA_arr = 0;

        bool constexpr LBATCHED = false;
        istat = rocblasCall_trsm_mem<LBATCHED, T, I>(side, transA, m, n, lda, ldb, batch_count,
                                                     &lsize_x_temp, &lsize_x_temp_arr, &lsize_invA,
                                                     &lsize_invA_arr);

        {
            bool const isok = (istat == rocblas_status_continue) || (istat == rocblas_status_success);
            if(!isok)
            {
                return (istat);
            }
        }

        size_x_temp = std::max(size_x_temp, lsize_x_temp);
        size_x_temp_arr = std::max(size_x_temp_arr, lsize_x_temp_arr);
        size_invA = std::max(size_invA, lsize_invA);
        size_invA_arr = std::max(size_invA, lsize_invA_arr);
    }

    adjust_for_alignment(size_x_temp);
    adjust_for_alignment(size_x_temp_arr);
    adjust_for_alignment(size_invA);
    adjust_for_alignment(size_invA_arr);

    *x_temp = size_x_temp;
    *x_temp_arr = size_x_temp_arr;
    *invA = size_invA;
    *invA_arr = size_invA_arr;

    return (istat);
}

template <typename T, typename I>
static rocblas_status rocblasCall_trsm_mem_alt(rocblas_side const side,
                                               rocblas_operation const transA,
                                               I const m,
                                               I const n,
                                               I const lda,
                                               I const ldb,
                                               I const batch_count,

                                               size_t* const p_size_work)
{
    *p_size_work = 0;

    {
        bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
        if(!has_work)
        {
            return (rocblas_status_success);
        }
    }

    size_t x_temp = 0;
    size_t x_temp_arr = 0;
    size_t invA = 0;
    size_t invA_arr = 0;

    rocblas_status const istat
        = rocblasCall_trsm_mem_max<T, I>(side, transA, m, n, lda, ldb, batch_count,

                                         &x_temp, &x_temp_arr, &invA, &invA_arr);
    {
        bool const isok = (istat == rocblas_status_continue) || (istat == rocblas_status_success);
        if(!isok)
        {
            return (istat);
        }
    }

    adjust_for_alignment(x_temp);
    adjust_for_alignment(x_temp_arr);
    adjust_for_alignment(invA);
    adjust_for_alignment(invA_arr);

    size_t size_work = x_temp + x_temp_arr + invA + invA_arr;

    {
        size_t size_workArr = sizeof(T*) * batch_count;
        adjust_for_alignment(size_workArr);

        size_work += size_workArr;
    }

    *p_size_work = size_work;

    return (istat);
}

template <typename T, typename I, typename Istride, typename UA, typename UB>
static rocblas_status rocblasCall_trsm_alt(rocblas_handle handle,
                                           rocblas_side const side,
                                           rocblas_fill const uplo,
                                           rocblas_operation const transA,
                                           rocblas_diagonal const diag,
                                           I const m,
                                           I const n,
                                           T const* const alpha,

                                           UA A,
                                           Istride const offset_A,
                                           I const lda,
                                           Istride const stride_A,

                                           UB B,
                                           Istride const offset_B,
                                           I const ldb,
                                           Istride const stride_B,

                                           I const batch_count,

                                           void* const work,
                                           size_t const size_work)
{
    {
        bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
        if(!has_work)
        {
            return (rocblas_status_success);
        };
    }

    auto const shift_A = offset_A;
    auto const shift_B = offset_B;

    std::byte* const pwork = (std::byte*)work;
    std::byte* pfree = pwork;

    rocblas_status istat = rocblas_status_success;

    hipStream_t stream;
    {
        istat = rocblas_get_stream(handle, &stream);
        bool const isok_get_stream
            = (istat == rocblas_status_success) || (istat == rocblas_status_continue);
        if(!isok_get_stream)
        {
            return (istat);
        }
    }

    // -----------------------
    // allocate scratch arrays
    // -----------------------

    size_t size_x_temp = 0;
    size_t size_x_temp_arr = 0;
    size_t size_invA = 0;
    size_t size_invA_arr = 0;

    bool const optimal_mem = true;

    {
        istat = rocblasCall_trsm_mem_max<T, I>(side, transA, m, n, lda, ldb, batch_count,

                                               &size_x_temp, &size_x_temp_arr, &size_invA,
                                               &size_invA_arr);

        bool const isok_trsm_mem
            = (istat == rocblas_status_success) || (istat == rocblas_status_continue);
        if(!isok_trsm_mem)
        {
            return (istat);
        }
    }

    adjust_for_alignment(size_x_temp);
    adjust_for_alignment(size_x_temp_arr);
    adjust_for_alignment(size_invA);
    adjust_for_alignment(size_invA_arr);

    void* const x_temp = (void*)pfree;
    pfree += size_x_temp;

    void* const x_temp_arr = (void*)pfree;
    pfree += size_x_temp_arr;

    void* const invA = (void*)pfree;
    pfree += size_invA;

    void* const invA_arr = (void*)pfree;
    pfree += size_invA_arr;

    size_t size_workArr = sizeof(T*) * batch_count;
    adjust_for_alignment(size_workArr);

    T** const workArr = (T**)pfree;
    pfree += size_workArr;

    MEM_CHECK(pfree);

    istat = rocblasCall_trsm<T, I>(handle, side, uplo, transA, diag, m, n, alpha,

                                   A, offset_A, lda, stride_A,

                                   B, offset_B, ldb, stride_B,

                                   batch_count, optimal_mem,

                                   x_temp, x_temp_arr, invA, invA_arr, workArr);

    return (istat);
}

template <typename T, typename I>
static rocblas_status
    rocblasCall_syrk_herk_mem(I const m, I const n, I const batch_count, size_t* p_size_work)
{
    size_t size_A_ptr = sizeof(T*) * batch_count;
    size_t size_B_ptr = sizeof(T*) * batch_count;

    adjust_for_alignment(size_A_ptr);
    adjust_for_alignment(size_B_ptr);

    size_t const size_work = size_A_ptr + size_B_ptr;

    *p_size_work = size_work;
    return (rocblas_status_success);
}

template <typename T, typename I, typename Istride, typename UA, typename UB, typename S = decltype(std::real(T{}))>
static rocblas_status rocblasCall_syrk_herk_alt(rocblas_handle handle,
                                                rocblas_fill const uplo,
                                                rocblas_operation const trans,

                                                I const m,
                                                I const n,

                                                S* const alpha,

                                                UA A,
                                                Istride const shiftA,
                                                I const lda,
                                                Istride const strideA,

                                                S* const beta,

                                                UB B,
                                                Istride const shiftB,
                                                I const ldb,
                                                Istride const strideB,

                                                I const batch_count,
                                                void* const work,
                                                size_t const size_work)
{
    bool constexpr is_pointer_batched_A = IS_POINTER_BATCHED(A, T);
    bool constexpr is_pointer_batched_B = IS_POINTER_BATCHED(B, T);

    {
        bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
        if(!has_work)
        {
            return (rocblas_status_success);
        }
    }

    rocblas_status istat = rocblas_status_success;
    hipStream_t stream;
    {
        istat = rocblas_get_stream(handle, &stream);
        bool const isok_get_stream
            = (istat == rocblas_status_success) || (istat == rocblas_status_continue);
        if(!isok_get_stream)
        {
            return (istat);
        }
    }

    std::byte* const pwork = (std::byte*)work;
    std::byte* pfree = pwork;

    bool constexpr is_both_same_type = (is_pointer_batched_A == is_pointer_batched_B);
    if constexpr(is_both_same_type)
    {
        size_t size_work_arr = sizeof(T*) * batch_count;
        adjust_for_alignment(size_work_arr);

        T** const work_arr = (T**)pfree;
        pfree += size_work_arr;

        MEM_CHECK(pfree);

        bool constexpr LBATCHED = is_pointer_batched_A || is_pointer_batched_B;
        istat = rocblasCall_syrk_herk<LBATCHED, T>(handle, uplo, trans, m, n, alpha,

                                                   A, shiftA, lda, strideA,

                                                   beta,

                                                   B, shiftB, ldb, strideB,

                                                   batch_count, work_arr);
    }
    else
    {
        T** A_ptr = nullptr;
        T** B_ptr = nullptr;

        if constexpr(is_pointer_batched_A)
        {
            A_ptr = (T**)A;
        }
        else
        {
            size_t size_A_ptr = sizeof(T*) * batch_count;
            adjust_for_alignment(size_A_ptr);

            A_ptr = (T**)pfree;
            pfree += size_A_ptr;

            MEM_CHECK(pfree);

            copy_array_to_ptr<T, I, Istride>(stream, batch_count,

                                             A, shiftA, lda, strideA,

                                             A_ptr);
        }

        if constexpr(is_pointer_batched_B)
        {
            B_ptr = B;
        }
        else
        {
            size_t size_B_ptr = sizeof(T*) * batch_count;
            adjust_for_alignment(size_B_ptr);

            B_ptr = (T**)pfree;
            pfree += size_B_ptr;

            MEM_CHECK(pfree);

            copy_array_to_ptr<T, I, Istride>(stream, batch_count,

                                             B, shiftB, ldb, strideB,

                                             B_ptr);
        }

        size_t size_work_arr = sizeof(T*) * batch_count;
        adjust_for_alignment(size_work_arr);

        T** const work_arr = (T**)pfree;
        pfree += size_work_arr;

        MEM_CHECK(pfree);

        bool constexpr LBATCHED = true;
        istat = rocblasCall_syrk_herk<LBATCHED, T>(handle, uplo, trans, m, n, alpha,

                                                   A_ptr, shiftA, lda, strideA,

                                                   beta,

                                                   B_ptr, shiftB, ldb, strideB,

                                                   batch_count, work_arr);
    }

    return (istat);
}

template <typename T, typename I>
static void rocsolver_potrf_getMemorySize_max(I const n,
                                              rocblas_fill const uplo,
                                              I const batch_count,

                                              size_t* const p_size_scalars,
                                              size_t* const p_size_work1,
                                              size_t* const p_size_work2,
                                              size_t* const p_size_work3,
                                              size_t* const p_size_work4,
                                              size_t* const p_size_pivots,
                                              size_t* const p_size_iinfo)
{
    size_t size_scalars = 3 * sizeof(T);
    size_t size_work1 = 0;
    size_t size_work2 = 0;
    size_t size_work3 = 0;
    size_t size_work4 = 0;
    size_t size_pivots = 0;
    size_t size_iinfo = 0;

    *p_size_scalars = size_scalars;

    *p_size_work1 = 0;
    *p_size_work2 = 0;
    *p_size_work3 = 0;
    *p_size_work4 = 0;

    *p_size_pivots = 0;
    *p_size_iinfo = 0;

    {
        bool const has_work = (n >= 1) && (batch_count >= 1);
        if(!has_work)
        {
            return;
        }
    }

    bool optim_mem = true;
    {
        bool constexpr LBATCHED = true;
        bool constexpr LSTRIDED = false;

        size_t lsize_scalars = 0;
        size_t lsize_work1 = 0;
        size_t lsize_work2 = 0;
        size_t lsize_work3 = 0;
        size_t lsize_work4 = 0;
        size_t lsize_pivots = 0;
        size_t lsize_iinfo = 0;

        rocsolver_potrf_getMemorySize<LBATCHED, LSTRIDED, T, I>(
            n, uplo, batch_count,

            &lsize_scalars, &lsize_work1, &lsize_work2, &lsize_work3, &lsize_work4, &lsize_pivots,
            &lsize_iinfo, &optim_mem);

        size_work1 = std::max(size_work1, lsize_work1);
        size_work2 = std::max(size_work2, lsize_work2);
        size_work3 = std::max(size_work3, lsize_work3);
        size_work4 = std::max(size_work4, lsize_work4);

        size_scalars = std::max(size_scalars, lsize_scalars);
        size_pivots = std::max(size_pivots, lsize_pivots);
        size_iinfo = std::max(size_iinfo, lsize_iinfo);
    }

    {
        bool constexpr LBATCHED = false;
        bool constexpr LSTRIDED = true;

        size_t lsize_scalars = 0;
        size_t lsize_work1 = 0;
        size_t lsize_work2 = 0;
        size_t lsize_work3 = 0;
        size_t lsize_work4 = 0;
        size_t lsize_pivots = 0;
        size_t lsize_iinfo = 0;

        rocsolver_potrf_getMemorySize<LBATCHED, LSTRIDED, T, I>(
            n, uplo, batch_count,

            &lsize_scalars, &lsize_work1, &lsize_work2, &lsize_work3, &lsize_work4, &lsize_pivots,
            &lsize_iinfo, &optim_mem);

        size_work1 = std::max(size_work1, lsize_work1);
        size_work2 = std::max(size_work2, lsize_work2);
        size_work3 = std::max(size_work3, lsize_work3);
        size_work4 = std::max(size_work4, lsize_work4);

        size_scalars = std::max(size_scalars, lsize_scalars);
        size_pivots = std::max(size_pivots, lsize_pivots);
        size_iinfo = std::max(size_iinfo, lsize_iinfo);
    }

    if(batch_count == 1)
    {
        bool constexpr LBATCHED = false;
        bool constexpr LSTRIDED = false;

        size_t lsize_scalars = 0;
        size_t lsize_work1 = 0;
        size_t lsize_work2 = 0;
        size_t lsize_work3 = 0;
        size_t lsize_work4 = 0;
        size_t lsize_pivots = 0;
        size_t lsize_iinfo = 0;

        rocsolver_potrf_getMemorySize<LBATCHED, LSTRIDED, T, I>(
            n, uplo, batch_count,

            &lsize_scalars, &lsize_work1, &lsize_work2, &lsize_work3, &lsize_work4, &lsize_pivots,
            &lsize_iinfo, &optim_mem);

        size_work1 = std::max(size_work1, lsize_work1);
        size_work2 = std::max(size_work2, lsize_work2);
        size_work3 = std::max(size_work3, lsize_work3);
        size_work4 = std::max(size_work4, lsize_work4);

        size_scalars = std::max(size_scalars, lsize_scalars);
        size_pivots = std::max(size_pivots, lsize_pivots);
        size_iinfo = std::max(size_iinfo, lsize_iinfo);
    }

    {
        adjust_for_alignment(size_work1);
        adjust_for_alignment(size_work2);
        adjust_for_alignment(size_work3);
        adjust_for_alignment(size_work4);

        adjust_for_alignment(size_scalars);
        adjust_for_alignment(size_pivots);
        adjust_for_alignment(size_iinfo);
    }

    *p_size_scalars = size_scalars;

    *p_size_work1 = size_work1;
    *p_size_work2 = size_work2;
    *p_size_work3 = size_work3;
    *p_size_work4 = size_work4;

    *p_size_pivots = size_pivots;
    *p_size_iinfo = size_iinfo;
}

template <typename T, typename I>
static void rocsolver_potrf_getMemorySize_alt(I const n,
                                              rocblas_fill const uplo,
                                              I const batch_count,

                                              size_t* const p_size_potrf)
{
    *p_size_potrf = 0;
    {
        bool const has_work = (n >= 1) && (batch_count >= 1);
        if(!has_work)
        {
            return;
        }
    }

    size_t size_potrf = 0;

    size_t size_scalars = 0;

    size_t size_work1 = 0;
    size_t size_work2 = 0;
    size_t size_work3 = 0;
    size_t size_work4 = 0;

    size_t size_pivots = 0;
    size_t size_iinfo = 0;

    rocsolver_potrf_getMemorySize_max<T, I>(n, uplo, batch_count,

                                            &size_scalars, &size_work1, &size_work2, &size_work3,
                                            &size_work4, &size_pivots, &size_iinfo);

    {
        adjust_for_alignment(size_scalars);

        adjust_for_alignment(size_work1);
        adjust_for_alignment(size_work2);
        adjust_for_alignment(size_work3);
        adjust_for_alignment(size_work4);

        adjust_for_alignment(size_pivots);
        adjust_for_alignment(size_iinfo);
    }

    size_potrf += size_scalars;
    size_potrf += size_work1 + size_work2 + size_work3 + size_work4;
    size_potrf += size_pivots + size_iinfo;

    *p_size_potrf = size_potrf;
}

template <typename T, typename I, typename Istride, typename UB, typename INFO>
rocblas_status rocsolver_potrf_template_alt(rocblas_handle handle,
                                            rocblas_fill const uplo,
                                            I const n,

                                            UB B,
                                            Istride const shiftB,
                                            I const ldb,
                                            Istride const strideB,

                                            INFO* const info,
                                            I const batch_count,

                                            void* const work,
                                            size_t const size_work)
{
    using S = decltype(std::real(T{}));

    {
        bool const has_work = (n >= 1) && (batch_count >= 1);
        if(!has_work)
        {
            return (rocblas_status_success);
        }
    }

    rocblas_status istat = rocblas_status_success;

    std::byte* const pwork = (std::byte*)work;
    std::byte* pfree = pwork;

    size_t size_scalars = 0;
    size_t size_work1 = 0;
    size_t size_work2 = 0;
    size_t size_work3 = 0;
    size_t size_work4 = 0;
    size_t size_pivots = 0;
    size_t size_iinfo = 0;

    bool optim_mem = true;

    rocsolver_potrf_getMemorySize_max<T, I>(n, uplo, batch_count,

                                            &size_scalars, &size_work1, &size_work2, &size_work3,
                                            &size_work4, &size_pivots, &size_iinfo);

    {
        adjust_for_alignment(size_scalars);

        adjust_for_alignment(size_work1);
        adjust_for_alignment(size_work2);
        adjust_for_alignment(size_work3);
        adjust_for_alignment(size_work4);

        adjust_for_alignment(size_pivots);
        adjust_for_alignment(size_iinfo);
    }

    // -----------------------------------------------------
    // allocate temporary storage for Cholesky factorization
    // -----------------------------------------------------

    T* const scalars = (T*)pfree;
    pfree += size_scalars;
    T* const pivots = (T*)pfree;
    pfree += size_pivots;
    INFO* const iinfo = (INFO*)pfree;
    pfree += size_iinfo;

    void* const work1 = (void*)pfree;
    pfree += size_work1;
    void* const work2 = (void*)pfree;
    pfree += size_work2;
    void* const work3 = (void*)pfree;
    pfree += size_work3;
    void* const work4 = (void*)pfree;
    pfree += size_work4;

    MEM_CHECK(pfree);

    {
        bool constexpr LBATCHED = IS_POINTER_BATCHED(B, T);
        bool constexpr LSTRIDED = !LBATCHED;
        istat = rocsolver_potrf_template<LBATCHED, LSTRIDED, T, I, INFO, S>(
            handle, uplo, n,

            B, shiftB, ldb, strideB,

            info, batch_count,

            scalars, work1, work2, work3, work4, pivots, iinfo, optim_mem);
    }

    return (istat);
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
static rocblas_status
    rocsolver_cholqr1_getMemorySize(I const m, I const n, I const batch_count, size_t* p_size_work)
{
    size_t size_work = 0;
    *p_size_work = 0;

    rocblas_status istat = rocblas_status_success;

    {
        bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
        if(!has_work)
        {
            return (rocblas_status_success);
        }
    }

    // ----------------------------------
    // storage for computing  B = A' * A
    // ----------------------------------

    size_t size_gemm = 0;
    size_t size_syrk_herk = 0;
    if constexpr(use_syrk)
    {
        istat = rocblasCall_syrk_herk_mem<T>(n, m, batch_count, &size_syrk_herk);
        adjust_for_alignment(size_syrk_herk);

        bool const isok_syrk_mem
            = (istat == rocblas_status_continue) || (istat == rocblas_status_success);
        if(!isok_syrk_mem)
        {
            return (istat);
        }

        size_t size_work_arr = sizeof(T*) * batch_count;
        adjust_for_alignment(size_work_arr);
        size_syrk_herk += size_work_arr;
    }
    else
    {
        size_gemm = sizeof(T*) * batch_count;
        adjust_for_alignment(size_gemm);
    }

    // ----------------------------------------------
    // storage for Cholesky factorization R = chol(B)
    // ----------------------------------------------
    size_t size_potrf = 0;
    {
        rocblas_fill const uplo = rocblas_fill_upper;
        rocsolver_potrf_getMemorySize_alt<T, I>(n, uplo, batch_count, &size_potrf);
    }

    // -------------------------------
    // storage for computing Q = A / R
    // -------------------------------
    size_t size_trsm = 0;
    {
        auto const mm = m;
        auto const nn = n;
        auto const ld1 = m;
        auto const ld2 = n;

        rocblas_side const side = rocblas_side_right;
        rocblas_fill const uplo = rocblas_fill_upper;
        rocblas_diagonal const diag = rocblas_diagonal_non_unit;
        rocblas_operation const trans1 = rocblas_operation_none;

        istat = rocblasCall_trsm_mem_alt<T, I>(side, trans1, mm, nn, ld1, ld2, batch_count,

                                               &size_trsm);

        adjust_for_alignment(size_trsm);

        bool const isok_trsm_mem
            = (istat == rocblas_status_success) || (istat == rocblas_status_continue);

        if(!isok_trsm_mem)
        {
            return (istat);
        }
    }

    size_t const size_blas = std::max({size_syrk_herk, size_gemm, size_trsm});

    size_work = std::max(size_potrf, size_blas);

    adjust_for_alignment(size_work);

    *p_size_work = size_work;

    return (istat);
}

template <bool BATCHED, bool STRIDED, typename T, typename I>
static rocblas_status
    rocsolver_cholqr2_getMemorySize(I const m, I const n, I const batch_count, size_t* p_size_work)
{
    using INFO = I;

    size_t size_work = 0;
    *p_size_work = 0;

    {
        bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
        if(!has_work)
        {
            return (rocblas_status_success);
        }
    }

    rocblas_status istat = rocblas_status_success;

    // --------------------------------
    // storage for R1, in computing [Q,R1] = cholqr1(A)
    // --------------------------------
    size_t size_R1 = 0;
    {
        I const ldr = n;
        size_t const strideR = static_cast<size_t>(ldr) * n;
        size_R1 = sizeof(T) * strideR * batch_count;
        adjust_for_alignment(size_R1);
    }

    // storage for 1st call to cholqr1
    size_t size_cholqr1 = 0;
    {
        istat
            = rocsolver_cholqr1_getMemorySize<BATCHED, STRIDED, T>(m, n, batch_count, &size_cholqr1);
        if(istat != rocblas_status_success)
        {
            return (istat);
        }

        adjust_for_alignment(size_cholqr1);
    }

    // --------------------------------------------
    // storage for iinfo, intended for 2nd call to cholqr1(A)
    // --------------------------------------------
    size_t size_iinfo = 0;
    {
        size_iinfo = sizeof(INFO) * batch_count;
        adjust_for_alignment(size_iinfo);
    }

    size_t size_trmm = 0;
    {
        size_trmm = sizeof(T*) * batch_count;
        adjust_for_alignment(size_trmm);
    }

    size_work += size_R1;
    size_work += std::max(size_trmm, (size_iinfo + size_cholqr1));

    *p_size_work = size_work;

    return (istat);
}

template <bool BATCHED, bool STRIDED, typename T, typename I>
static rocblas_status
    rocsolver_cholqr3_getMemorySize(I const m, I const n, I const batch_count, size_t* p_size_work)
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

    size_t size_R1 = 0;
    {
        I const ldr = n;
        size_R1 = sizeof(T) * ldr * n * batch_count;
        adjust_for_alignment(size_R1);
    }
    size_work += size_R1;

    size_t size_iinfo = 0;
    {
        size_iinfo = sizeof(I) * batch_count;
        adjust_for_alignment(size_iinfo);
    }

    size_work += size_iinfo;

    size_t size_cholqr1 = 0;

    {
        auto const istat = rocsolver_cholqr1_getMemorySize<BATCHED, STRIDED, T, I>(m, n, batch_count,
                                                                                   &size_cholqr1);
        if(istat != rocblas_status_success)
        {
            return (istat);
        }

        adjust_for_alignment(size_cholqr1);
    }

    size_t size_cholqr2 = 0;
    {
        auto const istat = rocsolver_cholqr2_getMemorySize<BATCHED, STRIDED, T, I>(m, n, batch_count,
                                                                                   &size_cholqr2);
        if(istat != rocblas_status_success)
        {
            return (istat);
        }

        adjust_for_alignment(size_cholqr2);
    }

    size_t size_trmm = 0;
    {
        size_trmm = sizeof(T*) * batch_count;
        adjust_for_alignment(size_trmm);
    }

    size_work += std::max({size_trmm, size_cholqr1, size_cholqr2});

    *p_size_work = size_work;
    return (rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T, typename I>
static rocblas_status rocsolver_cholqr_getMemorySize(I const m,
                                                     I const n,
                                                     I const batch_count,
                                                     rocsolver_cholqr_algo const algo,
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

    rocblas_status istat = rocblas_status_success;
    if(algo == rocsolver_cholqr_cholqr1)
    {
        istat
            = rocsolver_cholqr1_getMemorySize<BATCHED, STRIDED, T, I>(m, n, batch_count, &size_work);
    }
    else if((algo == rocsolver_cholqr_cholqr2) || (algo == rocsolver_cholqr_default))
    {
        istat
            = rocsolver_cholqr2_getMemorySize<BATCHED, STRIDED, T, I>(m, n, batch_count, &size_work);
    }
    else if((algo == rocsolver_cholqr_cholqr3_compute) || (algo == rocsolver_cholqr_cholqr3_user))
    {
        istat
            = rocsolver_cholqr3_getMemorySize<BATCHED, STRIDED, T, I>(m, n, batch_count, &size_work);
    }

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
          typename Istride,
          typename UA,
          typename UR,
          typename INFO = I,
          typename S = decltype(std::real(T{}))>
static rocblas_status rocsolver_cholqr1_template(

    rocblas_handle handle,

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

    I const batch_count,

    I* const info,

    void* work,
    size_t const size_work,

    S* const sigma_array = nullptr)
{
    bool constexpr is_complex = rocblas_is_complex<T>;

    bool constexpr is_pointer_batched_A = IS_POINTER_BATCHED(A, T);
    bool constexpr is_pointer_batched_R = IS_POINTER_BATCHED(R, T);

    bool constexpr is_both_same_type = (is_pointer_batched_A == is_pointer_batched_R);

    {
        bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
        if(!has_work)
        {
            return (rocblas_status_success);
        }
    }

    // ----------------------------------------------------
    // everything must be executed with scalars on the host
    // ----------------------------------------------------
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    rocblas_status istat = rocblas_status_success;

    S const zero = 0;
    S const one = 1;

    hipStream_t stream;
    try
    {
        {
            istat = rocblas_get_stream(handle, &stream);
            bool const isok_get_stream
                = (istat == rocblas_status_success) || (istat == rocblas_status_continue);
            if(!isok_get_stream)
            {
                throw istat;
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

        // -------------------------------------
        // form B = A' * A,  reuse storage for R
        // -------------------------------------
        auto const B = R;
        Istride const shiftB = shiftR;
        Istride const strideB = strideR;
        I const ldb = ldr;

        bool constexpr is_pointer_batched_B = is_pointer_batched_R;

        {
            auto const pfree_saved = pfree;

            MEM_CHECK_THROW(pfree);

            size_t const size_remain = (pwork + size_work) - pfree;

            // ------------------
            // compute B = A' * A
            // ------------------

            if constexpr(use_syrk)
            {
                // -------------------------------
                // Note output  matrix for SYRK is nn by nn
                // -------------------------------
                I const nn = n;
                I const kk = m;
                rocblas_fill const uplo = rocblas_fill_upper;
                S alpha = one;
                S beta = zero;

                rocblas_operation const trans1 = rocblas_operation_conjugate_transpose;

                istat = rocblasCall_syrk_herk_alt<T>(handle, uplo, trans1, nn, kk,

                                                     &alpha,

                                                     A, shiftA, lda, strideA,

                                                     &beta,

                                                     B, shiftB, ldb, strideB,

                                                     batch_count, (void*)pfree, size_remain);
            }
            else
            {
                I const mm = n;
                I const nn = n;
                I const kk = m;
                rocblas_operation const trans1 = rocblas_operation_conjugate_transpose;

                rocblas_operation const trans2 = rocblas_operation_none;

                T alpha = 1;
                T beta = 0;

                size_t size_workArr = sizeof(T*) * batch_count;
                adjust_for_alignment(size_workArr);

                T** const workArr = (T**)pfree;
                pfree += size_workArr;

                MEM_CHECK_THROW(pfree);

                istat = rocblasCall_gemm<T>(handle, trans1, trans2, mm, nn, kk,

                                            &alpha,

                                            A, shiftA, lda, strideA,

                                            A, shiftA, lda, strideA,

                                            &beta,

                                            B, shiftB, ldb, strideB,

                                            batch_count, workArr);
            }

            if(istat != rocblas_status_success)
            {
                throw(istat);
            }

            pfree = pfree_saved;
        }

        // -------------------------
        // optional, if sigma != 0
        // B <- B + sigma * identity
        // -------------------------
        if(sigma_array != nullptr)
        {
            add_shift<T, I, Istride>(stream, m, n, batch_count, sigma_array,

                                     B, shiftB, ldb, strideB);
        }

        // -----------------------------------
        // perform Cholesky factorization
        // B = R' * R,   R is upper triangular
        //
        // R will over-write B
        // -----------------------------------
        {
            auto const pfree_saved = pfree;

            MEM_CHECK_THROW(pfree);
            size_t const size_remain = ((pwork + size_work) - pfree);

            rocblas_fill const uplo = rocblas_fill_upper;

            istat = rocsolver_potrf_template_alt<T, I, Istride>(handle, uplo, n,

                                                                B, shiftB, ldb, strideB,

                                                                info, batch_count,

                                                                (void*)pfree, size_remain);

            if(istat != rocblas_status_success)
            {
                throw(istat);
            }

            pfree = pfree_saved;
        }

        // --------------------------------------
        // compute Q = A / R
        //
        // note Q over-writes original matrix A
        // --------------------------------------

        {
            auto const pfree_saved = pfree;

            MEM_CHECK_THROW(pfree);
            size_t const size_remain = (pwork + size_work) - pfree;

            size_t size_trsm = 0;

            auto const mm = m;
            auto const nn = n;
            auto const ld1 = lda;
            auto const ld2 = ldr;

            rocblas_side const side = rocblas_side_right;
            rocblas_fill const uplo = rocblas_fill_upper;
            rocblas_diagonal const diag = rocblas_diagonal_non_unit;
            rocblas_operation const trans1 = rocblas_operation_none;

            istat = rocblasCall_trsm_mem_alt<T, I>(side, trans1, mm, nn, ld1, ld2, batch_count,

                                                   &size_trsm);
            adjust_for_alignment(size_trsm);

            bool const isok_trsm_mem
                = (istat == rocblas_status_success) || (istat == rocblas_status_continue);

            if(!isok_trsm_mem)
            {
                throw(istat);
            }

            T alpha = 1;

            istat = rocblasCall_trsm_alt<T>(handle, side, uplo, trans1, diag,

                                            mm, nn, &alpha,

                                            R, shiftR, ldr, strideR,

                                            A, shiftA, lda, strideA,

                                            batch_count, (void*)pfree, size_remain);

            if(istat != rocblas_status_success)
            {
                throw(istat);
            }

            pfree = pfree_saved;
        }

        rocblas_set_pointer_mode(handle, old_mode);
        return (istat);
    }
    catch(rocblas_status istat)
    {
        rocblas_set_pointer_mode(handle, old_mode);
        return (istat);
    }
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

template <bool BATCHED, bool STRIDED, typename T, typename I, typename Istride, typename UA, typename UR, typename INFO = I>
static rocblas_status rocsolver_cholqr2_template(rocblas_handle handle,
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

                                                 I const batch_count,
                                                 INFO* const info,

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
        };
    }

    // ----------------------------------------------------
    // everything must be executed with scalars on the host
    // ----------------------------------------------------
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
                istat = (rocblas_status_internal_error);
                throw(istat);
            }
        }

        std::byte* const pwork = (std::byte*)work;
        std::byte* pfree = pwork;

        // -------------------------
        // (1) [Q,R1] = cholqr1( A )
        // -------------------------
        I const ldr1 = n;
        Istride const strideR1 = Istride{ldr1} * n;

        size_t size_R1 = sizeof(T) * strideR1 * batch_count;
        adjust_for_alignment(size_R1);

        Istride const shiftR1 = 0;

        T* const R1 = (T*)pfree;
        pfree += size_R1;

        MEM_CHECK_THROW(pfree);

        {
            auto const pfree_saved = pfree;
            MEM_CHECK_THROW(pfree);

            size_t const size_remain = (pwork + size_work) - pfree;

            {
                size_t size_cholqr1 = 0;
                istat = rocsolver_cholqr1_getMemorySize<BATCHED, STRIDED, T>(m, n, batch_count,
                                                                             &size_cholqr1);

                {
                    bool const isok_cholqr1_mem
                        = (istat == rocblas_status_success) || (istat == rocblas_status_continue);
                    if(!isok_cholqr1_mem)
                    {
                        throw(istat);
                    }
                }

                adjust_for_alignment(size_cholqr1);

                bool const is_mem_ok = (size_remain >= size_cholqr1);
                if(!is_mem_ok)
                {
                    istat = rocblas_status_memory_error;
                    throw(istat);
                }
            }

            istat = rocsolver_cholqr1_template<BATCHED, STRIDED, T>(handle,

                                                                    m, n,

                                                                    A, shiftA, lda, strideA,

                                                                    R1, shiftR1, ldr1, strideR1,

                                                                    batch_count, info,

                                                                    (void*)pfree, size_remain);

            if(istat != rocblas_status_success)
            {
                throw(istat);
            }
            pfree = pfree_saved;
        }

        // ------------------------
        // (2) [Q,R] = cholqr1( Q )
        // ------------------------

        // -----------------------------
        // Note: matrix Q over-writes matrix A
        // -----------------------------
        auto const Q = A;
        auto const shiftQ = shiftA;
        auto const ldq = lda;
        auto const strideQ = strideA;
        {
            auto const pfree_saved = pfree;

            size_t size_iinfo = sizeof(INFO) * batch_count;
            adjust_for_alignment(size_iinfo);

            INFO* iinfo = (INFO*)pfree;
            pfree += size_iinfo;

            MEM_CHECK_THROW(pfree);
            size_t const size_remain = (pwork + size_work) - pfree;

            istat = rocsolver_cholqr1_template<BATCHED, STRIDED, T>(handle,

                                                                    m, n,

                                                                    Q, shiftQ, ldq, strideQ,

                                                                    R, shiftR, ldr, strideR,

                                                                    batch_count, iinfo,

                                                                    (void*)pfree, size_remain);

            if(istat != rocblas_status_success)
            {
                throw(istat);
            }

            pfree = pfree_saved;
        }

        // --------------
        // (3) R = R * R1
        // --------------

        // -------------------
        // update  R <- R * R1
        //
        // (i)  set strictly lower triangular part of R be zero
        // (ii) perform TRMM
        // -------------------

        {
            // ----------------------------------------------------
            // (i)  set strictly lower triangular part of R be zero
            // ----------------------------------------------------

            char uplo = 'L';
            T alpha = 0;
            set_triangular(handle, uplo, n, n, alpha,

                           R, shiftR, ldr, strideR,

                           batch_count);
        }

        {
            // -----------------
            // (ii) perform TRMM
            // R <- R * R1
            // -----------------

            auto const pfree_saved = pfree;

            rocblas_side const side = rocblas_side_right;
            rocblas_fill const uplo = rocblas_fill_upper;
            rocblas_operation const trans1 = rocblas_operation_none;
            rocblas_diagonal const diag = rocblas_diagonal_non_unit;
            I const mm = n;
            I const nn = n;
            T alpha = 1;

            size_t size_workArr = sizeof(T*) * batch_count;
            adjust_for_alignment(size_workArr);

            T** const workArr = (T**)pfree;
            pfree += size_workArr;

            MEM_CHECK_THROW(pfree);

            Istride const stride_alpha = 0;

            istat = rocblasCall_trmm<T>(

                handle,

                side, uplo, trans1, diag,

                mm, nn,

                &alpha, stride_alpha,

                R1, shiftR1, ldr1, strideR1,

                R, shiftR, ldr, strideR,

                batch_count, workArr);

            if(istat != rocblas_status_success)
            {
                throw(istat);
            }

            pfree = pfree_saved;
        }

        rocblas_set_pointer_mode(handle, old_mode);
        return (istat);
    }
    catch(rocblas_status istat)
    {
        rocblas_set_pointer_mode(handle, old_mode);
        return (istat);
    }
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

        // -----------
        // allocate R1
        // -----------

        I const ldr1 = n;
        Istride const strideR1 = static_cast<Istride>(ldr1) * n;
        Istride const shiftR1 = 0;

        size_t size_R1 = sizeof(T) * strideR1 * batch_count;
        adjust_for_alignment(size_R1);

        T* const R1 = (T*)pfree;
        pfree += size_R1;

        MEM_CHECK_THROW(pfree);

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
            istat = rocsolver_cholqr1_template<BATCHED, STRIDED, T>(handle,

                                                                    m, n,

                                                                    A, shiftA, lda, strideA,

                                                                    R1, shiftR1, ldr1, strideR1,

                                                                    batch_count, info,

                                                                    (void*)pfree, size_remain,
                                                                    sigma_array);

            if(istat != rocblas_status_success)
            {
                throw(istat);
            }

            pfree = pfree_saved;
        }

        // ----------------
        // (2)   CholQR2(Q)
        // ----------------

        {
            auto const pfree_saved = pfree;

            size_t size_iinfo = sizeof(INFO) * batch_count;
            adjust_for_alignment(size_iinfo);

            INFO* iinfo = (INFO*)pfree;
            pfree += size_iinfo;

            MEM_CHECK_THROW(pfree);

            size_t const size_remain = (pwork + size_work) - pfree;

            istat = rocsolver_cholqr2_template<BATCHED, STRIDED, T>(handle, m, n,

                                                                    Q, shiftQ, ldq, strideQ,

                                                                    R, shiftR, ldr, strideR,

                                                                    batch_count, iinfo,

                                                                    (void*)pfree, size_remain);

            if(istat != rocblas_status_success)
            {
                throw(istat);
            }

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

        {
            // -----------------
            // (ii) perform TRMM
            // R <- R * R1
            // -----------------

            auto const pfree_saved = pfree;

            size_t size_workArr = sizeof(T*) * batch_count;
            adjust_for_alignment(size_workArr);

            T** const workArr = (T**)pfree;
            pfree += size_workArr;

            MEM_CHECK_THROW(pfree);

            rocblas_side const side = rocblas_side_right;
            rocblas_fill const uplo = rocblas_fill_upper;
            rocblas_operation const trans1 = rocblas_operation_none;
            rocblas_diagonal const diag = rocblas_diagonal_non_unit;
            I const mm = n;
            I const nn = n;
            T alpha = 1;

            Istride const stride_alpha = 0;

            istat = rocblasCall_trmm<T>(

                handle,

                side, uplo, trans1, diag,

                mm, nn,

                &alpha, stride_alpha,

                R1, shiftR1, ldr1, strideR1,

                R, shiftR, ldr, strideR,

                batch_count, workArr);

            if(istat != rocblas_status_success)
            {
                throw(istat);
            }

            pfree = pfree_saved;
        }

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

                                                void* work,
                                                size_t const size_work)

{
    rocblas_status istat = rocblas_status_success;

    if(algo == rocsolver_cholqr_cholqr1)
    {
        istat = rocsolver_cholqr1_template<BATCHED, STRIDED, T>(handle,

                                                                m, n,

                                                                A, shiftA, lda, strideA,

                                                                R, shiftR, ldr, strideR,

                                                                batch_count, info,

                                                                work, size_work);
    }
    else if((algo == rocsolver_cholqr_cholqr2) || (algo == rocsolver_cholqr_default))
    {
        istat = rocsolver_cholqr2_template<BATCHED, STRIDED, T>(handle, m, n,

                                                                A, shiftA, lda, strideA,

                                                                R, shiftR, ldr, strideR,

                                                                batch_count, info,

                                                                work, size_work);
    }
    else if((algo == rocsolver_cholqr_cholqr3_compute) || (algo == rocsolver_cholqr_cholqr3_user))
    {
        bool const compute_sigma = (algo == rocsolver_cholqr_cholqr3_compute);

        istat = rocsolver_cholqr3_template<BATCHED, STRIDED, T>(handle, m, n,

                                                                A, shiftA, lda, strideA,

                                                                R, shiftR, ldr, strideR,

                                                                compute_sigma, sigma_array,

                                                                info, batch_count,

                                                                work, size_work);
    }

    return (istat);
}

#undef IS_POINTER_BATCHED
#undef MEM_CHECK
#undef MEM_CHECK_THROW
ROCSOLVER_END_NAMESPACE

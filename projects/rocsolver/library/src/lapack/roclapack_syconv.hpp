/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
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
#include "rocsolver/rocsolver.h"

#include "rocprim/rocprim.hpp"

ROCSOLVER_BEGIN_NAMESPACE

#ifndef SYCONV_MAX_THDS
#define SYCONV_MAX_THDS 256
#endif

// -------------------------------------------------------------------------
// SYCONV is an auxiliary computational routine that converts a complex
// symmetric matrix A, which has already been factorized into L D L^t or U D
// U^t by SYTRF, into a different representation where the optimized triangular
// solver TRSM can be used. The off-diagonal entries in the 2x2 blocks are
// stored in array "E".  The routine has an option to reverse the conversion
// back into the original representation produced by SYTRF.
// -------------------------------------------------------------------------

// --------------------------------------------------------
// Routine to restore from array E[] the off-diagonal entry
// of 2x2 blocks into matrix A
// --------------------------------------------------------
template <typename T, typename I, typename UA, typename Istride>
static __global__
    __launch_bounds__(SYCONV_MAX_THDS) void syconv_restore_from_E(bool const is_upper,
                                                                  I const n,
                                                                  I const batch_count,
                                                                  UA A_arg,
                                                                  Istride const shiftA,
                                                                  I const lda,
                                                                  Istride const strideA,
                                                                  I const* const ipiv_arg,
                                                                  Istride const strideP,
                                                                  T const* const E_arg,
                                                                  Istride const strideE,
                                                                  I const* const icount_arg)
{
    if(n == 0 || batch_count == 0)
        return;

    auto idx2F
        = [](auto i, auto j, auto ld) { return ((i - 1) + (j - 1) * static_cast<int64_t>(ld)); };

    auto is_even = [](auto n) -> bool { return ((n % 2) == 0); };

    I const bid_start = blockIdx.z;
    I const bid_inc = gridDim.z;

    I const nthreads = (blockDim.x * blockDim.y) * blockDim.z;
    I const tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);

    I const ibx = blockIdx.x;
    I const nbx = gridDim.x;

    I const ij_start = tid + ibx * nthreads;
    I const ij_inc = nthreads * nbx;

    for(I bid = 0 + bid_start; bid < batch_count; bid += bid_inc)
    {
        T const* const E_bid = E_arg + bid * strideE;
        T* const A_bid = load_ptr_batch<T>(A_arg, bid, shiftA, strideA);
        I const* const icount_bid = icount_arg + bid * n;
        I const* const ipiv_bid = ipiv_arg + bid * strideP;

        auto A = [=](I i, I j) -> T& { return (A_bid[idx2F(i, j, lda)]); };
        auto E = [=](I i) -> T { return (E_bid[(i - 1)]); };
        auto ipiv = [=](I i) { return (ipiv_bid[(i - 1)]); };

        auto is_second_pivot = [=](I i) -> bool { return (is_even(icount_bid[(i - 1)])); };

        if(is_upper)
        {
            // ----------
            // upper part
            // ----------

            for(I i = 1 + ij_start; i <= n; i += ij_inc)
            {
                bool const is_2x2_block = (ipiv(i) < 0);
                if(is_2x2_block)
                {
                    if(is_second_pivot(i))
                    {
                        A(i - 1, i) = E(i);
                    }
                }
            } // end for i
        }
        else
        {
            // ----------
            // lower part
            // ----------

            for(I i = 1 + ij_start; i <= n; i += ij_inc)
            {
                if(ipiv(i) < 0)
                {
                    if(!is_second_pivot(i))
                    {
                        A(i + 1, i) = E(i);
                    }
                }
            } // end for i
        }

        __syncthreads();
    } // end for bid
}

//  ------------------------------------------------------
//  Routine to copy the off-diagonal entry of 2x2 blocks into array E
//  and zero the corresponding entry in matrix A
//  ------------------------------------------------------
template <typename T, typename I, typename UA, typename Istride>
static __global__
    __launch_bounds__(SYCONV_MAX_THDS) void syconv_convert_into_E(bool const is_upper,
                                                                  I const n,
                                                                  I const batch_count,
                                                                  UA A_arg,
                                                                  Istride const shiftA,
                                                                  I const lda,
                                                                  Istride const strideA,
                                                                  I const* const ipiv_arg,
                                                                  Istride const strideP,
                                                                  T* const E_arg,
                                                                  Istride const strideE,
                                                                  I const* const icount_arg

    )
{
    if(n == 0 || batch_count == 0)
        return;

    auto idx2F
        = [](auto i, auto j, auto ld) { return ((i - 1) + (j - 1) * static_cast<int64_t>(ld)); };

    auto is_even = [](auto n) -> bool { return ((n % 2) == 0); };

    I const bid_start = blockIdx.z;
    I const bid_inc = gridDim.z;

    I const nthreads = (blockDim.x * blockDim.y) * blockDim.z;
    I const tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);

    I const ibx = blockIdx.x;
    I const nbx = gridDim.x;

    I const ij_start = tid + ibx * nthreads;
    I const ij_inc = nthreads * nbx;

    T const zero = 0;

    for(I bid = 0 + bid_start; bid < batch_count; bid += bid_inc)
    {
        T* const E_bid = E_arg + bid * strideE;
        T* const A_bid = load_ptr_batch<T>(A_arg, bid, shiftA, strideA);
        I const* const icount_bid = icount_arg + bid * n;
        I const* const ipiv_bid = ipiv_arg + bid * strideP;

        auto A = [=](I i, I j) -> T& { return (A_bid[idx2F(i, j, lda)]); };
        auto E = [=](I i) -> T& { return (E_bid[(i - 1)]); };
        auto ipiv = [=](I i) { return (ipiv_bid[(i - 1)]); };

        auto is_second_pivot = [=](I i) -> bool { return (is_even(icount_bid[(i - 1)])); };

        if(is_upper)
        {
            // ----------
            // upper part
            // ----------

            for(I i = 1 + ij_start; i <= n; i += ij_inc)
            {
                bool const use_2x2_block = (ipiv(i) < 0);
                if(use_2x2_block)
                {
                    if(is_second_pivot(i))
                    {
                        E(i) = A(i - 1, i);
                        E(i - 1) = zero;

                        A(i - 1, i) = zero;
                    }
                }
                else
                {
                    E(i) = zero;
                }
            } // end for i
        }
        else
        {
            // ----------
            // lower part
            // ----------

            for(I i = 1 + ij_start; i <= n; i += ij_inc)
            {
                bool const use_2x2_block = ((i < n) && (ipiv(i) < 0));
                if(use_2x2_block)
                {
                    if(!is_second_pivot(i))
                    {
                        E(i) = A(i + 1, i);
                        E(i + 1) = zero;
                        A(i + 1, i) = zero;
                    }
                }
                else
                {
                    E(i) = zero;
                }
            } // end for i
        }

        __syncthreads();
    } // end for bid
}

// --------------------------------------------------------------
// Routine to setup array icount[] to count the number of
// negative index entries, in preparation for the prefix sum scan
// --------------------------------------------------------------
template <typename I, typename Istride>
static __global__
    __launch_bounds__(SYCONV_MAX_THDS) void syconv_setup_icount_kernel(I const n,
                                                                       I const batch_count,
                                                                       I* const ipiv_arg,
                                                                       Istride const strideP,
                                                                       I* const icount_arg)
{
    I const bid_start = blockIdx.z;
    I const bid_inc = gridDim.z;

    I const nthreads = (blockDim.x * blockDim.y) * blockDim.z;
    I const tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);

    I const ibx = blockIdx.x;
    I const nbx = gridDim.x;
    I const i_start = tid + ibx * nthreads;
    I const i_inc = nbx * nthreads;

    for(I bid = 0 + bid_start; bid < batch_count; bid += bid_inc)
    {
        I* const ipiv = ipiv_arg + static_cast<int64_t>(bid) * strideP;
        I* const icount = icount_arg + static_cast<int64_t>(bid) * n;

        for(I i = 0 + i_start; i < n; i += i_inc)
        {
            icount[i] = (ipiv[i] < 0);
        }
    }
}

// --------------------------------------------------------------------------------------
// Main kernel for syconv. This is called after E[] array has setup for conversion
// or perform the recovery but the recovery of the off-diagonal entries in 2x2 blocks are
// performed by kernel syconv_restore_from_E()
// --------------------------------------------------------------------------------------
template <typename T, typename I, typename UA, typename Istride>
static __global__ __launch_bounds__(SYCONV_MAX_THDS) void syconv_kernel(bool const is_upper,
                                                                        bool const is_convert,
                                                                        I const n,
                                                                        UA A_arg,
                                                                        Istride const shiftA,
                                                                        I const lda,
                                                                        Istride const strideA,
                                                                        I* const ipiv_arg,
                                                                        Istride const strideP,
                                                                        I const batch_count)
{
    if(n == 0 || batch_count == 0)
        return;

    I const bid_start = static_cast<I>(blockIdx.z);
    I const bid_inc = static_cast<I>(gridDim.z);

    I const nbx = static_cast<I>(gridDim.x);
    I const ibx = static_cast<I>(blockIdx.x);

    I const ij_start = static_cast<I>(threadIdx.x + threadIdx.y * blockDim.x
                                      + threadIdx.z * (blockDim.x * blockDim.y));
    I const ij_inc = static_cast<I>((blockDim.x * blockDim.y) * blockDim.z);

    auto ceildiv = [](auto m, auto b) { return ((m <= 0) ? 0 : (m - 1) / b + 1); };

    // ------------------------------------------------
    // note: this thread block handles only columns in
    // [gcol_start,gcol_end] inclusively and
    // gcol_start is using 1-based indexing
    // ------------------------------------------------
    I const nb = ceildiv(n, nbx);
    I const gcol_start = 1 + ibx * nb;
    I const gcol_end = std::min(n, gcol_start + nb - 1);

    // ------------------------
    // Fortran 1-based indexing
    // ------------------------
    auto idx2F
        = [](auto i, auto j, auto ld) { return ((i - 1) + (j - 1) * static_cast<int64_t>(ld)); };

    auto swap = [](T& x, T& y) {
        auto const temp = x;
        x = y;
        y = temp;
    };

    for(I bid = 0 + bid_start; bid < batch_count; bid += bid_inc)
    {
        T* const A_bid = load_ptr_batch<T>(A_arg, bid, shiftA, strideA);
        I* const ipiv_bid = ipiv_arg + bid * strideP;

        // ----------------------------------------------------
        // use 1-based indexing compatible with  Fortran/matlab
        // ----------------------------------------------------
        auto A = [=](auto i, auto j) -> T& { return (A_bid[idx2F(i, j, lda)]); };

        auto ipiv = [=](auto i) -> I { return (ipiv_bid[(i - 1)]); };

        // ------------------------------------------------------------
        // note __syncthreads() is not needed in syconv_swap_rows()
        // since each column is always consistently handled by the same thread
        //
        //
        // swap( A(row_k, col_start:col_end), A(row_kp, col_start:col_end)
        // ------------------------------------------------------------
        auto syconv_swap_rows
            = [=](I const row_k, I const row_kp, I const col_start, I const col_end) {
                  for(I icol = gcol_start + ij_start; icol <= gcol_end; icol += ij_inc)
                  {
                      bool const is_in_range = (col_start <= icol) && (icol <= col_end);
                      if(is_in_range)
                      {
                          swap(A(row_k, icol), A(row_kp, icol));
                      }
                  }
              };

        if(is_upper)
        {
            // ---------------------
            // A is upper triangular
            // ---------------------
            if(is_convert)
            {
                // -------------------------------
                //           convert A (A is upper triangular)
                //
                //           convert value
                // -------------------------------

                // -------------------------------
                //           convert permutations
                // -------------------------------
                I i = n;
                while(i >= 1)
                {
                    if(ipiv(i) > 0)
                    {
                        I const ip = ipiv(i);
                        if(i < n)
                        {
                            auto const col_start = (i + 1);
                            auto const col_end = n;
                            auto const irow_k = i;
                            auto const irow_kp = ip;
                            syconv_swap_rows(irow_k, irow_kp, col_start, col_end);
                        }
                    }
                    else
                    {
                        I const ip = -ipiv(i);
                        if(i < n)
                        {
                            auto const col_start = (i + 1);
                            auto const col_end = n;
                            auto const irow_k = (i - 1);
                            auto const irow_kp = ip;

                            syconv_swap_rows(irow_k, irow_kp, col_start, col_end);
                        }
                        i = i - 1;
                    }
                    i = i - 1;
                } // end while
            }
            else
            {
                // ------------------------------------------
                //           revert A (A is upper triangular)
                //
                //           revert permutations
                // ------------------------------------------
                I i = 1;

                while(i <= n)
                {
                    if(ipiv(i) > 0)
                    {
                        I const ip = ipiv(i);
                        if(i < n)
                        {
                            auto const col_start = (i + 1);
                            auto const col_end = n;
                            auto const irow_k = i;
                            auto const irow_kp = ip;

                            syconv_swap_rows(irow_k, irow_kp, col_start, col_end);
                        }
                    }
                    else
                    {
                        I const ip = -ipiv(i);
                        i = i + 1;
                        if(i < n)
                        {
                            auto const col_start = (i + 1);
                            auto const col_end = n;
                            auto const irow_k = (i - 1);
                            auto const irow_kp = ip;

                            syconv_swap_rows(irow_k, irow_kp, col_start, col_end);
                        }
                    }
                    i = i + 1;
                } // end while

            } // end if (is_revert)
        } // end if (is_upper)
        else
        {
            // ------------------------------
            //          A is lower triangular
            // ------------------------------

            if(is_convert)
            {
                // -------------------------------------------
                //           convert A (A is lower triangular)
                //
                //           convert value
                // -------------------------------------------

                // --------------------------------
                //             convert permutations
                // --------------------------------
                I i = 1;
                while(i <= n)
                {
                    if(ipiv(i) > 0)
                    {
                        I const ip = ipiv(i);
                        if(i > 1)
                        {
                            auto const col_start = 1;
                            auto const col_end = (i - 1);
                            auto const irow_k = i;
                            auto const irow_kp = ip;

                            syconv_swap_rows(irow_k, irow_kp, col_start, col_end);
                        }
                    }
                    else
                    {
                        I const ip = -ipiv(i);
                        if(i > 1)
                        {
                            auto const col_start = 1;
                            auto const col_end = (i - 1);
                            auto const irow_k = i + 1;
                            auto const irow_kp = ip;

                            syconv_swap_rows(irow_k, irow_kp, col_start, col_end);
                        }
                        i = i + 1;
                    }
                    i = i + 1;
                } // end while
            }
            else
            {
                //  -------------------------------
                //            revert A (A is lower triangular)
                //
                //            revert permutations
                //  -------------------------------
                I i = n;
                while(i >= 1)
                {
                    if(ipiv(i) > 0)
                    {
                        I const ip = ipiv(i);
                        if(i > 1)
                        {
                            auto const col_start = 1;
                            auto const col_end = (i - 1);
                            auto const irow_k = i;
                            auto const irow_kp = ip;

                            syconv_swap_rows(irow_k, irow_kp, col_start, col_end);
                        }
                    }
                    else
                    {
                        I const ip = -ipiv(i);
                        i = i - 1;
                        if(i > 1)
                        {
                            auto const col_start = 1;
                            auto const col_end = (i - 1);
                            auto const irow_k = (i + 1);
                            auto const irow_kp = ip;

                            syconv_swap_rows(irow_k, irow_kp, col_start, col_end);
                        }
                    }
                    i = i - 1;
                } // end while
            }
        }
        __syncthreads();
    } // end for bid
}

template <typename T, typename I, typename Istride, typename UA>
static inline rocblas_status rocsolver_syconv_argCheck(rocblas_handle handle,
                                                       [[maybe_unused]] bool const is_upper,
                                                       [[maybe_unused]] bool const is_convert,
                                                       I const n,
                                                       UA A,
                                                       Istride const shiftA,
                                                       I const lda,
                                                       I* const ipiv,
                                                       T* const E,
                                                       I const batch_count,
                                                       void* const work,
                                                       size_t const size_work)
{
    // order is important for unit tests:

    // 0. invalid handle
    if(handle == nullptr)
        return rocblas_status_invalid_handle;

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(n < 0 || lda < n || batch_count < 0)
        return rocblas_status_invalid_size;
    if(size_work < sizeof(I) * n * batch_count)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && batch_count && !A) || (n && batch_count && !ipiv) || (n && batch_count && !E)
       || (n && batch_count && !work))
        return rocblas_status_invalid_pointer;

    return (rocblas_status_continue);
}

template <typename T, typename I>
static inline rocblas_status rocsolver_syconv_getMemorySize(rocblas_handle handle,
                                                            I const n,
                                                            I const batch_count,
                                                            size_t* p_size_work)
{
    *p_size_work = 0;

    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    size_t const size_icount = sizeof(I) * n * batch_count;

    size_t size_rocprim = 0;
    {
        hipStream_t stream;
        rocblas_get_stream(handle, &stream);

        // Functor to generate offsets
        struct mul_n
        {
            I n;
            __host__ __device__ I operator()(I i) const
            {
                return i * n;
            }
        };

        auto counting = rocprim::make_counting_iterator<I>(0);
        auto offsets = rocprim::make_transform_iterator(counting, mul_n{n});

        void* temp = nullptr;
        size_t temp_bytes = 0;

        // ------------------------------
        // size query for scratch storage in in-place prefix sum scan
        //
        // NOTE:  rocprim expect argument temp_bytes as  size_t&
        //        and not as size_t *
        //
        // ------------------------------
        {
            // ------------------------------------------------
            // NOTE: rocprim needs to deduce the type from d_data
            // so pass in  I * pointer that has value nullptr
            // ------------------------------------------------
            I* const d_data = nullptr;

            HIP_CHECK(rocprim::segmented_inclusive_scan(temp, temp_bytes,
                                                        d_data, // input
                                                        d_data, // output (same pointer)
                                                        batch_count, offsets, offsets + 1,
                                                        rocprim::plus<I>(), stream));

            size_rocprim = temp_bytes;
        }
    }

    *p_size_work = size_icount + size_rocprim;

    return rocblas_status_success;
}

//  -----------------------------------
// if (is_convert == true) syconv_template converts a given factorization
// produced by SYTRF into L, U factors ready for TRSM.
// NOTE: The content of A will be modified.
//
// if (is_convert == false) syconv_template reverts to the original
// factorization produced by SYTRF
//  -----------------------------------
template <typename T, typename I, typename UA, typename Istride = rocblas_stride>
static rocblas_status rocsolver_syconv_template(rocblas_handle handle,
                                                bool const is_upper,
                                                bool const is_convert,
                                                I const n,
                                                UA A,
                                                Istride const shiftA,
                                                I const lda,
                                                Istride const strideA,
                                                I* const ipiv,
                                                Istride const strideP,
                                                T* const E,
                                                Istride const strideE,
                                                I const batch_count,
                                                void* const work,
                                                size_t const size_work)
{
    // --------------
    // check arguments
    // --------------
    rocblas_status st = rocsolver_syconv_argCheck(handle, is_upper, is_convert, n, A, shiftA, lda,
                                                  ipiv, E, batch_count, work, size_work);
    if(st != rocblas_status_continue)
        return st;

    auto ceildiv = [](auto n, auto b) { return ((n <= 0) ? 0 : (n - 1) / b + 1); };

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I const max_blocks = 64 * 1024 - 3;
    I const nbx = std::max(I(1), std::min(max_blocks, ceildiv(n, I(SYCONV_MAX_THDS))));
    I const nby = 1;
    I const nbz = std::max(I(1), std::min(max_blocks, batch_count));

    I const ln = ceildiv(n, nbx);
    I const nx = std::min(ln, I(SYCONV_MAX_THDS));
    I const ny = 1;
    I const nz = 1;

    std::byte* const pwork = reinterpret_cast<std::byte*>(work);
    std::byte* pfree = pwork;

    I* const icount = reinterpret_cast<I*>(pfree);
    pfree += sizeof(I) * n * batch_count;

    // -------------------------------------------------------------
    // setup icount[] to indicate whether there  is a negative pivot
    // -------------------------------------------------------------
    ROCSOLVER_LAUNCH_KERNEL(syconv_setup_icount_kernel, dim3(nbx, nby, nbz), dim3(nx, ny, nz), 0,
                            stream, n, batch_count, ipiv, strideP, icount);

    // -----------------------------------------------
    // compute segmented prefix sum scan using rocprim
    //
    // there are batch_count number of segments
    // and each segment is  length n
    // -----------------------------------------------
    // Functor to generate offsets
    struct mul_n
    {
        I n;
        __host__ __device__ I operator()(I i) const
        {
            return i * n;
        }
    };

    auto segmented_inclusive_scan_inplace = [=](I* d_data, I const n, I const batch_count) {
        auto counting = rocprim::make_counting_iterator<I>(0);
        auto offsets = rocprim::make_transform_iterator(counting, mul_n{n});

        void* temp = nullptr;
        size_t temp_bytes = 0;

        // ------------------------------
        // size query for scratch storage in in-place prefix sum scan
        //
        // NOTE:  rocprim expect argument temp_bytes as  size_t&
        //        and not as size_t *
        // ------------------------------
        HIP_CHECK(rocprim::segmented_inclusive_scan(temp, temp_bytes,
                                                    d_data, // input
                                                    d_data, // output (same pointer)
                                                    batch_count, offsets, offsets + 1,
                                                    rocprim::plus<I>(), stream));

        size_t const size_remain = (pwork + size_work) - pfree;
        if(temp_bytes > size_remain)
            return rocblas_status_memory_error;

        temp = reinterpret_cast<void*>(pfree);

        // --------------------------------
        // actual execution for (in place)
        // prefix sum scan
        // --------------------------------
        HIP_CHECK(rocprim::segmented_inclusive_scan(temp, temp_bytes, d_data, d_data, batch_count,
                                                    offsets, offsets + 1, rocprim::plus<I>(), stream));

        return rocblas_status_success;
    };

    ROCBLAS_CHECK(segmented_inclusive_scan_inplace(icount, n, batch_count));

    // ----------------------------------------
    // NOTE: icount now contains the prefix sum
    // ----------------------------------------
    if(is_convert)
    {
        ROCSOLVER_LAUNCH_KERNEL((syconv_convert_into_E<T, I, UA, Istride>), dim3(nbx, nby, nbz),
                                dim3(nx, ny, nz), 0, stream, is_upper, n, batch_count, A, shiftA,
                                lda, strideA, ipiv, strideP, E, strideE, icount);
    }

    ROCSOLVER_LAUNCH_KERNEL((syconv_kernel<T, I, UA>), dim3(nbx, nby, nbz), dim3(nx, ny, nz), 0,
                            stream, is_upper, is_convert, n, A, shiftA, lda, strideA, ipiv, strideP,
                            batch_count);

    // --------------------------------------------------
    // restore back to format initially produced by SYTRF
    // --------------------------------------------------
    if(!is_convert)
    {
        ROCSOLVER_LAUNCH_KERNEL((syconv_restore_from_E<T, I, UA, Istride>), dim3(nbx, nby, nbz),
                                dim3(nx, ny, nz), 0, stream, is_upper, n, batch_count, A, shiftA,
                                lda, strideA, ipiv, strideP, E, strideE, icount);
    }

    return rocblas_status_success;
}
ROCSOLVER_END_NAMESPACE
#undef SYCONV_MAX_THDS

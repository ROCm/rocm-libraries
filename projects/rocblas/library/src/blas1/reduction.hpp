/* ************************************************************************
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include "handle.hpp"
#include "rocblas.h"
#include "utility.hpp"
#include <type_traits>
#include <utility>

/*
 * ===========================================================================
 *    This file provide common device function used in various BLAS routines
 * ===========================================================================
 */

// BLAS Level 1 includes routines and functions performing vector-vector
// operations. Most BLAS 1 routines are about reduction: compute the norm,
// calculate the dot production of two vectors, find the maximum/minimum index
// of the element of the vector. As you may observed, although the computation
// type is different, the core algorithm is the same: scan all element of the
// vector(s) and reduce to one single result.
//
// The primary reduction algorithm now uses shuffle instructions that use a binary
// tree like reduction of masked out channels, but follows almost the
// same pattern as the recursive reduction algorithm on GPU that is called [parallel
// reduction](https://raw.githubusercontent.com/mateuszbuda/GPUExample/master/reduce3.png)
// which is also adopted in rocBLAS. At the beginning, all the threads in the thread
// block participate. After each step of reduction (like a tree), the number of
// participating threads decrease by half. At the end of the parallel reduction,
// only one thread (usually thread 0) owns the result in its thread block.
//
// Classically, the BLAS 1 reduction needs more than one GPU kernel to finish,
// because the lack of global synchronization of thread blocks without exiting
// the kernel. The first kernels gather partial results, write into a temporary
// working buffer. The second kernel finishes the final reduction.
//
// For example, BLAS 1 routine i*amax is to find index of the maximum absolute
// value element of a vector. In this routine:
//
// Kernel 1: launch many thread block as needed. Each thread block works on a
// subset of the vector. Each thread block use the parallel reduction to find a
// local index with the maximum absolute value of the subset. There are
// number-of-the-thread-blocks local results.The results are written into a
// temporary working buffer. The working buffer has number-of-the-thread-blocks
// elements.
//
// Kernel 2: launch only one thread block which reads the temporary work buffer and
// reduces to final result still with the parallel reduction.
//
// As you may see, if there is a mechanism to synchronize all the thread blocks
// after local index is obtained in kernel 1 (without ending the kernel), then
// Kernel 2's computation can be merged into Kernel 1. One such mechanism is called
// atomic operation. However, atomic operation is new and is not used in rocBLAS
// yet. rocBLAS still use the classic standard parallel reduction right now.

// Recursively compute reduction
template <rocblas_int k, typename REDUCE, typename T>
struct rocblas_reduction_s
{
    __forceinline__ __device__ void operator()(rocblas_int tx, T* x) const
    {
        // Reduce the lower half with the upper half
        if(tx < k)
            REDUCE{}(x[tx], x[tx + k]);
        __syncthreads();

        // Recurse down with k / 2
        rocblas_reduction_s<k / 2, REDUCE, T>{}(tx, x);
    }
};

// leaf node for terminating recursion
template <typename REDUCE, typename T>
struct rocblas_reduction_s<0, REDUCE, T>
{
    __forceinline__ __device__ void operator()(rocblas_int tx, T* x) const {}
};

/*! \brief general parallel reduction

    \details

    @param[in]
    n         rocblas_int. assume a power of 2
    @param[in]
    T         element type of vector x
    @param[in]
    REDUCE    reduction functor
    @param[in]
    tx        rocblas_int. thread id
    @param[inout]
    x         pointer storing vector x on the GPU.
              usually x is stored in shared memory;
              x[0] store the final result.
    ********************************************************************/
template <rocblas_int NB, typename REDUCE, typename T>
__attribute__((flatten)) __device__ void rocblas_reduction(rocblas_int tx, T* x)
{
    static_assert(NB > 1 && !(NB & (NB - 1)), "NB must be a power of 2");
    __syncthreads();
    rocblas_reduction_s<NB / 2, REDUCE, T>{}(tx, x);
}

/*! \brief parallel reduction: sum

    \details

    @param[in]
    n         rocblas_int. assume a power of 2
    @param[in]
    tx        rocblas_int. thread id
    @param[inout]
    x         pointer storing vector x on the GPU.
              usually x is stored in shared memory;
              x[0] store the final result.
    ********************************************************************/
struct rocblas_reduce_sum
{
    template <typename T>
    __forceinline__ __device__ void operator()(T& __restrict__ a, const T& __restrict__ b) const
    {
        a += b;
    }
};

template <rocblas_int NB, typename T>
__attribute__((flatten)) __device__ void rocblas_sum_reduce(rocblas_int tx, T* x)
{
    rocblas_reduction<NB, rocblas_reduce_sum>(tx, x);
}
// end sum_reduce

// Identity finalizer
struct rocblas_finalize_identity
{
    template <typename T>
    __forceinline__ __host__ __device__ T&& operator()(T&& x)
    {
        return std::forward<T>(x); // Perfect identity, preserving valueness
    }
};

// Emulates value initialization T{}. Allows specialization for certain types.
template <typename T>
struct rocblas_default_value
{
    __forceinline__ __host__ __device__ constexpr T operator()() const
    {
        return {};
    }
};

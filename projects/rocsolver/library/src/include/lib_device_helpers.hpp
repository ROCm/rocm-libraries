/* **************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cassert>

#include <hip/hip_runtime.h>

#include "ideal_sizes.hpp"
#include "lib_macros.hpp"
#include "libcommon.hpp"

#if defined(__GFX9__)
__device__ static constexpr int WarpSize = 64;
#else
__device__ static constexpr int WarpSize = 32;
#endif

ROCSOLVER_BEGIN_NAMESPACE

/*
 * ===========================================================================
 *    common location for device functions and kernels that are used across
 *    several rocSOLVER routines, excepting those device functions and kernels
 *    that reproduce LAPACK functionality (see lapack_device_functions.hpp).
 * ===========================================================================
 */

// **********************************************************
// device functions that are used by many kernels
// **********************************************************

template <typename S, typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
__device__ S aabs(T val)
{
    return std::abs(val);
}

template <typename S, typename T, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
__device__ S aabs(T val)
{
    return asum(val);
}

template <typename T>
__device__ __forceinline__ void swap(T& a, T& b)
{
    T temp = a;
    a = b;
    b = temp;
}

template <typename T>
__device__ __host__ void
    swap(const rocblas_int n, T* a, const rocblas_int inca, T* b, const rocblas_int incb)
{
    int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(tid < n)
        swap(a[inca * tid], b[incb * tid]);
}

template <typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
__device__ __inline__ T shift_left(T& value, int lane_delta)
{
    T r = value;
    r = __shfl_down(r, lane_delta);
    return r;
}

template <typename T, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
__device__ __inline__ T shift_left(T& value, int lane_delta)
{
    using S = decltype(std::real(T{}));
    S r = value.real();
    S i = value.imag();
    r = __shfl_down(r, lane_delta);
    i = __shfl_down(i, lane_delta);
    return rocblas_complex_num<S>(r, i);
}

/** SWAPVECT device function swap vectors a and b of dimension n **/
template <typename T>
__device__ void
    swapvect(const rocblas_int n, T* a, const rocblas_int inca, T* b, const rocblas_int incb)
{
    // check common special case
    if((inca == 1) && (incb == 1))
    {
        for(rocblas_int i = 0; i < n; ++i)
        {
            swap(a[i], b[i]);
        };
    }
    else
    {
        for(rocblas_int i = 0; i < n; ++i)
            swap(a[inca * i], b[incb * i]);
    };
}

/** FIND_MAX_TRIDIAG finds the element with the largest magnitude in the
    tridiagonal matrix **/
template <typename T>
__device__ __host__ T find_max_tridiag(const rocblas_int start, const rocblas_int end, T* D, T* E)
{
    T anorm = abs(D[end]);
    for(int i = start; i < end; i++)
        anorm = std::fmax(anorm, std::fmax(abs(D[i]), abs(E[i])));
    return anorm;
}

/** SCALE_TRIDIAG scales the elements of the tridiagonal matrix by a given
    scale factor **/
template <typename T>
__device__ __host__ void scale_tridiag(const rocblas_int start,
                                       const rocblas_int end,
                                       T* D,
                                       T* E,
                                       T scale,
                                       const rocblas_int tid = 0,
                                       const rocblas_int tid_inc = 1)
{
    if(tid == 0)
        D[end] *= scale;

    for(int i = tid + start; i < end; i += tid_inc)
    {
        D[i] *= scale;
        E[i] *= scale;
    }
}

template <typename S, typename I>
__device__ static void shell_sort_ascending(const I n, S* a, I* map = nullptr)
{
    // -----------------------------------------------
    // Sort array a[0...(n-1)] and generate permutation vector
    // in map[] if map[] is available
    // Note: performs in a single thread block
    // -----------------------------------------------
    if((n <= 0) || (a == nullptr))
    {
        return;
    };

    bool const is_root_thread
        = (hipThreadIdx_x == 0) && (hipThreadIdx_y == 0) && (hipThreadIdx_z == 0);

    int constexpr ngaps = 8;
    int const gaps[ngaps] = {701, 301, 132, 57, 23, 10, 4, 1}; // # Ciura gap sequence

    auto const k_start = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x
        + hipThreadIdx_z * (hipBlockDim_x * hipBlockDim_y);

    auto const k_inc = (hipBlockDim_x * hipBlockDim_y) * hipBlockDim_z;

    bool const has_map = (map != nullptr);

    __syncthreads();
    if(has_map)
    {
        for(auto k = k_start; k < n; k += k_inc)
        {
            map[k] = k;
        };
    };
    __syncthreads();

    if(is_root_thread)
    {
        for(auto igap = 0; igap < ngaps; igap++)
        {
            auto const gap = gaps[igap];

            for(I i = gap; i < n; i += 1)
            {
                {
                    S const temp = a[i];
                    auto const itemp = (has_map) ? map[i] : 0;

                    auto j = i;

                    while((j >= gap) && (a[j - gap] > temp))
                    {
                        a[j] = a[j - gap];
                        if(has_map)
                        {
                            map[j] = map[j - gap];
                        };
                        j -= gap;
                    };

                    a[j] = temp;
                    if(has_map)
                    {
                        map[j] = itemp;
                    };
                };
            };
        };
    };
    __syncthreads();
#ifdef NDEBUG
#else
    if(n >= 2)
    {
        for(auto k = k_start; k < (n - 1); k += k_inc)
        {
            assert(std::isnan(a[k]) || std::isnan(a[k + 1]) || a[k] <= a[k + 1]);
        };
    };
    __syncthreads();
#endif
}

template <typename S, typename I>
__device__ static void shell_sort_descending(const I n, S* a, I* map = nullptr)
{
    // -----------------------------------------------
    // Sort array a[0...(n-1)] and generate permutation vector
    // in map[] if map[] is available
    // Note: performs in a single thread block
    // -----------------------------------------------

    if((n <= 0) || (a == nullptr))
    {
        return;
    };

    bool const is_root_thread
        = (hipThreadIdx_x == 0) && (hipThreadIdx_y == 0) && (hipThreadIdx_z == 0);

    int constexpr ngaps = 8;
    int const gaps[ngaps] = {701, 301, 132, 57, 23, 10, 4, 1}; // # Ciura gap sequence

    auto const k_start = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x
        + hipThreadIdx_z * (hipBlockDim_x * hipBlockDim_y);

    auto const k_inc = (hipBlockDim_x * hipBlockDim_y) * hipBlockDim_z;

    bool const has_map = (map != nullptr);

    __syncthreads();
    if(has_map)
    {
        for(auto k = k_start; k < n; k += k_inc)
        {
            map[k] = k;
        };
    };
    __syncthreads();

    if(is_root_thread)
    {
        for(auto igap = 0; igap < ngaps; igap++)
        {
            auto const gap = gaps[igap];

            for(rocblas_int i = gap; i < n; i += 1)
            {
                {
                    S const temp = a[i];
                    auto const itemp = (has_map) ? map[i] : 0;

                    auto j = i;

                    while((j >= gap) && (a[j - gap] < temp))
                    {
                        a[j] = a[j - gap];
                        if(has_map)
                        {
                            map[j] = map[j - gap];
                        };
                        j -= gap;
                    };

                    a[j] = temp;
                    if(has_map)
                    {
                        map[j] = itemp;
                    };
                };
            };
        };
    };
    __syncthreads();
#ifdef NDEBUG
#else
    if(n >= 2)
    {
        for(auto k = k_start; k < (n - 1); k += k_inc)
        {
            assert(std::isnan(a[k]) || std::isnan(a[k + 1]) || a[k] >= a[k + 1]);
        };
    };
    __syncthreads();
#endif
}

template <typename S, typename I>
__device__ static void shell_sort(const I n, S* a, I* map = nullptr, const bool is_ascending = true)
{
    if((n <= 0) || (a == nullptr))
    {
        return;
    };

    if(is_ascending)
    {
        shell_sort_ascending(n, a, map);
    }
    else
    {
        shell_sort_descending(n, a, map);
    };
}

template <typename S, typename I>
__device__ static void selection_sort_ascending(const I n, S* D, I* map = nullptr)
{
    // ---------------------------------------------------
    // Sort entries in D[0...(n-1)]
    // generates permutation vector in map[] if map[] is available
    // Note: performs in a single thread block
    // ---------------------------------------------------
    if((n <= 0) || (D == nullptr))
    {
        return;
    };

    auto const tid = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x
        + hipThreadIdx_z * (hipBlockDim_x * hipBlockDim_y);

    auto const nthreads = (hipBlockDim_x * hipBlockDim_y) * hipBlockDim_z;

    auto const k_start = tid;
    auto const k_inc = nthreads;
    bool const is_root_thread = (tid == 0);

    bool const has_map = (map != nullptr);

    __syncthreads();

    if(has_map)
    {
        for(auto k = k_start; k < n; k += k_inc)
        {
            map[k] = k;
        };
    };

    __syncthreads();

    if(is_root_thread)
    {
        for(I ii = 1; ii < n; ii++)
        {
            auto l = ii - 1;
            auto m = l;
            auto p = D[l];
            auto ip = (has_map) ? map[l] : 0;

            for(auto j = ii; j < n; j++)
            {
                if(D[j] < p)
                {
                    m = j;
                    p = D[j];
                    if(has_map)
                    {
                        ip = map[j];
                    };
                }
            }
            if(m != l)
            {
                D[m] = D[l];
                D[l] = p;

                if(has_map)
                {
                    map[m] = map[l];
                    map[l] = ip;
                };
            }
        }
    }
    __syncthreads();
#ifdef NDEBUG
#else
    if(n >= 2)
    {
        for(auto k = k_start; k < (n - 1); k += k_inc)
        {
            assert(std::isnan(D[k]) || std::isnan(D[k + 1]) || D[k] <= D[k + 1]);
        };
    };
    __syncthreads();
#endif
}

template <typename S, typename I>
__device__ static void selection_sort_descending(const I n, S* D, I* map = nullptr)
{
    // ---------------------------------------------------
    // Sort entries in D[0...(n-1)]
    // generates permutation vector in map[] if map[] is available
    // Note: performs in a single thread block
    // ---------------------------------------------------
    if((n <= 0) || (D == nullptr))
    {
        return;
    };

    auto const tid = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x
        + hipThreadIdx_z * (hipBlockDim_x * hipBlockDim_y);

    auto const nthreads = (hipBlockDim_x * hipBlockDim_y) * hipBlockDim_z;

    auto const k_start = tid;
    auto const k_inc = nthreads;
    bool const is_root_thread = (tid == 0);

    bool const has_map = (map != nullptr);

    __syncthreads();

    if(has_map)
    {
        for(auto k = k_start; k < n; k += k_inc)
        {
            map[k] = k;
        };
    };

    __syncthreads();

    if(is_root_thread)
    {
        for(I ii = 1; ii < n; ii++)
        {
            auto l = ii - 1;
            auto m = l;
            auto p = D[l];
            auto ip = (has_map) ? map[l] : 0;

            for(auto j = ii; j < n; j++)
            {
                if(D[j] > p)
                {
                    m = j;
                    p = D[j];
                    if(has_map)
                    {
                        ip = map[j];
                    };
                }
            }
            if(m != l)
            {
                D[m] = D[l];
                D[l] = p;

                if(has_map)
                {
                    map[m] = map[l];
                    map[l] = ip;
                };
            }
        }
    }
    __syncthreads();
#ifdef NDEBUG
#else
    if(n >= 2)
    {
        for(auto k = k_start; k < (n - 1); k += k_inc)
        {
            assert(std::isnan(D[k]) || std::isnan(D[k + 1]) || D[k] >= D[k + 1]);
        };
    };
    __syncthreads();
#endif
}

template <typename S, typename I>
__device__ static void selection_sort(const I n, S* a, I* map = nullptr, const bool is_ascending = true)
{
    if((n <= 0) || (a == nullptr))
    {
        return;
    };

    if(is_ascending)
    {
        selection_sort_ascending(n, a, map);
    }
    else
    {
        selection_sort_descending(n, a, map);
    };
}

template <typename T, typename I>
__device__ static void permute_swap(const I n, T* C, I ldc, I* map, const I nev = -1)
{
    // --------------------------------------------
    // perform swaps to implement permutation vector
    // the permutation vector will be restored to the
    // identity permutation 0,1,2,...
    //
    // Note: performs in a thread block
    // --------------------------------------------
    {
        bool const has_work = (n >= 1) && (C != nullptr) && (ldc >= 1) && (map != nullptr);
        if(!has_work)
        {
            return;
        }
    }

    auto const tid = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x
        + hipThreadIdx_z * (hipBlockDim_x * hipBlockDim_y);

    auto const nthreads = (hipBlockDim_x * hipBlockDim_y) * hipBlockDim_z;

    auto const k_start = tid;
    auto const k_inc = nthreads;

    bool const is_root_thread = (tid == 0);

    auto const nn = (nev >= 0) ? nev : n;

    for(I i = 0; i < nn; i++)
    {
        __syncthreads();

        while(map[i] != i)
        {
            auto const map_i = map[i];
            auto const map_ii = map[map[i]];

            __syncthreads();

            if(is_root_thread)
            {
                map[map_i] = map_i;
                map[i] = map_ii;
            }

            __syncthreads();

            // ---------------------------------------
            // swap columns C( 0:(n-1),map_i) and C( 0:(n-1),map_ii)
            // ---------------------------------------

            for(auto k = k_start; k < n; k += k_inc)
            {
                auto const k_map_i = k + (map_i * static_cast<int64_t>(ldc));
                auto const k_map_ii = k + (map_ii * static_cast<int64_t>(ldc));

                auto const ctemp = C[k_map_i];
                C[k_map_i] = C[k_map_ii];
                C[k_map_ii] = ctemp;
            }

            __syncthreads();
        }
    }
#ifdef NDEBUG
#else
    // ----------------------------------------------------------
    // extra check that map[] is restored to identity permutation
    // ----------------------------------------------------------
    __syncthreads();
    for(auto k = k_start; k < nn; k += k_inc)
    {
        assert(std::isnan(map[k]) || map[k] == k);
    }
    __syncthreads();
#endif
}

// **********************************************************
// GPU kernels that are used by many rocsolver functions
// **********************************************************

enum copymat_direction
{
    copymat_to_buffer,
    copymat_from_buffer
};

/** A mask that is always true. Typically used to make the mask optional, by acting as the default when
    no other mask is provided. **/
struct no_mask
{
    __device__ constexpr bool operator[](rocblas_int) const noexcept
    {
        return true;
    }
};

/** An mask defined by an integer array (e.g., the info array). By default, a non-zero value for the ith integer
    indicates that the data for batch i should be copied over. Data will not be copied if the mask value is zero.
    This behaviour is reversed if the negate transform is passed to the constructor. **/
struct info_mask
{
    enum mask_transform
    {
        none,
        negate
    };

    explicit constexpr info_mask(rocblas_int* mask, mask_transform transform = none) noexcept
        : m_mask(mask)
        , m_negate(transform == negate)
    {
    }

    __device__ constexpr bool operator[](rocblas_int idx) const noexcept
    {
        return m_negate ^ !!m_mask[idx];
    }

    rocblas_int* m_mask;
    bool m_negate;
};

/** COPY_MAT copies m-by-n array A into buffer if copymat_to_buffer, or buffer into A if copymat_from_buffer
    An optional mask can be provided to limit the copy to selected matricies in the batch
    If uplo = rocblas_fill_upper, only the upper triangular part is copied
    If uplo = rocblas_fill_lower, only the lower triangular part is copied **/
template <typename T, typename U, typename Mask = no_mask>
ROCSOLVER_KERNEL void copy_mat(copymat_direction direction,
                               const rocblas_int m,
                               const rocblas_int n,
                               U A,
                               const rocblas_int shiftA,
                               const rocblas_int lda,
                               const rocblas_stride strideA,
                               T* buffer,
                               const Mask mask = no_mask{},
                               const rocblas_fill uplo = rocblas_fill_full,
                               const rocblas_diagonal diag = rocblas_diagonal_non_unit)
{
    const auto b = hipBlockIdx_z;
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    const rocblas_int ldb = m;
    const rocblas_stride strideB = rocblas_stride(ldb) * n;

    const bool copy = mask[b];

    const bool full = (uplo == rocblas_fill_full);
    const bool upper = (uplo == rocblas_fill_upper);
    const bool lower = (uplo == rocblas_fill_lower);
    const bool cdiag = (diag == rocblas_diagonal_non_unit);

    if(i < m && j < n && copy)
    {
        if(full || (upper && j > i) || (lower && i > j) || (cdiag && i == j))
        {
            T* Ap = load_ptr_batch<T>(A, b, shiftA, strideA);
            T* Bp = &buffer[b * strideB];

            if(direction == copymat_to_buffer)
                Bp[i + j * ldb] = Ap[i + j * lda];
            else // direction == copymat_from_buffer
                Ap[i + j * lda] = Bp[i + j * ldb];
        }
    }
}

/** COPY_MAT copies m-by-n array A into B
    An optional mask can be provided to limit the copy to selected matricies in the batch
    If uplo = rocblas_fill_upper, only the upper triangular part is copied
    If uplo = rocblas_fill_lower, only the lower triangular part is copied **/
template <typename T, typename U1, typename U2, typename Mask = no_mask>
ROCSOLVER_KERNEL void copy_mat(const rocblas_int m,
                               const rocblas_int n,
                               U1 A,
                               const rocblas_int shiftA,
                               const rocblas_int lda,
                               const rocblas_stride strideA,
                               U2 B,
                               const rocblas_int shiftB,
                               const rocblas_int ldb,
                               const rocblas_stride strideB,
                               const Mask mask = no_mask{},
                               const rocblas_fill uplo = rocblas_fill_full,
                               const rocblas_diagonal diag = rocblas_diagonal_non_unit)
{
    const auto b = hipBlockIdx_z;
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    const bool copy = mask[b];

    const bool full = (uplo == rocblas_fill_full);
    const bool upper = (uplo == rocblas_fill_upper);
    const bool lower = (uplo == rocblas_fill_lower);
    const bool cdiag = (diag == rocblas_diagonal_non_unit);

    if(i < m && j < n && copy)
    {
        if(full || (upper && j > i) || (lower && i > j) || (cdiag && i == j))
        {
            T* Ap = load_ptr_batch<T>(A, b, shiftA, strideA);
            T* Bp = load_ptr_batch<T>(B, b, shiftB, strideB);

            Bp[i + j * ldb] = Ap[i + j * lda];
        }
    }
}

/** COPY_MAT copies m-by-n array A into buffer if copymat_to_buffer, or buffer into A if copymat_from_buffer
    If uplo = rocblas_fill_upper, only the upper triangular part is copied
    If uplo = rocblas_fill_lower, only the lower triangular part is copied
    Only valid when A is complex and buffer real. If REAL, only works with real part of A;
    if !REAL only works with imaginary part of A**/
template <typename T, typename S, bool REAL, typename U, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
ROCSOLVER_KERNEL void copy_mat(copymat_direction direction,
                               const rocblas_int m,
                               const rocblas_int n,
                               U A,
                               const rocblas_int shiftA,
                               const rocblas_int lda,
                               const rocblas_stride strideA,
                               S* buffer,
                               const rocblas_fill uplo = rocblas_fill_full,
                               const rocblas_diagonal diag = rocblas_diagonal_non_unit)
{
    const auto b = hipBlockIdx_z;
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    const rocblas_int ldb = m;
    const rocblas_stride strideB = rocblas_stride(ldb) * n;

    const bool lower = (uplo == rocblas_fill_lower);
    const bool full = (uplo == rocblas_fill_full);
    const bool upper = (uplo == rocblas_fill_upper);
    const bool cdiag = (diag == rocblas_diagonal_non_unit);

    if(i < m && j < n)
    {
        if(full || (upper && j > i) || (lower && i > j) || (cdiag && i == j))
        {
            T* Ap = load_ptr_batch<T>(A, b, shiftA, strideA);
            S* Bp = &buffer[b * strideB];

            if(direction == copymat_to_buffer)
                Bp[i + j * ldb] = REAL ? Ap[i + j * lda].real() : Ap[i + j * lda].imag();
            else if(REAL)
                Ap[i + j * lda] = rocblas_complex_num<S>(Bp[i + j * ldb], Ap[i + j * lda].imag());
            else
                Ap[i + j * lda] = rocblas_complex_num<S>(Ap[i + j * lda].real(), Bp[i + j * ldb]);
        }
    }
}

/** COPY_TRANS_MAT copies m-by-n array A into B and transpose it depending on the value of trans.
    An optional mask can be provided to limit the copy to selected matricies in the batch
    If uplo = rocblas_fill_upper, only the upper triangular part is copied
    If uplo = rocblas_fill_lower, only the lower triangular part is copied **/
template <typename T1, typename T2, typename U1, typename U2, typename Mask = no_mask>
ROCSOLVER_KERNEL void copy_trans_mat(const rocblas_operation trans,
                                     const rocblas_int m,
                                     const rocblas_int n,
                                     U1 A,
                                     const rocblas_int shiftA,
                                     const rocblas_int lda,
                                     const rocblas_stride strideA,
                                     U2 B,
                                     const rocblas_int shiftB,
                                     const rocblas_int ldb,
                                     const rocblas_stride strideB,
                                     const Mask mask = no_mask{},
                                     const rocblas_fill uplo = rocblas_fill_full,
                                     const rocblas_diagonal diag = rocblas_diagonal_non_unit)
{
    const auto b = hipBlockIdx_z;
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    const bool copy = mask[b];

    const bool full = (uplo == rocblas_fill_full);
    const bool upper = (uplo == rocblas_fill_upper);
    const bool lower = (uplo == rocblas_fill_lower);
    const bool cdiag = (diag == rocblas_diagonal_non_unit);

    if(i < m && j < n && copy)
    {
        if(full || (upper && j > i) || (lower && i > j) || (cdiag && i == j))
        {
            T1* Ap = load_ptr_batch<T1>(A, b, shiftA, strideA);
            T2* Bp = load_ptr_batch<T2>(B, b, shiftB, strideB);

            if(trans == rocblas_operation_conjugate_transpose)
                Bp[j + i * ldb] = T2(conj(Ap[i + j * lda]));
            else if(trans == rocblas_operation_transpose)
                Bp[j + i * ldb] = T2(Ap[i + j * lda]);
            else
                Bp[i + j * ldb] = T2(Ap[i + j * lda]);
        }
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void init_ident(const rocblas_int m,
                                 const rocblas_int n,
                                 U A,
                                 const rocblas_int shiftA,
                                 const rocblas_int lda,
                                 const rocblas_stride strideA)
{
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    const auto b = hipBlockIdx_z;

    if(i < m && j < n)
    {
        T* a = load_ptr_batch<T>(A, b, shiftA, strideA);

        if(i == j)
            a[i + j * lda] = 1.0;
        else
            a[i + j * lda] = 0.0;
    }
}

template <typename T, typename I, typename U>
ROCSOLVER_KERNEL void reset_info(T* info, const I n, U val, I incr = 0)
{
    I idx = hipBlockIdx_x * static_cast<I>(hipBlockDim_x) + hipThreadIdx_x;

    if(idx < n)
        info[idx] = T(val) + incr * idx;
}

template <typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void reset_batch_info(U info, const rocblas_stride stride, const I n, S val)
{
    I idx = hipBlockIdx_x * static_cast<I>(hipBlockDim_x) + hipThreadIdx_x;
    I b = hipBlockIdx_y;

    T* inf = load_ptr_batch<T>(info, b, 0, stride);
    if(idx < n)
        inf[idx] = T(val);
}

template <typename T, typename I>
ROCSOLVER_KERNEL void get_array(T** out, T* in, rocblas_stride stride, I batch)
{
    I b = hipBlockIdx_x * static_cast<I>(hipBlockDim_x) + hipThreadIdx_x;

    if(b < batch)
        out[b] = in + b * stride;
}

template <typename T, typename I, typename U>
ROCSOLVER_KERNEL void shift_array(T** out, U in, rocblas_stride shift, I batch)
{
    I b = hipBlockIdx_x * static_cast<I>(hipBlockDim_x) + hipThreadIdx_x;

    if(b < batch)
        out[b] = in[b] + shift;
}

template <typename T, typename U>
ROCSOLVER_KERNEL void subtract_tau(const rocblas_int i,
                                   const rocblas_int j,
                                   U A,
                                   const rocblas_int shiftA,
                                   const rocblas_int lda,
                                   const rocblas_stride strideA,
                                   T* ipiv,
                                   const rocblas_stride strideP)
{
    const auto b = hipBlockIdx_x;
    T* Ap = load_ptr_batch<T>(A, b, shiftA, strideA);
    T* tau = ipiv + b * strideP;

    T t = -(*tau);
    *tau = t;
    Ap[i + j * lda] = 1.0 + t;
}

template <typename T>
ROCSOLVER_KERNEL void restau(const rocblas_int k, T* ipiv, const rocblas_stride strideP)
{
    const auto blocksizex = hipBlockDim_x;
    const auto b = hipBlockIdx_y;
    T* tau = ipiv + b * strideP;
    const auto i = hipBlockIdx_x * blocksizex + hipThreadIdx_x;

    if(i < k)
        tau[i] = -tau[i];
}

template <typename T,
          typename I,
          typename S,
          typename U,
          std::enable_if_t<!rocblas_is_complex<T> || rocblas_is_complex<S>, int> = 0>
ROCSOLVER_KERNEL void set_diag(S* D,
                               const rocblas_stride shiftd,
                               const rocblas_stride strided,
                               U A,
                               const rocblas_stride shifta,
                               const I lda,
                               const rocblas_stride stridea,
                               const I n,
                               bool set_one)
{
    I b = hipBlockIdx_x;
    I i = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    I j = i + i * lda;

    S* d = load_ptr_batch<S>(D, b, shiftd, strided);
    T* a = load_ptr_batch<T>(A, b, shifta, stridea);

    if(i < n)
    {
        d[i] = a[j];
        a[j] = set_one ? T(1) : a[j];
    }
}

template <typename T,
          typename I,
          typename S,
          typename U,
          std::enable_if_t<rocblas_is_complex<T> && !rocblas_is_complex<S>, int> = 0>
ROCSOLVER_KERNEL void set_diag(S* D,
                               const rocblas_stride shiftd,
                               const rocblas_stride strided,
                               U A,
                               const rocblas_stride shifta,
                               const I lda,
                               const rocblas_stride stridea,
                               const I n,
                               bool set_one)
{
    I b = hipBlockIdx_x;
    I i = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    I j = i + i * lda;

    S* d = load_ptr_batch<S>(D, b, shiftd, strided);
    T* a = load_ptr_batch<T>(A, b, shifta, stridea);

    if(i < n)
    {
        d[i] = a[j].real();
        a[j] = set_one ? T(1) : a[j];
    }
}

template <typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void restore_diag(S* D,
                                   const rocblas_stride shiftd,
                                   const rocblas_stride strided,
                                   U A,
                                   const rocblas_stride shifta,
                                   const I lda,
                                   const rocblas_stride stridea,
                                   const I n)
{
    I b = hipBlockIdx_x;
    I i = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    I j = i + i * lda;

    S* d = load_ptr_batch<S>(D, b, shiftd, strided);
    T* a = load_ptr_batch<T>(A, b, shifta, stridea);

    if(i < n)
        a[j] = d[i];
}

/** SET_ZERO inserts zeros in all the entries of a m-by-n matrix A.
    If uplo = lower, the lower triangular part of A is kept unchanged.
    If uplo = upper, the upper triangular part of A is kept unchanged **/
template <typename T, typename U>
ROCSOLVER_KERNEL void set_zero(const rocblas_int m,
                               const rocblas_int n,
                               U A,
                               const rocblas_int shiftA,
                               const rocblas_int lda,
                               const rocblas_stride strideA,
                               const rocblas_fill uplo = rocblas_fill_full)
{
    const auto b = hipBlockIdx_z;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    const bool lower = (uplo == rocblas_fill_lower);
    const bool full = (uplo == rocblas_fill_full);
    const bool upper = (uplo == rocblas_fill_upper);

    if(i < m && j < n)
    {
        if(full || (lower && j > i) || (upper && i > j))
        {
            T* Ap = load_ptr_batch<T>(A, b, shiftA, strideA);
            Ap[i + j * lda] = 0.0;
        }
    }
}

/** Eigensolver scalar case (n=1) **/
template <typename T, typename U, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
ROCSOLVER_KERNEL void syev_scalar_case(const rocblas_evect evect,
                                       U AA,
                                       const rocblas_stride strideA,
                                       T* DD,
                                       const rocblas_stride strideD,
                                       rocblas_int bc)
{
    int b = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(b < bc)
    {
        T* A = load_ptr_batch<T>(AA, b, 0, strideA);
        T* D = DD + b * strideD;
        D[0] = std::real(A[0]);

        if(evect == rocblas_evect_original)
            A[0] = T(1);
    }
}

/** Eigensolver scalar case (n=1) **/
template <typename T, typename S, typename U, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
ROCSOLVER_KERNEL void syev_scalar_case(const rocblas_evect evect,
                                       U AA,
                                       const rocblas_stride strideA,
                                       S* DD,
                                       const rocblas_stride strideD,
                                       rocblas_int bc)
{
    int b = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(b < bc)
    {
        T* A = load_ptr_batch<T>(AA, b, 0, strideA);
        S* D = DD + b * strideD;
        D[0] = A[0].real();

        if(evect == rocblas_evect_original)
            A[0] = T(1);
    }
}

template <typename T>
ROCSOLVER_KERNEL void sygv_update_info(T* info, T* iinfo, const rocblas_int n, const rocblas_int bc)
{
    int b = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(b < bc)
    {
        if(info[b] != 0)
            info[b] += n;
        else
            info[b] = iinfo[b];
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void copyshift_right(const bool copy,
                                      const rocblas_int dim,
                                      U A,
                                      const rocblas_int shiftA,
                                      const rocblas_int lda,
                                      const rocblas_stride strideA,
                                      T* W,
                                      const rocblas_int shiftW,
                                      const rocblas_int ldw,
                                      const rocblas_stride strideW)
{
    const auto b = hipBlockIdx_z;
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    T* Ap = load_ptr_batch<T>(A, b, shiftA, strideA);
    T* Wp = load_ptr_batch<T>(W, b, shiftW, strideW);

    // make first row the identity
    if(i == 0 && j == 0 && !copy)
        Ap[0] = 1.0;

    if(i < dim && j < dim && j <= i)
    {
        rocblas_int offset = j * (j + 1) / 2; // to acommodate in smaller array W

        if(copy)
        {
            // copy columns
            Wp[i + j * ldw - offset] = (j == 0 ? 0.0 : Ap[i + 1 + (j - 1) * lda]);
        }
        else
        {
            // shift columns to the right
            Ap[i + 1 + j * lda] = Wp[i + j * ldw - offset];

            // make first row the identity
            if(i == j)
                Ap[(j + 1) * lda] = 0.0;
        }
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void copyshift_left(const bool copy,
                                     const rocblas_int dim,
                                     U A,
                                     const rocblas_int shiftA,
                                     const rocblas_int lda,
                                     const rocblas_stride strideA,
                                     T* W,
                                     const rocblas_int shiftW,
                                     const rocblas_int ldw,
                                     const rocblas_stride strideW)
{
    const auto b = hipBlockIdx_z;
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    T* Ap = load_ptr_batch<T>(A, b, shiftA, strideA);
    T* Wp = load_ptr_batch<T>(W, b, shiftW, strideW);

    // make last row the identity
    if(i == 0 && j == 0 && !copy)
        Ap[dim + dim * lda] = 1.0;

    if(i < dim && j < dim && i <= j)
    {
        rocblas_int offset = j * ldw - j * (j + 1) / 2; // to acommodate in smaller array W

        if(copy)
        {
            // copy columns
            Wp[i + j * ldw - offset] = (j == dim - 1 ? 0.0 : Ap[i + (j + 2) * lda]);
        }
        else
        {
            // shift columns to the left
            Ap[i + (j + 1) * lda] = Wp[i + j * ldw - offset];

            // make last row the identity
            if(i == j)
                Ap[dim + j * lda] = 0.0;
        }
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void copyshift_down(const bool copy,
                                     const rocblas_int dim,
                                     U A,
                                     const rocblas_int shiftA,
                                     const rocblas_int lda,
                                     const rocblas_stride strideA,
                                     T* W,
                                     const rocblas_int shiftW,
                                     const rocblas_int ldw,
                                     const rocblas_stride strideW)
{
    const auto b = hipBlockIdx_z;
    const auto j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    const auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    T* Ap = load_ptr_batch<T>(A, b, shiftA, strideA);
    T* Wp = load_ptr_batch<T>(W, b, shiftW, strideW);

    // make first column the identity
    if(i == 0 && j == 0 && !copy)
        Ap[0] = 1.0;

    if(i < dim && j < dim && i <= j)
    {
        rocblas_int offset = j * ldw - j * (j + 1) / 2; // to acommodate in smaller array W

        if(copy)
        {
            // copy rows
            Wp[i + j * ldw - offset] = (i == 0 ? 0.0 : Ap[i - 1 + (j + 1) * lda]);
        }
        else
        {
            // shift rows downward
            Ap[i + (j + 1) * lda] = Wp[i + j * ldw - offset];

            // make first column the identity
            if(i == j)
                Ap[i + 1] = 0.0;
        }
    }
}

/** set_offdiag kernel copies the off-diagonal element of A, which is the non-zero element
    resulting by applying the Householder reflector to the working column, to E. Then set it
    to 1 to prepare for the application of the Householder reflector to the rest of the matrix **/
template <typename T, typename S, typename U, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
ROCSOLVER_KERNEL void set_offdiag(const rocblas_int batch_count,
                                  U A,
                                  const rocblas_int shiftA,
                                  const rocblas_stride strideA,
                                  S* E,
                                  const rocblas_stride strideE)
{
    rocblas_int b = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(b < batch_count)
    {
        T* a = load_ptr_batch<T>(A, b, shiftA, strideA);
        S* e = E + b * strideE;

        e[0] = a[0];
        a[0] = T(1);
    }
}

template <typename T, typename S, typename U, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
ROCSOLVER_KERNEL void set_offdiag(const rocblas_int batch_count,
                                  U A,
                                  const rocblas_int shiftA,
                                  const rocblas_stride strideA,
                                  S* E,
                                  const rocblas_stride strideE)
{
    rocblas_int b = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(b < batch_count)
    {
        T* a = load_ptr_batch<T>(A, b, shiftA, strideA);
        S* e = E + b * strideE;

        e[0] = a[0].real();
        a[0] = T(1);
    }
}

/** scale_axpy kernel executes axpy to update tau computing the scalar alpha with other
    results in different memopry locations **/
template <typename T, typename U>
ROCSOLVER_KERNEL void scale_axpy(const rocblas_int n,
                                 T* scl,
                                 T* S,
                                 const rocblas_stride strideS,
                                 U A,
                                 const rocblas_int shiftA,
                                 const rocblas_stride strideA,
                                 T* W,
                                 const rocblas_int shiftW,
                                 const rocblas_stride strideW)
{
    rocblas_int b = hipBlockIdx_y;
    rocblas_int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(i < n)
    {
        T* v = load_ptr_batch<T>(A, b, shiftA, strideA);
        T* w = load_ptr_batch<T>(W, b, shiftW, strideW);
        T* s = S + b * strideS;

        // scale
        T alpha = -0.5 * s[0] * scl[b];

        // axpy
        w[i] = alpha * v[i] + w[i];
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void check_singularity(const rocblas_int n,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        rocblas_int* info)
{
    // Checks for singularities in the matrix and updates info to indicate where
    // the first singularity (if any) occurs
    int b = hipBlockIdx_x;

    T* a = load_ptr_batch<T>(A, b, shiftA, strideA);

    __shared__ rocblas_int _info;

    if(hipThreadIdx_y == 0)
        _info = 0;
    __syncthreads();

    for(int i = hipThreadIdx_y; i < n; i += hipBlockDim_y)
    {
        if(a[i + i * lda] == 0)
        {
            rocblas_int _info_temp = _info;
            while(_info_temp == 0 || _info_temp > i + 1)
                _info_temp = atomicCAS(&_info, _info_temp, i + 1);
        }
    }
    __syncthreads();

    if(hipThreadIdx_y == 0)
        info[b] = _info;
}

/** SWAP swaps the values of vectors x and y of dimension n.
    Launch this kernel with a desired number of threads organized in
    NG groups in the x direction with NT threads in the x direction. **/
template <typename T, typename I>
ROCSOLVER_KERNEL void swap_kernel(I const n, T* const x, I const incx, T* const y, I const incy)
{
    if(n <= 0)
        return;

    I const i_start = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    I const i_inc = hipBlockDim_x * hipGridDim_x;

    if((incx == 1) && (incy == 1))
    {
        // ------------
        // special case
        // ------------
        for(I i = i_start; i < n; i += i_inc)
        {
            auto const temp = y[i];
            y[i] = x[i];
            x[i] = temp;
        }
    }
    else
    {
        // ---------------------------
        // code for unequal increments
        // ---------------------------

        for(I i = i_start; i < n; i += i_inc)
        {
            auto const ix = 0 + i * static_cast<int64_t>(incx);
            auto const iy = 0 + i * static_cast<int64_t>(incy);

            auto const temp = y[iy];
            y[iy] = x[ix];
            x[ix] = temp;
        }
    }
}

ROCSOLVER_END_NAMESPACE

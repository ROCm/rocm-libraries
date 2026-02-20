/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../blas2/rocblas_gemv.hpp"
#include "../blas3/trtri_trsm.hpp"
#include "check_numerics_vector.hpp"
#include "device_macros.hpp"

template <typename T>
static const T alpha_negative_one = T(-1);

template <typename T>
static const T beta_0 = T(0);

template <typename T>
static const T alpha_1 = T(1);
template <typename T>
static const T beta_1 = T(1);

template <rocblas_int NB, typename T, typename U>
ROCBLAS_KERNEL(NB)
rocblas_internal_flip_vector_kernel(U* __restrict__ data,
                                    rocblas_int    m,
                                    rocblas_int    size,
                                    rocblas_int    abs_incx,
                                    rocblas_stride offset,
                                    rocblas_stride stride,
                                    rocblas_int    batch_count)
{
    rocblas_int tx = blockIdx.x * blockDim.x + threadIdx.x;
    if(tx < size)
    {
        // Load appropriate pointers
        uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
        for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
        {
#endif
            T*          pdata = load_ptr_batch(data, batch, offset, stride);
            rocblas_int end   = (m - 1 - tx) * abs_incx;
            rocblas_int start = tx * abs_incx;
            T           temp  = pdata[end];
            pdata[end]        = pdata[start];
            pdata[start]      = temp;
#if DEVICE_GRID_YZ_16BIT
        }
#endif
    }
}

template <rocblas_int NB_X, typename T, typename U>
rocblas_status rocblas_internal_flip_vector(rocblas_handle handle,
                                            U*             data,
                                            rocblas_int    m,
                                            rocblas_int    abs_incx,
                                            rocblas_stride stride,
                                            rocblas_int    batch_count,
                                            rocblas_stride offset = 0)
{
    int         batches = handle->getBatchGridDim((int)batch_count);
    rocblas_int size    = (m + 1) / 2;
    rocblas_int blocksX = (size - 1) / NB_X + 1;
    dim3        grid(blocksX, 1, batches);
    dim3        threads(NB_X, 1, 1);

    ROCBLAS_LAUNCH_KERNEL((rocblas_internal_flip_vector_kernel<NB_X, T>),
                          grid,
                          threads,
                          0,
                          handle->get_stream(),
                          data,
                          m,
                          size,
                          abs_incx,
                          offset,
                          stride,
                          batch_count);

    return rocblas_status_success;
}

template <rocblas_int NB, typename T, typename U, typename V>
ROCBLAS_KERNEL(NB)
rocblas_internal_strided_vector_copy_kernel(U __restrict__ dst,
                                            rocblas_int    dst_incx,
                                            rocblas_stride dst_stride,
                                            V __restrict__ src,
                                            rocblas_int    src_incx,
                                            rocblas_stride src_stride,
                                            rocblas_int    size,
                                            rocblas_stride offset_dst = 0,
                                            rocblas_stride offset_src = 0)
{
    ptrdiff_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    if(tx < size)
    {
        const T* __restrict__ xsrc = load_ptr_batch(src, blockIdx.y, offset_src, src_stride);
        T* __restrict__ xdst       = load_ptr_batch(dst, blockIdx.y, offset_dst, dst_stride);
        xdst[tx * dst_incx]        = xsrc[tx * src_incx];
    }
}

template <rocblas_int NB_X, typename T, typename U, typename V>
rocblas_status rocblas_internal_strided_vector_copy(rocblas_handle handle,
                                                    U              dst,
                                                    rocblas_int    dst_incx,
                                                    rocblas_stride dst_stride,
                                                    V              src,
                                                    rocblas_int    src_incx,
                                                    rocblas_stride src_stride,
                                                    rocblas_int    size,
                                                    rocblas_int    batch_count,
                                                    rocblas_stride offset_dst = 0,
                                                    rocblas_stride offset_src = 0)
{
    rocblas_int blocksX = (size - 1) / NB_X + 1;
    dim3        grid(blocksX, batch_count, 1);
    dim3        threads(NB_X, 1, 1);

    ROCBLAS_LAUNCH_KERNEL((rocblas_internal_strided_vector_copy_kernel<NB_X, T>),
                          grid,
                          threads,
                          0,
                          handle->get_stream(),
                          dst,
                          dst_incx,
                          dst_stride,
                          src,
                          src_incx,
                          src_stride,
                          size,
                          offset_dst,
                          offset_src);

    return rocblas_status_success;
}

template <rocblas_int BLOCK, typename T, typename U, typename V>
rocblas_status rocblas_internal_trsv_left(rocblas_handle    handle,
                                          rocblas_fill      uplo,
                                          rocblas_operation transA,
                                          rocblas_int       m,
                                          U                 A,
                                          rocblas_stride    offset_Ain,
                                          rocblas_int       lda,
                                          rocblas_stride    stride_A,
                                          V                 B,
                                          rocblas_stride    offset_Bin,
                                          rocblas_int       incx,
                                          rocblas_stride    stride_B,
                                          U                 invA,
                                          rocblas_stride    offset_invAin,
                                          rocblas_stride    stride_invA,
                                          V                 X,
                                          rocblas_stride    stride_X,
                                          rocblas_int       batch_count)
{
    rocblas_int i, jb;

    if(transA == rocblas_operation_none)
    {
        if(uplo == rocblas_fill_lower)
        {
            // left, lower no-transpose
            jb = std::min(BLOCK, m);
            rocblas_internal_gemv_launcher(handle,
                                           transA,
                                           jb,
                                           jb,
                                           &alpha_1<T>,
                                           0,
                                           invA,
                                           offset_invAin,
                                           BLOCK,
                                           stride_invA,
                                           (U)B,
                                           offset_Bin,
                                           incx,
                                           stride_B,
                                           &beta_0<T>,
                                           0,
                                           X,
                                           0,
                                           1,
                                           stride_X,
                                           batch_count);

            if(BLOCK < m)
            {
                rocblas_internal_gemv_launcher(handle,
                                               transA,
                                               m - BLOCK,
                                               BLOCK,
                                               &alpha_negative_one<T>,
                                               0,
                                               A,
                                               offset_Ain + BLOCK,
                                               lda,
                                               stride_A,
                                               (U)X,
                                               0,
                                               1,
                                               stride_X,
                                               &beta_1<T>,
                                               0,
                                               B,
                                               offset_Bin + BLOCK * incx,
                                               incx,
                                               stride_B,
                                               batch_count);

                // remaining blocks
                for(i = BLOCK; i < m; i += BLOCK)
                {
                    jb = std::min(m - i, BLOCK);

                    rocblas_internal_gemv_launcher(handle,
                                                   transA,
                                                   jb,
                                                   jb,
                                                   &alpha_1<T>,
                                                   0,
                                                   invA,
                                                   offset_invAin + i * BLOCK,
                                                   BLOCK,
                                                   stride_invA,
                                                   (U)B,
                                                   offset_Bin + i * incx,
                                                   incx,
                                                   stride_B,
                                                   &beta_0<T>,
                                                   0,
                                                   X,
                                                   i,
                                                   1,
                                                   stride_X,
                                                   batch_count);
                    if(i + BLOCK < m)
                        rocblas_internal_gemv_launcher(handle,
                                                       transA,
                                                       m - i - BLOCK,
                                                       BLOCK,
                                                       &alpha_negative_one<T>,
                                                       0,
                                                       A,
                                                       offset_Ain + i + BLOCK + i * lda,
                                                       lda,
                                                       stride_A,
                                                       (U)X,
                                                       i,
                                                       1,
                                                       stride_X,
                                                       &beta_1<T>,
                                                       0,
                                                       B,
                                                       offset_Bin + (i + BLOCK) * incx,
                                                       incx,
                                                       stride_B,
                                                       batch_count);
                }
            }
        }
        else
        {
            // left, upper no-transpose
            jb = m % BLOCK == 0 ? BLOCK : m % BLOCK;
            i  = m - jb;

            // if m=n=35=lda=ldb, BLOCK =32, then jb = 3, i = 32; {3, 35, 3, 32, 35, 35}
            rocblas_internal_gemv_launcher(handle,
                                           transA,
                                           jb,
                                           jb,
                                           &alpha_1<T>,
                                           0,
                                           invA,
                                           offset_invAin + i * BLOCK,
                                           BLOCK,
                                           stride_invA,
                                           (U)B,
                                           offset_Bin + i * incx,
                                           incx,
                                           stride_B,
                                           &beta_0<T>,
                                           0,
                                           X,
                                           i,
                                           1,
                                           stride_X,
                                           batch_count);

            if(i >= BLOCK)
            {
                rocblas_internal_gemv_launcher(handle,
                                               transA,
                                               i,
                                               jb,
                                               &alpha_negative_one<T>,
                                               0,
                                               A,
                                               offset_Ain + i * lda,
                                               lda,
                                               stride_A,
                                               (U)X,
                                               i,
                                               1,
                                               stride_X,
                                               &beta_1<T>,
                                               0,
                                               B,
                                               offset_Bin,
                                               incx,
                                               stride_B,
                                               batch_count);

                // remaining blocks
                for(i = m - jb - BLOCK; i >= 0; i -= BLOCK)
                {
                    //{32, 35, 32, 32, 35, 35}
                    rocblas_internal_gemv_launcher(handle,
                                                   transA,
                                                   BLOCK,
                                                   BLOCK,
                                                   &alpha_1<T>,
                                                   0,
                                                   invA,
                                                   offset_invAin + i * BLOCK,
                                                   BLOCK,
                                                   stride_invA,
                                                   (U)B,
                                                   offset_Bin + i * incx,
                                                   incx,
                                                   stride_B,
                                                   &beta_0<T>,
                                                   0,
                                                   X,
                                                   i,
                                                   1,
                                                   stride_X,
                                                   batch_count);

                    if(i >= BLOCK)
                        rocblas_internal_gemv_launcher(handle,
                                                       transA,
                                                       i,
                                                       BLOCK,
                                                       &alpha_negative_one<T>,
                                                       0,
                                                       A,
                                                       offset_Ain + i * lda,
                                                       lda,
                                                       stride_A,
                                                       (U)X,
                                                       i,
                                                       1,
                                                       stride_X,
                                                       &beta_1<T>,
                                                       0,
                                                       B,
                                                       offset_Bin,
                                                       incx,
                                                       stride_B,
                                                       batch_count);
                }
            }
        }
    }
    else
    { // transA == rocblas_operation_transpose || transA == rocblas_operation_conjugate_transpose
        if(uplo == rocblas_fill_lower)
        {
            // left, lower transpose
            jb = m % BLOCK == 0 ? BLOCK : m % BLOCK;
            i  = m - jb;

            rocblas_internal_gemv_launcher(handle,
                                           transA,
                                           jb,
                                           jb,
                                           &alpha_1<T>,
                                           0,
                                           invA,
                                           offset_invAin + i * BLOCK,
                                           BLOCK,
                                           stride_invA,
                                           (U)B,
                                           offset_Bin + i * incx,
                                           incx,
                                           stride_B,
                                           &beta_0<T>,
                                           0,
                                           X,
                                           i,
                                           1,
                                           stride_X,
                                           batch_count);

            if(i - BLOCK >= 0)
            {
                rocblas_internal_gemv_launcher(handle,
                                               transA,
                                               jb,
                                               i,
                                               &alpha_negative_one<T>,
                                               0,
                                               A,
                                               offset_Ain + i,
                                               lda,
                                               stride_A,
                                               (U)X,
                                               i,
                                               1,
                                               stride_X,
                                               &beta_1<T>,
                                               0,
                                               B,
                                               offset_Bin,
                                               incx,
                                               stride_B,
                                               batch_count);

                // remaining blocks
                for(i = m - jb - BLOCK; i >= 0; i -= BLOCK)
                {
                    rocblas_internal_gemv_launcher(handle,
                                                   transA,
                                                   BLOCK,
                                                   BLOCK,
                                                   &alpha_1<T>,
                                                   0,
                                                   invA,
                                                   offset_invAin + i * BLOCK,
                                                   BLOCK,
                                                   stride_invA,
                                                   (U)B,
                                                   offset_Bin + i * incx,
                                                   incx,
                                                   stride_B,
                                                   &beta_0<T>,
                                                   0,
                                                   X,
                                                   i,
                                                   1,
                                                   stride_X,
                                                   batch_count);

                    if(i >= BLOCK)
                        rocblas_internal_gemv_launcher(handle,
                                                       transA,
                                                       BLOCK,
                                                       i,
                                                       &alpha_negative_one<T>,
                                                       0,
                                                       A,
                                                       offset_Ain + i,
                                                       lda,
                                                       stride_A,
                                                       (U)X,
                                                       i,
                                                       1,
                                                       stride_X,
                                                       &beta_1<T>,
                                                       0,
                                                       B,
                                                       offset_Bin,
                                                       incx,
                                                       stride_B,
                                                       batch_count);
                }
            }
        }
        else
        {
            // left, upper transpose
            jb = std::min(BLOCK, m);
            rocblas_internal_gemv_launcher(handle,
                                           transA,
                                           jb,
                                           jb,
                                           &alpha_1<T>,
                                           0,
                                           invA,
                                           offset_invAin,
                                           BLOCK,
                                           stride_invA,
                                           (U)B,
                                           offset_Bin,
                                           incx,
                                           stride_B,
                                           &beta_0<T>,
                                           0,
                                           X,
                                           0,
                                           1,
                                           stride_X,
                                           batch_count);

            if(BLOCK < m)
            {
                rocblas_internal_gemv_launcher(handle,
                                               transA,
                                               BLOCK,
                                               m - BLOCK,
                                               &alpha_negative_one<T>,
                                               0,
                                               A,
                                               offset_Ain + BLOCK * lda,
                                               lda,
                                               stride_A,
                                               (U)X,
                                               0,
                                               1,
                                               stride_X,
                                               &beta_1<T>,
                                               0,
                                               B,
                                               offset_Bin + BLOCK * incx,
                                               incx,
                                               stride_B,
                                               batch_count);

                // remaining blocks
                for(i = BLOCK; i < m; i += BLOCK)
                {
                    jb = std::min(m - i, BLOCK);
                    rocblas_internal_gemv_launcher(handle,
                                                   transA,
                                                   jb,
                                                   jb,
                                                   &alpha_1<T>,
                                                   0,
                                                   invA,
                                                   offset_invAin + i * BLOCK,
                                                   BLOCK,
                                                   stride_invA,
                                                   (U)B,
                                                   offset_Bin + i * incx,
                                                   incx,
                                                   stride_B,
                                                   &beta_0<T>,
                                                   0,
                                                   X,
                                                   i,
                                                   1,
                                                   stride_X,
                                                   batch_count);

                    if(i + BLOCK < m)
                        rocblas_internal_gemv_launcher(handle,
                                                       transA,
                                                       BLOCK,
                                                       m - i - BLOCK,
                                                       &alpha_negative_one<T>,
                                                       0,
                                                       A,
                                                       offset_Ain + i + (i + BLOCK) * lda,
                                                       lda,
                                                       stride_A,
                                                       (U)X,
                                                       i,
                                                       1,
                                                       stride_X,
                                                       &beta_1<T>,
                                                       0,
                                                       B,
                                                       offset_Bin + (i + BLOCK) * incx,
                                                       incx,
                                                       stride_B,
                                                       batch_count);
                }
            }
        }
    } // transpose

    return rocblas_status_success;
}

template <rocblas_int BLOCK, typename T, typename U, typename V>
rocblas_status rocblas_internal_special_trsv_template(rocblas_handle    handle,
                                                      rocblas_fill      uplo,
                                                      rocblas_operation transA,
                                                      rocblas_diagonal  diag,
                                                      rocblas_int       m,
                                                      U                 A,
                                                      rocblas_stride    offset_Ain,
                                                      rocblas_int       lda,
                                                      rocblas_stride    stride_A,
                                                      V                 B,
                                                      rocblas_stride    offset_Bin,
                                                      rocblas_int       incx,
                                                      rocblas_stride    stride_B,
                                                      U                 invA,
                                                      rocblas_stride    offset_invAin,
                                                      rocblas_stride    stride_invA,
                                                      V                 x_temp,
                                                      rocblas_stride    stride_X,
                                                      rocblas_int       batch_count)
{
    bool   parity = (transA == rocblas_operation_none) ^ (uplo == rocblas_fill_lower);
    size_t R      = m / BLOCK;

    for(size_t r = 0; r < R; r++)
    {
        size_t q = R - r;
        size_t j = parity ? q - 1 : r;

        // copy a BLOCK*n piece we are solving at a time
        rocblas_internal_strided_vector_copy<BLOCK, T>(handle,
                                                       x_temp,
                                                       1,
                                                       stride_X,
                                                       B,
                                                       incx,
                                                       stride_B,
                                                       BLOCK,
                                                       batch_count,
                                                       0,
                                                       offset_Bin + incx * j * BLOCK);

        if(r)
        {
            rocblas_int    M       = BLOCK;
            rocblas_int    N       = BLOCK;
            rocblas_stride offsetA = 0;
            rocblas_stride offsetB = parity ? q * BLOCK * incx : 0;

            if(transA == rocblas_operation_none)
            {
                N *= r;
                offsetA = parity ? BLOCK * ((lda + 1) * q - 1) : N;
            }
            else
            {
                M *= r;
                offsetA = parity ? BLOCK * ((lda + 1) * q - lda) : M * lda;
            }

            rocblas_internal_gemv_launcher(handle,
                                           transA,
                                           M,
                                           N,
                                           &alpha_negative_one<T>,
                                           0,
                                           A,
                                           offset_Ain + offsetA,
                                           lda,
                                           stride_A,
                                           (U)B,
                                           offset_Bin + offsetB,
                                           incx,
                                           stride_B,
                                           &beta_1<T>,
                                           0,
                                           x_temp,
                                           0,
                                           1,
                                           stride_X,
                                           batch_count);
        }

        rocblas_internal_gemv_launcher(handle,
                                       transA,
                                       BLOCK,
                                       BLOCK,
                                       &alpha_1<T>,
                                       0,
                                       invA,
                                       offset_invAin + j * BLOCK * BLOCK,
                                       BLOCK,
                                       stride_invA,
                                       (U)x_temp,
                                       0,
                                       1,
                                       stride_X,
                                       &beta_0<T>,
                                       0,
                                       B,
                                       offset_Bin + j * BLOCK * incx,
                                       incx,
                                       stride_B,
                                       batch_count);
    }

    return rocblas_status_success;
}

template <rocblas_int BLOCK, bool BATCHED, typename T, typename U, typename MEM>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trsv_inverse_template_mem(rocblas_handle handle,
                                               rocblas_int    m,
                                               rocblas_int    batch_count,
                                               MEM&           w_mem,
                                               void*&         w_mem_x_temp,
                                               void*&         w_mem_x_temp_arr,
                                               void*&         w_mem_invA,
                                               void*&         w_mem_invA_arr,
                                               const U*       supplied_invA      = nullptr,
                                               rocblas_int    supplied_invA_size = 0)
{
    // Whether size is an exact multiple of blocksize
    const bool exact_blocks = (m % BLOCK) == 0;

    // perf_status indicates whether optimal performance is obtainable with available memory
    rocblas_status perf_status = rocblas_status_success;

    size_t invA_bytes   = 0;
    size_t c_temp_bytes = 0;

    // For user-supplied invA, check to make sure size is large enough
    // If not large enough, indicate degraded performance and ignore supplied invA
    if(supplied_invA && supplied_invA_size / BLOCK < m)
    {
        supplied_invA = nullptr;
        if(!handle->is_device_memory_size_query())
        {
            static auto& once = rocblas_cerr
                                << "WARNING: TRSV invA_size argument is too small; invA argument "
                                   "is being ignored; TRSV performance is degraded"
                                << std::endl;
            perf_status = rocblas_status_perf_degraded;
        }
    }

    if(!supplied_invA)
    {
        // Only allocate bytes for invA if supplied_invA == nullptr or supplied_invA_size is too small
        invA_bytes = sizeof(T) * BLOCK * m * batch_count;

        // When m < BLOCK, C is unnecessary for trtri
        c_temp_bytes = (m / BLOCK) * (sizeof(T) * (BLOCK / 2) * (BLOCK / 2)) * batch_count;

        // For the TRTRI last diagonal block we need remainder space if m % BLOCK != 0
        if(!exact_blocks)
        {
            // TODO: Make this more accurate -- right now it's much larger than necessary
            size_t remainder_bytes = sizeof(T) * ROCBLAS_TRTRI_NB * BLOCK * 2 * batch_count;

            // C is the maximum of the temporary space needed for TRTRI
            c_temp_bytes = std::max(c_temp_bytes, remainder_bytes);
        }
    }

    // Temporary solution vector
    // If the special solver can be used, then only BLOCK words are needed instead of m words
    size_t x_temp_bytes
        = exact_blocks ? sizeof(T) * BLOCK * batch_count : sizeof(T) * m * batch_count;

    // X and C temporaries can share space, so the maximum size is allocated
    size_t x_c_temp_bytes = std::max(x_temp_bytes, c_temp_bytes);
    size_t arrBytes       = BATCHED ? sizeof(T*) * batch_count : 0;
    size_t xarrBytes      = BATCHED ? sizeof(T*) * batch_count : 0;

    // If this is a device memory size query, set optimal size and return changed status
    if(handle->is_device_memory_size_query())
        return handle->set_optimal_device_memory_size(
            x_c_temp_bytes, xarrBytes, invA_bytes, arrBytes);

    // Attempt to allocate optimal memory size, returning error if failure
    w_mem = handle->device_malloc(x_c_temp_bytes, xarrBytes, invA_bytes, arrBytes);
    if(!w_mem)
        return rocblas_status_memory_error;

    // Get pointers to allocated device memory

    w_mem_x_temp     = w_mem[0];
    w_mem_x_temp_arr = w_mem[1];
    w_mem_invA       = w_mem[2];
    w_mem_invA_arr   = w_mem[3];
    return perf_status;
}

template <rocblas_int BLOCK, bool BATCHED, typename T, typename U, typename V>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trsv_inverse_template(rocblas_handle    handle,
                                           rocblas_fill      uplo,
                                           rocblas_operation transA,
                                           rocblas_diagonal  diag,
                                           rocblas_int       m,
                                           U                 A,
                                           rocblas_stride    offset_A,
                                           rocblas_int       lda,
                                           rocblas_stride    stride_A,
                                           V                 B,
                                           rocblas_stride    offset_B,
                                           rocblas_int       incx,
                                           rocblas_stride    stride_B,
                                           rocblas_int       batch_count,
                                           void*             x_temp,
                                           void*             x_temparr,
                                           void*             invA               = nullptr,
                                           void*             invAarr            = nullptr,
                                           U                 supplied_invA      = nullptr,
                                           rocblas_int       supplied_invA_size = 0,
                                           rocblas_stride    offset_invA        = 0,
                                           rocblas_stride    stride_invA        = 0)
{
    if(batch_count == 0)
        return rocblas_status_success;

    rocblas_status status       = rocblas_status_success;
    const bool     exact_blocks = (m % BLOCK) == 0;
    size_t         x_temp_els   = exact_blocks ? BLOCK : m;

    // Temporarily switch to host pointer mode, restoring on return
    // cppcheck-suppress unreadVariable
    auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

    if(supplied_invA)
        invA = (U*)(supplied_invA);
    else
    {
        // batched trtri invert diagonal part (BLOCK*BLOCK) of A into invA
        auto c_temp = x_temp; // Uses same memory as x_temp
        stride_invA = BLOCK * m;
        if(BATCHED)
        {
            RETURN_IF_ROCBLAS_ERROR(setup_batched_array<BLOCK>(
                handle->get_stream(), (T*)c_temp, 0, (T**)x_temparr, batch_count));
            RETURN_IF_ROCBLAS_ERROR(setup_batched_array<BLOCK>(
                handle->get_stream(), (T*)invA, stride_invA, (T**)invAarr, batch_count));
        }

        status = rocblas_trtri_trsm_template<BLOCK, BATCHED, T>(handle,
                                                                (V)(BATCHED ? x_temparr : c_temp),
                                                                uplo,
                                                                diag,
                                                                m,
                                                                A,
                                                                offset_A,
                                                                lda,
                                                                stride_A,
                                                                (V)(BATCHED ? invAarr : invA),
                                                                offset_invA,
                                                                stride_invA,
                                                                batch_count);
        if(status != rocblas_status_success)
            return status;
    }

    // TODO: workaround to fix negative incx issue
    rocblas_int abs_incx = incx < 0 ? -incx : incx;
    if(incx < 0)
        rocblas_internal_flip_vector<BLOCK, T>(
            handle, B, m, abs_incx, stride_B, batch_count, offset_B);

    if(BATCHED)
    {
        RETURN_IF_ROCBLAS_ERROR(setup_batched_array<BLOCK>(
            handle->get_stream(), (T*)x_temp, x_temp_els, (T**)x_temparr, batch_count));
    }

    if(exact_blocks)
    {

        status = rocblas_internal_special_trsv_template<BLOCK, T>(handle,
                                                                  uplo,
                                                                  transA,
                                                                  diag,
                                                                  m,
                                                                  A,
                                                                  offset_A,
                                                                  lda,
                                                                  stride_A,
                                                                  B,
                                                                  offset_B,
                                                                  abs_incx,
                                                                  stride_B,
                                                                  (U)(BATCHED ? invAarr : invA),
                                                                  offset_invA,
                                                                  stride_invA,
                                                                  (V)(BATCHED ? x_temparr : x_temp),
                                                                  x_temp_els,
                                                                  batch_count);

        if(status != rocblas_status_success)
            return status;

        // TODO: workaround to fix negative incx issue
        if(incx < 0)
            rocblas_internal_flip_vector<BLOCK, T>(
                handle, B, m, abs_incx, stride_B, batch_count, offset_B);
    }
    else
    {

        status = rocblas_internal_trsv_left<BLOCK, T>(handle,
                                                      uplo,
                                                      transA,
                                                      m,
                                                      A,
                                                      offset_A,
                                                      lda,
                                                      stride_A,
                                                      B,
                                                      offset_B,
                                                      abs_incx,
                                                      stride_B,
                                                      (U)(BATCHED ? invAarr : invA),
                                                      offset_invA,
                                                      stride_invA,
                                                      (V)(BATCHED ? x_temparr : x_temp),
                                                      x_temp_els,
                                                      batch_count);

        if(status != rocblas_status_success)
            return status;

        // copy solution X into B
        // TODO: workaround to fix negative incx issue
        rocblas_internal_strided_vector_copy<BLOCK, T>(handle,
                                                       B,
                                                       abs_incx,
                                                       stride_B,
                                                       (V)(BATCHED ? x_temparr : x_temp),
                                                       incx < 0 ? -1 : 1,
                                                       x_temp_els,
                                                       m,
                                                       batch_count,
                                                       offset_B,
                                                       incx < 0 ? m - 1 : 0);
    }

    return status;
}

//TODO :-Add rocblas_check_numerics_tr_matrix_template for checking Matrix `A` which is a Triangular Matrix
template <typename T, typename U>
rocblas_status rocblas_internal_trsv_ex_check_numerics(const char*    function_name,
                                                       rocblas_handle handle,
                                                       rocblas_int    m,
                                                       T              A,
                                                       rocblas_stride offset_a,
                                                       rocblas_int    lda,
                                                       rocblas_stride stride_a,
                                                       U              x,
                                                       rocblas_stride offset_x,
                                                       rocblas_int    inc_x,
                                                       rocblas_stride stride_x,
                                                       rocblas_int    batch_count,
                                                       const int      check_numerics,
                                                       bool           is_input)
{
    rocblas_status check_numerics_status
        = rocblas_internal_check_numerics_vector_template(function_name,
                                                          handle,
                                                          m,
                                                          x,
                                                          offset_x,
                                                          inc_x,
                                                          stride_x,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);

    return check_numerics_status;
}

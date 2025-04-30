/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#define SSYMM_MIN_NB 32
#define DSYMM_MIN_NB 32
#define CSYMM_MIN_NB 32
#define ZSYMM_MIN_NB 32

#define SSYMM_BATCHED_MIN_NB 32
#define DSYMM_BATCHED_MIN_NB 32
#define CSYMM_BATCHED_MIN_NB 32
#define ZSYMM_BATCHED_MIN_NB 32

#include "definitions.hpp"
#include "device_macros.hpp"
#include "handle.hpp"
#include "rocblas_gemm.hpp"
#include "rocblas_symm_hemm.hpp"
#include "src64/blas3/rocblas_gemm_64.hpp"
#include <type_traits>

template <typename T>
static const T beta_1 = T(1);

template <int DIM_X, int DIM_Y, typename T>
ROCBLAS_KERNEL_ILF void
    rocblas_symm_scale_device(rocblas_int m, rocblas_int n, T beta, T* C, int64_t ldc)
{
    auto tx = blockIdx.x * DIM_X + threadIdx.x;

    for(ptrdiff_t ty = blockIdx.y * DIM_Y + threadIdx.y; tx < m && ty < n; ty += DIM_Y * gridDim.y)
    {
        C[ty * ldc + tx] = beta ? beta * C[ty * ldc + tx] : 0;
    }
}

/**
  *  Loads pointers and launches the actual calculation kernel.
  */
template <int DIM_X, int DIM_Y, typename T, typename U>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_symm_scale_kernel(rocblas_int    m,
                          rocblas_int    n,
                          T              beta_host_device,
                          U              CP_array,
                          rocblas_stride shift_c,
                          int64_t        ldc,
                          rocblas_stride stride_c,
                          rocblas_int    batch_count)
{
    auto beta = load_scalar(beta_host_device);
    if(beta == 1)
        return;

    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif
        auto C = load_ptr_batch(CP_array, batch, shift_c, stride_c);
        rocblas_symm_scale_device<DIM_X, DIM_Y>(m, n, beta, C, ldc);
#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

/**
  * kernel
  */
template <bool HERM, bool RIGHT, rocblas_int TILE_NK, typename T>
ROCBLAS_KERNEL_ILF void rocblas_symm_hemm_mult_add_device(bool        is_upper,
                                                          rocblas_int m,
                                                          rocblas_int n,
                                                          T           alpha,
                                                          const T* __restrict__ A,
                                                          int64_t lda,
                                                          const T* __restrict__ B,
                                                          int64_t ldb,
                                                          T* __restrict__ C,
                                                          int64_t ldc)
{
    // function not called when !alpha

    __shared__ T atile[TILE_NK][TILE_NK];
    __shared__ T btile[TILE_NK][TILE_NK];

    for(int blockIdxY = blockIdx.y; blockIdxY < (n - 1) / TILE_NK + 1; blockIdxY += gridDim.y)
    {
        int col_pos = blockIdxY * TILE_NK;
        int row_pos = blockIdx.x * TILE_NK;

        int row = row_pos + threadIdx.x;
        int col = col_pos + threadIdx.y; // blockIdx.y * TILE_NK + threadIdx.y

        int from, to;

        int k_end = !RIGHT ? m : n;
        for(int k_pos = 0; k_pos < k_end; k_pos += TILE_NK)
        {
            // tiling over dimension K

            int row_loc, col_loc;
            int r, c;

            // when HERM ^H instead of ^T fetch A

            if(!RIGHT)
            {
                // premultiply: alpha*A*B

                // fetch tile of symm matrix A
                row_loc = row_pos + threadIdx.x;
                col_loc = k_pos + threadIdx.y;

                from = is_upper ? row_loc : col_loc;
                to   = is_upper ? col_loc : row_loc;

                r = from > to ? col_loc : row_loc;
                c = from > to ? row_loc : col_loc;

                if(!HERM)
                {
                    atile[threadIdx.x][threadIdx.y] = (r < m && c < m) ? A[c * lda + r] : 0;
                }
                else
                {
                    // clang-format off
                    T e = (r < m && c < m)
                              ? (from > to ? conj(A[c * lda + r])
                                           : (from == to ? std::real(A[c * lda + r]) : A[c * lda + r]))
                              : 0;
                    // clang-format on
                    atile[threadIdx.x][threadIdx.y] = e;
                }

                // fetch tile of matrix B
                row_loc = k_pos + threadIdx.x;
                col_loc = col_pos + threadIdx.y;
                r       = row_loc;
                c       = col_loc;

                btile[threadIdx.x][threadIdx.y] = (r < m && c < n) ? B[c * ldb + r] : 0;

                __syncthreads();
            }
            else
            {
                // post multiply: alpha*B*A

                // fetch tile of matrix B  into tileA
                row_loc = row_pos + threadIdx.x;
                col_loc = k_pos + threadIdx.y;
                r       = row_loc;
                c       = col_loc;

                atile[threadIdx.x][threadIdx.y] = (r < m && c < n) ? B[c * ldb + r] : 0;

                // fetch tile of symm matrix A into tileB
                row_loc = k_pos + threadIdx.x;
                col_loc = col_pos + threadIdx.y;

                from = is_upper ? row_loc : col_loc;
                to   = is_upper ? col_loc : row_loc;

                r = from > to ? col_loc : row_loc;
                c = from > to ? row_loc : col_loc;

                if(!HERM)
                {
                    btile[threadIdx.x][threadIdx.y] = (r < n && c < n) ? A[c * lda + r] : 0;
                }
                else
                {
                    // clang-format off
                    T e = (r < n && c < n)
                              ? (from > to ? conj(A[c * lda + r])
                                           : (from == to ? std::real(A[c * lda + r]) : A[c * lda + r]))
                              : 0;
                    // clang-format on
                    btile[threadIdx.x][threadIdx.y] = e;
                }

                __syncthreads();
            }

            // m x n output, tile zero where invalid
            if(row < m && col < n)
            {
                T sum = T(0);
                for(int ki = 0; ki < TILE_NK; ++ki)
                {
                    sum += atile[threadIdx.x][ki] * btile[ki][threadIdx.y];
                }
                C[col * ldc + row] += alpha * sum;
            }

            __syncthreads();

        } // k_pos
    }
}

/**
  *  Loads pointers and launches the actual calculation kernel.
  */
template <bool        HERM,
          bool        RIGHT,
          rocblas_int DIM_XYT,
          typename TScal,
          typename TConstPtr,
          typename TPtr>
ROCBLAS_KERNEL(DIM_XYT* DIM_XYT)
rocblas_symm_hemm_kernel(bool           is_upper,
                         rocblas_int    m,
                         rocblas_int    n,
                         TScal          alpha_host_device,
                         TConstPtr      AP_array,
                         rocblas_stride shift_a,
                         int64_t        lda,
                         rocblas_stride stride_a,
                         TConstPtr      BP_array,
                         rocblas_stride shift_b,
                         int64_t        ldb,
                         rocblas_stride stride_b,
                         TPtr           CP_array,
                         rocblas_stride shift_c,
                         int64_t        ldc,
                         rocblas_stride stride_c,
                         rocblas_int    batch_count)
{
    auto alpha = load_scalar(alpha_host_device);
    if(alpha == 0)
        return;

    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        auto A = load_ptr_batch(AP_array, batch, shift_a, stride_a);
        auto B = load_ptr_batch(BP_array, batch, shift_b, stride_b);
        auto C = load_ptr_batch(CP_array, batch, shift_c, stride_c);

        // compute matrix multiplies and accumulate on the fly into C
        // when HERM does ^H in place of ^T for A fetches to symmetric empty side
        rocblas_symm_hemm_mult_add_device<HERM, RIGHT, DIM_XYT>(
            is_upper, m, n, alpha, A, lda, B, ldb, C, ldc);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

/**
  *  TScal     is always: const T* (either host or device)
  *  TConstPtr is either: const T* OR const T* const*
  *  TPtr      is either:       T* OR       T* const*
  */
template <bool HERM, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_symm_hemm_dispatch(rocblas_handle handle,
                                          rocblas_side   side,
                                          rocblas_fill   uplo,
                                          rocblas_int    m,
                                          rocblas_int    n,
                                          TScal          alpha,
                                          TConstPtr      AP,
                                          rocblas_stride offsetA,
                                          int64_t        lda,
                                          rocblas_stride strideA,
                                          TConstPtr      BP,
                                          rocblas_stride offsetB,
                                          int64_t        ldb,
                                          rocblas_stride strideB,
                                          TScal          beta,
                                          TPtr           CP,
                                          rocblas_stride offsetC,
                                          int64_t        ldc,
                                          rocblas_stride strideC,
                                          rocblas_int    batch_count)
{
    // quick return
    if(!m || !n || !batch_count)
        return rocblas_status_success;

    static constexpr int symm_SCALE_DIM_X = 128;
    static constexpr int symm_SCALE_DIM_Y = 8;
    rocblas_int          gx               = (m - 1) / (symm_SCALE_DIM_X) + 1;
    rocblas_int          gy = std::min(c_YZ_grid_launch_limit, (n - 1) / (symm_SCALE_DIM_Y) + 1);

    int batches = handle->getBatchGridDim((int)batch_count);

    dim3 symm_scale_grid(gx, gy, batches);
    dim3 symm_scale_threads(symm_SCALE_DIM_X, symm_SCALE_DIM_Y);

    static constexpr int symm_DIM_XY = 32;
    rocblas_int          bx          = (m - 1) / (symm_DIM_XY) + 1;
    rocblas_int          by = std::min(c_YZ_grid_launch_limit, (n - 1) / (symm_DIM_XY) + 1);
    dim3                 symm_grid(bx, by, batches);
    dim3                 symm_threads(symm_DIM_XY, symm_DIM_XY);

    // Launch a herk kernel for symm.
    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        // first scale C so we can use directly for output without work buffer
        ROCBLAS_LAUNCH_KERNEL((rocblas_symm_scale_kernel<symm_SCALE_DIM_X, symm_SCALE_DIM_Y>),
                              symm_scale_grid,
                              symm_scale_threads,
                              0,
                              handle->get_stream(),
                              m,
                              n,
                              beta,
                              CP,
                              offsetC,
                              ldc,
                              strideC,
                              batch_count);

        if(side == rocblas_side_left)
        {
            ROCBLAS_LAUNCH_KERNEL((rocblas_symm_hemm_kernel<HERM, false, symm_DIM_XY>),
                                  symm_grid,
                                  symm_threads,
                                  0,
                                  handle->get_stream(),
                                  uplo == rocblas_fill_upper,
                                  m,
                                  n,
                                  alpha,
                                  AP,
                                  offsetA,
                                  lda,
                                  strideA,
                                  BP,
                                  offsetB,
                                  ldb,
                                  strideB,
                                  CP,
                                  offsetC,
                                  ldc,
                                  strideC,
                                  batch_count);
        }
        else
        {
            ROCBLAS_LAUNCH_KERNEL((rocblas_symm_hemm_kernel<HERM, true, symm_DIM_XY>),
                                  symm_grid,
                                  symm_threads,
                                  0,
                                  handle->get_stream(),
                                  uplo == rocblas_fill_upper,
                                  m,
                                  n,
                                  alpha,
                                  AP,
                                  offsetA,
                                  lda,
                                  strideA,
                                  BP,
                                  offsetB,
                                  ldb,
                                  strideB,
                                  CP,
                                  offsetC,
                                  ldc,
                                  strideC,
                                  batch_count);
        }
    }
    else
    {
        if(*beta == 1 && (*alpha == 0))
            return rocblas_status_success;

        // first scale C so we can use directly for output without work buffer
        ROCBLAS_LAUNCH_KERNEL((rocblas_symm_scale_kernel<symm_SCALE_DIM_X, symm_SCALE_DIM_Y>),
                              symm_scale_grid,
                              symm_scale_threads,
                              0,
                              handle->get_stream(),
                              m,
                              n,
                              *beta,
                              CP,
                              offsetC,
                              ldc,
                              strideC,
                              batch_count);

        if(side == rocblas_side_left)
        {
            ROCBLAS_LAUNCH_KERNEL((rocblas_symm_hemm_kernel<HERM, false, symm_DIM_XY>),
                                  symm_grid,
                                  symm_threads,
                                  0,
                                  handle->get_stream(),
                                  uplo == rocblas_fill_upper,
                                  m,
                                  n,
                                  *alpha,
                                  AP,
                                  offsetA,
                                  lda,
                                  strideA,
                                  BP,
                                  offsetB,
                                  ldb,
                                  strideB,
                                  CP,
                                  offsetC,
                                  ldc,
                                  strideC,
                                  batch_count);
        }
        else
        {
            ROCBLAS_LAUNCH_KERNEL((rocblas_symm_hemm_kernel<HERM, true, symm_DIM_XY>),
                                  symm_grid,
                                  symm_threads,
                                  0,
                                  handle->get_stream(),
                                  uplo == rocblas_fill_upper,
                                  m,
                                  n,
                                  *alpha,
                                  AP,
                                  offsetA,
                                  lda,
                                  strideA,
                                  BP,
                                  offsetB,
                                  ldb,
                                  strideB,
                                  CP,
                                  offsetC,
                                  ldc,
                                  strideC,
                                  batch_count);
        }
    }
    return rocblas_status_success;
}

template <bool BATCHED, bool HERM, typename T, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_symm_hemm_template_non_batched(rocblas_handle handle,
                                                      rocblas_side   side,
                                                      rocblas_fill   uplo,
                                                      rocblas_int    m,
                                                      rocblas_int    n,
                                                      TScal          alpha,
                                                      TConstPtr      a,
                                                      rocblas_stride offsetA,
                                                      int64_t        lda,
                                                      TConstPtr      b,
                                                      rocblas_stride offsetB,
                                                      int64_t        ldb,
                                                      TScal          beta,
                                                      TPtr           c,
                                                      rocblas_stride offsetC,
                                                      int64_t        ldc)
{
    // nb_diag is a tuning parameter. It is the size of the diagonal blocks in the matrix
    // a. It is also the starting size for subdiagonal blocks in calls to gemm.
    rocblas_int nb_diag = 32; // size of diag blocks of triangle matrix a

    if(std::is_same_v<T, float>)
    {
        nb_diag = SSYMM_MIN_NB;
    }
    else if(std::is_same_v<T, double>)
    {
        nb_diag = DSYMM_MIN_NB;
    }
    else if(std::is_same_v<T, rocblas_float_complex>)
    {
        nb_diag = CSYMM_MIN_NB;
    }
    else if(std::is_same_v<T, rocblas_double_complex>)
    {
        nb_diag = ZSYMM_MIN_NB;
    }

    rocblas_operation trans_a
        = HERM ? rocblas_operation_conjugate_transpose : rocblas_operation_transpose;

    // rocblas_internal_gemm requires alpha and beta to be pointers to host memory.
    // If they are pointers to device memory, then copy *alpha and *beta to alpha_h and beta_h
    // and make alpha and beta point to alpha_h and beta_h, and set pointer mode to host.
    // Restore pointer mode in destructor when save_pointer_mode goes out of scope.
    T alpha_h, beta_h;
    RETURN_IF_ROCBLAS_ERROR(
        rocblas_copy_alpha_beta_to_host_if_on_device(handle, alpha, beta, alpha_h, beta_h, 1));
    auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

    if(*alpha == T(0) && *beta == T(1.0))
        return rocblas_status_success;

    rocblas_int ka = rocblas_side_left == side ? m : n; // dimension of triangle matrix a

    rocblas_int n_nb   = ka / nb_diag; // number of diag blocks of matrix a of size nb_diag
    rocblas_int nb_rem = ka % nb_diag; // remainder diag block size when ka not multiple of nb_diag

    rocblas_int symm_m = rocblas_side_left == side ? nb_diag : m; // diag block symm argument m
    rocblas_int symm_n = rocblas_side_left == side ? n : nb_diag; // diag block symm argument n

    int64_t diag_a_stride = 1 + lda; // stride for diag blocks in a
    int64_t diag_b_stride = rocblas_side_left == side ? 1 : ldb; // stride of b panels
    int64_t diag_c_stride = rocblas_side_left == side ? 1 : ldc; // stride of c panels

    rocblas_int i_diag; // index of diag block

    // calls to symm_strided_batched for diagonal blocks of size nb_diag
    // clang-format off
    RETURN_IF_ROCBLAS_ERROR( (rocblas_symm_hemm_dispatch<HERM>(handle,
             side, uplo, symm_m, symm_n, alpha,
             a, offsetA, lda, nb_diag * diag_a_stride,
             b, offsetB, ldb, nb_diag * diag_b_stride, beta,
             c, offsetC, ldc, nb_diag * diag_c_stride, n_nb)));

    // calls to symm for single remainder diagonal block of size nb_rem < nb_diag
    if(nb_rem != 0)
    {
        i_diag = n_nb * nb_diag; // diag block at a[i_diag, i_diag], size is nb_rem
        symm_m = rocblas_side_left == side ? nb_rem : m;
        symm_n = rocblas_side_left == side ? n : nb_rem;

        RETURN_IF_ROCBLAS_ERROR( (rocblas_symm_hemm_dispatch<HERM>(handle,
                 side, uplo, symm_m, symm_n, alpha,
                 a, i_diag * diag_a_stride + offsetA, lda, 0,
                 b, i_diag * diag_b_stride + offsetB, ldb, 0, beta,
                 c, i_diag * diag_c_stride + offsetC, ldc, 0, 1)));
    }

    int64_t stride, stride_rem, i_start;
    int64_t nb; // size of sub-diagonal blocks of matrix a

    // calls to gemm for sub-diagonal square blocks in matrix a with size m = n = nb.
    // Start with nb = nb_diag. Each iteration of the outer loop nb doubles, and the
    // number of gemm calls halves.
    for(nb = nb_diag, i_start = nb_diag; i_start < ka; i_start += nb, nb *= 2)
    {
        stride     = nb * 2; // stride for both indices of a, and for one index of b and c
        n_nb       = (ka - i_start) / stride; // number of calls to gemm
        stride_rem = (ka - i_start) % stride; // remainder when stride not multiple of ka-istart
        if(stride_rem >= nb)
        {
            stride_rem = 0;
            n_nb += 1;
        }

        int64_t i1       = i_start;
        int64_t i2       = i_start - nb;

        // Note:
        // SYMM, HEMM and other GEMM based functions, will follow the pattern of using 64-bit GEMM launcher for both 32-bit and 64-bit input sizes.
        // This is to avoid multiple instantiation of functions to call 32-bit & 64-bit.
        // The use of 64-bit GEMM API does not cause any performance degrade for 32-bit inputs.
        if(rocblas_side_right == side)
        {
            if(rocblas_fill_lower == uplo)
            {
                // lower sub-diagonal (from stored part of a)
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(handle,
                         rocblas_operation_none, rocblas_operation_none, m, nb, nb, alpha,
                         b,      i1 * ldb + offsetB, ldb,          stride * ldb,
                         a, i1 + i2 * lda + offsetA, lda, stride + stride * lda, &beta_1<T>,
                         c,      i2 * ldc + offsetC, ldc,          stride * ldc, n_nb)));

                // upper sub-diagonal (from transpose of stored part of a)
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(handle,
                         rocblas_operation_none, trans_a, m, nb, nb, alpha,
                         b,      i2 * ldb + offsetB, ldb,          stride * ldb,
                         a, i1 + i2 * lda + offsetA, lda, stride + stride * lda, &beta_1<T>,
                         c,      i1 * ldc + offsetC, ldc,          stride * ldc, n_nb)));
            }
            else
            {
                // upper sub-diagonal (from stored part of a)
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(handle,
                         rocblas_operation_none, rocblas_operation_none, m, nb, nb, alpha,
                         b, i2*ldb         + offsetB, ldb, stride*ldb,
                         a, i1-nb + i1*lda + offsetA, lda, stride*(1+lda), &beta_1<T>,
                         c, i1*ldc         + offsetC, ldc, stride*ldc, n_nb)));

                // lower sub-diagonal (from transpose of stored part of a)
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(handle,
                         rocblas_operation_none, trans_a, m, nb, nb, alpha,
                         b, i1*ldb         + offsetB, ldb, stride*ldb,
                         a, i1-nb + i1*lda + offsetA, lda, stride*(1+lda),  &beta_1<T>,
                         c, i2*ldc         + offsetC, ldc, stride*ldc, n_nb)));
            }
        }
        else
        {
            if(rocblas_fill_lower == uplo)
            {
                // lower sub-diagonal (from stored part of a)
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(handle,
                         rocblas_operation_none, rocblas_operation_none, nb, n, nb, alpha,
                         a, i1 + i2*lda + offsetA, lda, stride*(1+lda),
                         b, i2          + offsetB, ldb, stride,  &beta_1<T>,
                         c, i1          + offsetC, ldc, stride, n_nb)));

                // upper sub-diagonal (from transpose of stored part of a)
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(handle,
                         trans_a, rocblas_operation_none, nb, n, nb, alpha,
                         a, i1 + i2*lda + offsetA, lda, stride*(1+lda),
                         b, i1          + offsetB, ldb, stride,  &beta_1<T>,
                         c, i2          + offsetC, ldc, stride, n_nb)));
            }
            else
            {
                // upper sub-diagonal (from stored part of a)
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(handle,
                         rocblas_operation_none, rocblas_operation_none, nb, n, nb, alpha,
                         a, i2 + i1*lda + offsetA, lda, stride*(1+lda),
                         b, i1          + offsetB, ldb, stride,  &beta_1<T>,
                         c, i2          + offsetC, ldc, stride, n_nb)));

                // lower sub-diagonal (from transpose of stored part of a)
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(handle,
                         trans_a, rocblas_operation_none, nb, n, nb, alpha,
                         a, i2 + i1*lda + offsetA, lda, stride*(1+lda),
                         b, i2          + offsetB, ldb, stride,  &beta_1<T>,
                         c, i1          + offsetC, ldc, stride, n_nb)));
            }
        }
        // remainder gemm block of size nb_rem x nb where n_rem < nb
        if(stride_rem != 0)
        {
            int64_t i1     = i_start + n_nb * stride;
            int64_t i2     = i1 - nb;
            rocblas_int nb_rem = ka - i1;

            if(rocblas_side_right == side)
            {
                if(rocblas_fill_lower == uplo)
                {
                    // lower sub-diagonal (from stored part of a)
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(handle,
                             rocblas_operation_none, rocblas_operation_none, m, nb, nb_rem, alpha,
                             b,      i1 * ldb + offsetB, ldb, 0,
                             a, i1 + i2 * lda + offsetA, lda, 0,  &beta_1<T>,
                             c,      i2 * ldc + offsetC, ldc, 0, 1)));

                    // upper sub-diagonal (from transpose of stored part of a)
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(handle,
                             rocblas_operation_none, trans_a, m, nb_rem, nb, alpha,
                             b,      i2 * ldb + offsetB, ldb, 0,
                             a, i1 + i2 * lda + offsetA, lda, 0,  &beta_1<T>,
                             c,      i1 * ldc + offsetC, ldc, 0, 1)));
                }
                else
                {
                    // upper sub-diagonal (from stored part of a)
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(handle,
                             rocblas_operation_none, rocblas_operation_none, m, nb_rem, nb, alpha,
                             b,      i2*ldb + offsetB, ldb, 0,
                             a, i2 + i1*lda + offsetA, lda, 0,  &beta_1<T>,
                             c,      i1*ldc + offsetC, ldc, 0, 1)));

                    // lower sub-diagonal (from transpose of stored part of a)
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(handle,
                             rocblas_operation_none, trans_a, m, nb, nb_rem, alpha,
                             b,      i1*ldb + offsetB, ldb, 0,
                             a, i2 + i1*lda + offsetA, lda, 0,  &beta_1<T>,
                             c,      i2*ldc + offsetC, ldc, 0, 1)));
                }
            }
            else
            {
                if(rocblas_fill_lower == uplo)
                {
                    // lower sub-diagonal (from stored part of a)
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(handle,
                             rocblas_operation_none, rocblas_operation_none, nb_rem, n, nb, alpha,
                             a, i2*lda + i1 + offsetA, lda, 0,
                             b,          i2 + offsetB, ldb, 0,  &beta_1<T>,
                             c,          i1 + offsetC, ldc, 0, 1)));

                    // upper sub-diagonal (from transpose of stored part of a)
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(handle,
                             trans_a, rocblas_operation_none, nb, n, nb_rem, alpha,
                             a, i2*lda + i1 + offsetA, lda, 0,
                             b,          i1 + offsetB, ldb, 0,  &beta_1<T>,
                             c,          i2 + offsetC, ldc, 0, 1)));
                }
                else
                {
                    // upper sub-diagonal (from stored part of a)
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(handle,
                             rocblas_operation_none, rocblas_operation_none, nb, n, nb_rem, alpha,
                             a, i1*lda + i2 + offsetA, lda, 0,
                             b,          i1 + offsetB, ldb, 0,  &beta_1<T>,
                             c,          i2 + offsetC, ldc, 0, 1)));

                    // lower sub-diagonal (from transpose of stored part of a)
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(handle,
                             trans_a, rocblas_operation_none, nb_rem, n, nb, alpha,
                             a, i1*lda + i2 + offsetA, lda, 0,
                             b,          i2 + offsetB, ldb, 0,  &beta_1<T>,
                             c,          i1 + offsetC, ldc, 0, 1)));
                }
            }
        }
    }

    return rocblas_status_success;
}

template <bool HERM, typename TConstPtr, typename TPtr>
rocblas_status rocblas_hemm_symm_check_numerics(const char*    function_name,
                                                rocblas_handle handle,
                                                rocblas_side   side,
                                                rocblas_fill   uplo,
                                                int64_t    m,
                                                int64_t    n,
                                                TConstPtr      A,
                                                int64_t    lda,
                                                rocblas_stride stride_a,
                                                TConstPtr      B,
                                                int64_t    ldb,
                                                rocblas_stride stride_b,
                                                TPtr           C,
                                                int64_t    ldc,
                                                rocblas_stride stride_c,
                                                int64_t    batch_count,
                                                const int      check_numerics,
                                                bool           is_input)
{
    rocblas_status check_numerics_status = rocblas_status_success;
    if(is_input)
    {
        rocblas_int rows = (side == rocblas_side_left ? m : n);
        rocblas_int cols = (side == rocblas_side_left ? m : n);

        check_numerics_status = rocblas_internal_check_numerics_matrix_template(
            function_name,
            handle,
            rocblas_operation_none,
            uplo,
            HERM ? rocblas_client_hermitian_matrix : rocblas_client_symmetric_matrix,
            rows,
            cols,
            A,
            0,
            lda,
            stride_a,
            batch_count,
            check_numerics,
            is_input);
        if(check_numerics_status != rocblas_status_success)
            return check_numerics_status;

        check_numerics_status
            = rocblas_internal_check_numerics_matrix_template(function_name,
                                                              handle,
                                                              rocblas_operation_none,
                                                              rocblas_fill_full,
                                                              rocblas_client_general_matrix,
                                                              m,
                                                              n,
                                                              B,
                                                              0,
                                                              ldb,
                                                              stride_b,
                                                              batch_count,
                                                              check_numerics,
                                                              is_input);
        if(check_numerics_status != rocblas_status_success)
            return check_numerics_status;
    }

    check_numerics_status
        = rocblas_internal_check_numerics_matrix_template(function_name,
                                                          handle,
                                                          rocblas_operation_none,
                                                          rocblas_fill_full,
                                                          rocblas_client_general_matrix,
                                                          m,
                                                          n,
                                                          C,
                                                          0,
                                                          ldc,
                                                          stride_c,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);

    return check_numerics_status;
}

template <bool BATCHED, bool HERM, typename T, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_symm_hemm_template_batched(rocblas_handle handle,
                                             rocblas_side   side,
                                             rocblas_fill   uplo,
                                             rocblas_int    m,
                                             rocblas_int    n,
                                             TScal          alpha,
                                             TConstPtr      a,
                                             rocblas_stride offsetA,
                                             int64_t    lda,
                                             rocblas_stride strideA,
                                             TConstPtr      b,
                                             rocblas_stride offsetB,
                                             int64_t    ldb,
                                             rocblas_stride strideB,
                                             TScal          beta,
                                             TPtr           c,
                                             rocblas_stride offsetC,
                                             int64_t    ldc,
                                             rocblas_stride strideC,
                                             rocblas_int    batch_count)
{
    // nb_diag is a tuning parameter. It is the size of the diagonal blocks in the matrix
    // a. It is also the starting size for subdiagonal blocks in calls to gemm.
    rocblas_int nb_diag = 32; // size of diag blocks of triangle matrix a

    if     (std::is_same_v<T, float>)                  { nb_diag = SSYMM_BATCHED_MIN_NB; }
    else if(std::is_same_v<T, double>)                 { nb_diag = DSYMM_BATCHED_MIN_NB; }
    else if(std::is_same_v<T, rocblas_float_complex>)  { nb_diag = CSYMM_BATCHED_MIN_NB; }
    else if(std::is_same_v<T, rocblas_double_complex>) { nb_diag = ZSYMM_BATCHED_MIN_NB; }

    rocblas_operation trans_a = HERM ? rocblas_operation_conjugate_transpose : rocblas_operation_transpose;

    // rocblas_internal_gemm requires alpha and beta to be pointers to host memory.
    // If they are pointers to device memory, then copy *alpha and *beta to alpha_h and beta_h
    // and make alpha and beta point to alpha_h and beta_h, and set pointer mode to host.
    // Restore pointer mode in destructor when save_pointer_mode goes out of scope.
    T alpha_h, beta_h;
    RETURN_IF_ROCBLAS_ERROR(
        rocblas_copy_alpha_beta_to_host_if_on_device(handle, alpha, beta, alpha_h, beta_h, 1));
    auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

    if (*alpha == T(0) && *beta == T(1.0))
        return rocblas_status_success;

    rocblas_int ka = rocblas_side_left == side ? m : n; // dimension of triangle matrix a

    rocblas_int n_nb   = ka / nb_diag; // number of diag blocks of matrix a of size nb_diag
    rocblas_int nb_rem = ka % nb_diag; // remainder diag block size when ka not multiple of nb_diag

    rocblas_int symm_m = rocblas_side_left == side ? nb_diag : m; // diag block symm argument m
    rocblas_int symm_n = rocblas_side_left == side ? n : nb_diag; // diag block symm argument n

    int64_t     diag_a_stride = 1 + lda; // stride for diag blocks in a
    int64_t     diag_b_stride = rocblas_side_left == side ? 1 : ldb; // stride of b panels
    int64_t     diag_c_stride = rocblas_side_left == side ? 1 : ldc; // stride of c panels

    rocblas_int i_diag; // index of diag block

    // clang-format off
    // calls to symm_strided_batched for diagonal blocks of size nb_diag
    for(int i_nb = 0; i_nb < n_nb; i_nb++)
    {
        RETURN_IF_ROCBLAS_ERROR( (rocblas_symm_hemm_dispatch<HERM>(handle,
                 side, uplo, symm_m, symm_n, alpha,
                 a, i_nb * (nb_diag * diag_a_stride) + offsetA, lda, strideA,
                 b, i_nb * (nb_diag * diag_b_stride) + offsetB, ldb, strideB, beta,
                 c, i_nb * (nb_diag * diag_c_stride) + offsetC, ldc, strideC, batch_count)));
    }

    // calls to symm for single remainder diagonal block of size nb_rem < nb_diag
    if(nb_rem != 0)
    {
        i_diag = n_nb * nb_diag; // diag block at a[i_diag, i_diag], size is nb_rem
        symm_m = rocblas_side_left == side ? nb_rem : m;
        symm_n = rocblas_side_left == side ? n : nb_rem;

        RETURN_IF_ROCBLAS_ERROR( (rocblas_symm_hemm_dispatch<HERM>(handle,
                 side, uplo, symm_m, symm_n, alpha,
                 a, i_diag * diag_a_stride + offsetA, lda, strideA,
                 b, i_diag * diag_b_stride + offsetB, ldb, strideB, beta,
                 c, i_diag * diag_c_stride + offsetC, ldc, strideC, batch_count)));
    }

    int64_t stride, stride_rem, i_start;
    int64_t nb; // size of sub-diagonal blocks of matrix a
    // calls to gemm for sub-diagonal square blocks in matrix a with size m = n = nb.
    // Start with nb = nb_diag. Each iteration of the outer loop nb doubles, and the
    // number of gemm calls halves.
    for(nb = nb_diag, i_start = nb_diag; i_start < ka; i_start += nb, nb *= 2)
    {
        stride     = nb * 2; // stride for both indices of a, and for one index of b and c
        n_nb       = (ka - i_start) / stride; // number of calls to gemm
        stride_rem = (ka - i_start) % stride; // remainder when stride not multiple of ka-istart
        if(stride_rem >= nb)
        {
            stride_rem = 0;
            n_nb += 1;
        }

        int64_t i1       = i_start;
        int64_t i2       = i_start - nb;

        // Note:
        // SYMM, HEMM and other GEMM based functions, will follow the pattern of using 64-bit GEMM launcher for both 32-bit and 64-bit input sizes.
        // This is to avoid multiple instantiation of functions to call 32-bit & 64-bit.
        // The use of 64-bit GEMM API does not cause any performance degrade for 32-bit inputs.
        for(int i_nb = 0; i_nb < n_nb; i_nb++)
        {
            if(rocblas_side_right == side)
            {
                if(rocblas_fill_lower == uplo)
                {
                    // lower sub-diagonal (from stored part of a)
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(handle,
                             rocblas_operation_none, rocblas_operation_none, m, nb, nb, alpha,
                             b,      i1*ldb + offsetB + i_nb * stride * ldb    , ldb, strideB,
                             a, i1 + i2*lda + offsetA + i_nb * stride * (1+lda), lda, strideA, &beta_1<T>,
                             c,      i2*ldc + offsetC + i_nb * stride * ldc    , ldc, strideC, batch_count)));

                    // upper sub-diagonal (from transpose of stored part of a)
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(handle,
                             rocblas_operation_none, trans_a, m, nb, nb, alpha,
                             b,      i2*ldb + offsetB + i_nb * stride * ldb    , ldb, strideB,
                             a, i1 + i2*lda + offsetA + i_nb * stride * (1+lda), lda, strideA, &beta_1<T>,
                             c,      i1*ldc + offsetC + i_nb * stride * ldc    , ldc, strideC, batch_count)));
                }
                else
                {
                    // upper sub-diagonal (from stored part of a)
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(handle,
                             rocblas_operation_none, rocblas_operation_none, m, nb, nb, alpha,
                             b, i2*ldb         + offsetB + i_nb * stride * ldb    , ldb, strideB,
                             a, i1*lda + i1-nb + offsetA + i_nb * stride * (1+lda), lda, strideA, &beta_1<T>,
                             c, i1*ldc         + offsetC + i_nb * stride * ldc    , ldc, strideC, batch_count)));

                    // lower sub-diagonal (from transpose of stored part of a)
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(handle,
                             rocblas_operation_none, trans_a, m, nb, nb, alpha,
                             b, i1*ldb         + offsetB + i_nb * stride * ldb    , ldb, strideB,
                             a, i1*lda + i1-nb + offsetA + i_nb * stride * (1+lda), lda, strideA, &beta_1<T>,
                             c, i2*ldc         + offsetC + i_nb * stride * ldc    , ldc, strideC, batch_count)));
                }
            }
            else
            {
                if(rocblas_fill_lower == uplo)
                {
                    // lower sub-diagonal (from stored part of a)
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(handle,
                             rocblas_operation_none, rocblas_operation_none, nb, n, nb, alpha,
                             a, i1 + i2*lda + offsetA + i_nb * stride * (1+lda), lda, strideA,
                             b, i2          + offsetB + i_nb * stride          , ldb, strideB, &beta_1<T>,
                             c, i1          + offsetC + i_nb * stride          , ldc, strideC, batch_count)));

                    // upper sub-diagonal (from transpose of stored part of a)
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(handle,
                             trans_a, rocblas_operation_none, nb, n, nb, alpha,
                             a, i1 + i2*lda + offsetA + i_nb * stride * (1+lda), lda, strideA,
                             b, i1          + offsetB + i_nb * stride          , ldb, strideB, &beta_1<T>,
                             c, i2          + offsetC + i_nb * stride          , ldc, strideC, batch_count)));
                }
                else
                {
                    // upper sub-diagonal (from stored part of a)
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(handle,
                             rocblas_operation_none, rocblas_operation_none, nb, n, nb, alpha,
                             a, i2 + i1*lda + offsetA + i_nb * stride * (1+lda), lda, strideA,
                             b, i1          + offsetB + i_nb * stride          , ldb, strideB, &beta_1<T>,
                             c, i2          + offsetC + i_nb * stride          , ldc, strideC, batch_count)));

                    // lower sub-diagonal (from transpose of stored part of a)
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(handle,
                             trans_a, rocblas_operation_none, nb, n, nb, alpha,
                             a, i2 + i1*lda + offsetA + i_nb * stride * (1+lda), lda, strideA,
                             b, i2          + offsetB + i_nb * stride          , ldb, strideB, &beta_1<T>,
                             c, i1          + offsetC + i_nb * stride          , ldc, strideC, batch_count)));
                }
            }
        }
        // remainder gemm block of size nb_rem x nb where n_rem < nb
        if(stride_rem != 0)
        {
            int64_t i1     = i_start + n_nb * stride;
            int64_t i2     = i1 - nb;
            rocblas_int nb_rem = ka - i1;

            if(rocblas_side_right == side)
            {
                if(rocblas_fill_lower == uplo)
                {
                    // lower sub-diagonal (from stored part of a)
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(handle,
                             rocblas_operation_none, rocblas_operation_none, m, nb, nb_rem, alpha,
                             b,      i1*ldb + offsetB, ldb, strideB,
                             a, i1 + i2*lda + offsetA, lda, strideA, &beta_1<T>,
                             c,      i2*ldc + offsetC, ldc, strideC, batch_count)));

                    // upper sub-diagonal (from transpose of stored part of a)
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(handle,
                             rocblas_operation_none, trans_a, m, nb_rem, nb, alpha,
                             b,      i2*ldb + offsetB, ldb, strideB,
                             a, i1 + i2*lda + offsetA, lda, strideA, &beta_1<T>,
                             c,      i1*ldc + offsetC, ldc, strideC, batch_count)));
                }
                else
                {
                    // upper sub-diagonal (from stored part of a)
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(handle,
                             rocblas_operation_none, rocblas_operation_none, m, nb_rem, nb, alpha,
                             b,      i2*ldb + offsetB, ldb, strideB,
                             a, i2 + i1*lda + offsetA, lda, strideA, &beta_1<T>,
                             c,      i1*ldc + offsetC, ldc, strideC, batch_count)));

                    // lower sub-diagonal (from transpose of stored part of a)
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(handle,
                             rocblas_operation_none, trans_a, m, nb, nb_rem, alpha,
                             b,      i1*ldb + offsetB, ldb, strideB,
                             a, i2 + i1*lda + offsetA, lda, strideA, &beta_1<T>,
                             c,      i2*ldc + offsetC, ldc, strideC, batch_count)));
                }
            }
            else
            {
                if(rocblas_fill_lower == uplo)
                {
                    // lower sub-diagonal (from stored part of a)
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(handle,
                             rocblas_operation_none, rocblas_operation_none, nb_rem, n, nb, alpha,
                             a, i2*lda + i1 + offsetA, lda, strideA,
                             b,          i2 + offsetB, ldb, strideB, &beta_1<T>,
                             c,          i1 + offsetC, ldc, strideC, batch_count)));

                    // upper sub-diagonal (from transpose of stored part of a)
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(handle,
                             trans_a, rocblas_operation_none, nb, n, nb_rem, alpha,
                             a, i2*lda + i1 + offsetA, lda, strideA,
                             b,          i1 + offsetB, ldb, strideB, &beta_1<T>,
                             c,          i2 + offsetC, ldc, strideC, batch_count)));
                }
                else
                {
                    // upper sub-diagonal (from stored part of a)
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(handle,
                             rocblas_operation_none, rocblas_operation_none, nb, n, nb_rem, alpha,
                             a, i1*lda + i2 + offsetA, lda, strideA,
                             b,          i1 + offsetB, ldb, strideB, &beta_1<T>,
                             c,          i2 + offsetC, ldc, strideC, batch_count)));

                    // lower sub-diagonal (from transpose of stored part of a)
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(handle,
                             trans_a, rocblas_operation_none, nb_rem, n, nb, alpha,
                             a, i1*lda + i2 + offsetA, lda, strideA,
                             b,          i2 + offsetB, ldb, strideB, &beta_1<T>,
                             c,          i1 + offsetC, ldc, strideC, batch_count)));
                }
            }
        }
    }
    return rocblas_status_success;
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_symm_template(rocblas_handle handle,
                                   rocblas_side   side,
                                   rocblas_fill   uplo,
                                   rocblas_int    m,
                                   rocblas_int    n,
                                   const T*       alpha,
                                   const T*       A,
                                   rocblas_stride offsetA,
                                   rocblas_int    lda,
                                   rocblas_stride strideA,
                                   const T*       B,
                                   rocblas_stride offsetB,
                                   rocblas_int    ldb,
                                   rocblas_stride strideB,
                                   const T*       beta,
                                   T*             C,
                                   rocblas_stride offsetC,
                                   rocblas_int    ldc,
                                   rocblas_stride strideC,
                                   rocblas_int    batch_count)
{

    constexpr bool HERM = false;
    return  rocblas_internal_symm_hemm_launcher<HERM>(
            handle, side, uplo, m, n, alpha,
            A, offsetA, (int64_t)lda, strideA,
            B, offsetB, (int64_t)ldb, strideB, beta,
            C, offsetC, (int64_t)ldc, strideC, batch_count);
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_symm_batched_template(rocblas_handle handle,
                                   rocblas_side   side,
                                   rocblas_fill   uplo,
                                   rocblas_int    m,
                                   rocblas_int    n,
                                   const T*       alpha,
                                   const T* const* A,
                                   rocblas_stride offsetA,
                                   rocblas_int    lda,
                                   rocblas_stride strideA,
                                   const T* const* B,
                                   rocblas_stride offsetB,
                                   rocblas_int    ldb,
                                   rocblas_stride strideB,
                                   const T*       beta,
                                   T* const*      C,
                                   rocblas_stride offsetC,
                                   rocblas_int    ldc,
                                   rocblas_stride strideC,
                                   rocblas_int    batch_count)
{
    constexpr bool HERM = false;
    return rocblas_internal_symm_hemm_batched_launcher<HERM>(handle, side, uplo, m, n, alpha,
        A, offsetA, (int64_t)lda, strideA,
        B, offsetB, (int64_t)ldb, strideB, beta,
        C, offsetC, (int64_t)ldc, strideC, batch_count);
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_hemm_template(rocblas_handle handle,
                                   rocblas_side   side,
                                   rocblas_fill   uplo,
                                   rocblas_int    m,
                                   rocblas_int    n,
                                   const T*       alpha,
                                   const T*       A,
                                   rocblas_stride offsetA,
                                   rocblas_int    lda,
                                   rocblas_stride strideA,
                                   const T*       B,
                                   rocblas_stride offsetB,
                                   rocblas_int    ldb,
                                   rocblas_stride strideB,
                                   const T*       beta,
                                   T*             C,
                                   rocblas_stride offsetC,
                                   rocblas_int    ldc,
                                   rocblas_stride strideC,
                                   rocblas_int    batch_count)
{
    constexpr bool HERM = true;
    return  rocblas_internal_symm_hemm_launcher<HERM>(
            handle, side, uplo, m, n, alpha,
            A, offsetA, (int64_t)lda,strideA,
            B, offsetB, (int64_t)ldb, strideB, beta,
            C, offsetC, (int64_t)ldc,strideC, batch_count);
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_hemm_batched_template(rocblas_handle handle,
                                   rocblas_side   side,
                                   rocblas_fill   uplo,
                                   rocblas_int    m,
                                   rocblas_int    n,
                                   const T*       alpha,
                                   const T* const* A,
                                   rocblas_stride offsetA,
                                   rocblas_int    lda,
                                   rocblas_stride strideA,
                                   const T* const* B,
                                   rocblas_stride offsetB,
                                   rocblas_int    ldb,
                                   rocblas_stride strideB,
                                   const T*       beta,
                                   T* const*      C,
                                   rocblas_stride offsetC,
                                   rocblas_int    ldc,
                                   rocblas_stride strideC,
                                   rocblas_int    batch_count)
{
    constexpr bool HERM = true;
    return rocblas_internal_symm_hemm_batched_launcher<HERM>(handle, side, uplo, m, n, alpha,
        A, offsetA, (int64_t)lda, strideA,
        B, offsetB, (int64_t)ldb, strideB, beta,
        C, offsetC, (int64_t)ldc, strideC, batch_count);
}


template <bool HERM , typename T>
rocblas_status rocblas_internal_symm_hemm_launcher(rocblas_handle handle,
                                              rocblas_side   side,
                                              rocblas_fill   uplo,
                                              rocblas_int        m,
                                              rocblas_int        n,
                                              const T*       alpha,
                                              const T*       A,
                                              rocblas_stride offsetA,
                                              int64_t        lda,
                                              rocblas_stride strideA,
                                              const T*       B,
                                              rocblas_stride offsetB,
                                              int64_t        ldb,
                                              rocblas_stride strideB,
                                              const T*       beta,
                                              T*             C,
                                              rocblas_stride offsetC,
                                              int64_t        ldc,
                                              rocblas_stride strideC,
                                              rocblas_int        batch_count)
{

    if(batch_count == 1 )
    {
        return rocblas_symm_hemm_template_non_batched<false, HERM, T>(
            handle, side, uplo, m, n, alpha,
            A, offsetA, lda,
            B, offsetB, ldb, beta,
            C, offsetC, ldc);
    }
    else
    {

        return rocblas_symm_hemm_template_batched<false, HERM, T>(handle, side, uplo, m, n, alpha,
        A, offsetA, lda, strideA,
        B, offsetB, ldb, strideB, beta,
        C, offsetC, ldc, strideC, batch_count);
    }

}

template <bool HERM, typename T>
rocblas_status rocblas_internal_symm_hemm_batched_launcher(rocblas_handle  handle,
                                                           rocblas_side    side,
                                                           rocblas_fill    uplo,
                                                           rocblas_int     m,
                                                           rocblas_int     n,
                                                           const T*        alpha,
                                                           const T* const* A,
                                                           rocblas_stride  offsetA,
                                                           int64_t         lda,
                                                           rocblas_stride  strideA,
                                                           const T* const* B,
                                                           rocblas_stride  offsetB,
                                                           int64_t         ldb,
                                                           rocblas_stride  strideB,
                                                           const T*        beta,
                                                           T* const*       C,
                                                           rocblas_stride  offsetC,
                                                           int64_t         ldc,
                                                           rocblas_stride  strideC,
                                                           rocblas_int     batch_count)
{
    return rocblas_symm_hemm_template_batched<true, HERM, T>(handle, side, uplo, m, n, alpha,
        A, offsetA, lda, strideA,
        B, offsetB, ldb, strideB, beta,
        C, offsetC, ldc, strideC, batch_count);
}



// Instantiations below will need to be manually updated to match any change in
// template parameters in the files symm*.cpp

#ifdef INST_SYMM_TEMPLATE
#error INST_SYMM_TEMPLATE already defined
#endif

#define INST_SYMM_TEMPLATE(T_)                                                       \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status rocblas_internal_symm_template<T_> \
                                        (rocblas_handle handle,                             \
                                        rocblas_side   side,                                \
                                        rocblas_fill   uplo,                                \
                                        rocblas_int    m,                                   \
                                        rocblas_int    n,                                   \
                                        const T_*      alpha,                               \
                                        const T_*      A,                                   \
                                        rocblas_stride offsetA,                             \
                                        rocblas_int    lda,                                 \
                                        rocblas_stride strideA,                             \
                                        const T_*      B,                                   \
                                        rocblas_stride offsetB,                             \
                                        rocblas_int    ldb,                                 \
                                        rocblas_stride strideB,                             \
                                        const T_*      beta,                                \
                                        T_*            C,                                   \
                                        rocblas_stride offsetC,                             \
                                        rocblas_int    ldc,                                 \
                                        rocblas_stride strideC,                             \
                                        rocblas_int    batch_count);

INST_SYMM_TEMPLATE(float)
INST_SYMM_TEMPLATE(double)
INST_SYMM_TEMPLATE(rocblas_float_complex)
INST_SYMM_TEMPLATE(rocblas_double_complex)

#undef INST_SYMM_TEMPLATE

#ifdef INST_SYMM_BATCHED_TEMPLATE
#error INST_SYMM_BATCHED_TEMPLATE already defined
#endif

#define INST_SYMM_BATCHED_TEMPLATE(T_)                                                       \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status rocblas_internal_symm_batched_template<T_> \
                                                (rocblas_handle handle,                             \
                                                rocblas_side    side,                               \
                                                rocblas_fill    uplo,                               \
                                                rocblas_int     m,                                  \
                                                rocblas_int     n,                                  \
                                                const T_*       alpha,                              \
                                                const T_* const* A,                                 \
                                                rocblas_stride  offsetA,                            \
                                                rocblas_int     lda,                                \
                                                rocblas_stride  strideA,                            \
                                                const T_* const* B,                                 \
                                                rocblas_stride  offsetB,                            \
                                                rocblas_int     ldb,                                \
                                                rocblas_stride  strideB,                            \
                                                const T_*       beta,                               \
                                                T_* const*      C,                                  \
                                                rocblas_stride  offsetC,                            \
                                                rocblas_int     ldc,                                \
                                                rocblas_stride  strideC,                            \
                                                rocblas_int     batch_count);

INST_SYMM_BATCHED_TEMPLATE(float)
INST_SYMM_BATCHED_TEMPLATE(double)
INST_SYMM_BATCHED_TEMPLATE(rocblas_float_complex)
INST_SYMM_BATCHED_TEMPLATE(rocblas_double_complex)

#undef INST_SYMM_BATCHED_TEMPLATE

#ifdef INST_HEMM_TEMPLATE
#error INST_HEMM_TEMPLATE already defined
#endif

#define INST_HEMM_TEMPLATE(T_)                                                       \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status rocblas_internal_hemm_template<T_> \
                                        (rocblas_handle handle,                             \
                                        rocblas_side   side,                                \
                                        rocblas_fill   uplo,                                \
                                        rocblas_int    m,                                   \
                                        rocblas_int    n,                                   \
                                        const T_*      alpha,                               \
                                        const T_*      A,                                   \
                                        rocblas_stride offsetA,                             \
                                        rocblas_int    lda,                                 \
                                        rocblas_stride strideA,                             \
                                        const T_*      B,                                   \
                                        rocblas_stride offsetB,                             \
                                        rocblas_int    ldb,                                 \
                                        rocblas_stride strideB,                             \
                                        const T_*      beta,                                \
                                        T_*            C,                                   \
                                        rocblas_stride offsetC,                             \
                                        rocblas_int    ldc,                                 \
                                        rocblas_stride strideC,                             \
                                        rocblas_int    batch_count);

INST_HEMM_TEMPLATE(rocblas_float_complex)
INST_HEMM_TEMPLATE(rocblas_double_complex)

#undef INST_HEMM_TEMPLATE

#ifdef INST_HEMM_BATCHED_TEMPLATE
#error INST_HEMM_BATCHED_TEMPLATE already defined
#endif

#define INST_HEMM_BATCHED_TEMPLATE(T_)                                                             \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status rocblas_internal_hemm_batched_template<T_> \
                                                (rocblas_handle handle,                             \
                                                rocblas_side    side,                               \
                                                rocblas_fill    uplo,                               \
                                                rocblas_int     m,                                  \
                                                rocblas_int     n,                                  \
                                                const T_*       alpha,                              \
                                                const T_* const* A,                                 \
                                                rocblas_stride  offsetA,                            \
                                                rocblas_int     lda,                                \
                                                rocblas_stride  strideA,                            \
                                                const T_* const* B,                                 \
                                                rocblas_stride  offsetB,                            \
                                                rocblas_int     ldb,                                \
                                                rocblas_stride  strideB,                            \
                                                const T_*       beta,                               \
                                                T_* const*      C,                                  \
                                                rocblas_stride  offsetC,                            \
                                                rocblas_int     ldc,                                \
                                                rocblas_stride  strideC,                            \
                                                rocblas_int     batch_count);

INST_HEMM_BATCHED_TEMPLATE(rocblas_float_complex)
INST_HEMM_BATCHED_TEMPLATE(rocblas_double_complex)

#undef INST_HEMM_BATCHED_TEMPLATE

#ifdef INST_SYMM_HEMM_LAUNCHER
#error INST_SYMM_HEMM_LAUNCHER already defined
#endif

#define INST_SYMM_HEMM_LAUNCHER(HERM_, T_) \
    template rocblas_status rocblas_internal_symm_hemm_launcher<HERM_, T_> \
                                             (rocblas_handle handle,       \
                                              rocblas_side   side,         \
                                              rocblas_fill   uplo,         \
                                              rocblas_int        m,        \
                                              rocblas_int        n,        \
                                              const T_*       alpha,        \
                                              const T_*       A,            \
                                              rocblas_stride offsetA,      \
                                              int64_t        lda,          \
                                              rocblas_stride strideA,      \
                                              const T_*       B,            \
                                              rocblas_stride offsetB,      \
                                              int64_t        ldb,          \
                                              rocblas_stride strideB,      \
                                              const T_*       beta,         \
                                              T_*             C,            \
                                              rocblas_stride offsetC,      \
                                              int64_t        ldc,          \
                                              rocblas_stride strideC,      \
                                              rocblas_int        batch_count);
INST_SYMM_HEMM_LAUNCHER(false, float)
INST_SYMM_HEMM_LAUNCHER(false, double)
INST_SYMM_HEMM_LAUNCHER(false, rocblas_float_complex)
INST_SYMM_HEMM_LAUNCHER(false, rocblas_double_complex)
INST_SYMM_HEMM_LAUNCHER(true, rocblas_float_complex)
INST_SYMM_HEMM_LAUNCHER(true, rocblas_double_complex)

#undef INST_SYMM_HEMM_LAUNCHER

#ifdef INST_SYMM_HEMM_BATCHED_LAUNCHER
#error INST_SYMM_HEMM_BATCHED_LAUNCHER already defined
#endif

#define INST_SYMM_HEMM_BATCHED_LAUNCHER(HERM_, T_)                                 \
    template rocblas_status rocblas_internal_symm_hemm_batched_launcher<HERM_, T_> \
                                             (rocblas_handle   handle,             \
                                              rocblas_side     side,               \
                                              rocblas_fill     uplo,               \
                                              rocblas_int      m,                  \
                                              rocblas_int      n,                  \
                                              const T_*        alpha,              \
                                              const T_* const* A,                  \
                                              rocblas_stride   offsetA,            \
                                              int64_t          lda,                \
                                              rocblas_stride   strideA,            \
                                              const T_* const* B,                  \
                                              rocblas_stride   offsetB,            \
                                              int64_t          ldb,                \
                                              rocblas_stride   strideB,            \
                                              const T_*        beta,               \
                                              T_* const*       C,                  \
                                              rocblas_stride   offsetC,            \
                                              int64_t          ldc,                \
                                              rocblas_stride   strideC,            \
                                              rocblas_int      batch_count);
INST_SYMM_HEMM_BATCHED_LAUNCHER(false, float)
INST_SYMM_HEMM_BATCHED_LAUNCHER(false, double)
INST_SYMM_HEMM_BATCHED_LAUNCHER(false, rocblas_float_complex)
INST_SYMM_HEMM_BATCHED_LAUNCHER(false, rocblas_double_complex)
INST_SYMM_HEMM_BATCHED_LAUNCHER(true, rocblas_float_complex)
INST_SYMM_HEMM_BATCHED_LAUNCHER(true, rocblas_double_complex)

#undef INST_SYMM_HEMM_BATCHED_LAUNCHER

#ifdef INSTANTIATE_HEMM_SYMM_NUMERICS
#error INSTANTIATE_HEMM_SYMM_NUMERICS already defined
#endif

#define INSTANTIATE_HEMM_SYMM_NUMERICS(HERM_, TConstPtr_, TPtr_)                        \
template rocblas_status rocblas_hemm_symm_check_numerics                                \
                                  <HERM_, TConstPtr_, TPtr_>                            \
                                  (const char*       function_name,                     \
                                   rocblas_handle handle,                               \
                                   rocblas_side   side,                                 \
                                   rocblas_fill   uplo,                                 \
                                   int64_t    m,                                    \
                                   int64_t    n,                                    \
                                   TConstPtr_     A,                                    \
                                   int64_t    lda,                                  \
                                   rocblas_stride strideA,                              \
                                   TConstPtr_     B,                                    \
                                   int64_t    ldb,                                  \
                                   rocblas_stride strideB,                              \
                                   TPtr_          C,                                    \
                                   int64_t    ldc,                                  \
                                   rocblas_stride strideC,                              \
                                   int64_t    batch_count,                          \
                                   const int      check_numerics,                       \
                                   bool           is_input);

// instantiate for rocblas_Xhemm_Xsymm and rocblas_Xhemm_Xsymm_strided_batched
INSTANTIATE_HEMM_SYMM_NUMERICS(false, float const*, float*)
INSTANTIATE_HEMM_SYMM_NUMERICS(false, double const*, double*)
INSTANTIATE_HEMM_SYMM_NUMERICS(false,  rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_HEMM_SYMM_NUMERICS( true,  rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_HEMM_SYMM_NUMERICS(false, rocblas_double_complex const*, rocblas_double_complex*)
INSTANTIATE_HEMM_SYMM_NUMERICS( true, rocblas_double_complex const*, rocblas_double_complex*)

// instantiate for rocblas_Xhemm_Xsymm_batched
INSTANTIATE_HEMM_SYMM_NUMERICS(false, float const* const*, float* const*)
INSTANTIATE_HEMM_SYMM_NUMERICS(false, double const* const*, double* const*)
INSTANTIATE_HEMM_SYMM_NUMERICS(false,  rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_HEMM_SYMM_NUMERICS( true,  rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_HEMM_SYMM_NUMERICS(false, rocblas_double_complex const* const*, rocblas_double_complex* const*)
INSTANTIATE_HEMM_SYMM_NUMERICS( true, rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_HEMM_SYMM_NUMERICS

#undef INSTANTIATE_SYMM_TEMPLATE

#undef SSYMM_MIN_NB
#undef DSYMM_MIN_NB
#undef CSYMM_MIN_NB
#undef ZSYMM_MIN_NB

#undef SSYMM_BATCHED_MIN_NB
#undef DSYMM_BATCHED_MIN_NB
#undef CSYMM_BATCHED_MIN_NB
#undef ZSYMM_BATCHED_MIN_NB


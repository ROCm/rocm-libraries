/* ************************************************************************
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "check_numerics_matrix.hpp"
#include "check_numerics_vector.hpp"
#include "device_macros.hpp"
#include "rocblas_gbmv.hpp"

// uses shuffle reductions
#include "../blas1/rocblas_reduction.hpp"

/**
  *  Two kernels to handle all gbmv cases (transpose & conjugate, normal).
  */

template <int WARP, int DIM_Y, typename T>
__forceinline__ __device__ void rocblas_gbmvn_kernel_calc(rocblas_int m,
                                                          rocblas_int n,
                                                          rocblas_int kl,
                                                          rocblas_int ku,
                                                          T           alpha,
                                                          const T*    A,
                                                          int64_t     lda,
                                                          const T*    x,
                                                          int64_t     incx,
                                                          T           beta,
                                                          T*          y,
                                                          int64_t     incy)
{
    // With the banded format, each diagonal,
    // (from bottom left to top right) is like a row in a normal
    // matrix, so the blocks are "tilted" to the right.

    // y thread specifies row to process
    rocblas_int row = blockIdx.x * DIM_Y + threadIdx.y;
    if(row >= m)
        return;

    T res_A = T(0);

    if(alpha)
    {
        int bands_minus_1 = kl + ku;

        int brow = row + ku;
        if(brow > bands_minus_1)
            brow = bands_minus_1;

        int bcol = row - kl;
        if(bcol < 0)
            bcol = 0;

        // witin warp x threads compute a row dot product so move to adjacent band (unbanded matrix next column)
        brow -= threadIdx.x;
        bcol += threadIdx.x;

        // accumulate all bands partial results
        for(; brow >= 0; brow -= WARP, bcol += WARP)
        {
            if(bcol < n)
                res_A += A[brow + bcol * lda] * x[bcol * incx];
        }
        __syncthreads();

        res_A = rocblas_wavefront_reduce<WARP>(res_A);
        res_A *= alpha;
    }

    if(threadIdx.x == 0)
    {
        if(beta != 0)
            y[row * incy] = res_A + beta * y[row * incy];
        else
            y[row * incy] = res_A;
    }
}

/**
  * For the (conjugate-)transpose case. Iterates through each diagonal
  *  and creates partial sums for each tx.
  *  The conjugate basically switches A from upper -> lower or lower -> upper
  *  triangular matrix. Since A is banded format, the indexing changes, and we
  *  basically just iterate down columns.
  */
template <int WARP, int DIM_Y, typename T>
__forceinline__ __device__ void rocblas_gbmvt_kernel_calc(rocblas_operation transA,
                                                          rocblas_int       m,
                                                          rocblas_int       n,
                                                          rocblas_int       kl,
                                                          rocblas_int       ku,
                                                          T                 alpha,
                                                          const T*          A,
                                                          int64_t           lda,
                                                          const T*          x,
                                                          int64_t           incx,
                                                          T                 beta,
                                                          T*                y,
                                                          int64_t           incy)
{
    rocblas_int col = blockIdx.x * DIM_Y + threadIdx.y;
    if(col >= n)
        return;

    // With the banded format each diagonal
    // (from bottom left to top right) is like a row in a normal
    // matrix, so the blocks are "tilted" to the right.

    T res_A(0);

    if(alpha)
    {
        bool is_conj = transA == rocblas_operation_conjugate_transpose;

        // We have to convert banded to unbanded

        int tx = threadIdx.x;

        int nbands = kl + ku + 1;

        A += col * lda;

        // create WARP number of partial results
        for(int row = tx; row < nbands; row += WARP)
        {
            int ku_minus_row = ku - row;

            if(col < (m + ku_minus_row))
            {
                if(row > ku || (row <= ku && col >= ku_minus_row))
                {
                    res_A += ((is_conj ? conj(A[row]) : A[row]) * x[(col - ku_minus_row) * incx]);
                }
            }
        }
        __syncthreads();

        res_A = rocblas_wavefront_reduce<WARP>(res_A);
        res_A *= alpha;
    }

    if(threadIdx.x == 0)
    {
        if(beta != 0)
            y[col * incy] = res_A + beta * y[col * incy];
        else
            y[col * incy] = res_A;
    }
}

/**
  *  Loads pointers (in case of future batched versions) and launches
  *  the actual calculation kernel.
  *
  *  Summary of banded matrices:
  *  Banded matrices consist of the centre diagonal, along with 'kl' sub-diagonals and 'ku' super-diagonals.
  *
  *  These matrices are then compacted into a banded storage format. The main diagonal resides on the (ku+1)'th row,
  *  the the first super-diagonal on the RHS of the ku'th row, the first sub-diagonal on the LHS of the (ku+2)'th row, etc.
  *
  *  Ex: (m = 5, n = 5; ku = 1, kl = 2)
  *
  *  1 2 0 0 0              0 2 2 2 2
  *  3 1 2 0 0              1 1 1 1 1    <- main diag on (ku+1)'th row = (1+1)'th row = 2nd row
  *  4 3 1 2 0     ---->    3 3 3 3 0
  *  0 4 3 1 2              4 4 4 0 0
  *  0 0 4 3 1              0 0 0 0 0
  *
  *  Note: This definition uses 1-indexing as seen above.
  *
  *  The empty parts of these sparse matrices are not to be touched. As can be seen, the column
  *  of each element is preserved in the compaction, and the diagonals are "pushed" upwards and
  *  reside on the same row as the other elements of the same diagonal.
  */
template <int WARP, int DIM_Y, typename TStruct, typename V, typename W>
ROCBLAS_KERNEL(WARP* DIM_Y)
rocblas_gbmvn_kernel(bool           host_ptr_mode,
                     rocblas_int    m,
                     rocblas_int    n,
                     rocblas_int    kl,
                     rocblas_int    ku,
                     TStruct        alpha_device_host,
                     V              Aa,
                     rocblas_stride shifta,
                     int64_t        lda,
                     rocblas_stride strideA,
                     V              xa,
                     rocblas_stride shiftx,
                     int64_t        incx,
                     rocblas_stride stridex,
                     TStruct        beta_device_host,
                     W              ya,
                     rocblas_stride shifty,
                     int64_t        incy,
                     rocblas_stride stridey,
                     rocblas_int    batch_count)
{
    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        const auto alpha = host_ptr_mode ? alpha_device_host.value
                                         : load_scalar(alpha_device_host.ptr, batch, 0);
        const auto beta
            = host_ptr_mode ? beta_device_host.value : load_scalar(beta_device_host.ptr, batch, 0);

        if(!alpha && beta == 1)
        {
#if DEVICE_GRID_YZ_16BIT
            continue; //iterate to the next batch in the for loop rather than return.
#else
        return;
#endif
        }

        const auto* A = cond_load_ptr_batch(alpha, Aa, batch, shifta, strideA);
        const auto* x = cond_load_ptr_batch(alpha, xa, batch, shiftx, stridex);

        auto* y = load_ptr_batch(ya, batch, shifty, stridey);

        rocblas_gbmvn_kernel_calc<WARP, DIM_Y>(m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <int DIM_X, int DIM_Y, typename TStruct, typename V, typename W>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_gbmvt_kernel(bool              host_ptr_mode,
                     rocblas_operation transA,
                     rocblas_int       m,
                     rocblas_int       n,
                     rocblas_int       kl,
                     rocblas_int       ku,
                     TStruct           alpha_device_host,
                     V                 Aa,
                     rocblas_stride    shifta,
                     int64_t           lda,
                     rocblas_stride    strideA,
                     V                 xa,
                     rocblas_stride    shiftx,
                     int64_t           incx,
                     rocblas_stride    stridex,
                     TStruct           beta_device_host,
                     W                 ya,
                     rocblas_stride    shifty,
                     int64_t           incy,
                     rocblas_stride    stridey,
                     rocblas_int       batch_count)
{
    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif
        const auto alpha = host_ptr_mode ? alpha_device_host.value
                                         : load_scalar(alpha_device_host.ptr, batch, 0);
        const auto beta
            = host_ptr_mode ? beta_device_host.value : load_scalar(beta_device_host.ptr, batch, 0);

        if(!alpha && beta == 1)
        {
#if DEVICE_GRID_YZ_16BIT
            continue; //iterate to the next batch in the for loop rather than return.
#else
        return;
#endif
        }
        const auto* A = cond_load_ptr_batch(alpha, Aa, batch, shifta, strideA);
        const auto* x = cond_load_ptr_batch(alpha, xa, batch, shiftx, stridex);

        auto* y = load_ptr_batch(ya, batch, shifty, stridey);

        rocblas_gbmvt_kernel_calc<DIM_X, DIM_Y>(
            transA, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

/**
  *  Here, U is either a `const T* const*` or a `const T*`
  *  V is either a `T*` or a `T* const*`
  */
template <typename T, typename U, typename V>
rocblas_status rocblas_internal_gbmv_launcher(rocblas_handle    handle,
                                              rocblas_operation transA,
                                              rocblas_int       m,
                                              rocblas_int       n,
                                              rocblas_int       kl,
                                              rocblas_int       ku,
                                              const T*          alpha,
                                              U                 A,
                                              rocblas_stride    offseta,
                                              int64_t           lda,
                                              rocblas_stride    strideA,
                                              U                 x,
                                              rocblas_stride    offsetx,
                                              int64_t           incx,
                                              rocblas_stride    stridex,
                                              const T*          beta,
                                              V                 y,
                                              rocblas_stride    offsety,
                                              int64_t           incy,
                                              rocblas_stride    stridey,
                                              rocblas_int       batch_count)
{
    // quick return
    if(!m || !n || !batch_count)
        return rocblas_status_success;

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    auto shiftx
        = incx < 0 ? offsetx - ptrdiff_t(incx) * (transA == rocblas_operation_none ? n - 1 : m - 1)
                   : offsetx;
    auto shifty
        = incy < 0 ? offsety - ptrdiff_t(incy) * (transA == rocblas_operation_none ? m - 1 : n - 1)
                   : offsety;

    // The logic is similar to gemv just with modified
    // indices for the banded matrices.

    bool                        host_ptr_mode = handle->pointer_mode == rocblas_pointer_mode_host;
    rocblas_internal_val_ptr<T> alpha_device_host(host_ptr_mode, alpha);
    rocblas_internal_val_ptr<T> beta_device_host(host_ptr_mode, beta);

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        if(!*alpha && *beta == 1)
            return rocblas_status_success;
    }

    int  arch_major = handle->getArchMajor();
    bool is_arch_10_or_11_or_12
        = arch_major == 10 || arch_major == 11 || arch_major == 12 ? true : false;

    int batches = handle->getBatchGridDim((int)batch_count);

#define GBMV_COMMON_ARGS                                                                 \
    m, n, kl, ku, alpha_device_host, A, offseta, lda, strideA, x, shiftx, incx, stridex, \
        beta_device_host, y, shifty, incy, stridey, batch_count

    if(transA == rocblas_operation_none)
    {
        if(is_arch_10_or_11_or_12)
        {
            static constexpr int WARP        = 32; // warp size as using warp reduce for bands
            static constexpr int GBMVN_DIM_Y = 32; // problem sub block
            rocblas_int          blocks      = (m - 1) / (GBMVN_DIM_Y) + 1;
            dim3                 gbmvn_grid(blocks, 1, batches);
            dim3                 gbmvn_threads(WARP, GBMVN_DIM_Y);

            ROCBLAS_LAUNCH_KERNEL((rocblas_gbmvn_kernel<WARP, GBMVN_DIM_Y>),
                                  gbmvn_grid,
                                  gbmvn_threads,
                                  0,
                                  handle->get_stream(),
                                  host_ptr_mode,
                                  GBMV_COMMON_ARGS);
        }
        else
        {
            static constexpr int WARP        = 64;
            static constexpr int GBMVN_DIM_Y = 16;
            rocblas_int          blocks      = (m - 1) / (GBMVN_DIM_Y) + 1;
            dim3                 gbmvn_grid(blocks, 1, batches);
            dim3                 gbmvn_threads(WARP, GBMVN_DIM_Y);

            ROCBLAS_LAUNCH_KERNEL((rocblas_gbmvn_kernel<WARP, GBMVN_DIM_Y>),
                                  gbmvn_grid,
                                  gbmvn_threads,
                                  0,
                                  handle->get_stream(),
                                  host_ptr_mode,
                                  GBMV_COMMON_ARGS);
        }
    }
    else // trans/conj
    {
        if(is_arch_10_or_11_or_12)
        {
            static constexpr int WARP        = 32; // warp size as using warp reduce for bands
            static constexpr int GBMVT_DIM_Y = 32; // problem sub block
            rocblas_int          blocks      = (n - 1) / (GBMVT_DIM_Y) + 1;
            dim3                 gbmvt_grid(blocks, 1, batches);
            dim3                 gbmvt_threads(WARP, GBMVT_DIM_Y);

            ROCBLAS_LAUNCH_KERNEL((rocblas_gbmvt_kernel<WARP, GBMVT_DIM_Y>),
                                  gbmvt_grid,
                                  gbmvt_threads,
                                  0,
                                  handle->get_stream(),
                                  host_ptr_mode,
                                  transA,
                                  GBMV_COMMON_ARGS);
        }
        else
        {
            static constexpr int WARP        = 64;
            static constexpr int GBMVT_DIM_Y = 16;
            rocblas_int          blocks      = (n - 1) / (GBMVT_DIM_Y) + 1;
            dim3                 gbmvt_grid(blocks, 1, batches);
            dim3                 gbmvt_threads(WARP, GBMVT_DIM_Y);

            ROCBLAS_LAUNCH_KERNEL((rocblas_gbmvt_kernel<WARP, GBMVT_DIM_Y>),
                                  gbmvt_grid,
                                  gbmvt_threads,
                                  0,
                                  handle->get_stream(),
                                  host_ptr_mode,
                                  transA,
                                  GBMV_COMMON_ARGS);
        }
    }

#undef GBMV_COMMON_ARGS

    return rocblas_status_success;
}

//TODO :-Add rocblas_check_numerics_gb_matrix_template for checking Matrix `A` which is a General Band matrix
template <typename T, typename U>
rocblas_status rocblas_gbmv_check_numerics(const char*       function_name,
                                           rocblas_handle    handle,
                                           rocblas_operation trans_a,
                                           int64_t           m,
                                           int64_t           n,
                                           T                 A,
                                           rocblas_stride    offset_a,
                                           int64_t           lda,
                                           rocblas_stride    stride_a,
                                           T                 x,
                                           rocblas_stride    offset_x,
                                           int64_t           inc_x,
                                           rocblas_stride    stride_x,
                                           U                 y,
                                           rocblas_stride    offset_y,
                                           int64_t           inc_y,
                                           rocblas_stride    stride_y,
                                           int64_t           batch_count,
                                           const int         check_numerics,
                                           bool              is_input)
{
    if(is_input)
    {
        //TODO :-Add rocblas_check_numerics_gb_matrix_template for checking Matrix `A` which is a General Band matrix

        //Checking trans_a to transpose a vector 'x'
        int64_t        n_x = trans_a == rocblas_operation_none ? n : m;
        rocblas_status check_numerics_status
            = rocblas_internal_check_numerics_vector_template(function_name,
                                                              handle,
                                                              n_x,
                                                              x,
                                                              offset_x,
                                                              inc_x,
                                                              stride_x,
                                                              batch_count,
                                                              check_numerics,
                                                              is_input);
        return check_numerics_status;
    }
    else
    {
        //Checking trans_a to transpose a vector 'y'
        int64_t        n_y = trans_a == rocblas_operation_none ? m : n;
        rocblas_status check_numerics_status
            = rocblas_internal_check_numerics_vector_template(function_name,
                                                              handle,
                                                              n_y,
                                                              y,
                                                              offset_y,
                                                              inc_y,
                                                              stride_y,
                                                              batch_count,
                                                              check_numerics,
                                                              is_input);
        return check_numerics_status;
    }
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files *gbmv*.cpp

#ifdef INST_GBMV_LAUNCHER
#error INST_GBMV_LAUNCHER  already defined
#endif

#define INST_GBMV_LAUNCHER(T_, U_, V_)                                                            \
    template rocblas_status rocblas_internal_gbmv_launcher<T_, U_, V_>(rocblas_handle    handle,  \
                                                                       rocblas_operation transA,  \
                                                                       rocblas_int       m,       \
                                                                       rocblas_int       n,       \
                                                                       rocblas_int       kl,      \
                                                                       rocblas_int       ku,      \
                                                                       const T_*         alpha,   \
                                                                       U_                A,       \
                                                                       rocblas_stride    offseta, \
                                                                       int64_t           lda,     \
                                                                       rocblas_stride    strideA, \
                                                                       U_                x,       \
                                                                       rocblas_stride    offsetx, \
                                                                       int64_t           incx,    \
                                                                       rocblas_stride    stridex, \
                                                                       const T_*         beta,    \
                                                                       V_                y,       \
                                                                       rocblas_stride    offsety, \
                                                                       int64_t           incy,    \
                                                                       rocblas_stride    stridey, \
                                                                       rocblas_int batch_count);

INST_GBMV_LAUNCHER(double, double const* const*, double* const*)
INST_GBMV_LAUNCHER(rocblas_float_complex,
                   rocblas_float_complex const* const*,
                   rocblas_float_complex* const*)
INST_GBMV_LAUNCHER(rocblas_double_complex,
                   rocblas_double_complex const* const*,
                   rocblas_double_complex* const*)
INST_GBMV_LAUNCHER(float, float const*, float*)
INST_GBMV_LAUNCHER(double, double const*, double*)
INST_GBMV_LAUNCHER(rocblas_float_complex, rocblas_float_complex const*, rocblas_float_complex*)
INST_GBMV_LAUNCHER(rocblas_double_complex, rocblas_double_complex const*, rocblas_double_complex*)
INST_GBMV_LAUNCHER(float, float const* const*, float* const*)

#undef INST_GBMV_LAUNCHER

#ifdef INST_GBMV_NUMERICS
#error INST_GBMV_NUMERICS  already defined
#endif

#define INST_GBMV_NUMERICS(T_, U_)                                                                \
    template rocblas_status rocblas_gbmv_check_numerics<T_, U_>(const char*       function_name,  \
                                                                rocblas_handle    handle,         \
                                                                rocblas_operation trans_a,        \
                                                                int64_t           m,              \
                                                                int64_t           n,              \
                                                                T_                A,              \
                                                                rocblas_stride    offset_a,       \
                                                                int64_t           lda,            \
                                                                rocblas_stride    stride_a,       \
                                                                T_                x,              \
                                                                rocblas_stride    offset_x,       \
                                                                int64_t           inc_x,          \
                                                                rocblas_stride    stride_x,       \
                                                                U_                y,              \
                                                                rocblas_stride    offset_y,       \
                                                                int64_t           inc_y,          \
                                                                rocblas_stride    stride_y,       \
                                                                int64_t           batch_count,    \
                                                                const int         check_numerics, \
                                                                bool              is_input);

INST_GBMV_NUMERICS(float const*, float*)
INST_GBMV_NUMERICS(double const*, double*)
INST_GBMV_NUMERICS(rocblas_float_complex const*, rocblas_float_complex*)
INST_GBMV_NUMERICS(rocblas_double_complex const*, rocblas_double_complex*)
INST_GBMV_NUMERICS(float const* const*, float* const*)
INST_GBMV_NUMERICS(double const* const*, double* const*)
INST_GBMV_NUMERICS(rocblas_float_complex const* const*, rocblas_float_complex* const*)
INST_GBMV_NUMERICS(rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INST_GBMV_NUMERICS

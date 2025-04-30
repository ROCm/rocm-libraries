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

#include "../blas1/rocblas_copy.hpp"
#include "../blas1/rocblas_copy_kernels.hpp"
#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "rocblas_tbmv.hpp"

/**
  *  Helper for the non-transpose case. Iterates through each diagonal
  *  and creates partial sums for each ty.
  */
template <rocblas_int DIM_Y, typename T>
__device__ T rocblas_tbmvn_kernel_helper(rocblas_int ty,
                                         rocblas_int ind,
                                         bool        is_upper,
                                         bool        is_unit_diag,
                                         rocblas_int n,
                                         rocblas_int k,
                                         const T*    A,
                                         int64_t     lda,
                                         const T*    w_x_copy)
{
    T           res_A = 0.0;
    rocblas_int col;

    // Since the column is consistent, we can iterate up the diagonal
    // ty defines the column of banded & regular matrix
    for(col = ty; col < n; col += DIM_Y)
    {
        // We have to convert ind to banded matrix row
        rocblas_int row = is_upper ? ind + (k - col) : ind - col;

        if(ind < n)
        {
            // Regular case, simply multiply
            if(row < k && row > 0)
            {
                res_A += (A[row + col * size_t(lda)] * w_x_copy[col]);
            }
            else if(row == 0)
            {
                // If main diagonal && diag, don't reference matrix, assume 1.
                if(is_unit_diag && (!is_upper || k == 0 && is_upper))
                    res_A += w_x_copy[col];
                else
                    res_A += (A[row + col * size_t(lda)] * w_x_copy[col]);
            }
            else if(row == k)
            {
                // If diag, don't reference matrix, assume 1.
                if(is_unit_diag && is_upper)
                    res_A += w_x_copy[col];
                else
                    res_A += (A[row + col * size_t(lda)] * w_x_copy[col]);
            }
        }
    }
    return res_A;
}

/**
  *  Helper for the (conjugate-)transpose case. Iterates through each diagonal
  *  and creates partial sums for each ty.
  *  The conjugate basically switches A from upper -> lower or lower -> upper
  *  triangular matrix. Since A is compressed, the indexing changes, and we
  *  basically just iterate down columns.
  */
template <rocblas_int DIM_Y, typename T>
__device__ T rocblas_tbmvt_kernel_helper(bool        CONJ,
                                         rocblas_int ty,
                                         rocblas_int ind,
                                         bool        is_upper,
                                         bool        is_unit_diag,
                                         rocblas_int n,
                                         rocblas_int k,
                                         const T*    A,
                                         int64_t     lda,
                                         const T*    w_x_copy)
{
    T           res_A = 0.0;
    rocblas_int row;

    // for transpose case, ty defines the row
    for(row = ty; row < lda && row <= k; row += DIM_Y)
    {
        // We have to convert ind to banded matrix row
        rocblas_int col = ind;

        if(col < n)
        {
            if(is_upper)
            {
                // Regular case
                rocblas_int min_row = k - col;
                // cppcheck-suppress knownConditionTrueFalse
                if(row < k && row >= k - col && row != k)
                {
                    res_A += ((CONJ ? conj(A[row + col * size_t(lda)]) : A[row + col * size_t(lda)])
                              * w_x_copy[row - min_row]);
                }
                else if(row == k)
                {
                    // if main diagonal && diag then don't reference A, assume 1.
                    if(is_unit_diag)
                        res_A += w_x_copy[row - min_row];
                    else
                        res_A += ((CONJ ? conj(A[row + col * size_t(lda)])
                                        : A[row + col * size_t(lda)])
                                  * w_x_copy[row - min_row]);
                }
            }
            else
            {
                if(row <= k && row <= n - 1 - col && row > 0)
                {
                    res_A += ((CONJ ? conj(A[row + col * size_t(lda)]) : A[row + col * size_t(lda)])
                              * w_x_copy[row + col]);
                }
                else if(row == 0)
                {
                    if(is_unit_diag)
                        res_A += w_x_copy[row + col];
                    else
                        res_A += ((CONJ ? conj(A[row + col * size_t(lda)])
                                        : A[row + col * size_t(lda)])
                                  * w_x_copy[row + col]);
                }
            }
        }
    }
    return res_A;
}

/**
  *  A combined kernel to handle all tbmv cases (transpose, conjugate, normal).
  */
template <rocblas_int DIM_X, rocblas_int DIM_Y, typename T>
ROCBLAS_KERNEL_ILF void rocblas_tbmvx_kernel_calc(rocblas_operation transA,
                                                  bool              is_upper,
                                                  bool              is_unit_diag,
                                                  rocblas_int       n,
                                                  rocblas_int       k,
                                                  const T*          A,
                                                  int64_t           lda,
                                                  const T*          w_x_copy,
                                                  T*                x,
                                                  int64_t           incx)
{
    rocblas_int thread_id = threadIdx.x + threadIdx.y * blockDim.x;

    // threads are all configurated locally
    // Create "tilted" blocks. With the compaction, each diagonal,
    // (from top right to bottom left) is like a row in a normal
    // matrix, so the blocks are "tilted" to the right.
    rocblas_int tx = thread_id % DIM_X;
    rocblas_int ty = thread_id / DIM_X;

    rocblas_int ind = blockIdx.x * DIM_X + tx;

    __shared__ T sdata[DIM_X * DIM_Y];

    T res_A = 0.0;
    // Indexing is different for transpose/non-transpose case. To keep it clean
    // it's separated in two helper functions. They could potentially be combined
    // if more elegant logic is used.
    if(transA == rocblas_operation_none)
    {
        res_A = rocblas_tbmvn_kernel_helper<DIM_Y>(
            ty, ind, is_upper, is_unit_diag, n, k, A, lda, w_x_copy);
    }
    else
    {
        bool CONJ = transA == rocblas_operation_conjugate_transpose;
        res_A     = rocblas_tbmvt_kernel_helper<DIM_Y>(
            CONJ, ty, ind, is_upper, is_unit_diag, n, k, A, lda, w_x_copy);
    }
    // Store partial sums for the diagonal
    sdata[tx + ty * DIM_X] = res_A;
    __syncthreads();

    thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    ind       = blockIdx.x * DIM_X + thread_id;
    if(thread_id < DIM_X && ind < n)
    {
        // Add the partial sums of each diagonal and store
        for(rocblas_int i = 1; i < DIM_Y; i++)
        {
            sdata[thread_id] += sdata[thread_id + DIM_X * i];
        }

        // Update x.
        x[ind * int64_t(incx)] = (sdata[thread_id]);
    }
}

/**
  *  Loads pointers (in case of future batched versions) and launches
  *  the actual calculation kernel.
  *
  *  Summary of banded matrices:
  *  Two types of banded matrices exist, upper and lower. These matrices consist of
  *  the centre diagonal, along with 'k' sub-diagonals (if lower) or super-diagonals (if upper).
  *
  *  These matrices are then compacted into a banded storage format. For upper-triangular,
  *  the k'th super-diagonal resides on the right-hand side of the first row, k-1th on the second,
  *  etc, with the main diagonal on the k'th row.
  *
  *  Ex: (upper; n = 5; k = 2)
  *
  *  1 6 9 0 0              0 0 9 8 7
  *  0 2 7 8 0              0 6 7 8 9
  *  0 0 3 8 7     ---->    1 2 3 4 5
  *  0 0 0 4 9              0 0 0 0 0
  *  0 0 0 0 5              0 0 0 0 0
  *
  *  For lower-triangular, the main diagonal resides on the 0'th row, working up to the k'th
  *  sub-diagonal residing on the left-hand side of the k'th row.
  *
  *  Ex: (lower; n = 5; k = 2)
  *
  *  1 0 0 0 0              1 2 3 4 5
  *  6 2 0 0 0              6 7 8 9 0
  *  9 7 3 0 0     ---->    9 8 7 0 0
  *  0 8 8 4 0              0 0 0 0 0
  *  0 0 7 9 5              0 0 0 0 0
  *
  *  The empty parts of these sparse matrices are not to be touched. As can be seen, the column
  *  of each element is preserved in the compaction, and the diagonals are "pushed" upwards and
  *  reside on the same row as the other elements of the same diagonal.
  */
template <rocblas_int DIM_X, rocblas_int DIM_Y, typename U, typename V>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_tbmvx_kernel(rocblas_operation transA,
                     bool              is_upper,
                     bool              is_unit_diag,
                     rocblas_int       n,
                     rocblas_int       k,
                     U                 Aa,
                     rocblas_stride    shifta,
                     int64_t           lda,
                     rocblas_stride    strideA,
                     U                 w_xa_copy,
                     V                 xa,
                     rocblas_stride    shiftx,
                     int64_t           incx,
                     rocblas_stride    stridex,
                     rocblas_int       batch_count)
{
    rocblas_int num_threads = blockDim.x * blockDim.y * blockDim.z;
    if(DIM_X * DIM_Y != num_threads)
        return; // need to launch exactly the same number of threads as template parameters indicate

    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        const auto* A        = load_ptr_batch(Aa, batch, shifta, strideA);
        const auto* w_x_copy = load_ptr_batch(w_xa_copy, batch, 0, n);
        auto*       x        = load_ptr_batch(xa, batch, shiftx, stridex);

        rocblas_tbmvx_kernel_calc<DIM_X, DIM_Y>(
            transA, is_upper, is_unit_diag, n, k, A, lda, w_x_copy, x, incx);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

/**
  *  First, makes a copy of 'x', then uses a modified gemv algorithm
  *  to perform x := transA(A) * w_x_copy
  *  w_x_copy is workspace memory and should be of size sizeof(T) * n bytes * batch_count.
  *
  *  Here, TConstPtr is either a `const T* const*` or a `const T*`
  *  TPtr is either a `T*` or a `T* const*`
  */
template <typename TConstPtr, typename TPtr>
rocblas_status rocblas_internal_tbmv_launcher(rocblas_handle    handle,
                                              rocblas_fill      uplo,
                                              rocblas_operation transA,
                                              rocblas_diagonal  diag,
                                              rocblas_int       n,
                                              rocblas_int       k,
                                              TConstPtr         A,
                                              rocblas_stride    offseta,
                                              int64_t           lda,
                                              rocblas_stride    strideA,
                                              TPtr              x,
                                              rocblas_stride    offsetx,
                                              int64_t           incx,
                                              rocblas_stride    stridex,
                                              rocblas_int       batch_count,
                                              TPtr              w_x_copy)
{
    // quick return
    if(!n || !batch_count)
        return rocblas_status_success;

    // First we make a copy of x so we can avoid RAW race conditions in the kernel
    rocblas_status copy_status;
    if(incx > c_i32_max)
        copy_status
            = rocblas_internal_copy_launcher<int64_t, ROCBLAS_COPY_NB>(handle,
                                                                       int64_t(n),
                                                                       x,
                                                                       offsetx,
                                                                       incx,
                                                                       stridex,
                                                                       w_x_copy,
                                                                       0,
                                                                       int64_t(1),
                                                                       n,
                                                                       int64_t(batch_count));
    else
        copy_status = rocblas_internal_copy_template<rocblas_int>(
            handle, n, x, offsetx, incx, stridex, w_x_copy, 0, 1, n, batch_count);

    if(copy_status != rocblas_status_success)
        return copy_status;

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    ptrdiff_t shiftx = incx < 0 ? offsetx - ptrdiff_t(incx) * (n - 1) : offsetx;

    int batches = handle->getBatchGridDim((int)batch_count);

    // (gemv) TBMVX_DIM_Y must be at least 4, 8 * 8 is very slow only 40Gflop/s
    static constexpr int TBMVX_DIM_X = 64;
    static constexpr int TBMVX_DIM_Y = 16;
    rocblas_int          blocks      = (n - 1) / (TBMVX_DIM_X) + 1;
    dim3                 tbmvx_grid(blocks, 1, batches);
    dim3                 tbmvx_threads(TBMVX_DIM_X, TBMVX_DIM_Y);

    // Launch a modified gemv kernel. The logic is similar to gemv just with modified
    // indices for the banded matrices.
    ROCBLAS_LAUNCH_KERNEL((rocblas_tbmvx_kernel<TBMVX_DIM_X, TBMVX_DIM_Y>),
                          tbmvx_grid,
                          tbmvx_threads,
                          0,
                          handle->get_stream(),
                          transA,
                          uplo == rocblas_fill_upper,
                          diag == rocblas_diagonal_unit,
                          n,
                          k,
                          A,
                          offseta,
                          lda,
                          strideA,
                          (TConstPtr)w_x_copy,
                          x,
                          shiftx,
                          incx,
                          stridex,
                          batch_count);

    return rocblas_status_success;
}

//TODO :-Add rocblas_check_numerics_tb_matrix_template for checking Matrix `A` which is a Triangular Band Matrix
template <typename T, typename U>
rocblas_status rocblas_tbmv_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           int64_t        n,
                                           T              A,
                                           rocblas_stride offset_a,
                                           int64_t        lda,
                                           rocblas_stride stride_a,
                                           U              x,
                                           rocblas_stride offset_x,
                                           int64_t        inc_x,
                                           rocblas_stride stride_x,
                                           int64_t        batch_count,
                                           const int      check_numerics,
                                           bool           is_input)
{
    rocblas_status check_numerics_status
        = rocblas_internal_check_numerics_vector_template(function_name,
                                                          handle,
                                                          n,
                                                          x,
                                                          offset_x,
                                                          inc_x,
                                                          stride_x,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);

    return check_numerics_status;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files *tbmv*.cpp

// clang-format off

#ifdef INSTANTIATE_TBMV_LAUNCHER
#error INSTANTIATE_TBMV_LAUNCHER  already defined
#endif

#define INSTANTIATE_TBMV_LAUNCHER(U_, V_)                           \
template rocblas_status rocblas_internal_tbmv_launcher<U_, V_>      \
                                    (rocblas_handle    handle,      \
                                     rocblas_fill      uplo,        \
                                     rocblas_operation transA,      \
                                     rocblas_diagonal  diag,        \
                                     rocblas_int       n,           \
                                     rocblas_int       k,           \
                                     U_                A,           \
                                     rocblas_stride    offseta,     \
                                     int64_t           lda,         \
                                     rocblas_stride    strideA,     \
                                     V_                x,           \
                                     rocblas_stride    offsetx,     \
                                     int64_t           incx,        \
                                     rocblas_stride    stridex,     \
                                     rocblas_int       batch_count, \
                                     V_                w_x_copy);

INSTANTIATE_TBMV_LAUNCHER(float const*, float*)
INSTANTIATE_TBMV_LAUNCHER(double const*, double*)
INSTANTIATE_TBMV_LAUNCHER(rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_TBMV_LAUNCHER(rocblas_double_complex const*, rocblas_double_complex*)
INSTANTIATE_TBMV_LAUNCHER(float const* const*, float* const*)
INSTANTIATE_TBMV_LAUNCHER(double const* const*, double* const*)
INSTANTIATE_TBMV_LAUNCHER(rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_TBMV_LAUNCHER(rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_TBMV_LAUNCHER

#define INSTANTIATE_TBMV_NUMERICS(T, U_)                                  \
template rocblas_status rocblas_tbmv_check_numerics<T, U_>                \
                                          (const char*    function_name,  \
                                           rocblas_handle handle,         \
                                           int64_t        n,              \
                                           T              A,              \
                                           rocblas_stride offset_a,       \
                                           int64_t        lda,            \
                                           rocblas_stride stride_a,       \
                                           U_             x,              \
                                           rocblas_stride offset_x,       \
                                           int64_t        inc_x,          \
                                           rocblas_stride stride_x,       \
                                           int64_t        batch_count,    \
                                           const int      check_numerics, \
                                           bool           is_input);

INSTANTIATE_TBMV_NUMERICS(float const*, float*)
INSTANTIATE_TBMV_NUMERICS(double const*, double*)
INSTANTIATE_TBMV_NUMERICS(rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_TBMV_NUMERICS(rocblas_double_complex const*, rocblas_double_complex*)
INSTANTIATE_TBMV_NUMERICS(float const* const*, float* const*)
INSTANTIATE_TBMV_NUMERICS(double const* const*, double* const*)
INSTANTIATE_TBMV_NUMERICS(rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_TBMV_NUMERICS(rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_TBMV_NUMERICS
// clang-format on

/* ************************************************************************
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining A copy
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
#include "../blas1/rocblas_reduction.hpp"
#include "device_macros.hpp"
#include "rocblas.h"
#include "rocblas_trmv.hpp"
#include <cstddef>

template <rocblas_int DIM_X, rocblas_int DIM_Y, bool LOWER, bool UNIT, typename T>
ROCBLAS_KERNEL_ILF void rocblas_trmvn_kernel_calc(
    rocblas_int n, const T* A, int64_t lda, const T* x, int64_t incx, T* workspace)
{
    rocblas_int tid = threadIdx.x + threadIdx.y * blockDim.x;

    // tx corresponds to row in block, good for memory coalescing
    // ty corresponds to column
    rocblas_int tx = threadIdx.x;
    rocblas_int ty = threadIdx.y;

    rocblas_int row = blockIdx.x * DIM_X + tx;

    __shared__ T sdata[DIM_X * DIM_Y];
    T            res_A = 0;

    // handle diagonal separately
    if(ty == 0 && row < n)
    {
        if(UNIT)
            res_A = x[row * incx];
        else
            res_A = A[row + row * lda] * x[row * incx];
    }

    // multiply and sum across columns
    for(rocblas_int col = ty; col < n; col += DIM_Y)
    {
        if(row < n && ((!LOWER && col > row) || (LOWER && col < row)))
            res_A += A[row + col * lda] * x[col * incx];
    }

    // move partial sum to shared memory to sum further
    sdata[tx + ty * DIM_X] = res_A;

    __syncthreads();

    if(tid < DIM_X)
    {
        // sum DIM_Y elements to get result
        for(rocblas_int i = 1; i < DIM_Y; i++)
            sdata[tid] += sdata[tid + DIM_X * i];

        if(row < n)
            workspace[row] = sdata[tid];
    }
}

template <rocblas_int NB, bool LOWER, bool CONJ, bool UNIT, typename T>
ROCBLAS_KERNEL_ILF void rocblas_trmvt_kernel_calc(
    rocblas_int n, const T* A, int64_t lda, const T* x, int64_t incx, T* workspace)
{
    // tx is assigned to rows
    rocblas_int tx  = threadIdx.x;
    rocblas_int col = blockIdx.x;

    if(tx < n)
        A += tx;
    A += col * lda;

    T res = 0;

    // handle diagonal separately
    if(tx == 0)
    {
        if(UNIT)
            res += x[col * incx];
        else
            res += (CONJ ? conj(A[col]) : A[col]) * x[col * incx];
    }

    for(rocblas_int i = 0; tx + i < n; i += NB)
    {
        if((tx + i > col && LOWER) || (tx + i < col && !LOWER))
            res += (CONJ ? conj(A[i]) : A[i]) * x[(tx + i) * incx];
    }

    res = rocblas_dot_block_reduce<NB>(res);

    if(tx == 0)
    {
        workspace[col] = res;
    }
}

template <rocblas_int DIM_X,
          rocblas_int DIM_Y,
          bool        LOWER,
          bool        UNIT,
          typename TConstPtr,
          typename TPtr,
          typename TWork>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_trmvn_kernel(rocblas_int    n,
                     TConstPtr      A,
                     rocblas_stride shifta,
                     int64_t        lda,
                     rocblas_stride stride_A,
                     TPtr           x,
                     rocblas_stride shift_x,
                     int64_t        incx,
                     rocblas_stride stride_x,
                     TWork          workspace,
                     rocblas_stride stride_w,
                     rocblas_int    batch_count)
{
    static constexpr ptrdiff_t shiftw = 0;

    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        rocblas_trmvn_kernel_calc<DIM_X, DIM_Y, LOWER, UNIT>(
            n,
            load_ptr_batch(A, batch, shifta, stride_A),
            lda,
            load_ptr_batch(x, batch, shift_x, stride_x),
            incx,
            load_ptr_batch(workspace, batch, shiftw, stride_w));

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <rocblas_int NB,
          bool        LOWER,
          bool        CONJ,
          bool        UNIT,
          typename TConstPtr,
          typename TPtr,
          typename TWork>
ROCBLAS_KERNEL(NB)
rocblas_trmvt_kernel(rocblas_int    n,
                     TConstPtr      A,
                     ptrdiff_t      shifta,
                     int64_t        lda,
                     rocblas_stride stride_A,
                     TPtr           x,
                     ptrdiff_t      shift_x,
                     int64_t        incx,
                     rocblas_stride stride_x,
                     TWork          workspace,
                     rocblas_stride stride_w,
                     rocblas_int    batch_count)
{
    static constexpr ptrdiff_t shiftw = 0;

    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        rocblas_trmvt_kernel_calc<NB, LOWER, CONJ, UNIT>(
            n,
            load_ptr_batch(A, batch, shifta, stride_A),
            lda,
            load_ptr_batch(x, batch, shift_x, stride_x),
            incx,
            load_ptr_batch(workspace, batch, shiftw, stride_w));

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <typename TConstPtr, typename TPtr, typename TWork>
rocblas_status rocblas_internal_trmv_launcher(rocblas_handle    handle,
                                              rocblas_fill      uplo,
                                              rocblas_operation transA,
                                              rocblas_diagonal  diag,
                                              rocblas_int       n,
                                              TConstPtr         A,
                                              rocblas_stride    offset_A,
                                              int64_t           lda,
                                              rocblas_stride    stride_A,
                                              TPtr              x,
                                              rocblas_stride    offset_x,
                                              int64_t           incx,
                                              rocblas_stride    stride_x,
                                              TWork             workspace,
                                              rocblas_stride    stride_w,
                                              rocblas_int       batch_count)
{
    //
    // quick return
    //
    if(!n || !batch_count)
    {
        return rocblas_status_success;
    }

    hipStream_t rocblas_stream = handle->get_stream();

    int64_t shift_x = incx < 0 ? offset_x + incx * (1 - n) : offset_x;

    int batches = handle->getBatchGridDim((int)batch_count);

    static constexpr rocblas_int NB          = ROCBLAS_TRMV_NB;
    constexpr int                TRMVN_DIM_X = 64;
    constexpr int                TRMVN_DIM_Y = 16;

    dim3 trmvn_grid((n - 1) / TRMVN_DIM_X + 1, 1, batches);
    dim3 trmvn_threads(TRMVN_DIM_X, TRMVN_DIM_Y);

    dim3 trmvt_grid(n, 1, batches);
    dim3 trmvt_threads(NB);

#define TRMV_TEMPLATE_PARAMS                                                                 \
    0, rocblas_stream, n, A, offset_A, lda, stride_A, x, shift_x, incx, stride_x, workspace, \
        stride_w, batch_count

    if(uplo == rocblas_fill_upper)
    {
        if(diag == rocblas_diagonal_unit)
        {
            if(transA == rocblas_operation_none)
                ROCBLAS_LAUNCH_KERNEL((rocblas_trmvn_kernel<TRMVN_DIM_X, TRMVN_DIM_Y, false, true>),
                                      trmvn_grid,
                                      trmvn_threads,
                                      TRMV_TEMPLATE_PARAMS);
            else if(transA == rocblas_operation_transpose)
                ROCBLAS_LAUNCH_KERNEL((rocblas_trmvt_kernel<NB, false, false, true>),
                                      trmvt_grid,
                                      trmvt_threads,
                                      TRMV_TEMPLATE_PARAMS);
            else if(transA == rocblas_operation_conjugate_transpose)
                ROCBLAS_LAUNCH_KERNEL((rocblas_trmvt_kernel<NB, false, true, true>),
                                      trmvt_grid,
                                      trmvt_threads,
                                      TRMV_TEMPLATE_PARAMS);
        }
        else
        {
            if(transA == rocblas_operation_none)
                ROCBLAS_LAUNCH_KERNEL(
                    (rocblas_trmvn_kernel<TRMVN_DIM_X, TRMVN_DIM_Y, false, false>),
                    trmvn_grid,
                    trmvn_threads,
                    TRMV_TEMPLATE_PARAMS);
            else if(transA == rocblas_operation_transpose)
                ROCBLAS_LAUNCH_KERNEL((rocblas_trmvt_kernel<NB, false, false, false>),
                                      trmvt_grid,
                                      trmvt_threads,
                                      TRMV_TEMPLATE_PARAMS);
            else if(transA == rocblas_operation_conjugate_transpose)
                ROCBLAS_LAUNCH_KERNEL((rocblas_trmvt_kernel<NB, false, true, false>),
                                      trmvt_grid,
                                      trmvt_threads,
                                      TRMV_TEMPLATE_PARAMS);
        }
    }
    else if(uplo == rocblas_fill_lower)
    {
        if(diag == rocblas_diagonal_unit)
        {
            if(transA == rocblas_operation_none)
                ROCBLAS_LAUNCH_KERNEL((rocblas_trmvn_kernel<TRMVN_DIM_X, TRMVN_DIM_Y, true, true>),
                                      trmvn_grid,
                                      trmvn_threads,
                                      TRMV_TEMPLATE_PARAMS);
            else if(transA == rocblas_operation_transpose)
                ROCBLAS_LAUNCH_KERNEL((rocblas_trmvt_kernel<NB, true, false, true>),
                                      trmvt_grid,
                                      trmvt_threads,
                                      TRMV_TEMPLATE_PARAMS);
            else if(transA == rocblas_operation_conjugate_transpose)
                ROCBLAS_LAUNCH_KERNEL((rocblas_trmvt_kernel<NB, true, true, true>),
                                      trmvt_grid,
                                      trmvt_threads,
                                      TRMV_TEMPLATE_PARAMS);
        }
        else
        {
            if(transA == rocblas_operation_none)
                ROCBLAS_LAUNCH_KERNEL((rocblas_trmvn_kernel<TRMVN_DIM_X, TRMVN_DIM_Y, true, false>),
                                      trmvn_grid,
                                      trmvn_threads,
                                      TRMV_TEMPLATE_PARAMS);
            else if(transA == rocblas_operation_transpose)
                ROCBLAS_LAUNCH_KERNEL((rocblas_trmvt_kernel<NB, true, false, false>),
                                      trmvt_grid,
                                      trmvt_threads,
                                      TRMV_TEMPLATE_PARAMS);
            else if(transA == rocblas_operation_conjugate_transpose)
                ROCBLAS_LAUNCH_KERNEL((rocblas_trmvt_kernel<NB, true, true, false>),
                                      trmvt_grid,
                                      trmvt_threads,
                                      TRMV_TEMPLATE_PARAMS);
        }
    }

    //
    // Copy workspace to x.
    //
    {
        static constexpr rocblas_int offsetw = 0;
        static constexpr rocblas_int incw    = 1;
        return rocblas_internal_copy_launcher<int64_t, ROCBLAS_COPY_NB>(handle,
                                                                        n,
                                                                        workspace,
                                                                        offsetw,
                                                                        incw,
                                                                        stride_w,
                                                                        x,
                                                                        offset_x,
                                                                        incx,
                                                                        stride_x,
                                                                        batch_count);
    }

#undef TRMV_TEMPLATE_PARAMS
}

template <typename T>
rocblas_status rocblas_internal_trmv_template(rocblas_handle    handle,
                                              rocblas_fill      uplo,
                                              rocblas_operation transA,
                                              rocblas_diagonal  diag,
                                              rocblas_int       n,
                                              const T*          A,
                                              rocblas_stride    offset_A,
                                              rocblas_int       lda,
                                              rocblas_stride    stride_A,
                                              T*                x,
                                              rocblas_stride    offset_x,
                                              rocblas_int       incx,
                                              rocblas_stride    stride_x,
                                              T*                workspace,
                                              rocblas_stride    stride_w,
                                              rocblas_int       batch_count)
{
    return rocblas_internal_trmv_launcher<const T*>(handle,
                                                    uplo,
                                                    transA,
                                                    diag,
                                                    n,
                                                    A,
                                                    offset_A,
                                                    lda,
                                                    stride_A,
                                                    x,
                                                    offset_x,
                                                    incx,
                                                    stride_x,
                                                    workspace,
                                                    stride_w,
                                                    batch_count);
}

template <typename T>
rocblas_status rocblas_internal_trmv_batched_template(rocblas_handle    handle,
                                                      rocblas_fill      uplo,
                                                      rocblas_operation transA,
                                                      rocblas_diagonal  diag,
                                                      rocblas_int       n,
                                                      const T* const*   A,
                                                      rocblas_stride    offset_A,
                                                      rocblas_int       lda,
                                                      rocblas_stride    stride_A,
                                                      T* const*         x,
                                                      rocblas_stride    offset_x,
                                                      rocblas_int       incx,
                                                      rocblas_stride    stride_x,
                                                      T*                workspace,
                                                      rocblas_stride    stride_w,
                                                      rocblas_int       batch_count)
{
    return rocblas_internal_trmv_launcher<const T* const*>(handle,
                                                           uplo,
                                                           transA,
                                                           diag,
                                                           n,
                                                           A,
                                                           offset_A,
                                                           lda,
                                                           stride_A,
                                                           x,
                                                           offset_x,
                                                           incx,
                                                           stride_x,
                                                           workspace,
                                                           stride_w,
                                                           batch_count);
}

template <typename T, typename U>
rocblas_status rocblas_trmv_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           rocblas_fill   uplo,
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
    rocblas_status check_numerics_status = rocblas_status_success;
    if(is_input)
    {
        check_numerics_status
            = rocblas_internal_check_numerics_matrix_template(function_name,
                                                              handle,
                                                              rocblas_operation_none,
                                                              uplo,
                                                              rocblas_client_triangular_matrix,
                                                              n,
                                                              n,
                                                              A,
                                                              offset_a,
                                                              lda,
                                                              stride_a,
                                                              batch_count,
                                                              check_numerics,
                                                              is_input);

        if(check_numerics_status != rocblas_status_success)
            return check_numerics_status;
    }

    check_numerics_status = rocblas_internal_check_numerics_vector_template(function_name,
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
// template parameters in the files *trmv*.cpp

#ifdef INSTANTIATE_TRMV_TEMPLATE
#error INSTANTIATE_TRMV_TEMPLATE already defined
#endif

#define INSTANTIATE_TRMV_TEMPLATE(T_)                                                            \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status rocblas_internal_trmv_template<T_>( \
        rocblas_handle    handle,                                                                \
        rocblas_fill      uplo,                                                                  \
        rocblas_operation transA,                                                                \
        rocblas_diagonal  diag,                                                                  \
        rocblas_int       n,                                                                     \
        const T_*         A,                                                                     \
        rocblas_stride    offset_A,                                                              \
        rocblas_int       lda,                                                                   \
        rocblas_stride    stride_A,                                                              \
        T_*               x,                                                                     \
        rocblas_stride    offset_x,                                                              \
        rocblas_int       incx,                                                                  \
        rocblas_stride    stride_x,                                                              \
        T_*               workspace,                                                             \
        rocblas_stride    stride_w,                                                              \
        rocblas_int       batch_count);

INSTANTIATE_TRMV_TEMPLATE(float)
INSTANTIATE_TRMV_TEMPLATE(double)
INSTANTIATE_TRMV_TEMPLATE(rocblas_float_complex)
INSTANTIATE_TRMV_TEMPLATE(rocblas_double_complex)

#undef INSTANTIATE_TRMV_TEMPLATE

#ifdef INSTANTIATE_TRMV_TEMPLATE
#error INSTANTIATE_TRMV_TEMPLATE already defined
#endif

#define INSTANTIATE_TRMV_BATCHED_TEMPLATE(T_)                                   \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                    \
        rocblas_internal_trmv_batched_template<T_>(rocblas_handle    handle,    \
                                                   rocblas_fill      uplo,      \
                                                   rocblas_operation transA,    \
                                                   rocblas_diagonal  diag,      \
                                                   rocblas_int       n,         \
                                                   const T_* const*  A,         \
                                                   rocblas_stride    offset_A,  \
                                                   rocblas_int       lda,       \
                                                   rocblas_stride    stride_A,  \
                                                   T_* const*        x,         \
                                                   rocblas_stride    offset_x,  \
                                                   rocblas_int       incx,      \
                                                   rocblas_stride    stride_x,  \
                                                   T_*               workspace, \
                                                   rocblas_stride    stride_w,  \
                                                   rocblas_int       batch_count);

INSTANTIATE_TRMV_BATCHED_TEMPLATE(float)
INSTANTIATE_TRMV_BATCHED_TEMPLATE(double)
INSTANTIATE_TRMV_BATCHED_TEMPLATE(rocblas_float_complex)
INSTANTIATE_TRMV_BATCHED_TEMPLATE(rocblas_double_complex)

#undef INSTANTIATE_TRMV_BATCHED_TEMPLATE

#ifdef INSTANTIATE_TRMV_NUMERICS
#error INSTANTIATE_TRMV_NUMERICS already defined
#endif

#define INSTANTIATE_TRMV_NUMERICS(T_, U_)                                                      \
    template rocblas_status rocblas_trmv_check_numerics<T_, U_>(const char*    function_name,  \
                                                                rocblas_handle handle,         \
                                                                rocblas_fill   uplo,           \
                                                                int64_t        n,              \
                                                                T_             A,              \
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

INSTANTIATE_TRMV_NUMERICS(float const*, float*)
INSTANTIATE_TRMV_NUMERICS(double const*, double*)
INSTANTIATE_TRMV_NUMERICS(rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_TRMV_NUMERICS(rocblas_double_complex const*, rocblas_double_complex*)
INSTANTIATE_TRMV_NUMERICS(float const* const*, float* const*)
INSTANTIATE_TRMV_NUMERICS(double const* const*, double* const*)
INSTANTIATE_TRMV_NUMERICS(rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_TRMV_NUMERICS(rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_TRMV_NUMERICS

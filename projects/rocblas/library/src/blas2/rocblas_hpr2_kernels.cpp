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

#include "check_numerics_vector.hpp"
#include "device_macros.hpp"
#include "handle.hpp"
#include "rocblas_hpr2.hpp"

template <int DIM_X, int DIM_Y, int N_TX, typename T>
__forceinline__ __device__ void rocblas_hpr2_kernel_calc(bool        is_upper,
                                                         rocblas_int n,
                                                         T           alpha,
                                                         const T*    x,
                                                         int64_t     incx,
                                                         const T*    y,
                                                         int64_t     incy,
                                                         T*          AP)
{
    rocblas_int tx = (blockIdx.x * DIM_X * N_TX) + threadIdx.x;
    rocblas_int ty = blockIdx.y * DIM_Y + threadIdx.y;

    int index = is_upper ? ((ty * (ty + 1)) / 2) + tx : ((ty * (2 * n - ty + 1)) / 2) + (tx - ty);

#pragma unroll
    for(int i = 0; i < N_TX; i++, tx += DIM_X, index += DIM_X)
    {
        if(is_upper ? ty < n && tx < ty : tx < n && ty < tx)
        {
            AP[index] += alpha * x[tx * incx] * conj(y[ty * incy])
                         + conj(alpha) * y[tx * incy] * conj(x[ty * incx]);
        }
        else if(tx == ty && tx < n)
        {
            AP[index] = std::real(AP[index]) + alpha * x[tx * incx] * conj(y[ty * incy])
                        + conj(alpha) * y[tx * incy] * conj(x[ty * incx]);
        }
    }
}

template <int DIM_X, int DIM_Y, int N_TX, typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_hpr2_kernel(bool           is_upper,
                    rocblas_int    n,
                    TScal          alpha_device_host,
                    TConstPtr      xa,
                    rocblas_stride shift_x,
                    int64_t        incx,
                    rocblas_stride stride_x,
                    TConstPtr      ya,
                    rocblas_stride shift_y,
                    int64_t        incy,
                    rocblas_stride stride_y,
                    TPtr           APa,
                    rocblas_stride shift_A,
                    rocblas_stride stride_A,
                    rocblas_int    batch_count)
{
    auto alpha = load_scalar(alpha_device_host);

    if(!alpha)
        return;

    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif
        const auto* x  = load_ptr_batch(xa, batch, shift_x, stride_x);
        const auto* y  = load_ptr_batch(ya, batch, shift_y, stride_y);
        auto*       AP = load_ptr_batch(APa, batch, shift_A, stride_A);

        rocblas_hpr2_kernel_calc<DIM_X, DIM_Y, N_TX>(is_upper, n, alpha, x, incx, y, incy, AP);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

/**
 * TScal     is always: const T* (either host or device)
 * TConstPtr is either: const T* OR const T* const*
 * TPtr      is either:       T* OR       T* const*
 * Where T is the base type (rocblas_float_complex or rocblas_double_complex)
 */
template <typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_internal_hpr2_launcher(rocblas_handle handle,
                                              rocblas_fill   uplo,
                                              rocblas_int    n,
                                              TScal          alpha,
                                              TConstPtr      x,
                                              rocblas_stride offset_x,
                                              int64_t        incx,
                                              rocblas_stride stride_x,
                                              TConstPtr      y,
                                              rocblas_stride offset_y,
                                              int64_t        incy,
                                              rocblas_stride stride_y,
                                              TPtr           AP,
                                              rocblas_stride offset_A,
                                              rocblas_stride stride_A,
                                              rocblas_int    batch_count)
{
    // Quick return if possible. Not Argument error
    if(!n || !batch_count)
        return rocblas_status_success;

    // in case of negative inc, shift pointer to end of data for negative indexing tid*inc
    ptrdiff_t shift_x = incx < 0 ? offset_x - ptrdiff_t(incx) * (n - 1) : offset_x;
    ptrdiff_t shift_y = incy < 0 ? offset_y - ptrdiff_t(incy) * (n - 1) : offset_y;

    int batches = handle->getBatchGridDim((int)batch_count);

    static constexpr int HPR2_DIM_X = 64;
    static constexpr int HPR2_DIM_Y = 16;
    static constexpr int N_TX       = 2; // x items per x thread

    rocblas_int blocksX = (n - 1) / (HPR2_DIM_X * N_TX) + 1;
    rocblas_int blocksY = (n - 1) / HPR2_DIM_Y + 1;

    dim3 hpr2_grid(blocksX, blocksY, batches);
    dim3 hpr2_threads(HPR2_DIM_X, HPR2_DIM_Y);

    if(rocblas_pointer_mode_device == handle->pointer_mode)
    {
        ROCBLAS_LAUNCH_KERNEL((rocblas_hpr2_kernel<HPR2_DIM_X, HPR2_DIM_Y, N_TX>),
                              hpr2_grid,
                              hpr2_threads,
                              0,
                              handle->get_stream(),
                              uplo == rocblas_fill_upper,
                              n,
                              alpha,
                              x,
                              shift_x,
                              incx,
                              stride_x,
                              y,
                              shift_y,
                              incy,
                              stride_y,
                              AP,
                              offset_A,
                              stride_A,
                              batch_count);
    }
    else
        ROCBLAS_LAUNCH_KERNEL((rocblas_hpr2_kernel<HPR2_DIM_X, HPR2_DIM_Y, N_TX>),
                              hpr2_grid,
                              hpr2_threads,
                              0,
                              handle->get_stream(),
                              uplo == rocblas_fill_upper,
                              n,
                              *alpha,
                              x,
                              shift_x,
                              incx,
                              stride_x,
                              y,
                              shift_y,
                              incy,
                              stride_y,
                              AP,
                              offset_A,
                              stride_A,
                              batch_count);

    return rocblas_status_success;
}

//TODO :-Add rocblas_check_numerics_hp_matrix_template for checking Matrix `AP` which is a Hermitian Packed Matrix
template <typename T, typename U>
rocblas_status rocblas_hpr2_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           int64_t        n,
                                           T              A,
                                           rocblas_stride offset_a,
                                           rocblas_stride stride_a,
                                           U              x,
                                           rocblas_stride offset_x,
                                           int64_t        inc_x,
                                           rocblas_stride stride_x,
                                           U              y,
                                           rocblas_stride offset_y,
                                           int64_t        inc_y,
                                           rocblas_stride stride_y,
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
    if(check_numerics_status != rocblas_status_success)
        return check_numerics_status;

    check_numerics_status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                            handle,
                                                                            n,
                                                                            y,
                                                                            offset_y,
                                                                            inc_y,
                                                                            stride_y,
                                                                            batch_count,
                                                                            check_numerics,
                                                                            is_input);

    return check_numerics_status;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files *hpr2*.cpp

#ifdef INST_HPR2_LAUNCHER
#error INST_HPR2_LAUNCHER already defined
#endif

#define INST_HPR2_LAUNCHER(TScal_, TConstPtr_, TPtr_)                                  \
    template rocblas_status rocblas_internal_hpr2_launcher<TScal_, TConstPtr_, TPtr_>( \
        rocblas_handle handle,                                                         \
        rocblas_fill   uplo,                                                           \
        rocblas_int    n,                                                              \
        TScal_         alpha,                                                          \
        TConstPtr_     x,                                                              \
        rocblas_stride offset_x,                                                       \
        int64_t        incx,                                                           \
        rocblas_stride stride_x,                                                       \
        TConstPtr_     y,                                                              \
        rocblas_stride offset_y,                                                       \
        int64_t        incy,                                                           \
        rocblas_stride stride_y,                                                       \
        TPtr_          AP,                                                             \
        rocblas_stride offset_A,                                                       \
        rocblas_stride stride_A,                                                       \
        rocblas_int    batch_count);

INST_HPR2_LAUNCHER(rocblas_float_complex const*,
                   rocblas_float_complex const*,
                   rocblas_float_complex*)
INST_HPR2_LAUNCHER(rocblas_double_complex const*,
                   rocblas_double_complex const*,
                   rocblas_double_complex*)
INST_HPR2_LAUNCHER(rocblas_float_complex const*,
                   rocblas_float_complex const* const*,
                   rocblas_float_complex* const*)
INST_HPR2_LAUNCHER(rocblas_double_complex const*,
                   rocblas_double_complex const* const*,
                   rocblas_double_complex* const*)

#undef INST_HPR2_LAUNCHER

#ifdef INST_HPR2_NUMERICS
#error INST_HPR2_NUMERICS already defined
#endif

#define INST_HPR2_NUMERICS(T_, U_)                                                             \
    template rocblas_status rocblas_hpr2_check_numerics<T_, U_>(const char*    function_name,  \
                                                                rocblas_handle handle,         \
                                                                int64_t        n,              \
                                                                T_             A,              \
                                                                rocblas_stride offset_a,       \
                                                                rocblas_stride stride_a,       \
                                                                U_             x,              \
                                                                rocblas_stride offset_x,       \
                                                                int64_t        inc_x,          \
                                                                rocblas_stride stride_x,       \
                                                                U_             y,              \
                                                                rocblas_stride offset_y,       \
                                                                int64_t        inc_y,          \
                                                                rocblas_stride stride_y,       \
                                                                int64_t        batch_count,    \
                                                                const int      check_numerics, \
                                                                bool           is_input);

INST_HPR2_NUMERICS(rocblas_float_complex*, rocblas_float_complex const*)
INST_HPR2_NUMERICS(rocblas_double_complex*, rocblas_double_complex const*)
INST_HPR2_NUMERICS(rocblas_float_complex* const*, rocblas_float_complex const* const*)
INST_HPR2_NUMERICS(rocblas_double_complex* const*, rocblas_double_complex const* const*)

#undef INST_HPR2_NUMERICS

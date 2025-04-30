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
#include "rocblas_hpr.hpp"

template <int DIM_X, int DIM_Y, int N_TX, typename T, typename U>
__forceinline__ __device__ void
    rocblas_hpr_kernel_calc(bool is_upper, rocblas_int n, U alpha, const T* x, int64_t incx, T* AP)
{
    int tx = (blockIdx.x * DIM_X * N_TX) + threadIdx.x;
    int ty = blockIdx.y * DIM_Y + threadIdx.y;

    int index = is_upper ? ((ty * (ty + 1)) / 2) + tx : ((ty * (2 * n - ty + 1)) / 2) + (tx - ty);

#pragma unroll
    for(int i = 0; i < N_TX; i++, tx += DIM_X, index += DIM_X)
    {
        if(is_upper ? ty < n && tx < ty : tx < n && ty < tx)
            AP[index] += alpha * x[tx * incx] * conj(x[ty * incx]);
        else if(tx == ty && tx < n)
        {
            AP[index] = std::real(AP[index]) + alpha * std::norm(x[tx * incx]);
        }
    }
}

template <int DIM_X, int DIM_Y, int N_TX, typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_hpr_kernel(bool           is_upper,
                   rocblas_int    n,
                   TScal          alpha_device_host,
                   TConstPtr      xa,
                   rocblas_stride shift_x,
                   int64_t        incx,
                   rocblas_stride stride_x,
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
        auto*       AP = load_ptr_batch(APa, batch, shift_A, stride_A);

        rocblas_hpr_kernel_calc<DIM_X, DIM_Y, N_TX>(is_upper, n, alpha, x, incx, AP);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

/**
 * TScal     is always: const U* (either host or device)
 * TConstPtr is either: const T* OR const T* const*
 * TPtr      is either:       T* OR       T* const*
 * Where T is the base type (rocblas_float_complex or rocblas_double_complex)
 * and U is the scalar type (float or double)
 */
template <typename API_INT, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_hpr_launcher(rocblas_handle handle,
                                    rocblas_fill   uplo,
                                    API_INT        n,
                                    TScal          alpha,
                                    TConstPtr      x,
                                    rocblas_stride offset_x,
                                    int64_t        incx,
                                    rocblas_stride stride_x,
                                    TPtr           AP,
                                    rocblas_stride offset_A,
                                    rocblas_stride stride_A,
                                    API_INT        batch_count)
{
    // Quick return if possible. Not Argument error
    if(!n || !batch_count)
        return rocblas_status_success;

    // in case of negative inc, shift pointer to end of data for negative indexing tid*inc
    ptrdiff_t shift_x = incx < 0 ? offset_x - ptrdiff_t(incx) * (n - 1) : offset_x;

    int batches = handle->getBatchGridDim((int)batch_count);

    static constexpr int HPR_DIM_X = 64;
    static constexpr int HPR_DIM_Y = 16;
    static constexpr int N_TX      = 2; // x items per x thread

    rocblas_int blocksX = (n - 1) / (HPR_DIM_X * N_TX) + 1;
    rocblas_int blocksY = (n - 1) / HPR_DIM_Y + 1;

    dim3 hpr_grid(blocksX, blocksY, batches);
    dim3 hpr_threads(HPR_DIM_X, HPR_DIM_Y);

    if(rocblas_pointer_mode_device == handle->pointer_mode)
    {
        ROCBLAS_LAUNCH_KERNEL((rocblas_hpr_kernel<HPR_DIM_X, HPR_DIM_Y, N_TX>),
                              hpr_grid,
                              hpr_threads,
                              0,
                              handle->get_stream(),
                              uplo == rocblas_fill_upper,
                              n,
                              alpha,
                              x,
                              shift_x,
                              incx,
                              stride_x,
                              AP,
                              offset_A,
                              stride_A,
                              batch_count);
    }
    else
        ROCBLAS_LAUNCH_KERNEL((rocblas_hpr_kernel<HPR_DIM_X, HPR_DIM_Y, N_TX>),
                              hpr_grid,
                              hpr_threads,
                              0,
                              handle->get_stream(),
                              uplo == rocblas_fill_upper,
                              n,
                              *alpha,
                              x,
                              shift_x,
                              incx,
                              stride_x,
                              AP,
                              offset_A,
                              stride_A,
                              batch_count);

    return rocblas_status_success;
}

//TODO :-Add rocblas_check_numerics_hp_matrix_template for checking Matrix `AP` which is a Hermitian Packed Matrix
template <typename T, typename U>
rocblas_status rocblas_hpr_check_numerics(const char*    function_name,
                                          rocblas_handle handle,
                                          int64_t        n,
                                          T              AP,
                                          rocblas_stride offset_a,
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
// template parameters in the files *hpr*.cpp

#ifdef INST_HPR_LAUNCHER
#error INST_HPR_LAUNCHER already defined
#endif

#define INST_HPR_LAUNCHER(TScal_, TConstPtr_, TPtr_)                                      \
    template rocblas_status rocblas_hpr_launcher<rocblas_int, TScal_, TConstPtr_, TPtr_>( \
        rocblas_handle handle,                                                            \
        rocblas_fill   uplo,                                                              \
        rocblas_int    n,                                                                 \
        TScal_         alpha,                                                             \
        TConstPtr_     x,                                                                 \
        rocblas_stride offset_x,                                                          \
        int64_t        incx,                                                              \
        rocblas_stride stride_x,                                                          \
        TPtr_          AP,                                                                \
        rocblas_stride offset_A,                                                          \
        rocblas_stride stride_A,                                                          \
        rocblas_int    batch_count);

INST_HPR_LAUNCHER(float const*, rocblas_float_complex const*, rocblas_float_complex*)
INST_HPR_LAUNCHER(double const*, rocblas_double_complex const*, rocblas_double_complex*)
INST_HPR_LAUNCHER(float const*, rocblas_float_complex const* const*, rocblas_float_complex* const*)
INST_HPR_LAUNCHER(double const*,
                  rocblas_double_complex const* const*,
                  rocblas_double_complex* const*)

#undef INST_HPR_LAUNCHER

#ifdef INSTANTIATE_HPR_NUMERICS
#error INSTANTIATE_HPR_NUMERICS already defined
#endif

#define INSTANTIATE_HPR_NUMERICS(T_, U_)                                                      \
    template rocblas_status rocblas_hpr_check_numerics<T_, U_>(const char*    function_name,  \
                                                               rocblas_handle handle,         \
                                                               int64_t        n,              \
                                                               T_             AP,             \
                                                               rocblas_stride offset_a,       \
                                                               rocblas_stride stride_a,       \
                                                               U_             x,              \
                                                               rocblas_stride offset_x,       \
                                                               int64_t        inc_x,          \
                                                               rocblas_stride stride_x,       \
                                                               int64_t        batch_count,    \
                                                               const int      check_numerics, \
                                                               bool           is_input);

INSTANTIATE_HPR_NUMERICS(rocblas_float_complex*, rocblas_float_complex const*)
INSTANTIATE_HPR_NUMERICS(rocblas_double_complex*, rocblas_double_complex const*)
INSTANTIATE_HPR_NUMERICS(rocblas_float_complex* const*, rocblas_float_complex const* const*)
INSTANTIATE_HPR_NUMERICS(rocblas_double_complex* const*, rocblas_double_complex const* const*)

#undef INSTANTIATE_HPR_NUMERICS

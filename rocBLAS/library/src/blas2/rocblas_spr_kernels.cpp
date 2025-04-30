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

#include "device_macros.hpp"
#include "rocblas_spr.hpp"

//TODO :-Add rocblas_check_numerics_sp_matrix_template for checking Matrix `A` which is a Symmetric Packed Matrix
template <typename T, typename U>
rocblas_status rocblas_spr_check_numerics(const char*    function_name,
                                          rocblas_handle handle,
                                          int64_t        n,
                                          T              A,
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

template <int DIM_X, int DIM_Y, int N_TX, typename T>
__forceinline__ __device__ void
    rocblas_spr_kernel_calc(bool is_upper, rocblas_int n, T alpha, const T* x, int64_t incx, T* AP)
{
    int tx = (blockIdx.x * DIM_X * N_TX) + threadIdx.x;
    int ty = blockIdx.y * DIM_Y + threadIdx.y;

    int index = is_upper ? ((ty * (ty + 1)) / 2) + tx : ((ty * (2 * n - ty + 1)) / 2) + (tx - ty);

#pragma unroll
    for(int i = 0; i < N_TX; i++, tx += DIM_X, index += DIM_X)
    {
        if(is_upper ? ty < n && tx <= ty : tx < n && ty <= tx)
            AP[index] += alpha * x[tx * incx] * x[ty * incx];
    }
}

template <int DIM_X, int DIM_Y, int N_TX, typename TStruct, typename TConstPtr, typename TPtr>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_spr_kernel(bool           host_ptr_mode,
                   bool           is_upper,
                   rocblas_int    n,
                   TStruct        alpha_device_host,
                   TConstPtr      xa,
                   rocblas_stride shift_x,
                   int64_t        incx,
                   rocblas_stride stride_x,
                   TPtr           APa,
                   rocblas_stride shift_A,
                   rocblas_stride stride_A,
                   rocblas_int    batch_count)
{
    auto alpha = host_ptr_mode ? alpha_device_host.value : load_scalar(alpha_device_host.ptr);

    if(!alpha)
        return;

    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif
        auto*       AP = load_ptr_batch(APa, batch, shift_A, stride_A);
        const auto* x  = load_ptr_batch(xa, batch, shift_x, stride_x);

        rocblas_spr_kernel_calc<DIM_X, DIM_Y, N_TX>(is_upper, n, alpha, x, incx, AP);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

/**
 * TScal     is always: const T* (either host or device)
 * TConstPtr is either: const T* OR const T* const*
 * TPtr      is either:       T* OR       T* const*
 * Where T is the base type (float, double, rocblas_float_complex, etc.)
 */
template <typename API_INT, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_internal_spr_launcher(rocblas_handle handle,
                                             rocblas_fill   uplo,
                                             API_INT        n,
                                             TScal const*   alpha,
                                             TConstPtr      x,
                                             rocblas_stride offset_x,
                                             int64_t        incx,
                                             rocblas_stride stride_x,
                                             TPtr           AP,
                                             rocblas_stride offset_A,
                                             rocblas_stride stride_A,
                                             int64_t        batch_count)
{
    // Quick return if possible. Not Argument error
    if(!n || !batch_count)
        return rocblas_status_success;

    // in case of negative inc, shift pointer to end of data for negative indexing tid*inc
    ptrdiff_t shift_x = incx < 0 ? offset_x - ptrdiff_t(incx) * (n - 1) : offset_x;

    int batches = handle->getBatchGridDim((int)batch_count);

    static constexpr int SPR_DIM_X = 64;
    static constexpr int SPR_DIM_Y = 16;

    bool                            host_mode = handle->pointer_mode == rocblas_pointer_mode_host;
    rocblas_internal_val_ptr<TScal> alpha_device_host(host_mode, alpha);

    if constexpr(std::is_same_v<TScal, rocblas_double_complex>)
    {
        static constexpr int N_TX    = 1;
        rocblas_int          blocksX = (n - 1) / (SPR_DIM_X * N_TX) + 1;
        rocblas_int          blocksY = (n - 1) / SPR_DIM_Y + 1;

        dim3 spr_grid(blocksX, blocksY, batches);
        dim3 spr_threads(SPR_DIM_X, SPR_DIM_Y);
        ROCBLAS_LAUNCH_KERNEL((rocblas_spr_kernel<SPR_DIM_X, SPR_DIM_Y, N_TX>),
                              spr_grid,
                              spr_threads,
                              0,
                              handle->get_stream(),
                              host_mode,
                              uplo == rocblas_fill_upper,
                              n,
                              alpha_device_host,
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
    {
        static constexpr int N_TX    = 2;
        rocblas_int          blocksX = (n - 1) / (SPR_DIM_X * N_TX) + 1;
        rocblas_int          blocksY = (n - 1) / SPR_DIM_Y + 1;

        dim3 spr_grid(blocksX, blocksY, batches);
        dim3 spr_threads(SPR_DIM_X, SPR_DIM_Y);
        ROCBLAS_LAUNCH_KERNEL((rocblas_spr_kernel<SPR_DIM_X, SPR_DIM_Y, N_TX>),
                              spr_grid,
                              spr_threads,
                              0,
                              handle->get_stream(),
                              host_mode,
                              uplo == rocblas_fill_upper,
                              n,
                              alpha_device_host,
                              x,
                              shift_x,
                              incx,
                              stride_x,
                              AP,
                              offset_A,
                              stride_A,
                              batch_count);
    }

    return rocblas_status_success;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files *spr*.cpp

// clang-format off

#ifdef INSTANTIATE_SPR_LAUNCHER
#error INSTANTIATE_SPR_LAUNCHER already defined
#endif

#define INSTANTIATE_SPR_LAUNCHER(TI_, TScal_, TConstPtr_, TPtr_)                      \
template rocblas_status rocblas_internal_spr_launcher<TI_, TScal_, TConstPtr_, TPtr_> \
                                   (rocblas_handle handle,                            \
                                    rocblas_fill   uplo,                              \
                                    TI_    n,                                         \
                                    TScal_ const*  alpha,                             \
                                    TConstPtr_     x,                                 \
                                    rocblas_stride    offset_x,                       \
                                    int64_t    incx,                                      \
                                    rocblas_stride stride_x,                          \
                                    TPtr_          AP,                                \
                                    rocblas_stride    offset_A,                       \
                                    rocblas_stride stride_A,                          \
                                    int64_t    batch_count);

INSTANTIATE_SPR_LAUNCHER(rocblas_int, float, float const*, float*)
INSTANTIATE_SPR_LAUNCHER(rocblas_int, double, double const*, double*)
INSTANTIATE_SPR_LAUNCHER(rocblas_int, rocblas_float_complex, rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_SPR_LAUNCHER(rocblas_int, rocblas_double_complex, rocblas_double_complex const*, rocblas_double_complex*)
INSTANTIATE_SPR_LAUNCHER(rocblas_int, float, float const* const*, float* const*)
INSTANTIATE_SPR_LAUNCHER(rocblas_int, double, double const* const*, double* const*)
INSTANTIATE_SPR_LAUNCHER(rocblas_int, rocblas_float_complex, rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_SPR_LAUNCHER(rocblas_int, rocblas_double_complex, rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_SPR_LAUNCHER

#ifdef INSTANTIATE_SPR_NUMERICS
#error INSTANTIATE_SPR_NUMERICS already defined
#endif

#define INSTANTIATE_SPR_NUMERICS(T_, U_)                                 \
template rocblas_status rocblas_spr_check_numerics<T_, U_>               \
                                         (const char*    function_name,  \
                                          rocblas_handle handle,         \
                                          int64_t        n,              \
                                          T_             A,              \
                                          rocblas_stride offset_a,       \
                                          rocblas_stride stride_a,       \
                                          U_             x,              \
                                          rocblas_stride offset_x,       \
                                          int64_t        inc_x,          \
                                          rocblas_stride stride_x,       \
                                          int64_t        batch_count,    \
                                          const int      check_numerics, \
                                          bool           is_input);

INSTANTIATE_SPR_NUMERICS(float*, float const*)
INSTANTIATE_SPR_NUMERICS(double*, double const*)
INSTANTIATE_SPR_NUMERICS(rocblas_float_complex*, rocblas_float_complex const*)
INSTANTIATE_SPR_NUMERICS(rocblas_double_complex*, rocblas_double_complex const*)
INSTANTIATE_SPR_NUMERICS(float* const*, float const* const*)
INSTANTIATE_SPR_NUMERICS(double* const*, double const* const*)
INSTANTIATE_SPR_NUMERICS(rocblas_float_complex* const*, rocblas_float_complex const* const*)
INSTANTIATE_SPR_NUMERICS(rocblas_double_complex* const*, rocblas_double_complex const* const*)

#undef INSTANTIATE_SPR_NUMERICS

// clang-format on

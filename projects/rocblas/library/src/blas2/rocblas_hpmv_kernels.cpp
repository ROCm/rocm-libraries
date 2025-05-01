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
#include "check_numerics_vector.hpp"
#include "device_macros.hpp"
#include "rocblas_hpmv.hpp"

/**
  *  A combined kernel to handle all hpmv cases.
  */
template <int DIM_X, int DIM_Y, typename T>
__forceinline__ __device__ void rocblas_hpmv_kernel_calc(bool        is_upper,
                                                         rocblas_int n,
                                                         T           alpha,
                                                         const T*    AP,
                                                         const T*    x,
                                                         int64_t     incx,
                                                         T           beta,
                                                         T*          y,
                                                         int64_t     incy)
{
    rocblas_int thread_id = threadIdx.x + threadIdx.y * DIM_X;

    // threads are all configurated locally
    rocblas_int tx = thread_id % DIM_X;
    rocblas_int ty = thread_id / DIM_X;

    rocblas_int ind = blockIdx.x * DIM_X + tx;

    if(!alpha)
    {
        if(thread_id < DIM_X && ind < n)
        {
            rocblas_int idx = blockIdx.x * DIM_X + thread_id;
            if(idx < n)
                y[idx * int64_t(incy)] = beta ? beta * y[idx * int64_t(incy)] : 0;
        }
        return;
    }

    __shared__ T sdata[DIM_X * DIM_Y];
    T            res_A = 0.0;
    rocblas_int  col;

    for(col = ty; col < n; col += DIM_Y)
    {
        if(ind < n)
        {
            int  ind_x = ind;
            int  ind_y = col;
            bool CONJ  = false;

            if((ind > col && is_upper) || (ind < col && !is_upper))
            {
                // in the opposite triangle, get conjugate of value at transposed position
                ind_x = col;
                ind_y = ind;
                CONJ  = true;
            }

            // The indices used here for AP come from the summation of the number of elements
            // in previous columns.
            //                              col
            // For upper matrices, index = sigma(i) + row.
            //                              i=1
            //
            //                              col-1
            // For lower matrices, index = sigma(n-i) + row
            //                              i=0

            int64_t index = is_upper
                                ? ((int64_t(ind_y) * (ind_y + 1)) / 2) + ind_x
                                : ((int64_t(ind_y) * (2 * n - ind_y + 1)) / 2) + (ind_x - ind_y);

            res_A += (ind_x == ind_y ? std::real(AP[index])
                      : CONJ         ? conj(AP[index])
                                     : (AP[index]))
                     * x[col * int64_t(incx)];
        }
    }

    // Store partial sums for the diagonal
    sdata[tx + ty * DIM_X] = res_A;
    __syncthreads();

    if(thread_id < DIM_X && ind < n)
    {
        // Add the partial sums of each diagonal and store
        for(rocblas_int i = 1; i < DIM_Y; i++)
            sdata[thread_id] += sdata[thread_id + DIM_X * i];

        int64_t idx = blockIdx.x * DIM_X + thread_id;
        // Update y.
        if(idx < n)
            y[idx * int64_t(incy)] = beta ? alpha * sdata[thread_id] + beta * y[idx * int64_t(incy)]
                                          : alpha * sdata[thread_id];
    }
}

/**
  *  Loads pointers and launches the actual calculation kernel.
  */
template <int DIM_X, int DIM_Y, typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_hpmv_kernel(bool           is_upper,
                    rocblas_int    n,
                    TScal          alpha_device_host,
                    TConstPtr      APa,
                    rocblas_stride shifta,
                    rocblas_stride strideA,
                    TConstPtr      xa,
                    rocblas_stride shiftx,
                    int64_t        incx,
                    rocblas_stride stridex,
                    TScal          beta_device_host,
                    TPtr           ya,
                    rocblas_stride shifty,
                    int64_t        incy,
                    rocblas_stride stridey,
                    rocblas_int    batch_count)
{
    auto alpha = load_scalar(alpha_device_host);
    auto beta  = load_scalar(beta_device_host);

    if(!alpha && beta == 1)
        return;

    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif
        auto AP = cond_load_ptr_batch(alpha, APa, batch, shifta, strideA);
        auto x  = cond_load_ptr_batch(alpha, xa, batch, shiftx, stridex);

        auto y = load_ptr_batch(ya, batch, shifty, stridey);

        rocblas_hpmv_kernel_calc<DIM_X, DIM_Y>(is_upper, n, alpha, AP, x, incx, beta, y, incy);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

/**
  *  TScal     is always: const T* (either host or device)
  *  TConstPtr is either: const T* OR const T* const*
  *  TPtr      is either:       T* OR       T* const*
  */
template <typename API_INT, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_hpmv_launcher(rocblas_handle handle,
                                     rocblas_fill   uplo,
                                     API_INT        n,
                                     TScal          alpha,
                                     TConstPtr      AP,
                                     rocblas_stride offseta,
                                     rocblas_stride strideA,
                                     TConstPtr      x,
                                     rocblas_stride offsetx,
                                     int64_t        incx,
                                     rocblas_stride stridex,
                                     TScal          beta,
                                     TPtr           y,
                                     rocblas_stride offsety,
                                     int64_t        incy,
                                     rocblas_stride stridey,
                                     API_INT        batch_count)
{
    // quick return
    if(!n || !batch_count)
        return rocblas_status_success;

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    offsetx = incx < 0 ? offsetx - ptrdiff_t(incx) * (n - 1) : offsetx;
    offsety = incy < 0 ? offsety - ptrdiff_t(incy) * (n - 1) : offsety;

    int batches = handle->getBatchGridDim((int)batch_count);

    static constexpr int HPMV_DIM_X = 64;
    static constexpr int HPMV_DIM_Y = 16;

    rocblas_int blocks = (n - 1) / (HPMV_DIM_X) + 1;
    dim3        hpmv_grid(blocks, 1, batches);
    dim3        hpmv_threads(HPMV_DIM_X, HPMV_DIM_Y);

    // Launch a modified gemv kernel for hpmv.
    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        ROCBLAS_LAUNCH_KERNEL((rocblas_hpmv_kernel<HPMV_DIM_X, HPMV_DIM_Y>),
                              hpmv_grid,
                              hpmv_threads,
                              0,
                              handle->get_stream(),
                              uplo == rocblas_fill_upper,
                              n,
                              alpha,
                              AP,
                              offseta,
                              strideA,
                              x,
                              offsetx,
                              incx,
                              stridex,
                              beta,
                              y,
                              offsety,
                              incy,
                              stridey,
                              batch_count);
    }
    else
    {
        if(!*alpha && *beta == 1)
            return rocblas_status_success;

        ROCBLAS_LAUNCH_KERNEL((rocblas_hpmv_kernel<HPMV_DIM_X, HPMV_DIM_Y>),
                              hpmv_grid,
                              hpmv_threads,
                              0,
                              handle->get_stream(),
                              uplo == rocblas_fill_upper,
                              n,
                              *alpha,
                              AP,
                              offseta,
                              strideA,
                              x,
                              offsetx,
                              incx,
                              stridex,
                              *beta,
                              y,
                              offsety,
                              incy,
                              stridey,
                              batch_count);
    }

    return rocblas_status_success;
}

//TODO :-Add rocblas_check_numerics_hp_matrix_launcher for checking Matrix `AP` which is a Hermitian Packed matrix
template <typename T, typename U>
rocblas_status rocblas_hpmv_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           int64_t        n,
                                           T              AP,
                                           rocblas_stride offset_a,
                                           rocblas_stride stride_a,
                                           T              x,
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
// template parameters in the files *hpmv*.cpp

#ifdef INST_HPMV_LAUNCHER
#error INST_HPMV_LAUNCHER already defined
#endif

#define INST_HPMV_LAUNCHER(TI_, TScal_, TConstPtr_, TPtr_)                         \
    template rocblas_status rocblas_hpmv_launcher<TI_, TScal_, TConstPtr_, TPtr_>( \
        rocblas_handle handle,                                                     \
        rocblas_fill   uplo,                                                       \
        TI_            n,                                                          \
        TScal_         alpha,                                                      \
        TConstPtr_     AP,                                                         \
        rocblas_stride offseta,                                                    \
        rocblas_stride strideA,                                                    \
        TConstPtr_     x,                                                          \
        rocblas_stride offsetx,                                                    \
        int64_t        incx,                                                       \
        rocblas_stride stridex,                                                    \
        TScal_         beta,                                                       \
        TPtr_          y,                                                          \
        rocblas_stride offsety,                                                    \
        int64_t        incy,                                                       \
        rocblas_stride stridey,                                                    \
        TI_            batch_count);

INST_HPMV_LAUNCHER(rocblas_int,
                   rocblas_float_complex const*,
                   rocblas_float_complex const*,
                   rocblas_float_complex*)
INST_HPMV_LAUNCHER(rocblas_int,
                   rocblas_double_complex const*,
                   rocblas_double_complex const*,
                   rocblas_double_complex*)

INST_HPMV_LAUNCHER(rocblas_int,
                   rocblas_float_complex const*,
                   rocblas_float_complex const* const*,
                   rocblas_float_complex* const*)
INST_HPMV_LAUNCHER(rocblas_int,
                   rocblas_double_complex const*,
                   rocblas_double_complex const* const*,
                   rocblas_double_complex* const*)

#undef INST_HPMV_LAUNCHER

#ifdef INST_HPMV_NUMERICS
#error INST_HPMV_NUMERICS already defined
#endif

#define INST_HPMV_NUMERICS(T_, U_)                                                             \
    template rocblas_status rocblas_hpmv_check_numerics<T_, U_>(const char*    function_name,  \
                                                                rocblas_handle handle,         \
                                                                int64_t        n,              \
                                                                T_             AP,             \
                                                                rocblas_stride offset_a,       \
                                                                rocblas_stride stride_a,       \
                                                                T_             x,              \
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

INST_HPMV_NUMERICS(rocblas_float_complex const*, rocblas_float_complex*)
INST_HPMV_NUMERICS(rocblas_double_complex const*, rocblas_double_complex*)
INST_HPMV_NUMERICS(rocblas_float_complex const* const*, rocblas_float_complex* const*)
INST_HPMV_NUMERICS(rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INST_HPMV_NUMERICS

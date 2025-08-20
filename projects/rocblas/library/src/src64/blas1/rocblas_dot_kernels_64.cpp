/* ************************************************************************
 * Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "int64_helpers.hpp"
#include "rocblas.h"
#include "rocblas_block_sizes.h"
#include "rocblas_dot_64.hpp"

#include "blas1/rocblas_dot.hpp" // int32 API called
#include "blas1/rocblas_dot_kernels.hpp"

#include <cassert>

// assume workspace has already been allocated, recommended for repeated calling of dot_strided_batched product
// routine
template <typename API_INT, int NB, bool CONJ, typename T, typename U, typename V>
rocblas_status rocblas_internal_dot_launcher_64(rocblas_handle __restrict__ handle,
                                                int64_t n_64,
                                                const U __restrict__ x,
                                                rocblas_stride offsetx,
                                                int64_t        incx_64,
                                                rocblas_stride stridex,
                                                const U __restrict__ y,
                                                rocblas_stride offsety,
                                                int64_t        incy_64,
                                                rocblas_stride stridey,
                                                int64_t        batch_count_64,
                                                T* __restrict__ results,
                                                V* __restrict__ workspace)
{

    // Original launcher may only provide partial results
    // Second reduction used for all iterative results within a batch
    // batch_count loops are independent results so don't require additional reduction

    // Quick return if possible.
    if(n_64 <= 0 || batch_count_64 == 0)
    {
        if(handle->is_device_memory_size_query())
            return rocblas_status_size_unchanged;
        else if(handle->pointer_mode == rocblas_pointer_mode_device && batch_count_64 > 0)
        {
            RETURN_IF_HIP_ERROR(
                hipMemsetAsync(&results[0], 0, batch_count_64 * sizeof(T), handle->get_stream()));
        }
        else
        {
            memset(&results[0], 0, batch_count_64 * sizeof(T));
        }

        return rocblas_status_success;
    }

    static constexpr int WIN      = rocblas_dot_WIN<T>();
    int64_t              n_passes = (n_64 - 1) / c_i64_grid_X_chunk + 1;

    if(std::abs(incx_64) <= c_ILP64_i32_max && std::abs(incy_64) <= c_ILP64_i32_max)
    {
        if(n_64 <= c_ILP64_i32_max && batch_count_64 < c_i64_grid_YZ_chunk)
        {
            // valid to use original 32bit API with truncated 64bit args
            return rocblas_internal_dot_launcher<rocblas_int, NB, CONJ, T, U, V>(handle,
                                                                                 n_64,
                                                                                 x,
                                                                                 offsetx,
                                                                                 incx_64,
                                                                                 stridex,
                                                                                 y,
                                                                                 offsety,
                                                                                 incy_64,
                                                                                 stridey,
                                                                                 batch_count_64,
                                                                                 results,
                                                                                 workspace);
        }

        for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
        {
            auto    x_ptr       = adjust_ptr_batch(x, b_base, stridex);
            auto    y_ptr       = adjust_ptr_batch(y, b_base, stridey);
            int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

            T* output = &results[b_base];
            if(handle->pointer_mode == rocblas_pointer_mode_host)
            {
                output = (T*)(workspace);
            }

            // additional workspace if n_passes as results are only partial sum
            V* partial_results = (V*)(workspace);

            for(int64_t n_base = 0, pass = 0; n_base < n_64; n_base += c_i64_grid_X_chunk, pass++)
            {
                int32_t n = int32_t(std::min(n_64 - n_base, c_i64_grid_X_chunk));

                int64_t shiftx
                    = offsetx + (incx_64 < 0 ? -incx_64 * (n_64 - n - n_base) : n_base * incx_64);
                int64_t shifty
                    = offsety + (incy_64 < 0 ? -incy_64 * (n_64 - n - n_base) : n_base * incy_64);

                // 32bit API call but with new instantiation for V, U, V for high precision output
                rocblas_status status
                    = rocblas_internal_dot_launcher<rocblas_int, NB, CONJ, V, U, V>(
                        handle,
                        n,
                        x_ptr,
                        shiftx,
                        int32_t(incx_64),
                        stridex,
                        y_ptr,
                        shifty,
                        int32_t(incy_64),
                        stridey,
                        batch_count,
                        partial_results + pass * batch_count,
                        workspace + n_passes * batch_count);
                if(status != rocblas_status_success)
                    return status;
            }
            // reduce n partitions within batch chunk

            // sum partial_results to results always needed if only to down convert
            ROCBLAS_LAUNCH_KERNEL(
                (rocblas_reduction_kernel_part2<NB, WIN, rocblas_finalize_identity>),
                dim3(batch_count),
                dim3(NB),
                0,
                handle->get_stream(),
                n_passes,
                partial_results,
                output);

            if(handle->pointer_mode == rocblas_pointer_mode_host)
            {
                RETURN_IF_HIP_ERROR(hipMemcpyAsync(&results[b_base],
                                                   output,
                                                   sizeof(T) * batch_count,
                                                   hipMemcpyDeviceToHost,
                                                   handle->get_stream()));
            }

        } // for chunk of batches
    } // small incx/y
    else
    { // int64_t incx/y

        static constexpr int single_block_threshold = rocblas_dot_one_block_threshold<T>();

        if(n_64 > single_block_threshold)
        {
            // non single block closer to form above reusing 32bit API

            static constexpr bool ONE_BLOCK = false;

            for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
            {
                auto    x_ptr = adjust_ptr_batch(x, b_base, stridex);
                auto    y_ptr = adjust_ptr_batch(y, b_base, stridey);
                int32_t batch_count
                    = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

                T* output = &results[b_base];
                if(handle->pointer_mode == rocblas_pointer_mode_host)
                {
                    output = (T*)(workspace);
                }

                // additional workspace if n_passes as results are only partial sum
                V* partial_results = (V*)(workspace);

                {
                    // defeat synch copy of partial results by pushing pointer_mode_device

                    // cppcheck-suppress unreadVariable
                    auto saved_pointer_mode
                        = handle->push_pointer_mode(rocblas_pointer_mode_device);

                    for(int64_t n_base = 0, pass = 0; n_base < n_64;
                        n_base += c_i64_grid_X_chunk, pass++)
                    {
                        int32_t n      = int32_t(std::min(n_64 - n_base, c_i64_grid_X_chunk));
                        int64_t blocks = rocblas_reduction_kernel_block_count(n, NB * WIN);

                        // in case of negative inc only partial sums are doing summation in that direction
                        int64_t shiftx
                            = offsetx
                              + (incx_64 < 0 ? -incx_64 * (n_64 - n - n_base) : n_base * incx_64);
                        int64_t shifty
                            = offsety
                              + (incy_64 < 0 ? -incy_64 * (n_64 - n - n_base) : n_base * incy_64);

                        // 32bit template call but with new instantiation for V, U, V for high precision output
                        rocblas_status status
                            = rocblas_internal_dot_launcher<int64_t, NB, CONJ, V, U, V>(
                                handle,
                                n,
                                x_ptr,
                                shiftx,
                                (incx_64),
                                stridex,
                                y_ptr,
                                shifty,
                                (incy_64),
                                stridey,
                                batch_count,
                                partial_results + pass * batch_count,
                                workspace + n_passes * batch_count);
                        if(status != rocblas_status_success)
                            return status;

                    } // for n

                } // original pointer mode

                // reduce n partitions within batch chunk
                // sum partial_results to results always needed as may down convert
                ROCBLAS_LAUNCH_KERNEL(
                    (rocblas_reduction_kernel_part2<NB, WIN, rocblas_finalize_identity>),
                    dim3(batch_count),
                    dim3(NB),
                    0,
                    handle->get_stream(),
                    n_passes,
                    partial_results,
                    output);

                if(handle->pointer_mode == rocblas_pointer_mode_host)
                {
                    RETURN_IF_HIP_ERROR(hipMemcpyAsync(&results[b_base],
                                                       output,
                                                       sizeof(T) * batch_count,
                                                       hipMemcpyDeviceToHost,
                                                       handle->get_stream()));
                }

            } // for batch
        } // not single block
        else
        {
            // single block so no n looping required

            // we only reduce the block count to 1 so safe to ignore extra workspace allocated in caller
            int32_t n = n_64;

            // in case of negative inc shift pointer to end of data for negative indexing tid*inc
            int64_t shiftx = incx_64 < 0 ? offsetx - (incx_64) * (n - 1) : offsetx;
            int64_t shifty = incx_64 < 0 ? offsety - (incx_64) * (n - 1) : offsety;

            static constexpr int NB_OB  = 1024;
            static constexpr int WIN_OB = 32; // 32K max n threshold, assert guard below

            int64_t blocks = rocblas_reduction_kernel_block_count(n, NB_OB * WIN_OB);
            assert(blocks == 1);
            static constexpr bool ONE_BLOCK = true;

            for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
            {
                auto    x_ptr = adjust_ptr_batch(x, b_base, stridex);
                auto    y_ptr = adjust_ptr_batch(y, b_base, stridey);
                int32_t batch_count
                    = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

                T* output = &results[b_base];
                if(handle->pointer_mode == rocblas_pointer_mode_host)
                {
                    output = (T*)(workspace);
                }

                {
                    // defeat synch copy of partial results by pushing pointer_mode_device
                    auto saved_pointer_mode
                        = handle->push_pointer_mode(rocblas_pointer_mode_device);

                    rocblas_status status
                        = rocblas_internal_dot_launcher<int64_t, NB, CONJ, T, U, V>(handle,
                                                                                    n,
                                                                                    x_ptr,
                                                                                    shiftx,
                                                                                    (incx_64),
                                                                                    stridex,
                                                                                    y_ptr,
                                                                                    shifty,
                                                                                    (incy_64),
                                                                                    stridey,
                                                                                    batch_count,
                                                                                    output,
                                                                                    workspace);
                    if(status != rocblas_status_success)
                        return status;

                } // pointer mode override

                if(handle->pointer_mode == rocblas_pointer_mode_host)
                {
                    // non sync as partial results
                    RETURN_IF_HIP_ERROR(hipMemcpyAsync(&results[b_base],
                                                       output,
                                                       sizeof(T) * batch_count,
                                                       hipMemcpyDeviceToHost,
                                                       handle->get_stream()));
                }
            } // for batch
        } // single block
    }

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        // sync here to match legacy BLAS
        hipStreamSynchronize(handle->get_stream());
    }

    return rocblas_status_success;
}

template <typename T, typename Tex>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_dot_template_64(rocblas_handle __restrict__ handle,
                                     int64_t n,
                                     const T* __restrict__ x,
                                     rocblas_stride offsetx,
                                     int64_t        incx,
                                     rocblas_stride stridex,
                                     const T* __restrict__ y,
                                     rocblas_stride offsety,
                                     int64_t        incy,
                                     rocblas_stride stridey,
                                     int64_t        batch_count,
                                     T* __restrict__ results,
                                     Tex* __restrict__ workspace)
{
    return rocblas_internal_dot_launcher_64<int64_t, ROCBLAS_DOT_NB, false>(handle,
                                                                            n,
                                                                            x,
                                                                            offsetx,
                                                                            incx,
                                                                            stridex,
                                                                            y,
                                                                            offsety,
                                                                            incy,
                                                                            stridey,
                                                                            batch_count,
                                                                            results,
                                                                            workspace);
}

template <typename T, typename Tex>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_dotc_template_64(rocblas_handle __restrict__ handle,
                                      int64_t n,
                                      const T* __restrict__ x,
                                      rocblas_stride offsetx,
                                      int64_t        incx,
                                      rocblas_stride stridex,
                                      const T* __restrict__ y,
                                      rocblas_stride offsety,
                                      int64_t        incy,
                                      rocblas_stride stridey,
                                      int64_t        batch_count,
                                      T* __restrict__ results,
                                      Tex* __restrict__ workspace)
{
    return rocblas_internal_dot_launcher_64<int64_t, ROCBLAS_DOT_NB, true>(handle,
                                                                           n,
                                                                           x,
                                                                           offsetx,
                                                                           incx,
                                                                           stridex,
                                                                           y,
                                                                           offsety,
                                                                           incy,
                                                                           stridey,
                                                                           batch_count,
                                                                           results,
                                                                           workspace);
}

template <typename T, typename Tex>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_dot_batched_template_64(rocblas_handle __restrict__ handle,
                                             int64_t n,
                                             const T* const* __restrict__ x,
                                             rocblas_stride offsetx,
                                             int64_t        incx,
                                             rocblas_stride stridex,
                                             const T* const* __restrict__ y,
                                             rocblas_stride offsety,
                                             int64_t        incy,
                                             rocblas_stride stridey,
                                             int64_t        batch_count,
                                             T* __restrict__ results,
                                             Tex* __restrict__ workspace)
{
    return rocblas_internal_dot_launcher_64<int64_t, ROCBLAS_DOT_NB, false>(handle,
                                                                            n,
                                                                            x,
                                                                            offsetx,
                                                                            incx,
                                                                            stridex,
                                                                            y,
                                                                            offsety,
                                                                            incy,
                                                                            stridey,
                                                                            batch_count,
                                                                            results,
                                                                            workspace);
}

template <typename T, typename Tex>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_dotc_batched_template_64(rocblas_handle __restrict__ handle,
                                              int64_t n,
                                              const T* const* __restrict__ x,
                                              rocblas_stride offsetx,
                                              int64_t        incx,
                                              rocblas_stride stridex,
                                              const T* const* __restrict__ y,
                                              rocblas_stride offsety,
                                              int64_t        incy,
                                              rocblas_stride stridey,
                                              int64_t        batch_count,
                                              T* __restrict__ results,
                                              Tex* __restrict__ workspace)
{
    return rocblas_internal_dot_launcher_64<int64_t, ROCBLAS_DOT_NB, true>(handle,
                                                                           n,
                                                                           x,
                                                                           offsetx,
                                                                           incx,
                                                                           stridex,
                                                                           y,
                                                                           offsety,
                                                                           incy,
                                                                           stridey,
                                                                           batch_count,
                                                                           results,
                                                                           workspace);
}

// If there are any changes in template parameters in the files *dot*.cpp
// instantiations below will need to be manually updated to match the changes.

// clang-format off
#ifdef INST_DOT_TEMPLATE_64
#error INST_DOT_TEMPLATE_64 already defined
#endif

#define INST_DOT_TEMPLATE_64(T_, Tex_)                                          \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                           \
        rocblas_internal_dot_template_64<T_, Tex_>(rocblas_handle __restrict__ handle, \
                                                   int64_t n,                          \
                                                   const T_* __restrict__ x,           \
                                                   rocblas_stride offsetx,             \
                                                   int64_t        incx,                \
                                                   rocblas_stride stridex,             \
                                                   const T_* __restrict__ y,           \
                                                   rocblas_stride offsety,             \
                                                   int64_t        incy,                \
                                                   rocblas_stride stridey,             \
                                                   int64_t        batch_count,         \
                                                   T_* __restrict__ results,           \
                                                   Tex_* __restrict__ workspace)

INST_DOT_TEMPLATE_64(rocblas_half, rocblas_half);
INST_DOT_TEMPLATE_64(rocblas_bfloat16, float);
INST_DOT_TEMPLATE_64(float, float);
INST_DOT_TEMPLATE_64(double, double);
INST_DOT_TEMPLATE_64(rocblas_float_complex, rocblas_float_complex);
INST_DOT_TEMPLATE_64(rocblas_double_complex, rocblas_double_complex);

#undef INST_DOT_TEMPLATE

#ifdef INST_DOTC_TEMPLATE_64
#error INST_DOTC_TEMPLATE_64 already defined
#endif

#define INST_DOTC_TEMPLATE_64(T_, Tex_)                                          \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                            \
        rocblas_internal_dotc_template_64<T_, Tex_>(rocblas_handle __restrict__ handle, \
                                                    int64_t n,                          \
                                                    const T_* __restrict__ x,           \
                                                    rocblas_stride offsetx,             \
                                                    int64_t        incx,                \
                                                    rocblas_stride stridex,             \
                                                    const T_* __restrict__ y,           \
                                                    rocblas_stride offsety,             \
                                                    int64_t        incy,                \
                                                    rocblas_stride stridey,             \
                                                    int64_t        batch_count,         \
                                                    T_* __restrict__ results,           \
                                                    Tex_* __restrict__ workspace)

INST_DOTC_TEMPLATE_64(rocblas_float_complex, rocblas_float_complex);
INST_DOTC_TEMPLATE_64(rocblas_double_complex, rocblas_double_complex);

#undef INST_DOTC_TEMPLATE_64

#ifdef INST_DOT_BATCHED_TEMPLATE_64
#error INST_DOT_BATCHED_TEMPLATE_64 already defined
#endif

#define INST_DOT_BATCHED_TEMPLATE_64(T_, Tex_)                                          \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                                   \
        rocblas_internal_dot_batched_template_64<T_, Tex_>(rocblas_handle __restrict__ handle, \
                                                           int64_t n,                          \
                                                           const T_* const* __restrict__ x,    \
                                                           rocblas_stride offsetx,             \
                                                           int64_t        incx,                \
                                                           rocblas_stride stridex,             \
                                                           const T_* const* __restrict__ y,    \
                                                           rocblas_stride offsety,             \
                                                           int64_t        incy,                \
                                                           rocblas_stride stridey,             \
                                                           int64_t        batch_count,         \
                                                           T_* __restrict__ results,           \
                                                           Tex_* __restrict__ workspace)

INST_DOT_BATCHED_TEMPLATE_64(rocblas_half, rocblas_half);
INST_DOT_BATCHED_TEMPLATE_64(rocblas_bfloat16, float);
INST_DOT_BATCHED_TEMPLATE_64(float, float);
INST_DOT_BATCHED_TEMPLATE_64(double, double);
INST_DOT_BATCHED_TEMPLATE_64(rocblas_float_complex, rocblas_float_complex);
INST_DOT_BATCHED_TEMPLATE_64(rocblas_double_complex, rocblas_double_complex);

#undef INST_DOT_BATCHED_TEMPLATE_64

#ifdef INST_DOTC_BATCHED_TEMPLATE_64
#error INST_DOTC_BATCHED_TEMPLATE_64 already defined
#endif

#define INST_DOTC_BATCHED_TEMPLATE_64(T_, Tex_)                                          \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                                    \
        rocblas_internal_dotc_batched_template_64<T_, Tex_>(rocblas_handle __restrict__ handle, \
                                                            int64_t n,                          \
                                                            const T_* const* __restrict__ x,    \
                                                            rocblas_stride offsetx,             \
                                                            int64_t        incx,                \
                                                            rocblas_stride stridex,             \
                                                            const T_* const* __restrict__ y,    \
                                                            rocblas_stride offsety,             \
                                                            int64_t        incy,                \
                                                            rocblas_stride stridey,             \
                                                            int64_t        batch_count,         \
                                                            T_* __restrict__ results,           \
                                                            Tex_* __restrict__ workspace)

INST_DOTC_BATCHED_TEMPLATE_64(rocblas_float_complex, rocblas_float_complex);
INST_DOTC_BATCHED_TEMPLATE_64(rocblas_double_complex, rocblas_double_complex);

#undef INST_DOTC_BATCHED_TEMPLATE_64

// instantiate _ex forms from 32bit header file define but need high precision output added for multi-pass reductions

// real types are "supported" in dotc_ex
INST_DOT_EX_LAUNCHER(int64_t, ROCBLAS_DOT_NB, true, float, rocblas_half const*, float)
INST_DOT_EX_LAUNCHER(int64_t, ROCBLAS_DOT_NB, true, float, rocblas_half const* const*, float)
INST_DOT_EX_LAUNCHER(int64_t, ROCBLAS_DOT_NB, true, float, rocblas_half const*, rocblas_half)
INST_DOT_EX_LAUNCHER(int64_t, ROCBLAS_DOT_NB, true, float, rocblas_half const* const*, rocblas_half)
INST_DOT_EX_LAUNCHER(int64_t, ROCBLAS_DOT_NB, true, float, rocblas_bfloat16 const*, float)
INST_DOT_EX_LAUNCHER(int64_t, ROCBLAS_DOT_NB, true, float, rocblas_bfloat16 const* const*, float)

// higher precision partial results
INST_DOT_EX_LAUNCHER(int64_t, ROCBLAS_DOT_NB, false, float, rocblas_half const*, float)
INST_DOT_EX_LAUNCHER(int64_t, ROCBLAS_DOT_NB, false, float, rocblas_half const* const*, float)
INST_DOT_EX_LAUNCHER(int64_t, ROCBLAS_DOT_NB, false, float, rocblas_half const*, rocblas_half)
INST_DOT_EX_LAUNCHER(int64_t, ROCBLAS_DOT_NB, false, float, rocblas_half const* const*, rocblas_half)
INST_DOT_EX_LAUNCHER(int64_t, ROCBLAS_DOT_NB, false, float, rocblas_bfloat16 const*, float)
INST_DOT_EX_LAUNCHER(int64_t, ROCBLAS_DOT_NB, false, float, rocblas_bfloat16 const* const*, float)

// for ex interface
#ifdef INST_DOT_EX_LAUNCHER_64
#error INST_DOT_EX_LAUNCHER_64 already defined
#endif

#define INST_DOT_EX_LAUNCHER_64(CONJ_, T_, U_, V_)    \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status      \
        rocblas_internal_dot_launcher_64<int64_t, ROCBLAS_DOT_NB, CONJ_, T_, U_, V_>( \
            rocblas_handle __restrict__ handle,                   \
            int64_t n,                                            \
            U_ __restrict__ x,                                    \
            rocblas_stride offsetx,                               \
            int64_t        incx,                                  \
            rocblas_stride stridex,                               \
            U_ __restrict__ y,                                    \
            rocblas_stride offsety,                               \
            int64_t        incy,                                  \
            rocblas_stride stridey,                               \
            int64_t        batch_count,                           \
            T_* __restrict__ results,                             \
            V_* __restrict__ workspace)

// Mixed precision for dot_ex
INST_DOT_EX_LAUNCHER_64(false, rocblas_half, rocblas_half const*, float);
INST_DOT_EX_LAUNCHER_64(false, rocblas_half, rocblas_half const* const*, float);
INST_DOT_EX_LAUNCHER_64(false, double, float const*, double);
INST_DOT_EX_LAUNCHER_64(false, double, float const* const*, double);

// real types are "supported" in dotc_ex
INST_DOT_EX_LAUNCHER_64(true, rocblas_half, rocblas_half const*, float);
INST_DOT_EX_LAUNCHER_64(true, rocblas_half, rocblas_half const* const*, float);
INST_DOT_EX_LAUNCHER_64(true, rocblas_half, rocblas_half const*, rocblas_half);
INST_DOT_EX_LAUNCHER_64(true, rocblas_half, rocblas_half const* const*, rocblas_half);
INST_DOT_EX_LAUNCHER_64(true, rocblas_bfloat16, rocblas_bfloat16 const*, float);
INST_DOT_EX_LAUNCHER_64(true, rocblas_bfloat16, rocblas_bfloat16 const* const*, float);
INST_DOT_EX_LAUNCHER_64(true, double, float const*, double);
INST_DOT_EX_LAUNCHER_64(true, double, float const* const*, double);
INST_DOT_EX_LAUNCHER_64(true, float, float const*, float);
INST_DOT_EX_LAUNCHER_64(true, float, float const* const*, float);
INST_DOT_EX_LAUNCHER_64(true, double, double const*, double);
INST_DOT_EX_LAUNCHER_64(true, double, double const* const*, double);

#undef INST_DOT_EX_LAUNCHER_64

// clang-format on

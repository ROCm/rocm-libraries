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

#include "blas1/rocblas_asum_nrm2.hpp" // rocblas_int API called

#include "rocblas_asum_nrm2_kernels_64.hpp"

template <typename API_INT,
          int NB,
          typename FETCH,
          typename FINALIZE,
          typename TPtrX,
          typename To,
          typename Tr>
rocblas_status rocblas_internal_asum_nrm2_launcher_64(rocblas_handle handle,
                                                      API_INT        n_64,
                                                      TPtrX          x,
                                                      rocblas_stride shiftx,
                                                      API_INT        incx_64,
                                                      rocblas_stride stridex,
                                                      API_INT        batch_count_64,
                                                      To*            workspace,
                                                      Tr*            result)
{
    int64_t n_passes = (n_64 - 1) / c_i64_grid_X_chunk + 1;

    if(std::abs(incx_64) <= c_i32_max)
    {
        if(n_64 <= c_i32_max && batch_count_64 < c_i64_grid_YZ_chunk)
        {
            // valid to use original 32bit API with truncated 64bit args
            return rocblas_internal_asum_nrm2_launcher<rocblas_int, NB, FETCH, FINALIZE>(
                handle, n_64, x, shiftx, incx_64, stridex, batch_count_64, workspace, result);
        }

        for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
        {
            auto    x_ptr       = adjust_ptr_batch(x, b_base, stridex);
            int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

            Tr* output = &result[b_base];
            if(handle->pointer_mode == rocblas_pointer_mode_host)
            {
                output = (Tr*)(workspace);
            }

            // Additional workspace if n_passes as result are only partial sum
            To* partial_results = (To*)(workspace);

            for(int64_t n_base = 0, pass = 0; n_base < n_64; n_base += c_i64_grid_X_chunk, pass++)
            {
                int32_t n = int32_t(std::min(n_64 - n_base, c_i64_grid_X_chunk));

                int64_t offsetx = shiftx + n_base * incx_64; // negative inc is quick return

                // 32bit API call
                rocblas_status status = rocblas_internal_asum_nrm2_kernel_launcher<false,
                                                                                   rocblas_int,
                                                                                   NB,
                                                                                   FETCH,
                                                                                   FINALIZE>(
                    handle,
                    rocblas_int(n),
                    x_ptr,
                    offsetx,
                    rocblas_int(incx_64),
                    stridex,
                    batch_count,
                    workspace + n_passes * batch_count,
                    partial_results + pass * batch_count);
                if(status != rocblas_status_success)
                    return status;
            }
            // sum final partial_results to output
            ROCBLAS_LAUNCH_KERNEL((rocblas_reduction_kernel_part2_64<true, NB, FINALIZE>),
                                  dim3(batch_count),
                                  dim3(NB),
                                  0,
                                  handle->get_stream(),
                                  n_passes,
                                  partial_results,
                                  output);

            // Transfer final output to result
            if(handle->pointer_mode == rocblas_pointer_mode_host)
            {
                RETURN_IF_HIP_ERROR(hipMemcpyAsync(&result[b_base],
                                                   output,
                                                   sizeof(Tr) * batch_count,
                                                   hipMemcpyDeviceToHost,
                                                   handle->get_stream()));
            }
        }
    }
    else
    {
        for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
        {
            auto    x_ptr       = adjust_ptr_batch(x, b_base, stridex);
            int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

            Tr* output = &result[b_base];
            if(handle->pointer_mode == rocblas_pointer_mode_host)
            {
                output = (Tr*)(workspace);
            }

            // Additional workspace if n_passes as result are only partial sum
            To* partial_results = (To*)(workspace);

            for(int64_t n_base = 0, pass = 0; n_base < n_64; n_base += c_i64_grid_X_chunk, pass++)
            {
                int32_t n = int32_t(std::min(n_64 - n_base, c_i64_grid_X_chunk));

                // in case of negative inc only partial sums are doing summation in that direction
                int64_t offsetx = shiftx + n_base * incx_64; // negative inc is quick return

                // new instantiation for 64bit incx/y
                rocblas_status status = rocblas_internal_asum_nrm2_kernel_launcher<false,
                                                                                   int64_t,
                                                                                   NB,
                                                                                   FETCH,
                                                                                   FINALIZE>(
                    handle,
                    rocblas_int(n),
                    x_ptr,
                    offsetx,
                    incx_64,
                    stridex,
                    batch_count,
                    workspace + n_passes * batch_count,
                    partial_results + pass * batch_count);
                if(status != rocblas_status_success)
                    return status;
            }
            // sum final partial_results to output
            ROCBLAS_LAUNCH_KERNEL((rocblas_reduction_kernel_part2_64<true, NB, FINALIZE>),
                                  dim3(batch_count),
                                  dim3(NB),
                                  0,
                                  handle->get_stream(),
                                  n_passes,
                                  partial_results,
                                  output);

            // Transfer final output to result
            if(handle->pointer_mode == rocblas_pointer_mode_host)
            {
                RETURN_IF_HIP_ERROR(hipMemcpyAsync(&result[b_base],
                                                   output,
                                                   sizeof(Tr) * batch_count,
                                                   hipMemcpyDeviceToHost,
                                                   handle->get_stream()));
            }
        }
    }
    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        // sync here to match legacy BLAS
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->get_stream()));
    }
    return rocblas_status_success;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files nrm2*.cpp

// clang-format off
#ifdef INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER_64
#error INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER_64 already defined
#endif

#define INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER_64(NB_, FETCH_, FINALIZE_, T_, U_, V_)                              \
    template rocblas_status rocblas_internal_asum_nrm2_launcher_64<int64_t, NB_, FETCH_, FINALIZE_, T_, U_, V_>(            \
        rocblas_handle handle,                                                                                              \
        int64_t        n_64,                                                                                                \
        T_             x,                                                                                                   \
        rocblas_stride shiftx,                                                                                              \
        int64_t        incx_64,                                                                                             \
        rocblas_stride stridex,                                                                                             \
        int64_t        batch_count_64,                                                                                      \
        U_*            workspace,                                                                                           \
        V_*            result);


//ASUM instantiations
INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER_64(ROCBLAS_ASUM_NB, rocblas_fetch_asum<float>, rocblas_finalize_identity, float const*, float, float)
INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER_64(ROCBLAS_ASUM_NB, rocblas_fetch_asum<float>, rocblas_finalize_identity, float const* const*, float, float)

INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER_64(ROCBLAS_ASUM_NB, rocblas_fetch_asum<double>, rocblas_finalize_identity, double const*, double, double)
INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER_64(ROCBLAS_ASUM_NB, rocblas_fetch_asum<double>, rocblas_finalize_identity, double const* const*, double, double)

INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER_64(ROCBLAS_ASUM_NB, rocblas_fetch_asum<float>, rocblas_finalize_identity, rocblas_float_complex const*, float, float)
INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER_64(ROCBLAS_ASUM_NB, rocblas_fetch_asum<float>, rocblas_finalize_identity, rocblas_float_complex const* const*, float, float)

INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER_64(ROCBLAS_ASUM_NB, rocblas_fetch_asum<double>, rocblas_finalize_identity, rocblas_double_complex const*, double, double)
INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER_64(ROCBLAS_ASUM_NB, rocblas_fetch_asum<double>, rocblas_finalize_identity, rocblas_double_complex const* const*, double, double)

//nrm2 and nrm2_ex instantiations
INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER_64(ROCBLAS_NRM2_NB, rocblas_fetch_nrm2<float>, rocblas_finalize_nrm2, float const*, float, float)
INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER_64(ROCBLAS_NRM2_NB, rocblas_fetch_nrm2<float>, rocblas_finalize_nrm2, float const* const*, float, float)

INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER_64(ROCBLAS_NRM2_NB, rocblas_fetch_nrm2<double>, rocblas_finalize_nrm2, double const*, double, double)
INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER_64(ROCBLAS_NRM2_NB, rocblas_fetch_nrm2<double>, rocblas_finalize_nrm2, double const* const*, double, double)

INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER_64(ROCBLAS_NRM2_NB, rocblas_fetch_nrm2<float>, rocblas_finalize_nrm2, rocblas_float_complex const*, float, float)
INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER_64(ROCBLAS_NRM2_NB, rocblas_fetch_nrm2<float>, rocblas_finalize_nrm2, rocblas_float_complex const* const*, float, float)

INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER_64(ROCBLAS_NRM2_NB, rocblas_fetch_nrm2<double>, rocblas_finalize_nrm2, rocblas_double_complex const*, double, double)
INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER_64(ROCBLAS_NRM2_NB, rocblas_fetch_nrm2<double>, rocblas_finalize_nrm2, rocblas_double_complex const* const*, double, double)

INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER_64(ROCBLAS_NRM2_NB, rocblas_fetch_nrm2<float>, rocblas_finalize_nrm2, _Float16 const*, float, _Float16)
INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER_64(ROCBLAS_NRM2_NB, rocblas_fetch_nrm2<float>, rocblas_finalize_nrm2, _Float16 const* const*, float, _Float16)

INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER_64(ROCBLAS_NRM2_NB, rocblas_fetch_nrm2<float>, rocblas_finalize_nrm2, rocblas_bfloat16 const*, float, rocblas_bfloat16)
INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER_64(ROCBLAS_NRM2_NB, rocblas_fetch_nrm2<float>, rocblas_finalize_nrm2, rocblas_bfloat16 const* const*, float, rocblas_bfloat16)

#undef INSTANTIATE_ROCBLAS_INTERNAL_ASUM_NRM2_LAUNCHER_64
// clang-format on

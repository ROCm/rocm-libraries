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
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas.h"
#include "rocblas_block_sizes.h"
#include "rocblas_trsv_inverse.hpp"
#include "utility.hpp"

namespace
{
    template <rocblas_int BLOCK, typename T>
    rocblas_status rocblas_trsv_strided_batched_ex_impl(rocblas_handle    handle,
                                                        rocblas_fill      uplo,
                                                        rocblas_operation transA,
                                                        rocblas_diagonal  diag,
                                                        rocblas_int       m,
                                                        const T*          A,
                                                        rocblas_int       lda,
                                                        rocblas_stride    stride_A,
                                                        T*                B,
                                                        rocblas_int       incx,
                                                        rocblas_stride    stride_x,
                                                        rocblas_int       batch_count,
                                                        const T*          supplied_invA = nullptr,
                                                        rocblas_int       supplied_invA_size = 0,
                                                        rocblas_stride    stride_invA        = 0)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        rocblas_internal_logger logger;

        if(!handle->is_device_memory_size_query())
        {
            auto layer_mode = handle->layer_mode;
            if(layer_mode & rocblas_layer_mode_log_trace)
                logger.log_trace(handle,
                                 "rocblas_trsv_strided_batched_ex",
                                 uplo,
                                 transA,
                                 diag,
                                 m,
                                 A,
                                 lda,
                                 stride_A,
                                 B,
                                 incx,
                                 stride_x,
                                 batch_count);

            if(layer_mode & (rocblas_layer_mode_log_bench | rocblas_layer_mode_log_profile))
            {
                auto uplo_letter   = rocblas_fill_letter(uplo);
                auto transA_letter = rocblas_transpose_letter(transA);
                auto diag_letter   = rocblas_diag_letter(diag);

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    if(handle->pointer_mode == rocblas_pointer_mode_host)
                        logger.log_bench(handle,
                                         "./rocblas-bench -f trsv_strided_batched_ex -r",
                                         rocblas_precision_string<T>,
                                         "--uplo",
                                         uplo_letter,
                                         "--transposeA",
                                         transA_letter,
                                         "--diag",
                                         diag_letter,
                                         "-m",
                                         m,
                                         "--lda",
                                         lda,
                                         "--stride_a",
                                         stride_A,
                                         "--incx",
                                         incx,
                                         "--stride_x",
                                         stride_x,
                                         "--batch_count",
                                         batch_count);
                }

                if(layer_mode & rocblas_layer_mode_log_profile)
                    logger.log_profile(handle,
                                       "rocblas_trsv_strided_batched_ex",
                                       "uplo",
                                       uplo_letter,
                                       "transA",
                                       transA_letter,
                                       "diag",
                                       diag_letter,
                                       "M",
                                       m,
                                       "lda",
                                       lda,
                                       "stride_a",
                                       stride_A,
                                       "incx",
                                       incx,
                                       "stride_x",
                                       stride_x,
                                       "batch_count",
                                       batch_count);
            }
        }

        if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
            return rocblas_status_not_implemented;
        if(m < 0 || lda < m || lda < 1 || !incx || batch_count < 0)
            return rocblas_status_invalid_size;

        // quick return if possible.
        if(!m || !batch_count)
        {
            RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);
            return rocblas_status_success;
        }
        if(!handle->is_device_memory_size_query())
        {
            if(!A || !B)
                return rocblas_status_invalid_pointer;
        }

        // Proxy object holds the allocation. It must stay alive as long as mem_* pointers below are alive.
        auto  w_mem = handle->device_malloc(0);
        void* w_mem_x_temp;
        void* w_mem_x_temp_arr;
        void* w_mem_invA;
        void* w_mem_invA_arr;

        rocblas_status perf_status
            = rocblas_internal_trsv_inverse_template_mem<BLOCK, false, T>(handle,
                                                                          m,
                                                                          batch_count,
                                                                          w_mem,
                                                                          w_mem_x_temp,
                                                                          w_mem_x_temp_arr,
                                                                          w_mem_invA,
                                                                          w_mem_invA_arr,
                                                                          supplied_invA,
                                                                          supplied_invA_size);

        // If this was a device memory query or an error occurred, return status
        if(perf_status != rocblas_status_success && perf_status != rocblas_status_perf_degraded)
            return perf_status;

        auto check_numerics = handle->check_numerics;
        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status trsv_check_numerics_status
                = rocblas_internal_trsv_ex_check_numerics("rocblas_trsv_strided_batched_ex",
                                                          handle,
                                                          m,
                                                          A,
                                                          0,
                                                          lda,
                                                          stride_A,
                                                          B,
                                                          0,
                                                          incx,
                                                          stride_x,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);
            if(trsv_check_numerics_status != rocblas_status_success)
                return trsv_check_numerics_status;
        }

        rocblas_status status
            = rocblas_internal_trsv_inverse_template<BLOCK, false, T>(handle,
                                                                      uplo,
                                                                      transA,
                                                                      diag,
                                                                      m,
                                                                      A,
                                                                      0,
                                                                      lda,
                                                                      stride_A,
                                                                      B,
                                                                      0,
                                                                      incx,
                                                                      stride_x,
                                                                      batch_count,
                                                                      w_mem_x_temp,
                                                                      w_mem_x_temp_arr,
                                                                      w_mem_invA,
                                                                      w_mem_invA_arr,
                                                                      supplied_invA,
                                                                      supplied_invA_size,
                                                                      0,
                                                                      stride_invA);

        status = (status != rocblas_status_success) ? status : perf_status;
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status trsv_check_numerics_status
                = rocblas_internal_trsv_ex_check_numerics("rocblas_trsv_strided_batched_ex",
                                                          handle,
                                                          m,
                                                          A,
                                                          0,
                                                          lda,
                                                          stride_A,
                                                          B,
                                                          0,
                                                          incx,
                                                          stride_x,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);
            if(trsv_check_numerics_status != rocblas_status_success)
                return trsv_check_numerics_status;
        }
        return status;
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_trsv_strided_batched_ex(rocblas_handle    handle,
                                               rocblas_fill      uplo,
                                               rocblas_operation transA,
                                               rocblas_diagonal  diag,
                                               rocblas_int       m,
                                               const void*       A,
                                               rocblas_int       lda,
                                               rocblas_stride    stride_A,
                                               void*             x,
                                               rocblas_int       incx,
                                               rocblas_stride    stride_x,
                                               rocblas_int       batch_count,
                                               const void*       invA,
                                               rocblas_int       invA_size,
                                               rocblas_datatype  compute_type)
try
{
    switch(compute_type)
    {
    case rocblas_datatype_f64_r:
        return rocblas_trsv_strided_batched_ex_impl<ROCBLAS_TRSV_EX_NB>(
            handle,
            uplo,
            transA,
            diag,
            m,
            static_cast<const double*>(A),
            lda,
            stride_A,
            static_cast<double*>(x),
            incx,
            stride_x,
            batch_count,
            static_cast<const double*>(invA),
            invA_size);

    case rocblas_datatype_f32_r:
        return rocblas_trsv_strided_batched_ex_impl<ROCBLAS_TRSV_EX_NB>(
            handle,
            uplo,
            transA,
            diag,
            m,
            static_cast<const float*>(A),
            lda,
            stride_A,
            static_cast<float*>(x),
            incx,
            stride_x,
            batch_count,
            static_cast<const float*>(invA),
            invA_size);

    case rocblas_datatype_f64_c:
        return rocblas_trsv_strided_batched_ex_impl<ROCBLAS_TRSV_EX_NB>(
            handle,
            uplo,
            transA,
            diag,
            m,
            static_cast<const rocblas_double_complex*>(A),
            lda,
            stride_A,
            static_cast<rocblas_double_complex*>(x),
            incx,
            stride_x,
            batch_count,
            static_cast<const rocblas_double_complex*>(invA),
            invA_size);

    case rocblas_datatype_f32_c:
        return rocblas_trsv_strided_batched_ex_impl<ROCBLAS_TRSV_EX_NB>(
            handle,
            uplo,
            transA,
            diag,
            m,
            static_cast<const rocblas_float_complex*>(A),
            lda,
            stride_A,
            static_cast<rocblas_float_complex*>(x),
            incx,
            stride_x,
            batch_count,
            static_cast<const rocblas_float_complex*>(invA),
            invA_size);

    default:
        return rocblas_status_not_implemented;
    }
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"

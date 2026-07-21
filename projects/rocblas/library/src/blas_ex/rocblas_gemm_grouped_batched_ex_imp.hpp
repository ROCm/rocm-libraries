/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#pragma once

#include "blas_ex/rocblas_gemm_grouped_batched_ex.hpp"
#include "int64_helpers.hpp"
#include "logging.hpp"

#include <vector>

namespace
{
    template <typename API_INT>
    rocblas_status
        rocblas_copy_alpha_beta_ex_arrays_to_host_if_on_device(rocblas_handle     handle,
                                                               API_INT            group_count,
                                                               const void*        alpha_array,
                                                               const void*        beta_array,
                                                               std::vector<char>& alpha_h,
                                                               std::vector<char>& beta_h,
                                                               const void*&       alpha_host,
                                                               const void*&       beta_host,
                                                               rocblas_datatype   compute_type)
    {
        alpha_host = alpha_array;
        beta_host  = beta_array;

        if(group_count < 0)
            return rocblas_status_invalid_size;
        if(handle->pointer_mode == rocblas_pointer_mode_host || group_count == 0)
            return rocblas_status_success;

        if(!alpha_array || !beta_array)
            return rocblas_status_invalid_pointer;

        const size_t scalar_stride = rocblas_gemm_ex_compute_type_size(compute_type);
        if(scalar_stride == 0)
            return rocblas_status_invalid_value;

        const size_t bytes = static_cast<size_t>(group_count) * scalar_stride;
        alpha_h.resize(bytes);
        beta_h.resize(bytes);
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            alpha_h.data(), alpha_array, bytes, hipMemcpyDeviceToHost, handle->get_stream()));
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            beta_h.data(), beta_array, bytes, hipMemcpyDeviceToHost, handle->get_stream()));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->get_stream()));
        alpha_host = alpha_h.data();
        beta_host  = beta_h.data();
        return rocblas_status_success;
    }

    template <typename API_INT>
    rocblas_status rocblas_gemm_grouped_batched_ex_impl(rocblas_handle           handle,
                                                        const rocblas_operation* transa_array,
                                                        const rocblas_operation* transb_array,
                                                        const API_INT*           m_array,
                                                        const API_INT*           n_array,
                                                        const API_INT*           k_array,
                                                        const void*              alpha_array,
                                                        const void* const        Aarray[],
                                                        rocblas_datatype         a_type,
                                                        const API_INT*           lda_array,
                                                        const void* const        Barray[],
                                                        rocblas_datatype         b_type,
                                                        const API_INT*           ldb_array,
                                                        const void*              beta_array,
                                                        const void* const        Carray[],
                                                        rocblas_datatype         c_type,
                                                        const API_INT*           ldc_array,
                                                        void* const              Darray[],
                                                        rocblas_datatype         d_type,
                                                        const API_INT*           ldd_array,
                                                        API_INT                  group_count,
                                                        const API_INT*           group_size,
                                                        rocblas_datatype         compute_type,
                                                        rocblas_gemm_algo        algo,
                                                        uint32_t                 flags)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        const bool HPA = compute_type == rocblas_datatype_f32_r
                         && (a_type == rocblas_datatype_f16_r || a_type == rocblas_datatype_bf16_r);

        if(!HPA)
            RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        std::vector<char> alpha_h, beta_h;
        const void*       alpha_host = alpha_array;
        const void*       beta_host  = beta_array;
        RETURN_IF_ROCBLAS_ERROR(
            rocblas_copy_alpha_beta_ex_arrays_to_host_if_on_device(handle,
                                                                   group_count,
                                                                   alpha_array,
                                                                   beta_array,
                                                                   alpha_h,
                                                                   beta_h,
                                                                   alpha_host,
                                                                   beta_host,
                                                                   compute_type));
        auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

        rocblas_internal_logger logger;
        if(!handle->is_device_memory_size_query())
        {
            auto layer_mode = handle->layer_mode;
            if(layer_mode
               & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
                  | rocblas_layer_mode_log_profile))
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                    logger.log_trace(handle, ROCBLAS_API_STR(rocblas_gemm_grouped_batched_ex));

                if(layer_mode & rocblas_layer_mode_log_bench)
                    logger.log_bench(handle,
                                     ROCBLAS_API_BENCH " -f gemm_grouped_batched_ex",
                                     "--group_count",
                                     group_count);

                if(layer_mode & rocblas_layer_mode_log_profile)
                    logger.log_profile(handle,
                                       ROCBLAS_API_STR(rocblas_gemm_grouped_batched_ex),
                                       "group_count",
                                       group_count);
            }
        }

        auto validArgs = rocblas_gemm_grouped_batched_ex_arg_check(handle,
                                                                   transa_array,
                                                                   transb_array,
                                                                   m_array,
                                                                   n_array,
                                                                   k_array,
                                                                   alpha_host,
                                                                   Aarray,
                                                                   a_type,
                                                                   lda_array,
                                                                   Barray,
                                                                   b_type,
                                                                   ldb_array,
                                                                   beta_host,
                                                                   Carray,
                                                                   c_type,
                                                                   ldc_array,
                                                                   Darray,
                                                                   d_type,
                                                                   ldd_array,
                                                                   group_count,
                                                                   group_size,
                                                                   compute_type);

        if(validArgs != rocblas_status_continue)
        {
            if(validArgs == rocblas_status_success)
                RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);
            return validArgs;
        }

        return ROCBLAS_API(rocblas_internal_gemm_grouped_batched_ex_template)(handle,
                                                                              transa_array,
                                                                              transb_array,
                                                                              m_array,
                                                                              n_array,
                                                                              k_array,
                                                                              alpha_host,
                                                                              Aarray,
                                                                              a_type,
                                                                              lda_array,
                                                                              Barray,
                                                                              b_type,
                                                                              ldb_array,
                                                                              beta_host,
                                                                              Carray,
                                                                              c_type,
                                                                              ldc_array,
                                                                              Darray,
                                                                              d_type,
                                                                              ldd_array,
                                                                              group_count,
                                                                              group_size,
                                                                              compute_type,
                                                                              algo,
                                                                              0,
                                                                              flags);
    }
}

#define INST_GEMM_GROUPED_BATCHED_EX_C_API(TI_)                                             \
    extern "C" {                                                                            \
    rocblas_status                                                                          \
        ROCBLAS_API(rocblas_gemm_grouped_batched_ex)(rocblas_handle           handle,       \
                                                     const rocblas_operation* transa_array, \
                                                     const rocblas_operation* transb_array, \
                                                     const TI_*               m_array,      \
                                                     const TI_*               n_array,      \
                                                     const TI_*               k_array,      \
                                                     const void*              alpha_array,  \
                                                     const void* const        Aarray[],     \
                                                     rocblas_datatype         a_type,       \
                                                     const TI_*               lda_array,    \
                                                     const void* const        Barray[],     \
                                                     rocblas_datatype         b_type,       \
                                                     const TI_*               ldb_array,    \
                                                     const void*              beta_array,   \
                                                     const void* const        Carray[],     \
                                                     rocblas_datatype         c_type,       \
                                                     const TI_*               ldc_array,    \
                                                     void* const              Darray[],     \
                                                     rocblas_datatype         d_type,       \
                                                     const TI_*               ldd_array,    \
                                                     TI_                      group_count,  \
                                                     const TI_*               group_size,   \
                                                     rocblas_datatype         compute_type, \
                                                     rocblas_gemm_algo        algo,         \
                                                     uint32_t                 flags)        \
    try                                                                                     \
    {                                                                                       \
        return rocblas_gemm_grouped_batched_ex_impl<TI_>(handle,                            \
                                                         transa_array,                      \
                                                         transb_array,                      \
                                                         m_array,                           \
                                                         n_array,                           \
                                                         k_array,                           \
                                                         alpha_array,                       \
                                                         Aarray,                            \
                                                         a_type,                            \
                                                         lda_array,                         \
                                                         Barray,                            \
                                                         b_type,                            \
                                                         ldb_array,                         \
                                                         beta_array,                        \
                                                         Carray,                            \
                                                         c_type,                            \
                                                         ldc_array,                         \
                                                         Darray,                            \
                                                         d_type,                            \
                                                         ldd_array,                         \
                                                         group_count,                       \
                                                         group_size,                        \
                                                         compute_type,                      \
                                                         algo,                              \
                                                         flags);                            \
    }                                                                                       \
    catch(...)                                                                              \
    {                                                                                       \
        return exception_to_rocblas_status();                                               \
    }                                                                                       \
    } // extern "C"

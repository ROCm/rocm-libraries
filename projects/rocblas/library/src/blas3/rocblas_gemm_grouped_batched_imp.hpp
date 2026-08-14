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

#include "blas3/rocblas_gemm_grouped_batched.hpp"
#include "int64_helpers.hpp"
#include "logging.hpp"
#include "rocblas_gemm.hpp"

#include <vector>

namespace
{
    template <typename>
    constexpr char rocblas_gemm_grouped_batched_name[] = "unknown";
    template <>
    constexpr char rocblas_gemm_grouped_batched_name<float>[]
        = ROCBLAS_API_STR(rocblas_sgemm_grouped_batched);
    template <>
    constexpr char rocblas_gemm_grouped_batched_name<double>[]
        = ROCBLAS_API_STR(rocblas_dgemm_grouped_batched);

    template <typename API_INT, typename T>
    rocblas_status rocblas_copy_alpha_beta_arrays_to_host_if_on_device(rocblas_handle  handle,
                                                                       API_INT         group_count,
                                                                       const T*        alpha_array,
                                                                       const T*        beta_array,
                                                                       std::vector<T>& alpha_h,
                                                                       std::vector<T>& beta_h,
                                                                       const T*&       alpha_host,
                                                                       const T*&       beta_host)
    {
        alpha_host = alpha_array;
        beta_host  = beta_array;

        if(group_count < 0)
            return rocblas_status_invalid_size;
        if(handle->pointer_mode == rocblas_pointer_mode_host || group_count == 0)
            return rocblas_status_success;

        if(!alpha_array || !beta_array)
            return rocblas_status_invalid_pointer;

        alpha_h.resize(group_count);
        beta_h.resize(group_count);
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(alpha_h.data(),
                                           alpha_array,
                                           group_count * sizeof(T),
                                           hipMemcpyDeviceToHost,
                                           handle->get_stream()));
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(beta_h.data(),
                                           beta_array,
                                           group_count * sizeof(T),
                                           hipMemcpyDeviceToHost,
                                           handle->get_stream()));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->get_stream()));
        alpha_host = alpha_h.data();
        beta_host  = beta_h.data();
        return rocblas_status_success;
    }

    template <typename API_INT, typename T>
    rocblas_status rocblas_gemm_grouped_batched_impl(rocblas_handle           handle,
                                                     const rocblas_operation* transa_array,
                                                     const rocblas_operation* transb_array,
                                                     const API_INT*           m_array,
                                                     const API_INT*           n_array,
                                                     const API_INT*           k_array,
                                                     const T*                 alpha_array,
                                                     const T* const           Aarray[],
                                                     const API_INT*           lda_array,
                                                     const T* const           Barray[],
                                                     const API_INT*           ldb_array,
                                                     const T*                 beta_array,
                                                     T* const                 Carray[],
                                                     const API_INT*           ldc_array,
                                                     API_INT                  group_count,
                                                     const API_INT*           group_size)
    {
        if(!handle)
            return rocblas_status_invalid_handle;
        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        std::vector<T> alpha_h, beta_h;
        const T*       alpha_host = alpha_array;
        const T*       beta_host  = beta_array;
        RETURN_IF_ROCBLAS_ERROR(rocblas_copy_alpha_beta_arrays_to_host_if_on_device(
            handle, group_count, alpha_array, beta_array, alpha_h, beta_h, alpha_host, beta_host));
        auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

        auto                    layer_mode     = handle->layer_mode;
        auto                    check_numerics = handle->check_numerics;
        rocblas_internal_logger logger;
        if(layer_mode
           & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
              | rocblas_layer_mode_log_profile))
        {
            if(layer_mode & rocblas_layer_mode_log_trace)
                logger.log_trace(handle,
                                 rocblas_gemm_grouped_batched_name<T>,
                                 transa_array,
                                 transb_array,
                                 m_array,
                                 n_array,
                                 k_array,
                                 alpha_host,
                                 Aarray,
                                 lda_array,
                                 Barray,
                                 ldb_array,
                                 beta_host,
                                 Carray,
                                 ldc_array,
                                 group_count,
                                 group_size);

            if(layer_mode & rocblas_layer_mode_log_bench)
                logger.log_bench(handle,
                                 ROCBLAS_API_BENCH " -f gemm_grouped_batched -r",
                                 rocblas_precision_string<T>,
                                 "--group_count",
                                 group_count);

            if(layer_mode & rocblas_layer_mode_log_profile)
                logger.log_profile(
                    handle, rocblas_gemm_grouped_batched_name<T>, "group_count", group_count);
        }

        auto validArgs = rocblas_gemm_grouped_batched_arg_check(handle,
                                                                transa_array,
                                                                transb_array,
                                                                m_array,
                                                                n_array,
                                                                k_array,
                                                                alpha_host,
                                                                Aarray,
                                                                lda_array,
                                                                Barray,
                                                                ldb_array,
                                                                beta_host,
                                                                Carray,
                                                                ldc_array,
                                                                group_count,
                                                                group_size);

        if(validArgs != rocblas_status_continue)
            return validArgs;

        if(check_numerics)
        {
            int64_t idx = 0;
            for(API_INT g = 0; g < group_count; ++g)
            {
                bool           is_input = true;
                rocblas_status gemm_check_numerics_status
                    = rocblas_gemm_check_numerics(rocblas_gemm_grouped_batched_name<T>,
                                                  handle,
                                                  transa_array[g],
                                                  transb_array[g],
                                                  m_array[g],
                                                  n_array[g],
                                                  k_array[g],
                                                  Aarray + idx,
                                                  0,
                                                  lda_array[g],
                                                  0,
                                                  Barray + idx,
                                                  0,
                                                  ldb_array[g],
                                                  0,
                                                  Carray + idx,
                                                  0,
                                                  ldc_array[g],
                                                  0,
                                                  group_size[g],
                                                  check_numerics,
                                                  is_input);
                if(gemm_check_numerics_status != rocblas_status_success)
                    return gemm_check_numerics_status;
                idx += group_size[g];
            }
        }

        rocblas_status status
            = ROCBLAS_API(rocblas_internal_gemm_grouped_batched_template)(handle,
                                                                          transa_array,
                                                                          transb_array,
                                                                          m_array,
                                                                          n_array,
                                                                          k_array,
                                                                          alpha_host,
                                                                          Aarray,
                                                                          lda_array,
                                                                          Barray,
                                                                          ldb_array,
                                                                          beta_host,
                                                                          Carray,
                                                                          ldc_array,
                                                                          group_count,
                                                                          group_size);

        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            int64_t idx = 0;
            for(API_INT g = 0; g < group_count; ++g)
            {
                bool           is_input = false;
                rocblas_status gemm_check_numerics_status
                    = rocblas_gemm_check_numerics(rocblas_gemm_grouped_batched_name<T>,
                                                  handle,
                                                  transa_array[g],
                                                  transb_array[g],
                                                  m_array[g],
                                                  n_array[g],
                                                  k_array[g],
                                                  Aarray + idx,
                                                  0,
                                                  lda_array[g],
                                                  0,
                                                  Barray + idx,
                                                  0,
                                                  ldb_array[g],
                                                  0,
                                                  Carray + idx,
                                                  0,
                                                  ldc_array[g],
                                                  0,
                                                  group_size[g],
                                                  check_numerics,
                                                  is_input);
                if(gemm_check_numerics_status != rocblas_status_success)
                    return gemm_check_numerics_status;
                idx += group_size[g];
            }
        }

        return status;
    }
}

#ifdef IMPL_GROUPED
#error IMPL_GROUPED ALREADY DEFINED
#endif

#define IMPL_GROUPED(routine_name_, TI_, T_)                            \
    rocblas_status routine_name_(rocblas_handle           handle,       \
                                 const rocblas_operation* transa_array, \
                                 const rocblas_operation* transb_array, \
                                 const TI_*               m_array,      \
                                 const TI_*               n_array,      \
                                 const TI_*               k_array,      \
                                 const T_*                alpha_array,  \
                                 const T_* const          Aarray[],     \
                                 const TI_*               lda_array,    \
                                 const T_* const          Barray[],     \
                                 const TI_*               ldb_array,    \
                                 const T_*                beta_array,   \
                                 T_* const                Carray[],     \
                                 const TI_*               ldc_array,    \
                                 TI_                      group_count,  \
                                 const TI_*               group_size)   \
    try                                                                 \
    {                                                                   \
        return rocblas_gemm_grouped_batched_impl<TI_, T_>(handle,       \
                                                          transa_array, \
                                                          transb_array, \
                                                          m_array,      \
                                                          n_array,      \
                                                          k_array,      \
                                                          alpha_array,  \
                                                          Aarray,       \
                                                          lda_array,    \
                                                          Barray,       \
                                                          ldb_array,    \
                                                          beta_array,   \
                                                          Carray,       \
                                                          ldc_array,    \
                                                          group_count,  \
                                                          group_size);  \
    }                                                                   \
    catch(...)                                                          \
    {                                                                   \
        return exception_to_rocblas_status();                           \
    }

#define INST_GEMM_GROUPED_BATCHED_C_API(TI_)                               \
    extern "C" {                                                           \
    IMPL_GROUPED(ROCBLAS_API(rocblas_sgemm_grouped_batched), TI_, float);  \
    IMPL_GROUPED(ROCBLAS_API(rocblas_dgemm_grouped_batched), TI_, double); \
    } // extern "C"

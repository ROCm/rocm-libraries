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
#pragma once

#include "int64_helpers.hpp"
#include "logging.hpp"
#include "rocblas_block_sizes.h"
#include "rocblas_dot_ex.hpp"

namespace
{
    template <typename API_INT, bool CONJ>
    rocblas_status rocblas_dot_ex_impl(rocblas_handle   handle,
                                       API_INT          n,
                                       const void*      x,
                                       rocblas_datatype x_type,
                                       API_INT          incx,
                                       const void*      y,
                                       rocblas_datatype y_type,
                                       API_INT          incy,
                                       void*            result,
                                       rocblas_datatype result_type,
                                       rocblas_datatype execution_type,
                                       const char*      name,
                                       const char*      bench_name)
    {
        if(!handle)
        {
            return rocblas_status_invalid_handle;
        }

        size_t dev_bytes = rocblas_reduction_workspace_size<API_INT, ROCBLAS_DOT_NB>(
            n, incx, incy, 1, execution_type);
        if(handle->is_device_memory_size_query())
        {
            if(n <= 0)
                return rocblas_status_size_unchanged;
            else
                return handle->set_optimal_device_memory_size(dev_bytes);
        }

        auto                    layer_mode = handle->layer_mode;
        rocblas_internal_logger logger;
        if(layer_mode
           & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
              | rocblas_layer_mode_log_profile))
        {
            auto x_type_str      = rocblas_datatype_string(x_type);
            auto y_type_str      = rocblas_datatype_string(y_type);
            auto result_type_str = rocblas_datatype_string(result_type);
            auto ex_type_str     = rocblas_datatype_string(execution_type);

            if(layer_mode & rocblas_layer_mode_log_trace)
            {
                logger.log_trace(handle,
                                 name,
                                 n,
                                 x,
                                 x_type_str,
                                 incx,
                                 y,
                                 y_type_str,
                                 incy,
                                 result_type_str,
                                 ex_type_str);
            }

            if(layer_mode & rocblas_layer_mode_log_bench)
            {
                logger.log_bench(handle,
                                 ROCBLAS_API_BENCH " -f",
                                 bench_name,
                                 "-n",
                                 n,
                                 "--a_type",
                                 x_type_str,
                                 "--incx",
                                 incx,
                                 "--b_type",
                                 y_type_str,
                                 "--incy",
                                 incy,
                                 "--c_type",
                                 result_type_str,
                                 "--compute_type",
                                 ex_type_str);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
            {
                logger.log_profile(handle,
                                   name,
                                   "N",
                                   n,
                                   "a_type",
                                   x_type_str,
                                   "incx",
                                   incx,
                                   "b_type",
                                   y_type_str,
                                   "incy",
                                   incy,
                                   "c_type",
                                   result_type_str,
                                   "compute_type",
                                   ex_type_str);
            }
        }

        if(n <= 0)
        {
            if(!result)
                return rocblas_status_invalid_pointer;
            if(rocblas_pointer_mode_device == handle->pointer_mode)
                RETURN_IF_HIP_ERROR(hipMemsetAsync(
                    result, 0, rocblas_sizeof_datatype(result_type), handle->get_stream()));
            else
                memset(result, 0, rocblas_sizeof_datatype(result_type));
            return rocblas_status_success;
        }

        if(!x || !y || !result)
            return rocblas_status_invalid_pointer;

        auto w_mem = handle->device_malloc(dev_bytes);
        if(!w_mem)
            return rocblas_status_memory_error;

        static constexpr rocblas_int    batch_count_1 = 1;
        static constexpr rocblas_stride stride_0      = 0;
        return rocblas_dot_ex_template<API_INT, ROCBLAS_DOT_NB, false, CONJ>(handle,
                                                                             n,
                                                                             x,
                                                                             x_type,
                                                                             incx,
                                                                             stride_0,
                                                                             y,
                                                                             y_type,
                                                                             incy,
                                                                             stride_0,
                                                                             batch_count_1,
                                                                             result,
                                                                             result_type,
                                                                             execution_type,
                                                                             (void*)w_mem);
    }

}

#ifdef IMPL
#error IMPL ALREADY DEFINED
#endif

#define IMPL(name_, TI_, CONJ_, func_name_, bench_name_)           \
    rocblas_status name_(rocblas_handle   handle,                  \
                         TI_              n,                       \
                         const void*      x,                       \
                         rocblas_datatype x_type,                  \
                         TI_              incx,                    \
                         const void*      y,                       \
                         rocblas_datatype y_type,                  \
                         TI_              incy,                    \
                         void*            result,                  \
                         rocblas_datatype result_type,             \
                         rocblas_datatype execution_type)          \
    {                                                              \
        try                                                        \
        {                                                          \
            return rocblas_dot_ex_impl<TI_, CONJ_>(handle,         \
                                                   n,              \
                                                   x,              \
                                                   x_type,         \
                                                   incx,           \
                                                   y,              \
                                                   y_type,         \
                                                   incy,           \
                                                   result,         \
                                                   result_type,    \
                                                   execution_type, \
                                                   func_name_,     \
                                                   bench_name_);   \
        }                                                          \
        catch(...)                                                 \
        {                                                          \
            return exception_to_rocblas_status();                  \
        }                                                          \
    }

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

#define INST_DOT_EX_C_API(TI_)             \
    extern "C" {                           \
    IMPL(ROCBLAS_API(rocblas_dot_ex),      \
         TI_,                              \
         false,                            \
         ROCBLAS_API_STR(rocblas_dot_ex),  \
         ROCBLAS_API_STR(dot_ex));         \
    IMPL(ROCBLAS_API(rocblas_dotc_ex),     \
         TI_,                              \
         true,                             \
         ROCBLAS_API_STR(rocblas_dotc_ex), \
         ROCBLAS_API_STR(dotc_ex));        \
    } // extern "C"

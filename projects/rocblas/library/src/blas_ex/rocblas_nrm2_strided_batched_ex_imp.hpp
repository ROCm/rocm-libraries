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

#include "handle.hpp"
#include "int64_helpers.hpp"
#include "logging.hpp"
#include "rocblas.h"
#include "rocblas_block_sizes.h"
#include "rocblas_nrm2_ex.hpp"
#include "utility.hpp"

namespace
{
    // allocate workspace inside this API
    template <typename API_INT, rocblas_int NB>
    rocblas_status rocblas_nrm2_strided_batched_ex_impl(rocblas_handle   handle,
                                                        API_INT          n,
                                                        const void*      x,
                                                        rocblas_datatype x_type,
                                                        API_INT          incx,
                                                        rocblas_stride   stride_x,
                                                        API_INT          batch_count,
                                                        void*            results,
                                                        rocblas_datatype result_type,
                                                        rocblas_datatype execution_type)
    {
        if(!handle)
        {
            return rocblas_status_invalid_handle;
        }

        size_t dev_bytes = rocblas_reduction_workspace_size<API_INT, NB>(
            n, incx, incx, batch_count, execution_type);

        if(handle->is_device_memory_size_query())
        {
            if(n <= 0 || incx <= 0 || batch_count <= 0)
            {
                return rocblas_status_size_unchanged;
            }
            else
            {
                return handle->set_optimal_device_memory_size(dev_bytes);
            }
        }

        auto                    x_type_str      = rocblas_datatype_string(x_type);
        auto                    result_type_str = rocblas_datatype_string(result_type);
        auto                    ex_type_str     = rocblas_datatype_string(execution_type);
        auto                    layer_mode      = handle->layer_mode;
        rocblas_internal_logger logger;
        if(layer_mode & rocblas_layer_mode_log_trace)
        {
            logger.log_trace(handle,
                             ROCBLAS_API_STR(nrm2_strided_batched_ex),
                             n,
                             x,
                             x_type_str,
                             incx,
                             stride_x,
                             result_type_str,
                             batch_count,
                             ex_type_str);
        }

        if(layer_mode & rocblas_layer_mode_log_bench)
        {
            logger.log_bench(
                handle,
                ROCBLAS_API_BENCH " -f nrm2_strided_batched_ex",
                "-n",
                n,
                "--incx",
                incx,
                "--stride_x",
                stride_x,
                "--batch_count",
                batch_count,
                rocblas_internal_log_bench_ex_precisions(x_type, result_type, execution_type));
        }

        if(layer_mode & rocblas_layer_mode_log_profile)
        {
            logger.log_profile(handle,
                               ROCBLAS_API_STR(nrm2_strided_batched_ex),
                               "N",
                               n,
                               "a_type",
                               x_type_str,
                               "incx",
                               incx,
                               "stride_x",
                               stride_x,
                               "b_type",
                               result_type_str,
                               "batch_count",
                               batch_count,
                               "compute_type",
                               ex_type_str);
        }

        if(!results)
        {
            return rocblas_status_invalid_pointer;
        }

        // Quick return if possible.
        if(n <= 0 || incx <= 0 || batch_count <= 0)
        {
            if(rocblas_pointer_mode_device == handle->pointer_mode)
            {
                if(batch_count > 0)
                    RETURN_IF_HIP_ERROR(
                        hipMemsetAsync(results,
                                       0,
                                       rocblas_sizeof_datatype(result_type) * batch_count,
                                       handle->get_stream()));
            }
            else
            {
                if(batch_count > 0)
                    memset(results, 0, rocblas_sizeof_datatype(result_type) * batch_count);
            }
            return rocblas_status_success;
        }

        if(!x)
        {
            return rocblas_status_invalid_pointer;
        }

        auto w_mem = handle->device_malloc(dev_bytes);
        if(!w_mem)
        {
            return rocblas_status_memory_error;
        }

        static constexpr bool           isbatched = false;
        static constexpr rocblas_stride shiftx_0  = 0;

        return rocblas_nrm2_ex_template<API_INT, NB, isbatched>(handle,
                                                                n,
                                                                x,
                                                                x_type,
                                                                shiftx_0,
                                                                incx,
                                                                stride_x,
                                                                batch_count,
                                                                results,
                                                                result_type,
                                                                execution_type,
                                                                (void*)w_mem);
    }

} // namespace

/* ============================================================================================ */

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

#define INST_NRM2_STRIDED_BATCHED_EX_C_API(TI_)                                                  \
    extern "C" {                                                                                 \
    rocblas_status ROCBLAS_API(rocblas_nrm2_strided_batched_ex)(rocblas_handle   handle,         \
                                                                TI_              n,              \
                                                                const void*      x,              \
                                                                rocblas_datatype x_type,         \
                                                                TI_              incx,           \
                                                                rocblas_stride   stride_x,       \
                                                                TI_              batch_count,    \
                                                                void*            results,        \
                                                                rocblas_datatype result_type,    \
                                                                rocblas_datatype execution_type) \
    {                                                                                            \
        try                                                                                      \
        {                                                                                        \
            return rocblas_nrm2_strided_batched_ex_impl<TI_, ROCBLAS_NRM2_NB>(handle,            \
                                                                              n,                 \
                                                                              x,                 \
                                                                              x_type,            \
                                                                              incx,              \
                                                                              stride_x,          \
                                                                              batch_count,       \
                                                                              results,           \
                                                                              result_type,       \
                                                                              execution_type);   \
        }                                                                                        \
        catch(...)                                                                               \
        {                                                                                        \
            return exception_to_rocblas_status();                                                \
        }                                                                                        \
    }                                                                                            \
    } // extern "C"

/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "check_numerics_vector.hpp"
#include "fetch_template.hpp"
#include "handle.hpp"
#include "logging.hpp"
#include "reduction.hpp"
#include "rocblas.h"
#include "rocblas_reduction.hpp"
#include "utility.hpp"
#include <type_traits>
#include <utility>

//!
//! @brief Struct-operator a default_value of rocblas_index_value_t<T>
//!
template <typename T>
struct rocblas_default_value<rocblas_index_value_t<T>>
{
    __forceinline__ __host__ __device__ constexpr auto operator()() const
    {
        rocblas_index_value_t<T> x;
        x.index = 0;
        return x;
    }
};

//!
//! @brief Struct-operator to fetch absolute value
//!
template <typename To>
struct rocblas_fetch_amax_amin
{
    template <typename Ti>
    __forceinline__ __host__ __device__ rocblas_index_value_t<To>
                                        operator()(Ti x, rocblas_int index) const
    {
        return {index, fetch_asum(x)};
    }
};

// Replaces x with y if y.value > x.value or y.value == x.value and y.index < x.index
struct rocblas_reduce_amax
{
    template <typename To>
    __forceinline__ __host__ __device__ void
        operator()(rocblas_index_value_t<To>& __restrict__ x,
                   const rocblas_index_value_t<To>& __restrict__ y) const
    {
        // If y.index == 0 then y.value is invalid and should not be compared
        if(y.index != 0)
        {
            if(x.index == 0 || y.value > x.value)
                x = y; // if larger or smaller, update max/min and index
            else if(y.index < x.index && x.value == y.value)
                x.index = y.index; // if equal, choose smaller index
        }
    }
};

// Replaces x with y if y.value < x.value or y.value == x.value and y.index < x.index
struct rocblas_reduce_amin
{
    template <typename To>
    __forceinline__ __host__ __device__ void
        operator()(rocblas_index_value_t<To>& __restrict__ x,
                   const rocblas_index_value_t<To>& __restrict__ y) const
    {
        // If y.index == 0 then y.value is invalid and should not be compared
        if(y.index != 0)
        {
            if(x.index == 0 || y.value < x.value)
                x = y; // if larger or smaller, update max/min and index
            else if(y.index < x.index && x.value == y.value)
                x.index = y.index; // if equal, choose smaller index
        }
    }
};

template <typename API_INT, typename T, typename Tr>
rocblas_status rocblas_iamax_iamin_arg_check(rocblas_handle handle,
                                             API_INT        n,
                                             T              x,
                                             API_INT        incx,
                                             rocblas_stride stridex,
                                             API_INT        batch_count,
                                             Tr*            result)
{
    if(!result)
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
                    hipMemsetAsync(result, 0, batch_count * sizeof(Tr), handle->get_stream()));
        }
        else
        {
            if(batch_count > 0)
                memset(result, 0, batch_count * sizeof(Tr));
        }
        return rocblas_status_success;
    }

    if(!x)
    {
        return rocblas_status_invalid_pointer;
    }

    return rocblas_status_continue;
}

/**
 * @brief internal iamax template. Can be used with regular iamax or iamax_strided_batched.
 *        Used by rocSOLVER, includes offset params for arrays.
 */
template <typename T, typename S>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_iamax_template(rocblas_handle            handle,
                                    rocblas_int               n,
                                    const T*                  x,
                                    rocblas_stride            shiftx,
                                    rocblas_int               incx,
                                    rocblas_stride            stridex,
                                    rocblas_int               batch_count,
                                    rocblas_int*              result,
                                    rocblas_index_value_t<S>* workspace);

/**
 * @brief internal iamax_batched template.
 *        Used by rocSOLVER, includes offset params for arrays.
 */
template <typename T, typename S>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_iamax_batched_template(rocblas_handle            handle,
                                            rocblas_int               n,
                                            const T* const*           x,
                                            rocblas_stride            shiftx,
                                            rocblas_int               incx,
                                            rocblas_stride            stridex,
                                            rocblas_int               batch_count,
                                            rocblas_int*              result,
                                            rocblas_index_value_t<S>* workspace);

/**
 * @brief internal iamin template. Can be used with regular iamin or iamin_strided_batched.
 *        Used by rocSOLVER, includes offset params for arrays.
 */
template <typename T, typename S>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_iamin_template(rocblas_handle            handle,
                                    rocblas_int               n,
                                    const T*                  x,
                                    rocblas_stride            shiftx,
                                    rocblas_int               incx,
                                    rocblas_stride            stridex,
                                    rocblas_int               batch_count,
                                    rocblas_int*              result,
                                    rocblas_index_value_t<S>* workspace);

/**
 * @brief internal iamin_batched template.
 *        Used by rocSOLVER, includes offset params for arrays.
 */
template <typename T, typename S>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_iamin_batched_template(rocblas_handle            handle,
                                            rocblas_int               n,
                                            const T* const*           x,
                                            rocblas_stride            shiftx,
                                            rocblas_int               incx,
                                            rocblas_stride            stridex,
                                            rocblas_int               batch_count,
                                            rocblas_int*              result,
                                            rocblas_index_value_t<S>* workspace);

// 64bit APIs
template <typename T, typename S>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_iamax_template_64(rocblas_handle               handle,
                                       int64_t                      n,
                                       const T*                     x,
                                       rocblas_stride               shiftx,
                                       int64_t                      incx,
                                       rocblas_stride               stridex,
                                       int64_t                      batch_count,
                                       int64_t*                     result,
                                       rocblas_index_64_value_t<S>* workspace);

template <typename T, typename S>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_iamax_batched_template_64(rocblas_handle               handle,
                                               int64_t                      n,
                                               const T* const*              x,
                                               rocblas_stride               shiftx,
                                               int64_t                      incx,
                                               rocblas_stride               stridex,
                                               int64_t                      batch_count,
                                               int64_t*                     result,
                                               rocblas_index_64_value_t<S>* workspace);

template <typename T, typename S>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_iamin_template_64(rocblas_handle               handle,
                                       int64_t                      n,
                                       const T*                     x,
                                       rocblas_stride               shiftx,
                                       int64_t                      incx,
                                       rocblas_stride               stridex,
                                       int64_t                      batch_count,
                                       int64_t*                     result,
                                       rocblas_index_64_value_t<S>* workspace);

template <typename T, typename S>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_iamin_batched_template_64(rocblas_handle               handle,
                                               int64_t                      n,
                                               const T* const*              x,
                                               rocblas_stride               shiftx,
                                               int64_t                      incx,
                                               rocblas_stride               stridex,
                                               int64_t                      batch_count,
                                               int64_t*                     result,
                                               rocblas_index_64_value_t<S>* workspace);

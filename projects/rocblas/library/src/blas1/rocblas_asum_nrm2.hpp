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

#include "check_numerics_vector.hpp"
#include "fetch_template.hpp"
#include "int64_helpers.hpp"
#include "reduction.hpp"
#include "rocblas_dot.hpp"
#include "rocblas_reduction.hpp"

template <typename API_INT,
          int NB,
          typename FETCH,
          typename FINALIZE,
          typename TPtrX,
          typename To,
          typename Tr>
rocblas_status ROCBLAS_API(rocblas_internal_asum_nrm2_launcher)(rocblas_handle handle,
                                                                API_INT        n,
                                                                TPtrX          x,
                                                                rocblas_stride shiftx,
                                                                API_INT        incx,
                                                                rocblas_stride stridex,
                                                                API_INT        batch_count,
                                                                To*            workspace,
                                                                Tr*            result);

template <class To>
struct rocblas_fetch_asum
{
    template <typename Ti>
    __forceinline__ __device__ To operator()(Ti x) const
    {
        return {fetch_asum(x)};
    }
};

template <class To>
struct rocblas_fetch_nrm2
{
    template <class Ti>
    __forceinline__ __device__ To operator()(Ti x) const
    {
        return {fetch_abs2(x)};
    }
};

struct rocblas_finalize_nrm2
{
    template <class To>
    __forceinline__ __host__ __device__ To operator()(To x) const
    {
        return sqrt(x);
    }
};

template <typename API_INT, typename T, typename Tr>
rocblas_status rocblas_asum_nrm2_arg_check(rocblas_handle handle,
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
            {
                RETURN_IF_HIP_ERROR(
                    hipMemsetAsync(result, 0, batch_count * sizeof(Tr), handle->get_stream()));
            }
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

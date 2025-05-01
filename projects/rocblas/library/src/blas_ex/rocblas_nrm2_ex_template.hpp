/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../blas1/rocblas_asum_nrm2.hpp"
#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "int64_helpers.hpp"
#include "logging.hpp"

template <typename API_INT,
          int  NB,
          bool ISBATCHED,
          typename Tx,
          typename Tr  = Tx,
          typename Tex = Tr>
rocblas_status rocblas_nrm2_ex_typecasting(rocblas_handle handle,
                                           API_INT        n,
                                           const void*    x,
                                           rocblas_stride shiftx,
                                           API_INT        incx,
                                           rocblas_stride stridex,
                                           API_INT        batch_count,
                                           void*          results,
                                           void*          workspace)
{
    auto           check_numerics = handle->check_numerics;
    rocblas_status status         = rocblas_status_success;
    if(ISBATCHED)
    {
        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status nrm2_ex_check_numerics_status
                = rocblas_internal_check_numerics_vector_template(
                    ROCBLAS_API_STR(rocblas_nrm2_batched_ex),
                    handle,
                    n,
                    (const Tx* const*)x,
                    shiftx,
                    incx,
                    stridex,
                    batch_count,
                    check_numerics,
                    is_input);
            if(nrm2_ex_check_numerics_status != rocblas_status_success)
                return nrm2_ex_check_numerics_status;
        }

        status = ROCBLAS_API(rocblas_internal_asum_nrm2_launcher)<API_INT,
                                                                  NB,
                                                                  rocblas_fetch_nrm2<Tex>,
                                                                  rocblas_finalize_nrm2>(
            handle,
            n,
            (const Tx* const*)x,
            shiftx,
            incx,
            stridex,
            batch_count,
            (Tex*)workspace,
            (Tr*)results);

        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status nrm2_ex_check_numerics_status
                = rocblas_internal_check_numerics_vector_template(
                    ROCBLAS_API_STR(rocblas_nrm2_batched_ex),
                    handle,
                    n,
                    (const Tx* const*)x,
                    shiftx,
                    incx,
                    stridex,
                    batch_count,
                    check_numerics,
                    is_input);
            if(nrm2_ex_check_numerics_status != rocblas_status_success)
                return nrm2_ex_check_numerics_status;
        }
    }
    else
    {
        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status nrm2_ex_check_numerics_status
                = rocblas_internal_check_numerics_vector_template(
                    stridex ? ROCBLAS_API_STR(rocblas_nrm2_strided_batched_ex)
                            : ROCBLAS_API_STR(rocblas_nrm2_ex),
                    handle,
                    n,
                    (const Tx*)x,
                    shiftx,
                    incx,
                    stridex,
                    batch_count,
                    check_numerics,
                    is_input);
            if(nrm2_ex_check_numerics_status != rocblas_status_success)
                return nrm2_ex_check_numerics_status;
        }
        status = ROCBLAS_API(rocblas_internal_asum_nrm2_launcher)<API_INT,
                                                                  NB,
                                                                  rocblas_fetch_nrm2<Tex>,
                                                                  rocblas_finalize_nrm2>(
            handle,
            n,
            (const Tx*)x,
            shiftx,
            incx,
            stridex,
            batch_count,
            (Tex*)workspace,
            (Tr*)results);

        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status nrm2_ex_check_numerics_status
                = rocblas_internal_check_numerics_vector_template(
                    stridex ? ROCBLAS_API_STR(rocblas_nrm2_strided_batched_ex)
                            : ROCBLAS_API_STR(rocblas_nrm2_ex),
                    handle,
                    n,
                    (const Tx*)x,
                    shiftx,
                    incx,
                    stridex,
                    batch_count,
                    check_numerics,
                    is_input);
            if(nrm2_ex_check_numerics_status != rocblas_status_success)
                return nrm2_ex_check_numerics_status;
        }
    }
    return status;
}

template <typename API_INT, rocblas_int NB, bool ISBATCHED>
rocblas_status rocblas_nrm2_ex_template(rocblas_handle   handle,
                                        API_INT          n,
                                        const void*      x,
                                        rocblas_datatype x_type,
                                        rocblas_stride   shiftx,
                                        API_INT          incx,
                                        rocblas_stride   stridex,
                                        API_INT          batch_count,
                                        void*            results,
                                        rocblas_datatype result_type,
                                        rocblas_datatype execution_type,
                                        void*            workspace)
{
#define rocblas_nrm2_ex_typecasting_PARAM \
    handle, n, x, shiftx, incx, stridex, batch_count, results, workspace

    if(x_type == rocblas_datatype_f16_r && result_type == rocblas_datatype_f16_r
       && execution_type == rocblas_datatype_f32_r)
    {
        return rocblas_nrm2_ex_typecasting<API_INT,
                                           NB,
                                           ISBATCHED,
                                           rocblas_half,
                                           rocblas_half,
                                           float>(rocblas_nrm2_ex_typecasting_PARAM);
    }
    else if(x_type == rocblas_datatype_bf16_r && result_type == rocblas_datatype_bf16_r
            && execution_type == rocblas_datatype_f32_r)
    {
        return rocblas_nrm2_ex_typecasting<API_INT,
                                           NB,
                                           ISBATCHED,
                                           rocblas_bfloat16,
                                           rocblas_bfloat16,
                                           float>(rocblas_nrm2_ex_typecasting_PARAM);
    }
    else if(x_type == rocblas_datatype_f32_r && result_type == rocblas_datatype_f32_r
            && execution_type == rocblas_datatype_f32_r)
    {
        return rocblas_nrm2_ex_typecasting<API_INT, NB, ISBATCHED, float>(
            rocblas_nrm2_ex_typecasting_PARAM);
    }
    else if(x_type == rocblas_datatype_f64_r && result_type == rocblas_datatype_f64_r
            && execution_type == rocblas_datatype_f64_r)
    {
        return rocblas_nrm2_ex_typecasting<API_INT, NB, ISBATCHED, double>(
            rocblas_nrm2_ex_typecasting_PARAM);
    }
    else if(x_type == rocblas_datatype_f32_c && result_type == rocblas_datatype_f32_r
            && execution_type == rocblas_datatype_f32_r)
    {
        return rocblas_nrm2_ex_typecasting<API_INT, NB, ISBATCHED, rocblas_float_complex, float>(
            rocblas_nrm2_ex_typecasting_PARAM);
    }
    else if(x_type == rocblas_datatype_f64_c && result_type == rocblas_datatype_f64_r
            && execution_type == rocblas_datatype_f64_r)
    {
        return rocblas_nrm2_ex_typecasting<API_INT, NB, ISBATCHED, rocblas_double_complex, double>(
            rocblas_nrm2_ex_typecasting_PARAM);
    }

    return rocblas_status_not_implemented;
}

// clang-format off

#ifdef INSTANTIATE_NRM2_EX_TEMPLATE
#error INSTANTIATE_NRM2_EX_TEMPLATE  already defined
#endif

#define INSTANTIATE_NRM2_EX_TEMPLATE(TI_, NB, ISBATCHED)                       \
    template rocblas_status rocblas_nrm2_ex_template<TI_, NB, ISBATCHED>(      \
        rocblas_handle handle,                                                 \
        TI_              n,                                                    \
        const void*      x,                                                    \
        rocblas_datatype x_type,                                               \
        rocblas_stride   shiftx,                                               \
        TI_              incx,                                                 \
        rocblas_stride   stridex,                                              \
        TI_              batch_count,                                          \
        void*            results,                                              \
        rocblas_datatype result_type,                                          \
        rocblas_datatype execution_type,                                       \
        void*            workspace);
// clang-format on

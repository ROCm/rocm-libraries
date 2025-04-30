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

#pragma once

#ifdef BUILD_WITH_TENSILE
#include "../blas3/Tensile/gemm_tensile.hpp"
#endif

#include "../blas3/rocblas_gemm.hpp"
#include "handle.hpp"
#include "logging.hpp"

template <bool BATCHED, typename Ti, typename To = Ti, typename Tc = To>
rocblas_status gemm_ex_get_solutions_typecasting(rocblas_handle                      handle,
                                                 rocblas_operation                   trans_a,
                                                 rocblas_operation                   trans_b,
                                                 rocblas_int                         m,
                                                 rocblas_int                         n,
                                                 rocblas_int                         k,
                                                 const void*                         alpha,
                                                 const void*                         a,
                                                 rocblas_stride                      offsetAin,
                                                 rocblas_int                         lda,
                                                 rocblas_stride                      stride_a,
                                                 const void*                         b,
                                                 rocblas_stride                      offsetBin,
                                                 rocblas_int                         ldb,
                                                 rocblas_stride                      stride_b,
                                                 const void*                         beta,
                                                 const void*                         c,
                                                 rocblas_stride                      offsetCin,
                                                 rocblas_int                         ldc,
                                                 rocblas_stride                      stride_c,
                                                 void*                               d,
                                                 rocblas_stride                      offsetDin,
                                                 rocblas_int                         ldd,
                                                 rocblas_stride                      stride_d,
                                                 rocblas_int                         batch_count,
                                                 rocblas_gemm_flags                  flags,
                                                 rocblas_tensile_get_solution_option option,
                                                 rocblas_int*                        list_array,
                                                 rocblas_int*                        list_size)
{
    if(BATCHED)
    {
        RocblasContractionProblem<Ti, To, Tc> problem{handle,
                                                      trans_a,
                                                      trans_b,
                                                      m,
                                                      n,
                                                      k,
                                                      (const Tc*)alpha,
                                                      nullptr,
                                                      (const Ti* const*)a,
                                                      lda,
                                                      stride_a,
                                                      offsetAin,
                                                      nullptr,
                                                      (const Ti* const*)b,
                                                      ldb,
                                                      stride_b,
                                                      offsetBin,
                                                      (const Tc*)beta,
                                                      nullptr,
                                                      (const To* const*)c,
                                                      ldc,
                                                      stride_c,
                                                      offsetCin,
                                                      nullptr,
                                                      (To* const*)d,
                                                      ldd,
                                                      stride_d,
                                                      offsetDin,
                                                      batch_count,
                                                      false,
                                                      flags};
        return getAllSolutions(problem, option, list_array, list_size);
    }
    else
    {
        RocblasContractionProblem<Ti, To, Tc> problem{handle,
                                                      trans_a,
                                                      trans_b,
                                                      m,
                                                      n,
                                                      k,
                                                      (const Tc*)alpha,
                                                      (const Ti*)a,
                                                      nullptr,
                                                      lda,
                                                      stride_a,
                                                      offsetAin,
                                                      (const Ti*)b,
                                                      nullptr,
                                                      ldb,
                                                      stride_b,
                                                      offsetBin,
                                                      (const Tc*)beta,
                                                      (const To*)c,
                                                      nullptr,
                                                      ldc,
                                                      stride_c,
                                                      offsetCin,
                                                      (To*)d,
                                                      nullptr,
                                                      ldd,
                                                      stride_d,
                                                      offsetDin,
                                                      batch_count,
                                                      true,
                                                      flags};
        return getAllSolutions(problem, option, list_array, list_size);
    }
}

template <bool BATCHED>
rocblas_status rocblas_gemm_ex_get_solutions_template(rocblas_handle    handle,
                                                      rocblas_operation trans_a,
                                                      rocblas_operation trans_b,
                                                      rocblas_int       m,
                                                      rocblas_int       n,
                                                      rocblas_int       k,
                                                      const void*       alpha,
                                                      const void*       a,
                                                      rocblas_datatype  a_type,
                                                      rocblas_stride    offsetAin,
                                                      rocblas_int       lda,
                                                      rocblas_stride    stride_a,
                                                      const void*       b,
                                                      rocblas_datatype  b_type,
                                                      rocblas_stride    offsetBin,
                                                      rocblas_int       ldb,
                                                      rocblas_stride    stride_b,
                                                      const void*       beta,
                                                      const void*       c,
                                                      rocblas_datatype  c_type,
                                                      rocblas_stride    offsetCin,
                                                      rocblas_int       ldc,
                                                      rocblas_stride    stride_c,
                                                      void*             d,
                                                      rocblas_datatype  d_type,
                                                      rocblas_stride    offsetDin,
                                                      rocblas_int       ldd,
                                                      rocblas_stride    stride_d,
                                                      rocblas_int       batch_count,
                                                      rocblas_datatype  compute_type,
                                                      uint32_t          flags,
                                                      rocblas_tensile_get_solution_option option,
                                                      rocblas_int* list_array,
                                                      rocblas_int* list_size)
{
    // Note: k==0 is not an early exit, since C still needs to be multiplied by beta
    if(!m || !n || !batch_count)
        return rocblas_status_success;

    if(BATCHED)
    {
        stride_a = rocblas_stride(lda) * (trans_a == rocblas_operation_none ? k : m);
        stride_b = rocblas_stride(ldb) * (trans_b == rocblas_operation_none ? n : k);
        stride_c = rocblas_stride(ldc) * n;
        stride_d = rocblas_stride(ldd) * n;
    }

    rocblas_status rb_status = rocblas_status_not_implemented;

#define EX_TYPECASTING_PARM                                                                    \
    handle, trans_a, trans_b, m, n, k, alpha, a, offsetAin, lda, stride_a, b, offsetBin, ldb,  \
        stride_b, beta, c, offsetCin, ldc, stride_c, d, offsetDin, ldd, stride_d, batch_count, \
        rocblas_gemm_flags(flags), option, list_array, list_size

    if(a_type == rocblas_datatype_f64_r && b_type == rocblas_datatype_f64_r
       && c_type == rocblas_datatype_f64_r && d_type == rocblas_datatype_f64_r
       && compute_type == rocblas_datatype_f64_r)
    {
        rb_status = gemm_ex_get_solutions_typecasting<BATCHED, double>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f32_r && b_type == rocblas_datatype_f32_r
            && c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r
            && compute_type == rocblas_datatype_f32_r)
    {
        rb_status = gemm_ex_get_solutions_typecasting<BATCHED, float>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r)
    {
        if(c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r)
        {
            if(compute_type == rocblas_datatype_f16_r)
            {
                rb_status
                    = gemm_ex_get_solutions_typecasting<BATCHED, rocblas_half>(EX_TYPECASTING_PARM);
            }
            else if(compute_type == rocblas_datatype_f32_r)
            {
                rb_status
                    = gemm_ex_get_solutions_typecasting<BATCHED, rocblas_half, rocblas_half, float>(
                        EX_TYPECASTING_PARM);
            }
        }
        else if(c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r
                && compute_type == rocblas_datatype_f32_r)
        {
            rb_status = gemm_ex_get_solutions_typecasting<BATCHED, rocblas_half, float, float>(
                EX_TYPECASTING_PARM);
        }
    }
    else if(a_type == rocblas_datatype_bf16_r && b_type == rocblas_datatype_bf16_r
            && compute_type == rocblas_datatype_f32_r)
    {
        if(c_type == rocblas_datatype_bf16_r && d_type == rocblas_datatype_bf16_r)
        {
            rb_status = gemm_ex_get_solutions_typecasting<BATCHED,
                                                          rocblas_bfloat16,
                                                          rocblas_bfloat16,
                                                          float>(EX_TYPECASTING_PARM);
        }
        else if(c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r)
        {
            rb_status = gemm_ex_get_solutions_typecasting<BATCHED, rocblas_bfloat16, float, float>(
                EX_TYPECASTING_PARM);
        }
    }
    else if(a_type == rocblas_datatype_i8_r && b_type == rocblas_datatype_i8_r
            && c_type == rocblas_datatype_i32_r && d_type == rocblas_datatype_i32_r
            && compute_type == rocblas_datatype_i32_r)
    {
        rb_status
            = gemm_ex_get_solutions_typecasting<BATCHED, int8_t, int32_t>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f32_c && b_type == rocblas_datatype_f32_c
            && c_type == rocblas_datatype_f32_c && d_type == rocblas_datatype_f32_c
            && compute_type == rocblas_datatype_f32_c)
    {
        rb_status = gemm_ex_get_solutions_typecasting<BATCHED,
                                                      rocblas_float_complex,
                                                      rocblas_float_complex,
                                                      rocblas_float_complex>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f64_c && b_type == rocblas_datatype_f64_c
            && c_type == rocblas_datatype_f64_c && d_type == rocblas_datatype_f64_c
            && compute_type == rocblas_datatype_f64_c)
    {
        rb_status = gemm_ex_get_solutions_typecasting<BATCHED,
                                                      rocblas_double_complex,
                                                      rocblas_double_complex,
                                                      rocblas_double_complex>(EX_TYPECASTING_PARM);
    }
    else
    {
        rb_status = rocblas_status_not_implemented;
    }
    return rb_status;
}

#undef EX_TYPECASTING_PARM

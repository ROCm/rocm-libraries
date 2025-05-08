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

#include "handle.hpp"
#include "int64_helpers.hpp"

#include "rocblas_gemm_ex_64.hpp"

#include "blas3/rocblas_gemm_source.hpp"
#include "blas_ex/rocblas_gemm_ex.hpp" // int32 API called

template <bool BATCHED, typename Ti, typename To, typename TScal>
rocblas_status rocblas_internal_gemm_ex_typecasting_64(rocblas_handle     handle,
                                                       rocblas_operation  trans_a,
                                                       rocblas_operation  trans_b,
                                                       int64_t            m_64,
                                                       int64_t            n_64,
                                                       int64_t            k_64,
                                                       const void*        alpha,
                                                       const void*        a,
                                                       rocblas_stride     offsetAin,
                                                       int64_t            lda_64,
                                                       rocblas_stride     stride_a,
                                                       const void*        b,
                                                       rocblas_stride     offsetBin,
                                                       int64_t            ldb_64,
                                                       rocblas_stride     stride_b,
                                                       const void*        beta,
                                                       const void*        c,
                                                       rocblas_stride     offsetCin,
                                                       int64_t            ldc_64,
                                                       rocblas_stride     stride_c,
                                                       void*              d,
                                                       rocblas_stride     offsetDin,
                                                       int64_t            ldd_64,
                                                       rocblas_stride     stride_d,
                                                       int64_t            batch_count_64,
                                                       rocblas_gemm_algo  algo,
                                                       int32_t            solution_index,
                                                       rocblas_gemm_flags flags)
{
    using Ta = rocblas_const_batched_t<Ti, BATCHED>;
    using Tb = Ta;
    using Tc = rocblas_const_batched_t<To, BATCHED>;
    using Td = rocblas_batched_t<To, BATCHED>;

    TScal alpha_h, beta_h;
    RETURN_IF_ROCBLAS_ERROR(
        rocblas_copy_alpha_beta_to_host_if_on_device(handle, alpha, beta, alpha_h, beta_h, k_64));
    if(k_64 == 0 && !alpha)
    {
        // special case as rocblas_gemm_source_solution_64 needs values not pointers
        alpha_h = 0;
        alpha   = &alpha_h;
    }

    auto           check_numerics = handle->check_numerics;
    rocblas_status status         = rocblas_status_success;

    // check alignment of pointers before casting
    if(!isAligned(a, sizeof(Ta)) || !isAligned(b, sizeof(Tb)) || !isAligned(c, sizeof(Tc))
       || !isAligned(d, sizeof(Td)))
        return rocblas_status_invalid_size;

    auto rocblas_gemm_ex_name
        = BATCHED ? "rocblas_gemm_batched_ex_64"
                  : (stride_a ? "rocblas_gemm_strided_batched_ex_64" : "rocblas_gemm_ex_64");

    if(check_numerics && !std::is_same_v<Ti, signed char>)
    {
        bool           is_input = true;
        rocblas_status gemm_ex_check_numerics_status
            = rocblas_gemm_check_numerics(rocblas_gemm_ex_name,
                                          handle,
                                          trans_a,
                                          trans_b,
                                          m_64,
                                          n_64,
                                          k_64,
                                          (Ta)a,
                                          offsetAin,
                                          lda_64,
                                          stride_a,
                                          (Tb)b,
                                          offsetBin,
                                          ldb_64,
                                          stride_b,
                                          (Tc)c,
                                          offsetCin,
                                          ldc_64,
                                          stride_c,
                                          batch_count_64,
                                          check_numerics,
                                          is_input);
        if(gemm_ex_check_numerics_status != rocblas_status_success)
            return gemm_ex_check_numerics_status;
    }

    constexpr int64_t limit = c_i32_max * 16; // source kernels must have m and n blocks >= 16
    bool              source_dims_supported = (m_64 <= limit && n_64 <= limit) || k_64 == 0;
    if(!source_dims_supported)
        return rocblas_status_invalid_size;

    for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
    {
        int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

        rocblas_stride offsetA = offsetAin;
        rocblas_stride offsetB = offsetBin;
        rocblas_stride offsetC = offsetCin;
        rocblas_stride offsetD = offsetDin;

        auto A_ptr = adjust_ptr_batch((Ta)a, b_base, stride_a);
        auto B_ptr = adjust_ptr_batch((Tb)b, b_base, stride_b);
        auto C_ptr = adjust_ptr_batch((Tc)c, b_base, stride_c);
        auto D_ptr = adjust_ptr_batch((Td)d, b_base, stride_d);

        for(int64_t n_base = 0; n_base < n_64; n_base += c_i64_grid_YZ_chunk)
        {
            int32_t n = int32_t(std::min(n_64 - n_base, c_i64_grid_YZ_chunk));

            for(int64_t m_base = 0; m_base < m_64; m_base += c_i64_grid_X_chunk)
            {
                int32_t m = int32_t(std::min(m_64 - m_base, c_i64_grid_X_chunk));
                status    = rocblas_gemm_source_solution_64<BATCHED>(
                    handle,
                    trans_a,
                    trans_b,
                    m,
                    n,
                    k_64,
                    *(TScal*)alpha,
                    A_ptr,
                    lda_64,
                    stride_a,
                    offsetA + (trans_a == rocblas_operation_none ? m_base : m_base * lda_64),
                    B_ptr,
                    ldb_64,
                    stride_b,
                    offsetB + (trans_b == rocblas_operation_none ? n_base * ldb_64 : n_base),
                    *(TScal*)beta,
                    C_ptr,
                    ldc_64,
                    stride_c,
                    offsetC + m_base + n_base * ldc_64,
                    D_ptr,
                    ldd_64,
                    stride_d,
                    offsetD + m_base + n_base * ldd_64,
                    batch_count);
            }
        }

        if(status != rocblas_status_success)
            return status;
    }

    if(check_numerics && !std::is_same_v<Ti, signed char>)
    {
        bool           is_input = false;
        rocblas_status gemm_ex_check_numerics_status
            = rocblas_gemm_check_numerics(rocblas_gemm_ex_name,
                                          handle,
                                          trans_a,
                                          trans_b,
                                          m_64,
                                          n_64,
                                          k_64,
                                          (Ta)a,
                                          offsetAin,
                                          lda_64,
                                          stride_a,
                                          (Tb)b,
                                          offsetBin,
                                          ldb_64,
                                          stride_b,
                                          (Td)d,
                                          offsetDin,
                                          ldd_64,
                                          stride_d,
                                          batch_count_64,
                                          check_numerics,
                                          is_input);
        if(gemm_ex_check_numerics_status != rocblas_status_success)
            return gemm_ex_check_numerics_status;
    }

    return status;
}

template <bool BATCHED>
rocblas_status rocblas_gemm_ex_template_64(rocblas_handle    handle,
                                           rocblas_operation trans_a,
                                           rocblas_operation trans_b,
                                           int64_t           m_64,
                                           int64_t           n_64,
                                           int64_t           k_64,
                                           const void*       alpha,
                                           const void*       A,
                                           rocblas_datatype  a_type,
                                           rocblas_stride    offsetAin,
                                           int64_t           lda_64,
                                           rocblas_stride    stride_a,
                                           const void*       B,
                                           rocblas_datatype  b_type,
                                           rocblas_stride    offsetBin,
                                           int64_t           ldb_64,
                                           rocblas_stride    stride_b,
                                           const void*       beta,
                                           const void*       C,
                                           rocblas_datatype  c_type,
                                           rocblas_stride    offsetCin,
                                           int64_t           ldc_64,
                                           rocblas_stride    stride_c,
                                           void*             D,
                                           rocblas_datatype  d_type,
                                           rocblas_stride    offsetDin,
                                           int64_t           ldd_64,
                                           rocblas_stride    stride_d,
                                           int64_t           batch_count_64,
                                           rocblas_datatype  compute_type,
                                           rocblas_gemm_algo algo,
                                           int32_t           solution_index,
                                           uint32_t          flags)
{
    bool dims_32bit = m_64 <= c_ILP64_i32_max && n_64 <= c_ILP64_i32_max && k_64 <= c_ILP64_i32_max
                      && lda_64 <= c_ILP64_i32_max && ldb_64 <= c_ILP64_i32_max
                      && ldc_64 <= c_ILP64_i32_max && ldd_64 <= c_ILP64_i32_max;

    // if all dims are 32-bit, can use regular gemm_ex
    if(dims_32bit)
    {
        for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
        {
            int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

            rocblas_stride offsetA = offsetAin;
            rocblas_stride offsetB = offsetBin;
            rocblas_stride offsetC = offsetCin;
            rocblas_stride offsetD = offsetDin;

            auto A_ptr = A;
            auto B_ptr = B;
            auto C_ptr = C;
            auto D_ptr = D;

            if constexpr(BATCHED)
            {
                // avoiding typecasting
                A_ptr = (void**)A_ptr + b_base;
                B_ptr = (void**)B_ptr + b_base;
                C_ptr = (void**)C_ptr + b_base;
                D_ptr = (void**)D_ptr + b_base;
            }
            else if(b_base > 0)
            {
                offsetA += b_base * stride_a;
                offsetB += b_base * stride_b;
                offsetC += b_base * stride_c;
                offsetD += b_base * stride_d;
            }

            rocblas_status status = rocblas_gemm_ex_template<BATCHED>(handle,
                                                                      trans_a,
                                                                      trans_b,
                                                                      m_64,
                                                                      n_64,
                                                                      k_64,
                                                                      alpha,
                                                                      A_ptr,
                                                                      a_type,
                                                                      offsetA,
                                                                      lda_64,
                                                                      stride_a,
                                                                      B_ptr,
                                                                      b_type,
                                                                      offsetB,
                                                                      ldb_64,
                                                                      stride_b,
                                                                      beta,
                                                                      C_ptr,
                                                                      c_type,
                                                                      offsetC,
                                                                      ldc_64,
                                                                      stride_c,
                                                                      D_ptr,
                                                                      d_type,
                                                                      offsetD,
                                                                      ldd_64,
                                                                      stride_d,
                                                                      batch_count,
                                                                      compute_type,
                                                                      algo,
                                                                      solution_index,
                                                                      flags);

            if(status != rocblas_status_success)
                return status;
        }
    }
    else
    {
        if(!m_64 || !n_64 || !batch_count_64)
            return rocblas_status_success;

        if(BATCHED)
        {
            stride_a = lda_64 * (trans_a == rocblas_operation_none ? k_64 : m_64);
            stride_b = ldb_64 * (trans_b == rocblas_operation_none ? n_64 : k_64);
            stride_c = ldc_64 * n_64;
            stride_d = ldd_64 * n_64;
        }

        rocblas_status rb_status = rocblas_status_not_implemented;

#define EX_TYPECASTING_PARM                                                                      \
    handle, trans_a, trans_b, m_64, n_64, k_64, alpha, A, offsetAin, lda_64, stride_a, B,        \
        offsetBin, ldb_64, stride_b, beta, C, offsetCin, ldc_64, stride_c, D, offsetDin, ldd_64, \
        stride_d, batch_count_64, algo, solution_index, rocblas_gemm_flags(flags)

        if(a_type == rocblas_datatype_f64_r && b_type == rocblas_datatype_f64_r
           && c_type == rocblas_datatype_f64_r && d_type == rocblas_datatype_f64_r
           && compute_type == rocblas_datatype_f64_r)
        {
            rb_status
                = rocblas_internal_gemm_ex_typecasting_64<BATCHED, double>(EX_TYPECASTING_PARM);
        }
        else if(a_type == rocblas_datatype_f32_r && b_type == rocblas_datatype_f32_r
                && c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r
                && compute_type == rocblas_datatype_f32_r)
        {
            rb_status
                = rocblas_internal_gemm_ex_typecasting_64<BATCHED, float>(EX_TYPECASTING_PARM);
        }
        else if(a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r)
        {
            if(c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r)
            {
                if(compute_type == rocblas_datatype_f16_r)
                {
                    rb_status = rocblas_internal_gemm_ex_typecasting_64<BATCHED, rocblas_half>(
                        EX_TYPECASTING_PARM);
                }
                else if(compute_type == rocblas_datatype_f32_r)
                {
                    rb_status = rocblas_internal_gemm_ex_typecasting_64<BATCHED,
                                                                        rocblas_half,
                                                                        rocblas_half,
                                                                        float>(EX_TYPECASTING_PARM);
                }
            }
            else if(c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r
                    && compute_type == rocblas_datatype_f32_r)
            {
                rb_status
                    = rocblas_internal_gemm_ex_typecasting_64<BATCHED, rocblas_half, float, float>(
                        EX_TYPECASTING_PARM);
            }
        }
        else if(a_type == rocblas_datatype_bf16_r && b_type == rocblas_datatype_bf16_r
                && compute_type == rocblas_datatype_f32_r)
        {
            if(c_type == rocblas_datatype_bf16_r && d_type == rocblas_datatype_bf16_r)
            {
                rb_status = rocblas_internal_gemm_ex_typecasting_64<BATCHED,
                                                                    rocblas_bfloat16,
                                                                    rocblas_bfloat16,
                                                                    float>(EX_TYPECASTING_PARM);
            }
            else if(c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r)
            {
                rb_status = rocblas_internal_gemm_ex_typecasting_64<BATCHED,
                                                                    rocblas_bfloat16,
                                                                    float,
                                                                    float>(EX_TYPECASTING_PARM);
            }
        }
        else if(a_type == rocblas_datatype_i8_r && b_type == rocblas_datatype_i8_r
                && c_type == rocblas_datatype_i32_r && d_type == rocblas_datatype_i32_r
                && compute_type == rocblas_datatype_i32_r)
        {
            rb_status = rocblas_internal_gemm_ex_typecasting_64<BATCHED, int8_t, int32_t>(
                EX_TYPECASTING_PARM);
        }
        else if(a_type == rocblas_datatype_f32_c && b_type == rocblas_datatype_f32_c
                && c_type == rocblas_datatype_f32_c && d_type == rocblas_datatype_f32_c
                && compute_type == rocblas_datatype_f32_c)
        {
            rb_status = rocblas_internal_gemm_ex_typecasting_64<BATCHED,
                                                                rocblas_float_complex,
                                                                rocblas_float_complex,
                                                                rocblas_float_complex>(
                EX_TYPECASTING_PARM);
        }
        else if(a_type == rocblas_datatype_f64_c && b_type == rocblas_datatype_f64_c
                && c_type == rocblas_datatype_f64_c && d_type == rocblas_datatype_f64_c
                && compute_type == rocblas_datatype_f64_c)
        {
            rb_status = rocblas_internal_gemm_ex_typecasting_64<BATCHED,
                                                                rocblas_double_complex,
                                                                rocblas_double_complex,
                                                                rocblas_double_complex>(
                EX_TYPECASTING_PARM);
        }
        else
        {
            rb_status = rocblas_status_not_implemented;
        }

        return rb_status;
    }

    return rocblas_status_success;
}

#ifdef INSTANTIATE_GEMM_EX_64_TEMPLATE
#error INSTANTIATE_GEMM_EX_64_TEMPLATE already defined
#endif

#define INSTANTIATE_GEMM_EX_64_TEMPLATE(BATCHED_)                                                   \
    template rocblas_status rocblas_gemm_ex_template_64<BATCHED_>(rocblas_handle    handle,         \
                                                                  rocblas_operation trans_a,        \
                                                                  rocblas_operation trans_b,        \
                                                                  int64_t           m,              \
                                                                  int64_t           n,              \
                                                                  int64_t           k,              \
                                                                  const void*       alpha,          \
                                                                  const void*       a,              \
                                                                  rocblas_datatype  a_type,         \
                                                                  rocblas_stride    offsetAin,      \
                                                                  int64_t           lda,            \
                                                                  rocblas_stride    stride_a,       \
                                                                  const void*       b,              \
                                                                  rocblas_datatype  b_type,         \
                                                                  rocblas_stride    offsetBin,      \
                                                                  int64_t           ldb,            \
                                                                  rocblas_stride    stride_b,       \
                                                                  const void*       beta,           \
                                                                  const void*       c,              \
                                                                  rocblas_datatype  c_type,         \
                                                                  rocblas_stride    offsetCin,      \
                                                                  int64_t           ldc,            \
                                                                  rocblas_stride    stride_c,       \
                                                                  void*             d,              \
                                                                  rocblas_datatype  d_type,         \
                                                                  rocblas_stride    offsetDin,      \
                                                                  int64_t           ldd,            \
                                                                  rocblas_stride    stride_d,       \
                                                                  int64_t           batch_count,    \
                                                                  rocblas_datatype  compute_type,   \
                                                                  rocblas_gemm_algo algo,           \
                                                                  int32_t           solution_index, \
                                                                  uint32_t          flags);

INSTANTIATE_GEMM_EX_64_TEMPLATE(true)
INSTANTIATE_GEMM_EX_64_TEMPLATE(false)

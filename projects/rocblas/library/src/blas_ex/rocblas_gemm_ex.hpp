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

#ifdef BUILD_WITH_TENSILE
#include "../blas3/Tensile/gemm_tensile.hpp"
#endif

#include "../blas3/rocblas_gemm.hpp"
#include "handle.hpp"
#include "logging.hpp"

#define GEMM_EX_GEMV_SOLUTION_IDX -9

template <typename T>
rocblas_status rocblas_copy_alpha_beta_to_host_if_on_device(rocblas_handle   handle,
                                                            const T*&        alpha,
                                                            const T*&        beta,
                                                            rocblas_union_t& alpha_h,
                                                            rocblas_union_t& beta_h,
                                                            rocblas_int      k,
                                                            rocblas_datatype compute_type)
{
    switch(compute_type)
    {
    case rocblas_datatype_f16_r:
        return rocblas_copy_alpha_beta_to_host_if_on_device(
            handle, alpha, beta, alpha_h.h, beta_h.h, k);
    case rocblas_datatype_f32_r:
        return rocblas_copy_alpha_beta_to_host_if_on_device(
            handle, alpha, beta, alpha_h.s, beta_h.s, k);
    case rocblas_datatype_f64_r:
        return rocblas_copy_alpha_beta_to_host_if_on_device(
            handle, alpha, beta, alpha_h.d, beta_h.d, k);
    case rocblas_datatype_i32_r:
        return rocblas_copy_alpha_beta_to_host_if_on_device(
            handle, alpha, beta, alpha_h.i, beta_h.i, k);
    case rocblas_datatype_f32_c:
        return rocblas_copy_alpha_beta_to_host_if_on_device(
            handle, alpha, beta, alpha_h.c, beta_h.c, k);
    case rocblas_datatype_f64_c:
        return rocblas_copy_alpha_beta_to_host_if_on_device(
            handle, alpha, beta, alpha_h.z, beta_h.z, k);
    default:
        return rocblas_status_not_implemented;
    }
}

template <bool BATCHED, typename TScal, typename TiConstPtr, typename ToConstPtr, typename ToPtr>
constexpr bool rocblas_is_gemv_supported_types()
{
    using Tex = rocblas_type_from_ptr_t<TScal, false>;
    using Ti  = rocblas_type_from_ptr_t<TiConstPtr, BATCHED>;
    using Toc = rocblas_type_from_ptr_t<ToConstPtr, BATCHED>;
    using Tod = rocblas_type_from_ptr_t<ToPtr, BATCHED>;
    constexpr bool is_sgemv
        = std::is_same_v<
              Tex,
              float> && std::is_same_v<Tex, Ti> && std::is_same_v<Ti, Toc> && std::is_same_v<Toc, Tod>;
    constexpr bool is_dgemv
        = std::is_same_v<
              Tex,
              double> && std::is_same_v<Tex, Ti> && std::is_same_v<Ti, Toc> && std::is_same_v<Toc, Tod>;
    constexpr bool is_cgemv
        = std::is_same_v<
              Tex,
              rocblas_float_complex> && std::is_same_v<Tex, Ti> && std::is_same_v<Ti, Toc> && std::is_same_v<Toc, Tod>;
    constexpr bool is_zgemv
        = std::is_same_v<
              Tex,
              rocblas_double_complex> && std::is_same_v<Tex, Ti> && std::is_same_v<Ti, Toc> && std::is_same_v<Toc, Tod>;
    constexpr bool is_hshgemv
        = std::is_same_v<
              Tex,
              float> && std::is_same_v<Ti, rocblas_half> && std::is_same_v<Toc, rocblas_half> && std::is_same_v<Toc, Tod>;
    constexpr bool is_hssgemv
        = std::is_same_v<
              Tex,
              float> && std::is_same_v<Ti, rocblas_half> && std::is_same_v<Toc, float> && std::is_same_v<Toc, Tod>;
    constexpr bool is_tstgemv
        = std::is_same_v<
              Tex,
              float> && std::is_same_v<Ti, rocblas_bfloat16> && std::is_same_v<Toc, rocblas_bfloat16> && std::is_same_v<Toc, Tod>;
    constexpr bool is_tssgemv
        = std::is_same_v<
              Tex,
              float> && std::is_same_v<Ti, rocblas_bfloat16> && std::is_same_v<Toc, float> && std::is_same_v<Toc, Tod>;

    return is_sgemv || is_dgemv || is_cgemv || is_zgemv || is_hshgemv || is_hssgemv || is_tstgemv
           || is_tssgemv;
}

/*
 * Assuming valid types, this function will return whether or not our gemv source kernels
 * are valid for the gemm use case given the parameters.
 */
static inline bool rocblas_can_use_gemv_in_gemm(rocblas_operation trans_a,
                                                rocblas_operation trans_b,
                                                rocblas_int       m,
                                                rocblas_int       n,
                                                rocblas_int       k,
                                                void*             c,
                                                rocblas_stride    offset_c,
                                                rocblas_int       ldc,
                                                rocblas_stride    stride_c,
                                                void*             d,
                                                rocblas_stride    offset_d,
                                                rocblas_int       ldd,
                                                rocblas_stride    stride_d)
{
    // Can only use gemv in the case where m == 1 || n == 1
    if((m == 1 || n == 1) && k != 0)
    {
        // These are constraints on whether or not our gemv kernels can support this configuration. Need in-place, can't support CONJ in some cases
        bool gemv_constraints
            = (c == d && ldc == ldd && stride_c == stride_d && offset_c == offset_d)
              && !(n == 1 && trans_b == rocblas_operation_conjugate_transpose)
              && !(m == 1
                   && (trans_b == rocblas_operation_conjugate_transpose
                       || trans_a == rocblas_operation_conjugate_transpose));

        // Here, we aren't concerned about performance, just if we /can/ use gemv kernels or not
        return gemv_constraints;
    }

    return false;
}

/*
 * This function will return whether or not we /want/ to use gemv kernels for the given gemm
 * problem. This will determine if gemv kernels are valid, and use heuristics of some sort
 * to determine whether or not these kernels are desired.
 */
template <bool BATCHED, typename TScal, typename TiConstPtr, typename ToConstPtr, typename ToPtr>
bool rocblas_use_gemv_in_gemm(rocblas_handle    handle,
                              rocblas_operation trans_a,
                              rocblas_operation trans_b,
                              rocblas_int       m,
                              rocblas_int       n,
                              rocblas_int       k,
                              rocblas_int       lda,
                              rocblas_int       ldb,
                              ToConstPtr        c,
                              rocblas_stride    offset_c,
                              rocblas_int       ldc,
                              rocblas_stride    stride_c,
                              ToPtr             d,
                              rocblas_stride    offset_d,
                              rocblas_int       ldd,
                              rocblas_stride    stride_d);

template <bool BATCHED>
rocblas_status rocblas_gemm_ex_template(rocblas_handle    handle,
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
                                        rocblas_gemm_algo algo,
                                        int32_t           solution_index,
                                        uint32_t          flags);

template <typename API_INT>
inline rocblas_status rocblas_gemm_ex_arg_check(rocblas_handle    handle,
                                                rocblas_operation trans_a,
                                                rocblas_operation trans_b,
                                                API_INT           m,
                                                API_INT           n,
                                                API_INT           k,
                                                const void*       alpha,
                                                const void*       a,
                                                API_INT           ld_a,
                                                const void*       b,
                                                API_INT           ld_b,
                                                const void*       beta,
                                                const void*       c,
                                                rocblas_datatype  c_type,
                                                API_INT           ld_c,
                                                const void*       d,
                                                rocblas_datatype  d_type,
                                                API_INT           ld_d,
                                                rocblas_datatype  compute_type,
                                                API_INT           batch_count   = 1,
                                                bool              get_solutions = false)
{
    // handle must be valid
    if(!handle)
        return rocblas_status_invalid_handle;

    if(trans_a != rocblas_operation_none && trans_a != rocblas_operation_transpose
       && trans_a != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;
    if(trans_b != rocblas_operation_none && trans_b != rocblas_operation_transpose
       && trans_b != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;

    // sizes must not be negative
    if(m < 0 || n < 0 || k < 0 || batch_count < 0)
        return rocblas_status_invalid_size;

    // leading dimensions must be valid
    if(ld_c < m || ld_d < m || ld_a < (trans_a == rocblas_operation_none ? m : k)
       || ld_b < (trans_b == rocblas_operation_none ? k : n))
        return rocblas_status_invalid_size;

    // quick return
    // Note: k==0 is not a quick return, because C must still be multiplied by beta
    if(!m || !n || !batch_count)
        return rocblas_status_success;

    if(handle->is_device_memory_size_query())
        return rocblas_status_continue;

    // pointers must be valid
    if((k && !alpha) || !beta || (!d && !get_solutions))
        return rocblas_status_invalid_pointer;

    // If C is nullptr, beta must be zero
    if(!c && !get_solutions)
    {
        switch(compute_type)
        {
        case rocblas_datatype_f16_r:
            if(*(const rocblas_half*)beta)
                return rocblas_status_invalid_pointer;
            break;
        case rocblas_datatype_f32_r:
            if(*(const float*)beta)
                return rocblas_status_invalid_pointer;
            break;
        case rocblas_datatype_f64_r:
            if(*(const double*)beta)
                return rocblas_status_invalid_pointer;
            break;
        case rocblas_datatype_i32_r:
            if(*(const int32_t*)beta)
                return rocblas_status_invalid_pointer;
            break;
        case rocblas_datatype_f32_c:
            if(*(const rocblas_float_complex*)beta)
                return rocblas_status_invalid_pointer;
            break;
        case rocblas_datatype_f64_c:
            if(*(const rocblas_double_complex*)beta)
                return rocblas_status_invalid_pointer;
            break;
        default:
            break;
        }
    }

    // If k != 0 and either A or B is nullptr, alpha must be zero
    if(k && (!a || !b) && !get_solutions)
    {
        switch(compute_type)
        {
        case rocblas_datatype_f16_r:
            if(*(const rocblas_half*)alpha)
                return rocblas_status_invalid_pointer;
            break;
        case rocblas_datatype_f32_r:
            if(*(const float*)alpha)
                return rocblas_status_invalid_pointer;
            break;
        case rocblas_datatype_f64_r:
            if(*(const double*)alpha)
                return rocblas_status_invalid_pointer;
            break;
        case rocblas_datatype_i32_r:
            if(*(const int32_t*)alpha)
                return rocblas_status_invalid_pointer;
            break;
        case rocblas_datatype_f32_c:
            if(*(const rocblas_float_complex*)alpha)
                return rocblas_status_invalid_pointer;
            break;
        case rocblas_datatype_f64_c:
            if(*(const rocblas_double_complex*)alpha)
                return rocblas_status_invalid_pointer;
            break;
        default:
            break;
        }
    }

    if(c == d)
    {
        if(ld_c != ld_d)
            return rocblas_status_invalid_size;
        if(c_type != d_type)
            return rocblas_status_invalid_value;
    }

    return rocblas_status_continue;
}

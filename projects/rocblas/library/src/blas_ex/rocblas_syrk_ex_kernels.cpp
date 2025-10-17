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

#include "handle.hpp"
#include "logging.hpp"
#include "rocblas_gemm_ex.hpp"
#include "rocblas_syrk_ex.hpp"

#include "../blas3/herk_syrk_device.hpp"
#include "../blas3/rocblas_syrk_herk.hpp"
#include "../blas3/rocblas_syrk_imp.hpp"

template <typename Tex, typename Ta, typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syrk_ex_template(rocblas_handle    handle,
                                      rocblas_fill      uplo,
                                      rocblas_operation trans_A,
                                      rocblas_int       n,
                                      rocblas_int       k,
                                      const Tex*        alpha_in,
                                      const Ta*         A,
                                      rocblas_datatype  A_type,
                                      rocblas_stride    offset_A,
                                      rocblas_int       lda,
                                      rocblas_stride    stride_A,
                                      const Tex*        beta_in,
                                      T*                C,
                                      rocblas_datatype  C_type,
                                      rocblas_stride    offset_C,
                                      rocblas_int       ldc,
                                      rocblas_stride    stride_C,
                                      rocblas_datatype  compute_type,
                                      rocblas_int       batch_count)
{

    constexpr bool BATCHED    = false;
    constexpr bool HERM       = false;
    constexpr bool FORCEDGEMM = true;

    size_t size = rocblas_internal_syrk_herk_workspace<T, FORCEDGEMM>(handle, n, k, batch_count);

    //Allocate Workspace memory
    auto w_mem = handle->device_malloc(size);
    if(!w_mem)
        return rocblas_status_memory_error;

    hipStream_t rocblas_stream = handle->get_stream();

    // Note: alpha and beta always copied over to host by now
    if(*beta_in == 1 && (k == 0 || *alpha_in == 0))
        return rocblas_status_success;

    bool a_calc_invalid = !alpha_in || (*alpha_in != 0 && (!A));
    if(!C || (k && a_calc_invalid))
        return rocblas_status_invalid_pointer;

    // upgrade to complex if needed
    // TODO: Graph safety?
    const Tex alpha_val = (Tex)(*alpha_in);
    const Tex beta_val  = (Tex)(*beta_in);

    const Tex* alpha = &alpha_val;
    const Tex* beta  = &beta_val;

    rocblas_operation trans_orig
        = rocblas_operation_none == trans_A
              ? rocblas_operation_none
              : (HERM ? rocblas_operation_conjugate_transpose : rocblas_operation_transpose);
    rocblas_operation trans_opp
        = rocblas_operation_none == trans_A
              ? (HERM ? rocblas_operation_conjugate_transpose : rocblas_operation_transpose)
              : rocblas_operation_none;

    // Launch kernel to copy the data from triangular matrix to the workspace memory
    if(rocblas_fill_upper == uplo)
        RETURN_IF_ROCBLAS_ERROR((rocblas_copy_triangular_syrk_herk<true, true, HERM>(
            handle, n, C, ldc, stride_C, (T*)w_mem, batch_count)));
    else
        RETURN_IF_ROCBLAS_ERROR((rocblas_copy_triangular_syrk_herk<true, false, HERM>(
            handle, n, C, ldc, stride_C, (T*)w_mem, batch_count)));

    RETURN_IF_ROCBLAS_ERROR((rocblas_gemm_ex_template<BATCHED>(handle,
                                                               trans_orig,
                                                               trans_opp,
                                                               n,
                                                               n,
                                                               k,
                                                               alpha,
                                                               A,
                                                               A_type,
                                                               offset_A,
                                                               lda,
                                                               stride_A,
                                                               A,
                                                               A_type,
                                                               offset_A,
                                                               lda,
                                                               stride_A,
                                                               beta,
                                                               C,
                                                               C_type,
                                                               offset_C,
                                                               ldc,
                                                               stride_C,
                                                               C,
                                                               C_type,
                                                               offset_C,
                                                               ldc,
                                                               stride_C,
                                                               batch_count,
                                                               compute_type,
                                                               rocblas_gemm_algo_standard,
                                                               0,
                                                               0)));

    // Launch kernel to copy the data from workspace memory back to triangular matrix
    if(rocblas_fill_upper == uplo)
        RETURN_IF_ROCBLAS_ERROR((rocblas_copy_triangular_syrk_herk<false, true, HERM>(
            handle, n, C, ldc, stride_C, (T*)w_mem, batch_count)));
    else
        RETURN_IF_ROCBLAS_ERROR((rocblas_copy_triangular_syrk_herk<false, false, HERM>(
            handle, n, C, ldc, stride_C, (T*)w_mem, batch_count)));

    return rocblas_status_success;
}

template <bool BATCHED, typename T, typename U, typename Tex>
rocblas_status rocblas_syrk_ex_typecasting(rocblas_handle    handle,
                                           rocblas_fill      uplo,
                                           rocblas_operation trans_A,
                                           rocblas_int       n,
                                           rocblas_int       k,
                                           const void*       alpha,
                                           const void*       A,
                                           rocblas_datatype  A_type,
                                           rocblas_stride    offset_A,
                                           rocblas_int       lda,
                                           rocblas_stride    stride_A,
                                           const void*       beta,
                                           void*             C,
                                           rocblas_datatype  C_type,
                                           rocblas_stride    offset_C,
                                           rocblas_int       ldc,
                                           rocblas_stride    stride_C,
                                           rocblas_datatype  compute_type,
                                           rocblas_int       batch_count)
{
    auto check_numerics = handle->check_numerics;

    rocblas_status status = rocblas_syrk_arg_check<rocblas_int>(handle,
                                                                uplo,
                                                                trans_A,
                                                                n,
                                                                k,
                                                                (Tex*)alpha,
                                                                (const T*)A,
                                                                offset_A,
                                                                lda,
                                                                stride_A,
                                                                (Tex*)beta,
                                                                (U*)C,
                                                                offset_C,
                                                                ldc,
                                                                stride_C,
                                                                batch_count);
    if(status != rocblas_status_continue)
        return status;

    if(!BATCHED)
    {
        if(check_numerics)
        {
            auto check_numerics_string
                = stride_A ? "rocblas_syrk_strided_batched_ex" : "rocblas_syrk_ex";
            bool is_input = true;
            status        = rocblas_herk_syrk_check_numerics<false>(check_numerics_string,
                                                             handle,
                                                             uplo,
                                                             trans_A,
                                                             n,
                                                             k,
                                                             (const T*)A,
                                                             lda,
                                                             stride_A,
                                                             (U*)C,
                                                             ldc,
                                                             stride_C,
                                                             batch_count,
                                                             check_numerics,
                                                             is_input);
            if(status != rocblas_status_success)
                return status;
        }

        status = rocblas_internal_syrk_ex_template<Tex>(handle,
                                                        uplo,
                                                        trans_A,
                                                        n,
                                                        k,
                                                        (const Tex*)alpha,
                                                        (const T*)A,
                                                        A_type,
                                                        offset_A,
                                                        lda,
                                                        stride_A,
                                                        (const Tex*)beta,
                                                        (U*)C,
                                                        C_type,
                                                        offset_C,
                                                        ldc,
                                                        stride_C,
                                                        compute_type,
                                                        1);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            auto check_numerics_string
                = stride_A ? "rocblas_syrk_strided_batched_ex" : "rocblas_syrk_ex";
            bool is_input = false;
            status        = rocblas_herk_syrk_check_numerics<false>(check_numerics_string,
                                                             handle,
                                                             uplo,
                                                             trans_A,
                                                             n,
                                                             k,
                                                             (const T*)A,
                                                             lda,
                                                             stride_A,
                                                             (U*)C,
                                                             ldc,
                                                             stride_C,
                                                             batch_count,
                                                             check_numerics,
                                                             is_input);
            if(status != rocblas_status_success)
                return status;
        }
        return rocblas_status_success;
    }
    else
        return rocblas_status_not_implemented;
}

template <bool BATCHED>
rocblas_status rocblas_syrk_ex_template(rocblas_handle    handle,
                                        rocblas_fill      uplo,
                                        rocblas_operation trans_A,
                                        rocblas_int       n,
                                        rocblas_int       k,
                                        const void*       alpha,
                                        const void*       A,
                                        rocblas_datatype  A_type,
                                        rocblas_stride    offset_A,
                                        rocblas_int       lda,
                                        rocblas_stride    stride_A,
                                        const void*       beta,
                                        void*             C,
                                        rocblas_datatype  C_type,
                                        rocblas_stride    offset_C,
                                        rocblas_int       ldc,
                                        rocblas_stride    stride_C,
                                        rocblas_datatype  compute_type,
                                        rocblas_int       batch_count)
{
    if(!n || !batch_count)
        return rocblas_status_success;

    rocblas_status status = rocblas_status_not_implemented;

#define SYRK_EX_TYPECASTING_PARAM                                                            \
    handle, uplo, trans_A, n, k, alpha, A, A_type, offset_A, lda, stride_A, beta, C, C_type, \
        offset_C, ldc, stride_C, compute_type, batch_count

    if(A_type == rocblas_datatype_bf16_r && C_type == rocblas_datatype_bf16_r
       && compute_type == rocblas_datatype_f32_r)
        status = rocblas_syrk_ex_typecasting<BATCHED, rocblas_bfloat16, rocblas_bfloat16, float>(
            SYRK_EX_TYPECASTING_PARAM);
    else if(A_type == rocblas_datatype_bf16_r && C_type == rocblas_datatype_f32_r
            && compute_type == rocblas_datatype_f32_r)
        status = rocblas_syrk_ex_typecasting<BATCHED, rocblas_bfloat16, float, float>(
            SYRK_EX_TYPECASTING_PARAM);
    else if(A_type == rocblas_datatype_f16_r && C_type == rocblas_datatype_f16_r
            && compute_type == rocblas_datatype_f32_r)
        status = rocblas_syrk_ex_typecasting<BATCHED, rocblas_half, rocblas_half, float>(
            SYRK_EX_TYPECASTING_PARAM);
    else if(A_type == rocblas_datatype_f16_r && C_type == rocblas_datatype_f32_r
            && compute_type == rocblas_datatype_f32_r)
        status = rocblas_syrk_ex_typecasting<BATCHED, rocblas_half, float, float>(
            SYRK_EX_TYPECASTING_PARAM);
    // else if(A_type == rocblas_datatype_f32_r && C_type == rocblas_datatype_f32_r
    //         && compute_type == rocblas_datatype_f64_r)
    //     status = rocblas_syrk_ex_typecasting<BATCHED, float, float, double>(
    //         SYRK_EX_TYPECASTING_PARAM);
    // else if(A_type == rocblas_datatype_f32_r && C_type == rocblas_datatype_f64_r
    //         && compute_type == rocblas_datatype_f64_r)
    //     status = rocblas_syrk_ex_typecasting<BATCHED, float, double, double>(
    //         SYRK_EX_TYPECASTING_PARAM);
    else
        status = rocblas_status_not_implemented;

#undef SYRK_EX_TYPECASTING_PARAM

    return status;
}

#ifdef INSTANTIATE_SYRK_EX_TEMPLATE
#error INSTANTIATE_SYRK_EX_TEMPLATE  already defined
#endif

#define INSTANTIATE_SYRK_EX_TEMPLATE(BATCHED)                                                 \
    template rocblas_status rocblas_syrk_ex_template<BATCHED>(rocblas_handle    handle,       \
                                                              rocblas_fill      uplo,         \
                                                              rocblas_operation trans_A,      \
                                                              rocblas_int       n,            \
                                                              rocblas_int       k,            \
                                                              const void*       alpha,        \
                                                              const void*       A,            \
                                                              rocblas_datatype  A_type,       \
                                                              rocblas_stride    offset_A,     \
                                                              rocblas_int       lda,          \
                                                              rocblas_stride    stride_A,     \
                                                              const void*       beta,         \
                                                              void*             C,            \
                                                              rocblas_datatype  C_type,       \
                                                              rocblas_stride    offset_C,     \
                                                              rocblas_int       ldc,          \
                                                              rocblas_stride    stride_C,     \
                                                              rocblas_datatype  compute_type, \
                                                              rocblas_int       batch_count);

INSTANTIATE_SYRK_EX_TEMPLATE(false)
// INSTANTIATE_SYRK_EX_TEMPLATE(true)

#undef INSTANTIATE_SYRK_EX_TEMPLATE

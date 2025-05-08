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

#include "check_numerics_matrix.hpp"
#include "geam_ex_source.hpp"
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas_geam_ex.hpp"

template <typename T, typename TConstPtr, typename TPtr>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_geam_ex_template(rocblas_handle            handle,
                                      rocblas_operation         trans_A,
                                      rocblas_operation         trans_B,
                                      rocblas_int               m,
                                      rocblas_int               n,
                                      rocblas_int               k,
                                      const T*                  alpha,
                                      TConstPtr                 A,
                                      rocblas_stride            offset_A,
                                      rocblas_int               lda,
                                      rocblas_stride            stride_A,
                                      TConstPtr                 B,
                                      rocblas_stride            offset_B,
                                      rocblas_int               ldb,
                                      rocblas_stride            stride_B,
                                      const T*                  beta,
                                      TConstPtr                 C,
                                      rocblas_stride            offset_C,
                                      rocblas_int               ldc,
                                      rocblas_stride            stride_C,
                                      TPtr                      D,
                                      rocblas_stride            offset_D,
                                      rocblas_int               ldd,
                                      rocblas_stride            stride_D,
                                      rocblas_int               batch_count,
                                      rocblas_geam_ex_operation geam_ex_op)
{
    return geam_ex_source_solution<false, T>(handle,
                                             trans_A,
                                             trans_B,
                                             m,
                                             n,
                                             k,
                                             alpha,
                                             A,
                                             offset_A,
                                             lda,
                                             stride_A,
                                             B,
                                             offset_B,
                                             ldb,
                                             stride_B,
                                             beta,
                                             C,
                                             offset_C,
                                             ldc,
                                             stride_C,
                                             D,
                                             offset_D,
                                             ldd,
                                             stride_D,
                                             batch_count,
                                             geam_ex_op);
}

template <bool BATCHED, typename T>
rocblas_status rocblas_geam_ex_typecasting(rocblas_handle            handle,
                                           rocblas_operation         trans_A,
                                           rocblas_operation         trans_B,
                                           rocblas_int               m,
                                           rocblas_int               n,
                                           rocblas_int               k,
                                           const void*               alpha,
                                           const void*               A,
                                           rocblas_stride            offset_A,
                                           rocblas_int               lda,
                                           rocblas_stride            stride_A,
                                           const void*               B,
                                           rocblas_stride            offset_B,
                                           rocblas_int               ldb,
                                           rocblas_stride            stride_B,
                                           const void*               beta,
                                           const void*               C,
                                           rocblas_stride            offset_C,
                                           rocblas_int               ldc,
                                           rocblas_stride            stride_C,
                                           void*                     D,
                                           rocblas_stride            offset_D,
                                           rocblas_int               ldd,
                                           rocblas_stride            stride_D,
                                           rocblas_int               batch_count,
                                           rocblas_geam_ex_operation geam_ex_op)
{
    auto           check_numerics = handle->check_numerics;
    rocblas_status status         = rocblas_status_success;
    if(BATCHED)
    {
        if(check_numerics)
        {
            auto           check_numerics_string = "rocblas_geam_ex";
            bool           is_input              = true;
            rocblas_status gemm_min_plus_status
                = rocblas_geam_ex_check_numerics(check_numerics_string,
                                                 handle,
                                                 trans_A,
                                                 trans_B,
                                                 m,
                                                 n,
                                                 k,
                                                 (const T* const*)A,
                                                 lda,
                                                 stride_A,
                                                 (const T* const*)B,
                                                 ldb,
                                                 stride_B,
                                                 (const T* const*)C,
                                                 ldc,
                                                 stride_C,
                                                 (T* const*)D,
                                                 ldd,
                                                 stride_D,
                                                 batch_count,
                                                 check_numerics,
                                                 is_input);
            if(gemm_min_plus_status != rocblas_status_success)
                return gemm_min_plus_status;
        }

        status = rocblas_internal_geam_ex_template<T>(handle,
                                                      trans_A,
                                                      trans_B,
                                                      m,
                                                      n,
                                                      k,
                                                      (const T*)alpha,
                                                      (const T* const*)A,
                                                      offset_A,
                                                      lda,
                                                      stride_A,
                                                      (const T* const*)B,
                                                      offset_B,
                                                      ldb,
                                                      stride_B,
                                                      (const T*)beta,
                                                      (const T* const*)C,
                                                      offset_C,
                                                      ldc,
                                                      stride_C,
                                                      (T* const*)D,
                                                      offset_D,
                                                      ldd,
                                                      stride_D,
                                                      batch_count,
                                                      geam_ex_op);

        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            auto           check_numerics_string = "rocblas_geam_batched_ex";
            bool           is_input              = false;
            rocblas_status gemm_min_plus_status
                = rocblas_geam_ex_check_numerics(check_numerics_string,
                                                 handle,
                                                 trans_A,
                                                 trans_B,
                                                 m,
                                                 n,
                                                 k,
                                                 (const T* const*)A,
                                                 lda,
                                                 stride_A,
                                                 (const T* const*)B,
                                                 ldb,
                                                 stride_B,
                                                 (const T* const*)C,
                                                 ldc,
                                                 stride_C,
                                                 (T* const*)D,
                                                 ldd,
                                                 stride_D,
                                                 batch_count,
                                                 check_numerics,
                                                 is_input);
            if(gemm_min_plus_status != rocblas_status_success)
                return gemm_min_plus_status;
        }
    }
    else // not batched
    {
        if(check_numerics)
        {
            auto check_numerics_string
                = stride_A ? "rocblas_geam_strided_batched_ex" : "rocblas_geam_ex";
            bool           is_input = true;
            rocblas_status gemm_min_plus_status
                = rocblas_geam_ex_check_numerics(check_numerics_string,
                                                 handle,
                                                 trans_A,
                                                 trans_B,
                                                 m,
                                                 n,
                                                 k,
                                                 (const T*)A,
                                                 lda,
                                                 stride_A,
                                                 (const T*)B,
                                                 ldb,
                                                 stride_B,
                                                 (const T*)C,
                                                 ldc,
                                                 stride_C,
                                                 (T*)D,
                                                 ldd,
                                                 stride_D,
                                                 batch_count,
                                                 check_numerics,
                                                 is_input);
            if(gemm_min_plus_status != rocblas_status_success)
                return gemm_min_plus_status;
        }

        status = rocblas_internal_geam_ex_template<T>(handle,
                                                      trans_A,
                                                      trans_B,
                                                      m,
                                                      n,
                                                      k,
                                                      (const T*)alpha,
                                                      (const T*)A,
                                                      offset_A,
                                                      lda,
                                                      stride_A,
                                                      (const T*)B,
                                                      offset_B,
                                                      ldb,
                                                      stride_B,
                                                      (const T*)beta,
                                                      (const T*)C,
                                                      offset_C,
                                                      ldc,
                                                      stride_C,
                                                      (T*)D,
                                                      offset_D,
                                                      ldd,
                                                      stride_D,
                                                      batch_count,
                                                      geam_ex_op);

        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            auto check_numerics_string
                = stride_A ? "rocblas_geam_strided_batched_ex" : "rocblas_geam_ex";
            bool           is_input = false;
            rocblas_status gemm_min_plus_status
                = rocblas_geam_ex_check_numerics(check_numerics_string,
                                                 handle,
                                                 trans_A,
                                                 trans_B,
                                                 m,
                                                 n,
                                                 k,
                                                 (const T*)A,
                                                 lda,
                                                 stride_A,
                                                 (const T*)B,
                                                 ldb,
                                                 stride_B,
                                                 (const T*)C,
                                                 ldc,
                                                 stride_C,
                                                 (T*)D,
                                                 ldd,
                                                 stride_D,
                                                 batch_count,
                                                 check_numerics,
                                                 is_input);
            if(gemm_min_plus_status != rocblas_status_success)
                return gemm_min_plus_status;
        }
    }

    return rocblas_status_success;
}

template <bool BATCHED>
rocblas_status rocblas_geam_ex_template(rocblas_handle            handle,
                                        rocblas_operation         trans_A,
                                        rocblas_operation         trans_B,
                                        rocblas_int               m,
                                        rocblas_int               n,
                                        rocblas_int               k,
                                        const void*               alpha,
                                        const void*               A,
                                        rocblas_datatype          A_type,
                                        rocblas_stride            offset_A,
                                        rocblas_int               lda,
                                        rocblas_stride            stride_A,
                                        const void*               B,
                                        rocblas_datatype          B_type,
                                        rocblas_stride            offset_B,
                                        rocblas_int               ldb,
                                        rocblas_stride            stride_B,
                                        const void*               beta,
                                        const void*               C,
                                        rocblas_datatype          C_type,
                                        rocblas_stride            offset_C,
                                        rocblas_int               ldc,
                                        rocblas_stride            stride_C,
                                        void*                     D,
                                        rocblas_datatype          D_type,
                                        rocblas_stride            offset_D,
                                        rocblas_int               ldd,
                                        rocblas_stride            stride_D,
                                        rocblas_int               batch_count,
                                        rocblas_datatype          compute_type,
                                        rocblas_geam_ex_operation geam_ex_op)
{
    if(!m || !n || !batch_count)
        return rocblas_status_success;

    rocblas_status status = rocblas_status_not_implemented;

#define GEAM_EX_TYPECASTING_PARAM                                                            \
    handle, trans_A, trans_B, m, n, k, alpha, A, offset_A, lda, stride_A, B, offset_B, ldb,  \
        stride_B, beta, C, offset_C, ldc, stride_C, D, offset_D, ldd, stride_D, batch_count, \
        geam_ex_op

    if(A_type != B_type || B_type != C_type || C_type != D_type || D_type != compute_type)
        status = rocblas_status_not_implemented;
    else if(A_type == rocblas_datatype_f32_r)
        status = rocblas_geam_ex_typecasting<BATCHED, float>(GEAM_EX_TYPECASTING_PARAM);
    else if(A_type == rocblas_datatype_f64_r)
        status = rocblas_geam_ex_typecasting<BATCHED, double>(GEAM_EX_TYPECASTING_PARAM);
    else if(A_type == rocblas_datatype_f16_r)
        status = rocblas_geam_ex_typecasting<BATCHED, rocblas_half>(GEAM_EX_TYPECASTING_PARAM);
    else
        status = rocblas_status_not_implemented;

#undef GEAM_EX_TYPECASTING_PARAM

    return status;
}

template <typename TConstPtr, typename TPtr>
rocblas_status rocblas_geam_ex_check_numerics(const char*       function_name,
                                              rocblas_handle    handle,
                                              rocblas_operation trans_a,
                                              rocblas_operation trans_b,
                                              rocblas_int       m,
                                              rocblas_int       n,
                                              rocblas_int       k,
                                              TConstPtr         A,
                                              rocblas_int       lda,
                                              rocblas_stride    stride_a,
                                              TConstPtr         B,
                                              rocblas_int       ldb,
                                              rocblas_stride    stride_b,
                                              TConstPtr         C,
                                              rocblas_int       ldc,
                                              rocblas_stride    stride_c,
                                              TPtr              D,
                                              rocblas_int       ldd,
                                              rocblas_stride    stride_d,
                                              rocblas_int       batch_count,
                                              const int         check_numerics,
                                              bool              is_input)
{

    rocblas_status check_numerics_status
        = rocblas_internal_check_numerics_matrix_template(function_name,
                                                          handle,
                                                          trans_a,
                                                          rocblas_fill_full,
                                                          rocblas_client_general_matrix,
                                                          m,
                                                          k,
                                                          A,
                                                          0,
                                                          lda,
                                                          stride_a,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);
    if(check_numerics_status != rocblas_status_success)
        return check_numerics_status;

    check_numerics_status
        = rocblas_internal_check_numerics_matrix_template(function_name,
                                                          handle,
                                                          trans_b,
                                                          rocblas_fill_full,
                                                          rocblas_client_general_matrix,
                                                          k,
                                                          n,
                                                          B,
                                                          0,
                                                          ldb,
                                                          stride_b,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);
    if(check_numerics_status != rocblas_status_success)
        return check_numerics_status;

    check_numerics_status
        = rocblas_internal_check_numerics_matrix_template(function_name,
                                                          handle,
                                                          rocblas_operation_none,
                                                          rocblas_fill_full,
                                                          rocblas_client_general_matrix,
                                                          m,
                                                          n,
                                                          C,
                                                          0,
                                                          ldc,
                                                          stride_c,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);

    if(check_numerics_status != rocblas_status_success)
        return check_numerics_status;

    check_numerics_status
        = rocblas_internal_check_numerics_matrix_template(function_name,
                                                          handle,
                                                          rocblas_operation_none,
                                                          rocblas_fill_full,
                                                          rocblas_client_general_matrix,
                                                          m,
                                                          n,
                                                          D,
                                                          0,
                                                          ldd,
                                                          stride_d,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);

    return check_numerics_status;
}

// clang-format off

#ifdef INSTANTIATE_GEAM_EX_TEMPLATE
#error INSTANTIATE_GEAM_EX_TEMPLATE  already defined
#endif

#define INSTANTIATE_GEAM_EX_TEMPLATE(BATCHED)               \
template rocblas_status rocblas_geam_ex_template<BATCHED>      \
                                       (rocblas_handle     handle,       \
                                        rocblas_operation  trans_A,      \
                                        rocblas_operation  trans_B,      \
                                        rocblas_int        m,            \
                                        rocblas_int        n,            \
                                        rocblas_int        k,            \
                                        const void*        alpha,        \
                                        const void*        A,            \
                                        rocblas_datatype   A_type,       \
                                        rocblas_stride     offset_A,     \
                                        rocblas_int        lda,          \
                                        rocblas_stride     stride_A,     \
                                        const void*        B,            \
                                        rocblas_datatype   B_type,       \
                                        rocblas_stride     offset_B,     \
                                        rocblas_int        ldb,          \
                                        rocblas_stride     stride_B,     \
                                        const void*        beta,         \
                                        const void*        C,            \
                                        rocblas_datatype   C_type,       \
                                        rocblas_stride     offset_C,     \
                                        rocblas_int        ldc,          \
                                        rocblas_stride     stride_C,     \
                                        void*              D,            \
                                        rocblas_datatype   D_type,       \
                                        rocblas_stride     offset_D,     \
                                        rocblas_int        ldd,          \
                                        rocblas_stride     stride_D,     \
                                        rocblas_int        batch_count,  \
                                        rocblas_datatype   compute_type, \
                                        rocblas_geam_ex_operation geam_ex_op);

INSTANTIATE_GEAM_EX_TEMPLATE(false)
// INSTANTIATE_GEAM_EX_TEMPLATE(true)

#undef INSTANTIATE_GEAM_EX_TEMPLATE

// clang-format on

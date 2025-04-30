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

#include "rocblas_geam_ex.hpp"
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas.h"
#include "utility.hpp"

namespace
{
    rocblas_status rocblas_geam_ex_impl(rocblas_handle            handle,
                                        rocblas_operation         transA,
                                        rocblas_operation         transB,
                                        rocblas_int               m,
                                        rocblas_int               n,
                                        rocblas_int               k,
                                        const void*               alpha,
                                        const void*               A,
                                        rocblas_datatype          a_type,
                                        rocblas_int               lda,
                                        const void*               B,
                                        rocblas_datatype          b_type,
                                        rocblas_int               ldb,
                                        const void*               beta,
                                        const void*               C,
                                        rocblas_datatype          c_type,
                                        rocblas_int               ldc,
                                        void*                     D,
                                        rocblas_datatype          d_type,
                                        rocblas_int               ldd,
                                        rocblas_datatype          compute_type,
                                        rocblas_geam_ex_operation geam_ex_op)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        // Perform logging
        auto                    layer_mode = handle->layer_mode;
        rocblas_internal_logger logger;
        if(layer_mode
           & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
              | rocblas_layer_mode_log_profile))
        {
            char trans_a_letter, trans_b_letter;
            if(layer_mode & (rocblas_layer_mode_log_bench | rocblas_layer_mode_log_profile))
            {
                trans_a_letter = rocblas_transpose_letter(transA);
                trans_b_letter = rocblas_transpose_letter(transB);
            }
            auto a_type_string       = rocblas_datatype_string(a_type);
            auto b_type_string       = rocblas_datatype_string(b_type);
            auto c_type_string       = rocblas_datatype_string(c_type);
            auto d_type_string       = rocblas_datatype_string(d_type);
            auto compute_type_string = rocblas_datatype_string(compute_type);

            if(handle->pointer_mode == rocblas_pointer_mode_host)
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                {
                    rocblas_internal_ostream alphass, betass;
                    (void)rocblas_internal_log_trace_alpha_beta_ex(
                        compute_type, alpha, beta, alphass, betass);
                    logger.log_trace(handle,
                                     "rocblas_geam_ex",
                                     transA,
                                     transB,
                                     m,
                                     n,
                                     k,
                                     alphass.str(),
                                     A,
                                     a_type_string,
                                     lda,
                                     B,
                                     b_type_string,
                                     ldb,
                                     betass.str(),
                                     C,
                                     c_type_string,
                                     ldc,
                                     D,
                                     d_type_string,
                                     ldd,
                                     compute_type_string,
                                     geam_ex_op);
                }

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    std::string alphas, betas;
                    (void)rocblas_internal_log_bench_alpha_beta_ex(
                        compute_type, alpha, beta, alphas, betas);

                    logger.log_bench(handle,
                                     "./rocblas-bench -f geam_ex",
                                     "--transposeA",
                                     trans_a_letter,
                                     "--transposeB",
                                     trans_b_letter,
                                     "-m",
                                     m,
                                     "-n",
                                     n,
                                     "-k",
                                     k,
                                     alphas,
                                     "--a_type",
                                     a_type_string,
                                     "--lda",
                                     lda,
                                     "--b_type",
                                     b_type_string,
                                     "--ldb",
                                     ldb,
                                     betas,
                                     "--c_type",
                                     c_type_string,
                                     "--ldc",
                                     ldc,
                                     "--d_type",
                                     d_type_string,
                                     "--ldd",
                                     ldd,
                                     "--compute_type",
                                     compute_type_string,
                                     "--geam_ex_op",
                                     geam_ex_op);
                }

                if(layer_mode & rocblas_layer_mode_log_profile)
                {
                    logger.log_profile(handle,
                                       "rocblas_geam_ex",
                                       "a_type",
                                       a_type_string,
                                       "b_type",
                                       b_type_string,
                                       "c_type",
                                       c_type_string,
                                       "d_type",
                                       d_type_string,
                                       "compute_type",
                                       compute_type_string,
                                       "transA",
                                       trans_a_letter,
                                       "transB",
                                       trans_b_letter,
                                       "M",
                                       m,
                                       "N",
                                       n,
                                       "K",
                                       k,
                                       "alpha",
                                       rocblas_internal_value_category(alpha, compute_type),
                                       "lda",
                                       lda,
                                       "ldb",
                                       ldb,
                                       "beta",
                                       rocblas_internal_value_category(beta, compute_type),
                                       "ldc",
                                       ldc,
                                       "ldd",
                                       ldd,
                                       "geam_ex_op",
                                       geam_ex_op);
                }
            }
            else
            {
                if(layer_mode & rocblas_layer_mode_log_trace)
                {
                    logger.log_trace(handle,
                                     "rocblas_geam_ex",
                                     transA,
                                     transB,
                                     m,
                                     n,
                                     k,
                                     A,
                                     a_type_string,
                                     lda,
                                     B,
                                     b_type_string,
                                     ldb,
                                     C,
                                     c_type_string,
                                     ldc,
                                     D,
                                     d_type_string,
                                     ldd,
                                     compute_type_string,
                                     geam_ex_op);
                }
                if(layer_mode & rocblas_layer_mode_log_profile)
                {
                    logger.log_profile(handle,
                                       "rocblas_geam_ex",
                                       "a_type",
                                       a_type_string,
                                       "b_type",
                                       b_type_string,
                                       "c_type",
                                       c_type_string,
                                       "d_type",
                                       d_type_string,
                                       "compute_type",
                                       compute_type_string,
                                       "transA",
                                       trans_a_letter,
                                       "transB",
                                       trans_b_letter,
                                       "M",
                                       m,
                                       "N",
                                       n,
                                       "K",
                                       k,
                                       "lda",
                                       lda,
                                       "ldb",
                                       ldb,
                                       "ldc",
                                       ldc,
                                       "ldd",
                                       ldd,
                                       "geam_ex_op",
                                       geam_ex_op);
                }
            }
        }

        auto validArgs = rocblas_status_not_implemented;
        if(compute_type == rocblas_datatype_f16_r)
            validArgs = rocblas_geam_ex_arg_check<rocblas_half>(
                handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd);
        else if(compute_type == rocblas_datatype_f32_r)
            validArgs = rocblas_geam_ex_arg_check<float>(
                handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd);
        else if(compute_type == rocblas_datatype_f64_r)
            validArgs = rocblas_geam_ex_arg_check<double>(
                handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd);

        if(validArgs != rocblas_status_continue)
        {
            return validArgs;
        }

        rocblas_stride stride_zero = 0;
        rocblas_stride offset_zero = 0;
        rocblas_int    batch_count = 1;

        return rocblas_geam_ex_template<false>(handle,
                                               transA,
                                               transB,
                                               m,
                                               n,
                                               k,
                                               alpha,
                                               A,
                                               a_type,
                                               offset_zero,
                                               lda,
                                               stride_zero,
                                               B,
                                               b_type,
                                               offset_zero,
                                               ldb,
                                               stride_zero,
                                               beta,
                                               C,
                                               c_type,
                                               offset_zero,
                                               ldc,
                                               stride_zero,
                                               D,
                                               d_type,
                                               offset_zero,
                                               ldd,
                                               stride_zero,
                                               batch_count,
                                               compute_type,
                                               geam_ex_op);
    }
} // namespace

extern "C" rocblas_status rocblas_geam_ex(rocblas_handle            handle,
                                          rocblas_operation         transA,
                                          rocblas_operation         transB,
                                          rocblas_int               m,
                                          rocblas_int               n,
                                          rocblas_int               k,
                                          const void*               alpha,
                                          const void*               A,
                                          rocblas_datatype          a_type,
                                          rocblas_int               lda,
                                          const void*               B,
                                          rocblas_datatype          b_type,
                                          rocblas_int               ldb,
                                          const void*               beta,
                                          const void*               C,
                                          rocblas_datatype          c_type,
                                          rocblas_int               ldc,
                                          void*                     D,
                                          rocblas_datatype          d_type,
                                          rocblas_int               ldd,
                                          rocblas_datatype          compute_type,
                                          rocblas_geam_ex_operation geam_ex_op)
try
{
    return rocblas_geam_ex_impl(handle,
                                transA,
                                transB,
                                m,
                                n,
                                k,
                                alpha,
                                A,
                                a_type,
                                lda,
                                B,
                                b_type,
                                ldb,
                                beta,
                                C,
                                c_type,
                                ldc,
                                D,
                                d_type,
                                ldd,
                                compute_type,
                                geam_ex_op);
}
catch(...)
{
    return exception_to_rocblas_status();
}

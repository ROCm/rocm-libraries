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

#include "handle.hpp"
#include "rocblas.h"

#include "int64_helpers.hpp"
#include "logging.hpp"
#include "rocblas_gemm_ex.hpp"
#include "utility.hpp"

namespace
{
    template <typename API_INT>
    rocblas_status rocblas_gemm_batched_ex_impl(rocblas_handle    handle,
                                                rocblas_operation trans_a,
                                                rocblas_operation trans_b,
                                                API_INT           m,
                                                API_INT           n,
                                                API_INT           k,
                                                const void*       alpha,
                                                const void*       a,
                                                rocblas_datatype  a_type,
                                                API_INT           lda,
                                                const void*       b,
                                                rocblas_datatype  b_type,
                                                API_INT           ldb,
                                                const void*       beta,
                                                const void*       c,
                                                rocblas_datatype  c_type,
                                                API_INT           ldc,
                                                void*             d,
                                                rocblas_datatype  d_type,
                                                API_INT           ldd,
                                                API_INT           batch_count,
                                                rocblas_datatype  compute_type,
                                                rocblas_gemm_algo algo,
                                                int32_t           solution_index,
                                                uint32_t          flags)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        const bool HPA = compute_type == rocblas_datatype_f32_r
                         && (a_type == rocblas_datatype_f16_r || a_type == rocblas_datatype_bf16_r);

        if(!HPA)
            RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        // Copy alpha and beta to host if on device
        rocblas_union_t alpha_h, beta_h;
        RETURN_IF_ROCBLAS_ERROR(rocblas_copy_alpha_beta_to_host_if_on_device(
            handle, alpha, beta, alpha_h, beta_h, k, compute_type));
        auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

        rocblas_internal_logger logger;

        if(!handle->is_device_memory_size_query())
        {
            // Perform logging
            auto layer_mode = handle->layer_mode;
            if(layer_mode
               & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
                  | rocblas_layer_mode_log_profile))
            {
                char trans_a_letter, trans_b_letter;
                if(layer_mode & (rocblas_layer_mode_log_bench | rocblas_layer_mode_log_profile))
                {
                    trans_a_letter = rocblas_transpose_letter(trans_a);
                    trans_b_letter = rocblas_transpose_letter(trans_b);
                }
                auto a_type_string       = rocblas_datatype_string(a_type);
                auto b_type_string       = rocblas_datatype_string(b_type);
                auto c_type_string       = rocblas_datatype_string(c_type);
                auto d_type_string       = rocblas_datatype_string(d_type);
                auto compute_type_string = rocblas_datatype_string(compute_type);

                if(layer_mode & rocblas_layer_mode_log_trace)
                {
                    rocblas_internal_ostream alphass, betass;

                    (void)rocblas_internal_log_trace_alpha_beta_ex(
                        compute_type, alpha, beta, alphass, betass);
                    logger.log_trace(handle,
                                     ROCBLAS_API_STR(rocblas_gemm_batched_ex),
                                     trans_a,
                                     trans_b,
                                     m,
                                     n,
                                     k,
                                     alphass.str(),
                                     a,
                                     a_type_string,
                                     lda,
                                     b,
                                     b_type_string,
                                     ldb,
                                     betass.str(),
                                     c,
                                     c_type_string,
                                     ldc,
                                     d,
                                     d_type_string,
                                     ldd,
                                     batch_count,
                                     compute_type_string,
                                     algo,
                                     solution_index,
                                     rocblas_gemm_flags(flags));
                }

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    std::string alphas, betas;

                    (void)rocblas_internal_log_bench_alpha_beta_ex(
                        compute_type, alpha, beta, alphas, betas);

                    logger.log_bench(handle,
                                     ROCBLAS_API_BENCH " -f gemm_batched_ex",
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
                                     "--batch_count",
                                     batch_count,
                                     "--compute_type",
                                     compute_type_string,
                                     "--algo",
                                     algo,
                                     "--solution_index",
                                     solution_index,
                                     "--flags",
                                     flags);
                }

                if(layer_mode & rocblas_layer_mode_log_profile)
                {
                    logger.log_profile(handle,
                                       ROCBLAS_API_STR(rocblas_gemm_batched_ex),
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
                                       "batch_count",
                                       batch_count,
                                       "algo",
                                       algo,
                                       "solution_index",
                                       solution_index,
                                       "flags",
                                       rocblas_gemm_flags(flags));
                }
            }
        }

        auto validArgs = rocblas_gemm_ex_arg_check(handle,
                                                   trans_a,
                                                   trans_b,
                                                   m,
                                                   n,
                                                   k,
                                                   alpha,
                                                   a,
                                                   lda,
                                                   b,
                                                   ldb,
                                                   beta,
                                                   c,
                                                   c_type,
                                                   ldc,
                                                   d,
                                                   d_type,
                                                   ldd,
                                                   compute_type,
                                                   batch_count);

        if(validArgs != rocblas_status_continue)
        {
            if(validArgs == rocblas_status_success)
                RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);
            return validArgs;
        }

        auto stride_a = rocblas_stride(lda) * (trans_a == rocblas_operation_none ? k : m);
        auto stride_b = rocblas_stride(ldb) * (trans_b == rocblas_operation_none ? n : k);
        auto stride_c = rocblas_stride(ldc) * n;
        auto stride_d = rocblas_stride(ldd) * n;

        return ROCBLAS_API(rocblas_gemm_ex_template)<true>(handle,
                                                           trans_a,
                                                           trans_b,
                                                           m,
                                                           n,
                                                           k,
                                                           alpha,
                                                           a,
                                                           a_type,
                                                           0,
                                                           lda,
                                                           stride_a,
                                                           b,
                                                           b_type,
                                                           0,
                                                           ldb,
                                                           stride_b,
                                                           beta,
                                                           c,
                                                           c_type,
                                                           0,
                                                           ldc,
                                                           stride_c,
                                                           d,
                                                           d_type,
                                                           0,
                                                           ldd,
                                                           stride_d,
                                                           batch_count,
                                                           compute_type,
                                                           algo,
                                                           solution_index,
                                                           flags);
    }
}
// namespace

#define INST_GEMM_BATCHED_EX_C_API(TI_)                                                   \
    extern "C" {                                                                          \
    rocblas_status ROCBLAS_API(rocblas_gemm_batched_ex)(rocblas_handle    handle,         \
                                                        rocblas_operation trans_a,        \
                                                        rocblas_operation trans_b,        \
                                                        TI_               m,              \
                                                        TI_               n,              \
                                                        TI_               k,              \
                                                        const void*       alpha,          \
                                                        const void*       a,              \
                                                        rocblas_datatype  a_type,         \
                                                        TI_               lda,            \
                                                        const void*       b,              \
                                                        rocblas_datatype  b_type,         \
                                                        TI_               ldb,            \
                                                        const void*       beta,           \
                                                        const void*       c,              \
                                                        rocblas_datatype  c_type,         \
                                                        TI_               ldc,            \
                                                        void*             d,              \
                                                        rocblas_datatype  d_type,         \
                                                        TI_               ldd,            \
                                                        TI_               batch_count,    \
                                                        rocblas_datatype  compute_type,   \
                                                        rocblas_gemm_algo algo,           \
                                                        int32_t           solution_index, \
                                                        uint32_t          flags)          \
    try                                                                                   \
    {                                                                                     \
        return rocblas_gemm_batched_ex_impl(handle,                                       \
                                            trans_a,                                      \
                                            trans_b,                                      \
                                            m,                                            \
                                            n,                                            \
                                            k,                                            \
                                            alpha,                                        \
                                            a,                                            \
                                            a_type,                                       \
                                            lda,                                          \
                                            b,                                            \
                                            b_type,                                       \
                                            ldb,                                          \
                                            beta,                                         \
                                            c,                                            \
                                            c_type,                                       \
                                            ldc,                                          \
                                            d,                                            \
                                            d_type,                                       \
                                            ldd,                                          \
                                            batch_count,                                  \
                                            compute_type,                                 \
                                            algo,                                         \
                                            solution_index,                               \
                                            flags);                                       \
    }                                                                                     \
    catch(...)                                                                            \
    {                                                                                     \
        return exception_to_rocblas_status();                                             \
    }                                                                                     \
    }

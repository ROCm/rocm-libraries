/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "handle.hpp"
#include "int64_helpers.hpp"
#include "logging.hpp"
#include "rocblas.h"
#include "rocblas_block_sizes.h"
#include "rocblas_gemm_ex.hpp"
#include "rocblas_syrk_ex.hpp"
#include "utility.hpp"

namespace
{
    template <typename API_INT>
    rocblas_status rocblas_herk_ex_impl(rocblas_handle    handle,
                                        rocblas_fill      uplo,
                                        rocblas_operation transA,
                                        API_INT           n,
                                        API_INT           k,
                                        const void*       alpha,
                                        const void*       A,
                                        rocblas_datatype  a_type,
                                        API_INT           lda,
                                        const void*       beta,
                                        void*             C,
                                        rocblas_datatype  c_type,
                                        API_INT           ldc,
                                        rocblas_datatype  compute_type)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        constexpr bool HERM = true;

        //Check if the handle is in the device memory size query, as there are two algorithms one which requires extra workspace memory and one which doesn't
        if(handle->is_device_memory_size_query())
        {
            //If rocblas_use_only_gemm is true then it is required to allocate extra workspace memory
            if(rocblas_internal_syrk_herk_ex_use_gemm<HERM>(compute_type))
            {
                if(!n)
                    return rocblas_status_size_unchanged;
                size_t size = rocblas_internal_syrk_herk_ex_workspace<HERM>(handle, n, k, 1);
                return handle->set_optimal_device_memory_size(size);
            }
            else
                RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);
        }

        // Copy alpha and beta to host if on device
        rocblas_datatype alpha_beta_type = compute_type == rocblas_datatype_f64_c
                                               ? rocblas_datatype_f64_r
                                               : rocblas_datatype_f32_r;
        rocblas_union_t  alpha_h, beta_h;
        RETURN_IF_ROCBLAS_ERROR(rocblas_copy_alpha_beta_to_host_if_on_device(
            handle, alpha, beta, alpha_h, beta_h, k, alpha_beta_type));
        auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

        auto                    layer_mode     = handle->layer_mode;
        auto                    check_numerics = handle->check_numerics;
        rocblas_internal_logger logger;
        if(layer_mode
           & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
              | rocblas_layer_mode_log_profile))
        {
            auto uplo_letter   = rocblas_fill_letter(uplo);
            auto transA_letter = rocblas_transpose_letter(transA);
            auto a_type_str    = rocblas_datatype_string(a_type);
            auto c_type_str    = rocblas_datatype_string(c_type);
            auto ex_type_str   = rocblas_datatype_string(compute_type);

            rocblas_internal_logger logger;

            if(layer_mode & rocblas_layer_mode_log_trace)
            {
                rocblas_internal_ostream alphass, betass;
                (void)rocblas_internal_log_trace_alpha_beta_ex(
                    alpha_beta_type, alpha, beta, alphass, betass);

                logger.log_trace(handle,
                                 ROCBLAS_API_STR(rocblas_herk_ex),
                                 uplo,
                                 transA,
                                 n,
                                 k,
                                 alphass.str(),
                                 A,
                                 a_type_str,
                                 lda,
                                 betass.str(),
                                 C,
                                 c_type_str,
                                 ldc,
                                 ex_type_str);
            }

            if(layer_mode & rocblas_layer_mode_log_bench)
            {
                std::string alphas, betas;
                (void)rocblas_internal_log_bench_alpha_beta_ex(
                    alpha_beta_type, alpha, beta, alphas, betas);

                logger.log_bench(handle,
                                 ROCBLAS_API_BENCH " -f herk_ex",
                                 "--uplo",
                                 uplo_letter,
                                 "--transposeA",
                                 transA_letter,
                                 "-n",
                                 n,
                                 "-k",
                                 k,
                                 alphas,
                                 "--a_type",
                                 a_type_str,
                                 "--lda",
                                 lda,
                                 betas,
                                 "--c_type",
                                 c_type_str,
                                 "--ldc",
                                 ldc,
                                 "--compute_type",
                                 ex_type_str);
            }

            if(layer_mode & rocblas_layer_mode_log_profile)
            {
                logger.log_profile(handle,
                                   ROCBLAS_API_STR(rocblas_herk_ex),
                                   "a_type",
                                   a_type_str,
                                   "c_type",
                                   c_type_str,
                                   "compute_type",
                                   ex_type_str,
                                   "transA",
                                   transA_letter,
                                   "N",
                                   n,
                                   "K",
                                   k,
                                   "alpha",
                                   rocblas_internal_value_category(alpha, alpha_beta_type),
                                   "lda",
                                   lda,
                                   "beta",
                                   rocblas_internal_value_category(beta, alpha_beta_type),
                                   "ldc",
                                   ldc);
            }
        }

        static constexpr rocblas_stride offset_C = 0, offset_A = 0;
        static constexpr API_INT        batch_count = 1;
        static constexpr rocblas_stride stride_C = 0, stride_A = 0;

        return ROCBLAS_API(rocblas_herk_ex_template)<false>(handle,
                                                            uplo,
                                                            transA,
                                                            n,
                                                            k,
                                                            alpha,
                                                            A,
                                                            a_type,
                                                            offset_A,
                                                            lda,
                                                            stride_A,
                                                            beta,
                                                            C,
                                                            c_type,
                                                            offset_C,
                                                            ldc,
                                                            stride_C,
                                                            compute_type,
                                                            batch_count);
    }

} // namespace

#define INST_HERK_EX_C_API(TI_)                                                 \
    extern "C" {                                                                \
    rocblas_status ROCBLAS_API(rocblas_herk_ex)(rocblas_handle    handle,       \
                                                rocblas_fill      uplo,         \
                                                rocblas_operation trans_a,      \
                                                TI_               n,            \
                                                TI_               k,            \
                                                const void*       alpha,        \
                                                const void*       a,            \
                                                rocblas_datatype  a_type,       \
                                                TI_               lda,          \
                                                const void*       beta,         \
                                                void*             c,            \
                                                rocblas_datatype  c_type,       \
                                                TI_               ldc,          \
                                                rocblas_datatype  compute_type) \
    try                                                                         \
    {                                                                           \
        return rocblas_herk_ex_impl(handle,                                     \
                                    uplo,                                       \
                                    trans_a,                                    \
                                    n,                                          \
                                    k,                                          \
                                    alpha,                                      \
                                    a,                                          \
                                    a_type,                                     \
                                    lda,                                        \
                                    beta,                                       \
                                    c,                                          \
                                    c_type,                                     \
                                    ldc,                                        \
                                    compute_type);                              \
    }                                                                           \
    catch(...)                                                                  \
    {                                                                           \
        return exception_to_rocblas_status();                                   \
    }                                                                           \
    }

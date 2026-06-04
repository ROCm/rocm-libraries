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

#include "check_numerics_matrix.hpp"
#include "handle.hpp"

#ifdef BUILD_WITH_TENSILE

#include "int64_helpers.hpp"
#include "tensile_host.hpp"

/*******************************************************************************
 * Tensile Function call
 ******************************************************************************/
template <typename Ti, typename To, typename Tc>
inline rocblas_status rocblas_call_tensile(rocblas_handle     handle,
                                           const Tc*          alpha,
                                           const Tc*          beta,
                                           const Ti* const*   batchA,
                                           const Ti* const*   batchB,
                                           const To* const*   batchC,
                                           To* const*         batchD,
                                           rocblas_operation  trans_a,
                                           rocblas_operation  trans_b,
                                           rocblas_int        ld_d,
                                           rocblas_stride     stride_d,
                                           rocblas_stride     offset_d,
                                           rocblas_int        ld_c,
                                           rocblas_stride     stride_c,
                                           rocblas_stride     offset_c,
                                           rocblas_int        ld_a,
                                           rocblas_stride     stride_a,
                                           rocblas_stride     offset_a,
                                           rocblas_int        ld_b,
                                           rocblas_stride     stride_b,
                                           rocblas_stride     offset_b,
                                           rocblas_int        m,
                                           rocblas_int        n,
                                           rocblas_int        k,
                                           rocblas_int        batch_count = 1,
                                           rocblas_gemm_algo  algo = rocblas_gemm_algo_standard,
                                           int32_t            solution_index = 0,
                                           rocblas_gemm_flags flags = rocblas_gemm_flags_none)
{
    size_t grid_z_limit = handle->getBatchGridDim(batch_count);

    for(size_t b_base = 0; b_base < batch_count; b_base += grid_z_limit)
    {
        auto A_ptr = adjust_ptr_batch(batchA, b_base, stride_a);
        auto B_ptr = adjust_ptr_batch(batchB, b_base, stride_b);
        auto C_ptr = adjust_ptr_batch(batchC, b_base, stride_c);
        auto D_ptr = adjust_ptr_batch(batchD, b_base, stride_d);

        int32_t batches = int32_t(std::min(batch_count - b_base, grid_z_limit));

#if 0
    // if tensile supports we can remove special case handling here
    if(k == 0 || (alpha && !*alpha))
    {
        // !beta early return and beta always on host here so can dereference
        return rocblas_gemm_ex_scale_template(handle,
                                              m,
                                              n,
                                              *beta,
                                              C_ptr,
                                              offset_c,
                                              ld_c,
                                              stride_c,
                                              D_ptr,
                                              offset_d,
                                              ld_d,
                                              stride_d,
                                              batches);
    }
#endif

        RocblasContractionProblem<Ti, To, Tc> problem{
            handle,   trans_a, trans_b,  m,        n,       k,        alpha,    nullptr,
            A_ptr,    ld_a,    stride_a, offset_a, nullptr, B_ptr,    ld_b,     stride_b,
            offset_b, beta,    nullptr,  C_ptr,    ld_c,    stride_c, offset_c, nullptr,
            D_ptr,    ld_d,    stride_d, offset_d, batches, false,    flags};

        RETURN_IF_ROCBLAS_ERROR(runContractionProblem(problem, algo, solution_index));
    }
    return rocblas_status_success;
}

template <typename Ti, typename To, typename Tc>
inline rocblas_status rocblas_call_tensile(rocblas_handle     handle,
                                           const Tc*          alpha,
                                           const Tc*          beta,
                                           const Ti*          A,
                                           const Ti*          B,
                                           const To*          C,
                                           To*                D,
                                           rocblas_operation  trans_a,
                                           rocblas_operation  trans_b,
                                           rocblas_int        ld_d,
                                           rocblas_stride     stride_d,
                                           rocblas_stride     offset_d,
                                           rocblas_int        ld_c,
                                           rocblas_stride     stride_c,
                                           rocblas_stride     offset_c,
                                           rocblas_int        ld_a,
                                           rocblas_stride     stride_a,
                                           rocblas_stride     offset_a,
                                           rocblas_int        ld_b,
                                           rocblas_stride     stride_b,
                                           rocblas_stride     offset_b,
                                           rocblas_int        m,
                                           rocblas_int        n,
                                           rocblas_int        k,
                                           rocblas_int        batch_count = 1,
                                           rocblas_gemm_algo  algo = rocblas_gemm_algo_standard,
                                           int32_t            solution_index = 0,
                                           rocblas_gemm_flags flags = rocblas_gemm_flags_none)
{
    size_t grid_z_limit = handle->getBatchGridDim(batch_count);

    for(size_t b_base = 0; b_base < batch_count; b_base += grid_z_limit)
    {
        auto A_ptr = adjust_ptr_batch(A, b_base, stride_a);
        auto B_ptr = adjust_ptr_batch(B, b_base, stride_b);
        auto C_ptr = adjust_ptr_batch(C, b_base, stride_c);
        auto D_ptr = adjust_ptr_batch(D, b_base, stride_d);

        int32_t batches = int32_t(std::min(batch_count - b_base, grid_z_limit));

#if 0
    // if tensile supports we can remove special case handling here
    if(k == 0 || (alpha && !*alpha))
    {
        // !beta early return and beta always on host here so can dereference
        return rocblas_gemm_ex_scale_template(handle,
                                              m,
                                              n,
                                              *beta,
                                              C_ptr,
                                              offset_c,
                                              ld_c,
                                              stride_c,
                                              D_ptr,
                                              offset_d,
                                              ld_d,
                                              stride_d,
                                              batches);
    }
#endif

        // ALMIOPEN-1609 FIX: chunk k dimension when stride*k could overflow 32-bit
        // Note: k==0 is a valid BLAS operation (D = beta*C), must not skip it.
        // Tensile kernels use 32-bit offset = column * stride, which overflows
        // when stride (=lda or ldb) is large and k is large.
        // Safe k_chunk: k_chunk * max(lda,ldb) * sizeof(element) < 2^31
        const auto A_base = A_ptr + offset_a;
        const auto B_base = B_ptr + offset_b;
        const auto C_base = C_ptr + offset_c;
        auto       D_base = D_ptr + offset_d;

        // Calculate safe k_chunk_size to keep offsets within signed 32-bit
        constexpr int64_t overflow_limit_2_to_31 = 2147483647; // 2^31 - 1
        int64_t           max_stride             = std::max(int64_t(ld_a), int64_t(ld_b));
        int64_t           bytes_per_element      = sizeof(Ti);
        int64_t           safe_k
            = (max_stride > 0 && max_stride * bytes_per_element > 0)
                  ? std::max(int64_t(1), overflow_limit_2_to_31 / (max_stride * bytes_per_element))
                  : int64_t(k);
        // Only chunk if needed (when k would overflow)
        // Ensure k_chunk >= 1 to avoid infinite loop when k=0
        int64_t k_chunk = (safe_k >= k) ? std::max(int64_t(k), int64_t(1)) : safe_k;

        Tc beta_val = *beta;
        Tc one_val  = static_cast<Tc>(1);

        for(int64_t k_base = 0; k_base < std::max(int64_t(k), int64_t(1)); k_base += k_chunk)
        {
            int32_t kblock = int32_t(std::min(int64_t(k) - k_base, k_chunk));

            // Adjust A: advance k_base columns (transA=N) or k_base rows (transA=T)
            auto a_k_offset = (trans_a == rocblas_operation_none)
                                  ? int64_t(k_base) * ld_a // column-major: col offset
                                  : k_base; // transposed: row offset
            // Adjust B: advance k_base rows (transB=N) or k_base columns (transB=T)
            auto b_k_offset = (trans_b == rocblas_operation_none)
                                  ? k_base // column-major: row offset
                                  : int64_t(k_base) * ld_b; // transposed: col offset

            // First chunk uses original beta; subsequent use beta=1 to accumulate
            const Tc* chunk_beta = (k_base == 0) ? beta : &one_val;

            RocblasContractionProblem<Ti, To, Tc> problem{handle,
                                                          trans_a,
                                                          trans_b,
                                                          m,
                                                          n,
                                                          kblock,
                                                          alpha,
                                                          A_base + a_k_offset,
                                                          nullptr,
                                                          ld_a,
                                                          stride_a,
                                                          0 /* offset_a */,
                                                          B_base + b_k_offset,
                                                          nullptr,
                                                          ld_b,
                                                          stride_b,
                                                          0 /* offset_b */,
                                                          chunk_beta,
                                                          C_base,
                                                          nullptr,
                                                          ld_c,
                                                          stride_c,
                                                          0 /* offset_c */,
                                                          D_base,
                                                          nullptr,
                                                          ld_d,
                                                          stride_d,
                                                          0 /* offset_d */,
                                                          batches,
                                                          true,
                                                          flags};

            RETURN_IF_ROCBLAS_ERROR(runContractionProblem(problem, algo, solution_index));
        }
    }
    return rocblas_status_success;
}

#endif // BUILD_WITH_TENSILE

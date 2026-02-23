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

#include "handle.hpp"
#include "logging.hpp"
#include "rocblas_gemm_ex.hpp"
#include "rocblas_herk_ex.hpp"
#include "rocblas_syrk_herk_ex_kernels.hpp"

#include "../blas3/herk_syrk_device.hpp"
#include "../blas3/rocblas_syrk_herk.hpp"
#include "../blas3/rocblas_syrk_imp.hpp"

template <typename Tex, typename Tab_ex, typename Ta, typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_herk_ex_template(rocblas_handle    handle,
                                      rocblas_fill      uplo,
                                      rocblas_operation trans_A,
                                      rocblas_int       n,
                                      rocblas_int       k,
                                      const Tab_ex*     alpha_in,
                                      const Ta*         A,
                                      rocblas_datatype  A_type,
                                      rocblas_stride    offset_A,
                                      rocblas_int       lda,
                                      rocblas_stride    stride_A,
                                      const Tab_ex*     beta_in,
                                      T*                C,
                                      rocblas_datatype  C_type,
                                      rocblas_stride    offset_C,
                                      rocblas_int       ldc,
                                      rocblas_stride    stride_C,
                                      rocblas_datatype  compute_type,
                                      rocblas_int       batch_count)
{
    constexpr bool BATCHED = false;
    constexpr bool HERM    = true;

    /* TODO if constexpr(rocblas_internal_syrk_herk_ex_use_gemm<Tex>())
    {
        size_t size
            = rocblas_internal_syrk_herk_ex_workspace<HERM>(handle, n, k, batch_count);

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
        const Tab_ex alpha_val = (Tab_ex)(*alpha_in);
        const Tab_ex beta_val  = (Tab_ex)(*beta_in);

        const Tab_ex* alpha = &alpha_val;
        const Tab_ex* beta  = &beta_val;

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
    else */
    {
        using API_INT = rocblas_int;

        hipStream_t stream = handle->get_stream();

        // Note: alpha and beta always copied over to host by now
        if(*beta_in == 1 && (k == 0 || *alpha_in == 0))
            return rocblas_status_success;

        bool a_calc_invalid = !alpha_in || (*alpha_in != 0 && (!A));
        if(!C || (k && a_calc_invalid))
            return rocblas_status_invalid_pointer;

        int batches = handle->getBatchGridDim((int)batch_count);

        // syrkx has same behavior for alpha == 0 and k == 0. Special code is needed
        // for alpha == 0, no special code is needed for k == 0. It is more efficient
        // setting k = 0 than adding extra code to a kernel to handle alpha == 0
        if(*alpha_in == 0)
            k = 0;

        const Tab_ex alpha = (Tab_ex)(*alpha_in);
        const Tab_ex beta  = (Tab_ex)(*beta_in);

        if((n % 32 == 0) && (k % 8 == 0)) // restricted kernels
        {
            // Can also be used with:
            // n is mult of 64, k is mult of 4
            // const int dim_n = 16;
            // const int blk_n = 64;
            // const int blk_k = 4;

            // n is mult of 32, k is mult of 8
            const int dim_n = 16;
            const int blk_n = 32;
            const int blk_k = 8;
            dim3      dimBlock(dim_n, dim_n, 1);
            dim3      dimGrid(n / blk_n, n / blk_n, batches);

            // clang-format off
            if(alpha == Tab_ex(1.0) && beta == Tab_ex(1.0))
            {
                if((rocblas_operation_conjugate_transpose == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, 1, 1, HERM, 'C', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_none == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, 1, 1, HERM, 'N', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_conjugate_transpose == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, 1, 1, HERM, 'C', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_none == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, 1, 1, HERM, 'N', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
            }
            else if(alpha == Tab_ex(1.0) && beta == Tab_ex(-1.0))
            {
                if((rocblas_operation_conjugate_transpose == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, 1, -1, HERM, 'C', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_none == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, 1, -1, HERM, 'N', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_conjugate_transpose == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, 1, -1, HERM, 'C', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_none == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, 1, -1, HERM, 'N', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
            }
            else if(alpha == Tab_ex(1.0) && beta == Tab_ex(0.0))
            {
                if((rocblas_operation_conjugate_transpose == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, 1, 0, HERM, 'C', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_none == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, 1, 0, HERM, 'N', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_conjugate_transpose == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, 1, 0, HERM, 'C', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_none == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, 1, 0, HERM, 'N', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
            }
            else if(alpha == Tab_ex(-1.0) && beta == Tab_ex(0.0))
            {
                if((rocblas_operation_conjugate_transpose == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, -1, 0, HERM, 'C', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_none == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, -1, 0, HERM, 'N', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_conjugate_transpose == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                    <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, -1, 0, HERM, 'C', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_none == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_kernel
                        <API_INT, Tex, Ta, T, dim_n, blk_n, blk_k, -1, 0, HERM, 'N', 'U'>),
                        dimGrid, dimBlock, 0, stream, n, k, A, lda, stride_A, A, lda, stride_A, C, ldc, stride_C, batch_count);
            }
            else if(beta == Tab_ex(0))
            {
                // general alpha; beta == 0
                if((rocblas_operation_conjugate_transpose == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_general_kernel
                    <API_INT, Tex, Tab_ex, Ta, T, dim_n, blk_n, blk_k, true, HERM, 'C', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_none == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_general_kernel
                    <API_INT, Tex, Tab_ex, Ta, T, dim_n, blk_n, blk_k, true, HERM, 'N', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_conjugate_transpose == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_general_kernel
                    <API_INT, Tex, Tab_ex, Ta, T, dim_n, blk_n, blk_k, true, HERM, 'C', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_none == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_general_kernel
                    <API_INT, Tex, Tab_ex, Ta, T, dim_n, blk_n, blk_k, true, HERM, 'N', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
            }
            else
            {
                // general alpha, beta
                if((rocblas_operation_conjugate_transpose == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_general_kernel
                    <API_INT, Tex, Tab_ex, Ta, T, dim_n, blk_n, blk_k, false, HERM, 'C', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_none == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_general_kernel
                    <API_INT, Tex, Tab_ex, Ta, T, dim_n, blk_n, blk_k, false, HERM, 'N', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_conjugate_transpose == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_general_kernel
                    <API_INT, Tex, Tab_ex, Ta, T, dim_n, blk_n, blk_k, false, HERM, 'C', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_none == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_restricted_general_kernel
                    <API_INT, Tex, Tab_ex, Ta, T, dim_n, blk_n, blk_k, false, HERM, 'N', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
            }
            // clang-format on
        }
        else
        {
            const int dim_n = 16;
            const int blk_n = 32;
            const int blk_k = 8;
            dim3      dimBlock(dim_n, dim_n, 1);
            dim3      dimGrid(((n - 1) / blk_n) + 1, ((n - 1) / blk_n) + 1, batches);

            // clang-format off
            if(beta == Tab_ex(0))
            {
                // general n, k, alpha; beta == 0
                if((rocblas_operation_conjugate_transpose == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_general_kernel
                    <API_INT, Tex, Tab_ex, Ta, T, dim_n, blk_n, blk_k, true, HERM, 'C', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_none == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_general_kernel
                    <API_INT, Tex, Tab_ex, Ta, T, dim_n, blk_n, blk_k, true, HERM, 'N', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_conjugate_transpose == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_general_kernel
                    <API_INT, Tex, Tab_ex, Ta, T, dim_n, blk_n, blk_k, true, HERM, 'C', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_none == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_general_kernel
                    <API_INT, Tex, Tab_ex, Ta, T, dim_n, blk_n, blk_k, true, HERM, 'N', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
            }
            else
            {
                // general n, k, alpha, beta
                if((rocblas_operation_conjugate_transpose == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_general_kernel
                    <API_INT, Tex, Tab_ex, Ta, T, dim_n, blk_n, blk_k, false, HERM, 'C', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_none == trans_A) && (rocblas_fill_lower == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_general_kernel
                    <API_INT, Tex, Tab_ex, Ta, T, dim_n, blk_n, blk_k, false, HERM, 'N', 'L'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_conjugate_transpose == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_general_kernel
                    <API_INT, Tex, Tab_ex, Ta, T, dim_n, blk_n, blk_k, false, HERM, 'C', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
                else if((rocblas_operation_none == trans_A) && (rocblas_fill_upper == uplo))
                    ROCBLAS_LAUNCH_KERNEL((rocblas_syrkx_herkx_ex_general_kernel
                    <API_INT, Tex, Tab_ex, Ta, T, dim_n, blk_n, blk_k, false, HERM, 'N', 'U'>),
                    dimGrid, dimBlock, 0, stream, n, k, alpha, A, lda, stride_A, A, lda, stride_A, beta, C, ldc, stride_C, batch_count);
            }
            // clang-format on
        }

        return rocblas_status_success;
    }
}

template <bool BATCHED, typename T, typename U, typename Tex>
rocblas_status rocblas_herk_ex_typecasting(rocblas_handle    handle,
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
    using Tab_ex = real_t<Tex>;

    auto check_numerics = handle->check_numerics;

    rocblas_status status = rocblas_herk_arg_check<rocblas_int>(handle,
                                                                uplo,
                                                                trans_A,
                                                                n,
                                                                k,
                                                                (Tab_ex*)alpha,
                                                                (const T*)A,
                                                                offset_A,
                                                                lda,
                                                                stride_A,
                                                                (Tab_ex*)beta,
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
                = stride_A ? "rocblas_herk_strided_batched_ex" : "rocblas_herk_ex";
            bool is_input = true;
            status        = rocblas_herk_syrk_check_numerics<true>(check_numerics_string,
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

        status = rocblas_internal_herk_ex_template<Tex>(handle,
                                                        uplo,
                                                        trans_A,
                                                        n,
                                                        k,
                                                        (const Tab_ex*)alpha,
                                                        (const T*)A,
                                                        A_type,
                                                        offset_A,
                                                        lda,
                                                        stride_A,
                                                        (const Tab_ex*)beta,
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
                = stride_A ? "rocblas_herk_strided_batched_ex" : "rocblas_herk_ex";
            bool is_input = false;
            status        = rocblas_herk_syrk_check_numerics<true>(check_numerics_string,
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
rocblas_status rocblas_herk_ex_template(rocblas_handle    handle,
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

#define HERK_EX_TYPECASTING_PARAM                                                            \
    handle, uplo, trans_A, n, k, alpha, A, A_type, offset_A, lda, stride_A, beta, C, C_type, \
        offset_C, ldc, stride_C, compute_type, batch_count

    if(A_type == rocblas_datatype_f32_c && C_type == rocblas_datatype_f32_c
       && compute_type == rocblas_datatype_f64_c)
        status = rocblas_herk_ex_typecasting<BATCHED,
                                             rocblas_float_complex,
                                             rocblas_float_complex,
                                             rocblas_double_complex>(HERK_EX_TYPECASTING_PARAM);
    else if(A_type == rocblas_datatype_f32_c && C_type == rocblas_datatype_f64_c
            && compute_type == rocblas_datatype_f64_c)
        status = rocblas_herk_ex_typecasting<BATCHED,
                                             rocblas_float_complex,
                                             rocblas_double_complex,
                                             rocblas_double_complex>(HERK_EX_TYPECASTING_PARAM);
    else
        status = rocblas_status_not_implemented;

#undef HERK_EX_TYPECASTING_PARAM

    return status;
}

#ifdef INSTANTIATE_HERK_EX_TEMPLATE
#error INSTANTIATE_HERK_EX_TEMPLATE  already defined
#endif

#define INSTANTIATE_HERK_EX_TEMPLATE(BATCHED)                                                 \
    template rocblas_status rocblas_herk_ex_template<BATCHED>(rocblas_handle    handle,       \
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

INSTANTIATE_HERK_EX_TEMPLATE(false)
// INSTANTIATE_HERK_EX_TEMPLATE(true)

#undef INSTANTIATE_HERK_EX_TEMPLATE

/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

#include "csrmm/nnz_split/kernel_declarations.h"
#include "csrmm_device_nnz_split.h"
#include "rocsparse_common.h"
#include "rocsparse_csrmm.hpp"

#define NNZ_PER_BLOCK 256

namespace rocsparse
{
    template <typename T, typename I, typename J, typename A>
    rocsparse_status csrmm_buffer_size_template_nnz_split(rocsparse_handle          handle,
                                                          rocsparse_operation       trans_A,
                                                          rocsparse_csrmm_alg       alg,
                                                          J                         m,
                                                          J                         n,
                                                          J                         k,
                                                          I                         nnz,
                                                          const rocsparse_mat_descr descr,
                                                          const A*                  csr_val,
                                                          const I*                  csr_row_ptr,
                                                          const J*                  csr_col_ind,
                                                          size_t*                   buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(trans_A)
        {
        case rocsparse_operation_none:
        {
            I nblocks = (nnz - 1) / NNZ_PER_BLOCK + 1;

            *buffer_size = 0;
            *buffer_size += sizeof(J) * ((nblocks + 1 - 1) / 256 + 1) * 256; // row limits
            *buffer_size += sizeof(J) * ((nblocks - 1) / 256 + 1) * 256; // row block red
            *buffer_size += sizeof(T) * ((nblocks * n - 1) / 256 + 1) * 256; // val block red

            return rocsparse_status_success;
        }
        case rocsparse_operation_transpose:
        case rocsparse_operation_conjugate_transpose:
        {
            *buffer_size = 0;
            return rocsparse_status_success;
        }
        }
    }

    template <typename I, typename J, typename A>
    rocsparse_status csrmm_analysis_template_nnz_split(rocsparse_handle          handle,
                                                       rocsparse_operation       trans_A,
                                                       rocsparse_csrmm_alg       alg,
                                                       J                         m,
                                                       J                         n,
                                                       J                         k,
                                                       I                         nnz,
                                                       const rocsparse_mat_descr descr,
                                                       const A*                  csr_val,
                                                       const I*                  csr_row_ptr,
                                                       const J*                  csr_col_ind,
                                                       void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(trans_A)
        {
        case rocsparse_operation_none:
        {
            if(temp_buffer == nullptr)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
            }

            char* ptr        = reinterpret_cast<char*>(temp_buffer);
            J*    row_limits = reinterpret_cast<J*>(ptr);

            I nblocks = (nnz - 1) / NNZ_PER_BLOCK + 1;
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::csrmmnn_nnz_split_compute_row_limits<256, NNZ_PER_BLOCK>),
                dim3((nblocks - 1) / 256 + 1),
                dim3(256),
                0,
                handle->stream,
                m,
                nblocks,
                nnz,
                csr_row_ptr,
                row_limits,
                descr->base);

            return rocsparse_status_success;
        }
        case rocsparse_operation_transpose:
        case rocsparse_operation_conjugate_transpose:
        {
            return rocsparse_status_success;
        }
        }
    }
}

#define LAUNCH_CSRMMNN_NNZ_SPLIT_MAIN_KERNEL(CSRMMNT_DIM, WF_SIZE)        \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                   \
        (rocsparse::csrmmnn_nnz_split_main_kernel<CSRMMNT_DIM, WF_SIZE>), \
        dim3(nblocks),                                                    \
        dim3(CSRMMNT_DIM),                                                \
        0,                                                                \
        handle->stream,                                                   \
        conj_A,                                                           \
        conj_B,                                                           \
        main,                                                             \
        m,                                                                \
        n,                                                                \
        k,                                                                \
        nnz,                                                              \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),     \
        row_block_red,                                                    \
        val_block_red,                                                    \
        row_limits,                                                       \
        csr_row_ptr,                                                      \
        csr_col_ind,                                                      \
        csr_val,                                                          \
        dense_B,                                                          \
        ldb,                                                              \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),      \
        dense_C,                                                          \
        ldc,                                                              \
        order_C,                                                          \
        descr->base,                                                      \
        handle->pointer_mode == rocsparse_pointer_mode_host)

#define LAUNCH_CSRMMNN_NNZ_SPLIT_REMAINDER_KERNEL(CSRMMNT_DIM, WF_SIZE)        \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                        \
        (rocsparse::csrmmnn_nnz_split_remainder_kernel<CSRMMNT_DIM, WF_SIZE>), \
        dim3(nblocks),                                                         \
        dim3(CSRMMNT_DIM),                                                     \
        0,                                                                     \
        handle->stream,                                                        \
        conj_A,                                                                \
        conj_B,                                                                \
        main,                                                                  \
        m,                                                                     \
        n,                                                                     \
        k,                                                                     \
        nnz,                                                                   \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),          \
        row_block_red,                                                         \
        val_block_red,                                                         \
        row_limits,                                                            \
        csr_row_ptr,                                                           \
        csr_col_ind,                                                           \
        csr_val,                                                               \
        dense_B,                                                               \
        ldb,                                                                   \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),           \
        dense_C,                                                               \
        ldc,                                                                   \
        order_C,                                                               \
        descr->base,                                                           \
        handle->pointer_mode == rocsparse_pointer_mode_host)

namespace rocsparse
{
    template <unsigned int BLOCKSIZE,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    static rocsparse_status csrmmnn_nnz_split_dispatch(rocsparse_handle          handle,
                                                       bool                      conj_A,
                                                       bool                      conj_B,
                                                       J                         m,
                                                       J                         n,
                                                       J                         k,
                                                       I                         nnz,
                                                       const T*                  alpha_device_host,
                                                       const rocsparse_mat_descr descr,
                                                       const A*                  csr_val,
                                                       const I*                  csr_row_ptr,
                                                       const J*                  csr_col_ind,
                                                       const B*                  dense_B,
                                                       int64_t                   ldb,
                                                       const T*                  beta_device_host,
                                                       C*                        dense_C,
                                                       int64_t                   ldc,
                                                       rocsparse_order           order_C,
                                                       void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Scale C with beta
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::scale_2d_array(handle, m, n, ldc, 1, 0, beta_device_host, dense_C, order_C));

        I nblocks = (nnz - 1) / NNZ_PER_BLOCK + 1;

        char* ptr        = reinterpret_cast<char*>(temp_buffer);
        J*    row_limits = reinterpret_cast<J*>(ptr);
        ptr += sizeof(J) * ((nblocks + 1 - 1) / 256 + 1) * 256;
        J* row_block_red = reinterpret_cast<J*>(ptr);
        ptr += sizeof(J) * ((nblocks - 1) / 256 + 1) * 256;
        T* val_block_red = reinterpret_cast<T*>(ptr);

        J main      = 0;
        J remainder = 0;

        if(n >= 8)
        {
            remainder = n % 8;
            main      = n - remainder;
            LAUNCH_CSRMMNN_NNZ_SPLIT_MAIN_KERNEL(NNZ_PER_BLOCK, 8);
        }
        else if(n >= 4)
        {
            remainder = n % 4;
            main      = n - remainder;
            LAUNCH_CSRMMNN_NNZ_SPLIT_MAIN_KERNEL(NNZ_PER_BLOCK, 4);
        }
        else if(n >= 2)
        {
            remainder = n % 2;
            main      = n - remainder;
            LAUNCH_CSRMMNN_NNZ_SPLIT_MAIN_KERNEL(NNZ_PER_BLOCK, 2);
        }
        else if(n >= 1)
        {
            remainder = n % 1;
            main      = n - remainder;
            LAUNCH_CSRMMNN_NNZ_SPLIT_MAIN_KERNEL(NNZ_PER_BLOCK, 1);
        }
        else
        {
            remainder = n;
        }

        if(remainder > 0)
        {
            if(remainder <= 1)
            {
                LAUNCH_CSRMMNN_NNZ_SPLIT_REMAINDER_KERNEL(NNZ_PER_BLOCK, 1);
            }
            else if(remainder <= 2)
            {
                LAUNCH_CSRMMNN_NNZ_SPLIT_REMAINDER_KERNEL(NNZ_PER_BLOCK, 2);
            }
            else if(remainder <= 4)
            {
                LAUNCH_CSRMMNN_NNZ_SPLIT_REMAINDER_KERNEL(NNZ_PER_BLOCK, 4);
            }
            else if(remainder <= 8)
            {
                LAUNCH_CSRMMNN_NNZ_SPLIT_REMAINDER_KERNEL(NNZ_PER_BLOCK, 8);
            }
        }

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrmmnn_general_block_reduce<1024>),
                                           dim3(n),
                                           dim3(1024),
                                           0,
                                           handle->stream,
                                           nblocks,
                                           row_block_red,
                                           val_block_red,
                                           dense_C,
                                           ldc,
                                           order_C);

        return rocsparse_status_success;
    }
}

#define LAUNCH_CSRMMNT_NNZ_SPLIT_MAIN_KERNEL(CSRMMNT_DIM, WF_SIZE, LOOPS)        \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                          \
        (rocsparse::csrmmnt_nnz_split_main_kernel<CSRMMNT_DIM, WF_SIZE, LOOPS>), \
        dim3((nnz - 1) / CSRMMNT_DIM + 1),                                       \
        dim3(CSRMMNT_DIM),                                                       \
        0,                                                                       \
        handle->stream,                                                          \
        conj_A,                                                                  \
        conj_B,                                                                  \
        main,                                                                    \
        m,                                                                       \
        n,                                                                       \
        k,                                                                       \
        nnz,                                                                     \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),            \
        row_limits,                                                              \
        csr_row_ptr,                                                             \
        csr_col_ind,                                                             \
        csr_val,                                                                 \
        dense_B,                                                                 \
        ldb,                                                                     \
        dense_C,                                                                 \
        ldc,                                                                     \
        order_C,                                                                 \
        descr->base,                                                             \
        handle->pointer_mode == rocsparse_pointer_mode_host)

#define LAUNCH_CSRMMNT_NNZ_SPLIT_REMAINDER_KERNEL(CSRMMNT_DIM, WF_SIZE)        \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                        \
        (rocsparse::csrmmnt_nnz_split_remainder_kernel<CSRMMNT_DIM, WF_SIZE>), \
        dim3((nnz - 1) / CSRMMNT_DIM + 1),                                     \
        dim3(CSRMMNT_DIM),                                                     \
        0,                                                                     \
        handle->stream,                                                        \
        conj_A,                                                                \
        conj_B,                                                                \
        main,                                                                  \
        m,                                                                     \
        n,                                                                     \
        k,                                                                     \
        nnz,                                                                   \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),          \
        row_limits,                                                            \
        csr_row_ptr,                                                           \
        csr_col_ind,                                                           \
        csr_val,                                                               \
        dense_B,                                                               \
        ldb,                                                                   \
        dense_C,                                                               \
        ldc,                                                                   \
        order_C,                                                               \
        descr->base,                                                           \
        handle->pointer_mode == rocsparse_pointer_mode_host)

namespace rocsparse
{
    template <unsigned int BLOCKSIZE,
              unsigned int WF_SIZE,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    rocsparse_status csrmmnt_nnz_split_dispatch(rocsparse_handle          handle,
                                                bool                      conj_A,
                                                bool                      conj_B,
                                                J                         m,
                                                J                         n,
                                                J                         k,
                                                I                         nnz,
                                                const T*                  alpha_device_host,
                                                const rocsparse_mat_descr descr,
                                                const A*                  csr_val,
                                                const I*                  csr_row_ptr,
                                                const J*                  csr_col_ind,
                                                const B*                  dense_B,
                                                int64_t                   ldb,
                                                const T*                  beta_device_host,
                                                C*                        dense_C,
                                                int64_t                   ldc,
                                                rocsparse_order           order_C,
                                                void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Scale C with beta
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::scale_2d_array(handle, m, n, ldc, 1, 0, beta_device_host, dense_C, order_C));

        char* ptr        = reinterpret_cast<char*>(temp_buffer);
        J*    row_limits = reinterpret_cast<J*>(ptr);

        J main      = 0;
        J remainder = n;

        if(n >= 256)
        {
            remainder = n % 256;
            main      = n - remainder;
            LAUNCH_CSRMMNT_NNZ_SPLIT_MAIN_KERNEL(BLOCKSIZE, WF_SIZE, (256 / WF_SIZE));
        }
        else if(n >= 192)
        {
            remainder = n % 192;
            main      = n - remainder;
            LAUNCH_CSRMMNT_NNZ_SPLIT_MAIN_KERNEL(BLOCKSIZE, WF_SIZE, (192 / WF_SIZE));
        }
        else if(n >= 128)
        {
            remainder = n % 128;
            main      = n - remainder;
            LAUNCH_CSRMMNT_NNZ_SPLIT_MAIN_KERNEL(BLOCKSIZE, WF_SIZE, (128 / WF_SIZE));
        }
        else if(n >= 64)
        {
            remainder = n % 64;
            main      = n - remainder;
            LAUNCH_CSRMMNT_NNZ_SPLIT_MAIN_KERNEL(BLOCKSIZE, WF_SIZE, (64 / WF_SIZE));
        }

        if(remainder > 0)
        {
            if(remainder <= 1)
            {
                LAUNCH_CSRMMNT_NNZ_SPLIT_REMAINDER_KERNEL(BLOCKSIZE, 1);
            }
            else if(remainder <= 2)
            {
                LAUNCH_CSRMMNT_NNZ_SPLIT_REMAINDER_KERNEL(BLOCKSIZE, 2);
            }
            else if(remainder <= 4)
            {
                LAUNCH_CSRMMNT_NNZ_SPLIT_REMAINDER_KERNEL(BLOCKSIZE, 4);
            }
            else if(remainder <= 8)
            {
                LAUNCH_CSRMMNT_NNZ_SPLIT_REMAINDER_KERNEL(BLOCKSIZE, 8);
            }
            else if(remainder <= 16)
            {
                LAUNCH_CSRMMNT_NNZ_SPLIT_REMAINDER_KERNEL(BLOCKSIZE, 16);
            }
            else if(remainder <= 32 || WF_SIZE == 32)
            {
                LAUNCH_CSRMMNT_NNZ_SPLIT_REMAINDER_KERNEL(BLOCKSIZE, 32);
            }
            else
            {
                LAUNCH_CSRMMNT_NNZ_SPLIT_REMAINDER_KERNEL(BLOCKSIZE, 64);
            }
        }

        return rocsparse_status_success;
    }

#define ROCSPARSE_CSRMM_TEMPLATE_NNZ_SPLIT_IMPL(NAME) \
    NAME(handle,                                      \
         conj_A,                                      \
         conj_B,                                      \
         m,                                           \
         n,                                           \
         k,                                           \
         nnz,                                         \
         alpha_device_host,                           \
         descr,                                       \
         csr_val,                                     \
         csr_row_ptr,                                 \
         csr_col_ind,                                 \
         dense_B,                                     \
         ldb,                                         \
         beta_device_host,                            \
         dense_C,                                     \
         ldc,                                         \
         order_C,                                     \
         temp_buffer);

    template <typename T, typename I, typename J, typename A, typename B, typename C>
    rocsparse_status csrmm_template_nnz_split(rocsparse_handle          handle,
                                              rocsparse_operation       trans_A,
                                              rocsparse_operation       trans_B,
                                              J                         m,
                                              J                         n,
                                              J                         k,
                                              I                         nnz,
                                              const T*                  alpha_device_host,
                                              const rocsparse_mat_descr descr,
                                              const A*                  csr_val,
                                              const I*                  csr_row_ptr,
                                              const J*                  csr_col_ind,
                                              const B*                  dense_B,
                                              int64_t                   ldb,
                                              rocsparse_order           order_B,
                                              const T*                  beta_device_host,
                                              C*                        dense_C,
                                              int64_t                   ldc,
                                              rocsparse_order           order_C,
                                              void*                     temp_buffer,
                                              bool                      force_conj_A)
    {
        ROCSPARSE_ROUTINE_TRACE;

        bool conj_A = (trans_A == rocsparse_operation_conjugate_transpose || force_conj_A);
        bool conj_B = (trans_B == rocsparse_operation_conjugate_transpose);

        // Run different csrmm kernels
        if(trans_A == rocsparse_operation_none)
        {
            if(temp_buffer == nullptr)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
            }

            if((order_B == rocsparse_order_column && trans_B == rocsparse_operation_none)
               || (order_B == rocsparse_order_row && trans_B == rocsparse_operation_transpose)
               || (order_B == rocsparse_order_row
                   && trans_B == rocsparse_operation_conjugate_transpose))
            {
                return ROCSPARSE_CSRMM_TEMPLATE_NNZ_SPLIT_IMPL(
                    (rocsparse::csrmmnn_nnz_split_dispatch<NNZ_PER_BLOCK>));
            }
            else if((order_B == rocsparse_order_column && trans_B == rocsparse_operation_transpose)
                    || (order_B == rocsparse_order_column
                        && trans_B == rocsparse_operation_conjugate_transpose)
                    || (order_B == rocsparse_order_row && trans_B == rocsparse_operation_none))
            {
                if(handle->wavefront_size == 32)
                {
                    return ROCSPARSE_CSRMM_TEMPLATE_NNZ_SPLIT_IMPL(
                        (rocsparse::csrmmnt_nnz_split_dispatch<NNZ_PER_BLOCK, 32>));
                }
                else if(handle->wavefront_size == 64)
                {
                    return ROCSPARSE_CSRMM_TEMPLATE_NNZ_SPLIT_IMPL(
                        (rocsparse::csrmmnt_nnz_split_dispatch<NNZ_PER_BLOCK, 64>));
                }
            }
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }
}

#define INSTANTIATE_BUFFER_SIZE(TTYPE, ITYPE, JTYPE, ATYPE)                           \
    template rocsparse_status rocsparse::csrmm_buffer_size_template_nnz_split<TTYPE>( \
        rocsparse_handle          handle,                                             \
        rocsparse_operation       trans_A,                                            \
        rocsparse_csrmm_alg       alg,                                                \
        JTYPE                     m,                                                  \
        JTYPE                     n,                                                  \
        JTYPE                     k,                                                  \
        ITYPE                     nnz,                                                \
        const rocsparse_mat_descr descr,                                              \
        const ATYPE*              csr_val,                                            \
        const ITYPE*              csr_row_ptr,                                        \
        const JTYPE*              csr_col_ind,                                        \
        size_t*                   buffer_size)

// Uniform precisions
INSTANTIATE_BUFFER_SIZE(float, int32_t, int32_t, float);
INSTANTIATE_BUFFER_SIZE(float, int64_t, int32_t, float);
INSTANTIATE_BUFFER_SIZE(float, int64_t, int64_t, float);
INSTANTIATE_BUFFER_SIZE(double, int32_t, int32_t, double);
INSTANTIATE_BUFFER_SIZE(double, int64_t, int32_t, double);
INSTANTIATE_BUFFER_SIZE(double, int64_t, int64_t, double);
INSTANTIATE_BUFFER_SIZE(rocsparse_float_complex, int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE_BUFFER_SIZE(rocsparse_float_complex, int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE_BUFFER_SIZE(rocsparse_float_complex, int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE_BUFFER_SIZE(rocsparse_double_complex, int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE_BUFFER_SIZE(rocsparse_double_complex, int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE_BUFFER_SIZE(rocsparse_double_complex, int64_t, int64_t, rocsparse_double_complex);

// Mixed precisions
INSTANTIATE_BUFFER_SIZE(int32_t, int32_t, int32_t, int8_t);
INSTANTIATE_BUFFER_SIZE(int32_t, int64_t, int32_t, int8_t);
INSTANTIATE_BUFFER_SIZE(int32_t, int64_t, int64_t, int8_t);
INSTANTIATE_BUFFER_SIZE(float, int32_t, int32_t, int8_t);
INSTANTIATE_BUFFER_SIZE(float, int64_t, int32_t, int8_t);
INSTANTIATE_BUFFER_SIZE(float, int64_t, int64_t, int8_t);
INSTANTIATE_BUFFER_SIZE(float, int32_t, int32_t, _Float16);
INSTANTIATE_BUFFER_SIZE(float, int64_t, int32_t, _Float16);
INSTANTIATE_BUFFER_SIZE(float, int64_t, int64_t, _Float16);
INSTANTIATE_BUFFER_SIZE(float, int32_t, int32_t, rocsparse_bfloat16);
INSTANTIATE_BUFFER_SIZE(float, int64_t, int32_t, rocsparse_bfloat16);
INSTANTIATE_BUFFER_SIZE(float, int64_t, int64_t, rocsparse_bfloat16);
#undef INSTANTIATE_BUFFER_SIZE

#define INSTANTIATE_ANALYSIS(I, J, A)                                       \
    template rocsparse_status rocsparse::csrmm_analysis_template_nnz_split( \
        rocsparse_handle          handle,                                   \
        rocsparse_operation       trans_A,                                  \
        rocsparse_csrmm_alg       alg,                                      \
        J                         m,                                        \
        J                         n,                                        \
        J                         k,                                        \
        I                         nnz,                                      \
        const rocsparse_mat_descr descr,                                    \
        const A*                  csr_val,                                  \
        const I*                  csr_row_ptr,                              \
        const J*                  csr_col_ind,                              \
        void*                     temp_buffer)

// Uniform precisions
INSTANTIATE_ANALYSIS(int32_t, int32_t, float);
INSTANTIATE_ANALYSIS(int64_t, int32_t, float);
INSTANTIATE_ANALYSIS(int64_t, int64_t, float);
INSTANTIATE_ANALYSIS(int32_t, int32_t, double);
INSTANTIATE_ANALYSIS(int64_t, int32_t, double);
INSTANTIATE_ANALYSIS(int64_t, int64_t, double);
INSTANTIATE_ANALYSIS(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE_ANALYSIS(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE_ANALYSIS(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE_ANALYSIS(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE_ANALYSIS(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE_ANALYSIS(int64_t, int64_t, rocsparse_double_complex);

// Mixed precisions
INSTANTIATE_ANALYSIS(int32_t, int32_t, _Float16);
INSTANTIATE_ANALYSIS(int64_t, int32_t, _Float16);
INSTANTIATE_ANALYSIS(int64_t, int64_t, _Float16);
INSTANTIATE_ANALYSIS(int32_t, int32_t, rocsparse_bfloat16);
INSTANTIATE_ANALYSIS(int64_t, int32_t, rocsparse_bfloat16);
INSTANTIATE_ANALYSIS(int64_t, int64_t, rocsparse_bfloat16);
INSTANTIATE_ANALYSIS(int32_t, int32_t, int8_t);
INSTANTIATE_ANALYSIS(int64_t, int32_t, int8_t);
INSTANTIATE_ANALYSIS(int64_t, int64_t, int8_t);
#undef INSTANTIATE_ANALYSIS

#define INSTANTIATE(TTYPE, ITYPE, JTYPE, ATYPE, BTYPE, CTYPE)                                        \
    template rocsparse_status rocsparse::csrmm_template_nnz_split(rocsparse_handle    handle,        \
                                                                  rocsparse_operation trans_A,       \
                                                                  rocsparse_operation trans_B,       \
                                                                  JTYPE               m,             \
                                                                  JTYPE               n,             \
                                                                  JTYPE               k,             \
                                                                  ITYPE               nnz,           \
                                                                  const TTYPE* alpha_device_host,    \
                                                                  const rocsparse_mat_descr descr,   \
                                                                  const ATYPE*              csr_val, \
                                                                  const ITYPE*    csr_row_ptr,       \
                                                                  const JTYPE*    csr_col_ind,       \
                                                                  const BTYPE*    dense_B,           \
                                                                  int64_t         ldb,               \
                                                                  rocsparse_order order_B,           \
                                                                  const TTYPE*    beta_device_host,  \
                                                                  CTYPE*          dense_C,           \
                                                                  int64_t         ldc,               \
                                                                  rocsparse_order order_C,           \
                                                                  void*           temp_buffer,       \
                                                                  bool            force_conj_A)

// Uniform precisions
INSTANTIATE(float, int32_t, int32_t, float, float, float);
INSTANTIATE(float, int64_t, int32_t, float, float, float);
INSTANTIATE(float, int64_t, int64_t, float, float, float);
INSTANTIATE(double, int32_t, int32_t, double, double, double);
INSTANTIATE(double, int64_t, int32_t, double, double, double);
INSTANTIATE(double, int64_t, int64_t, double, double, double);
INSTANTIATE(rocsparse_float_complex,
            int32_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int64_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex,
            int32_t,
            int32_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int32_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int64_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);

// Mixed Precisions
INSTANTIATE(float, int32_t, int32_t, _Float16, _Float16, float);
INSTANTIATE(float, int64_t, int32_t, _Float16, _Float16, float);
INSTANTIATE(float, int64_t, int64_t, _Float16, _Float16, float);
INSTANTIATE(float, int32_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(float, int64_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(float, int64_t, int64_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(int32_t, int32_t, int32_t, int8_t, int8_t, int32_t);
INSTANTIATE(int32_t, int64_t, int32_t, int8_t, int8_t, int32_t);
INSTANTIATE(int32_t, int64_t, int64_t, int8_t, int8_t, int32_t);
INSTANTIATE(float, int32_t, int32_t, int8_t, int8_t, float);
INSTANTIATE(float, int64_t, int32_t, int8_t, int8_t, float);
INSTANTIATE(float, int64_t, int64_t, int8_t, int8_t, float);

#undef INSTANTIATE

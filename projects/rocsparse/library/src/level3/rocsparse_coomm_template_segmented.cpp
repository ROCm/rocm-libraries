/*! \file */
/* ************************************************************************
* Copyright (C) 2021-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

#include "coomm/segmented/kernel_declarations.h"
#include "coomm_device_segmented.h"

namespace rocsparse
{
    // RDNA (wave32) launch tuning for the COO SpMM segmented algorithm.
    //
    // The segmented main/remainder kernels (COOMMN_DIM = 256) and the final
    // block reduction (1024 threads) were sized for wave64 hardware. On wave32
    // parts (e.g. gfx1201 / RDNA4) each of those blocks maps to twice as many
    // wavefronts and holds twice the shared memory per element, which caps the
    // occupancy of the memory-bound segmented kernel. For large problems we drop
    // the segmented block to 128 threads (and the final reduction to 256); small
    // problems are launch-overhead bound and keep the original wave64 sizes.
    //
    // The gate depends only on the wavefront size and nnz (both available when
    // sizing the temporary buffer and when launching), so buffer sizing and the
    // kernel launch always agree on the block dimension. Numerics are unchanged:
    // the segmented / block reduction is exact for any power-of-two block size.
    static constexpr int64_t coomm_segmented_wave32_nnz_threshold = 262144;

    static inline uint32_t coomm_segmented_block_dim(rocsparse_handle handle, int64_t nnz)
    {
        if(handle->wavefront_size == 32 && nnz >= coomm_segmented_wave32_nnz_threshold)
        {
            return 128;
        }
        return 256;
    }

    template <typename T, typename I, typename A>
    rocsparse_status coomm_buffer_size_template_segmented(rocsparse_handle          handle,
                                                          rocsparse_operation       trans_A,
                                                          I                         m,
                                                          I                         n,
                                                          I                         k,
                                                          int64_t                   nnz,
                                                          I                         batch_count,
                                                          const rocsparse_mat_descr descr,
                                                          const A*                  coo_val,
                                                          const I*                  coo_row_ind,
                                                          const I*                  coo_col_ind,
                                                          size_t*                   buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

#define LOOPS 4
        // Block dimension must match the launch decision in coomm_template_segmented
        // so that nblocks (and thus the reduction buffer sizes) agree.
        const int64_t coommn_dim = rocsparse::coomm_segmented_block_dim(handle, nnz);
        const I       nblocks    = (nnz - 1) / (coommn_dim * LOOPS) + 1;
        // Reduction buffers are padded to a fixed 256-byte granularity.
        *buffer_size = size_t(256) + ((sizeof(I) * nblocks * batch_count - 1) / 256 + 1) * 256
                       + ((sizeof(T) * nblocks * n * batch_count - 1) / 256 + 1) * 256;
#undef LOOPS

        return rocsparse_status_success;
    }

#define LAUNCH_COOMMNN_SEGMENTED_MAIN_KERNEL(COOMMNN_DIM, WF_SIZE, LOOPS, TRANSB)        \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                                  \
        (rocsparse::coommnn_segmented_main_kernel<COOMMNN_DIM, WF_SIZE, LOOPS, TRANSB>), \
        dim3(nblocks, (main - 1) / WF_SIZE + 1, batch_count_C),                          \
        dim3(COOMMNN_DIM),                                                               \
        0,                                                                               \
        stream,                                                                          \
        conj_A,                                                                          \
        conj_B,                                                                          \
        m,                                                                               \
        n,                                                                               \
        k,                                                                               \
        nnz,                                                                             \
        batch_stride_A,                                                                  \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),                    \
        row_block_red,                                                                   \
        val_block_red,                                                                   \
        coo_row_ind,                                                                     \
        coo_col_ind,                                                                     \
        coo_val,                                                                         \
        dense_B,                                                                         \
        ldb,                                                                             \
        batch_stride_B,                                                                  \
        dense_C,                                                                         \
        ldc,                                                                             \
        batch_stride_C,                                                                  \
        order_C,                                                                         \
        descr->base,                                                                     \
        handle->pointer_mode == rocsparse_pointer_mode_host)

#define LAUNCH_COOMMNN_SEGMENTED_REMAINDER_KERNEL(COOMMNN_DIM, WF_SIZE, LOOPS, TRANSB)        \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                                       \
        (rocsparse::coommnn_segmented_remainder_kernel<COOMMNN_DIM, WF_SIZE, LOOPS, TRANSB>), \
        dim3(nblocks, 1, batch_count_C),                                                      \
        dim3(COOMMNN_DIM),                                                                    \
        0,                                                                                    \
        stream,                                                                               \
        conj_A,                                                                               \
        conj_B,                                                                               \
        main,                                                                                 \
        m,                                                                                    \
        n,                                                                                    \
        k,                                                                                    \
        nnz,                                                                                  \
        batch_stride_A,                                                                       \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),                         \
        row_block_red,                                                                        \
        val_block_red,                                                                        \
        coo_row_ind,                                                                          \
        coo_col_ind,                                                                          \
        coo_val,                                                                              \
        dense_B,                                                                              \
        ldb,                                                                                  \
        batch_stride_B,                                                                       \
        dense_C,                                                                              \
        ldc,                                                                                  \
        batch_stride_C,                                                                       \
        order_C,                                                                              \
        descr->base,                                                                          \
        handle->pointer_mode == rocsparse_pointer_mode_host)

// Full main + remainder launch sequence for one B orientation, parameterized by
// the segmented block dimension (COOMMNN_DIM) so that the same code drives both
// the wave64 (256) and the RDNA/wave32 (128) tuned instantiations.
#define LAUNCH_COOMMNN_SEGMENTED(COOMMNN_DIM, TRANSB)                              \
    {                                                                             \
        I main      = 0;                                                          \
        I remainder = 0;                                                          \
        if(n >= 8)                                                                \
        {                                                                         \
            remainder = n % 8;                                                    \
            main      = n - remainder;                                            \
            LAUNCH_COOMMNN_SEGMENTED_MAIN_KERNEL(COOMMNN_DIM, 8, LOOPS, TRANSB);  \
        }                                                                         \
        else if(n >= 4)                                                           \
        {                                                                         \
            remainder = n % 4;                                                    \
            main      = n - remainder;                                            \
            LAUNCH_COOMMNN_SEGMENTED_MAIN_KERNEL(COOMMNN_DIM, 4, LOOPS, TRANSB);  \
        }                                                                         \
        else if(n >= 2)                                                           \
        {                                                                         \
            remainder = n % 2;                                                    \
            main      = n - remainder;                                            \
            LAUNCH_COOMMNN_SEGMENTED_MAIN_KERNEL(COOMMNN_DIM, 2, LOOPS, TRANSB);  \
        }                                                                         \
        else if(n >= 1)                                                           \
        {                                                                         \
            remainder = n % 1;                                                    \
            main      = n - remainder;                                            \
            LAUNCH_COOMMNN_SEGMENTED_MAIN_KERNEL(COOMMNN_DIM, 1, LOOPS, TRANSB);  \
        }                                                                         \
        else                                                                      \
        {                                                                         \
            remainder = n;                                                        \
        }                                                                         \
        if(remainder > 0)                                                         \
        {                                                                         \
            if(remainder <= 1)                                                    \
            {                                                                     \
                LAUNCH_COOMMNN_SEGMENTED_REMAINDER_KERNEL(COOMMNN_DIM, 1, LOOPS, TRANSB); \
            }                                                                     \
            else if(remainder <= 2)                                               \
            {                                                                     \
                LAUNCH_COOMMNN_SEGMENTED_REMAINDER_KERNEL(COOMMNN_DIM, 2, LOOPS, TRANSB); \
            }                                                                     \
            else if(remainder <= 4)                                               \
            {                                                                     \
                LAUNCH_COOMMNN_SEGMENTED_REMAINDER_KERNEL(COOMMNN_DIM, 4, LOOPS, TRANSB); \
            }                                                                     \
            else if(remainder <= 8)                                               \
            {                                                                     \
                LAUNCH_COOMMNN_SEGMENTED_REMAINDER_KERNEL(COOMMNN_DIM, 8, LOOPS, TRANSB); \
            }                                                                     \
        }                                                                         \
    }

    template <typename T, typename I, typename A, typename B, typename C>
    rocsparse_status coomm_template_segmented(rocsparse_handle          handle,
                                              rocsparse_operation       trans_A,
                                              rocsparse_operation       trans_B,
                                              I                         m,
                                              I                         n,
                                              I                         k,
                                              int64_t                   nnz,
                                              I                         batch_count_A,
                                              int64_t                   batch_stride_A,
                                              const T*                  alpha_device_host,
                                              const rocsparse_mat_descr descr,
                                              const A*                  coo_val,
                                              const I*                  coo_row_ind,
                                              const I*                  coo_col_ind,
                                              const B*                  dense_B,
                                              int64_t                   ldb,
                                              I                         batch_count_B,
                                              int64_t                   batch_stride_B,
                                              rocsparse_order           order_B,
                                              const T*                  beta_device_host,
                                              C*                        dense_C,
                                              int64_t                   ldc,
                                              I                         batch_count_C,
                                              int64_t                   batch_stride_C,
                                              rocsparse_order           order_C,
                                              void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        const bool conj_A = (trans_A == rocsparse_operation_conjugate_transpose);
        const bool conj_B = (trans_B == rocsparse_operation_conjugate_transpose);

        // Stream
        hipStream_t stream = handle->stream;

        // Run different coomm kernels
        if(trans_A == rocsparse_operation_none)
        {
#define LOOPS 4
            // RDNA/wave32 tuning: drop the segmented block from 256 -> 128 threads
            // for large problems; small problems keep the wave64-era 256 block.
            const int64_t coommn_dim = rocsparse::coomm_segmented_block_dim(handle, nnz);
            const I       nblocks    = (nnz - 1) / (coommn_dim * LOOPS) + 1;

            // row and val block reduction buffer (padded to a fixed 256 granularity,
            // matching coomm_buffer_size_template_segmented).
            char* ptr = reinterpret_cast<char*>(temp_buffer);
            ptr += 256;
            I* row_block_red = reinterpret_cast<I*>(reinterpret_cast<void*>(ptr));
            ptr += ((sizeof(I) * nblocks * batch_count_C - 1) / 256 + 1) * 256;
            T* val_block_red = reinterpret_cast<T*>(reinterpret_cast<void*>(ptr));

            RETURN_IF_HIP_ERROR(rocsparse_hipMemsetAsync(
                row_block_red,
                0XFF,
                ((sizeof(I) * nblocks * batch_count_C - 1) / 256 + 1) * 256,
                stream));

            if((order_B == rocsparse_order_column && trans_B == rocsparse_operation_none)
               || (order_B == rocsparse_order_row && trans_B == rocsparse_operation_transpose)
               || (order_B == rocsparse_order_row
                   && trans_B == rocsparse_operation_conjugate_transpose))
            {
                if(coommn_dim == 128)
                {
                    LAUNCH_COOMMNN_SEGMENTED(128, false);
                }
                else
                {
                    LAUNCH_COOMMNN_SEGMENTED(256, false);
                }
            }
            else if((order_B == rocsparse_order_column && trans_B == rocsparse_operation_transpose)
                    || (order_B == rocsparse_order_column
                        && trans_B == rocsparse_operation_conjugate_transpose)
                    || (order_B == rocsparse_order_row && trans_B == rocsparse_operation_none))
            {
                if(coommn_dim == 128)
                {
                    LAUNCH_COOMMNN_SEGMENTED(128, true);
                }
                else
                {
                    LAUNCH_COOMMNN_SEGMENTED(256, true);
                }
            }
#undef LOOPS

            // RDNA/wave32 tuning: the final segmented reduction over nblocks was
            // sized at 1024 threads for wave64; use a 256-thread block on wave32
            // large problems (exact for any power-of-two block size).
            if(coommn_dim == 128)
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::coommnn_general_block_reduce<256>),
                                                   dim3(n, 1, batch_count_C),
                                                   256,
                                                   0,
                                                   stream,
                                                   n,
                                                   nblocks,
                                                   row_block_red,
                                                   val_block_red,
                                                   dense_C,
                                                   ldc,
                                                   batch_stride_C,
                                                   order_C);
            }
            else
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::coommnn_general_block_reduce<1024>),
                                                   dim3(n, 1, batch_count_C),
                                                   1024,
                                                   0,
                                                   stream,
                                                   n,
                                                   nblocks,
                                                   row_block_red,
                                                   val_block_red,
                                                   dense_C,
                                                   ldc,
                                                   batch_stride_C,
                                                   order_C);
            }
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        return rocsparse_status_success;
    }
}

#define INSTANTIATE_BUFFER_SIZE(TTYPE, ITYPE, ATYPE)                                  \
    template rocsparse_status rocsparse::coomm_buffer_size_template_segmented<TTYPE>( \
        rocsparse_handle          handle,                                             \
        rocsparse_operation       trans_A,                                            \
        ITYPE                     m,                                                  \
        ITYPE                     n,                                                  \
        ITYPE                     k,                                                  \
        int64_t                   nnz,                                                \
        ITYPE                     batch_count,                                        \
        const rocsparse_mat_descr descr,                                              \
        const ATYPE*              coo_val,                                            \
        const ITYPE*              coo_row_ind,                                        \
        const ITYPE*              coo_col_ind,                                        \
        size_t*                   buffer_size);

// Uniform precisions
INSTANTIATE_BUFFER_SIZE(float, int32_t, float);
INSTANTIATE_BUFFER_SIZE(float, int64_t, float);
INSTANTIATE_BUFFER_SIZE(double, int32_t, double);
INSTANTIATE_BUFFER_SIZE(double, int64_t, double);
INSTANTIATE_BUFFER_SIZE(rocsparse_float_complex, int32_t, rocsparse_float_complex);
INSTANTIATE_BUFFER_SIZE(rocsparse_float_complex, int64_t, rocsparse_float_complex);
INSTANTIATE_BUFFER_SIZE(rocsparse_double_complex, int32_t, rocsparse_double_complex);
INSTANTIATE_BUFFER_SIZE(rocsparse_double_complex, int64_t, rocsparse_double_complex);

// Mixed precisions
INSTANTIATE_BUFFER_SIZE(float, int32_t, _Float16);
INSTANTIATE_BUFFER_SIZE(float, int64_t, _Float16);
INSTANTIATE_BUFFER_SIZE(float, int32_t, rocsparse_bfloat16);
INSTANTIATE_BUFFER_SIZE(float, int64_t, rocsparse_bfloat16);
INSTANTIATE_BUFFER_SIZE(int32_t, int32_t, int8_t);
INSTANTIATE_BUFFER_SIZE(int32_t, int64_t, int8_t);
INSTANTIATE_BUFFER_SIZE(float, int32_t, int8_t);
INSTANTIATE_BUFFER_SIZE(float, int64_t, int8_t);
#undef INSTANTIATE_BUFFER_SIZE

#define INSTANTIATE(TTYPE, ITYPE, ATYPE, BTYPE, CTYPE)                                               \
    template rocsparse_status rocsparse::coomm_template_segmented(rocsparse_handle    handle,        \
                                                                  rocsparse_operation trans_A,       \
                                                                  rocsparse_operation trans_B,       \
                                                                  ITYPE               m,             \
                                                                  ITYPE               n,             \
                                                                  ITYPE               k,             \
                                                                  int64_t             nnz,           \
                                                                  ITYPE               batch_count_A, \
                                                                  int64_t      batch_stride_A,       \
                                                                  const TTYPE* alpha_device_host,    \
                                                                  const rocsparse_mat_descr descr,   \
                                                                  const ATYPE*              coo_val, \
                                                                  const ITYPE*    coo_row_ind,       \
                                                                  const ITYPE*    coo_col_ind,       \
                                                                  const BTYPE*    dense_B,           \
                                                                  int64_t         ldb,               \
                                                                  ITYPE           batch_count_B,     \
                                                                  int64_t         batch_stride_B,    \
                                                                  rocsparse_order order_B,           \
                                                                  const TTYPE*    beta_device_host,  \
                                                                  CTYPE*          dense_C,           \
                                                                  int64_t         ldc,               \
                                                                  ITYPE           batch_count_C,     \
                                                                  int64_t         batch_stride_C,    \
                                                                  rocsparse_order order_C,           \
                                                                  void*           temp_buffer);

// Uniform precisions
INSTANTIATE(float, int32_t, float, float, float);
INSTANTIATE(float, int64_t, float, float, float);
INSTANTIATE(double, int32_t, double, double, double);
INSTANTIATE(double, int64_t, double, double, double);
INSTANTIATE(rocsparse_float_complex,
            int32_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex,
            int32_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);

// Mixed Precisions
INSTANTIATE(float, int32_t, _Float16, _Float16, float);
INSTANTIATE(float, int64_t, _Float16, _Float16, float);
INSTANTIATE(float, int32_t, _Float16, _Float16, _Float16);
INSTANTIATE(float, int64_t, _Float16, _Float16, _Float16);
INSTANTIATE(float, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(float, int64_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(float, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, rocsparse_bfloat16);
INSTANTIATE(float, int64_t, rocsparse_bfloat16, rocsparse_bfloat16, rocsparse_bfloat16);
INSTANTIATE(int32_t, int32_t, int8_t, int8_t, int32_t);
INSTANTIATE(int32_t, int64_t, int8_t, int8_t, int32_t);
INSTANTIATE(float, int32_t, int8_t, int8_t, float);
INSTANTIATE(float, int64_t, int8_t, int8_t, float);

#undef INSTANTIATE

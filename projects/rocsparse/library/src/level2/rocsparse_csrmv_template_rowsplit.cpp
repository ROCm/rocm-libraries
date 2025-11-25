/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_common.h"
#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_csrmv.hpp"
#include "rocsparse_utility.hpp"

#include "csrmv_device.h"
#include "csrmv_symm_device.h"

namespace rocsparse
{
#define LAUNCH_CSRMVN_GENERAL(wfsize)                                 \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                               \
        (csrmvn_general_kernel<CSRMVN_DIM, wfsize>),                  \
        dim3(nblocks),                                                \
        dim3(CSRMVN_DIM),                                             \
        0,                                                            \
        stream,                                                       \
        conj,                                                         \
        m,                                                            \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host), \
        csr_row_ptr_begin,                                            \
        csr_row_ptr_end,                                              \
        csr_col_ind,                                                  \
        csr_val,                                                      \
        x,                                                            \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),  \
        y,                                                            \
        descr->base,                                                  \
        handle->pointer_mode == rocsparse_pointer_mode_host)

#define LAUNCH_CSRMVT(wfsize)                                         \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                               \
        (csrmvt_general_kernel<CSRMVT_DIM, wfsize>),                  \
        dim3(csrmvt_blocks),                                          \
        dim3(csrmvt_threads),                                         \
        0,                                                            \
        stream,                                                       \
        skip_diag,                                                    \
        conj,                                                         \
        m,                                                            \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host), \
        csr_row_ptr_begin,                                            \
        csr_row_ptr_end,                                              \
        csr_col_ind,                                                  \
        csr_val,                                                      \
        x,                                                            \
        y,                                                            \
        descr->base,                                                  \
        handle->pointer_mode == rocsparse_pointer_mode_host)

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              typename I,
              typename J,
              typename A,
              typename X,
              typename Y,
              typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrmvn_general_kernel(bool conj,
                               J    m,
                               ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                               const I* csr_row_ptr_begin,
                               const I* csr_row_ptr_end,
                               const J* __restrict__ csr_col_ind,
                               const A* __restrict__ csr_val,
                               const X* __restrict__ x,
                               ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                               Y* __restrict__ y,
                               rocsparse_index_base idx_base,
                               bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);
        if(alpha != 0 || beta != 1)
        {
            rocsparse::csrmvn_general_device<BLOCKSIZE, WF_SIZE>(conj,
                                                                 m,
                                                                 alpha,
                                                                 csr_row_ptr_begin,
                                                                 csr_row_ptr_end,
                                                                 csr_col_ind,
                                                                 csr_val,
                                                                 x,
                                                                 beta,
                                                                 y,
                                                                 idx_base);
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              typename I,
              typename J,
              typename A,
              typename X,
              typename Y,
              typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrmvt_general_kernel(bool skip_diag,
                               bool conj,
                               J    m,
                               ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                               const I* csr_row_ptr_begin,
                               const I* csr_row_ptr_end,
                               const J* __restrict__ csr_col_ind,
                               const A* __restrict__ csr_val,
                               const X* __restrict__ x,
                               Y* __restrict__ y,
                               rocsparse_index_base idx_base,
                               bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        if(alpha != 0)
        {
            rocsparse::csrmvt_general_device<BLOCKSIZE, WF_SIZE>(skip_diag,
                                                                 conj,
                                                                 m,
                                                                 alpha,
                                                                 csr_row_ptr_begin,
                                                                 csr_row_ptr_end,
                                                                 csr_col_ind,
                                                                 csr_val,
                                                                 x,
                                                                 y,
                                                                 idx_base);
        }
    }
}

template <typename T, typename I, typename J, typename A, typename X, typename Y>
rocsparse_status rocsparse::csrmv_rowsplit_template_dispatch(rocsparse_handle    handle,
                                                             rocsparse_operation trans,
                                                             J                   m,
                                                             J                   n,
                                                             I                   nnz,
                                                             const T*            alpha_device_host,
                                                             const rocsparse_mat_descr descr,
                                                             const A*                  csr_val,
                                                             const I* csr_row_ptr_begin,
                                                             const I* csr_row_ptr_end,
                                                             const J* csr_col_ind,
                                                             const X* x,
                                                             const T* beta_device_host,
                                                             Y*       y,
                                                             bool     force_conj)
{
    ROCSPARSE_ROUTINE_TRACE;

    bool conj = (trans == rocsparse_operation_conjugate_transpose || force_conj);

    // Stream
    hipStream_t stream = handle->stream;

    if(descr->type == rocsparse_matrix_type_hermitian)
    {
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        // LCOV_EXCL_STOP
    }

    // Average nnz per row
    J nnz_per_row = nnz / m;

    if(trans == rocsparse_operation_none || descr->type == rocsparse_matrix_type_symmetric)
    {
#define CSRMVN_DIM 256
        int wfsize = 0;
        if(nnz_per_row < 4)
        {
            wfsize = 2;
        }
        else if(nnz_per_row < 8)
        {
            wfsize = 4;
        }
        else if(nnz_per_row < 16)
        {
            wfsize = 8;
        }
        else if(nnz_per_row < 32)
        {
            wfsize = 16;
        }
        else if(nnz_per_row < 64 || handle->wavefront_size == 32)
        {
            wfsize = 32;
        }
        else
        {
            wfsize = 64;
        }

        J nblocks = std::min((m - 1) / (CSRMVN_DIM / wfsize) + 1, (J)2147483647);

        int maxthreads = handle->properties.maxThreadsPerBlock;
        int nprocs     = 2 * handle->properties.multiProcessorCount;
        int minblocks  = (nprocs * maxthreads - 1) / CSRMVN_DIM + 1;

        if(nblocks < minblocks)
        {
            J threads_per_row = CSRMVN_DIM * minblocks / m;

            if(threads_per_row >= 64)
            {
                wfsize = 64;
            }
            else if(threads_per_row >= 32)
            {
                wfsize = 32;
            }
            else if(threads_per_row >= 16)
            {
                wfsize = 16;
            }
            else if(threads_per_row >= 8)
            {
                wfsize = 8;
            }
            else if(threads_per_row >= 4)
            {
                wfsize = 4;
            }
            else
            {
                wfsize = 2;
            }

            wfsize = std::min(wfsize, handle->wavefront_size);
        }

        nblocks = std::min((m - 1) / (CSRMVN_DIM / wfsize) + 1, (J)2147483647);

        if(handle->wavefront_size == 32)
        {
            if(nblocks > 20 * minblocks)
            {
                nblocks = std::max((nblocks - 1) / wfsize + 1, (J)minblocks);
            }
        }

        if(wfsize == 2)
        {
            LAUNCH_CSRMVN_GENERAL(2);
        }
        else if(wfsize == 4)
        {
            LAUNCH_CSRMVN_GENERAL(4);
        }
        else if(wfsize == 8)
        {
            LAUNCH_CSRMVN_GENERAL(8);
        }
        else if(wfsize == 16)
        {
            LAUNCH_CSRMVN_GENERAL(16);
        }
        else if(wfsize == 32 || handle->wavefront_size == 32)
        {
            LAUNCH_CSRMVN_GENERAL(32);
        }
        else
        {
            LAUNCH_CSRMVN_GENERAL(64);
        }
#undef CSRMVN_DIM
    }

    if(trans != rocsparse_operation_none || descr->type == rocsparse_matrix_type_symmetric)
    {
#define CSRMVT_DIM 256
        if(descr->type != rocsparse_matrix_type_symmetric)
        {
            // Scale y with beta
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::scale_array(handle, n, beta_device_host, y));
        }

        bool skip_diag = (descr->type == rocsparse_matrix_type_symmetric);

        rocsparse_int max_blocks = 1024;
        rocsparse_int min_blocks = (m - 1) / CSRMVT_DIM + 1;

        dim3 csrmvt_blocks(rocsparse::min(min_blocks, max_blocks));
        dim3 csrmvt_threads(CSRMVT_DIM);

        if(nnz_per_row < 4)
        {
            LAUNCH_CSRMVT(4);
        }
        else if(nnz_per_row < 8)
        {
            LAUNCH_CSRMVT(8);
        }
        else if(nnz_per_row < 16)
        {
            LAUNCH_CSRMVT(16);
        }
        else if(nnz_per_row < 32 || handle->wavefront_size == 32)
        {
            LAUNCH_CSRMVT(32);
        }
        else
        {
            LAUNCH_CSRMVT(64);
        }
#undef CSRMVT_DIM
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(TTYPE, ITYPE, JTYPE, ATYPE, XTYPE, YTYPE)                     \
    template rocsparse_status rocsparse::csrmv_rowsplit_template_dispatch<TTYPE>( \
        rocsparse_handle          handle,                                         \
        rocsparse_operation       trans,                                          \
        JTYPE                     m,                                              \
        JTYPE                     n,                                              \
        ITYPE                     nnz,                                            \
        const TTYPE*              alpha_device_host,                              \
        const rocsparse_mat_descr descr,                                          \
        const ATYPE*              csr_val,                                        \
        const ITYPE*              csr_row_ptr_begin,                              \
        const ITYPE*              csr_row_ptr_end,                                \
        const JTYPE*              csr_col_ind,                                    \
        const XTYPE*              x,                                              \
        const TTYPE*              beta_device_host,                               \
        YTYPE*                    y,                                              \
        bool                      force_conj);

// Uniform precision
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

// Mixed percision
INSTANTIATE(int32_t, int32_t, int32_t, int8_t, int8_t, int32_t);
INSTANTIATE(int32_t, int64_t, int32_t, int8_t, int8_t, int32_t);
INSTANTIATE(int32_t, int64_t, int64_t, int8_t, int8_t, int32_t);
INSTANTIATE(float, int32_t, int32_t, int8_t, int8_t, float);
INSTANTIATE(float, int64_t, int32_t, int8_t, int8_t, float);
INSTANTIATE(float, int64_t, int64_t, int8_t, int8_t, float);
INSTANTIATE(float, int32_t, int32_t, _Float16, _Float16, float);
INSTANTIATE(float, int64_t, int32_t, _Float16, _Float16, float);
INSTANTIATE(float, int64_t, int64_t, _Float16, _Float16, float);
INSTANTIATE(float, int32_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(float, int64_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(float, int64_t, int64_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(rocsparse_float_complex,
            int32_t,
            int32_t,
            float,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int32_t,
            float,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int64_t,
            float,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(double, int32_t, int32_t, float, double, double);
INSTANTIATE(double, int64_t, int32_t, float, double, double);
INSTANTIATE(double, int64_t, int64_t, float, double, double);
INSTANTIATE(rocsparse_double_complex,
            int32_t,
            int32_t,
            double,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int32_t,
            double,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int64_t,
            double,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int32_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int64_t,
            rocsparse_float_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);

#undef INSTANTIATE

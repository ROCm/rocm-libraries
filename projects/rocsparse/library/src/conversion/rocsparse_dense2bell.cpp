/*! \file */
/* ************************************************************************
* Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_dense2bell.hpp"

#include "dense2bell.h"

#include <vector>

template <typename T>
rocsparse_status rocsparse::dense2bell_nnz_template(rocsparse_handle          handle, //0
                                                    rocsparse_order           order, //1
                                                    int64_t                   m, //2
                                                    int64_t                   n, //3
                                                    const rocsparse_mat_descr descr, //4
                                                    const T*                  A, //5
                                                    int64_t                   ld, //6
                                                    int64_t                   ell_block_size, //7
                                                    int64_t*                  nnzb_per_row, //8
                                                    int64_t*                  ell_cols) //9
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xdense2bell"),
                         order,
                         m,
                         n,
                         descr,
                         (const void*&)A,
                         ld,
                         ell_block_size,
                         (void*&)nnzb_per_row,
                         (void*&)ell_cols);

    // Stream
    hipStream_t stream = handle->stream;

    const int64_t mb = (m + ell_block_size - 1) / ell_block_size;
    const int64_t nb = (n + ell_block_size - 1) / ell_block_size;

    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::dense2bell_nnz_kernel<256>),
                                       dim3(mb),
                                       dim3(256),
                                       0,
                                       stream,
                                       m,
                                       n,
                                       A,
                                       ld,
                                       order,
                                       ell_block_size,
                                       nnzb_per_row);

    // Reduce nnzb_per_row (device, size mb) down to a single maximum using two
    // kernels so that no reduction is performed on the host.
    constexpr uint32_t REDUCE_BLOCKSIZE = 256;

    int64_t* workspace = nullptr;
    RETURN_IF_HIP_ERROR(
        rocsparse_hipMallocAsync(&workspace, sizeof(int64_t) * REDUCE_BLOCKSIZE, stream));

    // Stage 1: each of the REDUCE_BLOCKSIZE blocks reduces a grid-strided slice of
    // nnzb_per_row into workspace[blockIdx].
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
        (rocsparse::dense2bell_max_nnzb_part1_kernel<REDUCE_BLOCKSIZE>),
        dim3(REDUCE_BLOCKSIZE),
        dim3(REDUCE_BLOCKSIZE),
        0,
        stream,
        mb,
        nnzb_per_row,
        workspace);

    // Stage 2: a single block reduces the workspace array into workspace[0].
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
        (rocsparse::dense2bell_max_nnzb_part2_kernel<REDUCE_BLOCKSIZE>),
        dim3(1),
        dim3(REDUCE_BLOCKSIZE),
        0,
        stream,
        workspace);

    int64_t max_nnzb_per_row = 0;
    RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(
        &max_nnzb_per_row, workspace, sizeof(int64_t), hipMemcpyDeviceToHost, stream));
    RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(stream));

    RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(workspace, stream));

    // ell_cols is expressed in columns, i.e. the block width scaled by the block size.
    *ell_cols = max_nnzb_per_row * ell_block_size;

    return rocsparse_status_success;
}

template <typename I, typename T>
rocsparse_status rocsparse::dense2bell_template(rocsparse_handle          handle, //0
                                                rocsparse_order           order, //1
                                                int64_t                   m, //2
                                                int64_t                   n, //3
                                                const rocsparse_mat_descr descr, //4
                                                const T*                  A, //5
                                                int64_t                   ld, //6
                                                int64_t                   ell_block_size, //7
                                                int64_t                   ell_cols, //8
                                                T*                        bell_val, //9
                                                I*                        bell_col_ind) //10
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xdense2bell"),
                         order,
                         m,
                         n,
                         descr,
                         (const void*&)A,
                         ld,
                         ell_block_size,
                         ell_cols,
                         (void*&)bell_val,
                         (void*&)bell_col_ind);

    // Stream
    hipStream_t stream = handle->stream;

    const int64_t mb              = (m + ell_block_size - 1) / ell_block_size;
    const int64_t ell_block_width = ell_cols / ell_block_size;

    if(mb == 0 || ell_block_width == 0)
    {
        return rocsparse_status_success;
    }

    // Zero the value array so that padded ELL slots (and structural zeros that the
    // fill kernel skips at the matrix boundary) are well defined.
    RETURN_IF_HIP_ERROR(rocsparse_hipMemsetAsync(bell_val, 0, sizeof(T) * m * ell_cols, stream));

    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::dense2bell_fill_kernel<256>),
                                       dim3(mb),
                                       dim3(256),
                                       0,
                                       stream,
                                       m,
                                       n,
                                       A,
                                       ld,
                                       order,
                                       ell_block_size,
                                       ell_block_width,
                                       descr->base,
                                       bell_val,
                                       bell_col_ind);

    return rocsparse_status_success;
}

#define INSTANTIATE(TTYPE)                                               \
    template rocsparse_status rocsparse::dense2bell_nnz_template<TTYPE>( \
        rocsparse_handle          handle,                                \
        rocsparse_order           order,                                 \
        int64_t                   m,                                     \
        int64_t                   n,                                     \
        const rocsparse_mat_descr descr,                                 \
        const TTYPE*              A,                                     \
        int64_t                   ld,                                    \
        int64_t                   ell_block_size,                        \
        int64_t*                  nnzb_per_row,                          \
        int64_t*                  ell_cols);

INSTANTIATE(_Float16);
INSTANTIATE(rocsparse_bfloat16);
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
#undef INSTANTIATE

#define INSTANTIATE(ITYPE, TTYPE)                                           \
    template rocsparse_status rocsparse::dense2bell_template<ITYPE, TTYPE>( \
        rocsparse_handle          handle,                                   \
        rocsparse_order           order,                                    \
        int64_t                   m,                                        \
        int64_t                   n,                                        \
        const rocsparse_mat_descr descr,                                    \
        const TTYPE*              A,                                        \
        int64_t                   ld,                                       \
        int64_t                   ell_block_size,                           \
        int64_t                   ell_cols,                                 \
        TTYPE*                    bell_val,                                 \
        ITYPE*                    bell_col_ind);

INSTANTIATE(int32_t, _Float16);
INSTANTIATE(int32_t, rocsparse_bfloat16);
INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, _Float16);
INSTANTIATE(int64_t, rocsparse_bfloat16);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
#undef INSTANTIATE

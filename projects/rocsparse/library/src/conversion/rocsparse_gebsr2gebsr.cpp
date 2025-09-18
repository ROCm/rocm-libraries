/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/conversion/rocsparse_gebsr2gebsr.h"
#include "internal/conversion/rocsparse_coo2csr.h"
#include "internal/conversion/rocsparse_csr2gebsr.h"
#include "rocsparse_control.hpp"
#include "rocsparse_gebsr2gebsr.hpp"
#include "rocsparse_utility.hpp"

#include "gebsr2csr_device.h"
#include "gebsr2gebsr_device.h"
#include "rocsparse_common.hpp"
#include "rocsparse_csr2gebsr.hpp"
#include "rocsparse_gebsr2csr.hpp"

#include "rocsparse_common.h"
#include "rocsparse_primitives.hpp"

#define launch_gebsr2gebsr_fast_kernel(T, direction, block_size, segment_size)     \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                            \
        (rocsparse::gebsr2gebsr_fast_kernel<direction, block_size, segment_size>), \
        grid_size,                                                                 \
        block_size,                                                                \
        0,                                                                         \
        stream,                                                                    \
        mb,                                                                        \
        nb,                                                                        \
        descr_A->base,                                                             \
        bsr_val_A,                                                                 \
        bsr_row_ptr_A,                                                             \
        bsr_col_ind_A,                                                             \
        row_block_dim_A,                                                           \
        col_block_dim_A,                                                           \
        mb_c,                                                                      \
        nb_c,                                                                      \
        descr_C->base,                                                             \
        bsr_val_C,                                                                 \
        bsr_row_ptr_C,                                                             \
        bsr_col_ind_C,                                                             \
        row_block_dim_C,                                                           \
        col_block_dim_C);

template <typename T>
rocsparse_status rocsparse::gebsr2gebsr_buffer_size_template(rocsparse_handle          handle, //0
                                                             rocsparse_direction       dir, //1
                                                             rocsparse_int             mb, //2
                                                             rocsparse_int             nb, //3
                                                             rocsparse_int             nnzb, //4
                                                             const rocsparse_mat_descr descr_A, //5
                                                             const T*             bsr_val_A, //6
                                                             const rocsparse_int* bsr_row_ptr_A, //7
                                                             const rocsparse_int* bsr_col_ind_A, //8
                                                             rocsparse_int row_block_dim_A, //9
                                                             rocsparse_int col_block_dim_A, //10
                                                             rocsparse_int row_block_dim_C, //11
                                                             rocsparse_int col_block_dim_C, //12
                                                             size_t*       buffer_size) //13
{
    ROCSPARSE_ROUTINE_TRACE;

    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xgebsr2csr_buffer_size"),
                         mb,
                         nb,
                         nnzb,
                         descr_A,
                         (const void*&)bsr_val_A,
                         (const void*&)bsr_row_ptr_A,
                         (const void*&)bsr_col_ind_A,
                         row_block_dim_A,
                         col_block_dim_A,
                         row_block_dim_C,
                         col_block_dim_C,
                         (const void*&)buffer_size);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, dir);
    ROCSPARSE_CHECKARG_SIZE(2, mb);
    ROCSPARSE_CHECKARG_SIZE(3, nb);
    ROCSPARSE_CHECKARG_SIZE(4, nnzb);

    ROCSPARSE_CHECKARG_SIZE(9, row_block_dim_A);
    ROCSPARSE_CHECKARG(9, row_block_dim_A, (row_block_dim_A == 0), rocsparse_status_invalid_size);

    ROCSPARSE_CHECKARG_SIZE(10, col_block_dim_A);
    ROCSPARSE_CHECKARG(10, col_block_dim_A, (col_block_dim_A == 0), rocsparse_status_invalid_size);

    ROCSPARSE_CHECKARG_SIZE(11, row_block_dim_C);
    ROCSPARSE_CHECKARG(11, row_block_dim_C, (row_block_dim_C == 0), rocsparse_status_invalid_size);

    ROCSPARSE_CHECKARG_SIZE(12, col_block_dim_C);
    ROCSPARSE_CHECKARG(12, col_block_dim_C, (col_block_dim_C == 0), rocsparse_status_invalid_size);

    ROCSPARSE_CHECKARG_POINTER(5, descr_A);
    ROCSPARSE_CHECKARG(5,
                       descr_A,
                       (descr_A->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);
    ROCSPARSE_CHECKARG(5,
                       descr_A,
                       (rocsparse_matrix_type_general != descr_A->type),
                       rocsparse_status_not_implemented);

    ROCSPARSE_CHECKARG_ARRAY(6, nnzb, bsr_val_A);
    ROCSPARSE_CHECKARG_ARRAY(7, mb, bsr_row_ptr_A);
    ROCSPARSE_CHECKARG_ARRAY(8, nnzb, bsr_col_ind_A);

    ROCSPARSE_CHECKARG_POINTER(13, buffer_size);

    // Quick return if possible
    if(mb == 0 || nb == 0)
    {
        if(row_block_dim_C <= 32)
        {
            *buffer_size = 0;
        }
        else
        {
            // Perform the conversion gebsr->gebsr by performing gebsr->csr->gebsr.
            const rocsparse_int m = mb * row_block_dim_A;

            *buffer_size = sizeof(rocsparse_int) * (m + 1)
                           + sizeof(rocsparse_int) * row_block_dim_A * col_block_dim_A * nnzb
                           + sizeof(T) * row_block_dim_A * col_block_dim_A * nnzb;
        }

        return rocsparse_status_success;
    }

    if(row_block_dim_C <= 32)
    {
        *buffer_size = 0;
    }
    else
    {
        // Perform the conversion gebsr->gebsr by performing gebsr->csr->gebsr.
        rocsparse_int m = mb * row_block_dim_A;

        *buffer_size = sizeof(rocsparse_int) * (m + 1)
                       + sizeof(rocsparse_int) * row_block_dim_A * col_block_dim_A * nnzb
                       + sizeof(T) * row_block_dim_A * col_block_dim_A * nnzb;
    }

    return rocsparse_status_success;
}

namespace rocsparse
{
    static rocsparse_status gebsr2gebsr_quickreturn(rocsparse_handle          handle,
                                                    rocsparse_direction       dir,
                                                    rocsparse_int             mb,
                                                    rocsparse_int             nb,
                                                    rocsparse_int             nnzb,
                                                    const rocsparse_mat_descr descr_A,
                                                    const void*               bsr_val_A,
                                                    const void*               bsr_row_ptr_A,
                                                    const void*               bsr_col_ind_A,
                                                    rocsparse_int             row_block_dim_A,
                                                    rocsparse_int             col_block_dim_A,
                                                    const rocsparse_mat_descr descr_C,
                                                    void*                     bsr_val_C,
                                                    void*                     bsr_row_ptr_C,
                                                    void*                     bsr_col_ind_C,
                                                    rocsparse_int             row_block_dim_C,
                                                    rocsparse_int             col_block_dim_C,
                                                    void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(mb == 0 || nb == 0)
        {
            return rocsparse_status_success;
        }
        return rocsparse_status_continue;
    }
}

template <typename T>
rocsparse_status rocsparse::gebsr2gebsr_template(rocsparse_handle          handle, //0
                                                 rocsparse_direction       dir, //1
                                                 rocsparse_int             mb, //2
                                                 rocsparse_int             nb, //3
                                                 rocsparse_int             nnzb, //4
                                                 const rocsparse_mat_descr descr_A, //5
                                                 const T*                  bsr_val_A, //6
                                                 const rocsparse_int*      bsr_row_ptr_A, //7
                                                 const rocsparse_int*      bsr_col_ind_A, //8
                                                 rocsparse_int             row_block_dim_A, //9
                                                 rocsparse_int             col_block_dim_A, //10
                                                 const rocsparse_mat_descr descr_C, //11
                                                 T*                        bsr_val_C, //12
                                                 rocsparse_int*            bsr_row_ptr_C, //13
                                                 rocsparse_int*            bsr_col_ind_C, //14
                                                 rocsparse_int             row_block_dim_C, //15
                                                 rocsparse_int             col_block_dim_C, //16
                                                 void*                     temp_buffer) //17
{
    ROCSPARSE_ROUTINE_TRACE;

    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xgebsr2gebsr"),
                         mb,
                         nb,
                         nnzb,
                         descr_A,
                         (const void*&)bsr_val_A,
                         (const void*&)bsr_row_ptr_A,
                         (const void*&)bsr_col_ind_A,
                         row_block_dim_A,
                         col_block_dim_A,
                         descr_C,
                         (const void*&)bsr_val_C,
                         (const void*&)bsr_row_ptr_C,
                         (const void*&)bsr_col_ind_C,
                         row_block_dim_C,
                         col_block_dim_C,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    ROCSPARSE_CHECKARG_ENUM(1, dir);
    ROCSPARSE_CHECKARG_SIZE(2, mb);
    ROCSPARSE_CHECKARG_SIZE(3, nb);

    ROCSPARSE_CHECKARG_SIZE(4, nnzb);
    ROCSPARSE_CHECKARG_SIZE(9, row_block_dim_A);
    ROCSPARSE_CHECKARG(9, row_block_dim_A, (row_block_dim_A == 0), rocsparse_status_invalid_size);

    ROCSPARSE_CHECKARG_SIZE(10, col_block_dim_A);
    ROCSPARSE_CHECKARG(10, col_block_dim_A, (col_block_dim_A == 0), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_SIZE(15, row_block_dim_C);
    ROCSPARSE_CHECKARG(15, row_block_dim_C, (row_block_dim_C == 0), rocsparse_status_invalid_size);

    ROCSPARSE_CHECKARG_SIZE(16, col_block_dim_C);
    ROCSPARSE_CHECKARG(16, col_block_dim_C, (col_block_dim_C == 0), rocsparse_status_invalid_size);

    const rocsparse_status status = rocsparse::gebsr2gebsr_quickreturn(handle,
                                                                       dir,
                                                                       mb,
                                                                       nb,
                                                                       nnzb,
                                                                       descr_A,
                                                                       bsr_val_A,
                                                                       bsr_row_ptr_A,
                                                                       bsr_col_ind_A,
                                                                       row_block_dim_A,
                                                                       col_block_dim_A,
                                                                       descr_C,
                                                                       bsr_val_C,
                                                                       bsr_row_ptr_C,
                                                                       bsr_col_ind_C,
                                                                       row_block_dim_C,
                                                                       col_block_dim_C,
                                                                       temp_buffer);

    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG_POINTER(5, descr_A);
    ROCSPARSE_CHECKARG(5,
                       descr_A,
                       (descr_A->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    ROCSPARSE_CHECKARG(5,
                       descr_A,
                       (rocsparse_matrix_type_general != descr_A->type),
                       rocsparse_status_not_implemented);

    ROCSPARSE_CHECKARG_ARRAY(6, nnzb, bsr_val_A);
    ROCSPARSE_CHECKARG_ARRAY(7, mb, bsr_row_ptr_A);
    ROCSPARSE_CHECKARG_ARRAY(8, nnzb, bsr_col_ind_A);

    ROCSPARSE_CHECKARG_POINTER(11, descr_C);
    ROCSPARSE_CHECKARG(11,
                       descr_C,
                       (descr_C->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);
    ROCSPARSE_CHECKARG(11,
                       descr_C,
                       (rocsparse_matrix_type_general != descr_C->type),
                       rocsparse_status_not_implemented);

    ROCSPARSE_CHECKARG_ARRAY(13, mb, bsr_row_ptr_C);

    ROCSPARSE_CHECKARG(17,
                       temp_buffer,
                       (row_block_dim_C > 32 && temp_buffer == nullptr),
                       rocsparse_status_invalid_pointer);

    const rocsparse_int m    = mb * row_block_dim_A;
    const rocsparse_int n    = nb * col_block_dim_A;
    const rocsparse_int mb_c = (m + row_block_dim_C - 1) / row_block_dim_C;
    const rocsparse_int nb_c = (n + col_block_dim_C - 1) / col_block_dim_C;

    rocsparse_int start  = 0;
    rocsparse_int end    = 0;
    rocsparse_int nnzb_C = 0;
    if(bsr_row_ptr_C != nullptr)
    {
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(&end,
                                           &bsr_row_ptr_C[mb_c],
                                           sizeof(rocsparse_int),
                                           hipMemcpyDeviceToHost,
                                           handle->stream));
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(&start,
                                           &bsr_row_ptr_C[0],
                                           sizeof(rocsparse_int),
                                           hipMemcpyDeviceToHost,
                                           handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
    }
    nnzb_C = end - start;
    ROCSPARSE_CHECKARG_ARRAY(12, nnzb_C, bsr_val_C);
    ROCSPARSE_CHECKARG_ARRAY(14, nnzb_C, bsr_col_ind_C);

    RETURN_IF_HIP_ERROR(hipMemsetAsync(
        bsr_val_C, 0, sizeof(T) * nnzb_C * row_block_dim_C * col_block_dim_C, handle->stream));

    // Stream
    hipStream_t stream = handle->stream;

    // Common case where BSR block dimension is small
    if(row_block_dim_C <= 32)
    {
        // A 64 thread wavefront is decomposed as:
        //      |    bank 0        bank 1       bank 2         bank 3
        // row 0|  0  1  2  3 |  4  5  6  7 |  8  9 10 11 | 12 13 14 15 |
        // row 1| 16 17 18 19 | 20 21 22 23 | 24 25 26 27 | 28 29 30 31 |
        // row 2| 32 33 34 35 | 36 37 38 39 | 40 41 42 43 | 44 45 46 47 |
        // row 3| 48 49 50 51 | 52 53 54 55 | 56 57 58 59 | 60 61 62 63 |
        //
        // Segments can be of size 4 (quarter row), 8 (half row), 16 (full row),
        // 32 (half wavefront), or 64 (full wavefront). We assign one segment per
        // BSR block row where the segment size matches the block dimension as
        // closely as possible while still being greater than or equal to the
        // block dimension.

        // Note: block_size must be less than or equal to wavefront size
        rocsparse_int block_size   = row_block_dim_C > 16 ? 32 : 16;
        rocsparse_int segment_size = row_block_dim_C == 1 ? 2 : row_block_dim_C;

        // round segment_size up to next power of 2
        segment_size--;
        segment_size |= segment_size >> 1;
        segment_size |= segment_size >> 2;
        segment_size |= segment_size >> 4;
        segment_size |= segment_size >> 8;
        segment_size |= segment_size >> 16;
        segment_size++;

        const rocsparse_int segments_per_block = block_size / segment_size;
        const rocsparse_int grid_size = (mb_c + segments_per_block - 1) / segments_per_block;

        if(dir == rocsparse_direction_row)
        {
            if(row_block_dim_C <= 2)
            {
                launch_gebsr2gebsr_fast_kernel(T, rocsparse_direction_row, 16, 2);
            }
            else if(row_block_dim_C <= 4)
            {
                launch_gebsr2gebsr_fast_kernel(T, rocsparse_direction_row, 16, 4);
            }
            else if(row_block_dim_C <= 8)
            {
                launch_gebsr2gebsr_fast_kernel(T, rocsparse_direction_row, 16, 8);
            }
            else if(row_block_dim_C <= 16)
            {
                launch_gebsr2gebsr_fast_kernel(T, rocsparse_direction_row, 16, 16);
            }
            else
            {
                // (row_block_dim_C <= 32)
                launch_gebsr2gebsr_fast_kernel(T, rocsparse_direction_row, 32, 32);
            }
        }
        else
        {
            if(row_block_dim_C <= 2)
            {
                launch_gebsr2gebsr_fast_kernel(T, rocsparse_direction_column, 16, 2);
            }
            else if(row_block_dim_C <= 4)
            {
                launch_gebsr2gebsr_fast_kernel(T, rocsparse_direction_column, 16, 4);
            }
            else if(row_block_dim_C <= 8)
            {
                launch_gebsr2gebsr_fast_kernel(T, rocsparse_direction_column, 16, 8);
            }
            else if(row_block_dim_C <= 16)
            {
                launch_gebsr2gebsr_fast_kernel(T, rocsparse_direction_column, 16, 16);
            }
            else
            {
                //  (row_block_dim_C <= 32)
                launch_gebsr2gebsr_fast_kernel(T, rocsparse_direction_column, 32, 32);
            }
        }
    }
    else
    {

        rocsparse_int* ptr = reinterpret_cast<rocsparse_int*>(temp_buffer);

        rocsparse_int* csr_row_ptr = reinterpret_cast<rocsparse_int*>(ptr);
        ptr += (m + 1);
        rocsparse_int* csr_col_ind = reinterpret_cast<rocsparse_int*>(ptr);
        ptr += size_t(nnzb) * col_block_dim_A * row_block_dim_A;
        T* csr_val = reinterpret_cast<T*>(ptr);
        ptr += size_t(nnzb) * row_block_dim_A * col_block_dim_A;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gebsr2csr_template(handle,
                                                                dir,
                                                                mb,
                                                                nb,
                                                                descr_A,
                                                                bsr_val_A,
                                                                bsr_row_ptr_A,
                                                                bsr_col_ind_A,
                                                                row_block_dim_A,
                                                                col_block_dim_A,
                                                                descr_C,
                                                                csr_val,
                                                                csr_row_ptr,
                                                                csr_col_ind));

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2gebsr_template(handle,
                                                                dir,
                                                                m,
                                                                n,
                                                                descr_C,
                                                                csr_val,
                                                                csr_row_ptr,
                                                                csr_col_ind,
                                                                descr_C,
                                                                bsr_val_C,
                                                                bsr_row_ptr_C,
                                                                bsr_col_ind_C,
                                                                row_block_dim_C,
                                                                col_block_dim_C,
                                                                ptr));
    }

    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

#define launch_gebsr2gebsr_nnz_fast_kernel(block_size, segment_size)        \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                     \
        (rocsparse::gebsr2gebsr_nnz_fast_kernel<block_size, segment_size>), \
        dim3(grid_size),                                                    \
        dim3(block_size),                                                   \
        0,                                                                  \
        handle->stream,                                                     \
        mb,                                                                 \
        nb,                                                                 \
        descr_A->base,                                                      \
        bsr_row_ptr_A,                                                      \
        bsr_col_ind_A,                                                      \
        row_block_dim_A,                                                    \
        col_block_dim_A,                                                    \
        mb_c,                                                               \
        nb_c,                                                               \
        descr_C->base,                                                      \
        bsr_row_ptr_C,                                                      \
        row_block_dim_C,                                                    \
        col_block_dim_C);

// Performs gebsr2csr conversion without computing the values. Used in rocsparse_gebsr2gebsr_nnz()
extern "C" rocsparse_status rocsparse_gebsr2csr_nnz(rocsparse_handle          handle, //0
                                                    rocsparse_direction       direction, //1
                                                    rocsparse_int             mb, //2
                                                    rocsparse_int             nb, //3
                                                    const rocsparse_mat_descr bsr_descr, //4
                                                    const rocsparse_int*      bsr_row_ptr, //5
                                                    const rocsparse_int*      bsr_col_ind, //6
                                                    rocsparse_int             row_block_dim, //7
                                                    rocsparse_int             col_block_dim, //8
                                                    const rocsparse_mat_descr csr_descr, //9
                                                    rocsparse_int*            csr_row_ptr, //10
                                                    rocsparse_int*            csr_col_ind) //11
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check for valid handle
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, direction);
    ROCSPARSE_CHECKARG_SIZE(2, mb);
    ROCSPARSE_CHECKARG_SIZE(3, nb);

    ROCSPARSE_CHECKARG_POINTER(4, bsr_descr);
    ROCSPARSE_CHECKARG(4,
                       bsr_descr,
                       (bsr_descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);
    ROCSPARSE_CHECKARG(4,
                       bsr_descr,
                       (rocsparse_matrix_type_general != bsr_descr->type),
                       rocsparse_status_not_implemented);

    ROCSPARSE_CHECKARG_SIZE(7, row_block_dim);
    ROCSPARSE_CHECKARG(7, row_block_dim, (row_block_dim == 0), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_SIZE(8, col_block_dim);
    ROCSPARSE_CHECKARG(8, col_block_dim, (col_block_dim == 0), rocsparse_status_invalid_size);

    ROCSPARSE_CHECKARG_POINTER(9, csr_descr);
    ROCSPARSE_CHECKARG(9,
                       csr_descr,
                       (csr_descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);
    ROCSPARSE_CHECKARG(9,
                       csr_descr,
                       (rocsparse_matrix_type_general != csr_descr->type),
                       rocsparse_status_not_implemented);

    // Quick return if possible
    if(mb == 0 || nb == 0)
    {
        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG_ARRAY(5, mb, bsr_row_ptr);
    ROCSPARSE_CHECKARG_POINTER(10, csr_row_ptr);
    ROCSPARSE_CHECKARG_POINTER(11, csr_col_ind);

    if(bsr_col_ind == nullptr)
    {
        rocsparse_int start = 0;
        rocsparse_int end   = 0;

        RETURN_IF_HIP_ERROR(hipMemcpyAsync(&end,
                                           &bsr_row_ptr[mb * row_block_dim],
                                           sizeof(rocsparse_int),
                                           hipMemcpyDeviceToHost,
                                           handle->stream));
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &start, &bsr_row_ptr[0], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

        const rocsparse_int nnzb = (end - start);
        ROCSPARSE_CHECKARG_ARRAY(6, nnzb, bsr_col_ind);
    }

    // Stream
    hipStream_t stream = handle->stream;

    constexpr rocsparse_int block_size     = 256;
    rocsparse_int           wavefront_size = handle->wavefront_size;
    rocsparse_int           grid_size      = mb * row_block_dim / (block_size / wavefront_size);
    if(mb * row_block_dim % (block_size / wavefront_size) != 0)
    {
        grid_size++;
    }

    dim3 blocks(grid_size);
    dim3 threads(block_size);

    if(wavefront_size == 32)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gebsr2csr_nnz_kernel<block_size, 32>),
                                           blocks,
                                           threads,
                                           0,
                                           stream,
                                           mb,
                                           nb,
                                           bsr_descr->base,
                                           bsr_row_ptr,
                                           bsr_col_ind,
                                           row_block_dim,
                                           col_block_dim,
                                           csr_descr->base,
                                           csr_row_ptr,
                                           csr_col_ind);
    }
    else
    {
        rocsparse_host_assert(wavefront_size == 64, "Wrong wavefront size dispatch.");
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gebsr2csr_nnz_kernel<block_size, 64>),
                                           blocks,
                                           threads,
                                           0,
                                           stream,
                                           mb,
                                           nb,
                                           bsr_descr->base,
                                           bsr_row_ptr,
                                           bsr_col_ind,
                                           row_block_dim,
                                           col_block_dim,
                                           csr_descr->base,
                                           csr_row_ptr,
                                           csr_col_ind);
    }

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_gebsr2gebsr_nnz(rocsparse_handle          handle, //0
                                                      rocsparse_direction       dir, //1
                                                      rocsparse_int             mb, //2
                                                      rocsparse_int             nb, //3
                                                      rocsparse_int             nnzb, //4
                                                      const rocsparse_mat_descr descr_A, //5
                                                      const rocsparse_int*      bsr_row_ptr_A, //6
                                                      const rocsparse_int*      bsr_col_ind_A, //7
                                                      rocsparse_int             row_block_dim_A, //8
                                                      rocsparse_int             col_block_dim_A, //9
                                                      const rocsparse_mat_descr descr_C, //10
                                                      rocsparse_int*            bsr_row_ptr_C, //11
                                                      rocsparse_int  row_block_dim_C, //12
                                                      rocsparse_int  col_block_dim_C, //13
                                                      rocsparse_int* nnz_total_dev_host_ptr, //14
                                                      void*          temp_buffer) //15
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_gebsr2gebsr_nnz",
                         mb,
                         nb,
                         nnzb,
                         descr_A,
                         (const void*&)bsr_row_ptr_A,
                         (const void*&)bsr_col_ind_A,
                         row_block_dim_A,
                         col_block_dim_A,
                         descr_C,
                         (const void*&)bsr_row_ptr_C,
                         row_block_dim_C,
                         col_block_dim_C,
                         (const void*&)nnz_total_dev_host_ptr,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, dir);
    ROCSPARSE_CHECKARG_SIZE(2, mb);
    ROCSPARSE_CHECKARG_SIZE(3, nb);
    ROCSPARSE_CHECKARG_SIZE(4, nnzb);
    ROCSPARSE_CHECKARG_POINTER(5, descr_A);

    ROCSPARSE_CHECKARG(5,
                       descr_A,
                       (rocsparse_matrix_type_general != descr_A->type),
                       rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(5,
                       descr_A,
                       (descr_A->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    ROCSPARSE_CHECKARG_ARRAY(6, mb, bsr_row_ptr_A);
    ROCSPARSE_CHECKARG_ARRAY(7, nnzb, bsr_col_ind_A);

    ROCSPARSE_CHECKARG_SIZE(8, row_block_dim_A);
    ROCSPARSE_CHECKARG(8, row_block_dim_A, (row_block_dim_A == 0), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_SIZE(9, col_block_dim_A);
    ROCSPARSE_CHECKARG(9, col_block_dim_A, (col_block_dim_A == 0), rocsparse_status_invalid_size);

    ROCSPARSE_CHECKARG_POINTER(10, descr_C);
    ROCSPARSE_CHECKARG(10,
                       descr_C,
                       (rocsparse_matrix_type_general != descr_C->type),
                       rocsparse_status_not_implemented);

    ROCSPARSE_CHECKARG(10,
                       descr_C,
                       (descr_C->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    ROCSPARSE_CHECKARG_ARRAY(11, mb, bsr_row_ptr_C);

    ROCSPARSE_CHECKARG_SIZE(12, row_block_dim_C);
    ROCSPARSE_CHECKARG(12, row_block_dim_C, (row_block_dim_C == 0), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_SIZE(13, col_block_dim_C);
    ROCSPARSE_CHECKARG(13, col_block_dim_C, (col_block_dim_C == 0), rocsparse_status_invalid_size);

    ROCSPARSE_CHECKARG_POINTER(14, nnz_total_dev_host_ptr);

    ROCSPARSE_CHECKARG(15,
                       temp_buffer,
                       (row_block_dim_C > 32 && temp_buffer == nullptr),
                       rocsparse_status_invalid_pointer);

    // Stream
    hipStream_t stream = handle->stream;

    const rocsparse_int m    = mb * row_block_dim_A;
    const rocsparse_int n    = nb * col_block_dim_A;
    const rocsparse_int mb_c = (m + row_block_dim_C - 1) / row_block_dim_C;
    const rocsparse_int nb_c = (n + col_block_dim_C - 1) / col_block_dim_C;

    // Quick return if possible
    if(mb == 0 || nb == 0 || nnzb == 0)
    {
        if(nullptr != nnz_total_dev_host_ptr)
        {
            if(bsr_row_ptr_C != nullptr)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::valset(
                    handle, mb_c + 1, static_cast<rocsparse_int>(descr_C->base), bsr_row_ptr_C));
            }

            rocsparse_pointer_mode mode;
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_get_pointer_mode(handle, &mode));

            if(rocsparse_pointer_mode_device == mode)
            {
                RETURN_IF_HIP_ERROR(
                    hipMemsetAsync(nnz_total_dev_host_ptr, 0, sizeof(rocsparse_int), stream));
            }
            else
            {
                *nnz_total_dev_host_ptr = 0;
            }
        }

        return rocsparse_status_success;
    }

    // Common case where BSR block dimension is small
    if(row_block_dim_C <= 32)
    {
        // A 64 thread wavefront is decomposed as:
        //      |    bank 0        bank 1       bank 2         bank 3
        // row 0|  0  1  2  3 |  4  5  6  7 |  8  9 10 11 | 12 13 14 15 |
        // row 1| 16 17 18 19 | 20 21 22 23 | 24 25 26 27 | 28 29 30 31 |
        // row 2| 32 33 34 35 | 36 37 38 39 | 40 41 42 43 | 44 45 46 47 |
        // row 3| 48 49 50 51 | 52 53 54 55 | 56 57 58 59 | 60 61 62 63 |
        //
        // Segments can be of size 4 (quarter row), 8 (half row), 16 (full row),
        // 32 (half wavefront), or 64 (full wavefront). We assign one segment per
        // BSR block row of matrix C where the segment size matches the row block
        // dimension (row_block_dim_C) as closely as possible while still being
        // greater than or equal to the row block dimension.
        // Note: block_size must be less than or equal to wavefront size
        rocsparse_int block_size   = row_block_dim_C > 16 ? 32 : 16;
        rocsparse_int segment_size = row_block_dim_C == 1 ? 2 : row_block_dim_C;

        // round segment_size up to next power of 2
        segment_size--;
        segment_size |= segment_size >> 1;
        segment_size |= segment_size >> 2;
        segment_size |= segment_size >> 4;
        segment_size |= segment_size >> 8;
        segment_size |= segment_size >> 16;
        segment_size++;

        rocsparse_int segments_per_block = block_size / segment_size;
        rocsparse_int grid_size          = (mb_c + segments_per_block - 1) / segments_per_block;

        if(row_block_dim_C <= 2)
        {
            launch_gebsr2gebsr_nnz_fast_kernel(16, 2);
        }
        else if(row_block_dim_C <= 4)
        {
            launch_gebsr2gebsr_nnz_fast_kernel(16, 4);
        }
        else if(row_block_dim_C <= 8)
        {
            launch_gebsr2gebsr_nnz_fast_kernel(16, 8);
        }
        else if(row_block_dim_C <= 16)
        {
            launch_gebsr2gebsr_nnz_fast_kernel(16, 16);
        }
        else
        {
            // (row_block_dim_C <= 32)
            launch_gebsr2gebsr_nnz_fast_kernel(32, 32);
        }

        // Perform inclusive scan on bsr row pointer array
        size_t temp_storage_size_bytes;
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::primitives::inclusive_scan_buffer_size<rocsparse_int, rocsparse_int>(
                handle, mb_c + 1, &temp_storage_size_bytes)));

        bool  temp_alloc       = false;
        void* temp_storage_ptr = nullptr;
        if(handle->buffer_size >= temp_storage_size_bytes)
        {
            temp_storage_ptr = handle->buffer;
            temp_alloc       = false;
        }
        else
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(
                &temp_storage_ptr, temp_storage_size_bytes, handle->stream));
            temp_alloc = true;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::inclusive_scan(handle,
                                                                        bsr_row_ptr_C,
                                                                        bsr_row_ptr_C,
                                                                        mb_c + 1,
                                                                        temp_storage_size_bytes,
                                                                        temp_storage_ptr));

        if(temp_alloc)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(temp_storage_ptr, handle->stream));
        }

        // Compute nnz_total_dev_host_ptr
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(rocsparse::gebsr2gebsr_compute_nnz_total_kernel<1>,
                                               dim3(1),
                                               dim3(1),
                                               0,
                                               handle->stream,
                                               mb_c,
                                               bsr_row_ptr_C,
                                               nnz_total_dev_host_ptr);
        }
        else
        {
            rocsparse_int hstart = 0;
            rocsparse_int hend   = 0;
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&hend,
                                               &bsr_row_ptr_C[mb_c],
                                               sizeof(rocsparse_int),
                                               hipMemcpyDeviceToHost,
                                               handle->stream));
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&hstart,
                                               &bsr_row_ptr_C[0],
                                               sizeof(rocsparse_int),
                                               hipMemcpyDeviceToHost,
                                               handle->stream));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

            *nnz_total_dev_host_ptr = hend - hstart;
        }
    }
    else
    {
        rocsparse_int* ptr = reinterpret_cast<rocsparse_int*>(temp_buffer);

        rocsparse_int* csr_row_ptr = reinterpret_cast<rocsparse_int*>(ptr);
        ptr += (m + 1);
        rocsparse_int* csr_col_ind = reinterpret_cast<rocsparse_int*>(ptr);

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_gebsr2csr_nnz(handle,
                                                          dir,
                                                          mb,
                                                          nb,
                                                          descr_A,
                                                          bsr_row_ptr_A,
                                                          bsr_col_ind_A,
                                                          row_block_dim_A,
                                                          col_block_dim_A,
                                                          descr_C,
                                                          csr_row_ptr,
                                                          csr_col_ind));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_csr2gebsr_nnz(handle,
                                                          dir,
                                                          m,
                                                          n,
                                                          descr_C,
                                                          csr_row_ptr,
                                                          csr_col_ind,
                                                          descr_C,
                                                          bsr_row_ptr_C,
                                                          row_block_dim_C,
                                                          col_block_dim_C,
                                                          nnz_total_dev_host_ptr,
                                                          ptr));
    }

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_sgebsr2gebsr_buffer_size(rocsparse_handle          handle,
                                                               rocsparse_direction       dir,
                                                               rocsparse_int             mb,
                                                               rocsparse_int             nb,
                                                               rocsparse_int             nnzb,
                                                               const rocsparse_mat_descr descr_A,
                                                               const float*              bsr_val_A,
                                                               const rocsparse_int* bsr_row_ptr_A,
                                                               const rocsparse_int* bsr_col_ind_A,
                                                               rocsparse_int        row_block_dim_A,
                                                               rocsparse_int        col_block_dim_A,
                                                               rocsparse_int        row_block_dim_C,
                                                               rocsparse_int        col_block_dim_C,
                                                               size_t*              buffer_size)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gebsr2gebsr_buffer_size_template(handle,
                                                                          dir,
                                                                          mb,
                                                                          nb,
                                                                          nnzb,
                                                                          descr_A,
                                                                          bsr_val_A,
                                                                          bsr_row_ptr_A,
                                                                          bsr_col_ind_A,
                                                                          row_block_dim_A,
                                                                          col_block_dim_A,
                                                                          row_block_dim_C,
                                                                          col_block_dim_C,
                                                                          buffer_size));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_dgebsr2gebsr_buffer_size(rocsparse_handle          handle,
                                                               rocsparse_direction       dir,
                                                               rocsparse_int             mb,
                                                               rocsparse_int             nb,
                                                               rocsparse_int             nnzb,
                                                               const rocsparse_mat_descr descr_A,
                                                               const double*             bsr_val_A,
                                                               const rocsparse_int* bsr_row_ptr_A,
                                                               const rocsparse_int* bsr_col_ind_A,
                                                               rocsparse_int        row_block_dim_A,
                                                               rocsparse_int        col_block_dim_A,
                                                               rocsparse_int        row_block_dim_C,
                                                               rocsparse_int        col_block_dim_C,
                                                               size_t*              buffer_size)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gebsr2gebsr_buffer_size_template(handle,
                                                                          dir,
                                                                          mb,
                                                                          nb,
                                                                          nnzb,
                                                                          descr_A,
                                                                          bsr_val_A,
                                                                          bsr_row_ptr_A,
                                                                          bsr_col_ind_A,
                                                                          row_block_dim_A,
                                                                          col_block_dim_A,
                                                                          row_block_dim_C,
                                                                          col_block_dim_C,
                                                                          buffer_size));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status
    rocsparse_cgebsr2gebsr_buffer_size(rocsparse_handle               handle,
                                       rocsparse_direction            dir,
                                       rocsparse_int                  mb,
                                       rocsparse_int                  nb,
                                       rocsparse_int                  nnzb,
                                       const rocsparse_mat_descr      descr_A,
                                       const rocsparse_float_complex* bsr_val_A,
                                       const rocsparse_int*           bsr_row_ptr_A,
                                       const rocsparse_int*           bsr_col_ind_A,
                                       rocsparse_int                  row_block_dim_A,
                                       rocsparse_int                  col_block_dim_A,
                                       rocsparse_int                  row_block_dim_C,
                                       rocsparse_int                  col_block_dim_C,
                                       size_t*                        buffer_size)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gebsr2gebsr_buffer_size_template(handle,
                                                                          dir,
                                                                          mb,
                                                                          nb,
                                                                          nnzb,
                                                                          descr_A,
                                                                          bsr_val_A,
                                                                          bsr_row_ptr_A,
                                                                          bsr_col_ind_A,
                                                                          row_block_dim_A,
                                                                          col_block_dim_A,
                                                                          row_block_dim_C,
                                                                          col_block_dim_C,
                                                                          buffer_size));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status
    rocsparse_zgebsr2gebsr_buffer_size(rocsparse_handle                handle,
                                       rocsparse_direction             dir,
                                       rocsparse_int                   mb,
                                       rocsparse_int                   nb,
                                       rocsparse_int                   nnzb,
                                       const rocsparse_mat_descr       descr_A,
                                       const rocsparse_double_complex* bsr_val_A,
                                       const rocsparse_int*            bsr_row_ptr_A,
                                       const rocsparse_int*            bsr_col_ind_A,
                                       rocsparse_int                   row_block_dim_A,
                                       rocsparse_int                   col_block_dim_A,
                                       rocsparse_int                   row_block_dim_C,
                                       rocsparse_int                   col_block_dim_C,
                                       size_t*                         buffer_size)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gebsr2gebsr_buffer_size_template(handle,
                                                                          dir,
                                                                          mb,
                                                                          nb,
                                                                          nnzb,
                                                                          descr_A,
                                                                          bsr_val_A,
                                                                          bsr_row_ptr_A,
                                                                          bsr_col_ind_A,
                                                                          row_block_dim_A,
                                                                          col_block_dim_A,
                                                                          row_block_dim_C,
                                                                          col_block_dim_C,
                                                                          buffer_size));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_sgebsr2gebsr(rocsparse_handle          handle,
                                                   rocsparse_direction       dir,
                                                   rocsparse_int             mb,
                                                   rocsparse_int             nb,
                                                   rocsparse_int             nnzb,
                                                   const rocsparse_mat_descr descr_A,
                                                   const float*              bsr_val_A,
                                                   const rocsparse_int*      bsr_row_ptr_A,
                                                   const rocsparse_int*      bsr_col_ind_A,
                                                   rocsparse_int             row_block_dim_A,
                                                   rocsparse_int             col_block_dim_A,
                                                   const rocsparse_mat_descr descr_C,
                                                   float*                    bsr_val_C,
                                                   rocsparse_int*            bsr_row_ptr_C,
                                                   rocsparse_int*            bsr_col_ind_C,
                                                   rocsparse_int             row_block_dim_C,
                                                   rocsparse_int             col_block_dim_C,
                                                   void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gebsr2gebsr_template(handle,
                                                              dir,
                                                              mb,
                                                              nb,
                                                              nnzb,
                                                              descr_A,
                                                              bsr_val_A,
                                                              bsr_row_ptr_A,
                                                              bsr_col_ind_A,
                                                              row_block_dim_A,
                                                              col_block_dim_A,
                                                              descr_C,
                                                              bsr_val_C,
                                                              bsr_row_ptr_C,
                                                              bsr_col_ind_C,
                                                              row_block_dim_C,
                                                              col_block_dim_C,
                                                              temp_buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_dgebsr2gebsr(rocsparse_handle          handle,
                                                   rocsparse_direction       dir,
                                                   rocsparse_int             mb,
                                                   rocsparse_int             nb,
                                                   rocsparse_int             nnzb,
                                                   const rocsparse_mat_descr descr_A,
                                                   const double*             bsr_val_A,
                                                   const rocsparse_int*      bsr_row_ptr_A,
                                                   const rocsparse_int*      bsr_col_ind_A,
                                                   rocsparse_int             row_block_dim_A,
                                                   rocsparse_int             col_block_dim_A,
                                                   const rocsparse_mat_descr descr_C,
                                                   double*                   bsr_val_C,
                                                   rocsparse_int*            bsr_row_ptr_C,
                                                   rocsparse_int*            bsr_col_ind_C,
                                                   rocsparse_int             row_block_dim_C,
                                                   rocsparse_int             col_block_dim_C,
                                                   void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gebsr2gebsr_template(handle,
                                                              dir,
                                                              mb,
                                                              nb,
                                                              nnzb,
                                                              descr_A,
                                                              bsr_val_A,
                                                              bsr_row_ptr_A,
                                                              bsr_col_ind_A,
                                                              row_block_dim_A,
                                                              col_block_dim_A,
                                                              descr_C,
                                                              bsr_val_C,
                                                              bsr_row_ptr_C,
                                                              bsr_col_ind_C,
                                                              row_block_dim_C,
                                                              col_block_dim_C,
                                                              temp_buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_cgebsr2gebsr(rocsparse_handle               handle,
                                                   rocsparse_direction            dir,
                                                   rocsparse_int                  mb,
                                                   rocsparse_int                  nb,
                                                   rocsparse_int                  nnzb,
                                                   const rocsparse_mat_descr      descr_A,
                                                   const rocsparse_float_complex* bsr_val_A,
                                                   const rocsparse_int*           bsr_row_ptr_A,
                                                   const rocsparse_int*           bsr_col_ind_A,
                                                   rocsparse_int                  row_block_dim_A,
                                                   rocsparse_int                  col_block_dim_A,
                                                   const rocsparse_mat_descr      descr_C,
                                                   rocsparse_float_complex*       bsr_val_C,
                                                   rocsparse_int*                 bsr_row_ptr_C,
                                                   rocsparse_int*                 bsr_col_ind_C,
                                                   rocsparse_int                  row_block_dim_C,
                                                   rocsparse_int                  col_block_dim_C,
                                                   void*                          temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gebsr2gebsr_template(handle,
                                                              dir,
                                                              mb,
                                                              nb,
                                                              nnzb,
                                                              descr_A,
                                                              bsr_val_A,
                                                              bsr_row_ptr_A,
                                                              bsr_col_ind_A,
                                                              row_block_dim_A,
                                                              col_block_dim_A,
                                                              descr_C,
                                                              bsr_val_C,
                                                              bsr_row_ptr_C,
                                                              bsr_col_ind_C,
                                                              row_block_dim_C,
                                                              col_block_dim_C,
                                                              temp_buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_zgebsr2gebsr(rocsparse_handle                handle,
                                                   rocsparse_direction             dir,
                                                   rocsparse_int                   mb,
                                                   rocsparse_int                   nb,
                                                   rocsparse_int                   nnzb,
                                                   const rocsparse_mat_descr       descr_A,
                                                   const rocsparse_double_complex* bsr_val_A,
                                                   const rocsparse_int*            bsr_row_ptr_A,
                                                   const rocsparse_int*            bsr_col_ind_A,
                                                   rocsparse_int                   row_block_dim_A,
                                                   rocsparse_int                   col_block_dim_A,
                                                   const rocsparse_mat_descr       descr_C,
                                                   rocsparse_double_complex*       bsr_val_C,
                                                   rocsparse_int*                  bsr_row_ptr_C,
                                                   rocsparse_int*                  bsr_col_ind_C,
                                                   rocsparse_int                   row_block_dim_C,
                                                   rocsparse_int                   col_block_dim_C,
                                                   void*                           temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gebsr2gebsr_template(handle,
                                                              dir,
                                                              mb,
                                                              nb,
                                                              nnzb,
                                                              descr_A,
                                                              bsr_val_A,
                                                              bsr_row_ptr_A,
                                                              bsr_col_ind_A,
                                                              row_block_dim_A,
                                                              col_block_dim_A,
                                                              descr_C,
                                                              bsr_val_C,
                                                              bsr_row_ptr_C,
                                                              bsr_col_ind_C,
                                                              row_block_dim_C,
                                                              col_block_dim_C,
                                                              temp_buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

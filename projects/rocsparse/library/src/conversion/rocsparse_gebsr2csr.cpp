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

#include "internal/conversion/rocsparse_gebsr2csr.h"
#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_bsr2csr.hpp"
#include "rocsparse_calculate_nnz.hpp"
#include "rocsparse_common.h"
#include "rocsparse_gebsr2csr.hpp"

#include "gebsr2csr_device.h"

#define launch_gebsr2csr_block_per_row_1_32_kernel(block_size, brow_block_dim, bcol_block_dim) \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                                        \
        (rocsparse::                                                                           \
             gebsr2csr_block_per_row_1_32_kernel<block_size, brow_block_dim, bcol_block_dim>), \
        dim3(mb),                                                                              \
        dim3(block_size),                                                                      \
        0,                                                                                     \
        stream,                                                                                \
        dir,                                                                                   \
        mb,                                                                                    \
        nb,                                                                                    \
        bsr_descr->base,                                                                       \
        bsr_val,                                                                               \
        bsr_row_ptr,                                                                           \
        bsr_col_ind,                                                                           \
        row_block_dim,                                                                         \
        col_block_dim,                                                                         \
        csr_descr->base,                                                                       \
        csr_val,                                                                               \
        csr_row_ptr,                                                                           \
        csr_col_ind);

#define launch_gebsr2csr_block_per_row_33_128_kernel(                                 \
    block_size, brow_block_dim, bcol_block_dim, sub_row_block_dim, sub_col_block_dim) \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                               \
        (rocsparse::gebsr2csr_block_per_row_33_128_kernel<block_size,                 \
                                                          brow_block_dim,             \
                                                          bcol_block_dim,             \
                                                          sub_row_block_dim,          \
                                                          sub_col_block_dim>),        \
        dim3(mb),                                                                     \
        dim3(block_size),                                                             \
        0,                                                                            \
        stream,                                                                       \
        dir,                                                                          \
        mb,                                                                           \
        nb,                                                                           \
        bsr_descr->base,                                                              \
        bsr_val,                                                                      \
        bsr_row_ptr,                                                                  \
        bsr_col_ind,                                                                  \
        row_block_dim,                                                                \
        col_block_dim,                                                                \
        csr_descr->base,                                                              \
        csr_val,                                                                      \
        csr_row_ptr,                                                                  \
        csr_col_ind);

namespace rocsparse
{
    template <typename T>
    static rocsparse_status gebsr2csr_template_dispatch(rocsparse_handle          handle,
                                                        rocsparse_direction       dir,
                                                        rocsparse_int             mb,
                                                        rocsparse_int             nb,
                                                        const rocsparse_mat_descr bsr_descr,
                                                        const T*                  bsr_val,
                                                        const rocsparse_int*      bsr_row_ptr,
                                                        const rocsparse_int*      bsr_col_ind,
                                                        rocsparse_int             row_block_dim,
                                                        rocsparse_int             col_block_dim,
                                                        const rocsparse_mat_descr csr_descr,
                                                        T*                        csr_val,
                                                        rocsparse_int*            csr_row_ptr,
                                                        rocsparse_int*            csr_col_ind)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Stream
        hipStream_t stream = handle->stream;

        if(row_block_dim == col_block_dim)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsr2csr_core(handle,
                                                              dir,
                                                              mb,
                                                              nb,
                                                              bsr_descr,
                                                              bsr_val,
                                                              bsr_row_ptr,
                                                              bsr_col_ind,
                                                              row_block_dim,
                                                              csr_descr,
                                                              csr_val,
                                                              csr_row_ptr,
                                                              csr_col_ind));
            return rocsparse_status_success;
        }

        if(row_block_dim <= 2)
        {
            if(col_block_dim <= 2)
            {
                launch_gebsr2csr_block_per_row_1_32_kernel(256, 2, 2);
            }
            else if(col_block_dim <= 4)
            {
                launch_gebsr2csr_block_per_row_1_32_kernel(256, 2, 4);
            }
            else if(col_block_dim <= 8)
            {
                launch_gebsr2csr_block_per_row_1_32_kernel(256, 2, 8);
            }
            else if(col_block_dim <= 16)
            {
                launch_gebsr2csr_block_per_row_1_32_kernel(256, 2, 16);
            }
            else if(col_block_dim <= 32)
            {
                launch_gebsr2csr_block_per_row_1_32_kernel(256, 2, 32);
            }
            else if(col_block_dim <= 64)
            {
                launch_gebsr2csr_block_per_row_33_128_kernel(256, 2, 64, 2, 32);
            }
            else if(col_block_dim <= 128)
            {
                launch_gebsr2csr_block_per_row_33_128_kernel(256, 2, 128, 2, 32);
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
        }
        else if(row_block_dim <= 4)
        {
            if(col_block_dim <= 2)
            {
                launch_gebsr2csr_block_per_row_1_32_kernel(256, 4, 2);
            }
            else if(col_block_dim <= 4)
            {
                launch_gebsr2csr_block_per_row_1_32_kernel(256, 4, 4);
            }
            else if(col_block_dim <= 8)
            {
                launch_gebsr2csr_block_per_row_1_32_kernel(256, 4, 8);
            }
            else if(col_block_dim <= 16)
            {
                launch_gebsr2csr_block_per_row_1_32_kernel(1024, 4, 16);
            }
            else if(col_block_dim <= 32)
            {
                launch_gebsr2csr_block_per_row_1_32_kernel(1024, 4, 32);
            }
            else if(col_block_dim <= 64)
            {
                launch_gebsr2csr_block_per_row_33_128_kernel(1024, 4, 64, 4, 32);
            }
            else if(col_block_dim <= 128)
            {
                launch_gebsr2csr_block_per_row_33_128_kernel(1024, 4, 128, 4, 32);
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
        }
        else if(row_block_dim <= 8)
        {
            if(col_block_dim <= 2)
            {
                launch_gebsr2csr_block_per_row_1_32_kernel(1024, 8, 2);
            }
            else if(col_block_dim <= 4)
            {
                launch_gebsr2csr_block_per_row_1_32_kernel(1024, 8, 4);
            }
            else if(col_block_dim <= 8)
            {
                launch_gebsr2csr_block_per_row_1_32_kernel(1024, 8, 8);
            }
            else if(col_block_dim <= 16)
            {
                launch_gebsr2csr_block_per_row_1_32_kernel(1024, 8, 16);
            }
            else if(col_block_dim <= 32)
            {
                launch_gebsr2csr_block_per_row_1_32_kernel(1024, 8, 32);
            }
            else if(col_block_dim <= 64)
            {
                launch_gebsr2csr_block_per_row_33_128_kernel(1024, 8, 64, 8, 32);
            }
            else if(col_block_dim <= 128)
            {
                launch_gebsr2csr_block_per_row_33_128_kernel(1024, 8, 128, 8, 32);
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
        }
        else if(row_block_dim <= 16)
        {
            if(col_block_dim <= 2)
            {
                launch_gebsr2csr_block_per_row_1_32_kernel(1024, 16, 2);
            }
            else if(col_block_dim <= 4)
            {
                launch_gebsr2csr_block_per_row_1_32_kernel(1024, 16, 4);
            }
            else if(col_block_dim <= 8)
            {
                launch_gebsr2csr_block_per_row_1_32_kernel(1024, 16, 8);
            }
            else if(col_block_dim <= 16)
            {
                launch_gebsr2csr_block_per_row_1_32_kernel(1024, 16, 16);
            }
            else if(col_block_dim <= 32)
            {
                launch_gebsr2csr_block_per_row_1_32_kernel(1024, 16, 32);
            }
            else if(col_block_dim <= 64)
            {
                launch_gebsr2csr_block_per_row_33_128_kernel(1024, 16, 64, 16, 32);
            }
            else if(col_block_dim <= 128)
            {
                launch_gebsr2csr_block_per_row_33_128_kernel(1024, 16, 128, 16, 32);
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
        }
        else if(row_block_dim <= 32)
        {
            if(col_block_dim <= 2)
            {
                launch_gebsr2csr_block_per_row_1_32_kernel(1024, 32, 2);
            }
            else if(col_block_dim <= 4)
            {
                launch_gebsr2csr_block_per_row_1_32_kernel(1024, 32, 4);
            }
            else if(col_block_dim <= 8)
            {
                launch_gebsr2csr_block_per_row_1_32_kernel(1024, 32, 8);
            }
            else if(col_block_dim <= 16)
            {
                launch_gebsr2csr_block_per_row_1_32_kernel(1024, 32, 16);
            }
            else if(col_block_dim <= 32)
            {
                launch_gebsr2csr_block_per_row_1_32_kernel(1024, 32, 32);
            }
            else if(col_block_dim <= 64)
            {
                launch_gebsr2csr_block_per_row_33_128_kernel(1024, 32, 64, 32, 32);
            }
            else if(col_block_dim <= 128)
            {
                launch_gebsr2csr_block_per_row_33_128_kernel(1024, 32, 128, 32, 32);
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
        }
        else if(row_block_dim <= 64)
        {
            if(col_block_dim <= 2)
            {
                launch_gebsr2csr_block_per_row_33_128_kernel(1024, 64, 2, 32, 2);
            }
            else if(col_block_dim <= 4)
            {
                launch_gebsr2csr_block_per_row_33_128_kernel(1024, 64, 4, 32, 4);
            }
            else if(col_block_dim <= 8)
            {
                launch_gebsr2csr_block_per_row_33_128_kernel(1024, 64, 8, 32, 8);
            }
            else if(col_block_dim <= 16)
            {
                launch_gebsr2csr_block_per_row_33_128_kernel(1024, 64, 16, 32, 16);
            }
            else if(col_block_dim <= 32)
            {
                launch_gebsr2csr_block_per_row_33_128_kernel(1024, 64, 32, 32, 32);
            }
            else if(col_block_dim <= 64)
            {
                launch_gebsr2csr_block_per_row_33_128_kernel(1024, 64, 64, 32, 32);
            }
            else if(col_block_dim <= 128)
            {
                launch_gebsr2csr_block_per_row_33_128_kernel(1024, 64, 128, 32, 32);
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
        }
        else if(row_block_dim <= 128)
        {
            if(col_block_dim <= 2)
            {
                launch_gebsr2csr_block_per_row_33_128_kernel(1024, 128, 2, 32, 2);
            }
            else if(col_block_dim <= 4)
            {
                launch_gebsr2csr_block_per_row_33_128_kernel(1024, 128, 4, 32, 4);
            }
            else if(col_block_dim <= 8)
            {
                launch_gebsr2csr_block_per_row_33_128_kernel(1024, 128, 8, 32, 8);
            }
            else if(col_block_dim <= 16)
            {
                launch_gebsr2csr_block_per_row_33_128_kernel(1024, 128, 16, 32, 16);
            }
            else if(col_block_dim <= 32)
            {
                launch_gebsr2csr_block_per_row_33_128_kernel(1024, 128, 32, 32, 32);
            }
            else if(col_block_dim <= 64)
            {
                launch_gebsr2csr_block_per_row_33_128_kernel(1024, 128, 64, 32, 32);
            }
            else if(col_block_dim <= 128)
            {
                launch_gebsr2csr_block_per_row_33_128_kernel(1024, 128, 128, 32, 32);
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }

        return rocsparse_status_success;
    }
}

template <typename T>
rocsparse_status rocsparse::gebsr2csr_template(rocsparse_handle          handle, //0
                                               rocsparse_direction       dir, //1
                                               rocsparse_int             mb, //2
                                               rocsparse_int             nb, //3
                                               const rocsparse_mat_descr bsr_descr, //4
                                               const T*                  bsr_val, //5
                                               const rocsparse_int*      bsr_row_ptr, //6
                                               const rocsparse_int*      bsr_col_ind, //7
                                               rocsparse_int             row_block_dim, //8
                                               rocsparse_int             col_block_dim, //9
                                               const rocsparse_mat_descr csr_descr, //10
                                               T*                        csr_val, //11
                                               rocsparse_int*            csr_row_ptr, //12
                                               rocsparse_int*            csr_col_ind) //13
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xgebsr2csr"),
                         mb,
                         nb,
                         bsr_descr,
                         (const void*&)bsr_val,
                         (const void*&)bsr_row_ptr,
                         (const void*&)bsr_col_ind,
                         row_block_dim,
                         col_block_dim,
                         csr_descr,
                         (const void*&)csr_val,
                         (const void*&)csr_row_ptr,
                         (const void*&)csr_col_ind);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, dir);
    ROCSPARSE_CHECKARG_SIZE(2, mb);
    ROCSPARSE_CHECKARG_SIZE(3, nb);
    //
    ROCSPARSE_CHECKARG_POINTER(4, bsr_descr);
    ROCSPARSE_CHECKARG(4,
                       bsr_descr,
                       (bsr_descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);
    ROCSPARSE_CHECKARG(4,
                       bsr_descr,
                       (rocsparse_matrix_type_general != bsr_descr->type),
                       rocsparse_status_not_implemented);
    //
    ROCSPARSE_CHECKARG_POINTER(10, csr_descr);
    ROCSPARSE_CHECKARG(10,
                       csr_descr,
                       (csr_descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);
    ROCSPARSE_CHECKARG(10,
                       csr_descr,
                       (rocsparse_matrix_type_general != csr_descr->type),
                       rocsparse_status_not_implemented);

    ROCSPARSE_CHECKARG_SIZE(8, row_block_dim);
    ROCSPARSE_CHECKARG(8, row_block_dim, (row_block_dim == 0), rocsparse_status_invalid_size);

    ROCSPARSE_CHECKARG_SIZE(9, col_block_dim);
    ROCSPARSE_CHECKARG(9, col_block_dim, (col_block_dim == 0), rocsparse_status_invalid_size);

    ROCSPARSE_CHECKARG_ARRAY(6, mb, bsr_row_ptr);
    const rocsparse_int m = row_block_dim * mb;
    ROCSPARSE_CHECKARG_ARRAY(12, m, csr_row_ptr);

    // Quick return if possible
    if(mb == 0 || nb == 0)
    {
        if(csr_row_ptr != nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::valset(
                handle, m + 1, static_cast<rocsparse_int>(csr_descr->base), csr_row_ptr));
        }

        return rocsparse_status_success;
    }

    if(bsr_val == nullptr || bsr_col_ind == nullptr || csr_val == nullptr || csr_col_ind == nullptr)
    {
        int64_t nnzb;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::calculate_nnz(
            mb, rocsparse::get_indextype<rocsparse_int>(), bsr_row_ptr, &nnzb, handle->stream));
        ROCSPARSE_CHECKARG_ARRAY(5, nnzb * row_block_dim * col_block_dim, bsr_val);
        ROCSPARSE_CHECKARG_ARRAY(7, nnzb, bsr_col_ind);
        ROCSPARSE_CHECKARG_ARRAY(11, nnzb, csr_val);
        ROCSPARSE_CHECKARG_ARRAY(13, nnzb, csr_col_ind);
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gebsr2csr_template_dispatch(handle,
                                                                     dir,
                                                                     mb,
                                                                     nb,
                                                                     bsr_descr,
                                                                     bsr_val,
                                                                     bsr_row_ptr,
                                                                     bsr_col_ind,
                                                                     row_block_dim,
                                                                     col_block_dim,
                                                                     csr_descr,
                                                                     csr_val,
                                                                     csr_row_ptr,
                                                                     csr_col_ind));
    return rocsparse_status_success;
}

extern "C" {
#ifdef IMPL
#error IMPL IS ALREADY DEFINED
#endif

#define IMPL(name_, typer_)                                                    \
    rocsparse_status name_(rocsparse_handle          handle,                   \
                           rocsparse_direction       dir,                      \
                           rocsparse_int             mb,                       \
                           rocsparse_int             nb,                       \
                           const rocsparse_mat_descr bsr_descr,                \
                           const typer_*             bsr_val,                  \
                           const rocsparse_int*      bsr_row_ptr,              \
                           const rocsparse_int*      bsr_col_ind,              \
                           rocsparse_int             row_block_dim,            \
                           rocsparse_int             col_block_dim,            \
                           const rocsparse_mat_descr csr_descr,                \
                           typer_*                   csr_val,                  \
                           rocsparse_int*            csr_row_ptr,              \
                           rocsparse_int*            csr_col_ind)              \
    try                                                                        \
    {                                                                          \
        ROCSPARSE_ROUTINE_TRACE;                                               \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gebsr2csr_template(handle,        \
                                                                dir,           \
                                                                mb,            \
                                                                nb,            \
                                                                bsr_descr,     \
                                                                bsr_val,       \
                                                                bsr_row_ptr,   \
                                                                bsr_col_ind,   \
                                                                row_block_dim, \
                                                                col_block_dim, \
                                                                csr_descr,     \
                                                                csr_val,       \
                                                                csr_row_ptr,   \
                                                                csr_col_ind)); \
        return rocsparse_status_success;                                       \
    }                                                                          \
    catch(...)                                                                 \
    {                                                                          \
        RETURN_ROCSPARSE_EXCEPTION();                                          \
    }

IMPL(rocsparse_sgebsr2csr, float);
IMPL(rocsparse_dgebsr2csr, double);
IMPL(rocsparse_cgebsr2csr, rocsparse_float_complex);
IMPL(rocsparse_zgebsr2csr, rocsparse_double_complex);

#undef IMPL

} // extern "C"

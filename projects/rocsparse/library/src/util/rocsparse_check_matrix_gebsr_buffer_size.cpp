/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "internal/util/rocsparse_check_matrix_gebsr.h"

#include "rocsparse_check_matrix_gebsr.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_primitives.hpp"

#include "check_matrix_gebsr_device.h"

template <typename T, typename I, typename J>
rocsparse_status rocsparse::check_matrix_gebsr_buffer_size_core(rocsparse_handle      handle,
                                                                rocsparse_direction   dir,
                                                                J                     mb,
                                                                J                     nb,
                                                                I                     nnzb,
                                                                J                     row_block_dim,
                                                                J                     col_block_dim,
                                                                const T*              bsr_val,
                                                                const I*              bsr_row_ptr,
                                                                const J*              bsr_col_ind,
                                                                rocsparse_index_base  idx_base,
                                                                rocsparse_matrix_type matrix_type,
                                                                rocsparse_fill_mode   uplo,
                                                                rocsparse_storage_mode storage,
                                                                size_t*                buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    *buffer_size = 0;
    *buffer_size += ((sizeof(rocsparse_data_status) - 1) / 256 + 1) * 256; // data status

    if(storage == rocsparse_storage_mode_unsorted)
    {
        // Determine required rocprim buffer size
        size_t rocprim_buffer_size;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::sort_csr_column_indices_buffer_size(
            handle, mb, nb, nnzb, bsr_row_ptr, &rocprim_buffer_size));
        *buffer_size += ((rocprim_buffer_size - 1) / 256 + 1) * 256;

        // offset buffer
        *buffer_size += ((sizeof(I) * mb) / 256 + 1) * 256;

        // columns buffer
        *buffer_size += ((sizeof(J) * nnzb - 1) / 256 + 1) * 256;
        *buffer_size += ((sizeof(J) * nnzb - 1) / 256 + 1) * 256;
    }

    return rocsparse_status_success;
}

namespace rocsparse
{
    template <typename T, typename I, typename J>
    static rocsparse_status
        check_matrix_gebsr_buffer_size_quickreturn(rocsparse_handle       handle,
                                                   rocsparse_direction    dir,
                                                   J                      mb,
                                                   J                      nb,
                                                   I                      nnzb,
                                                   J                      row_block_dim,
                                                   J                      col_block_dim,
                                                   const T*               bsr_val,
                                                   const I*               bsr_row_ptr,
                                                   const J*               bsr_col_ind,
                                                   rocsparse_index_base   idx_base,
                                                   rocsparse_matrix_type  matrix_type,
                                                   rocsparse_fill_mode    uplo,
                                                   rocsparse_storage_mode storage,
                                                   size_t*                buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

        return rocsparse_status_continue;
    }
}

template <typename T, typename I, typename J>
rocsparse_status
    rocsparse::check_matrix_gebsr_buffer_size_checkarg(rocsparse_handle       handle, //0
                                                       rocsparse_direction    dir, //1
                                                       J                      mb, //2
                                                       J                      nb, //3
                                                       I                      nnzb, //4
                                                       J                      row_block_dim, //5
                                                       J                      col_block_dim, //6
                                                       const T*               bsr_val, //7
                                                       const I*               bsr_row_ptr, //8
                                                       const J*               bsr_col_ind, //9
                                                       rocsparse_index_base   idx_base, //10
                                                       rocsparse_matrix_type  matrix_type, //11
                                                       rocsparse_fill_mode    uplo, //12
                                                       rocsparse_storage_mode storage, //13
                                                       size_t*                buffer_size) //14
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, dir);
    ROCSPARSE_CHECKARG_SIZE(2, mb);
    ROCSPARSE_CHECKARG_SIZE(3, nb);
    ROCSPARSE_CHECKARG_SIZE(4, nnzb);

    ROCSPARSE_CHECKARG_SIZE(5, row_block_dim);
    ROCSPARSE_CHECKARG(5, row_block_dim, (row_block_dim == 0), rocsparse_status_invalid_size);

    ROCSPARSE_CHECKARG_SIZE(6, col_block_dim);
    ROCSPARSE_CHECKARG(6, col_block_dim, (col_block_dim == 0), rocsparse_status_invalid_size);

    ROCSPARSE_CHECKARG_ARRAY(7, nnzb, bsr_val);
    ROCSPARSE_CHECKARG_ARRAY(8, mb, bsr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(9, nnzb, bsr_col_ind);
    ROCSPARSE_CHECKARG_ENUM(10, idx_base);
    ROCSPARSE_CHECKARG_ENUM(11, matrix_type);
    ROCSPARSE_CHECKARG_ENUM(12, uplo);
    ROCSPARSE_CHECKARG_ENUM(13, storage);
    ROCSPARSE_CHECKARG_POINTER(14, buffer_size);

    const rocsparse_status status
        = rocsparse::check_matrix_gebsr_buffer_size_quickreturn(handle,
                                                                dir,
                                                                mb,
                                                                nb,
                                                                nnzb,
                                                                row_block_dim,
                                                                col_block_dim,
                                                                bsr_val,
                                                                bsr_row_ptr,
                                                                bsr_col_ind,
                                                                idx_base,
                                                                matrix_type,
                                                                uplo,
                                                                storage,
                                                                buffer_size);
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    return rocsparse_status_continue;
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                           \
    template rocsparse_status rocsparse::check_matrix_gebsr_buffer_size_core<TTYPE, ITYPE, JTYPE>( \
        rocsparse_handle       handle,                                                             \
        rocsparse_direction    dir,                                                                \
        JTYPE                  mb,                                                                 \
        JTYPE                  nb,                                                                 \
        ITYPE                  nnzb,                                                               \
        JTYPE                  row_block_dim,                                                      \
        JTYPE                  col_block_dim,                                                      \
        const TTYPE*           bsr_val,                                                            \
        const ITYPE*           bsr_row_ptr,                                                        \
        const JTYPE*           bsr_col_ind,                                                        \
        rocsparse_index_base   idx_base,                                                           \
        rocsparse_matrix_type  matrix_type,                                                        \
        rocsparse_fill_mode    uplo,                                                               \
        rocsparse_storage_mode storage,                                                            \
        size_t*                buffer_size);                                                                      \
    template rocsparse_status                                                                      \
        rocsparse::check_matrix_gebsr_buffer_size_checkarg<TTYPE, ITYPE, JTYPE>(                   \
            rocsparse_handle       handle,                                                         \
            rocsparse_direction    dir,                                                            \
            JTYPE                  mb,                                                             \
            JTYPE                  nb,                                                             \
            ITYPE                  nnzb,                                                           \
            JTYPE                  row_block_dim,                                                  \
            JTYPE                  col_block_dim,                                                  \
            const TTYPE*           bsr_val,                                                        \
            const ITYPE*           bsr_row_ptr,                                                    \
            const JTYPE*           bsr_col_ind,                                                    \
            rocsparse_index_base   idx_base,                                                       \
            rocsparse_matrix_type  matrix_type,                                                    \
            rocsparse_fill_mode    uplo,                                                           \
            rocsparse_storage_mode storage,                                                        \
            size_t*                buffer_size);

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);
#undef INSTANTIATE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
#define C_IMPL(NAME, T)                                                                       \
    extern "C" rocsparse_status NAME(rocsparse_handle       handle,                           \
                                     rocsparse_direction    dir,                              \
                                     rocsparse_int          mb,                               \
                                     rocsparse_int          nb,                               \
                                     rocsparse_int          nnzb,                             \
                                     rocsparse_int          row_block_dim,                    \
                                     rocsparse_int          col_block_dim,                    \
                                     const T*               bsr_val,                          \
                                     const rocsparse_int*   bsr_row_ptr,                      \
                                     const rocsparse_int*   bsr_col_ind,                      \
                                     rocsparse_index_base   idx_base,                         \
                                     rocsparse_matrix_type  matrix_type,                      \
                                     rocsparse_fill_mode    uplo,                             \
                                     rocsparse_storage_mode storage,                          \
                                     size_t*                buffer_size)                      \
    try                                                                                       \
    {                                                                                         \
        ROCSPARSE_ROUTINE_TRACE;                                                              \
        RETURN_IF_ROCSPARSE_ERROR(                                                            \
            (rocsparse::check_matrix_gebsr_buffer_size_impl<T, rocsparse_int, rocsparse_int>( \
                handle,                                                                       \
                dir,                                                                          \
                mb,                                                                           \
                nb,                                                                           \
                nnzb,                                                                         \
                row_block_dim,                                                                \
                col_block_dim,                                                                \
                bsr_val,                                                                      \
                bsr_row_ptr,                                                                  \
                bsr_col_ind,                                                                  \
                idx_base,                                                                     \
                matrix_type,                                                                  \
                uplo,                                                                         \
                storage,                                                                      \
                buffer_size)));                                                               \
        return rocsparse_status_success;                                                      \
    }                                                                                         \
    catch(...)                                                                                \
    {                                                                                         \
        RETURN_ROCSPARSE_EXCEPTION();                                                         \
    }

C_IMPL(rocsparse_scheck_matrix_gebsr_buffer_size, float);
C_IMPL(rocsparse_dcheck_matrix_gebsr_buffer_size, double);
C_IMPL(rocsparse_ccheck_matrix_gebsr_buffer_size, rocsparse_float_complex);
C_IMPL(rocsparse_zcheck_matrix_gebsr_buffer_size, rocsparse_double_complex);
#undef C_IMPL

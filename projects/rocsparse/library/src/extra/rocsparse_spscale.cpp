/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the Software), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "internal/generic/rocsparse_spscale.h"
#include "rocsparse_common.h"
#include "rocsparse_control.hpp"
#include "rocsparse_datatype_utils.hpp"
#include "rocsparse_indextype_utils.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    // Scale the value array of C in place by alpha (host or device pointer mode).
    template <typename T>
    static rocsparse_status
        spscale_scale_values(rocsparse_handle handle, int64_t nnz, const void* alpha, void* c_val)
    {
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::scale_array(
            handle, nnz, static_cast<const T*>(alpha), static_cast<T*>(c_val))));
        return rocsparse_status_success;
    }

    // Formats supported by rocsparse_spscale (all generic sparse formats).
    static bool spscale_is_supported_format(rocsparse_format format)
    {
        switch(format)
        {
        case rocsparse_format_coo:
        case rocsparse_format_coo_aos:
        case rocsparse_format_csr:
        case rocsparse_format_csc:
        case rocsparse_format_bsr:
        case rocsparse_format_ell:
        case rocsparse_format_bell:
        case rocsparse_format_sell:
            return true;
        }
        return false;
    }

    // Argument checks shared by rocsparse_spscale and rocsparse_spscale_buffer_size. The two
    // routines place mat_A and mat_C at different argument positions, so the caller passes the
    // matching argument indices (\p arg_A, \p arg_C) used for error reporting.
    static rocsparse_status spscale_checkarg(rocsparse_handle            handle,
                                             rocsparse_const_spmat_descr mat_A,
                                             rocsparse_spmat_descr       mat_C,
                                             int                         arg_A,
                                             int                         arg_C)
    {
        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_POINTER(arg_A, mat_A);
        ROCSPARSE_CHECKARG_POINTER(arg_C, mat_C);
        ROCSPARSE_CHECKARG(arg_A, mat_A, (mat_A->init == false), rocsparse_status_not_initialized);
        ROCSPARSE_CHECKARG(arg_C, mat_C, (mat_C->init == false), rocsparse_status_not_initialized);

        // A and C must share the same format, and the format must be one that is supported.
        ROCSPARSE_CHECKARG(
            arg_C, mat_C, (mat_C->format != mat_A->format), rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(arg_A,
                           mat_A,
                           (rocsparse::spscale_is_supported_format(mat_A->format) == false),
                           rocsparse_status_not_implemented);

        // A and C must have matching shape, nonzero count and types.
        ROCSPARSE_CHECKARG(
            arg_C, mat_C, (mat_C->rows != mat_A->rows), rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG(
            arg_C, mat_C, (mat_C->cols != mat_A->cols), rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG(arg_C, mat_C, (mat_C->nnz != mat_A->nnz), rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG(
            arg_C, mat_C, (mat_C->row_type != mat_A->row_type), rocsparse_status_type_mismatch);
        ROCSPARSE_CHECKARG(
            arg_C, mat_C, (mat_C->col_type != mat_A->col_type), rocsparse_status_type_mismatch);
        ROCSPARSE_CHECKARG(
            arg_C, mat_C, (mat_C->data_type != mat_A->data_type), rocsparse_status_type_mismatch);

        // Format specific layout parameters must also match between A and C.
        switch(mat_A->format)
        {
        case rocsparse_format_bsr:
        {
            ROCSPARSE_CHECKARG(arg_C,
                               mat_C,
                               (mat_C->block_dim != mat_A->block_dim),
                               rocsparse_status_invalid_size);
            break;
        }
        case rocsparse_format_ell:
        {
            ROCSPARSE_CHECKARG(arg_C,
                               mat_C,
                               (mat_C->ell_width != mat_A->ell_width),
                               rocsparse_status_invalid_size);
            break;
        }
        case rocsparse_format_bell:
        {
            ROCSPARSE_CHECKARG(
                arg_C, mat_C, (mat_C->ell_cols != mat_A->ell_cols), rocsparse_status_invalid_size);
            ROCSPARSE_CHECKARG(arg_C,
                               mat_C,
                               (mat_C->block_dim != mat_A->block_dim),
                               rocsparse_status_invalid_size);
            break;
        }
        case rocsparse_format_sell:
        {
            ROCSPARSE_CHECKARG(arg_C,
                               mat_C,
                               (mat_C->sell_slice_size != mat_A->sell_slice_size),
                               rocsparse_status_invalid_size);
            ROCSPARSE_CHECKARG(arg_C,
                               mat_C,
                               (mat_C->sell_colval_size != mat_A->sell_colval_size),
                               rocsparse_status_invalid_size);
            break;
        }
        case rocsparse_format_coo:
        case rocsparse_format_coo_aos:
        case rocsparse_format_csr:
        case rocsparse_format_csc:
            break;
        }

        // Batched matrices are not supported.
        ROCSPARSE_CHECKARG(
            arg_A, mat_A, (mat_A->batch_count != 1), rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(
            arg_C, mat_C, (mat_C->batch_count != 1), rocsparse_status_not_implemented);

        // Differing index base between A and C is not yet supported (base conversion follow-up).
        ROCSPARSE_CHECKARG(
            arg_C, mat_C, (mat_C->idx_base != mat_A->idx_base), rocsparse_status_not_implemented);

        return rocsparse_status_continue;
    }

    // Device-to-device copy of a single structure array. Only performed when both endpoints are
    // valid and there is something to copy.
    static rocsparse_status
        spscale_copy_array(rocsparse_handle handle, void* dst, const void* src, size_t bytes)
    {
        if(bytes > 0 && dst != nullptr && src != nullptr)
        {
            RETURN_IF_HIP_ERROR(
                hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, handle->stream));
        }
        return rocsparse_status_success;
    }

    static rocsparse_status spscale_core(rocsparse_handle            handle,
                                         const void*                 alpha,
                                         rocsparse_const_spmat_descr mat_A,
                                         rocsparse_spmat_descr       mat_C)
    {
        const int64_t rows = mat_A->rows;
        const int64_t cols = mat_A->cols;
        const int64_t nnz  = mat_A->nnz;

        const size_t row_size = rocsparse::indextype_sizeof(mat_A->row_type);
        const size_t col_size = rocsparse::indextype_sizeof(mat_A->col_type);
        const size_t val_size = rocsparse::datatype_sizeof(mat_A->data_type);

        // Number of value entries to copy and scale. This is format specific and is not always
        // equal to nnz (e.g. BSR stores block_dim^2 values per block-nonzero).
        int64_t val_length = 0;

        // Copy the sparsity structure of A into C. The index base of A and C is guaranteed to
        // match by the argument checks, so a plain device-to-device copy is sufficient.
        switch(mat_A->format)
        {
        case rocsparse_format_csr:
        {
            // row_data[rows + 1], col_data[nnz], val[nnz].
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::spscale_copy_array(
                handle, mat_C->row_data, mat_A->const_row_data, size_t(rows + 1) * row_size));
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::spscale_copy_array(
                handle, mat_C->col_data, mat_A->const_col_data, size_t(nnz) * col_size));
            val_length = nnz;
            break;
        }
        case rocsparse_format_csc:
        {
            // col_data[cols + 1], row_data[nnz], val[nnz].
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::spscale_copy_array(
                handle, mat_C->col_data, mat_A->const_col_data, size_t(cols + 1) * col_size));
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::spscale_copy_array(
                handle, mat_C->row_data, mat_A->const_row_data, size_t(nnz) * row_size));
            val_length = nnz;
            break;
        }
        case rocsparse_format_coo:
        {
            // row_data[nnz], col_data[nnz], val[nnz].
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::spscale_copy_array(
                handle, mat_C->row_data, mat_A->const_row_data, size_t(nnz) * row_size));
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::spscale_copy_array(
                handle, mat_C->col_data, mat_A->const_col_data, size_t(nnz) * col_size));
            val_length = nnz;
            break;
        }
        case rocsparse_format_coo_aos:
        {
            // ind_data[2 * nnz], val[nnz].
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::spscale_copy_array(
                handle, mat_C->ind_data, mat_A->const_ind_data, size_t(2 * nnz) * col_size));
            val_length = nnz;
            break;
        }
        case rocsparse_format_bsr:
        {
            // row_data[rows + 1], col_data[nnz], val[nnz * block_dim * block_dim].
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::spscale_copy_array(
                handle, mat_C->row_data, mat_A->const_row_data, size_t(rows + 1) * row_size));
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::spscale_copy_array(
                handle, mat_C->col_data, mat_A->const_col_data, size_t(nnz) * col_size));
            val_length = nnz * mat_A->block_dim * mat_A->block_dim;
            break;
        }
        case rocsparse_format_ell:
        {
            // col_data[rows * ell_width], val[rows * ell_width]. ELL has no row_data and the
            // descriptor nnz already equals rows * ell_width.
            const int64_t ell_length = rows * mat_A->ell_width;
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::spscale_copy_array(
                handle, mat_C->col_data, mat_A->const_col_data, size_t(ell_length) * col_size));
            val_length = ell_length;
            break;
        }
        case rocsparse_format_sell:
        {
            // row_data[nslices + 1], col_data[sell_colval_size], val[sell_colval_size].
            const int64_t nslices
                = (mat_A->sell_slice_size > 0)
                      ? (rows + mat_A->sell_slice_size - 1) / mat_A->sell_slice_size
                      : 0;
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::spscale_copy_array(
                handle, mat_C->row_data, mat_A->const_row_data, size_t(nslices + 1) * row_size));
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::spscale_copy_array(handle,
                                              mat_C->col_data,
                                              mat_A->const_col_data,
                                              size_t(mat_A->sell_colval_size) * col_size));
            val_length = mat_A->sell_colval_size;
            break;
        }
        case rocsparse_format_bell:
        {
            // Blocked-ELL: the descriptor stores rows and ell_cols in block units (rows is the
            // number of block-rows, ell_cols is the ELL block-width), matching how the client
            // bell_matrix and rocsparse_create_bell_descr are wired up. col_data holds one
            // block-column index per block, laid out as rows * ell_cols. val holds a
            // block_dim x block_dim dense block per ELL position, laid out as
            // rows * ell_cols * block_dim * block_dim scalars. There is no row_data.
            const int64_t block_dim      = mat_A->block_dim;
            const int64_t col_ind_length = rows * mat_A->ell_cols;
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::spscale_copy_array(
                handle, mat_C->col_data, mat_A->const_col_data, size_t(col_ind_length) * col_size));
            val_length = rows * mat_A->ell_cols * block_dim * block_dim;
            break;
        }
        }

        if(val_length > 0)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::spscale_copy_array(
                handle, mat_C->val_data, mat_A->const_val_data, size_t(val_length) * val_size));

            // Scale the copied values by alpha.
            switch(mat_A->data_type)
            {
            case rocsparse_datatype_f32_r:
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::spscale_scale_values<float>(
                    handle, val_length, alpha, mat_C->val_data));
                break;
            case rocsparse_datatype_f64_r:
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::spscale_scale_values<double>(
                    handle, val_length, alpha, mat_C->val_data));
                break;
            case rocsparse_datatype_f32_c:
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::spscale_scale_values<rocsparse_float_complex>(
                    handle, val_length, alpha, mat_C->val_data));
                break;
            case rocsparse_datatype_f64_c:
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::spscale_scale_values<rocsparse_double_complex>(
                    handle, val_length, alpha, mat_C->val_data));
                break;
            default:
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
        }

        return rocsparse_status_success;
    }
}

extern "C" rocsparse_status rocsparse_spscale_buffer_size(rocsparse_handle            handle,
                                                          rocsparse_const_spmat_descr mat_A,
                                                          rocsparse_spmat_descr       mat_C,
                                                          size_t*          buffer_size_in_bytes,
                                                          rocsparse_error* p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    rocsparse::log_trace(
        "rocsparse_spscale_buffer_size", handle, mat_A, mat_C, buffer_size_in_bytes);

    // p_error is reserved for forward compatibility and is not populated yet.
    (void)p_error;

    const rocsparse_status status = rocsparse::spscale_checkarg(handle, mat_A, mat_C, 1, 2);
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG_POINTER(3, buffer_size_in_bytes);

    // Scaling does not require any additional workspace for the supported formats.
    *buffer_size_in_bytes = 0;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_spscale(rocsparse_handle            handle,
                                              const void*                 alpha,
                                              rocsparse_const_spmat_descr mat_A,
                                              rocsparse_spmat_descr       mat_C,
                                              size_t                      buffer_size_in_bytes,
                                              void*                       temp_buffer,
                                              rocsparse_error*            p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    rocsparse::log_trace(
        "rocsparse_spscale", handle, alpha, mat_A, mat_C, buffer_size_in_bytes, temp_buffer);

    // No workspace is required for the currently supported formats.
    (void)buffer_size_in_bytes;
    (void)temp_buffer;
    // p_error is reserved for forward compatibility and is not populated yet.
    (void)p_error;

    ROCSPARSE_CHECKARG_POINTER(1, alpha);

    const rocsparse_status status = rocsparse::spscale_checkarg(handle, mat_A, mat_C, 2, 3);
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::spscale_core(handle, alpha, mat_A, mat_C));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

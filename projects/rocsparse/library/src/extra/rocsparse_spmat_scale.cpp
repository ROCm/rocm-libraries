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

#include "internal/generic/rocsparse_spmat_scale.h"
#include "rocsparse_common.h"
#include "rocsparse_control.hpp"
#include "rocsparse_datatype_utils.hpp"
#include "rocsparse_indextype_utils.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    // Scale the value array of the target in place by alpha (host or device pointer mode, taken
    // from the handle which the caller sets to match the alpha descriptor).
    template <typename T>
    static rocsparse_status spmat_scale_scale_values(rocsparse_handle handle,
                                                     int64_t          nnz,
                                                     const void*      alpha,
                                                     void*            target_val)
    {
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::scale_array(
            handle, nnz, static_cast<const T*>(alpha), static_cast<T*>(target_val))));
        return rocsparse_status_success;
    }

    // Formats supported by rocsparse_spmat_scale (all generic sparse formats).
    static bool spmat_scale_is_supported_format(rocsparse_format format)
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

    // Argument checks on the source and target sparse matrices. The caller passes the matching
    // argument indices (\p arg_source, \p arg_target) used for error reporting.
    static rocsparse_status spmat_scale_checkarg(rocsparse_handle            handle,
                                                 rocsparse_const_spmat_descr source,
                                                 rocsparse_spmat_descr       target,
                                                 int                         arg_source,
                                                 int                         arg_target)
    {
        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_POINTER(arg_source, source);
        ROCSPARSE_CHECKARG_POINTER(arg_target, target);
        ROCSPARSE_CHECKARG(
            arg_source, source, (source->init == false), rocsparse_status_not_initialized);
        ROCSPARSE_CHECKARG(
            arg_target, target, (target->init == false), rocsparse_status_not_initialized);

        // Source and target must share the same format, and the format must be supported.
        ROCSPARSE_CHECKARG(arg_target,
                           target,
                           (target->format != source->format),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(arg_source,
                           source,
                           (rocsparse::spmat_scale_is_supported_format(source->format) == false),
                           rocsparse_status_not_implemented);

        // Source and target must have matching shape, nonzero count and types.
        ROCSPARSE_CHECKARG(
            arg_target, target, (target->rows != source->rows), rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG(
            arg_target, target, (target->cols != source->cols), rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG(
            arg_target, target, (target->nnz != source->nnz), rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG(arg_target,
                           target,
                           (target->row_type != source->row_type),
                           rocsparse_status_type_mismatch);
        ROCSPARSE_CHECKARG(arg_target,
                           target,
                           (target->col_type != source->col_type),
                           rocsparse_status_type_mismatch);
        ROCSPARSE_CHECKARG(arg_target,
                           target,
                           (target->data_type != source->data_type),
                           rocsparse_status_type_mismatch);

        // Format specific layout parameters must also match between source and target.
        switch(source->format)
        {
        case rocsparse_format_bsr:
        {
            ROCSPARSE_CHECKARG(arg_target,
                               target,
                               (target->block_dim != source->block_dim),
                               rocsparse_status_invalid_size);
            break;
        }
        case rocsparse_format_ell:
        {
            ROCSPARSE_CHECKARG(arg_target,
                               target,
                               (target->ell_width != source->ell_width),
                               rocsparse_status_invalid_size);
            break;
        }
        case rocsparse_format_bell:
        {
            ROCSPARSE_CHECKARG(arg_target,
                               target,
                               (target->ell_cols != source->ell_cols),
                               rocsparse_status_invalid_size);
            ROCSPARSE_CHECKARG(arg_target,
                               target,
                               (target->block_dim != source->block_dim),
                               rocsparse_status_invalid_size);
            break;
        }
        case rocsparse_format_sell:
        {
            ROCSPARSE_CHECKARG(arg_target,
                               target,
                               (target->sell_slice_size != source->sell_slice_size),
                               rocsparse_status_invalid_size);
            ROCSPARSE_CHECKARG(arg_target,
                               target,
                               (target->sell_colval_size != source->sell_colval_size),
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
            arg_source, source, (source->batch_count != 1), rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(
            arg_target, target, (target->batch_count != 1), rocsparse_status_not_implemented);

        // Differing index base between source and target is not yet supported (base conversion
        // follow-up).
        ROCSPARSE_CHECKARG(arg_target,
                           target,
                           (target->idx_base != source->idx_base),
                           rocsparse_status_not_implemented);

        return rocsparse_status_continue;
    }

    // Device-to-device copy of a single structure array. Only performed when both endpoints are
    // valid and there is something to copy.
    static rocsparse_status
        spmat_scale_copy_array(rocsparse_handle handle, void* dst, const void* src, size_t bytes)
    {
        if(bytes > 0 && dst != nullptr && src != nullptr && dst != src)
        {
            RETURN_IF_HIP_ERROR(
                hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, handle->stream));
        }
        return rocsparse_status_success;
    }

    static rocsparse_status spmat_scale_core(rocsparse_handle            handle,
                                             const void*                 alpha,
                                             rocsparse_pointer_mode      alpha_pointer_mode,
                                             rocsparse_const_spmat_descr source,
                                             rocsparse_spmat_descr       target)
    {
        const int64_t rows = source->rows;
        const int64_t cols = source->cols;
        const int64_t nnz  = source->nnz;

        const size_t row_size = rocsparse::indextype_sizeof(source->row_type);
        const size_t col_size = rocsparse::indextype_sizeof(source->col_type);
        const size_t val_size = rocsparse::datatype_sizeof(source->data_type);

        // In-place scaling: the target already holds the source arrays, so no copy is required.
        const bool in_place = (target->val_data == source->const_val_data);

        // Number of value entries to copy and scale. This is format specific and is not always
        // equal to nnz (e.g. BSR stores block_dim^2 values per block-nonzero).
        int64_t val_length = 0;

        // Copy the sparsity structure of the source into the target. The index base of source and
        // target is guaranteed to match by the argument checks, so a plain device-to-device copy
        // is sufficient.
        switch(source->format)
        {
        case rocsparse_format_csr:
        {
            // row_data[rows + 1], col_data[nnz], val[nnz].
            if(!in_place)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::spmat_scale_copy_array(
                    handle, target->row_data, source->const_row_data, size_t(rows + 1) * row_size));
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::spmat_scale_copy_array(
                    handle, target->col_data, source->const_col_data, size_t(nnz) * col_size));
            }
            val_length = nnz;
            break;
        }
        case rocsparse_format_csc:
        {
            // col_data[cols + 1], row_data[nnz], val[nnz].
            if(!in_place)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::spmat_scale_copy_array(
                    handle, target->col_data, source->const_col_data, size_t(cols + 1) * col_size));
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::spmat_scale_copy_array(
                    handle, target->row_data, source->const_row_data, size_t(nnz) * row_size));
            }
            val_length = nnz;
            break;
        }
        case rocsparse_format_coo:
        {
            // row_data[nnz], col_data[nnz], val[nnz].
            if(!in_place)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::spmat_scale_copy_array(
                    handle, target->row_data, source->const_row_data, size_t(nnz) * row_size));
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::spmat_scale_copy_array(
                    handle, target->col_data, source->const_col_data, size_t(nnz) * col_size));
            }
            val_length = nnz;
            break;
        }
        case rocsparse_format_coo_aos:
        {
            // ind_data[2 * nnz], val[nnz].
            if(!in_place)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::spmat_scale_copy_array(
                    handle, target->ind_data, source->const_ind_data, size_t(2 * nnz) * col_size));
            }
            val_length = nnz;
            break;
        }
        case rocsparse_format_bsr:
        {
            // row_data[rows + 1], col_data[nnz], val[nnz * block_dim * block_dim].
            if(!in_place)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::spmat_scale_copy_array(
                    handle, target->row_data, source->const_row_data, size_t(rows + 1) * row_size));
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::spmat_scale_copy_array(
                    handle, target->col_data, source->const_col_data, size_t(nnz) * col_size));
            }
            val_length = nnz * source->block_dim * source->block_dim;
            break;
        }
        case rocsparse_format_ell:
        {
            // col_data[rows * ell_width], val[rows * ell_width]. ELL has no row_data and the
            // descriptor nnz already equals rows * ell_width.
            const int64_t ell_length = rows * source->ell_width;
            if(!in_place)
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::spmat_scale_copy_array(handle,
                                                      target->col_data,
                                                      source->const_col_data,
                                                      size_t(ell_length) * col_size));
            }
            val_length = ell_length;
            break;
        }
        case rocsparse_format_sell:
        {
            // row_data[nslices + 1], col_data[sell_colval_size], val[sell_colval_size].
            const int64_t nslices
                = (source->sell_slice_size > 0)
                      ? (rows + source->sell_slice_size - 1) / source->sell_slice_size
                      : 0;
            if(!in_place)
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::spmat_scale_copy_array(handle,
                                                      target->row_data,
                                                      source->const_row_data,
                                                      size_t(nslices + 1) * row_size));
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::spmat_scale_copy_array(handle,
                                                      target->col_data,
                                                      source->const_col_data,
                                                      size_t(source->sell_colval_size) * col_size));
            }
            val_length = source->sell_colval_size;
            break;
        }
        case rocsparse_format_bell:
        {
            // Blocked-ELL: the descriptor stores rows and ell_cols in block units (rows is the
            // number of block-rows, ell_cols is the ELL block-width). col_data holds one
            // block-column index per block, laid out as rows * ell_cols. val holds a
            // block_dim x block_dim dense block per ELL position, laid out as
            // rows * ell_cols * block_dim * block_dim scalars. There is no row_data.
            const int64_t block_dim      = source->block_dim;
            const int64_t col_ind_length = rows * source->ell_cols;
            if(!in_place)
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::spmat_scale_copy_array(handle,
                                                      target->col_data,
                                                      source->const_col_data,
                                                      size_t(col_ind_length) * col_size));
            }
            val_length = rows * source->ell_cols * block_dim * block_dim;
            break;
        }
        }

        if(val_length > 0)
        {
            if(!in_place)
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::spmat_scale_copy_array(handle,
                                                      target->val_data,
                                                      source->const_val_data,
                                                      size_t(val_length) * val_size));
            }

            // Scale the target values by alpha. scale_array reads the scalar according to the
            // handle pointer mode, so temporarily align it with the alpha descriptor.
            const rocsparse_pointer_mode saved_mode = handle->pointer_mode;
            handle->pointer_mode                    = alpha_pointer_mode;

            rocsparse_status scale_status = rocsparse_status_success;
            switch(source->data_type)
            {
            case rocsparse_datatype_f32_r:
                scale_status = rocsparse::spmat_scale_scale_values<float>(
                    handle, val_length, alpha, target->val_data);
                break;
            case rocsparse_datatype_f64_r:
                scale_status = rocsparse::spmat_scale_scale_values<double>(
                    handle, val_length, alpha, target->val_data);
                break;
            case rocsparse_datatype_f32_c:
                scale_status = rocsparse::spmat_scale_scale_values<rocsparse_float_complex>(
                    handle, val_length, alpha, target->val_data);
                break;
            case rocsparse_datatype_f64_c:
                scale_status = rocsparse::spmat_scale_scale_values<rocsparse_double_complex>(
                    handle, val_length, alpha, target->val_data);
                break;
            default:
                scale_status = rocsparse_status_not_implemented;
            }

            handle->pointer_mode = saved_mode;
            RETURN_IF_ROCSPARSE_ERROR(scale_status);
        }

        return rocsparse_status_success;
    }
}

extern "C" rocsparse_status rocsparse_spmat_scale(rocsparse_handle            handle,
                                                  rocsparse_const_dnvec_descr alpha,
                                                  rocsparse_spmat_descr       target,
                                                  rocsparse_const_spmat_descr source,
                                                  rocsparse_error*            p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    rocsparse::log_trace("rocsparse_spmat_scale", handle, alpha, target, source);

    // p_error is reserved for forward compatibility and is not populated yet.
    (void)p_error;

    ROCSPARSE_CHECKARG_POINTER(1, alpha);

    const rocsparse_status status = rocsparse::spmat_scale_checkarg(handle, source, target, 3, 2);
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    // alpha is a single scalar dense vector; its data type must match the matrices.
    ROCSPARSE_CHECKARG(1, alpha, (alpha->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(
        1, alpha, (alpha->size != 1 || alpha->batch_count != 1), rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(
        1, alpha, (alpha->data_type != source->data_type), rocsparse_status_type_mismatch);
    ROCSPARSE_CHECKARG_POINTER(1, alpha->const_values);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::spmat_scale_core(
        handle, alpha->const_values, alpha->pointer_mode, source, target));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

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

#include "rocsparse-config.h"

#include "internal/generic/rocsparse_spmat_scale.h"
#include "rocsparse_common.h"
#include "rocsparse_control.hpp"
#include "rocsparse_datatype_utils.hpp"
#include "rocsparse_indextype_utils.hpp"
#include "rocsparse_utility.hpp"

// rocsparse_spmat_scale is gated behind the ROCSPARSE_WITH_SPMAT_SCALE build-time feature flag.
#ifdef ROCSPARSE_WITH_SPMAT_SCALE

namespace rocsparse
{
    // Scale the value array of the target in place by alpha. The memory space of alpha is taken
    // from its descriptor and passed explicitly, so the handle pointer mode is never involved.
    template <typename T>
    static rocsparse_status spmat_scale_scale_values(rocsparse_handle       handle,
                                                     int64_t                nnz,
                                                     rocsparse_pointer_mode alpha_pointer_mode,
                                                     const void*            alpha,
                                                     void*                  target_val)
    {
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::scale_array(handle,
                                                          nnz,
                                                          alpha_pointer_mode,
                                                          static_cast<const T*>(alpha),
                                                          static_cast<T*>(target_val))));
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

        // Source and target must share the same format, and the format must be supported.
        ROCSPARSE_CHECKARG(arg_target,
                           target,
                           (target->format != source->format),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(arg_source,
                           source,
                           (rocsparse::spmat_scale_is_supported_format(source->format) == false),
                           rocsparse_status_not_implemented);

        // Source and target must have matching shape and nonzero count so their value arrays have
        // the same length, and the same value data type. The index arrays are never touched, so
        // the index types and index base of source and target are allowed to differ.
        ROCSPARSE_CHECKARG(
            arg_target, target, (target->rows != source->rows), rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG(
            arg_target, target, (target->cols != source->cols), rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG(
            arg_target, target, (target->nnz != source->nnz), rocsparse_status_invalid_size);
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

        return rocsparse_status_continue;
    }

    // Number of scalar value entries for the given format. This is not always nnz (e.g. BSR
    // stores block_dim^2 values per block-nonzero, ELL stores rows*ell_width, and so on).
    static int64_t spmat_scale_value_length(rocsparse_const_spmat_descr mat)
    {
        switch(mat->format)
        {
        case rocsparse_format_coo:
        case rocsparse_format_coo_aos:
        case rocsparse_format_csr:
        case rocsparse_format_csc:
            return mat->nnz;
        case rocsparse_format_bsr:
            return mat->nnz * mat->block_dim * mat->block_dim;
        case rocsparse_format_ell:
            return mat->rows * mat->ell_width;
        case rocsparse_format_sell:
            return mat->sell_colval_size;
        case rocsparse_format_bell:
            return mat->rows * mat->ell_cols * mat->block_dim * mat->block_dim;
        }
        return 0;
    }

    static rocsparse_status spmat_scale_core(rocsparse_handle            handle,
                                             const void*                 alpha,
                                             rocsparse_pointer_mode      alpha_pointer_mode,
                                             rocsparse_const_spmat_descr source,
                                             rocsparse_spmat_descr       target)
    {
        const size_t  val_size   = rocsparse::datatype_sizeof(source->data_type);
        const int64_t val_length = rocsparse::spmat_scale_value_length(source);

        if(val_length <= 0)
        {
            return rocsparse_status_success;
        }

        // This routine only writes the value array; the sparsity pattern of target is assumed to
        // already match source and is never copied. When target and source are distinct, the
        // source values are copied into target before scaling; when they alias, the scaling is
        // done in place.
        const bool in_place = (target->val_data == source->const_val_data);
        if(!in_place)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(target->val_data,
                                               source->const_val_data,
                                               size_t(val_length) * val_size,
                                               hipMemcpyDeviceToDevice,
                                               handle->stream));
        }

        // Scale the target values by alpha, passing alpha's memory space explicitly so the handle
        // pointer mode is not touched.
        switch(source->data_type)
        {
        case rocsparse_datatype_f32_r:
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::spmat_scale_scale_values<float>(
                handle, val_length, alpha_pointer_mode, alpha, target->val_data));
            break;
        case rocsparse_datatype_f64_r:
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::spmat_scale_scale_values<double>(
                handle, val_length, alpha_pointer_mode, alpha, target->val_data));
            break;
        case rocsparse_datatype_f32_c:
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::spmat_scale_scale_values<rocsparse_float_complex>(
                handle, val_length, alpha_pointer_mode, alpha, target->val_data));
            break;
        case rocsparse_datatype_f64_c:
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::spmat_scale_scale_values<rocsparse_double_complex>(
                handle, val_length, alpha_pointer_mode, alpha, target->val_data));
            break;
        default:
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }

        return rocsparse_status_success;
    }
}

extern "C" rocsparse_status rocsparse_spmat_scale(rocsparse_handle            handle,
                                                  rocsparse_const_dnvec_descr alpha,
                                                  rocsparse_const_spmat_descr source,
                                                  rocsparse_spmat_descr       target,
                                                  rocsparse_error*            p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // p_error is reserved for forward compatibility and is not populated yet.
    (void)p_error;

    // Argument positions in this routine's signature.
    static constexpr int arg_alpha  = 1;
    static constexpr int arg_source = 2;
    static constexpr int arg_target = 3;

    ROCSPARSE_CHECKARG_POINTER(arg_alpha, alpha);

    const rocsparse_status status
        = rocsparse::spmat_scale_checkarg(handle, source, target, arg_source, arg_target);
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    // alpha is a single scalar dense vector; its data type must match the matrices.
    ROCSPARSE_CHECKARG(arg_alpha,
                       alpha,
                       (alpha->size != 1 || alpha->batch_count != 1),
                       rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(
        arg_alpha, alpha, (alpha->data_type != source->data_type), rocsparse_status_type_mismatch);
    ROCSPARSE_CHECKARG_POINTER(arg_alpha, alpha->const_values);

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

#endif /* ROCSPARSE_WITH_SPMAT_SCALE */

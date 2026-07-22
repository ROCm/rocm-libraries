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

    static rocsparse_status spscale_checkarg(rocsparse_handle            handle,
                                             const void*                 alpha,
                                             rocsparse_const_spmat_descr mat_A,
                                             rocsparse_spmat_descr       mat_C)
    {
        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_POINTER(1, alpha);
        ROCSPARSE_CHECKARG_POINTER(2, mat_A);
        ROCSPARSE_CHECKARG_POINTER(3, mat_C);
        ROCSPARSE_CHECKARG(2, mat_A, (mat_A->init == false), rocsparse_status_not_initialized);
        ROCSPARSE_CHECKARG(3, mat_C, (mat_C->init == false), rocsparse_status_not_initialized);

        // Currently only CSR format is supported.
        ROCSPARSE_CHECKARG(
            2, mat_A, (mat_A->format != rocsparse_format_csr), rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(
            3, mat_C, (mat_C->format != rocsparse_format_csr), rocsparse_status_not_implemented);

        // A and C must have matching shape, nonzero count and types.
        ROCSPARSE_CHECKARG(3, mat_C, (mat_C->rows != mat_A->rows), rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG(3, mat_C, (mat_C->cols != mat_A->cols), rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG(3, mat_C, (mat_C->nnz != mat_A->nnz), rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG(
            3, mat_C, (mat_C->row_type != mat_A->row_type), rocsparse_status_type_mismatch);
        ROCSPARSE_CHECKARG(
            3, mat_C, (mat_C->col_type != mat_A->col_type), rocsparse_status_type_mismatch);
        ROCSPARSE_CHECKARG(
            3, mat_C, (mat_C->data_type != mat_A->data_type), rocsparse_status_type_mismatch);

        // Batched matrices are not supported.
        ROCSPARSE_CHECKARG(2, mat_A, (mat_A->batch_count != 1), rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(3, mat_C, (mat_C->batch_count != 1), rocsparse_status_not_implemented);

        // Differing index base between A and C is not yet supported (base conversion follow-up).
        ROCSPARSE_CHECKARG(
            3, mat_C, (mat_C->idx_base != mat_A->idx_base), rocsparse_status_not_implemented);

        return rocsparse_status_continue;
    }

    static rocsparse_status spscale_core(rocsparse_handle            handle,
                                         const void*                 alpha,
                                         rocsparse_const_spmat_descr mat_A,
                                         rocsparse_spmat_descr       mat_C)
    {
        const int64_t m   = mat_A->rows;
        const int64_t nnz = mat_A->nnz;

        const size_t row_bytes
            = static_cast<size_t>(m + 1) * rocsparse::indextype_sizeof(mat_A->row_type);
        const size_t col_bytes
            = static_cast<size_t>(nnz) * rocsparse::indextype_sizeof(mat_A->col_type);
        const size_t val_bytes
            = static_cast<size_t>(nnz) * rocsparse::datatype_sizeof(mat_A->data_type);

        // Copy the sparsity structure of A into C. The index base of A and C is guaranteed to
        // match by the argument checks, so a plain device-to-device copy is sufficient.
        if(mat_C->row_data != nullptr && mat_A->const_row_data != nullptr)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(mat_C->row_data,
                                               mat_A->const_row_data,
                                               row_bytes,
                                               hipMemcpyDeviceToDevice,
                                               handle->stream));
        }

        if(nnz > 0)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(mat_C->col_data,
                                               mat_A->const_col_data,
                                               col_bytes,
                                               hipMemcpyDeviceToDevice,
                                               handle->stream));
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(mat_C->val_data,
                                               mat_A->const_val_data,
                                               val_bytes,
                                               hipMemcpyDeviceToDevice,
                                               handle->stream));

            // Scale the copied values by alpha.
            switch(mat_A->data_type)
            {
            case rocsparse_datatype_f32_r:
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::spscale_scale_values<float>(handle, nnz, alpha, mat_C->val_data));
                break;
            case rocsparse_datatype_f64_r:
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::spscale_scale_values<double>(handle, nnz, alpha, mat_C->val_data));
                break;
            case rocsparse_datatype_f32_c:
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::spscale_scale_values<rocsparse_float_complex>(
                    handle, nnz, alpha, mat_C->val_data));
                break;
            case rocsparse_datatype_f64_c:
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::spscale_scale_values<rocsparse_double_complex>(
                    handle, nnz, alpha, mat_C->val_data));
                break;
            default:
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
        }

        return rocsparse_status_success;
    }
}

extern "C" rocsparse_status rocsparse_spscale_buffer_size(rocsparse_handle            handle,
                                                          const void*                 alpha,
                                                          rocsparse_const_spmat_descr mat_A,
                                                          rocsparse_spmat_descr       mat_C,
                                                          size_t*                     buffer_size)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    rocsparse::log_trace("rocsparse_spscale_buffer_size", handle, alpha, mat_A, mat_C, buffer_size);

    const rocsparse_status status = rocsparse::spscale_checkarg(handle, alpha, mat_A, mat_C);
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG_POINTER(4, buffer_size);

    // The CSR scaling does not require any additional workspace.
    *buffer_size = 0;

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
                                              size_t                      buffer_size,
                                              void*                       temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    rocsparse::log_trace(
        "rocsparse_spscale", handle, alpha, mat_A, mat_C, buffer_size, temp_buffer);

    // No workspace is required for the currently supported formats.
    (void)buffer_size;
    (void)temp_buffer;

    const rocsparse_status status = rocsparse::spscale_checkarg(handle, alpha, mat_A, mat_C);
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

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

#include "internal/conversion/rocsparse_hyb2csr.h"
#include "internal/conversion/rocsparse_coo2csr.h"
#include "rocsparse_control.hpp"
#include "rocsparse_hyb2csr.hpp"
#include "rocsparse_utility.hpp"

#include "hyb2csr_device.h"
#include "rocsparse_common.h"
#include "rocsparse_primitives.hpp"

namespace rocsparse
{
    static rocsparse_status hyb2csr_quickreturn(rocsparse_handle          handle,
                                                const rocsparse_mat_descr descr,
                                                const rocsparse_hyb_mat   hyb,
                                                void*                     csr_val,
                                                rocsparse_int*            csr_row_ptr,
                                                rocsparse_int*            csr_col_ind,
                                                void*                     temp_buffer)

    {
        ROCSPARSE_ROUTINE_TRACE;

        // Quick return if possible
        if(hyb->m == 0 || hyb->n == 0 || (hyb->ell_nnz == 0 && hyb->coo_nnz == 0))
        {
            if(hyb->m >= 0 && csr_row_ptr != nullptr)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::valset(
                    handle, hyb->m + 1, static_cast<rocsparse_int>(descr->base), csr_row_ptr));
            }

            return rocsparse_status_success;
        }
        return rocsparse_status_continue;
    }
}

template <typename T>
rocsparse_status rocsparse::hyb2csr_template(rocsparse_handle          handle,
                                             const rocsparse_mat_descr descr,
                                             const rocsparse_hyb_mat   hyb,
                                             T*                        csr_val,
                                             rocsparse_int*            csr_row_ptr,
                                             rocsparse_int*            csr_col_ind,
                                             void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xhyb2csr"),
                         (const void*&)descr,
                         (const void*&)hyb,
                         (const void*&)csr_val,
                         (const void*&)csr_row_ptr,
                         (const void*&)csr_col_ind,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    ROCSPARSE_CHECKARG(
        1, descr, (descr->type != rocsparse_matrix_type_general), rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(1,
                       descr,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);
    ROCSPARSE_CHECKARG_POINTER(2, hyb);
    ROCSPARSE_CHECKARG(2, hyb, (hyb->m < 0), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG(2, hyb, (hyb->n < 0), rocsparse_status_invalid_size);

    const rocsparse_status status = rocsparse::hyb2csr_quickreturn(
        handle, descr, hyb, csr_val, csr_row_ptr, csr_col_ind, temp_buffer);

    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG_POINTER(3, csr_val);
    ROCSPARSE_CHECKARG_POINTER(4, csr_row_ptr);
    ROCSPARSE_CHECKARG_POINTER(5, csr_col_ind);
    ROCSPARSE_CHECKARG_POINTER(6, temp_buffer);

    // Stream
    hipStream_t stream = handle->stream;

    // Temporary storage buffer
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    // COO row pointer buffer
    rocsparse_int* workspace = reinterpret_cast<rocsparse_int*>(ptr);

    // Get row offset pointers from COO part
    if(hyb->coo_nnz > 0)
    {
        // Shift ptr by workspace size
        ptr += ((sizeof(rocsparse_int) * hyb->m) / 256 + 1) * 256;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_coo2csr(
            handle, hyb->coo_row_ind, hyb->coo_nnz, hyb->m, workspace, descr->base));
    }

    // Compute row pointers
#define HYB2CSR_DIM 256
    dim3 hyb2csr_blocks((hyb->m - 1) / HYB2CSR_DIM + 1);
    dim3 hyb2csr_threads(HYB2CSR_DIM);

    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::hyb2csr_nnz_kernel<HYB2CSR_DIM>),
                                       hyb2csr_blocks,
                                       hyb2csr_threads,
                                       0,
                                       stream,
                                       hyb->m,
                                       hyb->n,
                                       hyb->ell_nnz,
                                       hyb->ell_width,
                                       hyb->ell_col_ind,
                                       hyb->coo_nnz,
                                       workspace,
                                       csr_row_ptr,
                                       descr->base);

    // Exclusive sum to obtain csr_row_ptr array
    size_t rocprim_size;
    void*  rocprim_buffer = reinterpret_cast<void*>(ptr);

    RETURN_IF_ROCSPARSE_ERROR(
        (rocsparse::primitives::exclusive_scan_buffer_size<rocsparse_int, rocsparse_int>(
            handle, static_cast<rocsparse_int>(descr->base), hyb->m + 1, &rocprim_size)));
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::primitives::exclusive_scan(handle,
                                              csr_row_ptr,
                                              csr_row_ptr,
                                              static_cast<rocsparse_int>(descr->base),
                                              hyb->m + 1,
                                              rocprim_size,
                                              rocprim_buffer));

    // Fill columns and values
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::hyb2csr_fill_kernel<HYB2CSR_DIM>),
                                       hyb2csr_blocks,
                                       hyb2csr_threads,
                                       0,
                                       stream,
                                       hyb->m,
                                       hyb->n,
                                       hyb->ell_nnz,
                                       hyb->ell_width,
                                       hyb->ell_col_ind,
                                       reinterpret_cast<T*>(hyb->ell_val),
                                       hyb->coo_nnz,
                                       workspace,
                                       hyb->coo_col_ind,
                                       reinterpret_cast<T*>(hyb->coo_val),
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       descr->base);
#undef HYB2CSR_DIM

    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

static rocsparse_status rocsparse_hyb2csr_buffer_size_quickreturn(rocsparse_handle          handle,
                                                                  const rocsparse_mat_descr descr,
                                                                  const rocsparse_hyb_mat   hyb,
                                                                  const void* csr_row_ptr,
                                                                  size_t*     buffer_size)

{
    ROCSPARSE_ROUTINE_TRACE;

    if(hyb->m == 0 || hyb->n == 0 || (hyb->ell_nnz == 0 && hyb->coo_nnz == 0))
    {
        *buffer_size = 0;
        return rocsparse_status_success;
    }
    return rocsparse_status_continue;
}

extern "C" rocsparse_status rocsparse_hyb2csr_buffer_size(rocsparse_handle          handle,
                                                          const rocsparse_mat_descr descr,
                                                          const rocsparse_hyb_mat   hyb,
                                                          const rocsparse_int*      csr_row_ptr,
                                                          size_t*                   buffer_size)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_hyb2csr_buffer_size",
                         (const void*&)descr,
                         (const void*&)hyb,
                         (const void*&)csr_row_ptr,
                         (const void*&)buffer_size);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    ROCSPARSE_CHECKARG(
        1, descr, (descr->type != rocsparse_matrix_type_general), rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(1,
                       descr,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);
    ROCSPARSE_CHECKARG_POINTER(2, hyb);
    ROCSPARSE_CHECKARG(2, hyb, (hyb->m < 0), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG(2, hyb, (hyb->n < 0), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_POINTER(4, buffer_size);

    const rocsparse_status status
        = rocsparse_hyb2csr_buffer_size_quickreturn(handle, descr, hyb, csr_row_ptr, buffer_size);

    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG_POINTER(3, csr_row_ptr);

    // Initialize buffer size
    *buffer_size = 0;

    // COO part requires conversion buffer
    if(hyb->coo_nnz > 0)
    {
        *buffer_size += ((sizeof(rocsparse_int) * hyb->m) / 256 + 1) * 256;
    }

    // Exclusive scan
    size_t rocprim_size;
    RETURN_IF_ROCSPARSE_ERROR(
        (rocsparse::primitives::exclusive_scan_buffer_size<rocsparse_int, rocsparse_int>(
            handle, static_cast<rocsparse_int>(descr->base), hyb->m + 1, &rocprim_size)));

    // rocprim buffer
    *buffer_size += rocprim_size;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_shyb2csr(rocsparse_handle          handle,
                                               const rocsparse_mat_descr descr,
                                               const rocsparse_hyb_mat   hyb,
                                               float*                    csr_val,
                                               rocsparse_int*            csr_row_ptr,
                                               rocsparse_int*            csr_col_ind,
                                               void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::hyb2csr_template(
        handle, descr, hyb, csr_val, csr_row_ptr, csr_col_ind, temp_buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_dhyb2csr(rocsparse_handle          handle,
                                               const rocsparse_mat_descr descr,
                                               const rocsparse_hyb_mat   hyb,
                                               double*                   csr_val,
                                               rocsparse_int*            csr_row_ptr,
                                               rocsparse_int*            csr_col_ind,
                                               void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::hyb2csr_template(
        handle, descr, hyb, csr_val, csr_row_ptr, csr_col_ind, temp_buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_chyb2csr(rocsparse_handle          handle,
                                               const rocsparse_mat_descr descr,
                                               const rocsparse_hyb_mat   hyb,
                                               rocsparse_float_complex*  csr_val,
                                               rocsparse_int*            csr_row_ptr,
                                               rocsparse_int*            csr_col_ind,
                                               void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::hyb2csr_template(
        handle, descr, hyb, csr_val, csr_row_ptr, csr_col_ind, temp_buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_zhyb2csr(rocsparse_handle          handle,
                                               const rocsparse_mat_descr descr,
                                               const rocsparse_hyb_mat   hyb,
                                               rocsparse_double_complex* csr_val,
                                               rocsparse_int*            csr_row_ptr,
                                               rocsparse_int*            csr_col_ind,
                                               void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::hyb2csr_template(
        handle, descr, hyb, csr_val, csr_row_ptr, csr_col_ind, temp_buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

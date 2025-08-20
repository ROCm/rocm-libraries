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

#include "internal/conversion/rocsparse_csr2hyb.h"
#include "rocsparse_control.hpp"
#include "rocsparse_csr2hyb.hpp"
#include "rocsparse_utility.hpp"

#include "csr2ell_device.h"
#include "csr2hyb_device.h"

#include "rocsparse_primitives.hpp"

namespace rocsparse
{
    template <typename T>
    static rocsparse_status clear_hyb_mat(rocsparse_handle        handle,
                                          rocsparse_int           m,
                                          rocsparse_int           n,
                                          rocsparse_hyb_mat       hyb,
                                          rocsparse_hyb_partition partition_type)
    {
        ROCSPARSE_ROUTINE_TRACE;

        hyb->m         = m;
        hyb->n         = n;
        hyb->partition = partition_type;
        hyb->ell_nnz   = 0;
        hyb->ell_width = 0;
        hyb->coo_nnz   = 0;

        if(std::is_same<T, float>{})
        {
            hyb->data_type_T = rocsparse_datatype_f32_r;
        }
        else if(std::is_same<T, double>{})
        {
            hyb->data_type_T = rocsparse_datatype_f64_r;
        }
        else if(std::is_same<T, rocsparse_float_complex>{})
        {
            hyb->data_type_T = rocsparse_datatype_f32_c;
        }
        else if(std::is_same<T, rocsparse_double_complex>{})
        {
            hyb->data_type_T = rocsparse_datatype_f64_c;
        }

        if(hyb->ell_col_ind)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(hyb->ell_col_ind, handle->stream));
        }
        if(hyb->ell_val)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(hyb->ell_val, handle->stream));
        }
        if(hyb->coo_row_ind)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(hyb->coo_row_ind, handle->stream));
        }
        if(hyb->coo_col_ind)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(hyb->coo_col_ind, handle->stream));
        }
        if(hyb->coo_val)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(hyb->coo_val, handle->stream));
        }

        return rocsparse_status_success;
    }
}

template <typename T>
rocsparse_status rocsparse::csr2hyb_template(rocsparse_handle          handle,
                                             rocsparse_int             m,
                                             rocsparse_int             n,
                                             const rocsparse_mat_descr descr,
                                             const T*                  csr_val,
                                             const rocsparse_int*      csr_row_ptr,
                                             const rocsparse_int*      csr_col_ind,
                                             rocsparse_hyb_mat         hyb,
                                             rocsparse_int             user_ell_width,
                                             rocsparse_hyb_partition   partition_type)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check for valid handle and matrix descriptor
    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xcsr2hyb"),
                         m,
                         n,
                         (const void*&)descr,
                         (const void*&)csr_val,
                         (const void*&)csr_row_ptr,
                         (const void*&)csr_col_ind,
                         (const void*&)hyb,
                         user_ell_width,
                         partition_type);

    // Check matrix type

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG_POINTER(3, descr);
    ROCSPARSE_CHECKARG(
        3, descr, (descr->type != rocsparse_matrix_type_general), rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(3,
                       descr,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);
    ROCSPARSE_CHECKARG_POINTER(7, hyb);
    ROCSPARSE_CHECKARG_ENUM(9, partition_type);

    // Quick return if possible
    if(m == 0 || n == 0)
    {
        // Clear HYB structure if already allocated
        RETURN_IF_ROCSPARSE_ERROR(clear_hyb_mat<T>(handle, m, n, hyb, partition_type));

        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG_POINTER(4, csr_val);
    ROCSPARSE_CHECKARG_POINTER(6, csr_col_ind);
    ROCSPARSE_CHECKARG_ARRAY(5, m, csr_row_ptr);

    // Stream
    hipStream_t stream = handle->stream;

    // Get number of CSR non-zeros
    rocsparse_int csr_nnz;
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        &csr_nnz, csr_row_ptr + m, sizeof(rocsparse_int), hipMemcpyDeviceToHost, stream));

    // Wait for host transfer to finish
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    // Correct by index base
    csr_nnz -= descr->base;

    // Maximum ELL row width allowed
    rocsparse_int max_row_nnz = 2 * (csr_nnz - 1) / m + 1;

    // Check user_ell_width
    if(partition_type == rocsparse_hyb_partition_user)
    {
        // ELL width cannot be 0 or negative
        if(user_ell_width < 0)
        {
            // LCOV_EXCL_START
            return rocsparse_status_invalid_value;
            // LCOV_EXCL_STOP
        }

        if(user_ell_width > max_row_nnz)
        {
            // LCOV_EXCL_START
            return rocsparse_status_invalid_value;
            // LCOV_EXCL_STOP
        }
    }

    // Clear HYB structure if already allocated
    RETURN_IF_ROCSPARSE_ERROR(clear_hyb_mat<T>(handle, m, n, hyb, partition_type));

    // Determine ELL width

#define CSR2ELL_DIM 512
    // Workspace size
    rocsparse_int blocks = (m - 1) / CSR2ELL_DIM + 1;

    if(partition_type == rocsparse_hyb_partition_user)
    {
        // ELL width given by user
        hyb->ell_width = user_ell_width;
    }
    else if(partition_type == rocsparse_hyb_partition_auto)
    {
        // ELL width determined by average nnz per row
        hyb->ell_width = (csr_nnz - 1) / m + 1;
    }
    else
    {
        // Allocate workspace
        rocsparse_int* workspace = nullptr;
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMallocAsync(&workspace, sizeof(rocsparse_int) * blocks, handle->stream));

        // HYB == ELL - no COO part - compute maximum nnz per row
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::ell_width_kernel_part1<CSR2ELL_DIM>),
                                           dim3(blocks),
                                           dim3(CSR2ELL_DIM),
                                           0,
                                           stream,
                                           m,
                                           csr_row_ptr,
                                           workspace);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::ell_width_kernel_part2<CSR2ELL_DIM>),
                                           dim3(1),
                                           dim3(CSR2ELL_DIM),
                                           0,
                                           stream,
                                           blocks,
                                           workspace);
        // Copy ell width back to host
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &hyb->ell_width, workspace, sizeof(rocsparse_int), hipMemcpyDeviceToHost, stream));

        // Wait for host transfer to finish
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(workspace, handle->stream));
    }

    // Re-check ELL width
    if(hyb->ell_width > max_row_nnz)
    {
        // LCOV_EXCL_START
        return rocsparse_status_invalid_value;
        // LCOV_EXCL_STOP
    }

    // Compute ELL non-zeros
    hyb->ell_nnz = hyb->ell_width * m;

    // Allocate ELL part
    if(hyb->ell_nnz > 0)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(
            &hyb->ell_col_ind, sizeof(rocsparse_int) * hyb->ell_nnz, handle->stream));
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMallocAsync(&hyb->ell_val, sizeof(T) * hyb->ell_nnz, handle->stream));
    }

    // Allocate workspace
    rocsparse_int* workspace = NULL;
    RETURN_IF_HIP_ERROR(
        rocsparse_hipMallocAsync(&workspace, sizeof(rocsparse_int) * (m + 1), handle->stream));

    // If there is a COO part, compute the COO non-zero elements per row
    if(partition_type != rocsparse_hyb_partition_max)
    {
        // If there is no ELL part, its easy...
        if(hyb->ell_nnz == 0)
        {
            hyb->coo_nnz = csr_nnz;
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(workspace,
                                               csr_row_ptr,
                                               sizeof(rocsparse_int) * (m + 1),
                                               hipMemcpyDeviceToDevice,
                                               stream));
        }
        else
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::hyb_coo_nnz<CSR2ELL_DIM>),
                                               dim3((m - 1) / CSR2ELL_DIM + 1),
                                               dim3(CSR2ELL_DIM),
                                               0,
                                               stream,
                                               m,
                                               hyb->ell_width,
                                               csr_row_ptr,
                                               workspace,
                                               descr->base);

            // Inclusive sum on workspace
            void*  d_temp_storage     = nullptr;
            size_t temp_storage_bytes = 0;

            // Obtain rocprim buffer size
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::primitives::inclusive_scan_buffer_size<rocsparse_int, rocsparse_int>(
                    handle, m + 1, &temp_storage_bytes)));

            // Allocate rocprim buffer
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMallocAsync(&d_temp_storage, temp_storage_bytes, handle->stream));

            // Do inclusive sum
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::inclusive_scan(
                handle, workspace, workspace, m + 1, temp_storage_bytes, d_temp_storage));

            // Clear rocprim buffer
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(d_temp_storage, handle->stream));

            // Obtain coo nnz from workspace
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&hyb->coo_nnz,
                                               workspace + m,
                                               sizeof(rocsparse_int),
                                               hipMemcpyDeviceToHost,
                                               stream));

            // Wait for host transfer to finish
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

            hyb->coo_nnz -= descr->base;
        }
    }

    // Allocate COO part
    if(hyb->coo_nnz > 0)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(
            &hyb->coo_row_ind, sizeof(rocsparse_int) * hyb->coo_nnz, handle->stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(
            &hyb->coo_col_ind, sizeof(rocsparse_int) * hyb->coo_nnz, handle->stream));
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMallocAsync(&hyb->coo_val, sizeof(T) * hyb->coo_nnz, handle->stream));
    }

    dim3 csr2ell_blocks((m - 1) / CSR2ELL_DIM + 1);
    dim3 csr2ell_threads(CSR2ELL_DIM);

    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csr2hyb_kernel<CSR2ELL_DIM>),
                                       csr2ell_blocks,
                                       csr2ell_threads,
                                       0,
                                       stream,
                                       m,
                                       csr_val,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       hyb->ell_width,
                                       hyb->ell_col_ind,
                                       (T*)hyb->ell_val,
                                       hyb->coo_row_ind,
                                       hyb->coo_col_ind,
                                       (T*)hyb->coo_val,
                                       workspace,
                                       descr->base);

    RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(workspace, handle->stream));
#undef CSR2ELL_DIM

    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_scsr2hyb(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               rocsparse_int             n,
                                               const rocsparse_mat_descr descr,
                                               const float*              csr_val,
                                               const rocsparse_int*      csr_row_ptr,
                                               const rocsparse_int*      csr_col_ind,
                                               rocsparse_hyb_mat         hyb,
                                               rocsparse_int             user_ell_width,
                                               rocsparse_hyb_partition   partition_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2hyb_template(handle,
                                                          m,
                                                          n,
                                                          descr,
                                                          csr_val,
                                                          csr_row_ptr,
                                                          csr_col_ind,
                                                          hyb,
                                                          user_ell_width,
                                                          partition_type));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_dcsr2hyb(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               rocsparse_int             n,
                                               const rocsparse_mat_descr descr,
                                               const double*             csr_val,
                                               const rocsparse_int*      csr_row_ptr,
                                               const rocsparse_int*      csr_col_ind,
                                               rocsparse_hyb_mat         hyb,
                                               rocsparse_int             user_ell_width,
                                               rocsparse_hyb_partition   partition_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2hyb_template(handle,
                                                          m,
                                                          n,
                                                          descr,
                                                          csr_val,
                                                          csr_row_ptr,
                                                          csr_col_ind,
                                                          hyb,
                                                          user_ell_width,
                                                          partition_type));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_ccsr2hyb(rocsparse_handle               handle,
                                               rocsparse_int                  m,
                                               rocsparse_int                  n,
                                               const rocsparse_mat_descr      descr,
                                               const rocsparse_float_complex* csr_val,
                                               const rocsparse_int*           csr_row_ptr,
                                               const rocsparse_int*           csr_col_ind,
                                               rocsparse_hyb_mat              hyb,
                                               rocsparse_int                  user_ell_width,
                                               rocsparse_hyb_partition        partition_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2hyb_template(handle,
                                                          m,
                                                          n,
                                                          descr,
                                                          csr_val,
                                                          csr_row_ptr,
                                                          csr_col_ind,
                                                          hyb,
                                                          user_ell_width,
                                                          partition_type));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_zcsr2hyb(rocsparse_handle                handle,
                                               rocsparse_int                   m,
                                               rocsparse_int                   n,
                                               const rocsparse_mat_descr       descr,
                                               const rocsparse_double_complex* csr_val,
                                               const rocsparse_int*            csr_row_ptr,
                                               const rocsparse_int*            csr_col_ind,
                                               rocsparse_hyb_mat               hyb,
                                               rocsparse_int                   user_ell_width,
                                               rocsparse_hyb_partition         partition_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2hyb_template(handle,
                                                          m,
                                                          n,
                                                          descr,
                                                          csr_val,
                                                          csr_row_ptr,
                                                          csr_col_ind,
                                                          hyb,
                                                          user_ell_width,
                                                          partition_type));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

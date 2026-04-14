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
#pragma once

#include "rocsparse_utility.hpp"

#include "rocsparse_common.h"
#include "rocsparse_control.hpp"
#include "rocsparse_dense2csx.hpp"
#include "rocsparse_primitives.hpp"

namespace rocsparse
{
    template <rocsparse_direction DIRA, typename I, typename J, typename T>
    rocsparse_status csx2dense_checkarg(rocsparse_handle          handle, //0
                                        J                         m, //1
                                        J                         n, //2
                                        const rocsparse_mat_descr descr, //3
                                        const T*                  A, //4
                                        int64_t                   lda, //5
                                        const I*                  nnz_per_row_column, //6
                                        T*                        csx_val_A, //7
                                        I*                        csx_row_col_ptr_A, //8
                                        J*                        csx_col_row_ind_A, //9
                                        rocsparse_order           order) //10
    {
        ROCSPARSE_ROUTINE_TRACE;

        static constexpr bool is_row_oriented = (rocsparse_direction_row == DIRA);

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_SIZE(1, m);
        ROCSPARSE_CHECKARG_SIZE(2, n);
        ROCSPARSE_CHECKARG(5,
                           lda,
                           (lda < (order == rocsparse_order_column ? m : n)),
                           rocsparse_status_invalid_size);

        if(m == 0 || n == 0)
        {
            if(csx_row_col_ptr_A != nullptr)
            {
                J dimdir = is_row_oriented ? m : n;

                RETURN_IF_ROCSPARSE_ERROR(rocsparse::valset(
                    handle, dimdir + 1, static_cast<I>(descr->base), csx_row_col_ptr_A));
            }

            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_POINTER(3, descr);
        ROCSPARSE_CHECKARG(3,
                           descr,
                           (descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);
        ROCSPARSE_CHECKARG(3,
                           descr,
                           (rocsparse_matrix_type_general != descr->type),
                           rocsparse_status_not_implemented);

        const size_t nnz = size_t(m) * n;
        ROCSPARSE_CHECKARG_ARRAY(4, nnz, A);

        switch(DIRA)
        {
        case rocsparse_direction_row:
        {
            const T* csr_val      = csx_val_A;
            const I* csr_row_ptr  = csx_row_col_ptr_A;
            const J* csr_col_ind  = csx_col_row_ind_A;
            const I* nnz_per_rows = nnz_per_row_column;

            ROCSPARSE_CHECKARG_ARRAY(6, m, nnz_per_rows);
            ROCSPARSE_CHECKARG_ARRAY(7, nnz, csr_val);
            ROCSPARSE_CHECKARG_ARRAY(8, m, csr_row_ptr);
            ROCSPARSE_CHECKARG_ARRAY(9, nnz, csr_col_ind);
            break;
        }
        case rocsparse_direction_column:
        {
            const T* csc_val         = csx_val_A;
            const I* csc_col_ptr     = csx_row_col_ptr_A;
            const J* csc_row_ind     = csx_col_row_ind_A;
            const I* nnz_per_columns = nnz_per_row_column;

            ROCSPARSE_CHECKARG_ARRAY(6, n, nnz_per_columns);
            ROCSPARSE_CHECKARG_ARRAY(7, nnz, csc_val);
            ROCSPARSE_CHECKARG_ARRAY(8, n, csc_col_ptr);
            ROCSPARSE_CHECKARG_ARRAY(9, nnz, csc_row_ind);
            break;
        }
        }

        return rocsparse_status_continue;
    }

    template <rocsparse_direction DIRA, typename I, typename J, typename T>
    rocsparse_status dense2csx_impl(rocsparse_handle          handle,
                                    rocsparse_order           order,
                                    J                         m,
                                    J                         n,
                                    const rocsparse_mat_descr descr,
                                    const T*                  A,
                                    int64_t                   lda,
                                    const I*                  nnz_per_row_column,
                                    T*                        csx_val_A,
                                    I*                        csx_row_col_ptr_A,
                                    J*                        csx_col_row_ind_A)
    {
        ROCSPARSE_ROUTINE_TRACE;

        static constexpr bool is_row_oriented = (rocsparse_direction_row == DIRA);

        //
        // Loggings
        //
        rocsparse::log_trace(handle,
                             is_row_oriented ? "rocsparse_dense2csr" : "rocsparse_dense2csc",
                             m,
                             n,
                             descr,
                             (const void*&)A,
                             lda,
                             (const void*&)nnz_per_row_column,
                             (const void*&)csx_val_A,
                             (const void*&)csx_row_col_ptr_A,
                             (const void*&)csx_col_row_ind_A);

        const rocsparse_status status = rocsparse::csx2dense_checkarg<DIRA>(handle,
                                                                            m,
                                                                            n,
                                                                            descr,
                                                                            A,
                                                                            lda,
                                                                            nnz_per_row_column,
                                                                            csx_val_A,
                                                                            csx_row_col_ptr_A,
                                                                            csx_col_row_ind_A,
                                                                            order);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        //
        // Compute csx_row_col_ptr_A with the right index base.
        //
        {
            J dimdir = is_row_oriented ? m : n;

            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                csx_row_col_ptr_A, &descr->base, sizeof(I), hipMemcpyHostToDevice, handle->stream));
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(csx_row_col_ptr_A + 1,
                                               nnz_per_row_column,
                                               sizeof(I) * dimdir,
                                               hipMemcpyDeviceToDevice,
                                               handle->stream));

            size_t temp_storage_bytes = 0;
            // Obtain rocprim buffer size
            RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::inclusive_scan_buffer_size<I, I>(
                handle, dimdir + 1, &temp_storage_bytes)));

            // Get rocprim buffer
            bool  d_temp_alloc;
            void* d_temp_storage;

            // Device buffer should be sufficient for rocprim in most cases
            if(handle->buffer_size >= temp_storage_bytes)
            {
                d_temp_storage = handle->buffer;
                d_temp_alloc   = false;
            }
            else
            {
                RETURN_IF_HIP_ERROR(
                    rocsparse_hipMallocAsync(&d_temp_storage, temp_storage_bytes, handle->stream));
                d_temp_alloc = true;
            }

            // Perform actual inclusive sum
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::inclusive_scan(handle,
                                                                            csx_row_col_ptr_A,
                                                                            csx_row_col_ptr_A,
                                                                            dimdir + 1,
                                                                            temp_storage_bytes,
                                                                            d_temp_storage));

            // Free rocprim buffer, if allocated
            if(d_temp_alloc == true)
            {
                RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(d_temp_storage, handle->stream));
            }
        }

        if(csx_col_row_ind_A == nullptr && csx_val_A == nullptr)
        {
            I start = 0;
            I end   = 0;

            if(is_row_oriented)
            {
                RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                    &end, &csx_row_col_ptr_A[m], sizeof(I), hipMemcpyDeviceToHost, handle->stream));
                RETURN_IF_HIP_ERROR(hipMemcpyAsync(&start,
                                                   &csx_row_col_ptr_A[0],
                                                   sizeof(I),
                                                   hipMemcpyDeviceToHost,
                                                   handle->stream));
            }
            else
            {
                RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                    &end, &csx_row_col_ptr_A[n], sizeof(I), hipMemcpyDeviceToHost, handle->stream));
                RETURN_IF_HIP_ERROR(hipMemcpyAsync(&start,
                                                   &csx_row_col_ptr_A[0],
                                                   sizeof(I),
                                                   hipMemcpyDeviceToHost,
                                                   handle->stream));
            }

            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

            I nnz = (end - start);

            if(nnz != 0)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
            }
        }

        //
        // Compute csx_val_A csx_col_row_ind_A with right index base and update the 0-based csx_row_col_ptr_A if necessary.
        //
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::dense2csx_template<DIRA>(
            handle, order, m, n, descr, A, lda, csx_val_A, csx_row_col_ptr_A, csx_col_row_ind_A));
        return rocsparse_status_success;
    }
}

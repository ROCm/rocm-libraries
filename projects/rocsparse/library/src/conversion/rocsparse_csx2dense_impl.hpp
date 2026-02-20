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
#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_csx2dense.hpp"

namespace rocsparse
{
    template <rocsparse_direction DIRA, typename I, typename J, typename T>
    rocsparse_status csx2dense_checkarg(rocsparse_handle          handle, //0
                                        J                         m, //1
                                        J                         n, //2
                                        const rocsparse_mat_descr descr, //3
                                        const T*                  csx_val, //4
                                        const I*                  csx_row_col_ptr, //5
                                        const J*                  csx_col_row_ind, //6
                                        T*                        A, //7
                                        int64_t                   lda, //8
                                        rocsparse_order           order) //9
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_POINTER(3, descr);
        ROCSPARSE_CHECKARG_SIZE(1, m);
        ROCSPARSE_CHECKARG_SIZE(2, n);
        ROCSPARSE_CHECKARG_ENUM(9, order);
        ROCSPARSE_CHECKARG(8,
                           lda,
                           (lda < (order == rocsparse_order_column ? m : n)),
                           rocsparse_status_invalid_size);
        if(m == 0 || n == 0)
        {
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG(3,
                           descr,
                           (descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);
        ROCSPARSE_CHECKARG(3,
                           descr,
                           (rocsparse_matrix_type_general != descr->type),
                           rocsparse_status_not_implemented);

        switch(DIRA)
        {
        case rocsparse_direction_row:
        {
            const I* csr_row_ptr = csx_row_col_ptr;
            const T* csr_val     = csx_val;
            const J* csr_col_ind = csx_col_row_ind;
            ROCSPARSE_CHECKARG_ARRAY(5, m, csr_row_ptr);
            if(csr_val == nullptr || csr_col_ind == nullptr)
            {
                rocsparse_int start = 0;
                rocsparse_int end   = 0;
                if(csr_row_ptr != nullptr)
                {
                    RETURN_IF_HIP_ERROR(hipMemcpyAsync(&end,
                                                       &csr_row_ptr[m],
                                                       sizeof(rocsparse_int),
                                                       hipMemcpyDeviceToHost,
                                                       handle->stream));
                    RETURN_IF_HIP_ERROR(hipMemcpyAsync(&start,
                                                       &csr_row_ptr[0],
                                                       sizeof(rocsparse_int),
                                                       hipMemcpyDeviceToHost,
                                                       handle->stream));
                    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
                }
                const rocsparse_int nnz = (end - start);
                ROCSPARSE_CHECKARG_ARRAY(4, nnz, csr_val);
                ROCSPARSE_CHECKARG_ARRAY(6, nnz, csr_col_ind);
            }
            break;
        }
        case rocsparse_direction_column:
        {
            const T* csc_val     = csx_val;
            const I* csc_col_ptr = csx_row_col_ptr;
            const J* csc_row_ind = csx_col_row_ind;
            ROCSPARSE_CHECKARG_ARRAY(5, n, csc_col_ptr);
            if(csc_val == nullptr || csc_row_ind == nullptr)
            {
                rocsparse_int start = 0;
                rocsparse_int end   = 0;
                if(csc_col_ptr != nullptr)
                {
                    RETURN_IF_HIP_ERROR(hipMemcpyAsync(&end,
                                                       &csc_col_ptr[m],
                                                       sizeof(rocsparse_int),
                                                       hipMemcpyDeviceToHost,
                                                       handle->stream));
                    RETURN_IF_HIP_ERROR(hipMemcpyAsync(&start,
                                                       &csc_col_ptr[0],
                                                       sizeof(rocsparse_int),
                                                       hipMemcpyDeviceToHost,
                                                       handle->stream));
                    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
                }
                const rocsparse_int nnz = (end - start);
                ROCSPARSE_CHECKARG_ARRAY(4, nnz, csc_val);
                ROCSPARSE_CHECKARG_ARRAY(6, nnz, csc_row_ind);
            }
            break;
        }
        }

        ROCSPARSE_CHECKARG_ARRAY(7, m * n, A);
        //
        // Quick return if possible, before checking for invalid pointers.
        //
        return rocsparse_status_continue;
    }

    template <rocsparse_direction DIRA, typename I, typename J, typename T>
    rocsparse_status csx2dense_impl(rocsparse_handle          handle, //0
                                    J                         m, //1
                                    J                         n, //2
                                    const rocsparse_mat_descr descr, //3
                                    const T*                  csx_val, //4
                                    const I*                  csx_row_col_ptr, //5
                                    const J*                  csx_col_row_ind, //6
                                    T*                        A, //7
                                    int64_t                   lda, //8
                                    rocsparse_order           order) //9
    {
        ROCSPARSE_ROUTINE_TRACE;

        static constexpr bool is_row_oriented = (rocsparse_direction_row == DIRA);

        rocsparse::log_trace(handle,
                             is_row_oriented ? "rocsparse_csr2dense" : "rocsparse_csc2dense",
                             m,
                             n,
                             descr,
                             (const void*&)A,
                             lda,
                             (const void*&)csx_val,
                             (const void*&)csx_row_col_ptr,
                             (const void*&)csx_col_row_ind);

        const rocsparse_status status = rocsparse::csx2dense_checkarg<DIRA, I, J, T>(
            handle, m, n, descr, csx_val, csx_row_col_ptr, csx_col_row_ind, A, lda, order);

        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        // Note: hipMemset2DAsync does not seem to be supported by hipgraph but should be in the future.
        // Once hipgraph supports hipMemset2DAsync then the kernel memset2d_kernel can be replaced
        // with the hipMemset2DAsync call below.
        //
        // const J mn = order == rocsparse_order_column ? m : n;
        // const J nm = order == rocsparse_order_column ? n : m;
        // RETURN_IF_HIP_ERROR(
        //     hipMemset2DAsync(A, sizeof(T) * lda, 0, sizeof(T) * mn, nm, handle->stream));

        // Set memory to zero.
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::valset_2d(
            handle, static_cast<I>(m), static_cast<I>(n), lda, static_cast<T>(0), A, order));

        //
        // Compute the conversion.
        //
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csx2dense_template<DIRA>(
            handle, m, n, descr, csx_val, csx_row_col_ptr, csx_col_row_ind, A, lda, order));
        return rocsparse_status_success;
    }
}

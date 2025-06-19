/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/extra/rocsparse_csrgeam.h"
#include "rocsparse_common.h"
#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_csrgeam.hpp"
#include "rocsparse_primitives.hpp"
#include "rocsparse_utility.hpp"

#include "csrgeam_device.h"

namespace rocsparse
{
    template <typename I, typename J>
    static rocsparse_status csrgeam_nnz_core_with_allocation(rocsparse_handle             handle,
                                                             const rocsparse_spgeam_descr descr,
                                                             rocsparse_operation          trans_A,
                                                             rocsparse_operation          trans_B,
                                                             int64_t                      m,
                                                             int64_t                      n,
                                                             const rocsparse_mat_descr    descr_A,
                                                             int64_t                      nnz_A,
                                                             const I* csr_row_ptr_A,
                                                             const J* csr_col_ind_A,
                                                             const rocsparse_mat_descr descr_B,
                                                             int64_t                   nnz_B,
                                                             const I* csr_row_ptr_B,
                                                             const J* csr_col_ind_B,
                                                             void*    temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse_host_assert(descr != nullptr, "Wrong logical dispatch.");

        // Stream
        hipStream_t stream = handle->stream;

#define CSRGEAM_DIM 256
        if(handle->wavefront_size == 32)
        {
            switch(descr->indextype)
            {
            case rocsparse_indextype_i32:
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csrgeam_nnz_multipass_device<CSRGEAM_DIM, 32>),
                    dim3((m - 1) / (CSRGEAM_DIM / 32) + 1),
                    dim3(CSRGEAM_DIM),
                    0,
                    stream,
                    m,
                    n,
                    csr_row_ptr_A,
                    csr_col_ind_A,
                    csr_row_ptr_B,
                    csr_col_ind_B,
                    static_cast<int32_t*>(descr->csr_row_ptr_C),
                    descr_A->base,
                    descr_B->base);
                break;
            }
            case rocsparse_indextype_i64:
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csrgeam_nnz_multipass_device<CSRGEAM_DIM, 32>),
                    dim3((m - 1) / (CSRGEAM_DIM / 32) + 1),
                    dim3(CSRGEAM_DIM),
                    0,
                    stream,
                    m,
                    n,
                    csr_row_ptr_A,
                    csr_col_ind_A,
                    csr_row_ptr_B,
                    csr_col_ind_B,
                    static_cast<int64_t*>(descr->csr_row_ptr_C),
                    descr_A->base,
                    descr_B->base);
                break;
            }
            default:
            {
                return rocsparse_status_internal_error;
            }
            }
        }
        else
        {
            switch(descr->indextype)
            {
            case rocsparse_indextype_i32:
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csrgeam_nnz_multipass_device<CSRGEAM_DIM, 64>),
                    dim3((m - 1) / (CSRGEAM_DIM / 64) + 1),
                    dim3(CSRGEAM_DIM),
                    0,
                    stream,
                    m,
                    n,
                    csr_row_ptr_A,
                    csr_col_ind_A,
                    csr_row_ptr_B,
                    csr_col_ind_B,
                    static_cast<int32_t*>(descr->csr_row_ptr_C),
                    descr_A->base,
                    descr_B->base);
                break;
            }
            case rocsparse_indextype_i64:
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csrgeam_nnz_multipass_device<CSRGEAM_DIM, 64>),
                    dim3((m - 1) / (CSRGEAM_DIM / 64) + 1),
                    dim3(CSRGEAM_DIM),
                    0,
                    stream,
                    m,
                    n,
                    csr_row_ptr_A,
                    csr_col_ind_A,
                    csr_row_ptr_B,
                    csr_col_ind_B,
                    static_cast<int64_t*>(descr->csr_row_ptr_C),
                    descr_A->base,
                    descr_B->base);
                break;
            }
            default:
            {
                return rocsparse_status_internal_error;
            }
            }
        }
#undef CSRGEAM_DIM

        // Exclusive sum to obtain row pointers of C
        switch(descr->indextype)
        {
        case rocsparse_indextype_i32:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::primitives::exclusive_scan_buffer_size<int32_t, int32_t>(
                    handle,
                    static_cast<int32_t>(rocsparse_index_base_zero),
                    m + 1,
                    &descr->rocprim_size)));
            break;
        }
        case rocsparse_indextype_i64:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::primitives::exclusive_scan_buffer_size<int64_t, int64_t>(
                    handle,
                    static_cast<int64_t>(rocsparse_index_base_zero),
                    m + 1,
                    &descr->rocprim_size)));
            break;
        }
        default:
        {
            return rocsparse_status_internal_error;
        }
        }

        if(handle->buffer_size >= descr->rocprim_size)
        {
            descr->rocprim_buffer = handle->buffer;
            descr->rocprim_alloc  = false;
        }
        else
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(
                &descr->rocprim_buffer, descr->rocprim_size, handle->stream));
            descr->rocprim_alloc = true;
        }

        switch(descr->indextype)
        {
        case rocsparse_indextype_i32:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::exclusive_scan(
                handle,
                static_cast<int32_t*>(descr->csr_row_ptr_C),
                static_cast<int32_t*>(descr->csr_row_ptr_C),
                static_cast<int32_t>(rocsparse_index_base_zero),
                m + 1,
                descr->rocprim_size,
                descr->rocprim_buffer));
            break;
        }
        case rocsparse_indextype_i64:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::exclusive_scan(
                handle,
                static_cast<int64_t*>(descr->csr_row_ptr_C),
                static_cast<int64_t*>(descr->csr_row_ptr_C),
                static_cast<int64_t>(rocsparse_index_base_zero),
                m + 1,
                descr->rocprim_size,
                descr->rocprim_buffer));
            break;
        }
        default:
        {
            return rocsparse_status_internal_error;
        }
        }

        if(descr->rocprim_alloc == true)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(descr->rocprim_buffer, handle->stream));
        }

        // Extract the number of non-zero elements of C
        switch(descr->indextype)
        {
        case rocsparse_indextype_i32:
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&descr->nnz_C,
                                               static_cast<int32_t*>(descr->csr_row_ptr_C) + m,
                                               sizeof(int32_t),
                                               hipMemcpyDeviceToHost,
                                               handle->stream));
            break;
        }
        case rocsparse_indextype_i64:
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&descr->nnz_C,
                                               static_cast<int64_t*>(descr->csr_row_ptr_C) + m,
                                               sizeof(int64_t),
                                               hipMemcpyDeviceToHost,
                                               handle->stream));
            break;
        }
        default:
        {
            return rocsparse_status_internal_error;
        }
        }

        return rocsparse_status_success;
    }

    template <typename I, typename J>
    static rocsparse_status csrgeam_nnz_core_without_allocation(rocsparse_handle             handle,
                                                                const rocsparse_spgeam_descr descr,
                                                                rocsparse_operation       trans_A,
                                                                rocsparse_operation       trans_B,
                                                                int64_t                   m,
                                                                int64_t                   n,
                                                                const rocsparse_mat_descr descr_A,
                                                                int64_t                   nnz_A,
                                                                const I* csr_row_ptr_A,
                                                                const J* csr_col_ind_A,
                                                                const rocsparse_mat_descr descr_B,
                                                                int64_t                   nnz_B,
                                                                const I* csr_row_ptr_B,
                                                                const J* csr_col_ind_B,
                                                                const rocsparse_mat_descr descr_C,
                                                                I*    csr_row_ptr_C,
                                                                I*    nnz_C,
                                                                void* temp_buffer,
                                                                bool  called_from_spgeam)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse_host_assert(csr_row_ptr_C != nullptr, "Wrong logical dispatch.");
        rocsparse_host_assert(nnz_C != nullptr, "Wrong logical dispatch.");

        // Stream
        hipStream_t stream = handle->stream;

#define CSRGEAM_DIM 256
        if(handle->wavefront_size == 32)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::csrgeam_nnz_multipass_device<CSRGEAM_DIM, 32>),
                dim3((m - 1) / (CSRGEAM_DIM / 32) + 1),
                dim3(CSRGEAM_DIM),
                0,
                stream,
                m,
                n,
                csr_row_ptr_A,
                csr_col_ind_A,
                csr_row_ptr_B,
                csr_col_ind_B,
                csr_row_ptr_C,
                descr_A->base,
                descr_B->base);
        }
        else
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::csrgeam_nnz_multipass_device<CSRGEAM_DIM, 64>),
                dim3((m - 1) / (CSRGEAM_DIM / 64) + 1),
                dim3(CSRGEAM_DIM),
                0,
                stream,
                m,
                n,
                csr_row_ptr_A,
                csr_col_ind_A,
                csr_row_ptr_B,
                csr_col_ind_B,
                csr_row_ptr_C,
                descr_A->base,
                descr_B->base);
        }
#undef CSRGEAM_DIM

        // Exclusive sum to obtain row pointers of C
        size_t rocprim_size;
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::exclusive_scan_buffer_size<I, I>(
            handle, static_cast<I>(descr_C->base), m + 1, &rocprim_size)));

        bool  rocprim_alloc;
        void* rocprim_buffer;

        if(handle->buffer_size >= rocprim_size)
        {
            rocprim_buffer = handle->buffer;
            rocprim_alloc  = false;
        }
        else
        {
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMallocAsync(&rocprim_buffer, rocprim_size, handle->stream));
            rocprim_alloc = true;
        }

        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::primitives::exclusive_scan(handle,
                                                  csr_row_ptr_C,
                                                  csr_row_ptr_C,
                                                  static_cast<I>(descr_C->base),
                                                  m + 1,
                                                  rocprim_size,
                                                  rocprim_buffer));

        if(rocprim_alloc == true)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(rocprim_buffer, handle->stream));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
        }

        // Checks the exclusive scan for integer overflow. If overflow detected, sets the
        // last entry in csr_row_ptr_C to -1
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrgeam_check_row_ptr<256>),
                                           dim3(((m + 1) - 1) / 256 + 1),
                                           dim3(256),
                                           0,
                                           stream,
                                           m,
                                           csr_row_ptr_C,
                                           descr_C->base);

        // Extract the number of non-zero elements of C
        if(handle->pointer_mode == rocsparse_pointer_mode_host || called_from_spgeam)
        {
            // Blocking mode
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                nnz_C, csr_row_ptr_C + m, sizeof(I), hipMemcpyDeviceToHost, handle->stream));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

            if(*nnz_C != -1)
            {
                // Adjust index base of nnz_C
                *nnz_C -= descr_C->base;
            }
        }
        else
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                nnz_C, csr_row_ptr_C + m, sizeof(I), hipMemcpyDeviceToDevice, stream));

            // Adjust index base of nnz_C
            if(descr_C->base == rocsparse_index_base_one)
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csrgeam_index_base<1>), dim3(1), dim3(1), 0, stream, nnz_C);
            }
        }

        return rocsparse_status_success;
    }

    template <typename I, typename J>
    static rocsparse_status csrgeam_nnz_core(rocsparse_handle             handle,
                                             const rocsparse_spgeam_descr descr,
                                             rocsparse_operation          trans_A,
                                             rocsparse_operation          trans_B,
                                             int64_t                      m,
                                             int64_t                      n,
                                             const rocsparse_mat_descr    descr_A,
                                             int64_t                      nnz_A,
                                             const I*                     csr_row_ptr_A,
                                             const J*                     csr_col_ind_A,
                                             const rocsparse_mat_descr    descr_B,
                                             int64_t                      nnz_B,
                                             const I*                     csr_row_ptr_B,
                                             const J*                     csr_col_ind_B,
                                             const rocsparse_mat_descr    descr_C,
                                             I*                           csr_row_ptr_C,
                                             I*                           nnz_C,
                                             void*                        temp_buffer,
                                             bool                         called_from_spgeam)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(csr_row_ptr_C == nullptr && nnz_C == nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgeam_nnz_core_with_allocation(handle,
                                                                                  descr,
                                                                                  trans_A,
                                                                                  trans_B,
                                                                                  m,
                                                                                  n,
                                                                                  descr_A,
                                                                                  nnz_A,
                                                                                  csr_row_ptr_A,
                                                                                  csr_col_ind_A,
                                                                                  descr_B,
                                                                                  nnz_B,
                                                                                  csr_row_ptr_B,
                                                                                  csr_col_ind_B,
                                                                                  temp_buffer));

            return rocsparse_status_success;
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::csrgeam_nnz_core_without_allocation(handle,
                                                               descr,
                                                               trans_A,
                                                               trans_B,
                                                               m,
                                                               n,
                                                               descr_A,
                                                               nnz_A,
                                                               csr_row_ptr_A,
                                                               csr_col_ind_A,
                                                               descr_B,
                                                               nnz_B,
                                                               csr_row_ptr_B,
                                                               csr_col_ind_B,
                                                               descr_C,
                                                               csr_row_ptr_C,
                                                               nnz_C,
                                                               temp_buffer,
                                                               called_from_spgeam));

            return rocsparse_status_success;
        }

        return rocsparse_status_internal_error;
    }

    template <typename I>
    static rocsparse_status csrgeam_nnz_quickreturn(rocsparse_handle             handle,
                                                    const rocsparse_spgeam_descr descr,
                                                    rocsparse_operation          trans_A,
                                                    rocsparse_operation          trans_B,
                                                    int64_t                      m,
                                                    int64_t                      n,
                                                    const rocsparse_mat_descr    descr_A,
                                                    int64_t                      nnz_A,
                                                    const void*                  csr_row_ptr_A,
                                                    const void*                  csr_col_ind_A,
                                                    const rocsparse_mat_descr    descr_B,
                                                    int64_t                      nnz_B,
                                                    const void*                  csr_row_ptr_B,
                                                    const void*                  csr_col_ind_B,
                                                    const rocsparse_mat_descr    descr_C,
                                                    I*                           csr_row_ptr_C,
                                                    I*                           nnz_C,
                                                    void*                        temp_buffer,
                                                    bool                         called_from_spgeam)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Quick return if possible
        if(m == 0 || n == 0 || (nnz_A == 0 && nnz_B == 0))
        {
            if(handle->pointer_mode == rocsparse_pointer_mode_host)
            {
                *nnz_C = 0;
            }
            else
            {
                RETURN_IF_HIP_ERROR(hipMemsetAsync(nnz_C, 0, sizeof(I), handle->stream));
            }

            if(nnz_A == 0 && nnz_B == 0)
            {
                if(csr_row_ptr_C != nullptr)
                {
                    RETURN_IF_ROCSPARSE_ERROR(rocsparse::valset(
                        handle, m + 1, static_cast<I>(descr_C->base), csr_row_ptr_C));
                }
            }
            return rocsparse_status_success;
        }

        return rocsparse_status_continue;
    }

    template <typename I>
    static rocsparse_status csrgeam_nnz_checkarg(rocsparse_handle          handle, //0
                                                 int64_t                   m, //1
                                                 int64_t                   n, //2
                                                 const rocsparse_mat_descr descr_A, //3
                                                 int64_t                   nnz_A, //4
                                                 const void*               csr_row_ptr_A, //5
                                                 const void*               csr_col_ind_A, //6
                                                 const rocsparse_mat_descr descr_B, //7
                                                 int64_t                   nnz_B, //8
                                                 const void*               csr_row_ptr_B, //9
                                                 const void*               csr_col_ind_B, //10
                                                 const rocsparse_mat_descr descr_C, //11
                                                 I*                        csr_row_ptr_C, //12
                                                 I*                        nnz_C) //13
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_POINTER(3, descr_A);
        ROCSPARSE_CHECKARG_POINTER(7, descr_B);
        ROCSPARSE_CHECKARG_POINTER(11, descr_C);

        ROCSPARSE_CHECKARG(3,
                           descr_A,
                           (descr_A->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(7,
                           descr_B,
                           (descr_B->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(11,
                           descr_C,
                           (descr_C->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);

        ROCSPARSE_CHECKARG(3,
                           descr_A,
                           (descr_A->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);
        ROCSPARSE_CHECKARG(7,
                           descr_B,
                           (descr_B->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);
        ROCSPARSE_CHECKARG(11,
                           descr_C,
                           (descr_C->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);

        ROCSPARSE_CHECKARG_SIZE(1, m);
        ROCSPARSE_CHECKARG_SIZE(2, n);
        ROCSPARSE_CHECKARG_SIZE(4, nnz_A);
        ROCSPARSE_CHECKARG_SIZE(8, nnz_B);

        ROCSPARSE_CHECKARG_POINTER(13, nnz_C);

        const rocsparse_spgeam_descr descr       = nullptr;
        void*                        temp_buffer = nullptr;
        const rocsparse_status       status      = rocsparse::csrgeam_nnz_quickreturn(handle,
                                                                           descr,
                                                                           rocsparse_operation_none,
                                                                           rocsparse_operation_none,
                                                                           m,
                                                                           n,
                                                                           descr_A,
                                                                           nnz_A,
                                                                           csr_row_ptr_A,
                                                                           csr_col_ind_A,
                                                                           descr_B,
                                                                           nnz_B,
                                                                           csr_row_ptr_B,
                                                                           csr_col_ind_B,
                                                                           descr_C,
                                                                           csr_row_ptr_C,
                                                                           nnz_C,
                                                                           temp_buffer,
                                                                           false);

        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_ARRAY(12, m, csr_row_ptr_C);
        ROCSPARSE_CHECKARG_ARRAY(5, m, csr_row_ptr_A);
        ROCSPARSE_CHECKARG_ARRAY(9, m, csr_row_ptr_B);
        ROCSPARSE_CHECKARG_ARRAY(6, nnz_A, csr_col_ind_A);
        ROCSPARSE_CHECKARG_ARRAY(10, nnz_B, csr_col_ind_B);

        return rocsparse_status_continue;
    }

    static rocsparse_status csrgeam_nnz_impl(rocsparse_handle          handle,
                                             rocsparse_int             m,
                                             rocsparse_int             n,
                                             const rocsparse_mat_descr descr_A,
                                             rocsparse_int             nnz_A,
                                             const rocsparse_int*      csr_row_ptr_A,
                                             const rocsparse_int*      csr_col_ind_A,
                                             const rocsparse_mat_descr descr_B,
                                             rocsparse_int             nnz_B,
                                             const rocsparse_int*      csr_row_ptr_B,
                                             const rocsparse_int*      csr_col_ind_B,
                                             const rocsparse_mat_descr descr_C,
                                             rocsparse_int*            csr_row_ptr_C,
                                             rocsparse_int*            nnz_C)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse::log_trace("rocsparse_csrgeam_nnz",
                             handle,
                             m,
                             n,
                             (const void*&)descr_A,
                             nnz_A,
                             (const void*&)csr_row_ptr_A,
                             (const void*&)csr_col_ind_A,
                             (const void*&)descr_B,
                             nnz_B,
                             (const void*&)csr_row_ptr_B,
                             (const void*&)csr_col_ind_B,
                             (const void*&)descr_C,
                             (const void*&)csr_row_ptr_C,
                             (const void*&)nnz_C);

        const rocsparse_status status = rocsparse::csrgeam_nnz_checkarg(handle,
                                                                        m,
                                                                        n,
                                                                        descr_A,
                                                                        nnz_A,
                                                                        csr_row_ptr_A,
                                                                        csr_col_ind_A,
                                                                        descr_B,
                                                                        nnz_B,
                                                                        csr_row_ptr_B,
                                                                        csr_col_ind_B,
                                                                        descr_C,
                                                                        csr_row_ptr_C,
                                                                        nnz_C);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        const rocsparse_spgeam_descr descr       = nullptr;
        void*                        temp_buffer = nullptr;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgeam_nnz_core(handle,
                                                              descr,
                                                              rocsparse_operation_none,
                                                              rocsparse_operation_none,
                                                              m,
                                                              n,
                                                              descr_A,
                                                              nnz_A,
                                                              csr_row_ptr_A,
                                                              csr_col_ind_A,
                                                              descr_B,
                                                              nnz_B,
                                                              csr_row_ptr_B,
                                                              csr_col_ind_B,
                                                              descr_C,
                                                              csr_row_ptr_C,
                                                              nnz_C,
                                                              temp_buffer,
                                                              false));
        return rocsparse_status_success;
    }
}

template <typename I, typename J>
rocsparse_status rocsparse::csrgeam_nnz_template(rocsparse_handle             handle,
                                                 const rocsparse_spgeam_descr descr,
                                                 rocsparse_operation          trans_A,
                                                 rocsparse_operation          trans_B,
                                                 int64_t                      m,
                                                 int64_t                      n,
                                                 const rocsparse_mat_descr    descr_A,
                                                 int64_t                      nnz_A,
                                                 const void*                  csr_row_ptr_A,
                                                 const void*                  csr_col_ind_A,
                                                 const rocsparse_mat_descr    descr_B,
                                                 int64_t                      nnz_B,
                                                 const void*                  csr_row_ptr_B,
                                                 const void*                  csr_col_ind_B,
                                                 const rocsparse_mat_descr    descr_C,
                                                 void*                        csr_row_ptr_C,
                                                 void*                        nnz_C,
                                                 void*                        temp_buffer,
                                                 bool                         called_from_spgeam)
{
    const rocsparse_status status = rocsparse::csrgeam_nnz_quickreturn(handle,
                                                                       descr,
                                                                       trans_A,
                                                                       trans_B,
                                                                       m,
                                                                       n,
                                                                       descr_A,
                                                                       nnz_A,
                                                                       (const I*)csr_row_ptr_A,
                                                                       (const J*)csr_col_ind_A,
                                                                       descr_B,
                                                                       nnz_B,
                                                                       (const I*)csr_row_ptr_B,
                                                                       (const J*)csr_col_ind_B,
                                                                       descr_C,
                                                                       (I*)csr_row_ptr_C,
                                                                       (I*)nnz_C,
                                                                       temp_buffer,
                                                                       called_from_spgeam);

    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgeam_nnz_core(handle,
                                                          descr,
                                                          trans_A,
                                                          trans_B,
                                                          m,
                                                          n,
                                                          descr_A,
                                                          nnz_A,
                                                          (const I*)csr_row_ptr_A,
                                                          (const J*)csr_col_ind_A,
                                                          descr_B,
                                                          nnz_B,
                                                          (const I*)csr_row_ptr_B,
                                                          (const J*)csr_col_ind_B,
                                                          descr_C,
                                                          (I*)csr_row_ptr_C,
                                                          (I*)nnz_C,
                                                          temp_buffer,
                                                          called_from_spgeam));
    return rocsparse_status_success;
}

#define INSTANTIATE(I, J)                                            \
    template rocsparse_status rocsparse::csrgeam_nnz_template<I, J>( \
        rocsparse_handle             handle,                         \
        const rocsparse_spgeam_descr descr,                          \
        rocsparse_operation          trans_A,                        \
        rocsparse_operation          trans_B,                        \
        int64_t                      m,                              \
        int64_t                      n,                              \
        const rocsparse_mat_descr    descr_A,                        \
        int64_t                      nnz_A,                          \
        const void*                  csr_row_ptr_A,                  \
        const void*                  csr_col_ind_A,                  \
        const rocsparse_mat_descr    descr_B,                        \
        int64_t                      nnz_B,                          \
        const void*                  csr_row_ptr_B,                  \
        const void*                  csr_col_ind_B,                  \
        const rocsparse_mat_descr    descr_C,                        \
        void*                        csr_row_ptr_C,                  \
        void*                        nnz_C,                          \
        void*                        temp_buffer,                    \
        bool                         called_from_spgeam);

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int64_t, int32_t);
INSTANTIATE(int64_t, int64_t);
#undef INSTANTIATE

extern "C" rocsparse_status rocsparse_csrgeam_nnz(rocsparse_handle          handle,
                                                  rocsparse_int             m,
                                                  rocsparse_int             n,
                                                  const rocsparse_mat_descr descr_A,
                                                  rocsparse_int             nnz_A,
                                                  const rocsparse_int*      csr_row_ptr_A,
                                                  const rocsparse_int*      csr_col_ind_A,
                                                  const rocsparse_mat_descr descr_B,
                                                  rocsparse_int             nnz_B,
                                                  const rocsparse_int*      csr_row_ptr_B,
                                                  const rocsparse_int*      csr_col_ind_B,
                                                  const rocsparse_mat_descr descr_C,
                                                  rocsparse_int*            csr_row_ptr_C,
                                                  rocsparse_int*            nnz_C)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgeam_nnz_impl(handle,
                                                          m,
                                                          n,
                                                          descr_A,
                                                          nnz_A,
                                                          csr_row_ptr_A,
                                                          csr_col_ind_A,
                                                          descr_B,
                                                          nnz_B,
                                                          csr_row_ptr_B,
                                                          csr_col_ind_B,
                                                          descr_C,
                                                          csr_row_ptr_C,
                                                          nnz_C));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

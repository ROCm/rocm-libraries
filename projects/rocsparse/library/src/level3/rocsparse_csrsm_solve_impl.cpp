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
#include "internal/level3/rocsparse_csrsm.h"
#include "rocsparse_csrsm.hpp"

#include "rocsparse_assign_async.hpp"
#include "rocsparse_common.h"
#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

#include "../level1/rocsparse_gthr.hpp"
#include "../level2/rocsparse_csrsv.hpp"
#include "csrsm_device.h"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, bool SLEEP, typename I, typename J, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrsm(rocsparse_operation transB,
               J                   m,
               J                   nrhs,
               ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
               const I* __restrict__ csr_row_ptr,
               const J* __restrict__ csr_col_ind,
               const T* __restrict__ csr_val,
               T* __restrict__ B,
               int64_t ldb,
               int* __restrict__ done_array,
               const J* __restrict__ map,
               J* __restrict__ zero_pivot,
               rocsparse_index_base idx_base,
               rocsparse_fill_mode  fill_mode,
               rocsparse_diag_type  diag_type,
               bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);

        rocsparse::csrsm_device<BLOCKSIZE, WFSIZE, SLEEP>(transB,
                                                          m,
                                                          nrhs,
                                                          alpha,
                                                          csr_row_ptr,
                                                          csr_col_ind,
                                                          csr_val,
                                                          B,
                                                          ldb,
                                                          done_array,
                                                          map,
                                                          zero_pivot,
                                                          idx_base,
                                                          fill_mode,
                                                          diag_type);
    }

    template <typename I, typename J, typename T>
    static rocsparse_status csrsm_solve_dispatch(rocsparse_handle          handle,
                                                 rocsparse_operation       trans_A,
                                                 rocsparse_operation       trans_B,
                                                 J                         m,
                                                 J                         nrhs,
                                                 I                         nnz,
                                                 const T*                  alpha,
                                                 const rocsparse_mat_descr descr,
                                                 const T*                  csr_val,
                                                 const I*                  csr_row_ptr,
                                                 const J*                  csr_col_ind,
                                                 T*                        B,
                                                 int64_t                   ldb,
                                                 rocsparse_order           order_B,
                                                 rocsparse_mat_info        info,
                                                 rocsparse_solve_policy    policy,
                                                 rocsparse_csrsm_info      csrsm_info,
                                                 void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Stream
        hipStream_t stream = handle->stream;

        // Buffer
        char* ptr = reinterpret_cast<char*>(temp_buffer);
        ptr += 256;

        // Each thread block performs at most blockdim columns of the
        // rhs matrix. Therefore, the number of blocks depend on nrhs
        // and the blocksize.
        // Because of this, we might need a larger done_array compared
        // to csrsv.
        int blockdim = 512;
        while(nrhs <= blockdim && blockdim > 32)
        {
            blockdim >>= 1;
        }
        blockdim <<= 1;

        const int narrays = (nrhs - 1) / blockdim + 1;

        // done array
        int* done_array = reinterpret_cast<int*>(ptr);
        ptr += ((sizeof(int) * m * narrays - 1) / 256 + 1) * 256;

        // Temporary array to store transpose of B
        T* Bt = B;
        if((trans_B == rocsparse_operation_none && order_B == rocsparse_order_column))
        {
            Bt = reinterpret_cast<T*>(ptr);
            ptr += ((sizeof(T) * m * nrhs - 1) / 256 + 1) * 256;
        }

        // Temporary array to store transpose of A
        T* At = nullptr;
        if(trans_A == rocsparse_operation_transpose
           || trans_A == rocsparse_operation_conjugate_transpose)
        {
            At = reinterpret_cast<T*>(ptr);
        }

        // Initialize buffers
        RETURN_IF_HIP_ERROR(hipMemsetAsync(done_array, 0, sizeof(int) * m * narrays, stream));

        const rocsparse::trm_info_t* trm_info = csrsm_info->get(trans_A, descr->fill_mode);

        // If diag type is unit, re-initialize zero pivot to remove structural zeros
        if(descr->diag_type == rocsparse_diag_type_unit)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::assign_async(reinterpret_cast<J*>(csrsm_info->get_zero_pivot()),
                                        std::numeric_limits<J>::max(),
                                        stream));
        }

        // Leading dimension
        int64_t ldimB = ldb;

        // Transpose B if B is not transposed yet to improve performance
        if((trans_B == rocsparse_operation_none && order_B == rocsparse_order_column))
        {
            // Leading dimension for transposed B
            ldimB = nrhs;

            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::dense_transpose_template(handle, m, nrhs, (T)1, B, ldb, Bt, ldimB));
        }

        // Pointers to differentiate between transpose mode
        const I* local_csr_row_ptr = csr_row_ptr;
        const J* local_csr_col_ind = csr_col_ind;
        const T* local_csr_val     = csr_val;

        rocsparse_fill_mode fill_mode = descr->fill_mode;

        // When computing transposed triangular solve, we first need to update the
        // transposed matrix values
        if(trans_A == rocsparse_operation_transpose
           || trans_A == rocsparse_operation_conjugate_transpose)
        {
            T* csrt_val = At;

            // Gather values
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::gthr_template<I, T>(handle,
                                                nnz,
                                                csr_val,
                                                csrt_val,
                                                (const I*)trm_info->get_transposed_perm(),
                                                rocsparse_index_base_zero)));

            if(trans_A == rocsparse_operation_conjugate_transpose)
            {
                // conjugate csrt_val
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::conjugate(handle, nnz, csrt_val));
            }

            local_csr_row_ptr = (const I*)trm_info->get_transposed_row_ptr();
            local_csr_col_ind = (const J*)trm_info->get_transposed_col_ind();
            local_csr_val     = (const T*)csrt_val;

            fill_mode = (fill_mode == rocsparse_fill_mode_lower) ? rocsparse_fill_mode_upper
                                                                 : rocsparse_fill_mode_lower;
        }
        {
            const dim3 csrsm_blocks(((nrhs - 1) / blockdim + 1) * m);
            const dim3 csrsm_threads(blockdim);

            // Determine gcnArch and ASIC revision

            const std::string gcn_arch_name = rocsparse::handle_get_arch_name(handle);
            const int         asicRev       = handle->asic_rev;

            // rocsparse_pointer_mode_device

            if(blockdim == 64)
            {

                if(gcn_arch_name == rocpsarse_arch_names::gfx908 && asicRev < 2)
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrsm<64, 64, true>),
                        csrsm_blocks,
                        csrsm_threads,
                        0,
                        stream,
                        trans_B,
                        m,
                        nrhs,
                        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha),
                        local_csr_row_ptr,
                        local_csr_col_ind,
                        local_csr_val,
                        Bt,
                        ldimB,
                        done_array,
                        (const J*)trm_info->get_row_map(),
                        (J*)csrsm_info->get_zero_pivot(),
                        descr->base,
                        fill_mode,
                        descr->diag_type,
                        handle->pointer_mode == rocsparse_pointer_mode_host);
                }
                else
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrsm<64, 64, false>),
                        csrsm_blocks,
                        csrsm_threads,
                        0,
                        stream,
                        trans_B,
                        m,
                        nrhs,
                        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha),
                        local_csr_row_ptr,
                        local_csr_col_ind,
                        local_csr_val,
                        Bt,
                        ldimB,
                        done_array,
                        (const J*)trm_info->get_row_map(),
                        (J*)csrsm_info->get_zero_pivot(),
                        descr->base,
                        fill_mode,
                        descr->diag_type,
                        handle->pointer_mode == rocsparse_pointer_mode_host);
                }
            }
            else if(blockdim == 128)
            {
                if(gcn_arch_name == rocpsarse_arch_names::gfx908 && asicRev < 2)
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrsm<128, 64, true>),
                        csrsm_blocks,
                        csrsm_threads,
                        0,
                        stream,
                        trans_B,
                        m,
                        nrhs,
                        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha),
                        local_csr_row_ptr,
                        local_csr_col_ind,
                        local_csr_val,
                        Bt,
                        ldimB,
                        done_array,
                        (const J*)trm_info->get_row_map(),
                        (J*)csrsm_info->get_zero_pivot(),
                        descr->base,
                        fill_mode,
                        descr->diag_type,
                        handle->pointer_mode == rocsparse_pointer_mode_host);
                }
                else
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrsm<128, 64, false>),
                        csrsm_blocks,
                        csrsm_threads,
                        0,
                        stream,
                        trans_B,
                        m,
                        nrhs,
                        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha),
                        local_csr_row_ptr,
                        local_csr_col_ind,
                        local_csr_val,
                        Bt,
                        ldimB,
                        done_array,
                        (const J*)trm_info->get_row_map(),
                        (J*)csrsm_info->get_zero_pivot(),
                        descr->base,
                        fill_mode,
                        descr->diag_type,
                        handle->pointer_mode == rocsparse_pointer_mode_host);
                }
            }
            else if(blockdim == 256)
            {
                if(gcn_arch_name == rocpsarse_arch_names::gfx908 && asicRev < 2)
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrsm<256, 64, true>),
                        csrsm_blocks,
                        csrsm_threads,
                        0,
                        stream,
                        trans_B,
                        m,
                        nrhs,
                        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha),
                        local_csr_row_ptr,
                        local_csr_col_ind,
                        local_csr_val,
                        Bt,
                        ldimB,
                        done_array,
                        (const J*)trm_info->get_row_map(),
                        (J*)csrsm_info->get_zero_pivot(),
                        descr->base,
                        fill_mode,
                        descr->diag_type,
                        handle->pointer_mode == rocsparse_pointer_mode_host);
                }
                else
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrsm<256, 64, false>),
                        csrsm_blocks,
                        csrsm_threads,
                        0,
                        stream,
                        trans_B,
                        m,
                        nrhs,
                        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha),
                        local_csr_row_ptr,
                        local_csr_col_ind,
                        local_csr_val,
                        Bt,
                        ldimB,
                        done_array,
                        (const J*)trm_info->get_row_map(),
                        (J*)csrsm_info->get_zero_pivot(),
                        descr->base,
                        fill_mode,
                        descr->diag_type,
                        handle->pointer_mode == rocsparse_pointer_mode_host);
                }
            }
            else if(blockdim == 512)
            {
                if(gcn_arch_name == rocpsarse_arch_names::gfx908 && asicRev < 2)
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrsm<512, 64, true>),
                        csrsm_blocks,
                        csrsm_threads,
                        0,
                        stream,
                        trans_B,
                        m,
                        nrhs,
                        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha),
                        local_csr_row_ptr,
                        local_csr_col_ind,
                        local_csr_val,
                        Bt,
                        ldimB,
                        done_array,
                        (const J*)trm_info->get_row_map(),
                        (J*)csrsm_info->get_zero_pivot(),
                        descr->base,
                        fill_mode,
                        descr->diag_type,
                        handle->pointer_mode == rocsparse_pointer_mode_host);
                }
                else
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrsm<512, 64, false>),
                        csrsm_blocks,
                        csrsm_threads,
                        0,
                        stream,
                        trans_B,
                        m,
                        nrhs,
                        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha),
                        local_csr_row_ptr,
                        local_csr_col_ind,
                        local_csr_val,
                        Bt,
                        ldimB,
                        done_array,
                        (const J*)trm_info->get_row_map(),
                        (J*)csrsm_info->get_zero_pivot(),
                        descr->base,
                        fill_mode,
                        descr->diag_type,
                        handle->pointer_mode == rocsparse_pointer_mode_host);
                }
            }
            else if(blockdim == 1024)
            {
                if(gcn_arch_name == rocpsarse_arch_names::gfx908 && asicRev < 2)
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrsm<1024, 64, true>),
                        csrsm_blocks,
                        csrsm_threads,
                        0,
                        stream,
                        trans_B,
                        m,
                        nrhs,
                        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha),
                        local_csr_row_ptr,
                        local_csr_col_ind,
                        local_csr_val,
                        Bt,
                        ldimB,
                        done_array,
                        (const J*)trm_info->get_row_map(),
                        (J*)csrsm_info->get_zero_pivot(),
                        descr->base,
                        fill_mode,
                        descr->diag_type,
                        handle->pointer_mode == rocsparse_pointer_mode_host);
                }
                else
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrsm<1024, 64, false>),
                        csrsm_blocks,
                        csrsm_threads,
                        0,
                        stream,
                        trans_B,
                        m,
                        nrhs,
                        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha),
                        local_csr_row_ptr,
                        local_csr_col_ind,
                        local_csr_val,
                        Bt,
                        ldimB,
                        done_array,
                        (const J*)trm_info->get_row_map(),
                        (J*)csrsm_info->get_zero_pivot(),
                        descr->base,
                        fill_mode,
                        descr->diag_type,
                        handle->pointer_mode == rocsparse_pointer_mode_host);
                }
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
            }
        }

        // Transpose B back if B was not initially transposed
        if((trans_B == rocsparse_operation_none && order_B == rocsparse_order_column))
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::dense_transpose_back(handle, m, nrhs, Bt, ldimB, B, ldb));
        }

        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, typename T>
    __launch_bounds__(BLOCKSIZE) __global__
        static void csrsm_solve_copy_y_to_B(const int64_t m, T* B, const int64_t ldb, const T* y)
    {
        const size_t tid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;
        if(tid < m)
        {
            B[tid * ldb] = y[tid];
        }
    }
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse::csrsm_solve_core(rocsparse_handle          handle,
                                             rocsparse_operation       trans_A,
                                             rocsparse_operation       trans_B,
                                             int64_t                   m_,
                                             int64_t                   nrhs_,
                                             int64_t                   nnz_,
                                             const void*               alpha_,
                                             const rocsparse_mat_descr descr,
                                             const void*               csr_val_,
                                             const void*               csr_row_ptr_,
                                             const void*               csr_col_ind_,
                                             void*                     B_,
                                             int64_t                   ldb,
                                             rocsparse_order           order_B,
                                             rocsparse_mat_info        info,
                                             rocsparse_solve_policy    policy,
                                             rocsparse_csrsm_info      csrsm_info,
                                             void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    const J  m           = static_cast<J>(m_);
    const J  nrhs        = static_cast<J>(nrhs_);
    const I  nnz         = static_cast<I>(nnz_);
    const T* alpha       = reinterpret_cast<const T*>(alpha_);
    const T* csr_val     = reinterpret_cast<const T*>(csr_val_);
    const I* csr_row_ptr = reinterpret_cast<const I*>(csr_row_ptr_);
    const J* csr_col_ind = reinterpret_cast<const J*>(csr_col_ind_);
    T*       B           = reinterpret_cast<T*>(B_);

    if(nrhs == 1)
    {
        //
        // Call csrsv.
        //
        T* y        = reinterpret_cast<T*>(temp_buffer);
        temp_buffer = reinterpret_cast<void*>(reinterpret_cast<char*>(temp_buffer)
                                              + ((sizeof(T) * m - 1) / 256 + 1) * 256);
        const int64_t b_inc
            = (trans_B == rocsparse_operation_none && order_B == rocsparse_order_column) ? 1 : ldb;

        RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrsv_solve_template<I, J, T>(handle,
                                                                            trans_A,
                                                                            m,
                                                                            nnz,
                                                                            alpha,
                                                                            descr,
                                                                            csr_val,
                                                                            csr_row_ptr,
                                                                            csr_col_ind,
                                                                            info,
                                                                            B,
                                                                            b_inc,
                                                                            y,
                                                                            policy,
                                                                            csrsm_info,
                                                                            temp_buffer)));

        if((trans_B == rocsparse_operation_none && order_B == rocsparse_order_column))
        {
            RETURN_IF_HIP_ERROR(
                hipMemcpyAsync(B, y, m * sizeof(T), hipMemcpyDeviceToDevice, handle->stream));
        }
        else
        {
            static constexpr uint32_t BLOCKSIZE = 1024;
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((csrsm_solve_copy_y_to_B<BLOCKSIZE, T>),
                                               dim3((m - 1) / BLOCKSIZE + 1),
                                               dim3(BLOCKSIZE),
                                               0,
                                               handle->stream,
                                               m,
                                               B,
                                               ldb,
                                               y);
        }

        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_solve_dispatch(handle,
                                                              trans_A,
                                                              trans_B,
                                                              m,
                                                              nrhs,
                                                              nnz,
                                                              alpha,
                                                              descr,
                                                              csr_val,
                                                              csr_row_ptr,
                                                              csr_col_ind,
                                                              B,
                                                              ldb,
                                                              order_B,
                                                              info,
                                                              policy,
                                                              csrsm_info,
                                                              temp_buffer));
    return rocsparse_status_success;
}

rocsparse_status rocsparse::csrsm_solve_quickreturn(rocsparse_handle          handle,
                                                    rocsparse_operation       trans_A,
                                                    rocsparse_operation       trans_B,
                                                    int64_t                   m,
                                                    int64_t                   nrhs,
                                                    int64_t                   nnz,
                                                    const void*               alpha,
                                                    const rocsparse_mat_descr descr,
                                                    const void*               csr_val,
                                                    const void*               csr_row_ptr,
                                                    const void*               csr_col_ind,
                                                    void*                     B,
                                                    int64_t                   ldb,
                                                    rocsparse_order           order_B,
                                                    rocsparse_mat_info        info,
                                                    rocsparse_solve_policy    policy,
                                                    void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Quick return if possible
    if(m == 0 || nrhs == 0)
    {
        return rocsparse_status_success;
    }
    return rocsparse_status_continue;
}

namespace rocsparse
{
    rocsparse_status csrsm_solve_checkarg(rocsparse_handle          handle, //0
                                          rocsparse_operation       trans_A, //1
                                          rocsparse_operation       trans_B, //2
                                          int64_t                   m, //3
                                          int64_t                   nrhs, //4
                                          int64_t                   nnz, //5
                                          const void*               alpha, //6
                                          const rocsparse_mat_descr descr, //7
                                          const void*               csr_val, //8
                                          const void*               csr_row_ptr, //9
                                          const void*               csr_col_ind, //10
                                          void*                     B, //11
                                          int64_t                   ldb, //12
                                          rocsparse_order           order_B, // non-classified
                                          rocsparse_mat_info        info, //13
                                          rocsparse_solve_policy    policy, //14
                                          void*                     temp_buffer) //15
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_ENUM(1, trans_A);
        ROCSPARSE_CHECKARG_ENUM(2, trans_B);
        ROCSPARSE_CHECKARG_SIZE(3, m);
        ROCSPARSE_CHECKARG_SIZE(4, nrhs);
        ROCSPARSE_CHECKARG_SIZE(5, nnz);

        ROCSPARSE_CHECKARG(12,
                           ldb,
                           (trans_B == rocsparse_operation_none && ldb < m),
                           rocsparse_status_invalid_size);

        ROCSPARSE_CHECKARG(12,
                           ldb,
                           ((trans_B == rocsparse_operation_transpose
                             || trans_B == rocsparse_operation_conjugate_transpose)
                            && ldb < nrhs),
                           rocsparse_status_invalid_size);

        const rocsparse_status status = rocsparse::csrsm_solve_quickreturn(handle,
                                                                           trans_A,
                                                                           trans_B,
                                                                           m,
                                                                           nrhs,
                                                                           nnz,
                                                                           alpha,
                                                                           descr,
                                                                           csr_val,
                                                                           csr_row_ptr,
                                                                           csr_col_ind,
                                                                           B,
                                                                           ldb,
                                                                           order_B,
                                                                           info,
                                                                           policy,
                                                                           temp_buffer);

        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_POINTER(7, descr);

        ROCSPARSE_CHECKARG(7,
                           descr,
                           (descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(7,
                           descr,
                           (descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);

        ROCSPARSE_CHECKARG_ARRAY(8, nnz, csr_val);
        ROCSPARSE_CHECKARG_ARRAY(9, m, csr_row_ptr);
        ROCSPARSE_CHECKARG_ARRAY(10, nnz, csr_col_ind);
        ROCSPARSE_CHECKARG_POINTER(13, info);
        ROCSPARSE_CHECKARG_ENUM(14, policy);

        ROCSPARSE_CHECKARG_POINTER(15, temp_buffer);
        ROCSPARSE_CHECKARG_POINTER(6, alpha);
        ROCSPARSE_CHECKARG_POINTER(11, B);
        return rocsparse_status_continue;
    }

    template <typename I, typename J, typename T>
    rocsparse_status csrsm_solve_impl(rocsparse_handle          handle,
                                      rocsparse_operation       trans_A,
                                      rocsparse_operation       trans_B,
                                      J                         m,
                                      J                         nrhs,
                                      I                         nnz,
                                      const T*                  alpha,
                                      const rocsparse_mat_descr descr,
                                      const T*                  csr_val,
                                      const I*                  csr_row_ptr,
                                      const J*                  csr_col_ind,
                                      T*                        B,
                                      int64_t                   ldb,
                                      rocsparse_order           order_B,
                                      rocsparse_mat_info        info,
                                      rocsparse_solve_policy    policy,
                                      void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse::log_trace(handle,
                             rocsparse::replaceX<T>("rocsparse_Xcsrsm_solve"),
                             trans_A,
                             trans_B,
                             m,
                             nrhs,
                             nnz,
                             LOG_TRACE_SCALAR_VALUE(handle, alpha),
                             (const void*&)descr,
                             (const void*&)csr_val,
                             (const void*&)csr_row_ptr,
                             (const void*&)csr_col_ind,
                             (const void*&)B,
                             ldb,
                             order_B,
                             (const void*&)info,
                             policy,
                             (const void*&)temp_buffer);

        const rocsparse_status status = rocsparse::csrsm_solve_checkarg(handle,
                                                                        trans_A,
                                                                        trans_B,
                                                                        m,
                                                                        nrhs,
                                                                        nnz,
                                                                        alpha,
                                                                        descr,
                                                                        csr_val,
                                                                        csr_row_ptr,
                                                                        csr_col_ind,
                                                                        B,
                                                                        ldb,
                                                                        order_B,
                                                                        info,
                                                                        policy,
                                                                        temp_buffer);

        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrsm_solve_core<I, J, T>(handle,
                                                                        trans_A,
                                                                        trans_B,
                                                                        m,
                                                                        nrhs,
                                                                        nnz,
                                                                        alpha,
                                                                        descr,
                                                                        csr_val,
                                                                        csr_row_ptr,
                                                                        csr_col_ind,
                                                                        B,
                                                                        ldb,
                                                                        order_B,
                                                                        info,
                                                                        policy,
                                                                        info->get_csrsm_info(),
                                                                        temp_buffer)));
        return rocsparse_status_success;
    }
}

#define INSTANTIATE(I, J, T)                                        \
    template rocsparse_status rocsparse::csrsm_solve_core<I, J, T>( \
        rocsparse_handle          handle,                           \
        rocsparse_operation       trans_A,                          \
        rocsparse_operation       trans_B,                          \
        int64_t                   m,                                \
        int64_t                   nrhs,                             \
        int64_t                   nnz,                              \
        const void*               alpha,                            \
        const rocsparse_mat_descr descr,                            \
        const void*               csr_val,                          \
        const void*               csr_row_ptr,                      \
        const void*               csr_col_ind,                      \
        void*                     B,                                \
        int64_t                   ldb,                              \
        rocsparse_order           order_B,                          \
        rocsparse_mat_info        info,                             \
        rocsparse_solve_policy    policy,                           \
        rocsparse_csrsm_info      csrsm_info,                       \
        void*                     temp_buffer);

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);

INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);

INSTANTIATE(int32_t, int64_t, float);
INSTANTIATE(int32_t, int64_t, double);
INSTANTIATE(int32_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int64_t, rocsparse_double_complex);

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
#define C_IMPL(NAME, TTYPE)                                                           \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,                \
                                     rocsparse_operation       trans_A,               \
                                     rocsparse_operation       trans_B,               \
                                     rocsparse_int             m,                     \
                                     rocsparse_int             nrhs,                  \
                                     rocsparse_int             nnz,                   \
                                     const TTYPE*              alpha,                 \
                                     const rocsparse_mat_descr descr,                 \
                                     const TTYPE*              csr_val,               \
                                     const rocsparse_int*      csr_row_ptr,           \
                                     const rocsparse_int*      csr_col_ind,           \
                                     TTYPE*                    B,                     \
                                     rocsparse_int             ldb,                   \
                                     rocsparse_mat_info        info,                  \
                                     rocsparse_solve_policy    policy,                \
                                     void*                     temp_buffer)           \
    try                                                                               \
    {                                                                                 \
        ROCSPARSE_ROUTINE_TRACE;                                                      \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_solve_impl(handle,                 \
                                                              trans_A,                \
                                                              trans_B,                \
                                                              m,                      \
                                                              nrhs,                   \
                                                              nnz,                    \
                                                              alpha,                  \
                                                              descr,                  \
                                                              csr_val,                \
                                                              csr_row_ptr,            \
                                                              csr_col_ind,            \
                                                              B,                      \
                                                              ldb,                    \
                                                              rocsparse_order_column, \
                                                              info,                   \
                                                              policy,                 \
                                                              temp_buffer));          \
        return rocsparse_status_success;                                              \
    }                                                                                 \
    catch(...)                                                                        \
    {                                                                                 \
        RETURN_ROCSPARSE_EXCEPTION();                                                 \
    }

C_IMPL(rocsparse_scsrsm_solve, float);
C_IMPL(rocsparse_dcsrsm_solve, double);
C_IMPL(rocsparse_ccsrsm_solve, rocsparse_float_complex);
C_IMPL(rocsparse_zcsrsm_solve, rocsparse_double_complex);

#undef C_IMPL

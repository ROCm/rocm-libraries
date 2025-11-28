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

#include "internal/precond/rocsparse_csrilu0.h"
#include "csrilu0_device.h"
#include "rocsparse_csrilu0.hpp"

namespace rocsparse
{
    template <typename T>
    static rocsparse_status csrilu0_numeric_boost_template(rocsparse_handle   handle,
                                                           rocsparse_mat_info info,
                                                           int                enable_boost,
                                                           size_t             boost_tol_size,
                                                           const void*        boost_tol,
                                                           const T*           boost_val)
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_POINTER(1, info);

        // Reset boost
        info->boost_enable = 0;

        // Numeric boost
        if(enable_boost)
        {
            // Check pointer arguments
            ROCSPARSE_CHECKARG_POINTER(3, boost_tol);
            ROCSPARSE_CHECKARG_POINTER(4, boost_val);

            info->boost_enable   = enable_boost;
            info->boost_tol_size = boost_tol_size;
            info->boost_tol      = boost_tol;
            info->boost_val      = reinterpret_cast<const void*>(boost_val);
        }

        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, bool SLEEP, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrilu0_binsearch(rocsparse_int        m,
                           const rocsparse_int* csr_row_ptr,
                           const rocsparse_int* csr_col_ind,
                           T*                   csr_val,
                           const rocsparse_int* csr_diag_ind,
                           int*                 done,
                           const rocsparse_int* map,
                           rocsparse_int*       zero_pivot,
                           rocsparse_int*       singular_pivot,
                           double               tol,
                           rocsparse_index_base idx_base,
                           int                  enable_boost,
                           size_t               size_boost_tol,
                           ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(float, boost_tol_32),
                           ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(double, boost_tol_64),
                           ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, boost_val),
                           bool is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(enable_boost, boost_tol_32);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(enable_boost, boost_tol_64);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(enable_boost, boost_val);
        const double boost_tol = (size_boost_tol == sizeof(double)) ? boost_tol_64 : boost_tol_32;

        rocsparse::csrilu0_binsearch_kernel<BLOCKSIZE, WFSIZE, SLEEP>(m,
                                                                      csr_row_ptr,
                                                                      csr_col_ind,
                                                                      csr_val,
                                                                      csr_diag_ind,
                                                                      done,
                                                                      map,
                                                                      zero_pivot,
                                                                      singular_pivot,
                                                                      tol,
                                                                      idx_base,
                                                                      enable_boost,
                                                                      boost_tol,
                                                                      boost_val);
    }

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, uint32_t HASH, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrilu0_hash(rocsparse_int        m,
                      const rocsparse_int* csr_row_ptr,
                      const rocsparse_int* csr_col_ind,
                      T*                   csr_val,
                      const rocsparse_int* csr_diag_ind,
                      int*                 done,
                      const rocsparse_int* map,
                      rocsparse_int*       zero_pivot,
                      rocsparse_int*       singular_pivot,
                      double               tol,
                      rocsparse_index_base idx_base,
                      int                  enable_boost,
                      size_t               size_boost_tol,
                      ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(float, boost_tol_32),
                      ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(double, boost_tol_64),
                      ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, boost_val),
                      bool is_host_mode)
    {

        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(enable_boost, boost_tol_32);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(enable_boost, boost_tol_64);
        const double boost_tol = (size_boost_tol == sizeof(double)) ? boost_tol_64 : boost_tol_32;

        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(enable_boost, boost_val);

        rocsparse::csrilu0_hash_kernel<BLOCKSIZE, WFSIZE, HASH>(m,
                                                                csr_row_ptr,
                                                                csr_col_ind,
                                                                csr_val,
                                                                csr_diag_ind,
                                                                done,
                                                                map,
                                                                zero_pivot,
                                                                singular_pivot,
                                                                tol,
                                                                idx_base,
                                                                enable_boost,
                                                                boost_tol,
                                                                boost_val);
    }

    template <typename T>
    static rocsparse_status csrilu0_dispatch(rocsparse_handle          handle,
                                             rocsparse_int             m,
                                             rocsparse_int             nnz,
                                             const rocsparse_mat_descr descr,
                                             T*                        csr_val,
                                             const rocsparse_int*      csr_row_ptr,
                                             const rocsparse_int*      csr_col_ind,
                                             rocsparse_mat_info        info,
                                             rocsparse_solve_policy    policy,
                                             void*                     temp_buffer,
                                             const void*               boost_tol,
                                             const T*                  boost_val)
    {
        ROCSPARSE_ROUTINE_TRACE;

        auto csrilu0_info = info->get_csrilu0_info();
        // Check for valid handle and matrix descriptor
        // Stream
        hipStream_t stream = handle->stream;

        // Buffer
        char* ptr = reinterpret_cast<char*>(temp_buffer);
        ptr += 256;

        // done array
        int* d_done_array = reinterpret_cast<int*>(ptr);

        // Initialize buffers
        RETURN_IF_HIP_ERROR(hipMemsetAsync(d_done_array, 0, sizeof(int) * m, stream));

        // Max nnz per row
        rocsparse::trm_info_t* trm_info
            = info->get_csrilu0_info(rocsparse_operation_none, rocsparse_fill_mode_lower);
        const rocsparse_int max_nnz = trm_info->get_max_nnz();

        // Determine gcnArch and ASIC revision
        const std::string gcn_arch_name = rocsparse::handle_get_arch_name(handle);

#define CSRILU0_DIM 256
        dim3 csrilu0_blocks((m * handle->wavefront_size - 1) / CSRILU0_DIM + 1);
        dim3 csrilu0_threads(CSRILU0_DIM);

        const float*  boost_tol_32 = (info->boost_enable) ? (const float*)boost_tol : nullptr;
        const double* boost_tol_64 = (info->boost_enable) ? (const double*)boost_tol : nullptr;

        if(gcn_arch_name == rocpsarse_arch_names::gfx908 && handle->asic_rev < 2)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::csrilu0_binsearch<CSRILU0_DIM, 64, true>),
                csrilu0_blocks,
                csrilu0_threads,
                0,
                stream,
                m,
                csr_row_ptr,
                csr_col_ind,
                csr_val,
                (const rocsparse_int*)trm_info->get_diag_ind(),
                d_done_array,
                (const rocsparse_int*)trm_info->get_row_map(),
                (rocsparse_int*)csrilu0_info->get_zero_pivot(),
                (rocsparse_int*)csrilu0_info->get_singular_pivot(),
                csrilu0_info->get_singular_tol(),
                descr->base,
                info->boost_enable,
                info->boost_tol_size,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_32),
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_64),
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_val),
                handle->pointer_mode == rocsparse_pointer_mode_host);
        }
        else
        {
            if(handle->wavefront_size == 32)
            {
                if(max_nnz < 32)
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrilu0_hash<CSRILU0_DIM, 32, 1>),
                        csrilu0_blocks,
                        csrilu0_threads,
                        0,
                        stream,
                        m,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        (const rocsparse_int*)trm_info->get_diag_ind(),
                        d_done_array,
                        (const rocsparse_int*)trm_info->get_row_map(),
                        (rocsparse_int*)csrilu0_info->get_zero_pivot(),
                        (rocsparse_int*)csrilu0_info->get_singular_pivot(),
                        csrilu0_info->get_singular_tol(),
                        descr->base,
                        info->boost_enable,
                        info->boost_tol_size,
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_32),
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_64),
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_val),
                        handle->pointer_mode == rocsparse_pointer_mode_host);
                }
                else if(max_nnz < 64)
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrilu0_hash<CSRILU0_DIM, 32, 2>),
                        csrilu0_blocks,
                        csrilu0_threads,
                        0,
                        stream,
                        m,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        (const rocsparse_int*)trm_info->get_diag_ind(),
                        d_done_array,
                        (const rocsparse_int*)trm_info->get_row_map(),
                        (rocsparse_int*)csrilu0_info->get_zero_pivot(),
                        (rocsparse_int*)csrilu0_info->get_singular_pivot(),
                        csrilu0_info->get_singular_tol(),
                        descr->base,
                        info->boost_enable,
                        info->boost_tol_size,
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_32),
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_64),
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_val),
                        handle->pointer_mode == rocsparse_pointer_mode_host);
                }
                else if(max_nnz < 128)
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrilu0_hash<CSRILU0_DIM, 32, 4>),
                        csrilu0_blocks,
                        csrilu0_threads,
                        0,
                        stream,
                        m,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        (const rocsparse_int*)trm_info->get_diag_ind(),
                        d_done_array,
                        (const rocsparse_int*)trm_info->get_row_map(),
                        (rocsparse_int*)csrilu0_info->get_zero_pivot(),
                        (rocsparse_int*)csrilu0_info->get_singular_pivot(),
                        csrilu0_info->get_singular_tol(),
                        descr->base,
                        info->boost_enable,
                        info->boost_tol_size,
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_32),
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_64),
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_val),
                        handle->pointer_mode == rocsparse_pointer_mode_host);
                }
                else if(max_nnz < 256)
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrilu0_hash<CSRILU0_DIM, 32, 8>),
                        csrilu0_blocks,
                        csrilu0_threads,
                        0,
                        stream,
                        m,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        (const rocsparse_int*)trm_info->get_diag_ind(),
                        d_done_array,
                        (const rocsparse_int*)trm_info->get_row_map(),
                        (rocsparse_int*)csrilu0_info->get_zero_pivot(),
                        (rocsparse_int*)csrilu0_info->get_singular_pivot(),
                        csrilu0_info->get_singular_tol(),
                        descr->base,
                        info->boost_enable,
                        info->boost_tol_size,
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_32),
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_64),
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_val),
                        handle->pointer_mode == rocsparse_pointer_mode_host);
                }
                else if(max_nnz < 512)
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrilu0_hash<CSRILU0_DIM, 32, 16>),
                        csrilu0_blocks,
                        csrilu0_threads,
                        0,
                        stream,
                        m,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        (const rocsparse_int*)trm_info->get_diag_ind(),
                        d_done_array,
                        (const rocsparse_int*)trm_info->get_row_map(),
                        (rocsparse_int*)csrilu0_info->get_zero_pivot(),
                        (rocsparse_int*)csrilu0_info->get_singular_pivot(),
                        csrilu0_info->get_singular_tol(),
                        descr->base,
                        info->boost_enable,
                        info->boost_tol_size,
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_32),
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_64),
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_val),
                        handle->pointer_mode == rocsparse_pointer_mode_host);
                }
                else
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrilu0_binsearch<CSRILU0_DIM, 32, false>),
                        csrilu0_blocks,
                        csrilu0_threads,
                        0,
                        stream,
                        m,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        (const rocsparse_int*)trm_info->get_diag_ind(),
                        d_done_array,
                        (const rocsparse_int*)trm_info->get_row_map(),
                        (rocsparse_int*)csrilu0_info->get_zero_pivot(),
                        (rocsparse_int*)csrilu0_info->get_singular_pivot(),
                        csrilu0_info->get_singular_tol(),
                        descr->base,
                        info->boost_enable,
                        info->boost_tol_size,
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_32),
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_64),
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_val),
                        handle->pointer_mode == rocsparse_pointer_mode_host);
                }
            }
            else if(handle->wavefront_size == 64)
            {
                if(max_nnz < 64)
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrilu0_hash<CSRILU0_DIM, 64, 1>),
                        csrilu0_blocks,
                        csrilu0_threads,
                        0,
                        stream,
                        m,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        (const rocsparse_int*)trm_info->get_diag_ind(),
                        d_done_array,
                        (const rocsparse_int*)trm_info->get_row_map(),
                        (rocsparse_int*)csrilu0_info->get_zero_pivot(),
                        (rocsparse_int*)csrilu0_info->get_singular_pivot(),
                        csrilu0_info->get_singular_tol(),
                        descr->base,
                        info->boost_enable,
                        info->boost_tol_size,
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_32),
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_64),
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_val),
                        handle->pointer_mode == rocsparse_pointer_mode_host);
                }
                else if(max_nnz < 128)
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrilu0_hash<CSRILU0_DIM, 64, 2>),
                        csrilu0_blocks,
                        csrilu0_threads,
                        0,
                        stream,
                        m,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        (const rocsparse_int*)trm_info->get_diag_ind(),
                        d_done_array,
                        (const rocsparse_int*)trm_info->get_row_map(),
                        (rocsparse_int*)csrilu0_info->get_zero_pivot(),
                        (rocsparse_int*)csrilu0_info->get_singular_pivot(),
                        csrilu0_info->get_singular_tol(),
                        descr->base,
                        info->boost_enable,
                        info->boost_tol_size,
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_32),
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_64),
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_val),
                        handle->pointer_mode == rocsparse_pointer_mode_host);
                }
                else if(max_nnz < 256)
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrilu0_hash<CSRILU0_DIM, 64, 4>),
                        csrilu0_blocks,
                        csrilu0_threads,
                        0,
                        stream,
                        m,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        (const rocsparse_int*)trm_info->get_diag_ind(),
                        d_done_array,
                        (const rocsparse_int*)trm_info->get_row_map(),
                        (rocsparse_int*)csrilu0_info->get_zero_pivot(),
                        (rocsparse_int*)csrilu0_info->get_singular_pivot(),
                        csrilu0_info->get_singular_tol(),
                        descr->base,
                        info->boost_enable,
                        info->boost_tol_size,
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_32),
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_64),
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_val),
                        handle->pointer_mode == rocsparse_pointer_mode_host);
                }
                else if(max_nnz < 512)
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrilu0_hash<CSRILU0_DIM, 64, 8>),
                        csrilu0_blocks,
                        csrilu0_threads,
                        0,
                        stream,
                        m,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        (const rocsparse_int*)trm_info->get_diag_ind(),
                        d_done_array,
                        (const rocsparse_int*)trm_info->get_row_map(),
                        (rocsparse_int*)csrilu0_info->get_zero_pivot(),
                        (rocsparse_int*)csrilu0_info->get_singular_pivot(),
                        csrilu0_info->get_singular_tol(),
                        descr->base,
                        info->boost_enable,
                        info->boost_tol_size,
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_32),
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_64),
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_val),
                        handle->pointer_mode == rocsparse_pointer_mode_host);
                }
                else if(max_nnz < 1024)
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrilu0_hash<CSRILU0_DIM, 64, 16>),
                        csrilu0_blocks,
                        csrilu0_threads,
                        0,
                        stream,
                        m,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        (const rocsparse_int*)trm_info->get_diag_ind(),
                        d_done_array,
                        (const rocsparse_int*)trm_info->get_row_map(),
                        (rocsparse_int*)csrilu0_info->get_zero_pivot(),
                        (rocsparse_int*)csrilu0_info->get_singular_pivot(),
                        csrilu0_info->get_singular_tol(),
                        descr->base,
                        info->boost_enable,
                        info->boost_tol_size,
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_32),
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_64),
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_val),
                        handle->pointer_mode == rocsparse_pointer_mode_host);
                }
                else
                {
                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (rocsparse::csrilu0_binsearch<CSRILU0_DIM, 64, false>),
                        csrilu0_blocks,
                        csrilu0_threads,
                        0,
                        stream,
                        m,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        (const rocsparse_int*)trm_info->get_diag_ind(),
                        d_done_array,
                        (const rocsparse_int*)trm_info->get_row_map(),
                        (rocsparse_int*)csrilu0_info->get_zero_pivot(),
                        (rocsparse_int*)csrilu0_info->get_singular_pivot(),
                        csrilu0_info->get_singular_tol(),
                        descr->base,
                        info->boost_enable,
                        info->boost_tol_size,
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_32),
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_tol_64),
                        ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, boost_val),
                        handle->pointer_mode == rocsparse_pointer_mode_host);
                }
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_arch_mismatch);
            }
        }
#undef CSRILU0_DIM

        return rocsparse_status_success;
    }

    static rocsparse_status csrilu0_quickreturn(rocsparse_handle          handle,
                                                int64_t                   m,
                                                int64_t                   nnz,
                                                const rocsparse_mat_descr descr,
                                                void*                     csr_val,
                                                const void*               csr_row_ptr,
                                                const void*               csr_col_ind,
                                                rocsparse_mat_info        info,
                                                rocsparse_solve_policy    policy,
                                                void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(m == 0)
        {
            return rocsparse_status_success;
        }
        return rocsparse_status_continue;
    }

    static rocsparse_status csrilu0_checkarg(rocsparse_handle          handle, //0
                                             int64_t                   m, //1
                                             int64_t                   nnz, //2
                                             const rocsparse_mat_descr descr, //3
                                             void*                     csr_val, //4
                                             const void*               csr_row_ptr, //5
                                             const void*               csr_col_ind, //6
                                             rocsparse_mat_info        info, //7
                                             rocsparse_solve_policy    policy, //8
                                             void*                     temp_buffer) //9
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_SIZE(1, m);

        const rocsparse_status status = rocsparse::csrilu0_quickreturn(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_SIZE(2, nnz);
        ROCSPARSE_CHECKARG_POINTER(3, descr);
        ROCSPARSE_CHECKARG(3,
                           descr,
                           (descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(3,
                           descr,
                           (descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);

        ROCSPARSE_CHECKARG_ARRAY(4, nnz, csr_val);
        ROCSPARSE_CHECKARG_ARRAY(5, m, csr_row_ptr);
        ROCSPARSE_CHECKARG_ARRAY(6, nnz, csr_col_ind);

        ROCSPARSE_CHECKARG_POINTER(7, info);
        ROCSPARSE_CHECKARG_ENUM(8, policy);
        ROCSPARSE_CHECKARG_ARRAY(9, m, temp_buffer);

        auto trm = info->get_csrilu0_info(rocsparse_operation_none, rocsparse_fill_mode_lower);
        ROCSPARSE_CHECKARG(7, info, (trm == nullptr), rocsparse_status_invalid_pointer);
        return rocsparse_status_continue;
    }

    template <typename T>
    static rocsparse_status csrilu0_core(rocsparse_handle          handle,
                                         rocsparse_int             m,
                                         rocsparse_int             nnz,
                                         const rocsparse_mat_descr descr,
                                         T*                        csr_val,
                                         const rocsparse_int*      csr_row_ptr,
                                         const rocsparse_int*      csr_col_ind,
                                         rocsparse_mat_info        info,
                                         rocsparse_solve_policy    policy,
                                         void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::csrilu0_dispatch(handle,
                                        m,
                                        nnz,
                                        descr,
                                        csr_val,
                                        csr_row_ptr,
                                        csr_col_ind,
                                        info,
                                        policy,
                                        temp_buffer,
                                        info->boost_tol,
                                        reinterpret_cast<const T*>(info->boost_val)));
        return rocsparse_status_success;
    }
}

template <typename T>
rocsparse_status rocsparse::csrilu0_template(rocsparse_handle          handle,
                                             rocsparse_int             m,
                                             rocsparse_int             nnz,
                                             const rocsparse_mat_descr descr,
                                             T*                        csr_val,
                                             const rocsparse_int*      csr_row_ptr,
                                             const rocsparse_int*      csr_col_ind,
                                             rocsparse_mat_info        info,
                                             rocsparse_solve_policy    policy,
                                             void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xcsrilu0"),
                         m,
                         nnz,
                         (const void*&)descr,
                         (const void*&)csr_val,
                         (const void*&)csr_row_ptr,
                         (const void*&)csr_col_ind,
                         (const void*&)info,
                         policy,
                         (const void*&)temp_buffer);

    const rocsparse_status status = rocsparse::csrilu0_checkarg(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrilu0_core(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer)));
    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_scsrilu0_numeric_boost(rocsparse_handle   handle,
                                                             rocsparse_mat_info info,
                                                             int                enable_boost,
                                                             const float*       boost_tol,
                                                             const float*       boost_val)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrilu0_numeric_boost_template(
        handle, info, enable_boost, sizeof(float), boost_tol, boost_val));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_dcsrilu0_numeric_boost(rocsparse_handle   handle,
                                                             rocsparse_mat_info info,
                                                             int                enable_boost,
                                                             const double*      boost_tol,
                                                             const double*      boost_val)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrilu0_numeric_boost_template(
        handle, info, enable_boost, sizeof(double), boost_tol, boost_val));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status
    rocsparse_ccsrilu0_numeric_boost(rocsparse_handle               handle,
                                     rocsparse_mat_info             info,
                                     int                            enable_boost,
                                     const float*                   boost_tol,
                                     const rocsparse_float_complex* boost_val)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrilu0_numeric_boost_template(
        handle, info, enable_boost, sizeof(float), boost_tol, boost_val));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status
    rocsparse_zcsrilu0_numeric_boost(rocsparse_handle                handle,
                                     rocsparse_mat_info              info,
                                     int                             enable_boost,
                                     const double*                   boost_tol,
                                     const rocsparse_double_complex* boost_val)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrilu0_numeric_boost_template(
        handle, info, enable_boost, sizeof(double), boost_tol, boost_val));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_dscsrilu0_numeric_boost(rocsparse_handle   handle,
                                                              rocsparse_mat_info info,
                                                              int                enable_boost,
                                                              const double*      boost_tol,
                                                              const float*       boost_val)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrilu0_numeric_boost_template(
        handle, info, enable_boost, sizeof(double), boost_tol, boost_val));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status
    rocsparse_dccsrilu0_numeric_boost(rocsparse_handle               handle,
                                      rocsparse_mat_info             info,
                                      int                            enable_boost,
                                      const double*                  boost_tol,
                                      const rocsparse_float_complex* boost_val)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrilu0_numeric_boost_template(
        handle, info, enable_boost, sizeof(double), boost_tol, boost_val));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_csrilu0_clear(rocsparse_handle   handle,
                                                    rocsparse_mat_info info)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, info);

    // Logging
    rocsparse::log_trace(handle, "rocsparse_csrilu0_clear", (const void*&)info);

    info->clear_csrilu0_info();
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_scsrilu0(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               rocsparse_int             nnz,
                                               const rocsparse_mat_descr descr,
                                               float*                    csr_val,
                                               const rocsparse_int*      csr_row_ptr,
                                               const rocsparse_int*      csr_col_ind,
                                               rocsparse_mat_info        info,
                                               rocsparse_solve_policy    policy,
                                               void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrilu0_template(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer)));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_dcsrilu0(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               rocsparse_int             nnz,
                                               const rocsparse_mat_descr descr,
                                               double*                   csr_val,
                                               const rocsparse_int*      csr_row_ptr,
                                               const rocsparse_int*      csr_col_ind,
                                               rocsparse_mat_info        info,
                                               rocsparse_solve_policy    policy,
                                               void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrilu0_template(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer)));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_ccsrilu0(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               rocsparse_int             nnz,
                                               const rocsparse_mat_descr descr,
                                               rocsparse_float_complex*  csr_val,
                                               const rocsparse_int*      csr_row_ptr,
                                               const rocsparse_int*      csr_col_ind,
                                               rocsparse_mat_info        info,
                                               rocsparse_solve_policy    policy,
                                               void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrilu0_template(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer)));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_zcsrilu0(rocsparse_handle          handle,
                                               rocsparse_int             m,
                                               rocsparse_int             nnz,
                                               const rocsparse_mat_descr descr,
                                               rocsparse_double_complex* csr_val,
                                               const rocsparse_int*      csr_row_ptr,
                                               const rocsparse_int*      csr_col_ind,
                                               rocsparse_mat_info        info,
                                               rocsparse_solve_policy    policy,
                                               void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrilu0_template(
        handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer)));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_csrilu0_zero_pivot(rocsparse_handle   handle,
                                                         rocsparse_mat_info info,
                                                         rocsparse_int*     position)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    rocsparse::log_trace(
        handle, "rocsparse_csrilu0_zero_pivot", (const void*&)info, (const void*&)position);

    ROCSPARSE_CHECKARG_POINTER(1, info);
    ROCSPARSE_CHECKARG_POINTER(2, position);

    auto csrilu0_info = info->get_csrilu0_info();
    {
        auto status = csrilu0_info->copy_zero_pivot_async(handle->pointer_mode,
                                                          rocsparse::get_indextype<rocsparse_int>(),
                                                          position,
                                                          handle->stream);
        if(status == rocsparse_status_zero_pivot)
        {
            return status;
        }
        RETURN_IF_ROCSPARSE_ERROR(status);
    }
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_csrilu0_singular_pivot(rocsparse_handle   handle,
                                                             rocsparse_mat_info info,
                                                             rocsparse_int*     position)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    rocsparse::log_trace(
        handle, "rocsparse_csrilu0_singular_pivot", (const void*&)info, (const void*&)position);

    ROCSPARSE_CHECKARG_POINTER(1, info);
    ROCSPARSE_CHECKARG_POINTER(2, position);

    auto    csrilu0_info = info->get_csrilu0_info();
    int64_t zero_pivot_value;
    auto    status = csrilu0_info->copy_zero_pivot_async(rocsparse_pointer_mode_host,
                                                      rocsparse::get_indextype<int64_t>(),
                                                      &zero_pivot_value,
                                                      handle->stream);
    if(status != rocsparse_status_zero_pivot)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
    }

    int64_t singular_pivot_value;
    status = csrilu0_info->copy_singular_pivot_async(rocsparse_pointer_mode_host,
                                                     rocsparse::get_indextype<int64_t>(),
                                                     &singular_pivot_value,
                                                     handle->stream);

    if(status != rocsparse_status_zero_pivot)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
    }

    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

    int64_t singular_pivot;
    if((singular_pivot_value == -1) || (zero_pivot_value == -1))
    {
        singular_pivot = ((singular_pivot_value == -1) ? zero_pivot_value : singular_pivot_value);
    }
    else
    {
        singular_pivot = std::min(zero_pivot_value, singular_pivot_value);
    }

    // Differentiate between pointer modes
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        // rocsparse_pointer_mode_device
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(position,
                                           &singular_pivot,
                                           sizeof(rocsparse_int),
                                           hipMemcpyHostToDevice,
                                           handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
    }
    else
    {
        // rocsparse_pointer_mode_host
        *position = singular_pivot;
    }

    return (rocsparse_status_success);
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status
    rocsparse_csrilu0_set_tolerance(rocsparse_handle handle, rocsparse_mat_info info, double tol)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, info);
    ROCSPARSE_CHECKARG(2, tol, (tol < 0), rocsparse_status_invalid_value);

    // Logging
    rocsparse::log_trace(handle, "rocsparse_csrilu0_set_tolerance", (const void*&)info, tol);
    auto csrilu0_info = info->get_csrilu0_info();
    csrilu0_info->set_singular_tol(tol);

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status
    rocsparse_csrilu0_get_tolerance(rocsparse_handle handle, rocsparse_mat_info info, double* tol)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, info);
    ROCSPARSE_CHECKARG_POINTER(2, tol);

    // Logging
    rocsparse::log_trace(handle, "rocsparse_csrilu0_get_tolerance", (const void*&)info, tol);
    const auto csrilu0_info = info->get_csrilu0_info();
    *tol                    = csrilu0_info->get_singular_tol();

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

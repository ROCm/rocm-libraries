/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/extra/rocsparse_bsrgeam.h"
#include "rocsparse_csrgeam.hpp"
#include "rocsparse_utility.hpp"

#include "bsrgeam_device.h"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, uint32_t WFSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrgeam_wf_per_row_multipass_2_3_kernel(rocsparse_direction dir,
                                                 rocsparse_int       mb,
                                                 rocsparse_int       nb,
                                                 rocsparse_int       block_dim,
                                                 ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                                 const rocsparse_int* __restrict__ bsr_row_ptr_A,
                                                 const rocsparse_int* __restrict__ bsr_col_ind_A,
                                                 const T* __restrict__ bsr_val_A,
                                                 ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                                 const rocsparse_int* __restrict__ bsr_row_ptr_B,
                                                 const rocsparse_int* __restrict__ bsr_col_ind_B,
                                                 const T* __restrict__ bsr_val_B,
                                                 const rocsparse_int* __restrict__ bsr_row_ptr_C,
                                                 rocsparse_int* __restrict__ bsr_col_ind_C,
                                                 T* __restrict__ bsr_val_C,
                                                 rocsparse_index_base idx_base_A,
                                                 rocsparse_index_base idx_base_B,
                                                 rocsparse_index_base idx_base_C,
                                                 bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);
        rocsparse::bsrgeam_wf_per_row_multipass_2_3_device<BLOCKSIZE, BLOCKDIM, WFSIZE>(
            dir,
            mb,
            nb,
            block_dim,
            alpha,
            bsr_row_ptr_A,
            bsr_col_ind_A,
            bsr_val_A,
            beta,
            bsr_row_ptr_B,
            bsr_col_ind_B,
            bsr_val_B,
            bsr_row_ptr_C,
            bsr_col_ind_C,
            bsr_val_C,
            idx_base_A,
            idx_base_B,
            idx_base_C);
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrgeam_wf_per_row_multipass_kernel(rocsparse_direction dir,
                                             rocsparse_int       mb,
                                             rocsparse_int       nb,
                                             rocsparse_int       block_dim,
                                             ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                             const rocsparse_int* __restrict__ bsr_row_ptr_A,
                                             const rocsparse_int* __restrict__ bsr_col_ind_A,
                                             const T* __restrict__ bsr_val_A,
                                             ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                             const rocsparse_int* __restrict__ bsr_row_ptr_B,
                                             const rocsparse_int* __restrict__ bsr_col_ind_B,
                                             const T* __restrict__ bsr_val_B,
                                             const rocsparse_int* __restrict__ bsr_row_ptr_C,
                                             rocsparse_int* __restrict__ bsr_col_ind_C,
                                             T* __restrict__ bsr_val_C,
                                             rocsparse_index_base idx_base_A,
                                             rocsparse_index_base idx_base_B,
                                             rocsparse_index_base idx_base_C,
                                             bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);
        rocsparse::bsrgeam_wf_per_row_multipass_device<BLOCKSIZE, BLOCKDIM>(dir,
                                                                            mb,
                                                                            nb,
                                                                            block_dim,
                                                                            alpha,
                                                                            bsr_row_ptr_A,
                                                                            bsr_col_ind_A,
                                                                            bsr_val_A,
                                                                            beta,
                                                                            bsr_row_ptr_B,
                                                                            bsr_col_ind_B,
                                                                            bsr_val_B,
                                                                            bsr_row_ptr_C,
                                                                            bsr_col_ind_C,
                                                                            bsr_val_C,
                                                                            idx_base_A,
                                                                            idx_base_B,
                                                                            idx_base_C);
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrgeam_block_per_row_multipass_kernel(rocsparse_direction dir,
                                                rocsparse_int       mb,
                                                rocsparse_int       nb,
                                                rocsparse_int       block_dim,
                                                ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                                const rocsparse_int* __restrict__ bsr_row_ptr_A,
                                                const rocsparse_int* __restrict__ bsr_col_ind_A,
                                                const T* __restrict__ bsr_val_A,
                                                ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                                const rocsparse_int* __restrict__ bsr_row_ptr_B,
                                                const rocsparse_int* __restrict__ bsr_col_ind_B,
                                                const T* __restrict__ bsr_val_B,
                                                const rocsparse_int* __restrict__ bsr_row_ptr_C,
                                                rocsparse_int* __restrict__ bsr_col_ind_C,
                                                T* __restrict__ bsr_val_C,
                                                rocsparse_index_base idx_base_A,
                                                rocsparse_index_base idx_base_B,
                                                rocsparse_index_base idx_base_C,
                                                bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);
        rocsparse::bsrgeam_block_per_row_multipass_device<BLOCKSIZE, BLOCKDIM>(dir,
                                                                               mb,
                                                                               nb,
                                                                               block_dim,
                                                                               alpha,
                                                                               bsr_row_ptr_A,
                                                                               bsr_col_ind_A,
                                                                               bsr_val_A,
                                                                               beta,
                                                                               bsr_row_ptr_B,
                                                                               bsr_col_ind_B,
                                                                               bsr_val_B,
                                                                               bsr_row_ptr_C,
                                                                               bsr_col_ind_C,
                                                                               bsr_val_C,
                                                                               idx_base_A,
                                                                               idx_base_B,
                                                                               idx_base_C);
    }

    template <uint32_t BLOCKSIZE, uint32_t BLOCKDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrgeam_block_per_row_multipass_kernel2(rocsparse_direction dir,
                                                 rocsparse_int       mb,
                                                 rocsparse_int       nb,
                                                 rocsparse_int       block_dim,
                                                 ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                                 const rocsparse_int* __restrict__ bsr_row_ptr_A,
                                                 const rocsparse_int* __restrict__ bsr_col_ind_A,
                                                 const T* __restrict__ bsr_val_A,
                                                 ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                                 const rocsparse_int* __restrict__ bsr_row_ptr_B,
                                                 const rocsparse_int* __restrict__ bsr_col_ind_B,
                                                 const T* __restrict__ bsr_val_B,
                                                 const rocsparse_int* __restrict__ bsr_row_ptr_C,
                                                 rocsparse_int* __restrict__ bsr_col_ind_C,
                                                 T* __restrict__ bsr_val_C,
                                                 rocsparse_index_base idx_base_A,
                                                 rocsparse_index_base idx_base_B,
                                                 rocsparse_index_base idx_base_C,
                                                 bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);
        rocsparse::bsrgeam_block_per_row_multipass_device2<BLOCKSIZE, BLOCKDIM>(dir,
                                                                                mb,
                                                                                nb,
                                                                                block_dim,
                                                                                alpha,
                                                                                bsr_row_ptr_A,
                                                                                bsr_col_ind_A,
                                                                                bsr_val_A,
                                                                                beta,
                                                                                bsr_row_ptr_B,
                                                                                bsr_col_ind_B,
                                                                                bsr_val_B,
                                                                                bsr_row_ptr_C,
                                                                                bsr_col_ind_C,
                                                                                bsr_val_C,
                                                                                idx_base_A,
                                                                                idx_base_B,
                                                                                idx_base_C);
    }

    template <typename T>
    static rocsparse_status bsrgeam_dispatch(rocsparse_handle          handle,
                                             rocsparse_direction       dir,
                                             rocsparse_int             mb,
                                             rocsparse_int             nb,
                                             rocsparse_int             block_dim,
                                             const T*                  alpha_device_host,
                                             const rocsparse_mat_descr descr_A,
                                             rocsparse_int             nnzb_A,
                                             const T*                  bsr_val_A,
                                             const rocsparse_int*      bsr_row_ptr_A,
                                             const rocsparse_int*      bsr_col_ind_A,
                                             const T*                  beta_device_host,
                                             const rocsparse_mat_descr descr_B,
                                             rocsparse_int             nnzb_B,
                                             const T*                  bsr_val_B,
                                             const rocsparse_int*      bsr_row_ptr_B,
                                             const rocsparse_int*      bsr_col_ind_B,
                                             const rocsparse_mat_descr descr_C,
                                             T*                        bsr_val_C,
                                             const rocsparse_int*      bsr_row_ptr_C,
                                             rocsparse_int*            bsr_col_ind_C)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(block_dim == 1)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }

        hipStream_t stream = handle->stream;
        if(block_dim == 2)
        {
#define BSRGEAM_DIM 256
            if(handle->wavefront_size == 32)
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::bsrgeam_wf_per_row_multipass_2_3_kernel<BSRGEAM_DIM, 2, 32>),
                    dim3((mb - 1) / (BSRGEAM_DIM / 32) + 1),
                    dim3(BSRGEAM_DIM),
                    0,
                    stream,
                    dir,
                    mb,
                    nb,
                    block_dim,
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                    bsr_row_ptr_A,
                    bsr_col_ind_A,
                    bsr_val_A,
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
                    bsr_row_ptr_B,
                    bsr_col_ind_B,
                    bsr_val_B,
                    bsr_row_ptr_C,
                    bsr_col_ind_C,
                    bsr_val_C,
                    descr_A->base,
                    descr_B->base,
                    descr_C->base,
                    handle->pointer_mode == rocsparse_pointer_mode_host);
            }
            else
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::bsrgeam_wf_per_row_multipass_2_3_kernel<BSRGEAM_DIM, 2, 64>),
                    dim3((mb - 1) / (BSRGEAM_DIM / 64) + 1),
                    dim3(BSRGEAM_DIM),
                    0,
                    stream,
                    dir,
                    mb,
                    nb,
                    block_dim,
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                    bsr_row_ptr_A,
                    bsr_col_ind_A,
                    bsr_val_A,
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
                    bsr_row_ptr_B,
                    bsr_col_ind_B,
                    bsr_val_B,
                    bsr_row_ptr_C,
                    bsr_col_ind_C,
                    bsr_val_C,
                    descr_A->base,
                    descr_B->base,
                    descr_C->base,
                    handle->pointer_mode == rocsparse_pointer_mode_host);
            }
#undef BSRGEAM_DIM
        }
        else if(block_dim == 3)
        {
#define BSRGEAM_DIM 256
            if(handle->wavefront_size == 32)
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::bsrgeam_wf_per_row_multipass_2_3_kernel<BSRGEAM_DIM, 3, 32>),
                    dim3((mb - 1) / (BSRGEAM_DIM / 32) + 1),
                    dim3(BSRGEAM_DIM),
                    0,
                    stream,
                    dir,
                    mb,
                    nb,
                    block_dim,
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                    bsr_row_ptr_A,
                    bsr_col_ind_A,
                    bsr_val_A,
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
                    bsr_row_ptr_B,
                    bsr_col_ind_B,
                    bsr_val_B,
                    bsr_row_ptr_C,
                    bsr_col_ind_C,
                    bsr_val_C,
                    descr_A->base,
                    descr_B->base,
                    descr_C->base,
                    handle->pointer_mode == rocsparse_pointer_mode_host);
            }
            else
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::bsrgeam_wf_per_row_multipass_2_3_kernel<BSRGEAM_DIM, 3, 64>),
                    dim3((mb - 1) / (BSRGEAM_DIM / 64) + 1),
                    dim3(BSRGEAM_DIM),
                    0,
                    stream,
                    dir,
                    mb,
                    nb,
                    block_dim,
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                    bsr_row_ptr_A,
                    bsr_col_ind_A,
                    bsr_val_A,
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
                    bsr_row_ptr_B,
                    bsr_col_ind_B,
                    bsr_val_B,
                    bsr_row_ptr_C,
                    bsr_col_ind_C,
                    bsr_val_C,
                    descr_A->base,
                    descr_B->base,
                    descr_C->base,
                    handle->pointer_mode == rocsparse_pointer_mode_host);
            }
#undef BSRGEAM_DIM
        }
        else if(block_dim == 4)
        {
#define BSRGEAM_DIM 64
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::bsrgeam_wf_per_row_multipass_kernel<BSRGEAM_DIM, 4>),
                dim3((mb - 1) / (BSRGEAM_DIM / (4 * 4)) + 1),
                dim3(BSRGEAM_DIM),
                0,
                stream,
                dir,
                mb,
                nb,
                block_dim,
                ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                bsr_row_ptr_A,
                bsr_col_ind_A,
                bsr_val_A,
                ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
                bsr_row_ptr_B,
                bsr_col_ind_B,
                bsr_val_B,
                bsr_row_ptr_C,
                bsr_col_ind_C,
                bsr_val_C,
                descr_A->base,
                descr_B->base,
                descr_C->base,
                handle->pointer_mode == rocsparse_pointer_mode_host);
#undef BSRGEAM_DIM
        }
        else if(block_dim <= 8)
        {
#define BSRGEAM_DIM 64
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::bsrgeam_wf_per_row_multipass_kernel<BSRGEAM_DIM, 8>),
                dim3((mb - 1) / (BSRGEAM_DIM / (8 * 8)) + 1),
                dim3(BSRGEAM_DIM),
                0,
                stream,
                dir,
                mb,
                nb,
                block_dim,
                ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                bsr_row_ptr_A,
                bsr_col_ind_A,
                bsr_val_A,
                ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
                bsr_row_ptr_B,
                bsr_col_ind_B,
                bsr_val_B,
                bsr_row_ptr_C,
                bsr_col_ind_C,
                bsr_val_C,
                descr_A->base,
                descr_B->base,
                descr_C->base,
                handle->pointer_mode == rocsparse_pointer_mode_host);
#undef BSRGEAM_DIM
        }
        else if(block_dim <= 16)
        {
#define BSRGEAM_DIM 256
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::bsrgeam_block_per_row_multipass_kernel2<BSRGEAM_DIM, 16>),
                dim3(mb),
                dim3(BSRGEAM_DIM),
                0,
                stream,
                dir,
                mb,
                nb,
                block_dim,
                ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                bsr_row_ptr_A,
                bsr_col_ind_A,
                bsr_val_A,
                ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
                bsr_row_ptr_B,
                bsr_col_ind_B,
                bsr_val_B,
                bsr_row_ptr_C,
                bsr_col_ind_C,
                bsr_val_C,
                descr_A->base,
                descr_B->base,
                descr_C->base,
                handle->pointer_mode == rocsparse_pointer_mode_host);
#undef BSRGEAM_DIM
        }
        else if(block_dim <= 32)
        {
#define BSRGEAM_DIM 256
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::bsrgeam_block_per_row_multipass_kernel2<BSRGEAM_DIM, 32>),
                dim3(mb),
                dim3(BSRGEAM_DIM),
                0,
                stream,
                dir,
                mb,
                nb,
                block_dim,
                ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                bsr_row_ptr_A,
                bsr_col_ind_A,
                bsr_val_A,
                ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
                bsr_row_ptr_B,
                bsr_col_ind_B,
                bsr_val_B,
                bsr_row_ptr_C,
                bsr_col_ind_C,
                bsr_val_C,
                descr_A->base,
                descr_B->base,
                descr_C->base,
                handle->pointer_mode == rocsparse_pointer_mode_host);
#undef BSRGEAM_DIM
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }

        return rocsparse_status_success;
    }

    template <typename T>
    static rocsparse_status bsrgeam_core(rocsparse_handle          handle,
                                         rocsparse_direction       dir,
                                         rocsparse_int             mb,
                                         rocsparse_int             nb,
                                         rocsparse_int             block_dim,
                                         const T*                  alpha_device_host,
                                         const rocsparse_mat_descr descr_A,
                                         rocsparse_int             nnzb_A,
                                         const T*                  bsr_val_A,
                                         const rocsparse_int*      bsr_row_ptr_A,
                                         const rocsparse_int*      bsr_col_ind_A,
                                         const T*                  beta_device_host,
                                         const rocsparse_mat_descr descr_B,
                                         rocsparse_int             nnzb_B,
                                         const T*                  bsr_val_B,
                                         const rocsparse_int*      bsr_row_ptr_B,
                                         const rocsparse_int*      bsr_col_ind_B,
                                         const rocsparse_mat_descr descr_C,
                                         T*                        bsr_val_C,
                                         const rocsparse_int*      bsr_row_ptr_C,
                                         rocsparse_int*            bsr_col_ind_C)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Stream
        if(block_dim == 1)
        {
            void* temp_buffer = nullptr;
            RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrgeam_template<T, rocsparse_int, rocsparse_int>(
                handle,
                rocsparse_operation_none,
                rocsparse_operation_none,
                mb,
                nb,
                alpha_device_host,
                descr_A,
                nnzb_A,
                bsr_val_A,
                bsr_row_ptr_A,
                bsr_col_ind_A,
                beta_device_host,
                descr_B,
                nnzb_B,
                bsr_val_B,
                bsr_row_ptr_B,
                bsr_col_ind_B,
                descr_C,
                bsr_val_C,
                bsr_row_ptr_C,
                bsr_col_ind_C,
                temp_buffer)));

            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrgeam_dispatch(handle,
                                                              dir,
                                                              mb,
                                                              nb,
                                                              block_dim,
                                                              alpha_device_host,
                                                              descr_A,
                                                              nnzb_A,
                                                              bsr_val_A,
                                                              bsr_row_ptr_A,
                                                              bsr_col_ind_A,
                                                              beta_device_host,
                                                              descr_B,
                                                              nnzb_B,
                                                              bsr_val_B,
                                                              bsr_row_ptr_B,
                                                              bsr_col_ind_B,
                                                              descr_C,
                                                              bsr_val_C,
                                                              bsr_row_ptr_C,
                                                              bsr_col_ind_C));
        return rocsparse_status_success;
    }

    static rocsparse_status bsrgeam_quickreturn(rocsparse_handle          handle,
                                                rocsparse_direction       dir,
                                                rocsparse_int             mb,
                                                rocsparse_int             nb,
                                                rocsparse_int             block_dim,
                                                const void*               alpha,
                                                const rocsparse_mat_descr descr_A,
                                                rocsparse_int             nnzb_A,
                                                const void*               bsr_val_A,
                                                const void*               bsr_row_ptr_A,
                                                const void*               bsr_col_ind_A,
                                                const void*               beta,
                                                const rocsparse_mat_descr descr_B,
                                                rocsparse_int             nnzb_B,
                                                const void*               bsr_val_B,
                                                const void*               bsr_row_ptr_B,
                                                const void*               bsr_col_ind_B,
                                                const rocsparse_mat_descr descr_C,
                                                void*                     bsr_val_C,
                                                const void*               bsr_row_ptr_C,
                                                void*                     bsr_col_ind_C)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(mb == 0 || nb == 0 || (nnzb_A == 0 && nnzb_B == 0))
        {
            return rocsparse_status_success;
        }

        return rocsparse_status_continue;
    }

    rocsparse_status bsrgeam_checkarg(rocsparse_handle          handle, //0
                                      rocsparse_direction       dir, //1
                                      rocsparse_int             mb, //2
                                      rocsparse_int             nb, //3
                                      rocsparse_int             block_dim, //4
                                      const void*               alpha, //5
                                      const rocsparse_mat_descr descr_A, //6
                                      rocsparse_int             nnzb_A, //7
                                      const void*               bsr_val_A, //8
                                      const rocsparse_int*      bsr_row_ptr_A, //9
                                      const rocsparse_int*      bsr_col_ind_A, //10
                                      const void*               beta, //11
                                      const rocsparse_mat_descr descr_B, //12
                                      rocsparse_int             nnzb_B, //13
                                      const void*               bsr_val_B, //14
                                      const rocsparse_int*      bsr_row_ptr_B, //15
                                      const rocsparse_int*      bsr_col_ind_B, //16
                                      const rocsparse_mat_descr descr_C, //17
                                      void*                     bsr_val_C, //18
                                      const rocsparse_int*      bsr_row_ptr_C, //19
                                      rocsparse_int*            bsr_col_ind_C) //20
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_POINTER(6, descr_A);
        ROCSPARSE_CHECKARG_POINTER(12, descr_B);
        ROCSPARSE_CHECKARG_POINTER(17, descr_C);

        ROCSPARSE_CHECKARG(6,
                           descr_A,
                           (descr_A->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(12,
                           descr_B,
                           (descr_B->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(17,
                           descr_C,
                           (descr_C->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);

        ROCSPARSE_CHECKARG(6,
                           descr_A,
                           (descr_A->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);
        ROCSPARSE_CHECKARG(12,
                           descr_B,
                           (descr_B->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);
        ROCSPARSE_CHECKARG(17,
                           descr_C,
                           (descr_C->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);

        ROCSPARSE_CHECKARG_ENUM(1, dir);
        ROCSPARSE_CHECKARG_SIZE(2, mb);
        ROCSPARSE_CHECKARG_SIZE(3, nb);

        ROCSPARSE_CHECKARG_SIZE(4, block_dim);
        ROCSPARSE_CHECKARG(4, block_dim, (block_dim == 0), rocsparse_status_invalid_size);

        ROCSPARSE_CHECKARG_SIZE(7, nnzb_A);
        ROCSPARSE_CHECKARG_SIZE(13, nnzb_B);

        const rocsparse_status status = rocsparse::bsrgeam_quickreturn(handle,
                                                                       dir,
                                                                       mb,
                                                                       nb,
                                                                       block_dim,
                                                                       alpha,
                                                                       descr_A,
                                                                       nnzb_A,
                                                                       bsr_val_A,
                                                                       bsr_row_ptr_A,
                                                                       bsr_col_ind_A,
                                                                       beta,
                                                                       descr_B,
                                                                       nnzb_B,
                                                                       bsr_val_B,
                                                                       bsr_row_ptr_B,
                                                                       bsr_col_ind_B,
                                                                       descr_C,
                                                                       bsr_val_C,
                                                                       bsr_row_ptr_C,
                                                                       bsr_col_ind_C);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_POINTER(5, alpha);
        ROCSPARSE_CHECKARG_ARRAY(8, nnzb_A, bsr_val_A);
        ROCSPARSE_CHECKARG_ARRAY(9, mb, bsr_row_ptr_A);
        ROCSPARSE_CHECKARG_ARRAY(10, nnzb_A, bsr_col_ind_A);
        ROCSPARSE_CHECKARG_POINTER(11, beta);
        ROCSPARSE_CHECKARG_ARRAY(14, nnzb_B, bsr_val_B);
        ROCSPARSE_CHECKARG_ARRAY(15, mb, bsr_row_ptr_B);
        ROCSPARSE_CHECKARG_ARRAY(16, nnzb_B, bsr_col_ind_B);
        ROCSPARSE_CHECKARG_ARRAY(19, mb, bsr_row_ptr_C);

        if(bsr_col_ind_C == nullptr || bsr_val_C == nullptr)
        {
            rocsparse_int start = 0;
            rocsparse_int end   = 0;
            if(bsr_row_ptr_C != nullptr)
            {
                RETURN_IF_HIP_ERROR(hipMemcpyAsync(&end,
                                                   &bsr_row_ptr_C[mb],
                                                   sizeof(rocsparse_int),
                                                   hipMemcpyDeviceToHost,
                                                   handle->stream));
                RETURN_IF_HIP_ERROR(hipMemcpyAsync(&start,
                                                   &bsr_row_ptr_C[0],
                                                   sizeof(rocsparse_int),
                                                   hipMemcpyDeviceToHost,
                                                   handle->stream));
                RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
            }

            const rocsparse_int nnzb_C = (end - start);
            ROCSPARSE_CHECKARG_ARRAY(18, nnzb_C, bsr_val_C);
            ROCSPARSE_CHECKARG_ARRAY(20, nnzb_C, bsr_col_ind_C);
        }

        return rocsparse_status_continue;
    }

    template <typename... P>
    static rocsparse_status bsrgeam_impl(P&&... p)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse::log_trace("rocsparse_Xbsrgeam", p...);

        const rocsparse_status status = rocsparse::bsrgeam_checkarg(p...);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrgeam_core(p...));
        return rocsparse_status_success;
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

#define C_IMPL(NAME, TYPE)                                                    \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,        \
                                     rocsparse_direction       dir,           \
                                     rocsparse_int             mb,            \
                                     rocsparse_int             nb,            \
                                     rocsparse_int             block_dim,     \
                                     const TYPE*               alpha,         \
                                     const rocsparse_mat_descr descr_A,       \
                                     rocsparse_int             nnzb_A,        \
                                     const TYPE*               bsr_val_A,     \
                                     const rocsparse_int*      bsr_row_ptr_A, \
                                     const rocsparse_int*      bsr_col_ind_A, \
                                     const TYPE*               beta,          \
                                     const rocsparse_mat_descr descr_B,       \
                                     rocsparse_int             nnzb_B,        \
                                     const TYPE*               bsr_val_B,     \
                                     const rocsparse_int*      bsr_row_ptr_B, \
                                     const rocsparse_int*      bsr_col_ind_B, \
                                     const rocsparse_mat_descr descr_C,       \
                                     TYPE*                     bsr_val_C,     \
                                     const rocsparse_int*      bsr_row_ptr_C, \
                                     rocsparse_int*            bsr_col_ind_C) \
    try                                                                       \
    {                                                                         \
        ROCSPARSE_ROUTINE_TRACE;                                              \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrgeam_impl(handle,             \
                                                          dir,                \
                                                          mb,                 \
                                                          nb,                 \
                                                          block_dim,          \
                                                          alpha,              \
                                                          descr_A,            \
                                                          nnzb_A,             \
                                                          bsr_val_A,          \
                                                          bsr_row_ptr_A,      \
                                                          bsr_col_ind_A,      \
                                                          beta,               \
                                                          descr_B,            \
                                                          nnzb_B,             \
                                                          bsr_val_B,          \
                                                          bsr_row_ptr_B,      \
                                                          bsr_col_ind_B,      \
                                                          descr_C,            \
                                                          bsr_val_C,          \
                                                          bsr_row_ptr_C,      \
                                                          bsr_col_ind_C));    \
        return rocsparse_status_success;                                      \
    }                                                                         \
    catch(...)                                                                \
    {                                                                         \
        RETURN_ROCSPARSE_EXCEPTION();                                         \
    }

C_IMPL(rocsparse_sbsrgeam, float);
C_IMPL(rocsparse_dbsrgeam, double);
C_IMPL(rocsparse_cbsrgeam, rocsparse_float_complex);
C_IMPL(rocsparse_zbsrgeam, rocsparse_double_complex);

#undef C_IMPL

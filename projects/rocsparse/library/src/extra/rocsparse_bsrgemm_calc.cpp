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

#include "rocsparse_bsrgemm_calc.hpp"
#include "../conversion/rocsparse_identity.hpp"
#include "bsrgemm_device.h"
#include "csrgemm_device.h"
#include "internal/extra/rocsparse_bsrgemm.h"
#include "rocsparse_bsrgemm.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_csrgemm.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_primitives.hpp"

#define BSRGEMM_MAXGROUPS 8
#define BSRGEMM_NNZ_HASH 79
#define BSRGEMM_FLL_HASH 137

namespace rocsparse
{
    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              uint32_t HASHSIZE,
              uint32_t HASHVAL,
              typename I,
              typename J,
              typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrgemm_fill_wf_per_row_2x2(rocsparse_direction dir,
                                     J                   mb,
                                     J                   nkb,
                                     const J* __restrict__ offset,
                                     const J* __restrict__ perm,
                                     ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                     const I* __restrict__ bsr_row_ptr_A,
                                     const J* __restrict__ bsr_col_ind_A,
                                     const T* __restrict__ bsr_val_A,
                                     const I* __restrict__ bsr_row_ptr_B,
                                     const J* __restrict__ bsr_col_ind_B,
                                     const T* __restrict__ bsr_val_B,
                                     ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                     const I* __restrict__ bsr_row_ptr_D,
                                     const J* __restrict__ bsr_col_ind_D,
                                     const T* __restrict__ bsr_val_D,
                                     const I* __restrict__ bsr_row_ptr_C,
                                     J* __restrict__ bsr_col_ind_C,
                                     T* __restrict__ bsr_val_C,
                                     rocsparse_index_base idx_base_A,
                                     rocsparse_index_base idx_base_B,
                                     rocsparse_index_base idx_base_C,
                                     rocsparse_index_base idx_base_D,
                                     bool                 mul,
                                     bool                 add,
                                     bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(mul, alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(add, beta);
        rocsparse::bsrgemm_fill_wf_per_row_2x2_device<BLOCKSIZE, WF_SIZE, HASHSIZE, HASHVAL>(
            dir,
            mb,
            nkb,
            offset,
            perm,
            alpha,
            bsr_row_ptr_A,
            bsr_col_ind_A,
            bsr_val_A,
            bsr_row_ptr_B,
            bsr_col_ind_B,
            bsr_val_B,
            beta,
            bsr_row_ptr_D,
            bsr_col_ind_D,
            bsr_val_D,
            bsr_row_ptr_C,
            bsr_col_ind_C,
            bsr_val_C,
            idx_base_A,
            idx_base_B,
            idx_base_C,
            idx_base_D,
            mul,
            add);
    }

    template <typename J, typename T, uint32_t HASHSIZE>
    constexpr uint32_t bsrgemm_fill_block_per_row_2x2_shared_memory_size()
    {
        return ((sizeof(J) + sizeof(T) * 4) * HASHSIZE);
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t HASHSIZE,
              uint32_t HASHVAL,
              typename I,
              typename J,
              typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrgemm_fill_block_per_row_2x2(rocsparse_direction dir,
                                        J                   mb,
                                        J                   nkb,
                                        const J* __restrict__ offset,
                                        const J* __restrict__ perm,
                                        ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                        const I* __restrict__ bsr_row_ptr_A,
                                        const J* __restrict__ bsr_col_ind_A,
                                        const T* __restrict__ bsr_val_A,
                                        const I* __restrict__ bsr_row_ptr_B,
                                        const J* __restrict__ bsr_col_ind_B,
                                        const T* __restrict__ bsr_val_B,
                                        ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                        const I* __restrict__ bsr_row_ptr_D,
                                        const J* __restrict__ bsr_col_ind_D,
                                        const T* __restrict__ bsr_val_D,
                                        const I* __restrict__ bsr_row_ptr_C,
                                        J* __restrict__ bsr_col_ind_C,
                                        T* __restrict__ bsr_val_C,
                                        rocsparse_index_base idx_base_A,
                                        rocsparse_index_base idx_base_B,
                                        rocsparse_index_base idx_base_C,
                                        rocsparse_index_base idx_base_D,
                                        bool                 mul,
                                        bool                 add,
                                        bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(mul, alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(add, beta);
        rocsparse::bsrgemm_fill_block_per_row_2x2_device<BLOCKSIZE, WFSIZE, HASHSIZE, HASHVAL>(
            dir,
            mb,
            nkb,
            offset,
            perm,
            alpha,
            bsr_row_ptr_A,
            bsr_col_ind_A,
            bsr_val_A,
            bsr_row_ptr_B,
            bsr_col_ind_B,
            bsr_val_B,
            beta,
            bsr_row_ptr_D,
            bsr_col_ind_D,
            bsr_val_D,
            bsr_row_ptr_C,
            bsr_col_ind_C,
            bsr_val_C,
            idx_base_A,
            idx_base_B,
            idx_base_C,
            idx_base_D,
            mul,
            add);
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              uint32_t HASHSIZE,
              uint32_t HASHVAL,
              uint32_t BLOCKDIM,
              typename I,
              typename J,
              typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrgemm_fill_wf_per_row(rocsparse_direction dir,
                                 J                   mb,
                                 J                   nkb,
                                 J                   block_dim,
                                 const J* __restrict__ offset,
                                 const J* __restrict__ perm,
                                 ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                 const I* __restrict__ bsr_row_ptr_A,
                                 const J* __restrict__ bsr_col_ind_A,
                                 const T* __restrict__ bsr_val_A,
                                 const I* __restrict__ bsr_row_ptr_B,
                                 const J* __restrict__ bsr_col_ind_B,
                                 const T* __restrict__ bsr_val_B,
                                 ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                 const I* __restrict__ bsr_row_ptr_D,
                                 const J* __restrict__ bsr_col_ind_D,
                                 const T* __restrict__ bsr_val_D,
                                 const I* __restrict__ bsr_row_ptr_C,
                                 J* __restrict__ bsr_col_ind_C,
                                 T* __restrict__ bsr_val_C,
                                 rocsparse_index_base idx_base_A,
                                 rocsparse_index_base idx_base_B,
                                 rocsparse_index_base idx_base_C,
                                 rocsparse_index_base idx_base_D,
                                 bool                 mul,
                                 bool                 add,
                                 bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(mul, alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(add, beta);
        rocsparse::bsrgemm_fill_wf_per_row_device<BLOCKSIZE, WF_SIZE, HASHSIZE, HASHVAL, BLOCKDIM>(
            dir,
            mb,
            nkb,
            block_dim,
            offset,
            perm,
            alpha,
            bsr_row_ptr_A,
            bsr_col_ind_A,
            bsr_val_A,
            bsr_row_ptr_B,
            bsr_col_ind_B,
            bsr_val_B,
            beta,
            bsr_row_ptr_D,
            bsr_col_ind_D,
            bsr_val_D,
            bsr_row_ptr_C,
            bsr_col_ind_C,
            bsr_val_C,
            idx_base_A,
            idx_base_B,
            idx_base_C,
            idx_base_D,
            mul,
            add);
    }

    template <uint32_t BLOCKSIZE,
              uint32_t HASHSIZE,
              uint32_t HASHVAL,
              uint32_t BLOCKDIM,
              typename I,
              typename J,
              typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrgemm_fill_block_per_row(rocsparse_direction dir,
                                    J                   mb,
                                    J                   nkb,
                                    J                   block_dim,
                                    const J* __restrict__ offset,
                                    const J* __restrict__ perm,
                                    ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                    const I* __restrict__ bsr_row_ptr_A,
                                    const J* __restrict__ bsr_col_ind_A,
                                    const T* __restrict__ bsr_val_A,
                                    const I* __restrict__ bsr_row_ptr_B,
                                    const J* __restrict__ bsr_col_ind_B,
                                    const T* __restrict__ bsr_val_B,
                                    ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                    const I* __restrict__ bsr_row_ptr_D,
                                    const J* __restrict__ bsr_col_ind_D,
                                    const T* __restrict__ bsr_val_D,
                                    const I* __restrict__ bsr_row_ptr_C,
                                    J* __restrict__ bsr_col_ind_C,
                                    T* __restrict__ bsr_val_C,
                                    rocsparse_index_base idx_base_A,
                                    rocsparse_index_base idx_base_B,
                                    rocsparse_index_base idx_base_C,
                                    rocsparse_index_base idx_base_D,
                                    bool                 mul,
                                    bool                 add,
                                    bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(mul, alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(add, beta);
        rocsparse::bsrgemm_fill_block_per_row_device<BLOCKSIZE, HASHSIZE, HASHVAL, BLOCKDIM>(
            dir,
            mb,
            nkb,
            block_dim,
            offset,
            perm,
            alpha,
            bsr_row_ptr_A,
            bsr_col_ind_A,
            bsr_val_A,
            bsr_row_ptr_B,
            bsr_col_ind_B,
            bsr_val_B,
            beta,
            bsr_row_ptr_D,
            bsr_col_ind_D,
            bsr_val_D,
            bsr_row_ptr_C,
            bsr_col_ind_C,
            bsr_val_C,
            idx_base_A,
            idx_base_B,
            idx_base_C,
            idx_base_D,
            mul,
            add);
    }

    template <uint32_t BLOCKSIZE,
              uint32_t CHUNKSIZE,
              uint32_t BLOCKDIM,
              typename I,
              typename J,
              typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrgemm_block_per_row_atomic_multipass(rocsparse_direction dir,
                                                J                   nb,
                                                J                   block_dim,
                                                const J* __restrict__ offset,
                                                const J* __restrict__ perm,
                                                ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                                const I* __restrict__ bsr_row_ptr_A,
                                                const J* __restrict__ bsr_col_ind_A,
                                                const T* __restrict__ bsr_val_A,
                                                const I* __restrict__ bsr_row_ptr_B,
                                                const J* __restrict__ bsr_col_ind_B,
                                                const T* __restrict__ bsr_val_B,
                                                ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                                const I* __restrict__ bsr_row_ptr_D,
                                                const J* __restrict__ bsr_col_ind_D,
                                                const T* __restrict__ bsr_val_D,
                                                const I* __restrict__ bsr_row_ptr_C,
                                                J* __restrict__ bsr_col_ind_C,
                                                T* __restrict__ bsr_val_C,
                                                I* __restrict__ workspace_B,
                                                rocsparse_index_base idx_base_A,
                                                rocsparse_index_base idx_base_B,
                                                rocsparse_index_base idx_base_C,
                                                rocsparse_index_base idx_base_D,
                                                bool                 mul,
                                                bool                 add,
                                                bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(mul, alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(add, beta);
        rocsparse::bsrgemm_block_per_row_atomic_multipass_device<BLOCKSIZE, CHUNKSIZE, BLOCKDIM>(
            dir,
            nb,
            block_dim,
            offset,
            perm,
            alpha,
            bsr_row_ptr_A,
            bsr_col_ind_A,
            bsr_val_A,
            bsr_row_ptr_B,
            bsr_col_ind_B,
            bsr_val_B,
            beta,
            bsr_row_ptr_D,
            bsr_col_ind_D,
            bsr_val_D,
            bsr_row_ptr_C,
            bsr_col_ind_C,
            bsr_val_C,
            workspace_B,
            idx_base_A,
            idx_base_B,
            idx_base_C,
            idx_base_D,
            mul,
            add);
    }

    template <uint32_t BLOCKSIZE,
              uint32_t CHUNKSIZE,
              uint32_t BLOCKDIM,
              typename I,
              typename J,
              typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrgemm_block_per_row_multipass(rocsparse_direction dir,
                                         J                   nb,
                                         J                   block_dim,
                                         const J* __restrict__ offset,
                                         const J* __restrict__ perm,
                                         ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                         const I* __restrict__ bsr_row_ptr_A,
                                         const J* __restrict__ bsr_col_ind_A,
                                         const T* __restrict__ bsr_val_A,
                                         const I* __restrict__ bsr_row_ptr_B,
                                         const J* __restrict__ bsr_col_ind_B,
                                         const T* __restrict__ bsr_val_B,
                                         ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                         const I* __restrict__ bsr_row_ptr_D,
                                         const J* __restrict__ bsr_col_ind_D,
                                         const T* __restrict__ bsr_val_D,
                                         const I* __restrict__ bsr_row_ptr_C,
                                         J* __restrict__ bsr_col_ind_C,
                                         T* __restrict__ bsr_val_C,
                                         I* __restrict__ workspace_B,
                                         rocsparse_index_base idx_base_A,
                                         rocsparse_index_base idx_base_B,
                                         rocsparse_index_base idx_base_C,
                                         rocsparse_index_base idx_base_D,
                                         bool                 mul,
                                         bool                 add,
                                         bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(mul, alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(add, beta);
        rocsparse::bsrgemm_block_per_row_multipass_device<BLOCKSIZE, CHUNKSIZE, BLOCKDIM>(
            dir,
            nb,
            block_dim,
            offset,
            perm,
            alpha,
            bsr_row_ptr_A,
            bsr_col_ind_A,
            bsr_val_A,
            bsr_row_ptr_B,
            bsr_col_ind_B,
            bsr_val_B,
            beta,
            bsr_row_ptr_D,
            bsr_col_ind_D,
            bsr_val_D,
            bsr_row_ptr_C,
            bsr_col_ind_C,
            bsr_val_C,
            workspace_B,
            idx_base_A,
            idx_base_B,
            idx_base_C,
            idx_base_D,
            mul,
            add);
    }

    template <typename I,
              typename J,
              typename T,
              typename std::enable_if<std::is_same<T, rocsparse_double_complex>::value, int>::type
              = 0>
    static inline rocsparse_status bsrgemm_2x2_group_6_launcher(rocsparse_handle    handle,
                                                                rocsparse_direction dir,
                                                                J                   group_size,
                                                                const J*            group_offset,
                                                                const J*            perm,
                                                                J                   mb,
                                                                J                   nb,
                                                                J                   kb,
                                                                const T* alpha_device_host,
                                                                const I* bsr_row_ptr_A,
                                                                const J* bsr_col_ind_A,
                                                                const T* bsr_val_A,
                                                                const I* bsr_row_ptr_B,
                                                                const J* bsr_col_ind_B,
                                                                const T* bsr_val_B,
                                                                const T* beta_device_host,
                                                                const I* bsr_row_ptr_D,
                                                                const J* bsr_col_ind_D,
                                                                const T* bsr_val_D,
                                                                const I* bsr_row_ptr_C,
                                                                J*       bsr_col_ind_C,
                                                                T*       bsr_val_C,
                                                                rocsparse_index_base base_A,
                                                                rocsparse_index_base base_B,
                                                                rocsparse_index_base base_C,
                                                                rocsparse_index_base base_D,
                                                                bool                 mul,
                                                                bool                 add)
    {
        ROCSPARSE_ROUTINE_TRACE;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    }

    template <
        typename I,
        typename J,
        typename T,
        typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, double>::value
                                    || std::is_same<T, rocsparse_float_complex>::value,
                                int>::type
        = 0>
    static inline rocsparse_status bsrgemm_2x2_group_6_launcher(rocsparse_handle    handle,
                                                                rocsparse_direction dir,
                                                                J                   group_size,
                                                                const J*            group_offset,
                                                                const J*            perm,
                                                                J                   mb,
                                                                J                   nb,
                                                                J                   kb,
                                                                const T* alpha_device_host,
                                                                const I* bsr_row_ptr_A,
                                                                const J* bsr_col_ind_A,
                                                                const T* bsr_val_A,
                                                                const I* bsr_row_ptr_B,
                                                                const J* bsr_col_ind_B,
                                                                const T* bsr_val_B,
                                                                const T* beta_device_host,
                                                                const I* bsr_row_ptr_D,
                                                                const J* bsr_col_ind_D,
                                                                const T* bsr_val_D,
                                                                const I* bsr_row_ptr_C,
                                                                J*       bsr_col_ind_C,
                                                                T*       bsr_val_C,
                                                                rocsparse_index_base base_A,
                                                                rocsparse_index_base base_B,
                                                                rocsparse_index_base base_C,
                                                                rocsparse_index_base base_D,
                                                                bool                 mul,
                                                                bool                 add)
    {
        ROCSPARSE_ROUTINE_TRACE;

#define BSRGEMM_BLOCKSIZE 256
#define BSRGEMM_HASHSIZE 512
        RETURN_IF_HIP_ERROR(hipFuncSetAttribute(
            (const void*)rocsparse::bsrgemm_fill_block_per_row_2x2<BSRGEMM_BLOCKSIZE,
                                                                   16,
                                                                   BSRGEMM_HASHSIZE,
                                                                   BSRGEMM_FLL_HASH,
                                                                   I,
                                                                   J,
                                                                   T>,
            hipFuncAttributeMaxDynamicSharedMemorySize,
            bsrgemm_fill_block_per_row_2x2_shared_memory_size<J, T, BSRGEMM_HASHSIZE>()));

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::bsrgemm_fill_block_per_row_2x2<BSRGEMM_BLOCKSIZE,
                                                       16,
                                                       BSRGEMM_HASHSIZE,
                                                       BSRGEMM_FLL_HASH>),
            dim3(group_size),
            dim3(BSRGEMM_BLOCKSIZE),
            (bsrgemm_fill_block_per_row_2x2_shared_memory_size<J, T, BSRGEMM_HASHSIZE>()),
            handle->stream,
            dir,
            group_size,
            rocsparse::max(kb, nb),
            group_offset,
            perm,
            ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, alpha_device_host),
            bsr_row_ptr_A,
            bsr_col_ind_A,
            bsr_val_A,
            bsr_row_ptr_B,
            bsr_col_ind_B,
            bsr_val_B,
            ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, beta_device_host),
            bsr_row_ptr_D,
            bsr_col_ind_D,
            bsr_val_D,
            bsr_row_ptr_C,
            bsr_col_ind_C,
            bsr_val_C,
            base_A,
            base_B,
            base_C,
            base_D,
            mul,
            add,
            handle->pointer_mode == rocsparse_pointer_mode_host);
#undef BSRGEMM_BLOCKSIZE
#undef BSRGEMM_HASHSIZE

        return rocsparse_status_success;
    }

    template <typename I, typename J, typename T>
    static inline rocsparse_status bsrgemm_calc_2x2_template(rocsparse_handle    handle,
                                                             rocsparse_direction dir,
                                                             rocsparse_operation trans_A,
                                                             rocsparse_operation trans_B,
                                                             J                   mb,
                                                             J                   nb,
                                                             J                   kb,
                                                             J                   block_dim,
                                                             const T*            alpha_device_host,
                                                             const rocsparse_mat_descr descr_A,
                                                             I                         nnzb_A,
                                                             const T*                  bsr_val_A,
                                                             const I* bsr_row_ptr_A,
                                                             const J* bsr_col_ind_A,
                                                             const rocsparse_mat_descr descr_B,
                                                             I                         nnzb_B,
                                                             const T*                  bsr_val_B,
                                                             const I* bsr_row_ptr_B,
                                                             const J* bsr_col_ind_B,
                                                             const T* beta_device_host,
                                                             const rocsparse_mat_descr descr_D,
                                                             I                         nnzb_D,
                                                             const T*                  bsr_val_D,
                                                             const I* bsr_row_ptr_D,
                                                             const J* bsr_col_ind_D,
                                                             const rocsparse_mat_descr descr_C,
                                                             T*                        bsr_val_C,
                                                             const I*                 bsr_row_ptr_C,
                                                             J*                       bsr_col_ind_C,
                                                             const rocsparse_mat_info info_C,
                                                             J*                       group_size,
                                                             J*                       group_offset,
                                                             J*                       perm,
                                                             I*                       workspace)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Stream
        hipStream_t stream = handle->stream;

        // Index base
        rocsparse_index_base base_A
            = info_C->csrgemm_info->mul ? descr_A->base : rocsparse_index_base_zero;
        rocsparse_index_base base_B
            = info_C->csrgemm_info->mul ? descr_B->base : rocsparse_index_base_zero;
        rocsparse_index_base base_D
            = info_C->csrgemm_info->add ? descr_D->base : rocsparse_index_base_zero;

        // Compute columns and accumulate values for each group
        // Group 0: 0 - 8 non-zeros per row
        if(group_size[0] > 0)
        {
#define BSRGEMM_BLOCKSIZE 256
#define BSRGEMM_WFSIZE 16
#define BSRGEMM_HASHSIZE 8
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::bsrgemm_fill_wf_per_row_2x2<BSRGEMM_BLOCKSIZE,
                                                        BSRGEMM_WFSIZE,
                                                        BSRGEMM_HASHSIZE,
                                                        BSRGEMM_FLL_HASH>),
                dim3((BSRGEMM_WFSIZE * (int64_t)group_size[0] - 1) / 256 + 1),
                dim3(BSRGEMM_BLOCKSIZE),
                0,
                stream,
                dir,
                group_size[0],
                rocsparse::max(kb, nb),
                &group_offset[0],
                perm,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, alpha_device_host),
                bsr_row_ptr_A,
                bsr_col_ind_A,
                bsr_val_A,
                bsr_row_ptr_B,
                bsr_col_ind_B,
                bsr_val_B,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, beta_device_host),
                bsr_row_ptr_D,
                bsr_col_ind_D,
                bsr_val_D,
                bsr_row_ptr_C,
                bsr_col_ind_C,
                bsr_val_C,
                base_A,
                base_B,
                descr_C->base,
                base_D,
                info_C->csrgemm_info->mul,
                info_C->csrgemm_info->add,
                handle->pointer_mode == rocsparse_pointer_mode_host);
#undef BSRGEMM_BLOCKSIZE
#undef BSRGEMM_WFSIZE
#undef BSRGEMM_HASHSIZE
        }

        // Group 1: 9 - 16 non-zeros per row
        if(group_size[1] > 0)
        {
#define BSRGEMM_BLOCKSIZE 256
#define BSRGEMM_WFSIZE 16
#define BSRGEMM_HASHSIZE 16
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::bsrgemm_fill_wf_per_row_2x2<BSRGEMM_BLOCKSIZE,
                                                        BSRGEMM_WFSIZE,
                                                        BSRGEMM_HASHSIZE,
                                                        BSRGEMM_FLL_HASH>),
                dim3((BSRGEMM_WFSIZE * (int64_t)group_size[1] - 1) / 256 + 1),
                dim3(BSRGEMM_BLOCKSIZE),
                0,
                stream,
                dir,
                group_size[1],
                rocsparse::max(kb, nb),
                &group_offset[1],
                perm,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, alpha_device_host),
                bsr_row_ptr_A,
                bsr_col_ind_A,
                bsr_val_A,
                bsr_row_ptr_B,
                bsr_col_ind_B,
                bsr_val_B,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, beta_device_host),
                bsr_row_ptr_D,
                bsr_col_ind_D,
                bsr_val_D,
                bsr_row_ptr_C,
                bsr_col_ind_C,
                bsr_val_C,
                base_A,
                base_B,
                descr_C->base,
                base_D,
                info_C->csrgemm_info->mul,
                info_C->csrgemm_info->add,
                handle->pointer_mode == rocsparse_pointer_mode_host);
#undef BSRGEMM_BLOCKSIZE
#undef BSRGEMM_WFSIZE
#undef BSRGEMM_HASHSIZE
        }

        // Group 2: 17 - 32 non-zeros per row
        if(group_size[2] > 0)
        {
#define BSRGEMM_BLOCKSIZE 256
#define BSRGEMM_WFSIZE 16
#define BSRGEMM_HASHSIZE 32
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::bsrgemm_fill_wf_per_row_2x2<BSRGEMM_BLOCKSIZE,
                                                        BSRGEMM_WFSIZE,
                                                        BSRGEMM_HASHSIZE,
                                                        BSRGEMM_FLL_HASH>),
                dim3((BSRGEMM_WFSIZE * (int64_t)group_size[2] - 1) / 256 + 1),
                dim3(BSRGEMM_BLOCKSIZE),
                0,
                stream,
                dir,
                group_size[2],
                rocsparse::max(kb, nb),
                &group_offset[2],
                perm,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, alpha_device_host),
                bsr_row_ptr_A,
                bsr_col_ind_A,
                bsr_val_A,
                bsr_row_ptr_B,
                bsr_col_ind_B,
                bsr_val_B,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, beta_device_host),
                bsr_row_ptr_D,
                bsr_col_ind_D,
                bsr_val_D,
                bsr_row_ptr_C,
                bsr_col_ind_C,
                bsr_val_C,
                base_A,
                base_B,
                descr_C->base,
                base_D,
                info_C->csrgemm_info->mul,
                info_C->csrgemm_info->add,
                handle->pointer_mode == rocsparse_pointer_mode_host);
#undef BSRGEMM_BLOCKSIZE
#undef BSRGEMM_WFSIZE
#undef BSRGEMM_HASHSIZE
        }

        // Group 3: 33 - 64 non-zeros per row
        if(group_size[3] > 0)
        {
#define BSRGEMM_BLOCKSIZE 256
#define BSRGEMM_HASHSIZE 64
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::bsrgemm_fill_block_per_row_2x2<BSRGEMM_BLOCKSIZE,
                                                           16,
                                                           BSRGEMM_HASHSIZE,
                                                           BSRGEMM_FLL_HASH>),
                dim3(group_size[3]),
                dim3(BSRGEMM_BLOCKSIZE),
                (bsrgemm_fill_block_per_row_2x2_shared_memory_size<J, T, BSRGEMM_HASHSIZE>()),
                stream,
                dir,
                group_size[3],
                rocsparse::max(kb, nb),
                &group_offset[3],
                perm,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, alpha_device_host),
                bsr_row_ptr_A,
                bsr_col_ind_A,
                bsr_val_A,
                bsr_row_ptr_B,
                bsr_col_ind_B,
                bsr_val_B,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, beta_device_host),
                bsr_row_ptr_D,
                bsr_col_ind_D,
                bsr_val_D,
                bsr_row_ptr_C,
                bsr_col_ind_C,
                bsr_val_C,
                base_A,
                base_B,
                descr_C->base,
                base_D,
                info_C->csrgemm_info->mul,
                info_C->csrgemm_info->add,
                handle->pointer_mode == rocsparse_pointer_mode_host);
#undef BSRGEMM_BLOCKSIZE
#undef BSRGEMM_HASHSIZE
        }

        // Group 4: 65 - 128 non-zeros per row
        if(group_size[4] > 0)
        {
#define BSRGEMM_BLOCKSIZE 256
#define BSRGEMM_HASHSIZE 128
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::bsrgemm_fill_block_per_row_2x2<BSRGEMM_BLOCKSIZE,
                                                           16,
                                                           BSRGEMM_HASHSIZE,
                                                           BSRGEMM_FLL_HASH>),
                dim3(group_size[4]),
                dim3(BSRGEMM_BLOCKSIZE),
                (bsrgemm_fill_block_per_row_2x2_shared_memory_size<J, T, BSRGEMM_HASHSIZE>()),
                stream,
                dir,
                group_size[4],
                rocsparse::max(kb, nb),
                &group_offset[4],
                perm,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, alpha_device_host),
                bsr_row_ptr_A,
                bsr_col_ind_A,
                bsr_val_A,
                bsr_row_ptr_B,
                bsr_col_ind_B,
                bsr_val_B,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, beta_device_host),
                bsr_row_ptr_D,
                bsr_col_ind_D,
                bsr_val_D,
                bsr_row_ptr_C,
                bsr_col_ind_C,
                bsr_val_C,
                base_A,
                base_B,
                descr_C->base,
                base_D,
                info_C->csrgemm_info->mul,
                info_C->csrgemm_info->add,
                handle->pointer_mode == rocsparse_pointer_mode_host);
#undef BSRGEMM_BLOCKSIZE
#undef BSRGEMM_HASHSIZE
        }

        // Group 5: 129 - 256 non-zeros per row
        if(group_size[5] > 0)
        {
#define BSRGEMM_BLOCKSIZE 256
#define BSRGEMM_HASHSIZE 256
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::bsrgemm_fill_block_per_row_2x2<BSRGEMM_BLOCKSIZE,
                                                           16,
                                                           BSRGEMM_HASHSIZE,
                                                           BSRGEMM_FLL_HASH>),
                dim3(group_size[5]),
                dim3(BSRGEMM_BLOCKSIZE),
                (bsrgemm_fill_block_per_row_2x2_shared_memory_size<J, T, BSRGEMM_HASHSIZE>()),
                stream,
                dir,
                group_size[5],
                rocsparse::max(kb, nb),
                &group_offset[5],
                perm,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, alpha_device_host),
                bsr_row_ptr_A,
                bsr_col_ind_A,
                bsr_val_A,
                bsr_row_ptr_B,
                bsr_col_ind_B,
                bsr_val_B,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, beta_device_host),
                bsr_row_ptr_D,
                bsr_col_ind_D,
                bsr_val_D,
                bsr_row_ptr_C,
                bsr_col_ind_C,
                bsr_val_C,
                base_A,
                base_B,
                descr_C->base,
                base_D,
                info_C->csrgemm_info->mul,
                info_C->csrgemm_info->add,
                handle->pointer_mode == rocsparse_pointer_mode_host);
#undef BSRGEMM_BLOCKSIZE
#undef BSRGEMM_HASHSIZE
        }

        // Group 6: 257 - 512 non-zeros per row
        if(group_size[6] > 0 && !std::is_same<T, rocsparse_double_complex>())
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::bsrgemm_2x2_group_6_launcher(handle,
                                                        dir,
                                                        group_size[6],
                                                        &group_offset[6],
                                                        perm,
                                                        mb,
                                                        nb,
                                                        kb,
                                                        alpha_device_host,
                                                        bsr_row_ptr_A,
                                                        bsr_col_ind_A,
                                                        bsr_val_A,
                                                        bsr_row_ptr_B,
                                                        bsr_col_ind_B,
                                                        bsr_val_B,
                                                        beta_device_host,
                                                        bsr_row_ptr_D,
                                                        bsr_col_ind_D,
                                                        bsr_val_D,
                                                        bsr_row_ptr_C,
                                                        bsr_col_ind_C,
                                                        bsr_val_C,
                                                        base_A,
                                                        base_B,
                                                        descr_C->base,
                                                        base_D,
                                                        info_C->csrgemm_info->mul,
                                                        info_C->csrgemm_info->add));
        }

        // Group 7: more than 512 non-zero blocks per row (or consumes too much shared memory to use pervious methods)
        if(group_size[7] > 0)
        {
#define BSRGEMM_BLOCKSIZE 256
#define BSRGEMM_CHUNKSIZE 256
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::bsrgemm_block_per_row_atomic_multipass<BSRGEMM_BLOCKSIZE,
                                                                   BSRGEMM_CHUNKSIZE,
                                                                   2>),
                dim3(group_size[7]),
                dim3(BSRGEMM_BLOCKSIZE),
                0,
                stream,
                dir,
                nb,
                block_dim,
                &group_offset[7],
                perm,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, alpha_device_host),
                bsr_row_ptr_A,
                bsr_col_ind_A,
                bsr_val_A,
                bsr_row_ptr_B,
                bsr_col_ind_B,
                bsr_val_B,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, beta_device_host),
                bsr_row_ptr_D,
                bsr_col_ind_D,
                bsr_val_D,
                bsr_row_ptr_C,
                bsr_col_ind_C,
                bsr_val_C,
                workspace,
                base_A,
                base_B,
                descr_C->base,
                base_D,
                info_C->csrgemm_info->mul,
                info_C->csrgemm_info->add,
                handle->pointer_mode == rocsparse_pointer_mode_host);
#undef BSRGEMM_BLOCKSIZE
#undef BSRGEMM_CHUNKSIZE
        }

        return rocsparse_status_success;
    }

    template <typename I, typename J, typename T>
    static inline rocsparse_status bsrgemm_calc_3_4_template(rocsparse_handle    handle,
                                                             rocsparse_direction dir,
                                                             rocsparse_operation trans_A,
                                                             rocsparse_operation trans_B,
                                                             J                   mb,
                                                             J                   nb,
                                                             J                   kb,
                                                             J                   block_dim,
                                                             const T*            alpha_device_host,
                                                             const rocsparse_mat_descr descr_A,
                                                             I                         nnzb_A,
                                                             const T*                  bsr_val_A,
                                                             const I* bsr_row_ptr_A,
                                                             const J* bsr_col_ind_A,
                                                             const rocsparse_mat_descr descr_B,
                                                             I                         nnzb_B,
                                                             const T*                  bsr_val_B,
                                                             const I* bsr_row_ptr_B,
                                                             const J* bsr_col_ind_B,
                                                             const T* beta_device_host,
                                                             const rocsparse_mat_descr descr_D,
                                                             I                         nnzb_D,
                                                             const T*                  bsr_val_D,
                                                             const I* bsr_row_ptr_D,
                                                             const J* bsr_col_ind_D,
                                                             const rocsparse_mat_descr descr_C,
                                                             T*                        bsr_val_C,
                                                             const I*                 bsr_row_ptr_C,
                                                             J*                       bsr_col_ind_C,
                                                             const rocsparse_mat_info info_C,
                                                             J*                       group_size,
                                                             J*                       group_offset,
                                                             J*                       perm,
                                                             I*                       workspace)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Stream
        hipStream_t stream = handle->stream;

        // Index base
        rocsparse_index_base base_A
            = info_C->csrgemm_info->mul ? descr_A->base : rocsparse_index_base_zero;
        rocsparse_index_base base_B
            = info_C->csrgemm_info->mul ? descr_B->base : rocsparse_index_base_zero;
        rocsparse_index_base base_D
            = info_C->csrgemm_info->add ? descr_D->base : rocsparse_index_base_zero;

        // Group 0: 0 - 8 non-zeros per row
        if(group_size[0] > 0)
        {
#define BSRGEMM_BLOCKSIZE 256
#define BSRGEMM_WFSIZE 64
#define BSRGEMM_HASHSIZE 8
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::bsrgemm_fill_wf_per_row<BSRGEMM_BLOCKSIZE,
                                                    BSRGEMM_WFSIZE,
                                                    BSRGEMM_HASHSIZE,
                                                    BSRGEMM_FLL_HASH,
                                                    4>),
                dim3((BSRGEMM_WFSIZE * (int64_t)group_size[0] - 1) / BSRGEMM_BLOCKSIZE + 1),
                dim3(BSRGEMM_BLOCKSIZE),
                0,
                stream,
                dir,
                group_size[0],
                rocsparse::max(kb, nb),
                block_dim,
                &group_offset[0],
                perm,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, alpha_device_host),
                bsr_row_ptr_A,
                bsr_col_ind_A,
                bsr_val_A,
                bsr_row_ptr_B,
                bsr_col_ind_B,
                bsr_val_B,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, beta_device_host),
                bsr_row_ptr_D,
                bsr_col_ind_D,
                bsr_val_D,
                bsr_row_ptr_C,
                bsr_col_ind_C,
                bsr_val_C,
                base_A,
                base_B,
                descr_C->base,
                base_D,
                info_C->csrgemm_info->mul,
                info_C->csrgemm_info->add,
                handle->pointer_mode == rocsparse_pointer_mode_host);
#undef BSRGEMM_BLOCKSIZE
#undef BSRGEMM_WFSIZE
#undef BSRGEMM_HASHSIZE
        }

        // Group 1: 9 - 16 non-zeros per row
        if(group_size[1] > 0)
        {
#define BSRGEMM_BLOCKSIZE 256
#define BSRGEMM_WFSIZE 64
#define BSRGEMM_HASHSIZE 16
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::bsrgemm_fill_wf_per_row<BSRGEMM_BLOCKSIZE,
                                                    BSRGEMM_WFSIZE,
                                                    BSRGEMM_HASHSIZE,
                                                    BSRGEMM_FLL_HASH,
                                                    4>),
                dim3((BSRGEMM_WFSIZE * (int64_t)group_size[1] - 1) / BSRGEMM_BLOCKSIZE + 1),
                dim3(BSRGEMM_BLOCKSIZE),
                0,
                stream,
                dir,
                group_size[1],
                rocsparse::max(kb, nb),
                block_dim,
                &group_offset[1],
                perm,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, alpha_device_host),
                bsr_row_ptr_A,
                bsr_col_ind_A,
                bsr_val_A,
                bsr_row_ptr_B,
                bsr_col_ind_B,
                bsr_val_B,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, beta_device_host),
                bsr_row_ptr_D,
                bsr_col_ind_D,
                bsr_val_D,
                bsr_row_ptr_C,
                bsr_col_ind_C,
                bsr_val_C,
                base_A,
                base_B,
                descr_C->base,
                base_D,
                info_C->csrgemm_info->mul,
                info_C->csrgemm_info->add,
                handle->pointer_mode == rocsparse_pointer_mode_host);
#undef BSRGEMM_BLOCKSIZE
#undef BSRGEMM_WFSIZE
#undef BSRGEMM_HASHSIZE
        }

        // Group 2: 17 - 32 non-zeros per row
        if(group_size[2] > 0)
        {
#define BSRGEMM_BLOCKSIZE 256
#define BSRGEMM_CHUNKSIZE 32
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::bsrgemm_block_per_row_atomic_multipass<BSRGEMM_BLOCKSIZE,
                                                                   BSRGEMM_CHUNKSIZE,
                                                                   4>),
                dim3(group_size[2]),
                dim3(BSRGEMM_BLOCKSIZE),
                0,
                stream,
                dir,
                nb,
                block_dim,
                &group_offset[2],
                perm,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, alpha_device_host),
                bsr_row_ptr_A,
                bsr_col_ind_A,
                bsr_val_A,
                bsr_row_ptr_B,
                bsr_col_ind_B,
                bsr_val_B,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, beta_device_host),
                bsr_row_ptr_D,
                bsr_col_ind_D,
                bsr_val_D,
                bsr_row_ptr_C,
                bsr_col_ind_C,
                bsr_val_C,
                workspace,
                base_A,
                base_B,
                descr_C->base,
                base_D,
                info_C->csrgemm_info->mul,
                info_C->csrgemm_info->add,
                handle->pointer_mode == rocsparse_pointer_mode_host);
#undef BSRGEMM_BLOCKSIZE
#undef BSRGEMM_CHUNKSIZE
        }

        // Group 3: 33 - 64 non-zeros per row
        if(group_size[3] > 0)
        {
#define BSRGEMM_BLOCKSIZE 256
#define BSRGEMM_CHUNKSIZE 64
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::bsrgemm_block_per_row_atomic_multipass<BSRGEMM_BLOCKSIZE,
                                                                   BSRGEMM_CHUNKSIZE,
                                                                   4>),
                dim3(group_size[3]),
                dim3(BSRGEMM_BLOCKSIZE),
                0,
                stream,
                dir,
                nb,
                block_dim,
                &group_offset[3],
                perm,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, alpha_device_host),
                bsr_row_ptr_A,
                bsr_col_ind_A,
                bsr_val_A,
                bsr_row_ptr_B,
                bsr_col_ind_B,
                bsr_val_B,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, beta_device_host),
                bsr_row_ptr_D,
                bsr_col_ind_D,
                bsr_val_D,
                bsr_row_ptr_C,
                bsr_col_ind_C,
                bsr_val_C,
                workspace,
                base_A,
                base_B,
                descr_C->base,
                base_D,
                info_C->csrgemm_info->mul,
                info_C->csrgemm_info->add,
                handle->pointer_mode == rocsparse_pointer_mode_host);
#undef BSRGEMM_BLOCKSIZE
#undef BSRGEMM_CHUNKSIZE
        }

        // Group 4: 65 - 128 non-zeros per row
        if(group_size[4] > 0)
        {
#define BSRGEMM_BLOCKSIZE 256
#define BSRGEMM_CHUNKSIZE 128
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::bsrgemm_block_per_row_atomic_multipass<BSRGEMM_BLOCKSIZE,
                                                                   BSRGEMM_CHUNKSIZE,
                                                                   4>),
                dim3(group_size[4]),
                dim3(BSRGEMM_BLOCKSIZE),
                0,
                stream,
                dir,
                nb,
                block_dim,
                &group_offset[4],
                perm,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, alpha_device_host),
                bsr_row_ptr_A,
                bsr_col_ind_A,
                bsr_val_A,
                bsr_row_ptr_B,
                bsr_col_ind_B,
                bsr_val_B,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, beta_device_host),
                bsr_row_ptr_D,
                bsr_col_ind_D,
                bsr_val_D,
                bsr_row_ptr_C,
                bsr_col_ind_C,
                bsr_val_C,
                workspace,
                base_A,
                base_B,
                descr_C->base,
                base_D,
                info_C->csrgemm_info->mul,
                info_C->csrgemm_info->add,
                handle->pointer_mode == rocsparse_pointer_mode_host);
#undef BSRGEMM_BLOCKSIZE
#undef BSRGEMM_CHUNKSIZE
        }

        // Group 5: 129 - 256 non-zeros per row
        if(group_size[5] > 0)
        {
#define BSRGEMM_BLOCKSIZE 256
#define BSRGEMM_CHUNKSIZE 128
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::bsrgemm_block_per_row_atomic_multipass<BSRGEMM_BLOCKSIZE,
                                                                   BSRGEMM_CHUNKSIZE,
                                                                   4>),
                dim3(group_size[5]),
                dim3(BSRGEMM_BLOCKSIZE),
                0,
                stream,
                dir,
                nb,
                block_dim,
                &group_offset[5],
                perm,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, alpha_device_host),
                bsr_row_ptr_A,
                bsr_col_ind_A,
                bsr_val_A,
                bsr_row_ptr_B,
                bsr_col_ind_B,
                bsr_val_B,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, beta_device_host),
                bsr_row_ptr_D,
                bsr_col_ind_D,
                bsr_val_D,
                bsr_row_ptr_C,
                bsr_col_ind_C,
                bsr_val_C,
                workspace,
                base_A,
                base_B,
                descr_C->base,
                base_D,
                info_C->csrgemm_info->mul,
                info_C->csrgemm_info->add,
                handle->pointer_mode == rocsparse_pointer_mode_host);
#undef BSRGEMM_BLOCKSIZE
#undef BSRGEMM_CHUNKSIZE
        }

        // Group 6: 257 - 512 non-zeros per row
        if(group_size[6] > 0)
        {
#define BSRGEMM_BLOCKSIZE 256
#define BSRGEMM_CHUNKSIZE 128
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::bsrgemm_block_per_row_atomic_multipass<BSRGEMM_BLOCKSIZE,
                                                                   BSRGEMM_CHUNKSIZE,
                                                                   4>),
                dim3(group_size[6]),
                dim3(BSRGEMM_BLOCKSIZE),
                0,
                stream,
                dir,
                nb,
                block_dim,
                &group_offset[6],
                perm,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, alpha_device_host),
                bsr_row_ptr_A,
                bsr_col_ind_A,
                bsr_val_A,
                bsr_row_ptr_B,
                bsr_col_ind_B,
                bsr_val_B,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, beta_device_host),
                bsr_row_ptr_D,
                bsr_col_ind_D,
                bsr_val_D,
                bsr_row_ptr_C,
                bsr_col_ind_C,
                bsr_val_C,
                workspace,
                base_A,
                base_B,
                descr_C->base,
                base_D,
                info_C->csrgemm_info->mul,
                info_C->csrgemm_info->add,
                handle->pointer_mode == rocsparse_pointer_mode_host);
#undef BSRGEMM_BLOCKSIZE
#undef BSRGEMM_CHUNKSIZE
        }

        // Group 7: more than 512 non-zero blocks per row (or consumes too much shared memory to use pervious methods)
        if(group_size[7] > 0)
        {
#define BSRGEMM_BLOCKSIZE 256
#define BSRGEMM_CHUNKSIZE 128
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::bsrgemm_block_per_row_atomic_multipass<BSRGEMM_BLOCKSIZE,
                                                                   BSRGEMM_CHUNKSIZE,
                                                                   4>),
                dim3(group_size[7]),
                dim3(BSRGEMM_BLOCKSIZE),
                0,
                stream,
                dir,
                nb,
                block_dim,
                &group_offset[7],
                perm,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, alpha_device_host),
                bsr_row_ptr_A,
                bsr_col_ind_A,
                bsr_val_A,
                bsr_row_ptr_B,
                bsr_col_ind_B,
                bsr_val_B,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, beta_device_host),
                bsr_row_ptr_D,
                bsr_col_ind_D,
                bsr_val_D,
                bsr_row_ptr_C,
                bsr_col_ind_C,
                bsr_val_C,
                workspace,
                base_A,
                base_B,
                descr_C->base,
                base_D,
                info_C->csrgemm_info->mul,
                info_C->csrgemm_info->add,
                handle->pointer_mode == rocsparse_pointer_mode_host);
#undef BSRGEMM_BLOCKSIZE
#undef BSRGEMM_CHUNKSIZE
        }

        return rocsparse_status_success;
    }

    template <typename I, typename J, typename T>
    static inline rocsparse_status bsrgemm_calc_5_8_template(rocsparse_handle    handle,
                                                             rocsparse_direction dir,
                                                             rocsparse_operation trans_A,
                                                             rocsparse_operation trans_B,
                                                             J                   mb,
                                                             J                   nb,
                                                             J                   kb,
                                                             J                   block_dim,
                                                             const T*            alpha_device_host,
                                                             const rocsparse_mat_descr descr_A,
                                                             I                         nnzb_A,
                                                             const T*                  bsr_val_A,
                                                             const I* bsr_row_ptr_A,
                                                             const J* bsr_col_ind_A,
                                                             const rocsparse_mat_descr descr_B,
                                                             I                         nnzb_B,
                                                             const T*                  bsr_val_B,
                                                             const I* bsr_row_ptr_B,
                                                             const J* bsr_col_ind_B,
                                                             const T* beta_device_host,
                                                             const rocsparse_mat_descr descr_D,
                                                             I                         nnzb_D,
                                                             const T*                  bsr_val_D,
                                                             const I* bsr_row_ptr_D,
                                                             const J* bsr_col_ind_D,
                                                             const rocsparse_mat_descr descr_C,
                                                             T*                        bsr_val_C,
                                                             const I*                 bsr_row_ptr_C,
                                                             J*                       bsr_col_ind_C,
                                                             const rocsparse_mat_info info_C,
                                                             J*                       group_size,
                                                             J*                       group_offset,
                                                             J*                       perm,
                                                             I*                       workspace)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Stream
        hipStream_t stream = handle->stream;

        // Index base
        rocsparse_index_base base_A
            = info_C->csrgemm_info->mul ? descr_A->base : rocsparse_index_base_zero;
        rocsparse_index_base base_B
            = info_C->csrgemm_info->mul ? descr_B->base : rocsparse_index_base_zero;
        rocsparse_index_base base_D
            = info_C->csrgemm_info->add ? descr_D->base : rocsparse_index_base_zero;

        // Group 0: 0 - 8 non-zeros per row
        if(group_size[0] > 0)
        {
#define BSRGEMM_BLOCKSIZE 256
#define BSRGEMM_WFSIZE 64
#define BSRGEMM_HASHSIZE 8
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::bsrgemm_fill_wf_per_row<BSRGEMM_BLOCKSIZE,
                                                    BSRGEMM_WFSIZE,
                                                    BSRGEMM_HASHSIZE,
                                                    BSRGEMM_FLL_HASH,
                                                    8>),
                dim3((BSRGEMM_WFSIZE * (int64_t)group_size[0] - 1) / BSRGEMM_BLOCKSIZE + 1),
                dim3(BSRGEMM_BLOCKSIZE),
                0,
                stream,
                dir,
                group_size[0],
                rocsparse::max(kb, nb),
                block_dim,
                &group_offset[0],
                perm,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, alpha_device_host),
                bsr_row_ptr_A,
                bsr_col_ind_A,
                bsr_val_A,
                bsr_row_ptr_B,
                bsr_col_ind_B,
                bsr_val_B,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, beta_device_host),
                bsr_row_ptr_D,
                bsr_col_ind_D,
                bsr_val_D,
                bsr_row_ptr_C,
                bsr_col_ind_C,
                bsr_val_C,
                base_A,
                base_B,
                descr_C->base,
                base_D,
                info_C->csrgemm_info->mul,
                info_C->csrgemm_info->add,
                handle->pointer_mode == rocsparse_pointer_mode_host);
#undef BSRGEMM_BLOCKSIZE
#undef BSRGEMM_WFSIZE
#undef BSRGEMM_HASHSIZE
        }

        // Group 1: 9 - 16 non-zeros per row
        if(group_size[1] > 0)
        {
#define BSRGEMM_BLOCKSIZE 256
#define BSRGEMM_CHUNKSIZE 16
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::bsrgemm_block_per_row_atomic_multipass<BSRGEMM_BLOCKSIZE,
                                                                   BSRGEMM_CHUNKSIZE,
                                                                   8>),
                dim3(group_size[1]),
                dim3(BSRGEMM_BLOCKSIZE),
                0,
                stream,
                dir,
                nb,
                block_dim,
                &group_offset[1],
                perm,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, alpha_device_host),
                bsr_row_ptr_A,
                bsr_col_ind_A,
                bsr_val_A,
                bsr_row_ptr_B,
                bsr_col_ind_B,
                bsr_val_B,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, beta_device_host),
                bsr_row_ptr_D,
                bsr_col_ind_D,
                bsr_val_D,
                bsr_row_ptr_C,
                bsr_col_ind_C,
                bsr_val_C,
                workspace,
                base_A,
                base_B,
                descr_C->base,
                base_D,
                info_C->csrgemm_info->mul,
                info_C->csrgemm_info->add,
                handle->pointer_mode == rocsparse_pointer_mode_host);
#undef BSRGEMM_BLOCKSIZE
#undef BSRGEMM_CHUNKSIZE
        }

        // Group 2: 17 - 32 non-zeros per row
        if(group_size[2] > 0)
        {
#define BSRGEMM_BLOCKSIZE 256
#define BSRGEMM_CHUNKSIZE 32
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::bsrgemm_block_per_row_atomic_multipass<BSRGEMM_BLOCKSIZE,
                                                                   BSRGEMM_CHUNKSIZE,
                                                                   8>),
                dim3(group_size[2]),
                dim3(BSRGEMM_BLOCKSIZE),
                0,
                stream,
                dir,
                nb,
                block_dim,
                &group_offset[2],
                perm,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, alpha_device_host),
                bsr_row_ptr_A,
                bsr_col_ind_A,
                bsr_val_A,
                bsr_row_ptr_B,
                bsr_col_ind_B,
                bsr_val_B,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, beta_device_host),
                bsr_row_ptr_D,
                bsr_col_ind_D,
                bsr_val_D,
                bsr_row_ptr_C,
                bsr_col_ind_C,
                bsr_val_C,
                workspace,
                base_A,
                base_B,
                descr_C->base,
                base_D,
                info_C->csrgemm_info->mul,
                info_C->csrgemm_info->add,
                handle->pointer_mode == rocsparse_pointer_mode_host);
#undef BSRGEMM_BLOCKSIZE
#undef BSRGEMM_CHUNKSIZE
        }

        // Group 3: 33 - 64 non-zeros per row
        if(group_size[3] > 0)
        {
#define BSRGEMM_BLOCKSIZE 256
#define BSRGEMM_CHUNKSIZE 32
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::bsrgemm_block_per_row_atomic_multipass<BSRGEMM_BLOCKSIZE,
                                                                   BSRGEMM_CHUNKSIZE,
                                                                   8>),
                dim3(group_size[3]),
                dim3(BSRGEMM_BLOCKSIZE),
                0,
                stream,
                dir,
                nb,
                block_dim,
                &group_offset[3],
                perm,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, alpha_device_host),
                bsr_row_ptr_A,
                bsr_col_ind_A,
                bsr_val_A,
                bsr_row_ptr_B,
                bsr_col_ind_B,
                bsr_val_B,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, beta_device_host),
                bsr_row_ptr_D,
                bsr_col_ind_D,
                bsr_val_D,
                bsr_row_ptr_C,
                bsr_col_ind_C,
                bsr_val_C,
                workspace,
                base_A,
                base_B,
                descr_C->base,
                base_D,
                info_C->csrgemm_info->mul,
                info_C->csrgemm_info->add,
                handle->pointer_mode == rocsparse_pointer_mode_host);
#undef BSRGEMM_BLOCKSIZE
#undef BSRGEMM_CHUNKSIZE
        }

        // Group 4: 65 - 128 non-zeros per row
        if(group_size[4] > 0)
        {
#define BSRGEMM_BLOCKSIZE 256
#define BSRGEMM_CHUNKSIZE 32
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::bsrgemm_block_per_row_atomic_multipass<BSRGEMM_BLOCKSIZE,
                                                                   BSRGEMM_CHUNKSIZE,
                                                                   8>),
                dim3(group_size[4]),
                dim3(BSRGEMM_BLOCKSIZE),
                0,
                stream,
                dir,
                nb,
                block_dim,
                &group_offset[4],
                perm,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, alpha_device_host),
                bsr_row_ptr_A,
                bsr_col_ind_A,
                bsr_val_A,
                bsr_row_ptr_B,
                bsr_col_ind_B,
                bsr_val_B,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, beta_device_host),
                bsr_row_ptr_D,
                bsr_col_ind_D,
                bsr_val_D,
                bsr_row_ptr_C,
                bsr_col_ind_C,
                bsr_val_C,
                workspace,
                base_A,
                base_B,
                descr_C->base,
                base_D,
                info_C->csrgemm_info->mul,
                info_C->csrgemm_info->add,
                handle->pointer_mode == rocsparse_pointer_mode_host);
#undef BSRGEMM_BLOCKSIZE
#undef BSRGEMM_CHUNKSIZE
        }

        // Group 5: 129 - 256 non-zeros per row
        if(group_size[5] > 0)
        {
#define BSRGEMM_BLOCKSIZE 256
#define BSRGEMM_CHUNKSIZE 32
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::bsrgemm_block_per_row_atomic_multipass<BSRGEMM_BLOCKSIZE,
                                                                   BSRGEMM_CHUNKSIZE,
                                                                   8>),
                dim3(group_size[5]),
                dim3(BSRGEMM_BLOCKSIZE),
                0,
                stream,
                dir,
                nb,
                block_dim,
                &group_offset[5],
                perm,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, alpha_device_host),
                bsr_row_ptr_A,
                bsr_col_ind_A,
                bsr_val_A,
                bsr_row_ptr_B,
                bsr_col_ind_B,
                bsr_val_B,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, beta_device_host),
                bsr_row_ptr_D,
                bsr_col_ind_D,
                bsr_val_D,
                bsr_row_ptr_C,
                bsr_col_ind_C,
                bsr_val_C,
                workspace,
                base_A,
                base_B,
                descr_C->base,
                base_D,
                info_C->csrgemm_info->mul,
                info_C->csrgemm_info->add,
                handle->pointer_mode == rocsparse_pointer_mode_host);
#undef BSRGEMM_BLOCKSIZE
#undef BSRGEMM_CHUNKSIZE
        }

        // Group 6: 257 - 512 non-zeros per row
        if(group_size[6] > 0)
        {
#define BSRGEMM_BLOCKSIZE 256
#define BSRGEMM_CHUNKSIZE 32
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::bsrgemm_block_per_row_atomic_multipass<BSRGEMM_BLOCKSIZE,
                                                                   BSRGEMM_CHUNKSIZE,
                                                                   8>),
                dim3(group_size[6]),
                dim3(BSRGEMM_BLOCKSIZE),
                0,
                stream,
                dir,
                nb,
                block_dim,
                &group_offset[6],
                perm,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, alpha_device_host),
                bsr_row_ptr_A,
                bsr_col_ind_A,
                bsr_val_A,
                bsr_row_ptr_B,
                bsr_col_ind_B,
                bsr_val_B,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, beta_device_host),
                bsr_row_ptr_D,
                bsr_col_ind_D,
                bsr_val_D,
                bsr_row_ptr_C,
                bsr_col_ind_C,
                bsr_val_C,
                workspace,
                base_A,
                base_B,
                descr_C->base,
                base_D,
                info_C->csrgemm_info->mul,
                info_C->csrgemm_info->add,
                handle->pointer_mode == rocsparse_pointer_mode_host);
#undef BSRGEMM_BLOCKSIZE
#undef BSRGEMM_CHUNKSIZE
        }

        // Group 7: more than 512 non-zero blocks per row (or consumes too much shared memory to use pervious methods)
        if(group_size[7] > 0)
        {
#define BSRGEMM_BLOCKSIZE 256
#define BSRGEMM_CHUNKSIZE 32
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::bsrgemm_block_per_row_atomic_multipass<BSRGEMM_BLOCKSIZE,
                                                                   BSRGEMM_CHUNKSIZE,
                                                                   8>),
                dim3(group_size[7]),
                dim3(BSRGEMM_BLOCKSIZE),
                0,
                stream,
                dir,
                nb,
                block_dim,
                &group_offset[7],
                perm,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, alpha_device_host),
                bsr_row_ptr_A,
                bsr_col_ind_A,
                bsr_val_A,
                bsr_row_ptr_B,
                bsr_col_ind_B,
                bsr_val_B,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, beta_device_host),
                bsr_row_ptr_D,
                bsr_col_ind_D,
                bsr_val_D,
                bsr_row_ptr_C,
                bsr_col_ind_C,
                bsr_val_C,
                workspace,
                base_A,
                base_B,
                descr_C->base,
                base_D,
                info_C->csrgemm_info->mul,
                info_C->csrgemm_info->add,
                handle->pointer_mode == rocsparse_pointer_mode_host);
#undef BSRGEMM_BLOCKSIZE
#undef BSRGEMM_CHUNKSIZE
        }

        return rocsparse_status_success;
    }

    template <typename I, typename J, typename T>
    static inline rocsparse_status bsrgemm_calc_9_16_template(rocsparse_handle    handle,
                                                              rocsparse_direction dir,
                                                              rocsparse_operation trans_A,
                                                              rocsparse_operation trans_B,
                                                              J                   mb,
                                                              J                   nb,
                                                              J                   kb,
                                                              J                   block_dim,
                                                              const T*            alpha_device_host,
                                                              const rocsparse_mat_descr descr_A,
                                                              I                         nnzb_A,
                                                              const T*                  bsr_val_A,
                                                              const I* bsr_row_ptr_A,
                                                              const J* bsr_col_ind_A,
                                                              const rocsparse_mat_descr descr_B,
                                                              I                         nnzb_B,
                                                              const T*                  bsr_val_B,
                                                              const I* bsr_row_ptr_B,
                                                              const J* bsr_col_ind_B,
                                                              const T* beta_device_host,
                                                              const rocsparse_mat_descr descr_D,
                                                              I                         nnzb_D,
                                                              const T*                  bsr_val_D,
                                                              const I* bsr_row_ptr_D,
                                                              const J* bsr_col_ind_D,
                                                              const rocsparse_mat_descr descr_C,
                                                              T*                        bsr_val_C,
                                                              const I* bsr_row_ptr_C,
                                                              J*       bsr_col_ind_C,
                                                              const rocsparse_mat_info info_C,
                                                              J*                       group_size,
                                                              J*                       group_offset,
                                                              J*                       perm,
                                                              I*                       workspace)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Stream
        hipStream_t stream = handle->stream;

        // Index base
        rocsparse_index_base base_A
            = info_C->csrgemm_info->mul ? descr_A->base : rocsparse_index_base_zero;
        rocsparse_index_base base_B
            = info_C->csrgemm_info->mul ? descr_B->base : rocsparse_index_base_zero;
        rocsparse_index_base base_D
            = info_C->csrgemm_info->add ? descr_D->base : rocsparse_index_base_zero;

#define BSRGEMM_BLOCKSIZE 256
#define BSRGEMM_CHUNKSIZE 8
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::bsrgemm_block_per_row_multipass<BSRGEMM_BLOCKSIZE, BSRGEMM_CHUNKSIZE, 16>),
            dim3(group_size[0]),
            dim3(BSRGEMM_BLOCKSIZE),
            0,
            stream,
            dir,
            nb,
            block_dim,
            &group_offset[0],
            perm,
            ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, alpha_device_host),
            bsr_row_ptr_A,
            bsr_col_ind_A,
            bsr_val_A,
            bsr_row_ptr_B,
            bsr_col_ind_B,
            bsr_val_B,
            ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, beta_device_host),
            bsr_row_ptr_D,
            bsr_col_ind_D,
            bsr_val_D,
            bsr_row_ptr_C,
            bsr_col_ind_C,
            bsr_val_C,
            workspace,
            base_A,
            base_B,
            descr_C->base,
            base_D,
            info_C->csrgemm_info->mul,
            info_C->csrgemm_info->add,
            handle->pointer_mode == rocsparse_pointer_mode_host);
#undef BSRGEMM_BLOCKSIZE
#undef BSRGEMM_CHUNKSIZE

        return rocsparse_status_success;
    }

    template <typename I, typename J, typename T>
    static inline rocsparse_status bsrgemm_calc_17_32_template(rocsparse_handle    handle,
                                                               rocsparse_direction dir,
                                                               rocsparse_operation trans_A,
                                                               rocsparse_operation trans_B,
                                                               J                   mb,
                                                               J                   nb,
                                                               J                   kb,
                                                               J                   block_dim,
                                                               const T* alpha_device_host,
                                                               const rocsparse_mat_descr descr_A,
                                                               I                         nnzb_A,
                                                               const T*                  bsr_val_A,
                                                               const I* bsr_row_ptr_A,
                                                               const J* bsr_col_ind_A,
                                                               const rocsparse_mat_descr descr_B,
                                                               I                         nnzb_B,
                                                               const T*                  bsr_val_B,
                                                               const I* bsr_row_ptr_B,
                                                               const J* bsr_col_ind_B,
                                                               const T* beta_device_host,
                                                               const rocsparse_mat_descr descr_D,
                                                               I                         nnzb_D,
                                                               const T*                  bsr_val_D,
                                                               const I* bsr_row_ptr_D,
                                                               const J* bsr_col_ind_D,
                                                               const rocsparse_mat_descr descr_C,
                                                               T*                        bsr_val_C,
                                                               const I* bsr_row_ptr_C,
                                                               J*       bsr_col_ind_C,
                                                               const rocsparse_mat_info info_C,
                                                               J*                       group_size,
                                                               J* group_offset,
                                                               J* perm,
                                                               I* workspace)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Stream
        hipStream_t stream = handle->stream;

        // Index base
        rocsparse_index_base base_A
            = info_C->csrgemm_info->mul ? descr_A->base : rocsparse_index_base_zero;
        rocsparse_index_base base_B
            = info_C->csrgemm_info->mul ? descr_B->base : rocsparse_index_base_zero;
        rocsparse_index_base base_D
            = info_C->csrgemm_info->add ? descr_D->base : rocsparse_index_base_zero;

#define BSRGEMM_BLOCKSIZE 256
#define BSRGEMM_CHUNKSIZE 2
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::bsrgemm_block_per_row_multipass<BSRGEMM_BLOCKSIZE, BSRGEMM_CHUNKSIZE, 32>),
            dim3(group_size[0]),
            dim3(BSRGEMM_BLOCKSIZE),
            0,
            stream,
            dir,
            nb,
            block_dim,
            &group_offset[0],
            perm,
            ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, alpha_device_host),
            bsr_row_ptr_A,
            bsr_col_ind_A,
            bsr_val_A,
            bsr_row_ptr_B,
            bsr_col_ind_B,
            bsr_val_B,
            ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, beta_device_host),
            bsr_row_ptr_D,
            bsr_col_ind_D,
            bsr_val_D,
            bsr_row_ptr_C,
            bsr_col_ind_C,
            bsr_val_C,
            workspace,
            base_A,
            base_B,
            descr_C->base,
            base_D,
            info_C->csrgemm_info->mul,
            info_C->csrgemm_info->add,
            handle->pointer_mode == rocsparse_pointer_mode_host);
#undef BSRGEMM_BLOCKSIZE
#undef BSRGEMM_CHUNKSIZE

        return rocsparse_status_success;
    }
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse::bsrgemm_calc_template_dispatch(rocsparse_handle    handle,
                                                           rocsparse_direction dir,
                                                           rocsparse_operation trans_A,
                                                           rocsparse_operation trans_B,
                                                           J                   mb,
                                                           J                   nb,
                                                           J                   kb,
                                                           J                   block_dim,
                                                           const T*            alpha_device_host,
                                                           const rocsparse_mat_descr descr_A,
                                                           I                         nnzb_A,
                                                           const T*                  bsr_val_A,
                                                           const I*                  bsr_row_ptr_A,
                                                           const J*                  bsr_col_ind_A,
                                                           const rocsparse_mat_descr descr_B,
                                                           I                         nnzb_B,
                                                           const T*                  bsr_val_B,
                                                           const I*                  bsr_row_ptr_B,
                                                           const J*                  bsr_col_ind_B,
                                                           const T* beta_device_host,
                                                           const rocsparse_mat_descr descr_D,
                                                           I                         nnzb_D,
                                                           const T*                  bsr_val_D,
                                                           const I*                  bsr_row_ptr_D,
                                                           const J*                  bsr_col_ind_D,
                                                           const rocsparse_mat_descr descr_C,
                                                           T*                        bsr_val_C,
                                                           const I*                  bsr_row_ptr_C,
                                                           J*                        bsr_col_ind_C,
                                                           const rocsparse_mat_info  info_C,
                                                           void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Stream
    hipStream_t stream = handle->stream;

    // Temporary buffer
    char* buffer = reinterpret_cast<char*>(temp_buffer);

    // rocprim buffer
    size_t rocprim_size;
    void*  rocprim_buffer;

    // Determine maximum non-zero entries per row of all rows
    J* workspace1 = reinterpret_cast<J*>(buffer);

#define BSRGEMM_DIM 256
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrgemm_max_row_nnz_part1<BSRGEMM_DIM>),
                                       dim3(BSRGEMM_DIM),
                                       dim3(BSRGEMM_DIM),
                                       0,
                                       stream,
                                       mb,
                                       bsr_row_ptr_C,
                                       workspace1);

    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrgemm_max_row_nnz_part2<BSRGEMM_DIM>),
                                       dim3(1),
                                       dim3(BSRGEMM_DIM),
                                       0,
                                       stream,
                                       workspace1);
#undef BSRGEMM_DIM

    J nnzb_max;
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(&nnzb_max, workspace1, sizeof(J), hipMemcpyDeviceToHost, stream));

    // Wait for host transfer to finish
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    // Group offset buffer
    J* d_group_offset = reinterpret_cast<J*>(buffer);
    buffer += sizeof(J) * 256;

    // Group size buffer
    J h_group_size[BSRGEMM_MAXGROUPS];

    // Initialize group sizes with zero
    memset(&h_group_size[0], 0, sizeof(J) * BSRGEMM_MAXGROUPS);

    // Permutation array
    J* d_perm = nullptr;

    // If maximum of row nnzb exceeds 8, we process the rows in groups of
    // similar sized row nnzb
    if(nnzb_max > 8 && block_dim <= 8)
    {
        // Group size buffer
        J* d_group_size = reinterpret_cast<J*>(buffer);
        buffer += sizeof(J) * 256 * BSRGEMM_MAXGROUPS;

        // Permutation temporary arrays
        J* tmp_vals = reinterpret_cast<J*>(buffer);
        buffer += ((sizeof(J) * mb - 1) / 256 + 1) * 256;

        J* tmp_perm = reinterpret_cast<J*>(buffer);
        buffer += ((sizeof(J) * mb - 1) / 256 + 1) * 256;

        int* tmp_keys = reinterpret_cast<int*>(buffer);
        buffer += ((sizeof(int) * mb - 1) / 256 + 1) * 256;

        int* tmp_groups = reinterpret_cast<int*>(buffer);
        buffer += ((sizeof(int) * mb - 1) / 256 + 1) * 256;

        // Determine number of rows per group
#define BSRGEMM_DIM 256
        if(block_dim == 2)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::bsrgemm_group_reduce_part2<BSRGEMM_DIM, BSRGEMM_MAXGROUPS, 2, T>),
                dim3(BSRGEMM_DIM),
                dim3(BSRGEMM_DIM),
                0,
                stream,
                mb,
                bsr_row_ptr_C,
                d_group_size,
                tmp_groups);
        }
        else
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::bsrgemm_group_reduce_part2<BSRGEMM_DIM, BSRGEMM_MAXGROUPS, 8, T>),
                dim3(BSRGEMM_DIM),
                dim3(BSRGEMM_DIM),
                0,
                stream,
                mb,
                bsr_row_ptr_C,
                d_group_size,
                tmp_groups);
        }

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::bsrgemm_group_reduce_part3<BSRGEMM_DIM, BSRGEMM_MAXGROUPS>),
            dim3(1),
            dim3(BSRGEMM_DIM),
            0,
            stream,
            d_group_size);
#undef BSRGEMM_DIM

        // Exclusive sum to obtain group offsets
        rocprim_buffer = reinterpret_cast<void*>(buffer);
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::exclusive_scan_buffer_size<J, J>(
            handle, static_cast<J>(0), BSRGEMM_MAXGROUPS, &rocprim_size)));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::exclusive_scan(handle,
                                                                        d_group_size,
                                                                        d_group_offset,
                                                                        static_cast<J>(0),
                                                                        BSRGEMM_MAXGROUPS,
                                                                        rocprim_size,
                                                                        rocprim_buffer));

        // Copy group sizes to host
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(&h_group_size,
                                           d_group_size,
                                           sizeof(J) * BSRGEMM_MAXGROUPS,
                                           hipMemcpyDeviceToHost,
                                           stream));

        // Wait for host transfer to finish
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

        // Create identity permutation for group access
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::create_identity_permutation_template(handle, mb, tmp_perm));

        rocsparse::primitives::double_buffer<int> d_keys(tmp_groups, tmp_keys);
        rocsparse::primitives::double_buffer<J>   d_vals(tmp_perm, tmp_vals);

        uint32_t startbit = 0;
        uint32_t endbit   = rocsparse::clz(BSRGEMM_MAXGROUPS);

        // Sort pairs (by groups)
        rocprim_buffer = reinterpret_cast<void*>(buffer);
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::radix_sort_pairs_buffer_size<int, J>(
            handle, mb, startbit, endbit, &rocprim_size)));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::radix_sort_pairs(
            handle, d_keys, d_vals, mb, startbit, endbit, rocprim_size, rocprim_buffer));

        d_perm = d_vals.current();

        // Release tmp_groups buffer
        buffer -= ((sizeof(int) * mb - 1) / 256 + 1) * 256;

        // Release tmp_keys buffer
        buffer -= ((sizeof(int) * mb - 1) / 256 + 1) * 256;
    }
    else
    {
        // First group processes all rows
        h_group_size[0] = mb;
        RETURN_IF_HIP_ERROR(hipMemsetAsync(d_group_offset, 0, sizeof(J), stream));
    }

    I* workspace2 = reinterpret_cast<I*>(buffer);

    if(block_dim == 2)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrgemm_calc_2x2_template(handle,
                                                                       dir,
                                                                       trans_A,
                                                                       trans_B,
                                                                       mb,
                                                                       nb,
                                                                       kb,
                                                                       block_dim,
                                                                       alpha_device_host,
                                                                       descr_A,
                                                                       nnzb_A,
                                                                       bsr_val_A,
                                                                       bsr_row_ptr_A,
                                                                       bsr_col_ind_A,
                                                                       descr_B,
                                                                       nnzb_B,
                                                                       bsr_val_B,
                                                                       bsr_row_ptr_B,
                                                                       bsr_col_ind_B,
                                                                       beta_device_host,
                                                                       descr_D,
                                                                       nnzb_D,
                                                                       bsr_val_D,
                                                                       bsr_row_ptr_D,
                                                                       bsr_col_ind_D,
                                                                       descr_C,
                                                                       bsr_val_C,
                                                                       bsr_row_ptr_C,
                                                                       bsr_col_ind_C,
                                                                       info_C,
                                                                       &h_group_size[0],
                                                                       d_group_offset,
                                                                       d_perm,
                                                                       workspace2));
        return rocsparse_status_success;
    }
    else if(block_dim <= 4)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrgemm_calc_3_4_template(handle,
                                                                       dir,
                                                                       trans_A,
                                                                       trans_B,
                                                                       mb,
                                                                       nb,
                                                                       kb,
                                                                       block_dim,
                                                                       alpha_device_host,
                                                                       descr_A,
                                                                       nnzb_A,
                                                                       bsr_val_A,
                                                                       bsr_row_ptr_A,
                                                                       bsr_col_ind_A,
                                                                       descr_B,
                                                                       nnzb_B,
                                                                       bsr_val_B,
                                                                       bsr_row_ptr_B,
                                                                       bsr_col_ind_B,
                                                                       beta_device_host,
                                                                       descr_D,
                                                                       nnzb_D,
                                                                       bsr_val_D,
                                                                       bsr_row_ptr_D,
                                                                       bsr_col_ind_D,
                                                                       descr_C,
                                                                       bsr_val_C,
                                                                       bsr_row_ptr_C,
                                                                       bsr_col_ind_C,
                                                                       info_C,
                                                                       &h_group_size[0],
                                                                       d_group_offset,
                                                                       d_perm,
                                                                       workspace2));
        return rocsparse_status_success;
    }
    else if(block_dim <= 8)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrgemm_calc_5_8_template(handle,
                                                                       dir,
                                                                       trans_A,
                                                                       trans_B,
                                                                       mb,
                                                                       nb,
                                                                       kb,
                                                                       block_dim,
                                                                       alpha_device_host,
                                                                       descr_A,
                                                                       nnzb_A,
                                                                       bsr_val_A,
                                                                       bsr_row_ptr_A,
                                                                       bsr_col_ind_A,
                                                                       descr_B,
                                                                       nnzb_B,
                                                                       bsr_val_B,
                                                                       bsr_row_ptr_B,
                                                                       bsr_col_ind_B,
                                                                       beta_device_host,
                                                                       descr_D,
                                                                       nnzb_D,
                                                                       bsr_val_D,
                                                                       bsr_row_ptr_D,
                                                                       bsr_col_ind_D,
                                                                       descr_C,
                                                                       bsr_val_C,
                                                                       bsr_row_ptr_C,
                                                                       bsr_col_ind_C,
                                                                       info_C,
                                                                       &h_group_size[0],
                                                                       d_group_offset,
                                                                       d_perm,
                                                                       workspace2));
        return rocsparse_status_success;
    }
    else if(block_dim <= 16)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrgemm_calc_9_16_template(handle,
                                                                        dir,
                                                                        trans_A,
                                                                        trans_B,
                                                                        mb,
                                                                        nb,
                                                                        kb,
                                                                        block_dim,
                                                                        alpha_device_host,
                                                                        descr_A,
                                                                        nnzb_A,
                                                                        bsr_val_A,
                                                                        bsr_row_ptr_A,
                                                                        bsr_col_ind_A,
                                                                        descr_B,
                                                                        nnzb_B,
                                                                        bsr_val_B,
                                                                        bsr_row_ptr_B,
                                                                        bsr_col_ind_B,
                                                                        beta_device_host,
                                                                        descr_D,
                                                                        nnzb_D,
                                                                        bsr_val_D,
                                                                        bsr_row_ptr_D,
                                                                        bsr_col_ind_D,
                                                                        descr_C,
                                                                        bsr_val_C,
                                                                        bsr_row_ptr_C,
                                                                        bsr_col_ind_C,
                                                                        info_C,
                                                                        &h_group_size[0],
                                                                        d_group_offset,
                                                                        d_perm,
                                                                        workspace2));
        return rocsparse_status_success;
    }
    else if(block_dim <= 32)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrgemm_calc_17_32_template(handle,
                                                                         dir,
                                                                         trans_A,
                                                                         trans_B,
                                                                         mb,
                                                                         nb,
                                                                         kb,
                                                                         block_dim,
                                                                         alpha_device_host,
                                                                         descr_A,
                                                                         nnzb_A,
                                                                         bsr_val_A,
                                                                         bsr_row_ptr_A,
                                                                         bsr_col_ind_A,
                                                                         descr_B,
                                                                         nnzb_B,
                                                                         bsr_val_B,
                                                                         bsr_row_ptr_B,
                                                                         bsr_col_ind_B,
                                                                         beta_device_host,
                                                                         descr_D,
                                                                         nnzb_D,
                                                                         bsr_val_D,
                                                                         bsr_row_ptr_D,
                                                                         bsr_col_ind_D,
                                                                         descr_C,
                                                                         bsr_val_C,
                                                                         bsr_row_ptr_C,
                                                                         bsr_col_ind_C,
                                                                         info_C,
                                                                         &h_group_size[0],
                                                                         d_group_offset,
                                                                         d_perm,
                                                                         workspace2));
        return rocsparse_status_success;
    }
    else
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }
}

#define INSTANTIATE(I, J, T)                                             \
    template rocsparse_status rocsparse::bsrgemm_calc_template_dispatch( \
        rocsparse_handle          handle,                                \
        rocsparse_direction       dir,                                   \
        rocsparse_operation       trans_A,                               \
        rocsparse_operation       trans_B,                               \
        J                         mb,                                    \
        J                         nb,                                    \
        J                         kb,                                    \
        J                         block_dim,                             \
        const T*                  alpha_device_host,                     \
        const rocsparse_mat_descr descr_A,                               \
        I                         nnzb_A,                                \
        const T*                  bsr_val_A,                             \
        const I*                  bsr_row_ptr_A,                         \
        const J*                  bsr_col_ind_A,                         \
        const rocsparse_mat_descr descr_B,                               \
        I                         nnzb_B,                                \
        const T*                  bsr_val_B,                             \
        const I*                  bsr_row_ptr_B,                         \
        const J*                  bsr_col_ind_B,                         \
        const T*                  beta_device_host,                      \
        const rocsparse_mat_descr descr_D,                               \
        I                         nnzb_D,                                \
        const T*                  bsr_val_D,                             \
        const I*                  bsr_row_ptr_D,                         \
        const J*                  bsr_col_ind_D,                         \
        const rocsparse_mat_descr descr_C,                               \
        T*                        bsr_val_C,                             \
        const I*                  bsr_row_ptr_C,                         \
        J*                        bsr_col_ind_C,                         \
        const rocsparse_mat_info  info_C,                                \
        void*                     temp_buffer)

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);

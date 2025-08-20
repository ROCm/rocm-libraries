/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_csrgemm_nnz_calc.hpp"
#include "../conversion/rocsparse_identity.hpp"
#include "csrgemm_device.h"
#include "rocsparse_control.hpp"
#include "rocsparse_csrgemm.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_primitives.hpp"

namespace rocsparse
{
    template <uint32_t HASHSIZE, typename J>
    constexpr uint32_t csrgemm_nnz_block_per_row_shared_memory_size()
    {
        return (sizeof(J) * HASHSIZE);
    }
}

template <typename I, typename J>
rocsparse_status rocsparse::csrgemm_nnz_calc(rocsparse_handle          handle,
                                             rocsparse_operation       trans_A,
                                             rocsparse_operation       trans_B,
                                             J                         m,
                                             J                         n,
                                             J                         k,
                                             const rocsparse_mat_descr descr_A,
                                             I                         nnz_A,
                                             const I*                  csr_row_ptr_A,
                                             const J*                  csr_col_ind_A,
                                             const rocsparse_mat_descr descr_B,
                                             I                         nnz_B,
                                             const I*                  csr_row_ptr_B,
                                             const J*                  csr_col_ind_B,
                                             const rocsparse_mat_descr descr_D,
                                             I                         nnz_D,
                                             const I*                  csr_row_ptr_D,
                                             const J*                  csr_col_ind_D,
                                             const rocsparse_mat_descr descr_C,
                                             I*                        csr_row_ptr_C,
                                             I*                        nnz_C,
                                             const rocsparse_mat_info  info_C,
                                             void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Stream
    hipStream_t stream = handle->stream;

    // Index base
    rocsparse_index_base base_A
        = info_C->csrgemm_info->mul ? descr_A->base : rocsparse_index_base_zero;
    rocsparse_index_base base_B
        = info_C->csrgemm_info->mul ? descr_B->base : rocsparse_index_base_zero;
    rocsparse_index_base base_D = info_C->csrgemm_info->add
                                      ? ((descr_D) ? descr_D->base : rocsparse_index_base_zero)
                                      : rocsparse_index_base_zero;

    bool mul = info_C->csrgemm_info->mul;
    bool add = info_C->csrgemm_info->add;

    // Temporary buffer
    char* buffer = reinterpret_cast<char*>(temp_buffer);

    // rocprim buffer
    size_t rocprim_size;
    void*  rocprim_buffer;

    // Compute number of intermediate products for each row
#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 8
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
        (rocsparse::csrgemm_intermediate_products<CSRGEMM_DIM, CSRGEMM_SUB>),
        dim3((m - 1) / (CSRGEMM_DIM / CSRGEMM_SUB) + 1),
        dim3(CSRGEMM_DIM),
        0,
        stream,
        m,
        csr_row_ptr_A,
        csr_col_ind_A,
        csr_row_ptr_B,
        csr_row_ptr_D,
        csr_row_ptr_C,
        base_A,
        mul,
        add);
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM

    // Determine maximum of all intermediate products
    RETURN_IF_ROCSPARSE_ERROR(
        (rocsparse::primitives::find_max_buffer_size<I, I>(handle, m, &rocprim_size)));
    rocprim_buffer = reinterpret_cast<void*>(buffer);
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::find_max(
        handle, csr_row_ptr_C, csr_row_ptr_C + m, m, rocprim_size, rocprim_buffer));

    I int_max;
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(&int_max, csr_row_ptr_C + m, sizeof(I), hipMemcpyDeviceToHost, stream));
    // Wait for host transfer to finish
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    // Group offset buffer
    J* d_group_offset = reinterpret_cast<J*>(buffer);
    buffer += sizeof(J) * 256;

    // Group size buffer
    J h_group_size[CSRGEMM_MAXGROUPS];

    // Initialize group sizes with zero
    memset(&h_group_size[0], 0, sizeof(J) * CSRGEMM_MAXGROUPS);

    // Permutation array
    J* d_perm = nullptr;

    // If maximum of intermediate products exceeds 32, we process the rows in groups of
    // similar sized intermediate products
    if(int_max > 32)
    {
        // Group size buffer
        J* d_group_size = reinterpret_cast<J*>(buffer);
        buffer += sizeof(J) * 256 * CSRGEMM_MAXGROUPS;

#define CSRGEMM_DIM 256
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_group_reduce_part1<CSRGEMM_DIM, CSRGEMM_MAXGROUPS>),
            dim3(CSRGEMM_DIM),
            dim3(CSRGEMM_DIM),
            0,
            stream,
            m,
            csr_row_ptr_C,
            d_group_size,
            handle->shared_mem_per_block_optin);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_group_reduce_part3<CSRGEMM_DIM, CSRGEMM_MAXGROUPS>),
            dim3(1),
            dim3(CSRGEMM_DIM),
            0,
            stream,
            d_group_size);
#undef CSRGEMM_DIM

        // Exclusive sum to obtain group offsets
        rocprim_buffer = reinterpret_cast<void*>(buffer);
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::exclusive_scan_buffer_size<J, J>(
            handle, static_cast<J>(0), CSRGEMM_MAXGROUPS, &rocprim_size)));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::exclusive_scan(handle,
                                                                        d_group_size,
                                                                        d_group_offset,
                                                                        static_cast<J>(0),
                                                                        CSRGEMM_MAXGROUPS,
                                                                        rocprim_size,
                                                                        rocprim_buffer));

        // Copy group sizes to host
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(&h_group_size,
                                           d_group_size,
                                           sizeof(J) * CSRGEMM_MAXGROUPS,
                                           hipMemcpyDeviceToHost,
                                           stream));

        // Wait for host transfer to finish
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

        // Permutation temporary arrays
        J* tmp_vals = reinterpret_cast<J*>(buffer);
        buffer += ((sizeof(J) * m - 1) / 256 + 1) * 256;

        J* tmp_perm = reinterpret_cast<J*>(buffer);
        buffer += ((sizeof(J) * m - 1) / 256 + 1) * 256;

        I* tmp_keys = reinterpret_cast<I*>(buffer);
        buffer += ((sizeof(I) * m - 1) / 256 + 1) * 256;

        // Create identity permutation for group access
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::create_identity_permutation_template(handle, m, tmp_perm));

        rocsparse::primitives::double_buffer<I> d_keys(csr_row_ptr_C, tmp_keys);
        rocsparse::primitives::double_buffer<J> d_vals(tmp_perm, tmp_vals);

        uint32_t startbit = 0;
        uint32_t endbit   = rocsparse::clz(CSRGEMM_MAXGROUPS);

        // Sort pairs (by groups)
        rocprim_buffer = reinterpret_cast<void*>(buffer);
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::radix_sort_pairs_buffer_size<I, J>(
            handle, m, startbit, endbit, &rocprim_size)));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::radix_sort_pairs(
            handle, d_keys, d_vals, m, startbit, endbit, rocprim_size, rocprim_buffer));

        d_perm = d_vals.current();

        // Release tmp_keys buffer
        buffer -= ((sizeof(I) * m - 1) / 256 + 1) * 256;
    }
    else
    {
        // First group processes all rows
        h_group_size[0] = m;
        RETURN_IF_HIP_ERROR(hipMemsetAsync(d_group_offset, 0, sizeof(J), stream));
    }

    // Compute non-zero entries per row for each group

    // Group 0: 0 - 32 intermediate products
    if(h_group_size[0] > 0)
    {
#define CSRGEMM_DIM 128
#define CSRGEMM_SUB 4
#define CSRGEMM_HASHSIZE 32
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_nnz_wf_per_row<CSRGEMM_DIM,
                                               CSRGEMM_SUB,
                                               CSRGEMM_HASHSIZE,
                                               CSRGEMM_NNZ_HASH>),
            dim3((h_group_size[0] - 1) / (CSRGEMM_DIM / CSRGEMM_SUB) + 1),
            dim3(CSRGEMM_DIM),
            0,
            stream,
            h_group_size[0],
            &d_group_offset[0],
            d_perm,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_row_ptr_C,
            base_A,
            base_B,
            base_D,
            mul,
            add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 1: 33 - 64 intermediate products
    if(h_group_size[1] > 0)
    {
#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 8
#define CSRGEMM_HASHSIZE 64
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_nnz_wf_per_row<CSRGEMM_DIM,
                                               CSRGEMM_SUB,
                                               CSRGEMM_HASHSIZE,
                                               CSRGEMM_NNZ_HASH>),
            dim3((h_group_size[1] - 1) / (CSRGEMM_DIM / CSRGEMM_SUB) + 1),
            dim3(CSRGEMM_DIM),
            0,
            stream,
            h_group_size[1],
            &d_group_offset[1],
            d_perm,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_row_ptr_C,
            base_A,
            base_B,
            base_D,
            mul,
            add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 2: 65 - 512 intermediate products
    if(h_group_size[2] > 0)
    {
#define CSRGEMM_DIM 128
#define CSRGEMM_SUB 8
#define CSRGEMM_HASHSIZE 512
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_nnz_block_per_row<CSRGEMM_DIM,
                                                  CSRGEMM_SUB,
                                                  CSRGEMM_HASHSIZE,
                                                  CSRGEMM_NNZ_HASH>),
            dim3(h_group_size[2]),
            dim3(CSRGEMM_DIM),
            (csrgemm_nnz_block_per_row_shared_memory_size<CSRGEMM_HASHSIZE, J>()),
            stream,
            &d_group_offset[2],
            d_perm,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_row_ptr_C,
            base_A,
            base_B,
            base_D,
            info_C->csrgemm_info->mul,
            info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 3: 513 - 1024 intermediate products
    if(h_group_size[3] > 0)
    {
#define CSRGEMM_DIM 128
#define CSRGEMM_SUB 8
#define CSRGEMM_HASHSIZE 1024
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_nnz_block_per_row<CSRGEMM_DIM,
                                                  CSRGEMM_SUB,
                                                  CSRGEMM_HASHSIZE,
                                                  CSRGEMM_NNZ_HASH>),
            dim3(h_group_size[3]),
            dim3(CSRGEMM_DIM),
            (csrgemm_nnz_block_per_row_shared_memory_size<CSRGEMM_HASHSIZE, J>()),
            stream,
            &d_group_offset[3],
            d_perm,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_row_ptr_C,
            base_A,
            base_B,
            base_D,
            info_C->csrgemm_info->mul,
            info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 4: 1025 - 2048 intermediate products
    if(h_group_size[4] > 0)
    {
#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 16
#define CSRGEMM_HASHSIZE 2048
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_nnz_block_per_row<CSRGEMM_DIM,
                                                  CSRGEMM_SUB,
                                                  CSRGEMM_HASHSIZE,
                                                  CSRGEMM_NNZ_HASH>),
            dim3(h_group_size[4]),
            dim3(CSRGEMM_DIM),
            (csrgemm_nnz_block_per_row_shared_memory_size<CSRGEMM_HASHSIZE, J>()),
            stream,
            &d_group_offset[4],
            d_perm,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_row_ptr_C,
            base_A,
            base_B,
            base_D,
            info_C->csrgemm_info->mul,
            info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 5: 2049 - 4096 intermediate products
    if(h_group_size[5] > 0)
    {
#define CSRGEMM_DIM 512
#define CSRGEMM_SUB 16
#define CSRGEMM_HASHSIZE 4096
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_nnz_block_per_row<CSRGEMM_DIM,
                                                  CSRGEMM_SUB,
                                                  CSRGEMM_HASHSIZE,
                                                  CSRGEMM_NNZ_HASH>),
            dim3(h_group_size[5]),
            dim3(CSRGEMM_DIM),
            (csrgemm_nnz_block_per_row_shared_memory_size<CSRGEMM_HASHSIZE, J>()),
            stream,
            &d_group_offset[5],
            d_perm,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_row_ptr_C,
            base_A,
            base_B,
            base_D,
            info_C->csrgemm_info->mul,
            info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 6: 4097 - 8192 intermediate products
    if(h_group_size[6] > 0)
    {
#define CSRGEMM_DIM 1024
#define CSRGEMM_SUB 32
#define CSRGEMM_HASHSIZE 8192
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_nnz_block_per_row<CSRGEMM_DIM,
                                                  CSRGEMM_SUB,
                                                  CSRGEMM_HASHSIZE,
                                                  CSRGEMM_NNZ_HASH>),
            dim3(h_group_size[6]),
            dim3(CSRGEMM_DIM),
            (csrgemm_nnz_block_per_row_shared_memory_size<CSRGEMM_HASHSIZE, J>()),
            stream,
            &d_group_offset[6],
            d_perm,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_row_ptr_C,
            base_A,
            base_B,
            base_D,
            info_C->csrgemm_info->mul,
            info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 7: 8193 - 16384 intermediate products
    if(h_group_size[7] > 0)
    {
#define CSRGEMM_DIM 1024
#define CSRGEMM_SUB 32
#define CSRGEMM_HASHSIZE 16384
        RETURN_IF_HIP_ERROR(hipFuncSetAttribute(
            (const void*)rocsparse::csrgemm_nnz_block_per_row<CSRGEMM_DIM,
                                                              CSRGEMM_SUB,
                                                              CSRGEMM_HASHSIZE,
                                                              CSRGEMM_NNZ_HASH,
                                                              I,
                                                              J>,
            hipFuncAttributeMaxDynamicSharedMemorySize,
            csrgemm_nnz_block_per_row_shared_memory_size<CSRGEMM_HASHSIZE, J>()));

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_nnz_block_per_row<CSRGEMM_DIM,
                                                  CSRGEMM_SUB,
                                                  CSRGEMM_HASHSIZE,
                                                  CSRGEMM_NNZ_HASH>),
            dim3(h_group_size[7]),
            dim3(CSRGEMM_DIM),
            (csrgemm_nnz_block_per_row_shared_memory_size<CSRGEMM_HASHSIZE, J>()),
            handle->stream,
            &d_group_offset[7],
            d_perm,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_row_ptr_C,
            base_A,
            base_B,
            base_D,
            info_C->csrgemm_info->mul,
            info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 8: 16385 - 32768 intermediate products
    if(h_group_size[8] > 0)
    {
#define CSRGEMM_DIM 1024
#define CSRGEMM_SUB 32
#define CSRGEMM_HASHSIZE 32768
        RETURN_IF_HIP_ERROR(hipFuncSetAttribute(
            (const void*)rocsparse::csrgemm_nnz_block_per_row<CSRGEMM_DIM,
                                                              CSRGEMM_SUB,
                                                              CSRGEMM_HASHSIZE,
                                                              CSRGEMM_NNZ_HASH,
                                                              I,
                                                              J>,
            hipFuncAttributeMaxDynamicSharedMemorySize,
            csrgemm_nnz_block_per_row_shared_memory_size<CSRGEMM_HASHSIZE, J>()));

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_nnz_block_per_row<CSRGEMM_DIM,
                                                  CSRGEMM_SUB,
                                                  CSRGEMM_HASHSIZE,
                                                  CSRGEMM_NNZ_HASH>),
            dim3(h_group_size[8]),
            dim3(CSRGEMM_DIM),
            (csrgemm_nnz_block_per_row_shared_memory_size<CSRGEMM_HASHSIZE, J>()),
            handle->stream,
            &d_group_offset[8],
            d_perm,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_row_ptr_C,
            base_A,
            base_B,
            base_D,
            info_C->csrgemm_info->mul,
            info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 9: 32769 - 65536 intermediate products
    if(h_group_size[9] > 0)
    {
#define CSRGEMM_DIM 1024
#define CSRGEMM_SUB 32
#define CSRGEMM_HASHSIZE 65536
        RETURN_IF_HIP_ERROR(hipFuncSetAttribute(
            (const void*)rocsparse::csrgemm_nnz_block_per_row<CSRGEMM_DIM,
                                                              CSRGEMM_SUB,
                                                              CSRGEMM_HASHSIZE,
                                                              CSRGEMM_NNZ_HASH,
                                                              I,
                                                              J>,
            hipFuncAttributeMaxDynamicSharedMemorySize,
            csrgemm_nnz_block_per_row_shared_memory_size<CSRGEMM_HASHSIZE, J>()));

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_nnz_block_per_row<CSRGEMM_DIM,
                                                  CSRGEMM_SUB,
                                                  CSRGEMM_HASHSIZE,
                                                  CSRGEMM_NNZ_HASH>),
            dim3(h_group_size[9]),
            dim3(CSRGEMM_DIM),
            (csrgemm_nnz_block_per_row_shared_memory_size<CSRGEMM_HASHSIZE, J>()),
            handle->stream,
            &d_group_offset[9],
            d_perm,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_row_ptr_C,
            base_A,
            base_B,
            base_D,
            info_C->csrgemm_info->mul,
            info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 10: more than 65536 intermediate products or shared memory exceeded
    if(h_group_size[10] > 0)
    {
        // Matrices B and D must be sorted in order to run this path
        if(descr_B->storage_mode == rocsparse_storage_mode_unsorted
           || (info_C->csrgemm_info->add ? descr_D->storage_mode == rocsparse_storage_mode_unsorted
                                         : false))
        {
            return rocsparse_status_requires_sorted_storage;
        }

#define CSRGEMM_DIM 512
#define CSRGEMM_SUB 16
#define CSRGEMM_CHUNKSIZE 2048
        I* workspace_B = nullptr;

        if(info_C->csrgemm_info->mul == true)
        {
            // Allocate additional buffer for C = alpha * A * B
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMallocAsync(&workspace_B, sizeof(I) * nnz_A, handle->stream));
        }

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::
                 csrgemm_nnz_block_per_row_multipass<CSRGEMM_DIM, CSRGEMM_SUB, CSRGEMM_CHUNKSIZE>),
            dim3(h_group_size[10]),
            dim3(CSRGEMM_DIM),
            0,
            stream,
            n,
            &d_group_offset[10],
            d_perm,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_row_ptr_C,
            workspace_B,
            base_A,
            base_B,
            base_D,
            mul,
            add);

        if(info_C->csrgemm_info->mul == true)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(workspace_B, handle->stream));
        }
#undef CSRGEMM_CHUNKSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Exclusive sum to obtain row pointers of C
    rocprim_buffer = reinterpret_cast<void*>(buffer);
    RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::exclusive_scan_buffer_size<I, I>(
        handle, static_cast<I>(descr_C->base), m + 1, &rocprim_size)));
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::exclusive_scan(handle,
                                                                    csr_row_ptr_C,
                                                                    csr_row_ptr_C,
                                                                    static_cast<I>(descr_C->base),
                                                                    m + 1,
                                                                    rocprim_size,
                                                                    rocprim_buffer));

    // Store nnz of C
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        RETURN_IF_HIP_ERROR(
            hipMemcpyAsync(nnz_C, csr_row_ptr_C + m, sizeof(I), hipMemcpyDeviceToDevice, stream));

        // Adjust nnz by index base
        if(descr_C->base == rocsparse_index_base_one)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::csrgemm_index_base<1>), dim3(1), dim3(1), 0, stream, nnz_C);
        }
    }
    else
    {
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            nnz_C, csr_row_ptr_C + m, sizeof(I), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

        // Adjust nnz by index base
        *nnz_C -= descr_C->base;
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(I, J)                                                                          \
    template rocsparse_status rocsparse::csrgemm_nnz_calc(rocsparse_handle          handle,        \
                                                          rocsparse_operation       trans_A,       \
                                                          rocsparse_operation       trans_B,       \
                                                          J                         m,             \
                                                          J                         n,             \
                                                          J                         k,             \
                                                          const rocsparse_mat_descr descr_A,       \
                                                          I                         nnz_A,         \
                                                          const I*                  csr_row_ptr_A, \
                                                          const J*                  csr_col_ind_A, \
                                                          const rocsparse_mat_descr descr_B,       \
                                                          I                         nnz_B,         \
                                                          const I*                  csr_row_ptr_B, \
                                                          const J*                  csr_col_ind_B, \
                                                          const rocsparse_mat_descr descr_D,       \
                                                          I                         nnz_D,         \
                                                          const I*                  csr_row_ptr_D, \
                                                          const J*                  csr_col_ind_D, \
                                                          const rocsparse_mat_descr descr_C,       \
                                                          I*                        csr_row_ptr_C, \
                                                          I*                        nnz_C,         \
                                                          const rocsparse_mat_info  info_C,        \
                                                          void*                     temp_buffer)

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int32_t, int64_t);
INSTANTIATE(int64_t, int32_t);
INSTANTIATE(int64_t, int64_t);

#undef INSTANTIATE

/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_csrgemm_calc.hpp"
#include "../conversion/rocsparse_identity.hpp"
#include "control.h"
#include "csrgemm_device.h"
#include "internal/extra/rocsparse_csrgemm.h"
#include "rocsparse_csrgemm.hpp"
#include "utility.h"

#include "rocsparse_primitives.h"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t HASHSIZE,
              uint32_t HASHVAL,
              typename I,
              typename J,
              typename T,
              typename U>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_fill_wf_per_row(J m,
                                 J nk,
                                 const J* __restrict__ offset,
                                 const J* __restrict__ perm,
                                 U alpha_device_host,
                                 const I* __restrict__ csr_row_ptr_A,
                                 const J* __restrict__ csr_col_ind_A,
                                 const T* __restrict__ csr_val_A,
                                 const I* __restrict__ csr_row_ptr_B,
                                 const J* __restrict__ csr_col_ind_B,
                                 const T* __restrict__ csr_val_B,
                                 U beta_device_host,
                                 const I* __restrict__ csr_row_ptr_D,
                                 const J* __restrict__ csr_col_ind_D,
                                 const T* __restrict__ csr_val_D,
                                 const I* __restrict__ csr_row_ptr_C,
                                 J* __restrict__ csr_col_ind_C,
                                 T* __restrict__ csr_val_C,
                                 rocsparse_index_base idx_base_A,
                                 rocsparse_index_base idx_base_B,
                                 rocsparse_index_base idx_base_C,
                                 rocsparse_index_base idx_base_D,
                                 bool                 mul,
                                 bool                 add)
    {
        rocsparse::csrgemm_fill_wf_per_row_device<BLOCKSIZE, WFSIZE, HASHSIZE, HASHVAL>(
            m,
            nk,
            offset,
            perm,
            (mul == true) ? rocsparse::load_scalar_device_host(alpha_device_host)
                          : static_cast<T>(0),
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_val_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_val_B,
            (add == true) ? rocsparse::load_scalar_device_host(beta_device_host)
                          : static_cast<T>(0),
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_val_D,
            csr_row_ptr_C,
            csr_col_ind_C,
            csr_val_C,
            idx_base_A,
            idx_base_B,
            idx_base_C,
            idx_base_D,
            mul,
            add);
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t HASHSIZE,
              uint32_t HASHVAL,
              typename I,
              typename J,
              typename T,
              typename U>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_fill_block_per_row(J nk,
                                    const J* __restrict__ offset,
                                    const J* __restrict__ perm,
                                    U alpha_device_host,
                                    const I* __restrict__ csr_row_ptr_A,
                                    const J* __restrict__ csr_col_ind_A,
                                    const T* __restrict__ csr_val_A,
                                    const I* __restrict__ csr_row_ptr_B,
                                    const J* __restrict__ csr_col_ind_B,
                                    const T* __restrict__ csr_val_B,
                                    U beta_device_host,
                                    const I* __restrict__ csr_row_ptr_D,
                                    const J* __restrict__ csr_col_ind_D,
                                    const T* __restrict__ csr_val_D,
                                    const I* __restrict__ csr_row_ptr_C,
                                    J* __restrict__ csr_col_ind_C,
                                    T* __restrict__ csr_val_C,
                                    rocsparse_index_base idx_base_A,
                                    rocsparse_index_base idx_base_B,
                                    rocsparse_index_base idx_base_C,
                                    rocsparse_index_base idx_base_D,
                                    bool                 mul,
                                    bool                 add)
    {
        rocsparse::csrgemm_fill_block_per_row_device<BLOCKSIZE, WFSIZE, HASHSIZE, HASHVAL>(
            nk,
            offset,
            perm,
            (mul == true) ? rocsparse::load_scalar_device_host(alpha_device_host)
                          : static_cast<T>(0),
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_val_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_val_B,
            (add == true) ? rocsparse::load_scalar_device_host(beta_device_host)
                          : static_cast<T>(0),
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_val_D,
            csr_row_ptr_C,
            csr_col_ind_C,
            csr_val_C,
            idx_base_A,
            idx_base_B,
            idx_base_C,
            idx_base_D,
            mul,
            add);
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WFSIZE,
              uint32_t CHUNKSIZE,
              typename I,
              typename J,
              typename T,
              typename U>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_fill_block_per_row_multipass(J n,
                                              const J* __restrict__ offset,
                                              const J* __restrict__ perm,
                                              U alpha_device_host,
                                              const I* __restrict__ csr_row_ptr_A,
                                              const J* __restrict__ csr_col_ind_A,
                                              const T* __restrict__ csr_val_A,
                                              const I* __restrict__ csr_row_ptr_B,
                                              const J* __restrict__ csr_col_ind_B,
                                              const T* __restrict__ csr_val_B,
                                              U beta_device_host,
                                              const I* __restrict__ csr_row_ptr_D,
                                              const J* __restrict__ csr_col_ind_D,
                                              const T* __restrict__ csr_val_D,
                                              const I* __restrict__ csr_row_ptr_C,
                                              J* __restrict__ csr_col_ind_C,
                                              T* __restrict__ csr_val_C,
                                              I* __restrict__ workspace_B,
                                              rocsparse_index_base idx_base_A,
                                              rocsparse_index_base idx_base_B,
                                              rocsparse_index_base idx_base_C,
                                              rocsparse_index_base idx_base_D,
                                              bool                 mul,
                                              bool                 add)
    {
        rocsparse::csrgemm_fill_block_per_row_multipass_device<BLOCKSIZE, WFSIZE, CHUNKSIZE>(
            n,
            offset,
            perm,
            (mul == true) ? rocsparse::load_scalar_device_host(alpha_device_host)
                          : static_cast<T>(0),
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_val_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_val_B,
            (add == true) ? rocsparse::load_scalar_device_host(beta_device_host)
                          : static_cast<T>(0),
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_val_D,
            csr_row_ptr_C,
            csr_col_ind_C,
            csr_val_C,
            workspace_B,
            idx_base_A,
            idx_base_B,
            idx_base_C,
            idx_base_D,
            mul,
            add);
    }

    // Disable for rocsparse_double_complex, as well as double and rocsparse_float_complex
    // if I == J == int64_t, as required size would exceed available memory
    template <typename I,
              typename J,
              typename T,
              typename U,
              typename std::enable_if<
                  std::is_same<T, rocsparse_double_complex>::value
                      || (std::is_same<T, double>::value && std::is_same<I, int64_t>::value
                          && std::is_same<J, int64_t>::value)
                      || (std::is_same<T, rocsparse_float_complex>::value
                          && std::is_same<I, int64_t>::value && std::is_same<J, int64_t>::value),
                  int>::type
              = 0>
    static inline rocsparse_status csrgemm_launcher(rocsparse_handle     handle,
                                                    J                    group_size,
                                                    const J*             group_offset,
                                                    const J*             perm,
                                                    J                    m,
                                                    J                    n,
                                                    J                    k,
                                                    U                    alpha_device_host,
                                                    const I*             csr_row_ptr_A,
                                                    const J*             csr_col_ind_A,
                                                    const T*             csr_val_A,
                                                    const I*             csr_row_ptr_B,
                                                    const J*             csr_col_ind_B,
                                                    const T*             csr_val_B,
                                                    U                    beta_device_host,
                                                    const I*             csr_row_ptr_D,
                                                    const J*             csr_col_ind_D,
                                                    const T*             csr_val_D,
                                                    const I*             csr_row_ptr_C,
                                                    J*                   csr_col_ind_C,
                                                    T*                   csr_val_C,
                                                    rocsparse_index_base base_A,
                                                    rocsparse_index_base base_B,
                                                    rocsparse_index_base base_C,
                                                    rocsparse_index_base base_D,
                                                    bool                 mul,
                                                    bool                 add)
    {
        return rocsparse_status_internal_error;
    }

    template <typename I,
              typename J,
              typename T,
              typename U,
              typename std::enable_if<
                  std::is_same<T, float>::value
                      || (std::is_same<T, double>::value
                          && (std::is_same<I, int32_t>::value || std::is_same<J, int32_t>::value))
                      || (std::is_same<T, rocsparse_float_complex>::value
                          && (std::is_same<I, int32_t>::value || std::is_same<J, int32_t>::value)),
                  int>::type
              = 0>
    static inline rocsparse_status csrgemm_launcher(rocsparse_handle     handle,
                                                    J                    group_size,
                                                    const J*             group_offset,
                                                    const J*             perm,
                                                    J                    m,
                                                    J                    n,
                                                    J                    k,
                                                    U                    alpha_device_host,
                                                    const I*             csr_row_ptr_A,
                                                    const J*             csr_col_ind_A,
                                                    const T*             csr_val_A,
                                                    const I*             csr_row_ptr_B,
                                                    const J*             csr_col_ind_B,
                                                    const T*             csr_val_B,
                                                    U                    beta_device_host,
                                                    const I*             csr_row_ptr_D,
                                                    const J*             csr_col_ind_D,
                                                    const T*             csr_val_D,
                                                    const I*             csr_row_ptr_C,
                                                    J*                   csr_col_ind_C,
                                                    T*                   csr_val_C,
                                                    rocsparse_index_base base_A,
                                                    rocsparse_index_base base_B,
                                                    rocsparse_index_base base_C,
                                                    rocsparse_index_base base_D,
                                                    bool                 mul,
                                                    bool                 add)
    {
#define CSRGEMM_DIM 1024
#define CSRGEMM_SUB 64
#define CSRGEMM_HASHSIZE 4096
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_fill_block_per_row<CSRGEMM_DIM,
                                                   CSRGEMM_SUB,
                                                   CSRGEMM_HASHSIZE,
                                                   CSRGEMM_FLL_HASH>),
            dim3(group_size),
            dim3(CSRGEMM_DIM),
            0,
            handle->stream,
            rocsparse::max(k, n),
            group_offset,
            perm,
            alpha_device_host,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_val_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_val_B,
            beta_device_host,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_val_D,
            csr_row_ptr_C,
            csr_col_ind_C,
            csr_val_C,
            base_A,
            base_B,
            base_C,
            base_D,
            mul,
            add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM

        return rocsparse_status_success;
    }
}

template <typename I, typename J, typename T, typename U>
rocsparse_status rocsparse::csrgemm_calc_template(rocsparse_handle          handle,
                                                  rocsparse_operation       trans_A,
                                                  rocsparse_operation       trans_B,
                                                  J                         m,
                                                  J                         n,
                                                  J                         k,
                                                  U                         alpha_device_host,
                                                  const rocsparse_mat_descr descr_A,
                                                  I                         nnz_A,
                                                  const T*                  csr_val_A,
                                                  const I*                  csr_row_ptr_A,
                                                  const J*                  csr_col_ind_A,
                                                  const rocsparse_mat_descr descr_B,
                                                  I                         nnz_B,
                                                  const T*                  csr_val_B,
                                                  const I*                  csr_row_ptr_B,
                                                  const J*                  csr_col_ind_B,
                                                  U                         beta_device_host,
                                                  const rocsparse_mat_descr descr_D,
                                                  I                         nnz_D,
                                                  const T*                  csr_val_D,
                                                  const I*                  csr_row_ptr_D,
                                                  const J*                  csr_col_ind_D,
                                                  const rocsparse_mat_descr descr_C,
                                                  T*                        csr_val_C,
                                                  const I*                  csr_row_ptr_C,
                                                  J*                        csr_col_ind_C,
                                                  const rocsparse_mat_info  info_C,
                                                  void*                     temp_buffer)
{
    // Stream
    hipStream_t stream = handle->stream;

    // Index base
    rocsparse_index_base base_A
        = info_C->csrgemm_info->mul ? descr_A->base : rocsparse_index_base_zero;
    rocsparse_index_base base_B
        = info_C->csrgemm_info->mul ? descr_B->base : rocsparse_index_base_zero;
    rocsparse_index_base base_D
        = info_C->csrgemm_info->add ? descr_D->base : rocsparse_index_base_zero;

    // Flag for exceeding shared memory
    constexpr bool exceeding_smem
        = std::is_same<T, rocsparse_double_complex>::value
          || (std::is_same<T, double>::value && std::is_same<I, int64_t>::value
              && std::is_same<J, int64_t>::value)
          || (std::is_same<T, rocsparse_float_complex>::value && std::is_same<I, int64_t>::value
              && std::is_same<J, int64_t>::value);

    // Temporary buffer
    char* buffer = reinterpret_cast<char*>(temp_buffer);

    // rocprim buffer
    size_t rocprim_size;
    void*  rocprim_buffer;

    // Determine maximum non-zero entries per row of all rows
    J* workspace = reinterpret_cast<J*>(buffer);

#define CSRGEMM_DIM 256
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrgemm_max_row_nnz_part1<CSRGEMM_DIM>),
                                       dim3(CSRGEMM_DIM),
                                       dim3(CSRGEMM_DIM),
                                       0,
                                       stream,
                                       m,
                                       csr_row_ptr_C,
                                       workspace);

    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrgemm_max_row_nnz_part2<CSRGEMM_DIM>),
                                       dim3(1),
                                       dim3(CSRGEMM_DIM),
                                       0,
                                       stream,
                                       workspace);
#undef CSRGEMM_DIM

    J nnz_max;
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(&nnz_max, workspace, sizeof(J), hipMemcpyDeviceToHost, stream));

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

    // If maximum of row nnz exceeds 16, we process the rows in groups of
    // similar sized row nnz
    if(nnz_max > 16)
    {
        // Group size buffer
        J* d_group_size = reinterpret_cast<J*>(buffer);
        buffer += sizeof(J) * 256 * CSRGEMM_MAXGROUPS;

        // Permutation temporary arrays
        J* tmp_vals = reinterpret_cast<J*>(buffer);
        buffer += ((sizeof(J) * m - 1) / 256 + 1) * 256;

        J* tmp_perm = reinterpret_cast<J*>(buffer);
        buffer += ((sizeof(J) * m - 1) / 256 + 1) * 256;

        int* tmp_keys = reinterpret_cast<int*>(buffer);
        buffer += ((sizeof(int) * m - 1) / 256 + 1) * 256;

        int* tmp_groups = reinterpret_cast<int*>(buffer);
        buffer += ((sizeof(int) * m - 1) / 256 + 1) * 256;

        // Determine number of rows per group
#define CSRGEMM_DIM 256
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_group_reduce_part2<CSRGEMM_DIM, CSRGEMM_MAXGROUPS, exceeding_smem>),
            dim3(CSRGEMM_DIM),
            dim3(CSRGEMM_DIM),
            0,
            stream,
            m,
            csr_row_ptr_C,
            d_group_size,
            tmp_groups);

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

        // Create identity permutation for group access
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::create_identity_permutation_template(handle, m, tmp_perm));

        rocsparse::primitives::double_buffer<int> d_keys(tmp_groups, tmp_keys);
        rocsparse::primitives::double_buffer<J>   d_vals(tmp_perm, tmp_vals);

        // Sort pairs (by groups)
        rocprim_buffer = reinterpret_cast<void*>(buffer);
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::radix_sort_pairs_buffer_size<int, J>(
            handle, m, 0, 3, &rocprim_size)));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::radix_sort_pairs(
            handle, d_keys, d_vals, m, 0, 3, rocprim_size, rocprim_buffer));

        d_perm = d_vals.current();

        // Release tmp_groups buffer
        // buffer -= ((sizeof(int) * m - 1) / 256 + 1) * 256;

        // Release tmp_keys buffer
        // buffer -= ((sizeof(int) * m - 1) / 256 + 1) * 256;
    }
    else
    {
        // First group processes all rows
        h_group_size[0] = m;
        RETURN_IF_HIP_ERROR(hipMemsetAsync(d_group_offset, 0, sizeof(J), stream));
    }

    // Compute columns and accumulate values for each group
    // Group 0: 0 - 16 non-zeros per row
    if(h_group_size[0] > 0)
    {
#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 8
#define CSRGEMM_HASHSIZE 16
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_fill_wf_per_row<CSRGEMM_DIM,
                                                CSRGEMM_SUB,
                                                CSRGEMM_HASHSIZE,
                                                CSRGEMM_FLL_HASH>),
            dim3((h_group_size[0] - 1) / (CSRGEMM_DIM / CSRGEMM_SUB) + 1),
            dim3(CSRGEMM_DIM),
            0,
            stream,
            h_group_size[0],
            rocsparse::max(k, n),
            &d_group_offset[0],
            d_perm,
            alpha_device_host,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_val_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_val_B,
            beta_device_host,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_val_D,
            csr_row_ptr_C,
            csr_col_ind_C,
            csr_val_C,
            base_A,
            base_B,
            descr_C->base,
            base_D,
            info_C->csrgemm_info->mul,
            info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 1: 17 - 32 non-zeros per row
    if(h_group_size[1] > 0)
    {
#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 16
#define CSRGEMM_HASHSIZE 32
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_fill_wf_per_row<CSRGEMM_DIM,
                                                CSRGEMM_SUB,
                                                CSRGEMM_HASHSIZE,
                                                CSRGEMM_FLL_HASH>),
            dim3((h_group_size[1] - 1) / (CSRGEMM_DIM / CSRGEMM_SUB) + 1),
            dim3(CSRGEMM_DIM),
            0,
            stream,
            h_group_size[1],
            rocsparse::max(k, n),
            &d_group_offset[1],
            d_perm,
            alpha_device_host,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_val_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_val_B,
            beta_device_host,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_val_D,
            csr_row_ptr_C,
            csr_col_ind_C,
            csr_val_C,
            base_A,
            base_B,
            descr_C->base,
            base_D,
            info_C->csrgemm_info->mul,
            info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 2: 33 - 256 non-zeros per row
    if(h_group_size[2] > 0)
    {
#define CSRGEMM_DIM 128
#define CSRGEMM_SUB 16
#define CSRGEMM_HASHSIZE 256
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_fill_block_per_row<CSRGEMM_DIM,
                                                   CSRGEMM_SUB,
                                                   CSRGEMM_HASHSIZE,
                                                   CSRGEMM_FLL_HASH>),
            dim3(h_group_size[2]),
            dim3(CSRGEMM_DIM),
            0,
            stream,
            rocsparse::max(k, n),
            &d_group_offset[2],
            d_perm,
            alpha_device_host,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_val_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_val_B,
            beta_device_host,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_val_D,
            csr_row_ptr_C,
            csr_col_ind_C,
            csr_val_C,
            base_A,
            base_B,
            descr_C->base,
            base_D,
            info_C->csrgemm_info->mul,
            info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 3: 257 - 512 non-zeros per row
    if(h_group_size[3] > 0)
    {
#define CSRGEMM_DIM 256
#define CSRGEMM_SUB 32
#define CSRGEMM_HASHSIZE 512
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_fill_block_per_row<CSRGEMM_DIM,
                                                   CSRGEMM_SUB,
                                                   CSRGEMM_HASHSIZE,
                                                   CSRGEMM_FLL_HASH>),
            dim3(h_group_size[3]),
            dim3(CSRGEMM_DIM),
            0,
            stream,
            rocsparse::max(k, n),
            &d_group_offset[3],
            d_perm,
            alpha_device_host,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_val_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_val_B,
            beta_device_host,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_val_D,
            csr_row_ptr_C,
            csr_col_ind_C,
            csr_val_C,
            base_A,
            base_B,
            descr_C->base,
            base_D,
            info_C->csrgemm_info->mul,
            info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 4: 513 - 1024 non-zeros per row
    if(h_group_size[4] > 0)
    {
#define CSRGEMM_DIM 512
#define CSRGEMM_SUB 32
#define CSRGEMM_HASHSIZE 1024
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_fill_block_per_row<CSRGEMM_DIM,
                                                   CSRGEMM_SUB,
                                                   CSRGEMM_HASHSIZE,
                                                   CSRGEMM_FLL_HASH>),
            dim3(h_group_size[4]),
            dim3(CSRGEMM_DIM),
            0,
            stream,
            rocsparse::max(k, n),
            &d_group_offset[4],
            d_perm,
            alpha_device_host,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_val_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_val_B,
            beta_device_host,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_val_D,
            csr_row_ptr_C,
            csr_col_ind_C,
            csr_val_C,
            base_A,
            base_B,
            descr_C->base,
            base_D,
            info_C->csrgemm_info->mul,
            info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 5: 1025 - 2048 non-zeros per row
    if(h_group_size[5] > 0)
    {
#define CSRGEMM_DIM 1024
#define CSRGEMM_SUB 32
#define CSRGEMM_HASHSIZE 2048
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrgemm_fill_block_per_row<CSRGEMM_DIM,
                                                   CSRGEMM_SUB,
                                                   CSRGEMM_HASHSIZE,
                                                   CSRGEMM_FLL_HASH>),
            dim3(h_group_size[5]),
            dim3(CSRGEMM_DIM),
            0,
            stream,
            rocsparse::max(k, n),
            &d_group_offset[5],
            d_perm,
            alpha_device_host,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_val_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_val_B,
            beta_device_host,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_val_D,
            csr_row_ptr_C,
            csr_col_ind_C,
            csr_val_C,
            base_A,
            base_B,
            descr_C->base,
            base_D,
            info_C->csrgemm_info->mul,
            info_C->csrgemm_info->add);
#undef CSRGEMM_HASHSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM
    }

    // Group 6: 2049 - 4096 non-zeros per row
    if(h_group_size[6] > 0 && !exceeding_smem)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_launcher(handle,
                                                              h_group_size[6],
                                                              &d_group_offset[6],
                                                              d_perm,
                                                              m,
                                                              n,
                                                              k,
                                                              alpha_device_host,
                                                              csr_row_ptr_A,
                                                              csr_col_ind_A,
                                                              csr_val_A,
                                                              csr_row_ptr_B,
                                                              csr_col_ind_B,
                                                              csr_val_B,
                                                              beta_device_host,
                                                              csr_row_ptr_D,
                                                              csr_col_ind_D,
                                                              csr_val_D,
                                                              csr_row_ptr_C,
                                                              csr_col_ind_C,
                                                              csr_val_C,
                                                              base_A,
                                                              base_B,
                                                              descr_C->base,
                                                              base_D,
                                                              info_C->csrgemm_info->mul,
                                                              info_C->csrgemm_info->add));
    }

    // Group 7: more than 4096 non-zeros per row
    if(h_group_size[7] > 0)
    {
        // Matrices B and D must be sorted in order to run this path
        if(descr_B->storage_mode == rocsparse_storage_mode_unsorted
           || (info_C->csrgemm_info->add ? descr_D->storage_mode == rocsparse_storage_mode_unsorted
                                         : false))
        {
            return rocsparse_status_requires_sorted_storage;
        }

        I* workspace_B = nullptr;

        if(info_C->csrgemm_info->mul == true)
        {
            // Allocate additional buffer for C = alpha * A * B
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMallocAsync((void**)&workspace_B, sizeof(I) * nnz_A, handle->stream));
        }

#define CSRGEMM_DIM 512
#define CSRGEMM_SUB 16
#define CSRGEMM_CHUNKSIZE 2048
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::
                 csrgemm_fill_block_per_row_multipass<CSRGEMM_DIM, CSRGEMM_SUB, CSRGEMM_CHUNKSIZE>),
            dim3(h_group_size[7]),
            dim3(CSRGEMM_DIM),
            0,
            stream,
            n,
            &d_group_offset[7],
            d_perm,
            alpha_device_host,
            csr_row_ptr_A,
            csr_col_ind_A,
            csr_val_A,
            csr_row_ptr_B,
            csr_col_ind_B,
            csr_val_B,
            beta_device_host,
            csr_row_ptr_D,
            csr_col_ind_D,
            csr_val_D,
            csr_row_ptr_C,
            csr_col_ind_C,
            csr_val_C,
            workspace_B,
            base_A,
            base_B,
            descr_C->base,
            base_D,
            info_C->csrgemm_info->mul,
            info_C->csrgemm_info->add);
#undef CSRGEMM_CHUNKSIZE
#undef CSRGEMM_SUB
#undef CSRGEMM_DIM

        if(info_C->csrgemm_info->mul == true)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(workspace_B, handle->stream));
        }
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(I, J, T, U)                                                                     \
    template rocsparse_status rocsparse::csrgemm_calc_template(rocsparse_handle    handle,          \
                                                               rocsparse_operation trans_A,         \
                                                               rocsparse_operation trans_B,         \
                                                               J                   m,               \
                                                               J                   n,               \
                                                               J                   k,               \
                                                               U alpha_device_host,                 \
                                                               const rocsparse_mat_descr descr_A,   \
                                                               I                         nnz_A,     \
                                                               const T*                  csr_val_A, \
                                                               const I* csr_row_ptr_A,              \
                                                               const J* csr_col_ind_A,              \
                                                               const rocsparse_mat_descr descr_B,   \
                                                               I                         nnz_B,     \
                                                               const T*                  csr_val_B, \
                                                               const I* csr_row_ptr_B,              \
                                                               const J* csr_col_ind_B,              \
                                                               U        beta_device_host,           \
                                                               const rocsparse_mat_descr descr_D,   \
                                                               I                         nnz_D,     \
                                                               const T*                  csr_val_D, \
                                                               const I* csr_row_ptr_D,              \
                                                               const J* csr_col_ind_D,              \
                                                               const rocsparse_mat_descr descr_C,   \
                                                               T*                        csr_val_C, \
                                                               const I* csr_row_ptr_C,              \
                                                               J*       csr_col_ind_C,              \
                                                               const rocsparse_mat_info info_C,     \
                                                               void*                    temp_buffer)

INSTANTIATE(int32_t, int32_t, float, float);
INSTANTIATE(int32_t, int32_t, double, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex, rocsparse_double_complex);

INSTANTIATE(int64_t, int64_t, float, float);
INSTANTIATE(int64_t, int64_t, double, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex, rocsparse_double_complex);

INSTANTIATE(int64_t, int32_t, float, float);
INSTANTIATE(int64_t, int32_t, double, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex, rocsparse_double_complex);

INSTANTIATE(int32_t, int32_t, float, const float*);
INSTANTIATE(int32_t, int32_t, double, const double*);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex, const rocsparse_float_complex*);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex, const rocsparse_double_complex*);

INSTANTIATE(int64_t, int64_t, float, const float*);
INSTANTIATE(int64_t, int64_t, double, const double*);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex, const rocsparse_float_complex*);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex, const rocsparse_double_complex*);

INSTANTIATE(int64_t, int32_t, float, const float*);
INSTANTIATE(int64_t, int32_t, double, const double*);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex, const rocsparse_float_complex*);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex, const rocsparse_double_complex*);

#undef INSTANTIATE

/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_csrmv_alg.hpp"
#include "rocsparse_handle.hpp"

#include <vector>

namespace rocsparse
{
    // Replaces csrmv_alg_default with a specific algorithm. When csrmv_info holds
    // the algorithm analysis selected for the same operation, that one is used.
    rocsparse_status csrmv_alg_default2csrmv_alg(rocsparse::csrmv_alg&        alg,
                                                 rocsparse_operation          operation,
                                                 const _rocsparse_csrmv_info* csrmv_info);

    // Host row blocks of CSR-Adaptive, built by the first phase of the adaptive
    // analysis. The second phase uploads them to the csrmv info.
    template <typename I, typename J>
    struct csrmv_adaptive_partition
    {
        std::vector<I> row_ptr;
        std::vector<I> row_blocks;
        std::vector<J> wg_ids;
        int64_t        first_row{};
        int64_t        last_row{};
        uint32_t       gen_wg{};
    };

    template <typename I, typename J>
    rocsparse_status csrmv_adaptive_compute_partition(rocsparse_handle                handle,
                                                      J                               m,
                                                      I                               nnz,
                                                      const I*                        csr_row_ptr,
                                                      csrmv_adaptive_partition<I, J>& partition);

    // Replaces csrmv_alg_default with the algorithm to analyse. For a
    // non-transposed matrix this builds the adaptive partition, chooses nnz-split
    // when the partition is a poor fit and adaptive otherwise. Explicit
    // algorithms are left unchanged and the partition is left empty.
    template <typename I, typename J>
    rocsparse_status csrmv_select_algorithm(rocsparse_handle                handle,
                                            rocsparse_operation             trans,
                                            J                               m,
                                            I                               nnz,
                                            const I*                        csr_row_ptr,
                                            rocsparse::csrmv_alg&           alg,
                                            csrmv_adaptive_partition<I, J>& partition);

    // Helper function to extract gamma arrays and z vectors for batched operations
    // gamma_device_array and z_array must be preallocated device buffers
    template <typename T, typename Y>
    rocsparse_status csrmv_extract_gamma_and_z_arrays(rocsparse_handle             handle,
                                                      rocsparse_int                num_extra,
                                                      rocsparse_const_dnvec_descr  gamma_vec,
                                                      rocsparse_const_dnvec_descr* z_vecs,
                                                      T*        gamma_device_array,
                                                      const Y** z_array);

    // partition is the result of csrmv_adaptive_compute_partition for this
    // matrix, or nullptr to compute it here.
    template <typename I, typename J, typename A>
    rocsparse_status
        csrmv_analysis_adaptive_template_dispatch(rocsparse_handle          handle,
                                                  rocsparse_operation       trans,
                                                  J                         m,
                                                  J                         n,
                                                  I                         nnz,
                                                  const rocsparse_mat_descr descr,
                                                  const A*                  csr_val,
                                                  const I*                  csr_row_ptr,
                                                  const J*                  csr_col_ind,
                                                  rocsparse_csrmv_info*     p_csrmv_info,
                                                  const csrmv_adaptive_partition<I, J>* partition);

    template <typename I, typename J, typename A>
    rocsparse_status csrmv_analysis_lrb_template_dispatch(rocsparse_handle          handle,
                                                          rocsparse_operation       trans,
                                                          J                         m,
                                                          J                         n,
                                                          I                         nnz,
                                                          const rocsparse_mat_descr descr,
                                                          const A*                  csr_val,
                                                          const I*                  csr_row_ptr,
                                                          const J*                  csr_col_ind,
                                                          rocsparse_csrmv_info*     p_csrmv_info);

    // known_max_row_nnz is the longest row of the matrix when the caller already
    // has it, or nullptr to compute it here.
    template <typename I, typename J, typename A>
    rocsparse_status csrmv_analysis_nnzsplit_template_dispatch(rocsparse_handle          handle,
                                                               rocsparse_operation       trans,
                                                               J                         m,
                                                               J                         n,
                                                               I                         nnz,
                                                               const rocsparse_mat_descr descr,
                                                               const A*                  csr_val,
                                                               const I*              csr_row_ptr,
                                                               const J*              csr_col_ind,
                                                               rocsparse_csrmv_info* p_csrmv_info,
                                                               const int64_t* known_max_row_nnz
                                                               = nullptr);

    template <typename I, typename J, typename A>
    rocsparse_status csrmv_analysis_template(rocsparse_handle          handle,
                                             rocsparse_operation       trans,
                                             rocsparse::csrmv_alg      alg,
                                             int64_t                   m,
                                             int64_t                   n,
                                             int64_t                   nnz,
                                             const rocsparse_mat_descr descr,
                                             const void*               csr_val,
                                             const void*               csr_row_ptr,
                                             const void*               csr_col_ind,
                                             rocsparse_csrmv_info*     p_csrmv_info);

    template <typename T, typename I, typename J, typename A, typename X, typename Y>
    rocsparse_status csrmv_rowsplit_template_dispatch(rocsparse_handle          handle,
                                                      rocsparse_operation       trans,
                                                      J                         m,
                                                      J                         n,
                                                      I                         nnz,
                                                      const T*                  alpha_device_host,
                                                      const rocsparse_mat_descr descr,
                                                      const A*                  csr_val,
                                                      const I*                  csr_row_ptr_begin,
                                                      const I*                  csr_row_ptr_end,
                                                      const J*                  csr_col_ind,
                                                      const X*                  x,
                                                      const T*                  beta_device_host,
                                                      Y*                        y,
                                                      bool                      force_conj);

    template <typename T, typename I, typename J, typename A, typename X, typename Y>
    rocsparse_status csrmv_rowsplit_template_dispatch(rocsparse_handle            handle,
                                                      rocsparse_operation         trans,
                                                      J                           m,
                                                      J                           n,
                                                      I                           nnz,
                                                      const T*                    alpha_device_host,
                                                      const rocsparse_mat_descr   descr,
                                                      const A*                    csr_val,
                                                      const I*                    csr_row_ptr_begin,
                                                      const I*                    csr_row_ptr_end,
                                                      const J*                    csr_col_ind,
                                                      const X*                    x,
                                                      const T*                    beta_device_host,
                                                      Y*                          y,
                                                      rocsparse_int               num_extra,
                                                      rocsparse_const_dnvec_descr gamma_vec,
                                                      rocsparse_const_dnvec_descr* z_vecs,
                                                      bool                         force_conj);

    template <typename T, typename I, typename J, typename A, typename X, typename Y>
    rocsparse_status csrmv_adaptive_template_dispatch(rocsparse_handle          handle,
                                                      rocsparse_operation       trans,
                                                      J                         m,
                                                      J                         n,
                                                      I                         nnz,
                                                      const T*                  alpha_device_host,
                                                      const rocsparse_mat_descr descr,
                                                      const A*                  csr_val,
                                                      const I*                  csr_row_ptr,
                                                      const J*                  csr_col_ind,
                                                      rocsparse_csrmv_info      csrmv_info,
                                                      const X*                  x,
                                                      const T*                  beta_device_host,
                                                      Y*                        y,
                                                      bool                      force_conj);

    template <typename T, typename I, typename J, typename A, typename X, typename Y>
    rocsparse_status csrmv_adaptive_template_dispatch(rocsparse_handle            handle,
                                                      rocsparse_operation         trans,
                                                      J                           m,
                                                      J                           n,
                                                      I                           nnz,
                                                      const T*                    alpha_device_host,
                                                      const rocsparse_mat_descr   descr,
                                                      const A*                    csr_val,
                                                      const I*                    csr_row_ptr,
                                                      const J*                    csr_col_ind,
                                                      rocsparse_csrmv_info        csrmv_info,
                                                      const X*                    x,
                                                      const T*                    beta_device_host,
                                                      Y*                          y,
                                                      rocsparse_int               num_extra,
                                                      rocsparse_const_dnvec_descr gamma_vec,
                                                      rocsparse_const_dnvec_descr* z_vecs,
                                                      bool                         force_conj);

    template <typename T, typename I, typename J, typename A, typename X, typename Y>
    rocsparse_status csrmv_lrb_template_dispatch(rocsparse_handle             handle,
                                                 rocsparse_operation          trans,
                                                 J                            m,
                                                 J                            n,
                                                 I                            nnz,
                                                 const T*                     alpha_device_host,
                                                 const rocsparse_mat_descr    descr,
                                                 const A*                     csr_val,
                                                 const I*                     csr_row_ptr,
                                                 const J*                     csr_col_ind,
                                                 rocsparse_csrmv_info         info,
                                                 const X*                     x,
                                                 const T*                     beta_device_host,
                                                 Y*                           y,
                                                 rocsparse_int                num_extra,
                                                 rocsparse_const_dnvec_descr  gamma_vec,
                                                 rocsparse_const_dnvec_descr* z_vecs,
                                                 bool                         force_conj);

    template <typename T, typename I, typename J, typename A, typename X, typename Y>
    rocsparse_status csrmv_nnzsplit_template_dispatch(rocsparse_handle            handle,
                                                      rocsparse_operation         trans,
                                                      J                           m,
                                                      J                           n,
                                                      I                           nnz,
                                                      const T*                    alpha_device_host,
                                                      const rocsparse_mat_descr   descr,
                                                      const A*                    csr_val,
                                                      const I*                    csr_row_ptr,
                                                      const J*                    csr_col_ind,
                                                      rocsparse_csrmv_info        csrmv_info,
                                                      const X*                    x,
                                                      const T*                    beta_device_host,
                                                      Y*                          y,
                                                      rocsparse_int               num_extra,
                                                      rocsparse_const_dnvec_descr gamma_vec,
                                                      rocsparse_const_dnvec_descr* z_vecs,
                                                      bool                         force_conj);

    template <typename T, typename I, typename J, typename A, typename X, typename Y>
    rocsparse_status csrmv_nnzsplit_template_dispatch(rocsparse_handle          handle,
                                                      rocsparse_operation       trans,
                                                      J                         m,
                                                      J                         n,
                                                      I                         nnz,
                                                      const T*                  alpha_device_host,
                                                      const rocsparse_mat_descr descr,
                                                      const A*                  csr_val,
                                                      const I*                  csr_row_ptr,
                                                      const J*                  csr_col_ind,
                                                      rocsparse_csrmv_info      csrmv_info,
                                                      const X*                  x,
                                                      const T*                  beta_device_host,
                                                      Y*                        y,
                                                      bool                      force_conj);

    template <typename T, typename I, typename J, typename A, typename X, typename Y>
    rocsparse_status csrmv_template(rocsparse_handle          handle,
                                    rocsparse_operation       trans,
                                    rocsparse::csrmv_alg      alg,
                                    int64_t                   m,
                                    int64_t                   n,
                                    int64_t                   nnz,
                                    const void*               alpha,
                                    const rocsparse_mat_descr descr,
                                    const void*               csr_val,
                                    const void*               csr_row_ptr_begin,
                                    const void*               csr_row_ptr_end,
                                    const void*               csr_col_ind,
                                    rocsparse_csrmv_info      csrmv_info,
                                    const void*               x,
                                    const void*               beta,
                                    void*                     y,
                                    bool                      force_conj,
                                    bool                      fallback_algorithm);

    template <typename T, typename I, typename J, typename A, typename X, typename Y>
    rocsparse_status csrmv_template(rocsparse_handle             handle,
                                    rocsparse_operation          trans,
                                    rocsparse::csrmv_alg         alg,
                                    int64_t                      m,
                                    int64_t                      n,
                                    int64_t                      nnz,
                                    const void*                  alpha,
                                    const rocsparse_mat_descr    descr,
                                    const void*                  csr_val,
                                    const void*                  csr_row_ptr_begin,
                                    const void*                  csr_row_ptr_end,
                                    const void*                  csr_col_ind,
                                    rocsparse_csrmv_info         csrmv_info,
                                    const void*                  x,
                                    const void*                  beta,
                                    void*                        y,
                                    rocsparse_int                num_extra,
                                    rocsparse_const_dnvec_descr  gamma_vec,
                                    rocsparse_const_dnvec_descr* z_vecs,
                                    bool                         force_conj,
                                    bool                         fallback_algorithm);

    rocsparse_status csrmv_analysis(rocsparse_handle          handle,
                                    rocsparse_operation       trans,
                                    rocsparse::csrmv_alg      alg,
                                    int64_t                   m,
                                    int64_t                   n,
                                    int64_t                   nnz,
                                    const rocsparse_mat_descr descr,
                                    rocsparse_datatype        csr_val_datatype,
                                    const void*               csr_val,
                                    rocsparse_indextype       csr_row_ptr_indextype,
                                    const void*               csr_row_ptr,
                                    rocsparse_indextype       csr_col_ind_indextype,
                                    const void*               csr_col_ind,
                                    rocsparse_csrmv_info*     p_csrmv_info);

    rocsparse_status csrmv(rocsparse_handle             handle,
                           rocsparse_operation          trans,
                           rocsparse::csrmv_alg         alg,
                           int64_t                      m,
                           int64_t                      n,
                           int64_t                      nnz,
                           rocsparse_datatype           alpha_device_host_datatype,
                           const void*                  alpha_device_host,
                           const rocsparse_mat_descr    descr,
                           rocsparse_datatype           csr_val_datatype,
                           const void*                  csr_val,
                           rocsparse_indextype          csr_row_ptr_indextype,
                           const void*                  csr_row_ptr,
                           rocsparse_indextype          csr_row_ptr_end_indextype,
                           const void*                  csr_row_ptr_end,
                           rocsparse_indextype          csr_col_ind_indextype,
                           const void*                  csr_col_ind,
                           rocsparse_csrmv_info         csrmv_info,
                           rocsparse_datatype           x_datatype,
                           const void*                  x,
                           rocsparse_datatype           beta_device_host_datatype,
                           const void*                  beta_device_host,
                           rocsparse_datatype           y_datatype,
                           void*                        y,
                           rocsparse_int                num_extra,
                           rocsparse_const_dnvec_descr  gamma_vec,
                           rocsparse_const_dnvec_descr* z_vecs,
                           bool                         fallback_algorithm);

}

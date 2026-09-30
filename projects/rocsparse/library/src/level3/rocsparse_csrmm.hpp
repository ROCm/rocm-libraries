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

#include "rocsparse_handle.hpp"
#include "rocsparse_line_nnz_profile.hpp"

typedef enum rocsparse_csrmm_alg_
{
    rocsparse_csrmm_alg_default = 0,
    rocsparse_csrmm_alg_row_split,
    rocsparse_csrmm_alg_nnz_split,
    rocsparse_csrmm_alg_merge_path
} rocsparse_csrmm_alg;

namespace rocsparse
{
    // A dense multiply is batched iff its output C carries more than one batch:
    // the inputs A and B may legitimately be broadcast (batch count 1) across a
    // batched C, so C's batch count is the single authoritative indicator.
    inline bool spmm_is_batched(int64_t batch_count_C)
    {
        return batch_count_C > 1;
    }

    template <typename T, typename I, typename J, typename A>
    rocsparse_status csrmm_buffer_size_template(rocsparse_handle          handle,
                                                rocsparse_operation       trans_A,
                                                rocsparse_csrmm_alg       alg,
                                                int64_t                   m,
                                                int64_t                   n,
                                                int64_t                   k,
                                                int64_t                   nnz,
                                                const rocsparse_mat_descr descr,
                                                const void*               csr_val,
                                                const void*               csr_row_ptr,
                                                const void*               csr_col_ind,
                                                size_t*                   buffer_size);

    template <typename I, typename J, typename A>
    rocsparse_status csrmm_analysis_template(rocsparse_handle          handle,
                                             rocsparse_operation       trans_A,
                                             rocsparse_csrmm_alg       alg,
                                             int64_t                   m,
                                             int64_t                   n,
                                             int64_t                   k,
                                             int64_t                   nnz,
                                             int64_t                   batch_count,
                                             int64_t                   offsets_batch_stride_A,
                                             const rocsparse_mat_descr descr,
                                             const void*               csr_val,
                                             const void*               csr_row_ptr,
                                             const void*               csr_col_ind,
                                             void*                     temp_buffer);

    template <typename T, typename I, typename J, typename A, typename B, typename C>
    rocsparse_status csrmm_template_dispatch(rocsparse_handle    handle,
                                             rocsparse_operation trans_A,
                                             rocsparse_operation trans_B,
                                             rocsparse_csrmm_alg alg,
                                             J                   m,
                                             J                   n,
                                             J                   k,
                                             I                   nnz,
                                             J                   batch_count_A,
                                             int64_t             offsets_batch_stride_A,
                                             int64_t             columns_values_batch_stride_A,
                                             const T*            alpha_device_host,
                                             const rocsparse_mat_descr descr,
                                             const A*                  csr_val,
                                             const I*                  csr_row_ptr,
                                             const J*                  csr_col_ind,
                                             const B*                  dense_B,
                                             int64_t                   ldb,
                                             J                         batch_count_B,
                                             int64_t                   batch_stride_B,
                                             rocsparse_order           order_B,
                                             const T*                  beta_device_host,
                                             C*                        dense_C,
                                             int64_t                   ldc,
                                             J                         batch_count_C,
                                             int64_t                   batch_stride_C,
                                             rocsparse_order           order_C,
                                             void*                     temp_buffer,
                                             bool                      force_conj_A);

    template <typename T, typename I, typename J, typename A, typename B, typename C>
    rocsparse_status csrmm_template(rocsparse_handle          handle,
                                    rocsparse_operation       trans_A,
                                    rocsparse_operation       trans_B,
                                    rocsparse_csrmm_alg       alg,
                                    int64_t                   m,
                                    int64_t                   n,
                                    int64_t                   k,
                                    int64_t                   nnz,
                                    int64_t                   batch_count_A,
                                    int64_t                   offsets_batch_stride_A,
                                    int64_t                   columns_values_batch_stride_A,
                                    const void*               alpha,
                                    const rocsparse_mat_descr descr,
                                    const void*               csr_val,
                                    const void*               csr_row_ptr,
                                    const void*               csr_col_ind,
                                    const void*               dense_B,
                                    int64_t                   ldb,
                                    int64_t                   batch_count_B,
                                    int64_t                   batch_stride_B,
                                    rocsparse_order           order_B,
                                    const void*               beta,
                                    void*                     dense_C,
                                    int64_t                   ldc,
                                    int64_t                   batch_count_C,
                                    int64_t                   batch_stride_C,
                                    rocsparse_order           order_C,
                                    void*                     temp_buffer,
                                    bool                      force_conj_A);

    rocsparse_status csrmm_buffer_size(rocsparse_handle          handle,
                                       rocsparse_operation       trans_A,
                                       rocsparse_csrmm_alg       alg,
                                       int64_t                   m,
                                       int64_t                   n,
                                       int64_t                   k,
                                       int64_t                   nnz,
                                       int64_t                   batch_count_C,
                                       const rocsparse_mat_descr descr,
                                       rocsparse_datatype        compute_datatype,
                                       rocsparse_datatype        csr_val_datatype,
                                       const void*               csr_val,
                                       rocsparse_indextype       csr_row_ptr_indextype,
                                       const void*               csr_row_ptr,
                                       rocsparse_indextype       csr_col_ind_indextype,
                                       const void*               csr_col_ind,
                                       size_t*                   buffer_size);

    // When \p alg is the format default, csrmm_analysis resolves it to a concrete
    // load-balanced algorithm: it computes the matrix's line-nnz profile from
    // \p csr_row_ptr into \p alg_info->profile (a non-op if already known) and
    // applies csrmm_select_default_alg with \p alg_info->is_batched. Passing a
    // null \p alg_info (or a null profile inside it, or an explicit \p alg) skips
    // the auto-selection and runs the analysis for the given algorithm unchanged.
    // The reduction inside compute_line_nnz_profile synchronizes, so this must
    // only be called on the non-capturing analysis stage.
    rocsparse_status csrmm_analysis(rocsparse_handle                        handle,
                                    rocsparse_operation                     trans_A,
                                    rocsparse_csrmm_alg                     alg,
                                    int64_t                                 m,
                                    int64_t                                 n,
                                    int64_t                                 k,
                                    int64_t                                 nnz,
                                    int64_t                                 batch_count,
                                    int64_t                                 offsets_batch_stride_A,
                                    const rocsparse_mat_descr               descr,
                                    rocsparse_datatype                      csr_val_datatype,
                                    const void*                             csr_val,
                                    rocsparse_indextype                     csr_row_ptr_indextype,
                                    const void*                             csr_row_ptr,
                                    rocsparse_indextype                     csr_col_ind_indextype,
                                    const void*                             csr_col_ind,
                                    const rocsparse::spmm_default_alg_info* alg_info,
                                    void*                                   temp_buffer);

    // Auto-selection of a load-balanced CSR SpMM algorithm for the format
    // default (rocsparse_csrmm_alg_default). The expensive structural input -
    // the matrix's line-length profile - is computed once by the shared
    // rocsparse::compute_line_nnz_profile (declared in rocsparse_line_nnz_profile.hpp)
    // and cached on the descriptor; this selector is only the policy on top of it.
    //
    // csrmm_select_default_alg is a pure, O(1), handle-free mapping: it upgrades
    // the default to the non-zero-split kernel when one line is long enough,
    // relative to the device's parallelism, that the row-split kernel would
    // serialize on it (e.g. a power-law "hub" row). The crossover is the
    // dimensionless, architecture-portable test
    //
    //     profile.max * cu_count >= C * profile.nnz
    //
    // where cu_count is the device compute-unit count (queried from the handle)
    // and C is a single dimensionless constant; this replaces the previous pair
    // of architecture-specific absolute thresholds and self-scales with the GPU.
    // With nothing cached, or for an explicit (non-default) algorithm, the
    // algorithm is returned unchanged. The profile is computed once on the
    // preprocess (analysis) stage; the selector is then re-applied identically on
    // the compute stage (from the cached profile) so the chosen algorithm stays
    // consistent while keeping the capture-sensitive compute stage free of kernel
    // launches and copies. Because buffer_size runs before the profile exists, it
    // sizes for the largest auto-selectable kernel (nnz-split) as a safe upper
    // bound.
    rocsparse_status csrmm_select_default_alg(rocsparse_operation                trans_A,
                                              bool                               is_batched,
                                              int32_t                            cu_count,
                                              const rocsparse::line_nnz_profile& profile,
                                              rocsparse_csrmm_alg&               alg);

    // When \p alg is the format default and \p alg_info carries a non-null, known
    // profile, csrmm resolves it to the same load-balanced algorithm chosen at
    // analysis by re-applying csrmm_select_default_alg from the cached profile and
    // alg_info->is_batched. This is a pure, launch-free O(1) step, so the
    // capture-sensitive compute stage stays free of kernels and synchronizations.
    // A null \p alg_info or null/unknown profile keeps the row-split default.
    rocsparse_status csrmm(rocsparse_handle                        handle,
                           rocsparse_operation                     trans_A,
                           rocsparse_operation                     trans_B,
                           rocsparse_csrmm_alg                     alg,
                           int64_t                                 m,
                           int64_t                                 n,
                           int64_t                                 k,
                           int64_t                                 nnz,
                           int64_t                                 batch_count_A,
                           int64_t                                 offsets_batch_stride_A,
                           int64_t                                 columns_values_batch_stride_A,
                           rocsparse_datatype                      alpha_datatype,
                           const void*                             alpha,
                           const rocsparse_mat_descr               descr,
                           rocsparse_datatype                      csr_val_datatype,
                           const void*                             csr_val,
                           rocsparse_indextype                     csr_row_ptr_indextype,
                           const void*                             csr_row_ptr,
                           rocsparse_indextype                     csr_col_ind_indextype,
                           const void*                             csr_col_ind,
                           rocsparse_datatype                      dense_B_datatype,
                           const void*                             dense_B,
                           int64_t                                 ldb,
                           int64_t                                 batch_count_B,
                           int64_t                                 batch_stride_B,
                           rocsparse_order                         order_B,
                           rocsparse_datatype                      beta_datatype,
                           const void*                             beta,
                           rocsparse_datatype                      dense_C_datatype,
                           void*                                   dense_C,
                           int64_t                                 ldc,
                           int64_t                                 batch_count_C,
                           int64_t                                 batch_stride_C,
                           rocsparse_order                         order_C,
                           const rocsparse::spmm_default_alg_info* alg_info,
                           void*                                   temp_buffer,
                           bool                                    force_conj_A);
}

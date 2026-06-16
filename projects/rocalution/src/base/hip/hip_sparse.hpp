/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCALUTION_HIP_HIP_SPARSE_HPP_
#define ROCALUTION_HIP_HIP_SPARSE_HPP_

#include "../../utils/type_traits.hpp"
#include <rocsparse/rocsparse.h>

namespace rocalution
{
    // ValueType to rocsparse_datatype
    template <typename ValueType>
    rocsparse_datatype rocsparseTdatatype();

    // rocsparse csrmv analysis
    template <typename ValueType>
    rocsparse_status rocsparseTcsrmv_analysis(rocsparse_handle          handle,
                                              rocsparse_operation       trans,
                                              int                       m,
                                              int                       n,
                                              int                       nnz,
                                              const rocsparse_mat_descr descr,
                                              const ValueType*          csr_val,
                                              const int*                csr_row_ptr,
                                              const int*                csr_col_ind,
                                              rocsparse_mat_info        info);

    // rocsparse csrmv
    template <typename ValueType>
    rocsparse_status rocsparseTcsrmv(rocsparse_handle          handle,
                                     rocsparse_operation       trans,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const ValueType*          alpha,
                                     const rocsparse_mat_descr descr,
                                     const ValueType*          csr_val,
                                     const int*                csr_row_ptr,
                                     const int*                csr_col_ind,
                                     rocsparse_mat_info        info,
                                     const ValueType*          x,
                                     const ValueType*          beta,
                                     ValueType*                y);

    // rocsparse csrsv buffer size
    template <typename ValueType>
    rocsparse_status rocsparseTcsrsv_buffer_size(rocsparse_handle          handle,
                                                 rocsparse_operation       trans,
                                                 int                       m,
                                                 int                       nnz,
                                                 const rocsparse_mat_descr descr,
                                                 const ValueType*          csr_val,
                                                 const int*                csr_row_ptr,
                                                 const int*                csr_col_ind,
                                                 rocsparse_mat_info        info,
                                                 size_t*                   buffer_size);

    // rocsparse csrsv analysis
    template <typename ValueType>
    rocsparse_status rocsparseTcsrsv_analysis(rocsparse_handle          handle,
                                              rocsparse_operation       trans,
                                              int                       m,
                                              int                       nnz,
                                              const rocsparse_mat_descr descr,
                                              const ValueType*          csr_val,
                                              const int*                csr_row_ptr,
                                              const int*                csr_col_ind,
                                              rocsparse_mat_info        info,
                                              rocsparse_analysis_policy analysis,
                                              rocsparse_solve_policy    solve,
                                              void*                     temp_buffer);

    // rocsparse csrsv
    template <typename ValueType>
    rocsparse_status rocsparseTcsrsv(rocsparse_handle          handle,
                                     rocsparse_operation       trans,
                                     int                       m,
                                     int                       nnz,
                                     const ValueType*          alpha,
                                     const rocsparse_mat_descr descr,
                                     const ValueType*          csr_val,
                                     const int*                csr_row_ptr,
                                     const int*                csr_col_ind,
                                     rocsparse_mat_info        info,
                                     const ValueType*          x,
                                     ValueType*                y,
                                     rocsparse_solve_policy    policy,
                                     void*                     temp_buffer);

    // rocsprarse csritsv buffer size
    template <typename ValueType>
    rocsparse_status rocsparseTcsritsv_buffer_size(rocsparse_handle          handle,
                                                   rocsparse_operation       trans,
                                                   rocsparse_int             m,
                                                   rocsparse_int             nnz,
                                                   const rocsparse_mat_descr descr,
                                                   const ValueType*          csr_val,
                                                   const rocsparse_int*      csr_row_ptr,
                                                   const rocsparse_int*      csr_col_ind,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size);

    // rocsprarse csritsv analysis
    template <typename ValueType>
    rocsparse_status rocsparseTcsritsv_analysis(rocsparse_handle          handle,
                                                rocsparse_operation       trans,
                                                rocsparse_int             m,
                                                rocsparse_int             nnz,
                                                const rocsparse_mat_descr descr,
                                                const ValueType*          csr_val,
                                                const rocsparse_int*      csr_row_ptr,
                                                const rocsparse_int*      csr_col_ind,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                void*                     temp_buffer);

    // rocsprarse csritsv solve
    template <typename ValueType>
    rocsparse_status rocsparseTcsritsv_solve(rocsparse_handle                   handle,
                                             rocsparse_int*                     host_nmaxiter,
                                             const numeric_traits_t<ValueType>* host_tol,
                                             numeric_traits_t<ValueType>*       host_history,
                                             rocsparse_operation                trans,
                                             rocsparse_int                      m,
                                             rocsparse_int                      nnz,
                                             const ValueType*                   alpha,
                                             const rocsparse_mat_descr          descr,
                                             const ValueType*                   csr_val,
                                             const rocsparse_int*               csr_row_ptr,
                                             const rocsparse_int*               csr_col_ind,
                                             rocsparse_mat_info                 info,
                                             const ValueType*                   x,
                                             ValueType*                         y,
                                             rocsparse_solve_policy             policy,
                                             void*                              temp_buffer);

    // rocsparse bsrsv buffer size
    template <typename ValueType>
    rocsparse_status rocsparseTbsrsv_buffer_size(rocsparse_handle          handle,
                                                 rocsparse_direction       dir,
                                                 rocsparse_operation       trans,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const rocsparse_mat_descr descr,
                                                 const ValueType*          bsr_val,
                                                 const int*                bsr_row_ptr,
                                                 const int*                bsr_col_ind,
                                                 int                       bsr_dim,
                                                 rocsparse_mat_info        info,
                                                 size_t*                   buffer_size);

    // rocsparse bsrsv analysis
    template <typename ValueType>
    rocsparse_status rocsparseTbsrsv_analysis(rocsparse_handle          handle,
                                              rocsparse_direction       dir,
                                              rocsparse_operation       trans,
                                              int                       mb,
                                              int                       nnzb,
                                              const rocsparse_mat_descr descr,
                                              const ValueType*          bsr_val,
                                              const int*                bsr_row_ptr,
                                              const int*                bsr_col_ind,
                                              int                       bsr_dim,
                                              rocsparse_mat_info        info,
                                              rocsparse_analysis_policy analysis,
                                              rocsparse_solve_policy    solve,
                                              void*                     temp_buffer);

    // rocsparse bsrsv
    template <typename ValueType>
    rocsparse_status rocsparseTbsrsv(rocsparse_handle          handle,
                                     rocsparse_direction       dir,
                                     rocsparse_operation       trans,
                                     int                       mb,
                                     int                       nnzb,
                                     const ValueType*          alpha,
                                     const rocsparse_mat_descr descr,
                                     const ValueType*          bsr_val,
                                     const int*                bsr_row_ptr,
                                     const int*                bsr_col_ind,
                                     int                       bsr_dim,
                                     rocsparse_mat_info        info,
                                     const ValueType*          x,
                                     ValueType*                y,
                                     rocsparse_solve_policy    policy,
                                     void*                     temp_buffer);

    // rocsparse coomv
    template <typename ValueType>
    rocsparse_status rocsparseTcoomv(rocsparse_handle          handle,
                                     rocsparse_operation       trans,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const ValueType*          alpha,
                                     const rocsparse_mat_descr descr,
                                     const ValueType*          coo_val,
                                     const int*                coo_row_ind,
                                     const int*                coo_col_ind,
                                     const ValueType*          x,
                                     const ValueType*          beta,
                                     ValueType*                y);

    // rocsparse ellmv
    template <typename ValueType>
    rocsparse_status rocsparseTellmv(rocsparse_handle          handle,
                                     rocsparse_operation       trans,
                                     int                       m,
                                     int                       n,
                                     const ValueType*          alpha,
                                     const rocsparse_mat_descr descr,
                                     const ValueType*          ell_val,
                                     const int*                ell_col_ind,
                                     int                       ell_width,
                                     const ValueType*          x,
                                     const ValueType*          beta,
                                     ValueType*                y);

    // rocsparse bsrmv
    template <typename ValueType>
    rocsparse_status rocsparseTbsrmv(rocsparse_handle          handle,
                                     rocsparse_direction       dir,
                                     rocsparse_operation       trans,
                                     int                       mb,
                                     int                       nb,
                                     int                       nnzb,
                                     const ValueType*          alpha,
                                     const rocsparse_mat_descr descr,
                                     const ValueType*          bsr_val,
                                     const int*                bsr_row_ptr,
                                     const int*                bsr_col_ind,
                                     int                       bsr_dim,
                                     const ValueType*          x,
                                     const ValueType*          beta,
                                     ValueType*                y);

    // rocsparse csrgeam
    template <typename ValueType>
    rocsparse_status rocsparseTcsrgeam(rocsparse_handle          handle,
                                       int                       m,
                                       int                       n,
                                       const ValueType*          alpha,
                                       const rocsparse_mat_descr descr_A,
                                       int                       nnz_A,
                                       const ValueType*          csr_val_A,
                                       const int*                csr_row_ptr_A,
                                       const int*                csr_col_ind_A,
                                       const ValueType*          beta,
                                       const rocsparse_mat_descr descr_B,
                                       int                       nnz_B,
                                       const ValueType*          csr_val_B,
                                       const int*                csr_row_ptr_B,
                                       const int*                csr_col_ind_B,
                                       const rocsparse_mat_descr descr_C,
                                       ValueType*                csr_val_C,
                                       const int*                csr_row_ptr_C,
                                       int*                      csr_col_ind_C);

    // rocsparse csrgemm
    template <typename ValueType>
    rocsparse_status rocsparseTcsrgemm_buffer_size(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_B,
                                                   int                       m,
                                                   int                       n,
                                                   int                       k,
                                                   const ValueType*          alpha,
                                                   const rocsparse_mat_descr descr_A,
                                                   int                       nnz_A,
                                                   const int*                csr_row_ptr_A,
                                                   const int*                csr_col_ind_A,
                                                   const rocsparse_mat_descr descr_B,
                                                   int                       nnz_B,
                                                   const int*                csr_row_ptr_B,
                                                   const int*                csr_col_ind_B,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size);

    template <typename ValueType>
    rocsparse_status rocsparseTcsrgemm(rocsparse_handle          handle,
                                       rocsparse_operation       trans_A,
                                       rocsparse_operation       trans_B,
                                       int                       m,
                                       int                       n,
                                       int                       k,
                                       const ValueType*          alpha,
                                       const rocsparse_mat_descr descr_A,
                                       int                       nnz_A,
                                       const ValueType*          csr_val_A,
                                       const int*                csr_row_ptr_A,
                                       const int*                csr_col_ind_A,
                                       const rocsparse_mat_descr descr_B,
                                       int                       nnz_B,
                                       const ValueType*          csr_val_B,
                                       const int*                csr_row_ptr_B,
                                       const int*                csr_col_ind_B,
                                       const rocsparse_mat_descr descr_C,
                                       ValueType*                csr_val_C,
                                       const int*                csr_row_ptr_C,
                                       int*                      csr_col_ind_C,
                                       rocsparse_mat_info        info,
                                       void*                     temp_buffer);

    // rocsparse csric0 buffer size
    template <typename ValueType>
    rocsparse_status rocsparseTcsric0_buffer_size(rocsparse_handle          handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const rocsparse_mat_descr descr,
                                                  ValueType*                csr_val,
                                                  const int*                csr_row_ptr,
                                                  const int*                csr_col_ind,
                                                  rocsparse_mat_info        info,
                                                  size_t*                   buffer_size);

    // rocsparse csric0 analysis
    template <typename ValueType>
    rocsparse_status rocsparseTcsric0_analysis(rocsparse_handle          handle,
                                               int                       m,
                                               int                       nnz,
                                               const rocsparse_mat_descr descr,
                                               ValueType*                csr_val,
                                               const int*                csr_row_ptr,
                                               const int*                csr_col_ind,
                                               rocsparse_mat_info        info,
                                               rocsparse_analysis_policy analysis,
                                               rocsparse_solve_policy    solve,
                                               void*                     temp_buffer);

    // rocsparse csric0
    template <typename ValueType>
    rocsparse_status rocsparseTcsric0(rocsparse_handle          handle,
                                      int                       m,
                                      int                       nnz,
                                      const rocsparse_mat_descr descr,
                                      ValueType*                csr_val,
                                      const int*                csr_row_ptr,
                                      const int*                csr_col_ind,
                                      rocsparse_mat_info        info,
                                      rocsparse_solve_policy    policy,
                                      void*                     temp_buffer);

    // rocsparse bsric0 buffer size
    template <typename ValueType>
    rocsparse_status rocsparseTbsric0_buffer_size(rocsparse_handle          handle,
                                                  rocsparse_direction       dir,
                                                  int                       mb,
                                                  int                       nnzb,
                                                  const rocsparse_mat_descr descr,
                                                  ValueType*                bsr_val,
                                                  const int*                bsr_row_ptr,
                                                  const int*                bsr_col_ind,
                                                  int                       bsr_dim,
                                                  rocsparse_mat_info        info,
                                                  size_t*                   buffer_size);

    // rocsparse bsric0 analysis
    template <typename ValueType>
    rocsparse_status rocsparseTbsric0_analysis(rocsparse_handle          handle,
                                               rocsparse_direction       dir,
                                               int                       mb,
                                               int                       nnzb,
                                               const rocsparse_mat_descr descr,
                                               ValueType*                bsr_val,
                                               const int*                bsr_row_ptr,
                                               const int*                bsr_col_ind,
                                               int                       bsr_dim,
                                               rocsparse_mat_info        info,
                                               rocsparse_analysis_policy analysis,
                                               rocsparse_solve_policy    solve,
                                               void*                     temp_buffer);

    // rocsparse bsric0
    template <typename ValueType>
    rocsparse_status rocsparseTbsric0(rocsparse_handle          handle,
                                      rocsparse_direction       dir,
                                      int                       mb,
                                      int                       nnzb,
                                      const rocsparse_mat_descr descr,
                                      ValueType*                bsr_val,
                                      const int*                bsr_row_ptr,
                                      const int*                bsr_col_ind,
                                      int                       bsr_dim,
                                      rocsparse_mat_info        info,
                                      rocsparse_solve_policy    policy,
                                      void*                     temp_buffer);

    // rocsparse csrilu0 buffer size
    template <typename ValueType>
    rocsparse_status rocsparseTcsrilu0_buffer_size(rocsparse_handle          handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const rocsparse_mat_descr descr,
                                                   ValueType*                csr_val,
                                                   const int*                csr_row_ptr,
                                                   const int*                csr_col_ind,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size);

    // rocsparse csrilu0 analysis
    template <typename ValueType>
    rocsparse_status rocsparseTcsrilu0_analysis(rocsparse_handle          handle,
                                                int                       m,
                                                int                       nnz,
                                                const rocsparse_mat_descr descr,
                                                ValueType*                csr_val,
                                                const int*                csr_row_ptr,
                                                const int*                csr_col_ind,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                void*                     temp_buffer);

    // rocsparse csrilu0
    template <typename ValueType>
    rocsparse_status rocsparseTcsrilu0(rocsparse_handle          handle,
                                       int                       m,
                                       int                       nnz,
                                       const rocsparse_mat_descr descr,
                                       ValueType*                csr_val,
                                       const int*                csr_row_ptr,
                                       const int*                csr_col_ind,
                                       rocsparse_mat_info        info,
                                       rocsparse_solve_policy    policy,
                                       void*                     temp_buffer);

    // rocsparse bsrilu0 buffer size
    template <typename ValueType>
    rocsparse_status rocsparseTbsrilu0_buffer_size(rocsparse_handle          handle,
                                                   rocsparse_direction       dir,
                                                   int                       mb,
                                                   int                       nnzb,
                                                   const rocsparse_mat_descr descr,
                                                   ValueType*                bsr_val,
                                                   const int*                bsr_row_ptr,
                                                   const int*                bsr_col_ind,
                                                   int                       bsr_dim,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size);

    // rocsparse bsrilu0 analysis
    template <typename ValueType>
    rocsparse_status rocsparseTbsrilu0_analysis(rocsparse_handle          handle,
                                                rocsparse_direction       dir,
                                                int                       mb,
                                                int                       nnzb,
                                                const rocsparse_mat_descr descr,
                                                ValueType*                bsr_val,
                                                const int*                bsr_row_ptr,
                                                const int*                bsr_col_ind,
                                                int                       bsr_dim,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                void*                     temp_buffer);

    // rocsparse bsrilu0
    template <typename ValueType>
    rocsparse_status rocsparseTbsrilu0(rocsparse_handle          handle,
                                       rocsparse_direction       dir,
                                       int                       mb,
                                       int                       nnzb,
                                       const rocsparse_mat_descr descr,
                                       ValueType*                bsr_val,
                                       const int*                bsr_row_ptr,
                                       const int*                bsr_col_ind,
                                       int                       bsr_dim,
                                       rocsparse_mat_info        info,
                                       rocsparse_solve_policy    policy,
                                       void*                     temp_buffer);

    // rocsparse_csr2bsr
    template <typename ValueType>
    rocsparse_status rocsparseTcsr2bsr(rocsparse_handle          handle,
                                       rocsparse_direction       dir,
                                       int                       m,
                                       int                       n,
                                       const rocsparse_mat_descr csr_descr,
                                       const ValueType*          csr_val,
                                       const int*                csr_row_ptr,
                                       const int*                csr_col_ind,
                                       int                       block_dim,
                                       const rocsparse_mat_descr bsr_descr,
                                       ValueType*                bsr_val,
                                       int*                      bsr_row_ptr,
                                       int*                      bsr_col_ind);

    // rocsparse_bsr2csr
    template <typename ValueType>
    rocsparse_status rocsparseTbsr2csr(rocsparse_handle          handle,
                                       rocsparse_direction       dir,
                                       int                       mb,
                                       int                       nb,
                                       const rocsparse_mat_descr bsr_descr,
                                       const ValueType*          bsr_val,
                                       const int*                bsr_row_ptr,
                                       const int*                bsr_col_ind,
                                       int                       block_dim,
                                       const rocsparse_mat_descr csr_descr,
                                       ValueType*                csr_val,
                                       int*                      csr_row_ptr,
                                       int*                      csr_col_ind);

    // rocsparse csr2csc
    template <typename ValueType>
    rocsparse_status rocsparseTcsr2csc(rocsparse_handle     handle,
                                       int                  m,
                                       int                  n,
                                       int                  nnz,
                                       const ValueType*     csr_val,
                                       const int*           csr_row_ptr,
                                       const int*           csr_col_ind,
                                       ValueType*           csc_val,
                                       int*                 csc_row_ind,
                                       int*                 csc_col_ptr,
                                       rocsparse_action     copy_values,
                                       rocsparse_index_base idx_base,
                                       void*                temp_buffer);

    // rocsparse csr2ell
    template <typename ValueType>
    rocsparse_status rocsparseTcsr2ell(rocsparse_handle          handle,
                                       int                       m,
                                       const rocsparse_mat_descr csr_descr,
                                       const ValueType*          csr_val,
                                       const int*                csr_row_ptr,
                                       const int*                csr_col_ind,
                                       const rocsparse_mat_descr ell_descr,
                                       int                       ell_width,
                                       ValueType*                ell_val,
                                       int*                      ell_col_ind);

    // rocsparse ell2csr
    template <typename ValueType>
    rocsparse_status rocsparseTell2csr(rocsparse_handle          handle,
                                       int                       m,
                                       int                       n,
                                       const rocsparse_mat_descr ell_descr,
                                       int                       ell_width,
                                       const ValueType*          ell_val,
                                       const int*                ell_col_ind,
                                       const rocsparse_mat_descr csr_descr,
                                       ValueType*                csr_val,
                                       const int*                csr_row_ptr,
                                       int*                      csr_col_ind);

    // rocsparse csr2dense
    template <typename ValueType>
    rocsparse_status rocsparseTcsr2dense(rocsparse_handle          handle,
                                         int                       m,
                                         int                       n,
                                         const rocsparse_mat_descr csr_descr,
                                         const ValueType*          csr_val,
                                         const int*                csr_row_ptr,
                                         const int*                csr_col_ind,
                                         ValueType*                A,
                                         int                       ld);

    // rocsparse dense2csr
    template <typename ValueType>
    rocsparse_status rocsparseTdense2csr(rocsparse_handle          handle,
                                         int                       m,
                                         int                       n,
                                         const rocsparse_mat_descr descr_A,
                                         const ValueType*          A,
                                         int                       lda,
                                         const int*                nnz_per_row,
                                         ValueType*                csr_val,
                                         int*                      csr_row_ptr,
                                         int*                      csr_col_ind);

    // rocsparse nnz
    template <typename ValueType>
    rocsparse_status rocsparseTnnz(rocsparse_handle          handle,
                                   rocsparse_direction       dir_A,
                                   int                       m,
                                   int                       n,
                                   const rocsparse_mat_descr descr_A,
                                   const ValueType*          A,
                                   int                       lda,
                                   int*                      nnz_per_row_column,
                                   int*                      nnz_total);

    // rocsparse gthr
    template <typename ValueType>
    rocsparse_status rocsparseTgthr(rocsparse_handle     handle,
                                    int                  nnz,
                                    ValueType*           y,
                                    ValueType*           x_val,
                                    int*                 x_ind,
                                    rocsparse_index_base idx_base);
    // rocsparse nnz compress
    template <typename ValueType>
    rocsparse_status rocsparseTnnz_compress(rocsparse_handle          handle,
                                            rocsparse_int             m,
                                            const rocsparse_mat_descr descr_A,
                                            const ValueType*          csr_val_A,
                                            const rocsparse_int*      csr_row_ptr_A,
                                            rocsparse_int*            nnz_per_row,
                                            rocsparse_int*            nnz_C,
                                            ValueType                 tol);

    // rocsparse csr2csr compress
    template <typename ValueType>
    rocsparse_status rocsparseTcsr2csr_compress(rocsparse_handle          handle,
                                                rocsparse_int             m,
                                                rocsparse_int             n,
                                                const rocsparse_mat_descr descr_A,
                                                const ValueType*          csr_val_A,
                                                const rocsparse_int*      csr_row_ptr_A,
                                                const rocsparse_int*      csr_col_ind_A,
                                                rocsparse_int             nnz_A,
                                                const rocsparse_int*      nnz_per_row,
                                                ValueType*                csr_val_C,
                                                rocsparse_int*            csr_row_ptr_C,
                                                rocsparse_int*            csr_col_ind_C,
                                                ValueType                 tol);

    // rocsparse csritilu0 compute
    template <typename ValueType>
    rocsparse_status rocsparseTcsritilu0_compute(rocsparse_handle            handle,
                                                 rocsparse_itilu0_alg        alg,
                                                 rocsparse_int               option,
                                                 rocsparse_int*              nmaxiter,
                                                 numeric_traits_t<ValueType> tol,
                                                 rocsparse_int               m,
                                                 rocsparse_int               nnz,
                                                 const rocsparse_int*        csr_row_ptr,
                                                 const rocsparse_int*        csr_col_ind,
                                                 const ValueType*            csr_val,
                                                 ValueType*                  ilu0,
                                                 rocsparse_index_base        idx_base,
                                                 size_t                      buffer_size,
                                                 void*                       buffer);

    // rocsparse csritilu0 history
    template <typename ValueType>
    rocsparse_status rocsparseTcsritilu0_history(rocsparse_handle             handle,
                                                 rocsparse_itilu0_alg         alg,
                                                 rocsparse_int*               niter,
                                                 numeric_traits_t<ValueType>* data,
                                                 size_t                       buffer_size,
                                                 void*                        buffer);

} // namespace rocalution

#endif // ROCALUTION_HIP_HIP_SPARSE_HPP_

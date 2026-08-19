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

#include "hip_sparse.hpp"
#include "../../utils/def.hpp"
#include "../../utils/log.hpp"

#include <complex>
#include <rocsparse/rocsparse.h>

namespace rocalution
{
    // ValueType to rocsparse_datatype
    template <>
    rocsparse_datatype rocsparseTdatatype<float>()
    {
        return rocsparse_datatype_f32_r;
    }
    template <>
    rocsparse_datatype rocsparseTdatatype<double>()
    {
        return rocsparse_datatype_f64_r;
    }
    template <>
    rocsparse_datatype rocsparseTdatatype<std::complex<float>>()
    {
        return rocsparse_datatype_f32_c;
    }
    template <>
    rocsparse_datatype rocsparseTdatatype<std::complex<double>>()
    {
        return rocsparse_datatype_f64_c;
    }
    template <>
    rocsparse_datatype rocsparseTdatatype<int8_t>()
    {
        return rocsparse_datatype_i8_r;
    }
    template <>
    rocsparse_datatype rocsparseTdatatype<uint8_t>()
    {
        return rocsparse_datatype_u8_r;
    }
    template <>
    rocsparse_datatype rocsparseTdatatype<int32_t>()
    {
        return rocsparse_datatype_i32_r;
    }
    template <>
    rocsparse_datatype rocsparseTdatatype<uint32_t>()
    {
        return rocsparse_datatype_u32_r;
    }

    // rocsparse csrmv analysis
    template <>
    rocsparse_status rocsparseTcsrmv_analysis(rocsparse_handle          handle,
                                              rocsparse_operation       trans,
                                              int                       m,
                                              int                       n,
                                              int                       nnz,
                                              const rocsparse_mat_descr descr,
                                              const float*              csr_val,
                                              const int*                csr_row_ptr,
                                              const int*                csr_col_ind,
                                              rocsparse_mat_info        info)
    {
        return rocsparse_scsrmv_analysis(
            handle, trans, m, n, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info);
    }

    template <>
    rocsparse_status rocsparseTcsrmv_analysis(rocsparse_handle          handle,
                                              rocsparse_operation       trans,
                                              int                       m,
                                              int                       n,
                                              int                       nnz,
                                              const rocsparse_mat_descr descr,
                                              const double*             csr_val,
                                              const int*                csr_row_ptr,
                                              const int*                csr_col_ind,
                                              rocsparse_mat_info        info)
    {
        return rocsparse_dcsrmv_analysis(
            handle, trans, m, n, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info);
    }

    template <>
    rocsparse_status rocsparseTcsrmv_analysis(rocsparse_handle           handle,
                                              rocsparse_operation        trans,
                                              int                        m,
                                              int                        n,
                                              int                        nnz,
                                              const rocsparse_mat_descr  descr,
                                              const std::complex<float>* csr_val,
                                              const int*                 csr_row_ptr,
                                              const int*                 csr_col_ind,
                                              rocsparse_mat_info         info)
    {
        return rocsparse_ccsrmv_analysis(handle,
                                         trans,
                                         m,
                                         n,
                                         nnz,
                                         descr,
                                         (const rocsparse_float_complex*)csr_val,
                                         csr_row_ptr,
                                         csr_col_ind,
                                         info);
    }

    template <>
    rocsparse_status rocsparseTcsrmv_analysis(rocsparse_handle            handle,
                                              rocsparse_operation         trans,
                                              int                         m,
                                              int                         n,
                                              int                         nnz,
                                              const rocsparse_mat_descr   descr,
                                              const std::complex<double>* csr_val,
                                              const int*                  csr_row_ptr,
                                              const int*                  csr_col_ind,
                                              rocsparse_mat_info          info)
    {
        return rocsparse_zcsrmv_analysis(handle,
                                         trans,
                                         m,
                                         n,
                                         nnz,
                                         descr,
                                         (const rocsparse_double_complex*)csr_val,
                                         csr_row_ptr,
                                         csr_col_ind,
                                         info);
    }

    // rocsparse csrmv
    template <>
    rocsparse_status rocsparseTcsrmv(rocsparse_handle          handle,
                                     rocsparse_operation       trans,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const float*              alpha,
                                     const rocsparse_mat_descr descr,
                                     const float*              csr_val,
                                     const int*                csr_row_ptr,
                                     const int*                csr_col_ind,
                                     rocsparse_mat_info        info,
                                     const float*              x,
                                     const float*              beta,
                                     float*                    y)
    {
        return rocsparse_scsrmv(handle,
                                trans,
                                m,
                                n,
                                nnz,
                                alpha,
                                descr,
                                csr_val,
                                csr_row_ptr,
                                csr_col_ind,
                                info,
                                x,
                                beta,
                                y);
    }

    template <>
    rocsparse_status rocsparseTcsrmv(rocsparse_handle          handle,
                                     rocsparse_operation       trans,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const double*             alpha,
                                     const rocsparse_mat_descr descr,
                                     const double*             csr_val,
                                     const int*                csr_row_ptr,
                                     const int*                csr_col_ind,
                                     rocsparse_mat_info        info,
                                     const double*             x,
                                     const double*             beta,
                                     double*                   y)
    {
        return rocsparse_dcsrmv(handle,
                                trans,
                                m,
                                n,
                                nnz,
                                alpha,
                                descr,
                                csr_val,
                                csr_row_ptr,
                                csr_col_ind,
                                info,
                                x,
                                beta,
                                y);
    }

    template <>
    rocsparse_status rocsparseTcsrmv(rocsparse_handle           handle,
                                     rocsparse_operation        trans,
                                     int                        m,
                                     int                        n,
                                     int                        nnz,
                                     const std::complex<float>* alpha,
                                     const rocsparse_mat_descr  descr,
                                     const std::complex<float>* csr_val,
                                     const int*                 csr_row_ptr,
                                     const int*                 csr_col_ind,
                                     rocsparse_mat_info         info,
                                     const std::complex<float>* x,
                                     const std::complex<float>* beta,
                                     std::complex<float>*       y)
    {
        return rocsparse_ccsrmv(handle,
                                trans,
                                m,
                                n,
                                nnz,
                                (const rocsparse_float_complex*)alpha,
                                descr,
                                (const rocsparse_float_complex*)csr_val,
                                csr_row_ptr,
                                csr_col_ind,
                                info,
                                (const rocsparse_float_complex*)x,
                                (const rocsparse_float_complex*)beta,
                                (rocsparse_float_complex*)y);
    }

    template <>
    rocsparse_status rocsparseTcsrmv(rocsparse_handle            handle,
                                     rocsparse_operation         trans,
                                     int                         m,
                                     int                         n,
                                     int                         nnz,
                                     const std::complex<double>* alpha,
                                     const rocsparse_mat_descr   descr,
                                     const std::complex<double>* csr_val,
                                     const int*                  csr_row_ptr,
                                     const int*                  csr_col_ind,
                                     rocsparse_mat_info          info,
                                     const std::complex<double>* x,
                                     const std::complex<double>* beta,
                                     std::complex<double>*       y)
    {
        return rocsparse_zcsrmv(handle,
                                trans,
                                m,
                                n,
                                nnz,
                                (const rocsparse_double_complex*)alpha,
                                descr,
                                (const rocsparse_double_complex*)csr_val,
                                csr_row_ptr,
                                csr_col_ind,
                                info,
                                (const rocsparse_double_complex*)x,
                                (const rocsparse_double_complex*)beta,
                                (rocsparse_double_complex*)y);
    }

    // rocsparse csrsv buffer size
    template <>
    rocsparse_status rocsparseTcsrsv_buffer_size(rocsparse_handle          handle,
                                                 rocsparse_operation       trans,
                                                 int                       m,
                                                 int                       nnz,
                                                 const rocsparse_mat_descr descr,
                                                 const float*              csr_val,
                                                 const int*                csr_row_ptr,
                                                 const int*                csr_col_ind,
                                                 rocsparse_mat_info        info,
                                                 size_t*                   buffer_size)
    {
        return rocsparse_scsrsv_buffer_size(
            handle, trans, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsrsv_buffer_size(rocsparse_handle          handle,
                                                 rocsparse_operation       trans,
                                                 int                       m,
                                                 int                       nnz,
                                                 const rocsparse_mat_descr descr,
                                                 const double*             csr_val,
                                                 const int*                csr_row_ptr,
                                                 const int*                csr_col_ind,
                                                 rocsparse_mat_info        info,
                                                 size_t*                   buffer_size)
    {
        return rocsparse_dcsrsv_buffer_size(
            handle, trans, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsrsv_buffer_size(rocsparse_handle           handle,
                                                 rocsparse_operation        trans,
                                                 int                        m,
                                                 int                        nnz,
                                                 const rocsparse_mat_descr  descr,
                                                 const std::complex<float>* csr_val,
                                                 const int*                 csr_row_ptr,
                                                 const int*                 csr_col_ind,
                                                 rocsparse_mat_info         info,
                                                 size_t*                    buffer_size)
    {
        return rocsparse_ccsrsv_buffer_size(handle,
                                            trans,
                                            m,
                                            nnz,
                                            descr,
                                            (const rocsparse_float_complex*)csr_val,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            info,
                                            buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsrsv_buffer_size(rocsparse_handle            handle,
                                                 rocsparse_operation         trans,
                                                 int                         m,
                                                 int                         nnz,
                                                 const rocsparse_mat_descr   descr,
                                                 const std::complex<double>* csr_val,
                                                 const int*                  csr_row_ptr,
                                                 const int*                  csr_col_ind,
                                                 rocsparse_mat_info          info,
                                                 size_t*                     buffer_size)
    {
        return rocsparse_zcsrsv_buffer_size(handle,
                                            trans,
                                            m,
                                            nnz,
                                            descr,
                                            (const rocsparse_double_complex*)csr_val,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            info,
                                            buffer_size);
    }

    // rocsparse csrsv analysis
    template <>
    rocsparse_status rocsparseTcsrsv_analysis(rocsparse_handle          handle,
                                              rocsparse_operation       trans,
                                              int                       m,
                                              int                       nnz,
                                              const rocsparse_mat_descr descr,
                                              const float*              csr_val,
                                              const int*                csr_row_ptr,
                                              const int*                csr_col_ind,
                                              rocsparse_mat_info        info,
                                              rocsparse_analysis_policy analysis,
                                              rocsparse_solve_policy    solve,
                                              void*                     temp_buffer)
    {
        return rocsparse_scsrsv_analysis(handle,
                                         trans,
                                         m,
                                         nnz,
                                         descr,
                                         csr_val,
                                         csr_row_ptr,
                                         csr_col_ind,
                                         info,
                                         analysis,
                                         solve,
                                         temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrsv_analysis(rocsparse_handle          handle,
                                              rocsparse_operation       trans,
                                              int                       m,
                                              int                       nnz,
                                              const rocsparse_mat_descr descr,
                                              const double*             csr_val,
                                              const int*                csr_row_ptr,
                                              const int*                csr_col_ind,
                                              rocsparse_mat_info        info,
                                              rocsparse_analysis_policy analysis,
                                              rocsparse_solve_policy    solve,
                                              void*                     temp_buffer)
    {
        return rocsparse_dcsrsv_analysis(handle,
                                         trans,
                                         m,
                                         nnz,
                                         descr,
                                         csr_val,
                                         csr_row_ptr,
                                         csr_col_ind,
                                         info,
                                         analysis,
                                         solve,
                                         temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrsv_analysis(rocsparse_handle           handle,
                                              rocsparse_operation        trans,
                                              int                        m,
                                              int                        nnz,
                                              const rocsparse_mat_descr  descr,
                                              const std::complex<float>* csr_val,
                                              const int*                 csr_row_ptr,
                                              const int*                 csr_col_ind,
                                              rocsparse_mat_info         info,
                                              rocsparse_analysis_policy  analysis,
                                              rocsparse_solve_policy     solve,
                                              void*                      temp_buffer)
    {
        return rocsparse_ccsrsv_analysis(handle,
                                         trans,
                                         m,
                                         nnz,
                                         descr,
                                         (const rocsparse_float_complex*)csr_val,
                                         csr_row_ptr,
                                         csr_col_ind,
                                         info,
                                         analysis,
                                         solve,
                                         temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrsv_analysis(rocsparse_handle            handle,
                                              rocsparse_operation         trans,
                                              int                         m,
                                              int                         nnz,
                                              const rocsparse_mat_descr   descr,
                                              const std::complex<double>* csr_val,
                                              const int*                  csr_row_ptr,
                                              const int*                  csr_col_ind,
                                              rocsparse_mat_info          info,
                                              rocsparse_analysis_policy   analysis,
                                              rocsparse_solve_policy      solve,
                                              void*                       temp_buffer)
    {
        return rocsparse_zcsrsv_analysis(handle,
                                         trans,
                                         m,
                                         nnz,
                                         descr,
                                         (const rocsparse_double_complex*)csr_val,
                                         csr_row_ptr,
                                         csr_col_ind,
                                         info,
                                         analysis,
                                         solve,
                                         temp_buffer);
    }

    // rocsparse csrsv
    template <>
    rocsparse_status rocsparseTcsrsv(rocsparse_handle          handle,
                                     rocsparse_operation       trans,
                                     int                       m,
                                     int                       nnz,
                                     const float*              alpha,
                                     const rocsparse_mat_descr descr,
                                     const float*              csr_val,
                                     const int*                csr_row_ptr,
                                     const int*                csr_col_ind,
                                     rocsparse_mat_info        info,
                                     const float*              x,
                                     float*                    y,
                                     rocsparse_solve_policy    policy,
                                     void*                     temp_buffer)
    {
        return rocsparse_scsrsv_solve(handle,
                                      trans,
                                      m,
                                      nnz,
                                      alpha,
                                      descr,
                                      csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      info,
                                      x,
                                      y,
                                      policy,
                                      temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrsv(rocsparse_handle          handle,
                                     rocsparse_operation       trans,
                                     int                       m,
                                     int                       nnz,
                                     const double*             alpha,
                                     const rocsparse_mat_descr descr,
                                     const double*             csr_val,
                                     const int*                csr_row_ptr,
                                     const int*                csr_col_ind,
                                     rocsparse_mat_info        info,
                                     const double*             x,
                                     double*                   y,
                                     rocsparse_solve_policy    policy,
                                     void*                     temp_buffer)
    {
        return rocsparse_dcsrsv_solve(handle,
                                      trans,
                                      m,
                                      nnz,
                                      alpha,
                                      descr,
                                      csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      info,
                                      x,
                                      y,
                                      policy,
                                      temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrsv(rocsparse_handle           handle,
                                     rocsparse_operation        trans,
                                     int                        m,
                                     int                        nnz,
                                     const std::complex<float>* alpha,
                                     const rocsparse_mat_descr  descr,
                                     const std::complex<float>* csr_val,
                                     const int*                 csr_row_ptr,
                                     const int*                 csr_col_ind,
                                     rocsparse_mat_info         info,
                                     const std::complex<float>* x,
                                     std::complex<float>*       y,
                                     rocsparse_solve_policy     policy,
                                     void*                      temp_buffer)
    {
        return rocsparse_ccsrsv_solve(handle,
                                      trans,
                                      m,
                                      nnz,
                                      (const rocsparse_float_complex*)alpha,
                                      descr,
                                      (const rocsparse_float_complex*)csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      info,
                                      (const rocsparse_float_complex*)x,
                                      (rocsparse_float_complex*)y,
                                      policy,
                                      temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrsv(rocsparse_handle            handle,
                                     rocsparse_operation         trans,
                                     int                         m,
                                     int                         nnz,
                                     const std::complex<double>* alpha,
                                     const rocsparse_mat_descr   descr,
                                     const std::complex<double>* csr_val,
                                     const int*                  csr_row_ptr,
                                     const int*                  csr_col_ind,
                                     rocsparse_mat_info          info,
                                     const std::complex<double>* x,
                                     std::complex<double>*       y,
                                     rocsparse_solve_policy      policy,
                                     void*                       temp_buffer)
    {
        return rocsparse_zcsrsv_solve(handle,
                                      trans,
                                      m,
                                      nnz,
                                      (const rocsparse_double_complex*)alpha,
                                      descr,
                                      (const rocsparse_double_complex*)csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      info,
                                      (const rocsparse_double_complex*)x,
                                      (rocsparse_double_complex*)y,
                                      policy,
                                      temp_buffer);
    }

    // rocsprarse csritsv buffer size
    template <>
    rocsparse_status rocsparseTcsritsv_buffer_size(rocsparse_handle          handle,
                                                   rocsparse_operation       trans,
                                                   rocsparse_int             m,
                                                   rocsparse_int             nnz,
                                                   const rocsparse_mat_descr descr,
                                                   const float*              csr_val,
                                                   const rocsparse_int*      csr_row_ptr,
                                                   const rocsparse_int*      csr_col_ind,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size)
    {
        return rocsparse_scsritsv_buffer_size(
            handle, trans, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsritsv_buffer_size(rocsparse_handle          handle,
                                                   rocsparse_operation       trans,
                                                   rocsparse_int             m,
                                                   rocsparse_int             nnz,
                                                   const rocsparse_mat_descr descr,
                                                   const double*             csr_val,
                                                   const rocsparse_int*      csr_row_ptr,
                                                   const rocsparse_int*      csr_col_ind,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size)
    {
        return rocsparse_dcsritsv_buffer_size(
            handle, trans, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsritsv_buffer_size(rocsparse_handle           handle,
                                                   rocsparse_operation        trans,
                                                   rocsparse_int              m,
                                                   rocsparse_int              nnz,
                                                   const rocsparse_mat_descr  descr,
                                                   const std::complex<float>* csr_val,
                                                   const rocsparse_int*       csr_row_ptr,
                                                   const rocsparse_int*       csr_col_ind,
                                                   rocsparse_mat_info         info,
                                                   size_t*                    buffer_size)
    {
        return rocsparse_ccsritsv_buffer_size(handle,
                                              trans,
                                              m,
                                              nnz,
                                              descr,
                                              (const rocsparse_float_complex*)csr_val,
                                              csr_row_ptr,
                                              csr_col_ind,
                                              info,
                                              buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsritsv_buffer_size(rocsparse_handle            handle,
                                                   rocsparse_operation         trans,
                                                   rocsparse_int               m,
                                                   rocsparse_int               nnz,
                                                   const rocsparse_mat_descr   descr,
                                                   const std::complex<double>* csr_val,
                                                   const rocsparse_int*        csr_row_ptr,
                                                   const rocsparse_int*        csr_col_ind,
                                                   rocsparse_mat_info          info,
                                                   size_t*                     buffer_size)
    {
        return rocsparse_zcsritsv_buffer_size(handle,
                                              trans,
                                              m,
                                              nnz,
                                              descr,
                                              (const rocsparse_double_complex*)csr_val,
                                              csr_row_ptr,
                                              csr_col_ind,
                                              info,
                                              buffer_size);
    }

    // rocsprarse csritsv analysis
    template <>
    rocsparse_status rocsparseTcsritsv_analysis(rocsparse_handle          handle,
                                                rocsparse_operation       trans,
                                                rocsparse_int             m,
                                                rocsparse_int             nnz,
                                                const rocsparse_mat_descr descr,
                                                const float*              csr_val,
                                                const rocsparse_int*      csr_row_ptr,
                                                const rocsparse_int*      csr_col_ind,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                void*                     temp_buffer)
    {
        return rocsparse_scsritsv_analysis(handle,
                                           trans,
                                           m,
                                           nnz,
                                           descr,
                                           csr_val,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           info,
                                           analysis,
                                           solve,
                                           temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsritsv_analysis(rocsparse_handle          handle,
                                                rocsparse_operation       trans,
                                                rocsparse_int             m,
                                                rocsparse_int             nnz,
                                                const rocsparse_mat_descr descr,
                                                const double*             csr_val,
                                                const rocsparse_int*      csr_row_ptr,
                                                const rocsparse_int*      csr_col_ind,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                void*                     temp_buffer)
    {
        return rocsparse_dcsritsv_analysis(handle,
                                           trans,
                                           m,
                                           nnz,
                                           descr,
                                           csr_val,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           info,
                                           analysis,
                                           solve,
                                           temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsritsv_analysis(rocsparse_handle           handle,
                                                rocsparse_operation        trans,
                                                rocsparse_int              m,
                                                rocsparse_int              nnz,
                                                const rocsparse_mat_descr  descr,
                                                const std::complex<float>* csr_val,
                                                const rocsparse_int*       csr_row_ptr,
                                                const rocsparse_int*       csr_col_ind,
                                                rocsparse_mat_info         info,
                                                rocsparse_analysis_policy  analysis,
                                                rocsparse_solve_policy     solve,
                                                void*                      temp_buffer)
    {
        return rocsparse_ccsritsv_analysis(handle,
                                           trans,
                                           m,
                                           nnz,
                                           descr,
                                           (const rocsparse_float_complex*)csr_val,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           info,
                                           analysis,
                                           solve,
                                           temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsritsv_analysis(rocsparse_handle            handle,
                                                rocsparse_operation         trans,
                                                rocsparse_int               m,
                                                rocsparse_int               nnz,
                                                const rocsparse_mat_descr   descr,
                                                const std::complex<double>* csr_val,
                                                const rocsparse_int*        csr_row_ptr,
                                                const rocsparse_int*        csr_col_ind,
                                                rocsparse_mat_info          info,
                                                rocsparse_analysis_policy   analysis,
                                                rocsparse_solve_policy      solve,
                                                void*                       temp_buffer)
    {
        return rocsparse_zcsritsv_analysis(handle,
                                           trans,
                                           m,
                                           nnz,
                                           descr,
                                           (const rocsparse_double_complex*)csr_val,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           info,
                                           analysis,
                                           solve,
                                           temp_buffer);
    }

    // rocsprarse csritsv analysis
    template <>
    rocsparse_status rocsparseTcsritsv_solve(rocsparse_handle          handle,
                                             rocsparse_int*            host_nmaxiter,
                                             const float*              host_tol,
                                             float*                    host_history,
                                             rocsparse_operation       trans,
                                             rocsparse_int             m,
                                             rocsparse_int             nnz,
                                             const float*              alpha,
                                             const rocsparse_mat_descr descr,
                                             const float*              csr_val,
                                             const rocsparse_int*      csr_row_ptr,
                                             const rocsparse_int*      csr_col_ind,
                                             rocsparse_mat_info        info,
                                             const float*              x,
                                             float*                    y,
                                             rocsparse_solve_policy    policy,
                                             void*                     temp_buffer)
    {
        return rocsparse_scsritsv_solve(handle,
                                        host_nmaxiter,
                                        host_tol,
                                        host_history,
                                        trans,
                                        m,
                                        nnz,
                                        alpha,
                                        descr,
                                        csr_val,
                                        csr_row_ptr,
                                        csr_col_ind,
                                        info,
                                        x,
                                        y,
                                        policy,
                                        temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsritsv_solve(rocsparse_handle          handle,
                                             rocsparse_int*            host_nmaxiter,
                                             const double*             host_tol,
                                             double*                   host_history,
                                             rocsparse_operation       trans,
                                             rocsparse_int             m,
                                             rocsparse_int             nnz,
                                             const double*             alpha,
                                             const rocsparse_mat_descr descr,
                                             const double*             csr_val,
                                             const rocsparse_int*      csr_row_ptr,
                                             const rocsparse_int*      csr_col_ind,
                                             rocsparse_mat_info        info,
                                             const double*             x,
                                             double*                   y,
                                             rocsparse_solve_policy    policy,
                                             void*                     temp_buffer)
    {
        return rocsparse_dcsritsv_solve(handle,
                                        host_nmaxiter,
                                        host_tol,
                                        host_history,
                                        trans,
                                        m,
                                        nnz,
                                        alpha,
                                        descr,
                                        csr_val,
                                        csr_row_ptr,
                                        csr_col_ind,
                                        info,
                                        x,
                                        y,
                                        policy,
                                        temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsritsv_solve(rocsparse_handle           handle,
                                             rocsparse_int*             host_nmaxiter,
                                             const float*               host_tol,
                                             float*                     host_history,
                                             rocsparse_operation        trans,
                                             rocsparse_int              m,
                                             rocsparse_int              nnz,
                                             const std::complex<float>* alpha,
                                             const rocsparse_mat_descr  descr,
                                             const std::complex<float>* csr_val,
                                             const rocsparse_int*       csr_row_ptr,
                                             const rocsparse_int*       csr_col_ind,
                                             rocsparse_mat_info         info,
                                             const std::complex<float>* x,
                                             std::complex<float>*       y,
                                             rocsparse_solve_policy     policy,
                                             void*                      temp_buffer)
    {
        return rocsparse_ccsritsv_solve(handle,
                                        host_nmaxiter,
                                        host_tol,
                                        host_history,
                                        trans,
                                        m,
                                        nnz,
                                        (const rocsparse_float_complex*)alpha,
                                        descr,
                                        (const rocsparse_float_complex*)csr_val,
                                        csr_row_ptr,
                                        csr_col_ind,
                                        info,
                                        (const rocsparse_float_complex*)x,
                                        (rocsparse_float_complex*)y,
                                        policy,
                                        temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsritsv_solve(rocsparse_handle            handle,
                                             rocsparse_int*              host_nmaxiter,
                                             const double*               host_tol,
                                             double*                     host_history,
                                             rocsparse_operation         trans,
                                             rocsparse_int               m,
                                             rocsparse_int               nnz,
                                             const std::complex<double>* alpha,
                                             const rocsparse_mat_descr   descr,
                                             const std::complex<double>* csr_val,
                                             const rocsparse_int*        csr_row_ptr,
                                             const rocsparse_int*        csr_col_ind,
                                             rocsparse_mat_info          info,
                                             const std::complex<double>* x,
                                             std::complex<double>*       y,
                                             rocsparse_solve_policy      policy,
                                             void*                       temp_buffer)
    {
        return rocsparse_zcsritsv_solve(handle,
                                        host_nmaxiter,
                                        host_tol,
                                        host_history,
                                        trans,
                                        m,
                                        nnz,
                                        (const rocsparse_double_complex*)alpha,
                                        descr,
                                        (const rocsparse_double_complex*)csr_val,
                                        csr_row_ptr,
                                        csr_col_ind,
                                        info,
                                        (const rocsparse_double_complex*)x,
                                        (rocsparse_double_complex*)y,
                                        policy,
                                        temp_buffer);
    }

    // rocsparse bsrsv buffer size
    template <>
    rocsparse_status rocsparseTbsrsv_buffer_size(rocsparse_handle          handle,
                                                 rocsparse_direction       dir,
                                                 rocsparse_operation       trans,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const rocsparse_mat_descr descr,
                                                 const float*              bsr_val,
                                                 const int*                bsr_row_ptr,
                                                 const int*                bsr_col_ind,
                                                 int                       bsr_dim,
                                                 rocsparse_mat_info        info,
                                                 size_t*                   buffer_size)
    {
        return rocsparse_sbsrsv_buffer_size(handle,
                                            dir,
                                            trans,
                                            mb,
                                            nnzb,
                                            descr,
                                            bsr_val,
                                            bsr_row_ptr,
                                            bsr_col_ind,
                                            bsr_dim,
                                            info,
                                            buffer_size);
    }

    template <>
    rocsparse_status rocsparseTbsrsv_buffer_size(rocsparse_handle          handle,
                                                 rocsparse_direction       dir,
                                                 rocsparse_operation       trans,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const rocsparse_mat_descr descr,
                                                 const double*             bsr_val,
                                                 const int*                bsr_row_ptr,
                                                 const int*                bsr_col_ind,
                                                 int                       bsr_dim,
                                                 rocsparse_mat_info        info,
                                                 size_t*                   buffer_size)
    {
        return rocsparse_dbsrsv_buffer_size(handle,
                                            dir,
                                            trans,
                                            mb,
                                            nnzb,
                                            descr,
                                            bsr_val,
                                            bsr_row_ptr,
                                            bsr_col_ind,
                                            bsr_dim,
                                            info,
                                            buffer_size);
    }

    template <>
    rocsparse_status rocsparseTbsrsv_buffer_size(rocsparse_handle           handle,
                                                 rocsparse_direction        dir,
                                                 rocsparse_operation        trans,
                                                 int                        mb,
                                                 int                        nnzb,
                                                 const rocsparse_mat_descr  descr,
                                                 const std::complex<float>* bsr_val,
                                                 const int*                 bsr_row_ptr,
                                                 const int*                 bsr_col_ind,
                                                 int                        bsr_dim,
                                                 rocsparse_mat_info         info,
                                                 size_t*                    buffer_size)
    {
        return rocsparse_cbsrsv_buffer_size(handle,
                                            dir,
                                            trans,
                                            mb,
                                            nnzb,
                                            descr,
                                            (const rocsparse_float_complex*)bsr_val,
                                            bsr_row_ptr,
                                            bsr_col_ind,
                                            bsr_dim,
                                            info,
                                            buffer_size);
    }

    template <>
    rocsparse_status rocsparseTbsrsv_buffer_size(rocsparse_handle            handle,
                                                 rocsparse_direction         dir,
                                                 rocsparse_operation         trans,
                                                 int                         mb,
                                                 int                         nnzb,
                                                 const rocsparse_mat_descr   descr,
                                                 const std::complex<double>* bsr_val,
                                                 const int*                  bsr_row_ptr,
                                                 const int*                  bsr_col_ind,
                                                 int                         bsr_dim,
                                                 rocsparse_mat_info          info,
                                                 size_t*                     buffer_size)
    {
        return rocsparse_zbsrsv_buffer_size(handle,
                                            dir,
                                            trans,
                                            mb,
                                            nnzb,
                                            descr,
                                            (const rocsparse_double_complex*)bsr_val,
                                            bsr_row_ptr,
                                            bsr_col_ind,
                                            bsr_dim,
                                            info,
                                            buffer_size);
    }

    // rocsparse bsrsv analysis
    template <>
    rocsparse_status rocsparseTbsrsv_analysis(rocsparse_handle          handle,
                                              rocsparse_direction       dir,
                                              rocsparse_operation       trans,
                                              int                       mb,
                                              int                       nnzb,
                                              const rocsparse_mat_descr descr,
                                              const float*              bsr_val,
                                              const int*                bsr_row_ptr,
                                              const int*                bsr_col_ind,
                                              int                       bsr_dim,
                                              rocsparse_mat_info        info,
                                              rocsparse_analysis_policy analysis,
                                              rocsparse_solve_policy    solve,
                                              void*                     temp_buffer)
    {
        return rocsparse_sbsrsv_analysis(handle,
                                         dir,
                                         trans,
                                         mb,
                                         nnzb,
                                         descr,
                                         bsr_val,
                                         bsr_row_ptr,
                                         bsr_col_ind,
                                         bsr_dim,
                                         info,
                                         analysis,
                                         solve,
                                         temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTbsrsv_analysis(rocsparse_handle          handle,
                                              rocsparse_direction       dir,
                                              rocsparse_operation       trans,
                                              int                       mb,
                                              int                       nnzb,
                                              const rocsparse_mat_descr descr,
                                              const double*             bsr_val,
                                              const int*                bsr_row_ptr,
                                              const int*                bsr_col_ind,
                                              int                       bsr_dim,
                                              rocsparse_mat_info        info,
                                              rocsparse_analysis_policy analysis,
                                              rocsparse_solve_policy    solve,
                                              void*                     temp_buffer)
    {
        return rocsparse_dbsrsv_analysis(handle,
                                         dir,
                                         trans,
                                         mb,
                                         nnzb,
                                         descr,
                                         bsr_val,
                                         bsr_row_ptr,
                                         bsr_col_ind,
                                         bsr_dim,
                                         info,
                                         analysis,
                                         solve,
                                         temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTbsrsv_analysis(rocsparse_handle           handle,
                                              rocsparse_direction        dir,
                                              rocsparse_operation        trans,
                                              int                        mb,
                                              int                        nnzb,
                                              const rocsparse_mat_descr  descr,
                                              const std::complex<float>* bsr_val,
                                              const int*                 bsr_row_ptr,
                                              const int*                 bsr_col_ind,
                                              int                        bsr_dim,
                                              rocsparse_mat_info         info,
                                              rocsparse_analysis_policy  analysis,
                                              rocsparse_solve_policy     solve,
                                              void*                      temp_buffer)
    {
        return rocsparse_cbsrsv_analysis(handle,
                                         dir,
                                         trans,
                                         mb,
                                         nnzb,
                                         descr,
                                         (const rocsparse_float_complex*)bsr_val,
                                         bsr_row_ptr,
                                         bsr_col_ind,
                                         bsr_dim,
                                         info,
                                         analysis,
                                         solve,
                                         temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTbsrsv_analysis(rocsparse_handle            handle,
                                              rocsparse_direction         dir,
                                              rocsparse_operation         trans,
                                              int                         mb,
                                              int                         nnzb,
                                              const rocsparse_mat_descr   descr,
                                              const std::complex<double>* bsr_val,
                                              const int*                  bsr_row_ptr,
                                              const int*                  bsr_col_ind,
                                              int                         bsr_dim,
                                              rocsparse_mat_info          info,
                                              rocsparse_analysis_policy   analysis,
                                              rocsparse_solve_policy      solve,
                                              void*                       temp_buffer)
    {
        return rocsparse_zbsrsv_analysis(handle,
                                         dir,
                                         trans,
                                         mb,
                                         nnzb,
                                         descr,
                                         (const rocsparse_double_complex*)bsr_val,
                                         bsr_row_ptr,
                                         bsr_col_ind,
                                         bsr_dim,
                                         info,
                                         analysis,
                                         solve,
                                         temp_buffer);
    }

    // rocsparse bsrsv
    template <>
    rocsparse_status rocsparseTbsrsv(rocsparse_handle          handle,
                                     rocsparse_direction       dir,
                                     rocsparse_operation       trans,
                                     int                       mb,
                                     int                       nnzb,
                                     const float*              alpha,
                                     const rocsparse_mat_descr descr,
                                     const float*              bsr_val,
                                     const int*                bsr_row_ptr,
                                     const int*                bsr_col_ind,
                                     int                       bsr_dim,
                                     rocsparse_mat_info        info,
                                     const float*              x,
                                     float*                    y,
                                     rocsparse_solve_policy    policy,
                                     void*                     temp_buffer)
    {
        return rocsparse_sbsrsv_solve(handle,
                                      dir,
                                      trans,
                                      mb,
                                      nnzb,
                                      alpha,
                                      descr,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      bsr_dim,
                                      info,
                                      x,
                                      y,
                                      policy,
                                      temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTbsrsv(rocsparse_handle          handle,
                                     rocsparse_direction       dir,
                                     rocsparse_operation       trans,
                                     int                       mb,
                                     int                       nnzb,
                                     const double*             alpha,
                                     const rocsparse_mat_descr descr,
                                     const double*             bsr_val,
                                     const int*                bsr_row_ptr,
                                     const int*                bsr_col_ind,
                                     int                       bsr_dim,
                                     rocsparse_mat_info        info,
                                     const double*             x,
                                     double*                   y,
                                     rocsparse_solve_policy    policy,
                                     void*                     temp_buffer)
    {
        return rocsparse_dbsrsv_solve(handle,
                                      dir,
                                      trans,
                                      mb,
                                      nnzb,
                                      alpha,
                                      descr,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      bsr_dim,
                                      info,
                                      x,
                                      y,
                                      policy,
                                      temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTbsrsv(rocsparse_handle           handle,
                                     rocsparse_direction        dir,
                                     rocsparse_operation        trans,
                                     int                        mb,
                                     int                        nnzb,
                                     const std::complex<float>* alpha,
                                     const rocsparse_mat_descr  descr,
                                     const std::complex<float>* bsr_val,
                                     const int*                 bsr_row_ptr,
                                     const int*                 bsr_col_ind,
                                     int                        bsr_dim,
                                     rocsparse_mat_info         info,
                                     const std::complex<float>* x,
                                     std::complex<float>*       y,
                                     rocsparse_solve_policy     policy,
                                     void*                      temp_buffer)
    {
        return rocsparse_cbsrsv_solve(handle,
                                      dir,
                                      trans,
                                      mb,
                                      nnzb,
                                      (const rocsparse_float_complex*)alpha,
                                      descr,
                                      (const rocsparse_float_complex*)bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      bsr_dim,
                                      info,
                                      (const rocsparse_float_complex*)x,
                                      (rocsparse_float_complex*)y,
                                      policy,
                                      temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTbsrsv(rocsparse_handle            handle,
                                     rocsparse_direction         dir,
                                     rocsparse_operation         trans,
                                     int                         mb,
                                     int                         nnzb,
                                     const std::complex<double>* alpha,
                                     const rocsparse_mat_descr   descr,
                                     const std::complex<double>* bsr_val,
                                     const int*                  bsr_row_ptr,
                                     const int*                  bsr_col_ind,
                                     int                         bsr_dim,
                                     rocsparse_mat_info          info,
                                     const std::complex<double>* x,
                                     std::complex<double>*       y,
                                     rocsparse_solve_policy      policy,
                                     void*                       temp_buffer)
    {
        return rocsparse_zbsrsv_solve(handle,
                                      dir,
                                      trans,
                                      mb,
                                      nnzb,
                                      (const rocsparse_double_complex*)alpha,
                                      descr,
                                      (const rocsparse_double_complex*)bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      bsr_dim,
                                      info,
                                      (const rocsparse_double_complex*)x,
                                      (rocsparse_double_complex*)y,
                                      policy,
                                      temp_buffer);
    }

    // rocsparse coomv
    template <>
    rocsparse_status rocsparseTcoomv(rocsparse_handle          handle,
                                     rocsparse_operation       trans,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const float*              alpha,
                                     const rocsparse_mat_descr descr,
                                     const float*              coo_val,
                                     const int*                coo_row_ind,
                                     const int*                coo_col_ind,
                                     const float*              x,
                                     const float*              beta,
                                     float*                    y)
    {
        return rocsparse_scoomv(
            handle, trans, m, n, nnz, alpha, descr, coo_val, coo_row_ind, coo_col_ind, x, beta, y);
    }

    template <>
    rocsparse_status rocsparseTcoomv(rocsparse_handle          handle,
                                     rocsparse_operation       trans,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const double*             alpha,
                                     const rocsparse_mat_descr descr,
                                     const double*             coo_val,
                                     const int*                coo_row_ind,
                                     const int*                coo_col_ind,
                                     const double*             x,
                                     const double*             beta,
                                     double*                   y)
    {
        return rocsparse_dcoomv(
            handle, trans, m, n, nnz, alpha, descr, coo_val, coo_row_ind, coo_col_ind, x, beta, y);
    }

    template <>
    rocsparse_status rocsparseTcoomv(rocsparse_handle           handle,
                                     rocsparse_operation        trans,
                                     int                        m,
                                     int                        n,
                                     int                        nnz,
                                     const std::complex<float>* alpha,
                                     const rocsparse_mat_descr  descr,
                                     const std::complex<float>* coo_val,
                                     const int*                 coo_row_ind,
                                     const int*                 coo_col_ind,
                                     const std::complex<float>* x,
                                     const std::complex<float>* beta,
                                     std::complex<float>*       y)
    {
        return rocsparse_ccoomv(handle,
                                trans,
                                m,
                                n,
                                nnz,
                                (const rocsparse_float_complex*)alpha,
                                descr,
                                (const rocsparse_float_complex*)coo_val,
                                coo_row_ind,
                                coo_col_ind,
                                (const rocsparse_float_complex*)x,
                                (const rocsparse_float_complex*)beta,
                                (rocsparse_float_complex*)y);
    }

    template <>
    rocsparse_status rocsparseTcoomv(rocsparse_handle            handle,
                                     rocsparse_operation         trans,
                                     int                         m,
                                     int                         n,
                                     int                         nnz,
                                     const std::complex<double>* alpha,
                                     const rocsparse_mat_descr   descr,
                                     const std::complex<double>* coo_val,
                                     const int*                  coo_row_ind,
                                     const int*                  coo_col_ind,
                                     const std::complex<double>* x,
                                     const std::complex<double>* beta,
                                     std::complex<double>*       y)
    {
        return rocsparse_zcoomv(handle,
                                trans,
                                m,
                                n,
                                nnz,
                                (const rocsparse_double_complex*)alpha,
                                descr,
                                (const rocsparse_double_complex*)coo_val,
                                coo_row_ind,
                                coo_col_ind,
                                (const rocsparse_double_complex*)x,
                                (const rocsparse_double_complex*)beta,
                                (rocsparse_double_complex*)y);
    }

    // rocsparse ellmv
    template <>
    rocsparse_status rocsparseTellmv(rocsparse_handle          handle,
                                     rocsparse_operation       trans,
                                     int                       m,
                                     int                       n,
                                     const float*              alpha,
                                     const rocsparse_mat_descr descr,
                                     const float*              ell_val,
                                     const int*                ell_col_ind,
                                     int                       ell_width,
                                     const float*              x,
                                     const float*              beta,
                                     float*                    y)
    {
        return rocsparse_sellmv(
            handle, trans, m, n, alpha, descr, ell_val, ell_col_ind, ell_width, x, beta, y);
    }

    template <>
    rocsparse_status rocsparseTellmv(rocsparse_handle          handle,
                                     rocsparse_operation       trans,
                                     int                       m,
                                     int                       n,
                                     const double*             alpha,
                                     const rocsparse_mat_descr descr,
                                     const double*             ell_val,
                                     const int*                ell_col_ind,
                                     int                       ell_width,
                                     const double*             x,
                                     const double*             beta,
                                     double*                   y)
    {
        return rocsparse_dellmv(
            handle, trans, m, n, alpha, descr, ell_val, ell_col_ind, ell_width, x, beta, y);
    }

    template <>
    rocsparse_status rocsparseTellmv(rocsparse_handle           handle,
                                     rocsparse_operation        trans,
                                     int                        m,
                                     int                        n,
                                     const std::complex<float>* alpha,
                                     const rocsparse_mat_descr  descr,
                                     const std::complex<float>* ell_val,
                                     const int*                 ell_col_ind,
                                     int                        ell_width,
                                     const std::complex<float>* x,
                                     const std::complex<float>* beta,
                                     std::complex<float>*       y)
    {
        return rocsparse_cellmv(handle,
                                trans,
                                m,
                                n,
                                (const rocsparse_float_complex*)alpha,
                                descr,
                                (const rocsparse_float_complex*)ell_val,
                                ell_col_ind,
                                ell_width,
                                (const rocsparse_float_complex*)x,
                                (const rocsparse_float_complex*)beta,
                                (rocsparse_float_complex*)y);
    }

    template <>
    rocsparse_status rocsparseTellmv(rocsparse_handle            handle,
                                     rocsparse_operation         trans,
                                     int                         m,
                                     int                         n,
                                     const std::complex<double>* alpha,
                                     const rocsparse_mat_descr   descr,
                                     const std::complex<double>* ell_val,
                                     const int*                  ell_col_ind,
                                     int                         ell_width,
                                     const std::complex<double>* x,
                                     const std::complex<double>* beta,
                                     std::complex<double>*       y)
    {
        return rocsparse_zellmv(handle,
                                trans,
                                m,
                                n,
                                (const rocsparse_double_complex*)alpha,
                                descr,
                                (const rocsparse_double_complex*)ell_val,
                                ell_col_ind,
                                ell_width,
                                (const rocsparse_double_complex*)x,
                                (const rocsparse_double_complex*)beta,
                                (rocsparse_double_complex*)y);
    }

    // rocsparse bsrmv
    template <>
    rocsparse_status rocsparseTbsrmv(rocsparse_handle          handle,
                                     rocsparse_direction       dir,
                                     rocsparse_operation       trans,
                                     int                       mb,
                                     int                       nb,
                                     int                       nnzb,
                                     const float*              alpha,
                                     const rocsparse_mat_descr descr,
                                     const float*              bsr_val,
                                     const int*                bsr_row_ptr,
                                     const int*                bsr_col_ind,
                                     int                       bsr_dim,
                                     const float*              x,
                                     const float*              beta,
                                     float*                    y)
    {
        return rocsparse_sbsrmv(handle,
                                dir,
                                trans,
                                mb,
                                nb,
                                nnzb,
                                alpha,
                                descr,
                                bsr_val,
                                bsr_row_ptr,
                                bsr_col_ind,
                                bsr_dim,
#if ROCSPARSE_VERSION_MAJOR >= 3
                                nullptr,
#endif
                                x,
                                beta,
                                y);
    }

    template <>
    rocsparse_status rocsparseTbsrmv(rocsparse_handle          handle,
                                     rocsparse_direction       dir,
                                     rocsparse_operation       trans,
                                     int                       mb,
                                     int                       nb,
                                     int                       nnzb,
                                     const double*             alpha,
                                     const rocsparse_mat_descr descr,
                                     const double*             bsr_val,
                                     const int*                bsr_row_ptr,
                                     const int*                bsr_col_ind,
                                     int                       bsr_dim,
                                     const double*             x,
                                     const double*             beta,
                                     double*                   y)
    {
        return rocsparse_dbsrmv(handle,
                                dir,
                                trans,
                                mb,
                                nb,
                                nnzb,
                                alpha,
                                descr,
                                bsr_val,
                                bsr_row_ptr,
                                bsr_col_ind,
                                bsr_dim,
#if ROCSPARSE_VERSION_MAJOR >= 3
                                nullptr,
#endif
                                x,
                                beta,
                                y);
    }

    template <>
    rocsparse_status rocsparseTbsrmv(rocsparse_handle           handle,
                                     rocsparse_direction        dir,
                                     rocsparse_operation        trans,
                                     int                        mb,
                                     int                        nb,
                                     int                        nnzb,
                                     const std::complex<float>* alpha,
                                     const rocsparse_mat_descr  descr,
                                     const std::complex<float>* bsr_val,
                                     const int*                 bsr_row_ptr,
                                     const int*                 bsr_col_ind,
                                     int                        bsr_dim,
                                     const std::complex<float>* x,
                                     const std::complex<float>* beta,
                                     std::complex<float>*       y)
    {
        return rocsparse_cbsrmv(handle,
                                dir,
                                trans,
                                mb,
                                nb,
                                nnzb,
                                (const rocsparse_float_complex*)alpha,
                                descr,
                                (const rocsparse_float_complex*)bsr_val,
                                bsr_row_ptr,
                                bsr_col_ind,
                                bsr_dim,
#if ROCSPARSE_VERSION_MAJOR >= 3
                                nullptr,
#endif
                                (const rocsparse_float_complex*)x,
                                (const rocsparse_float_complex*)beta,
                                (rocsparse_float_complex*)y);
    }

    template <>
    rocsparse_status rocsparseTbsrmv(rocsparse_handle            handle,
                                     rocsparse_direction         dir,
                                     rocsparse_operation         trans,
                                     int                         mb,
                                     int                         nb,
                                     int                         nnzb,
                                     const std::complex<double>* alpha,
                                     const rocsparse_mat_descr   descr,
                                     const std::complex<double>* bsr_val,
                                     const int*                  bsr_row_ptr,
                                     const int*                  bsr_col_ind,
                                     int                         bsr_dim,
                                     const std::complex<double>* x,
                                     const std::complex<double>* beta,
                                     std::complex<double>*       y)
    {
        return rocsparse_zbsrmv(handle,
                                dir,
                                trans,
                                mb,
                                nb,
                                nnzb,
                                (const rocsparse_double_complex*)alpha,
                                descr,
                                (const rocsparse_double_complex*)bsr_val,
                                bsr_row_ptr,
                                bsr_col_ind,
                                bsr_dim,
#if ROCSPARSE_VERSION_MAJOR >= 3
                                nullptr,
#endif
                                (const rocsparse_double_complex*)x,
                                (const rocsparse_double_complex*)beta,
                                (rocsparse_double_complex*)y);
    }

    // rocsparse csrgeam
    template <>
    rocsparse_status rocsparseTcsrgeam(rocsparse_handle          handle,
                                       int                       m,
                                       int                       n,
                                       const float*              alpha,
                                       const rocsparse_mat_descr descr_A,
                                       int                       nnz_A,
                                       const float*              csr_val_A,
                                       const int*                csr_row_ptr_A,
                                       const int*                csr_col_ind_A,
                                       const float*              beta,
                                       const rocsparse_mat_descr descr_B,
                                       int                       nnz_B,
                                       const float*              csr_val_B,
                                       const int*                csr_row_ptr_B,
                                       const int*                csr_col_ind_B,
                                       const rocsparse_mat_descr descr_C,
                                       float*                    csr_val_C,
                                       const int*                csr_row_ptr_C,
                                       int*                      csr_col_ind_C)
    {
        return rocsparse_scsrgeam(handle,
                                  m,
                                  n,
                                  alpha,
                                  descr_A,
                                  nnz_A,
                                  csr_val_A,
                                  csr_row_ptr_A,
                                  csr_col_ind_A,
                                  beta,
                                  descr_B,
                                  nnz_B,
                                  csr_val_B,
                                  csr_row_ptr_B,
                                  csr_col_ind_B,
                                  descr_C,
                                  csr_val_C,
                                  csr_row_ptr_C,
                                  csr_col_ind_C);
    }

    template <>
    rocsparse_status rocsparseTcsrgeam(rocsparse_handle          handle,
                                       int                       m,
                                       int                       n,
                                       const double*             alpha,
                                       const rocsparse_mat_descr descr_A,
                                       int                       nnz_A,
                                       const double*             csr_val_A,
                                       const int*                csr_row_ptr_A,
                                       const int*                csr_col_ind_A,
                                       const double*             beta,
                                       const rocsparse_mat_descr descr_B,
                                       int                       nnz_B,
                                       const double*             csr_val_B,
                                       const int*                csr_row_ptr_B,
                                       const int*                csr_col_ind_B,
                                       const rocsparse_mat_descr descr_C,
                                       double*                   csr_val_C,
                                       const int*                csr_row_ptr_C,
                                       int*                      csr_col_ind_C)
    {
        return rocsparse_dcsrgeam(handle,
                                  m,
                                  n,
                                  alpha,
                                  descr_A,
                                  nnz_A,
                                  csr_val_A,
                                  csr_row_ptr_A,
                                  csr_col_ind_A,
                                  beta,
                                  descr_B,
                                  nnz_B,
                                  csr_val_B,
                                  csr_row_ptr_B,
                                  csr_col_ind_B,
                                  descr_C,
                                  csr_val_C,
                                  csr_row_ptr_C,
                                  csr_col_ind_C);
    }

    template <>
    rocsparse_status rocsparseTcsrgeam(rocsparse_handle           handle,
                                       int                        m,
                                       int                        n,
                                       const std::complex<float>* alpha,
                                       const rocsparse_mat_descr  descr_A,
                                       int                        nnz_A,
                                       const std::complex<float>* csr_val_A,
                                       const int*                 csr_row_ptr_A,
                                       const int*                 csr_col_ind_A,
                                       const std::complex<float>* beta,
                                       const rocsparse_mat_descr  descr_B,
                                       int                        nnz_B,
                                       const std::complex<float>* csr_val_B,
                                       const int*                 csr_row_ptr_B,
                                       const int*                 csr_col_ind_B,
                                       const rocsparse_mat_descr  descr_C,
                                       std::complex<float>*       csr_val_C,
                                       const int*                 csr_row_ptr_C,
                                       int*                       csr_col_ind_C)
    {
        return rocsparse_ccsrgeam(handle,
                                  m,
                                  n,
                                  (const rocsparse_float_complex*)alpha,
                                  descr_A,
                                  nnz_A,
                                  (const rocsparse_float_complex*)csr_val_A,
                                  csr_row_ptr_A,
                                  csr_col_ind_A,
                                  (const rocsparse_float_complex*)beta,
                                  descr_B,
                                  nnz_B,
                                  (const rocsparse_float_complex*)csr_val_B,
                                  csr_row_ptr_B,
                                  csr_col_ind_B,
                                  descr_C,
                                  (rocsparse_float_complex*)csr_val_C,
                                  csr_row_ptr_C,
                                  csr_col_ind_C);
    }

    template <>
    rocsparse_status rocsparseTcsrgeam(rocsparse_handle            handle,
                                       int                         m,
                                       int                         n,
                                       const std::complex<double>* alpha,
                                       const rocsparse_mat_descr   descr_A,
                                       int                         nnz_A,
                                       const std::complex<double>* csr_val_A,
                                       const int*                  csr_row_ptr_A,
                                       const int*                  csr_col_ind_A,
                                       const std::complex<double>* beta,
                                       const rocsparse_mat_descr   descr_B,
                                       int                         nnz_B,
                                       const std::complex<double>* csr_val_B,
                                       const int*                  csr_row_ptr_B,
                                       const int*                  csr_col_ind_B,
                                       const rocsparse_mat_descr   descr_C,
                                       std::complex<double>*       csr_val_C,
                                       const int*                  csr_row_ptr_C,
                                       int*                        csr_col_ind_C)
    {
        return rocsparse_zcsrgeam(handle,
                                  m,
                                  n,
                                  (const rocsparse_double_complex*)alpha,
                                  descr_A,
                                  nnz_A,
                                  (const rocsparse_double_complex*)csr_val_A,
                                  csr_row_ptr_A,
                                  csr_col_ind_A,
                                  (const rocsparse_double_complex*)beta,
                                  descr_B,
                                  nnz_B,
                                  (const rocsparse_double_complex*)csr_val_B,
                                  csr_row_ptr_B,
                                  csr_col_ind_B,
                                  descr_C,
                                  (rocsparse_double_complex*)csr_val_C,
                                  csr_row_ptr_C,
                                  csr_col_ind_C);
    }

    // rocsparse csrgemm
    template <>
    rocsparse_status rocsparseTcsrgemm_buffer_size(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_B,
                                                   int                       m,
                                                   int                       n,
                                                   int                       k,
                                                   const float*              alpha,
                                                   const rocsparse_mat_descr descr_A,
                                                   int                       nnz_A,
                                                   const int*                csr_row_ptr_A,
                                                   const int*                csr_col_ind_A,
                                                   const rocsparse_mat_descr descr_B,
                                                   int                       nnz_B,
                                                   const int*                csr_row_ptr_B,
                                                   const int*                csr_col_ind_B,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size)
    {
        return rocsparse_scsrgemm_buffer_size(handle,
                                              trans_A,
                                              trans_B,
                                              m,
                                              n,
                                              k,
                                              alpha,
                                              descr_A,
                                              nnz_A,
                                              csr_row_ptr_A,
                                              csr_col_ind_A,
                                              descr_B,
                                              nnz_B,
                                              csr_row_ptr_B,
                                              csr_col_ind_B,
                                              NULL,
                                              NULL,
                                              0,
                                              NULL,
                                              NULL,
                                              info,
                                              buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsrgemm_buffer_size(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_B,
                                                   int                       m,
                                                   int                       n,
                                                   int                       k,
                                                   const double*             alpha,
                                                   const rocsparse_mat_descr descr_A,
                                                   int                       nnz_A,
                                                   const int*                csr_row_ptr_A,
                                                   const int*                csr_col_ind_A,
                                                   const rocsparse_mat_descr descr_B,
                                                   int                       nnz_B,
                                                   const int*                csr_row_ptr_B,
                                                   const int*                csr_col_ind_B,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size)
    {
        return rocsparse_dcsrgemm_buffer_size(handle,
                                              trans_A,
                                              trans_B,
                                              m,
                                              n,
                                              k,
                                              alpha,
                                              descr_A,
                                              nnz_A,
                                              csr_row_ptr_A,
                                              csr_col_ind_A,
                                              descr_B,
                                              nnz_B,
                                              csr_row_ptr_B,
                                              csr_col_ind_B,
                                              NULL,
                                              NULL,
                                              0,
                                              NULL,
                                              NULL,
                                              info,
                                              buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsrgemm_buffer_size(rocsparse_handle           handle,
                                                   rocsparse_operation        trans_A,
                                                   rocsparse_operation        trans_B,
                                                   int                        m,
                                                   int                        n,
                                                   int                        k,
                                                   const std::complex<float>* alpha,
                                                   const rocsparse_mat_descr  descr_A,
                                                   int                        nnz_A,
                                                   const int*                 csr_row_ptr_A,
                                                   const int*                 csr_col_ind_A,
                                                   const rocsparse_mat_descr  descr_B,
                                                   int                        nnz_B,
                                                   const int*                 csr_row_ptr_B,
                                                   const int*                 csr_col_ind_B,
                                                   rocsparse_mat_info         info,
                                                   size_t*                    buffer_size)
    {
        return rocsparse_ccsrgemm_buffer_size(handle,
                                              trans_A,
                                              trans_B,
                                              m,
                                              n,
                                              k,
                                              (const rocsparse_float_complex*)alpha,
                                              descr_A,
                                              nnz_A,
                                              csr_row_ptr_A,
                                              csr_col_ind_A,
                                              descr_B,
                                              nnz_B,
                                              csr_row_ptr_B,
                                              csr_col_ind_B,
                                              NULL,
                                              NULL,
                                              0,
                                              NULL,
                                              NULL,
                                              info,
                                              buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsrgemm_buffer_size(rocsparse_handle            handle,
                                                   rocsparse_operation         trans_A,
                                                   rocsparse_operation         trans_B,
                                                   int                         m,
                                                   int                         n,
                                                   int                         k,
                                                   const std::complex<double>* alpha,
                                                   const rocsparse_mat_descr   descr_A,
                                                   int                         nnz_A,
                                                   const int*                  csr_row_ptr_A,
                                                   const int*                  csr_col_ind_A,
                                                   const rocsparse_mat_descr   descr_B,
                                                   int                         nnz_B,
                                                   const int*                  csr_row_ptr_B,
                                                   const int*                  csr_col_ind_B,
                                                   rocsparse_mat_info          info,
                                                   size_t*                     buffer_size)
    {
        return rocsparse_zcsrgemm_buffer_size(handle,
                                              trans_A,
                                              trans_B,
                                              m,
                                              n,
                                              k,
                                              (const rocsparse_double_complex*)alpha,
                                              descr_A,
                                              nnz_A,
                                              csr_row_ptr_A,
                                              csr_col_ind_A,
                                              descr_B,
                                              nnz_B,
                                              csr_row_ptr_B,
                                              csr_col_ind_B,
                                              NULL,
                                              NULL,
                                              0,
                                              NULL,
                                              NULL,
                                              info,
                                              buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsrgemm(rocsparse_handle          handle,
                                       rocsparse_operation       trans_A,
                                       rocsparse_operation       trans_B,
                                       int                       m,
                                       int                       n,
                                       int                       k,
                                       const float*              alpha,
                                       const rocsparse_mat_descr descr_A,
                                       int                       nnz_A,
                                       const float*              csr_val_A,
                                       const int*                csr_row_ptr_A,
                                       const int*                csr_col_ind_A,
                                       const rocsparse_mat_descr descr_B,
                                       int                       nnz_B,
                                       const float*              csr_val_B,
                                       const int*                csr_row_ptr_B,
                                       const int*                csr_col_ind_B,
                                       const rocsparse_mat_descr descr_C,
                                       float*                    csr_val_C,
                                       const int*                csr_row_ptr_C,
                                       int*                      csr_col_ind_C,
                                       rocsparse_mat_info        info,
                                       void*                     temp_buffer)
    {
        return rocsparse_scsrgemm(handle,
                                  trans_A,
                                  trans_B,
                                  m,
                                  n,
                                  k,
                                  alpha,
                                  descr_A,
                                  nnz_A,
                                  csr_val_A,
                                  csr_row_ptr_A,
                                  csr_col_ind_A,
                                  descr_B,
                                  nnz_B,
                                  csr_val_B,
                                  csr_row_ptr_B,
                                  csr_col_ind_B,
                                  NULL,
                                  NULL,
                                  0,
                                  NULL,
                                  NULL,
                                  NULL,
                                  descr_C,
                                  csr_val_C,
                                  csr_row_ptr_C,
                                  csr_col_ind_C,
                                  info,
                                  temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrgemm(rocsparse_handle          handle,
                                       rocsparse_operation       trans_A,
                                       rocsparse_operation       trans_B,
                                       int                       m,
                                       int                       n,
                                       int                       k,
                                       const double*             alpha,
                                       const rocsparse_mat_descr descr_A,
                                       int                       nnz_A,
                                       const double*             csr_val_A,
                                       const int*                csr_row_ptr_A,
                                       const int*                csr_col_ind_A,
                                       const rocsparse_mat_descr descr_B,
                                       int                       nnz_B,
                                       const double*             csr_val_B,
                                       const int*                csr_row_ptr_B,
                                       const int*                csr_col_ind_B,
                                       const rocsparse_mat_descr descr_C,
                                       double*                   csr_val_C,
                                       const int*                csr_row_ptr_C,
                                       int*                      csr_col_ind_C,
                                       rocsparse_mat_info        info,
                                       void*                     temp_buffer)
    {
        return rocsparse_dcsrgemm(handle,
                                  trans_A,
                                  trans_B,
                                  m,
                                  n,
                                  k,
                                  alpha,
                                  descr_A,
                                  nnz_A,
                                  csr_val_A,
                                  csr_row_ptr_A,
                                  csr_col_ind_A,
                                  descr_B,
                                  nnz_B,
                                  csr_val_B,
                                  csr_row_ptr_B,
                                  csr_col_ind_B,
                                  NULL,
                                  NULL,
                                  0,
                                  NULL,
                                  NULL,
                                  NULL,
                                  descr_C,
                                  csr_val_C,
                                  csr_row_ptr_C,
                                  csr_col_ind_C,
                                  info,
                                  temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrgemm(rocsparse_handle           handle,
                                       rocsparse_operation        trans_A,
                                       rocsparse_operation        trans_B,
                                       int                        m,
                                       int                        n,
                                       int                        k,
                                       const std::complex<float>* alpha,
                                       const rocsparse_mat_descr  descr_A,
                                       int                        nnz_A,
                                       const std::complex<float>* csr_val_A,
                                       const int*                 csr_row_ptr_A,
                                       const int*                 csr_col_ind_A,
                                       const rocsparse_mat_descr  descr_B,
                                       int                        nnz_B,
                                       const std::complex<float>* csr_val_B,
                                       const int*                 csr_row_ptr_B,
                                       const int*                 csr_col_ind_B,
                                       const rocsparse_mat_descr  descr_C,
                                       std::complex<float>*       csr_val_C,
                                       const int*                 csr_row_ptr_C,
                                       int*                       csr_col_ind_C,
                                       rocsparse_mat_info         info,
                                       void*                      temp_buffer)
    {
        return rocsparse_ccsrgemm(handle,
                                  trans_A,
                                  trans_B,
                                  m,
                                  n,
                                  k,
                                  (const rocsparse_float_complex*)alpha,
                                  descr_A,
                                  nnz_A,
                                  (const rocsparse_float_complex*)csr_val_A,
                                  csr_row_ptr_A,
                                  csr_col_ind_A,
                                  descr_B,
                                  nnz_B,
                                  (const rocsparse_float_complex*)csr_val_B,
                                  csr_row_ptr_B,
                                  csr_col_ind_B,
                                  NULL,
                                  NULL,
                                  0,
                                  NULL,
                                  NULL,
                                  NULL,
                                  descr_C,
                                  (rocsparse_float_complex*)csr_val_C,
                                  csr_row_ptr_C,
                                  csr_col_ind_C,
                                  info,
                                  temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrgemm(rocsparse_handle            handle,
                                       rocsparse_operation         trans_A,
                                       rocsparse_operation         trans_B,
                                       int                         m,
                                       int                         n,
                                       int                         k,
                                       const std::complex<double>* alpha,
                                       const rocsparse_mat_descr   descr_A,
                                       int                         nnz_A,
                                       const std::complex<double>* csr_val_A,
                                       const int*                  csr_row_ptr_A,
                                       const int*                  csr_col_ind_A,
                                       const rocsparse_mat_descr   descr_B,
                                       int                         nnz_B,
                                       const std::complex<double>* csr_val_B,
                                       const int*                  csr_row_ptr_B,
                                       const int*                  csr_col_ind_B,
                                       const rocsparse_mat_descr   descr_C,
                                       std::complex<double>*       csr_val_C,
                                       const int*                  csr_row_ptr_C,
                                       int*                        csr_col_ind_C,
                                       rocsparse_mat_info          info,
                                       void*                       temp_buffer)
    {
        return rocsparse_zcsrgemm(handle,
                                  trans_A,
                                  trans_B,
                                  m,
                                  n,
                                  k,
                                  (const rocsparse_double_complex*)alpha,
                                  descr_A,
                                  nnz_A,
                                  (const rocsparse_double_complex*)csr_val_A,
                                  csr_row_ptr_A,
                                  csr_col_ind_A,
                                  descr_B,
                                  nnz_B,
                                  (const rocsparse_double_complex*)csr_val_B,
                                  csr_row_ptr_B,
                                  csr_col_ind_B,
                                  NULL,
                                  NULL,
                                  0,
                                  NULL,
                                  NULL,
                                  NULL,
                                  descr_C,
                                  (rocsparse_double_complex*)csr_val_C,
                                  csr_row_ptr_C,
                                  csr_col_ind_C,
                                  info,
                                  temp_buffer);
    }

    // rocsparse csric0 buffer size
    template <>
    rocsparse_status rocsparseTcsric0_buffer_size(rocsparse_handle          handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const rocsparse_mat_descr descr,
                                                  float*                    csr_val,
                                                  const int*                csr_row_ptr,
                                                  const int*                csr_col_ind,
                                                  rocsparse_mat_info        info,
                                                  size_t*                   buffer_size)
    {
        return rocsparse_scsric0_buffer_size(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsric0_buffer_size(rocsparse_handle          handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const rocsparse_mat_descr descr,
                                                  double*                   csr_val,
                                                  const int*                csr_row_ptr,
                                                  const int*                csr_col_ind,
                                                  rocsparse_mat_info        info,
                                                  size_t*                   buffer_size)
    {
        return rocsparse_dcsric0_buffer_size(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsric0_buffer_size(rocsparse_handle          handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const rocsparse_mat_descr descr,
                                                  std::complex<float>*      csr_val,
                                                  const int*                csr_row_ptr,
                                                  const int*                csr_col_ind,
                                                  rocsparse_mat_info        info,
                                                  size_t*                   buffer_size)
    {
        return rocsparse_ccsric0_buffer_size(handle,
                                             m,
                                             nnz,
                                             descr,
                                             (rocsparse_float_complex*)csr_val,
                                             csr_row_ptr,
                                             csr_col_ind,
                                             info,
                                             buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsric0_buffer_size(rocsparse_handle          handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const rocsparse_mat_descr descr,
                                                  std::complex<double>*     csr_val,
                                                  const int*                csr_row_ptr,
                                                  const int*                csr_col_ind,
                                                  rocsparse_mat_info        info,
                                                  size_t*                   buffer_size)
    {
        return rocsparse_zcsric0_buffer_size(handle,
                                             m,
                                             nnz,
                                             descr,
                                             (rocsparse_double_complex*)csr_val,
                                             csr_row_ptr,
                                             csr_col_ind,
                                             info,
                                             buffer_size);
    }

    // rocsparse csric0 analysis
    template <>
    rocsparse_status rocsparseTcsric0_analysis(rocsparse_handle          handle,
                                               int                       m,
                                               int                       nnz,
                                               const rocsparse_mat_descr descr,
                                               float*                    csr_val,
                                               const int*                csr_row_ptr,
                                               const int*                csr_col_ind,
                                               rocsparse_mat_info        info,
                                               rocsparse_analysis_policy analysis,
                                               rocsparse_solve_policy    solve,
                                               void*                     temp_buffer)
    {
        return rocsparse_scsric0_analysis(handle,
                                          m,
                                          nnz,
                                          descr,
                                          csr_val,
                                          csr_row_ptr,
                                          csr_col_ind,
                                          info,
                                          analysis,
                                          solve,
                                          temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsric0_analysis(rocsparse_handle          handle,
                                               int                       m,
                                               int                       nnz,
                                               const rocsparse_mat_descr descr,
                                               double*                   csr_val,
                                               const int*                csr_row_ptr,
                                               const int*                csr_col_ind,
                                               rocsparse_mat_info        info,
                                               rocsparse_analysis_policy analysis,
                                               rocsparse_solve_policy    solve,
                                               void*                     temp_buffer)
    {
        return rocsparse_dcsric0_analysis(handle,
                                          m,
                                          nnz,
                                          descr,
                                          csr_val,
                                          csr_row_ptr,
                                          csr_col_ind,
                                          info,
                                          analysis,
                                          solve,
                                          temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsric0_analysis(rocsparse_handle          handle,
                                               int                       m,
                                               int                       nnz,
                                               const rocsparse_mat_descr descr,
                                               std::complex<float>*      csr_val,
                                               const int*                csr_row_ptr,
                                               const int*                csr_col_ind,
                                               rocsparse_mat_info        info,
                                               rocsparse_analysis_policy analysis,
                                               rocsparse_solve_policy    solve,
                                               void*                     temp_buffer)
    {
        return rocsparse_ccsric0_analysis(handle,
                                          m,
                                          nnz,
                                          descr,
                                          (rocsparse_float_complex*)csr_val,
                                          csr_row_ptr,
                                          csr_col_ind,
                                          info,
                                          analysis,
                                          solve,
                                          temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsric0_analysis(rocsparse_handle          handle,
                                               int                       m,
                                               int                       nnz,
                                               const rocsparse_mat_descr descr,
                                               std::complex<double>*     csr_val,
                                               const int*                csr_row_ptr,
                                               const int*                csr_col_ind,
                                               rocsparse_mat_info        info,
                                               rocsparse_analysis_policy analysis,
                                               rocsparse_solve_policy    solve,
                                               void*                     temp_buffer)
    {
        return rocsparse_zcsric0_analysis(handle,
                                          m,
                                          nnz,
                                          descr,
                                          (rocsparse_double_complex*)csr_val,
                                          csr_row_ptr,
                                          csr_col_ind,
                                          info,
                                          analysis,
                                          solve,
                                          temp_buffer);
    }

    // rocsparse csric0
    template <>
    rocsparse_status rocsparseTcsric0(rocsparse_handle          handle,
                                      int                       m,
                                      int                       nnz,
                                      const rocsparse_mat_descr descr,
                                      float*                    csr_val,
                                      const int*                csr_row_ptr,
                                      const int*                csr_col_ind,
                                      rocsparse_mat_info        info,
                                      rocsparse_solve_policy    policy,
                                      void*                     temp_buffer)
    {
        return rocsparse_scsric0(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsric0(rocsparse_handle          handle,
                                      int                       m,
                                      int                       nnz,
                                      const rocsparse_mat_descr descr,
                                      double*                   csr_val,
                                      const int*                csr_row_ptr,
                                      const int*                csr_col_ind,
                                      rocsparse_mat_info        info,
                                      rocsparse_solve_policy    policy,
                                      void*                     temp_buffer)
    {
        return rocsparse_dcsric0(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsric0(rocsparse_handle          handle,
                                      int                       m,
                                      int                       nnz,
                                      const rocsparse_mat_descr descr,
                                      std::complex<float>*      csr_val,
                                      const int*                csr_row_ptr,
                                      const int*                csr_col_ind,
                                      rocsparse_mat_info        info,
                                      rocsparse_solve_policy    policy,
                                      void*                     temp_buffer)
    {
        return rocsparse_ccsric0(handle,
                                 m,
                                 nnz,
                                 descr,
                                 (rocsparse_float_complex*)csr_val,
                                 csr_row_ptr,
                                 csr_col_ind,
                                 info,
                                 policy,
                                 temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsric0(rocsparse_handle          handle,
                                      int                       m,
                                      int                       nnz,
                                      const rocsparse_mat_descr descr,
                                      std::complex<double>*     csr_val,
                                      const int*                csr_row_ptr,
                                      const int*                csr_col_ind,
                                      rocsparse_mat_info        info,
                                      rocsparse_solve_policy    policy,
                                      void*                     temp_buffer)
    {
        return rocsparse_zcsric0(handle,
                                 m,
                                 nnz,
                                 descr,
                                 (rocsparse_double_complex*)csr_val,
                                 csr_row_ptr,
                                 csr_col_ind,
                                 info,
                                 policy,
                                 temp_buffer);
    }

    // rocsparse bsric0 buffer size
    template <>
    rocsparse_status rocsparseTbsric0_buffer_size(rocsparse_handle          handle,
                                                  rocsparse_direction       dir,
                                                  int                       mb,
                                                  int                       nnzb,
                                                  const rocsparse_mat_descr descr,
                                                  float*                    bsr_val,
                                                  const int*                bsr_row_ptr,
                                                  const int*                bsr_col_ind,
                                                  int                       bsr_dim,
                                                  rocsparse_mat_info        info,
                                                  size_t*                   buffer_size)
    {
        return rocsparse_sbsric0_buffer_size(handle,
                                             dir,
                                             mb,
                                             nnzb,
                                             descr,
                                             bsr_val,
                                             bsr_row_ptr,
                                             bsr_col_ind,
                                             bsr_dim,
                                             info,
                                             buffer_size);
    }

    template <>
    rocsparse_status rocsparseTbsric0_buffer_size(rocsparse_handle          handle,
                                                  rocsparse_direction       dir,
                                                  int                       mb,
                                                  int                       nnzb,
                                                  const rocsparse_mat_descr descr,
                                                  double*                   bsr_val,
                                                  const int*                bsr_row_ptr,
                                                  const int*                bsr_col_ind,
                                                  int                       bsr_dim,
                                                  rocsparse_mat_info        info,
                                                  size_t*                   buffer_size)
    {
        return rocsparse_dbsric0_buffer_size(handle,
                                             dir,
                                             mb,
                                             nnzb,
                                             descr,
                                             bsr_val,
                                             bsr_row_ptr,
                                             bsr_col_ind,
                                             bsr_dim,
                                             info,
                                             buffer_size);
    }

    template <>
    rocsparse_status rocsparseTbsric0_buffer_size(rocsparse_handle          handle,
                                                  rocsparse_direction       dir,
                                                  int                       mb,
                                                  int                       nnzb,
                                                  const rocsparse_mat_descr descr,
                                                  std::complex<float>*      bsr_val,
                                                  const int*                bsr_row_ptr,
                                                  const int*                bsr_col_ind,
                                                  int                       bsr_dim,
                                                  rocsparse_mat_info        info,
                                                  size_t*                   buffer_size)
    {
        return rocsparse_cbsric0_buffer_size(handle,
                                             dir,
                                             mb,
                                             nnzb,
                                             descr,
                                             (rocsparse_float_complex*)bsr_val,
                                             bsr_row_ptr,
                                             bsr_col_ind,
                                             bsr_dim,
                                             info,
                                             buffer_size);
    }

    template <>
    rocsparse_status rocsparseTbsric0_buffer_size(rocsparse_handle          handle,
                                                  rocsparse_direction       dir,
                                                  int                       mb,
                                                  int                       nnzb,
                                                  const rocsparse_mat_descr descr,
                                                  std::complex<double>*     bsr_val,
                                                  const int*                bsr_row_ptr,
                                                  const int*                bsr_col_ind,
                                                  int                       bsr_dim,
                                                  rocsparse_mat_info        info,
                                                  size_t*                   buffer_size)
    {
        return rocsparse_zbsric0_buffer_size(handle,
                                             dir,
                                             mb,
                                             nnzb,
                                             descr,
                                             (rocsparse_double_complex*)bsr_val,
                                             bsr_row_ptr,
                                             bsr_col_ind,
                                             bsr_dim,
                                             info,
                                             buffer_size);
    }

    // rocsparse bsric0 analysis
    template <>
    rocsparse_status rocsparseTbsric0_analysis(rocsparse_handle          handle,
                                               rocsparse_direction       dir,
                                               int                       mb,
                                               int                       nnzb,
                                               const rocsparse_mat_descr descr,
                                               float*                    bsr_val,
                                               const int*                bsr_row_ptr,
                                               const int*                bsr_col_ind,
                                               int                       bsr_dim,
                                               rocsparse_mat_info        info,
                                               rocsparse_analysis_policy analysis,
                                               rocsparse_solve_policy    solve,
                                               void*                     temp_buffer)
    {
        return rocsparse_sbsric0_analysis(handle,
                                          dir,
                                          mb,
                                          nnzb,
                                          descr,
                                          bsr_val,
                                          bsr_row_ptr,
                                          bsr_col_ind,
                                          bsr_dim,
                                          info,
                                          analysis,
                                          solve,
                                          temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTbsric0_analysis(rocsparse_handle          handle,
                                               rocsparse_direction       dir,
                                               int                       mb,
                                               int                       nnzb,
                                               const rocsparse_mat_descr descr,
                                               double*                   bsr_val,
                                               const int*                bsr_row_ptr,
                                               const int*                bsr_col_ind,
                                               int                       bsr_dim,
                                               rocsparse_mat_info        info,
                                               rocsparse_analysis_policy analysis,
                                               rocsparse_solve_policy    solve,
                                               void*                     temp_buffer)
    {
        return rocsparse_dbsric0_analysis(handle,
                                          dir,
                                          mb,
                                          nnzb,
                                          descr,
                                          bsr_val,
                                          bsr_row_ptr,
                                          bsr_col_ind,
                                          bsr_dim,
                                          info,
                                          analysis,
                                          solve,
                                          temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTbsric0_analysis(rocsparse_handle          handle,
                                               rocsparse_direction       dir,
                                               int                       mb,
                                               int                       nnzb,
                                               const rocsparse_mat_descr descr,
                                               std::complex<float>*      bsr_val,
                                               const int*                bsr_row_ptr,
                                               const int*                bsr_col_ind,
                                               int                       bsr_dim,
                                               rocsparse_mat_info        info,
                                               rocsparse_analysis_policy analysis,
                                               rocsparse_solve_policy    solve,
                                               void*                     temp_buffer)
    {
        return rocsparse_cbsric0_analysis(handle,
                                          dir,
                                          mb,
                                          nnzb,
                                          descr,
                                          (rocsparse_float_complex*)bsr_val,
                                          bsr_row_ptr,
                                          bsr_col_ind,
                                          bsr_dim,
                                          info,
                                          analysis,
                                          solve,
                                          temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTbsric0_analysis(rocsparse_handle          handle,
                                               rocsparse_direction       dir,
                                               int                       mb,
                                               int                       nnzb,
                                               const rocsparse_mat_descr descr,
                                               std::complex<double>*     bsr_val,
                                               const int*                bsr_row_ptr,
                                               const int*                bsr_col_ind,
                                               int                       bsr_dim,
                                               rocsparse_mat_info        info,
                                               rocsparse_analysis_policy analysis,
                                               rocsparse_solve_policy    solve,
                                               void*                     temp_buffer)
    {
        return rocsparse_zbsric0_analysis(handle,
                                          dir,
                                          mb,
                                          nnzb,
                                          descr,
                                          (rocsparse_double_complex*)bsr_val,
                                          bsr_row_ptr,
                                          bsr_col_ind,
                                          bsr_dim,
                                          info,
                                          analysis,
                                          solve,
                                          temp_buffer);
    }

    // rocsparse bsric0
    template <>
    rocsparse_status rocsparseTbsric0(rocsparse_handle          handle,
                                      rocsparse_direction       dir,
                                      int                       mb,
                                      int                       nnzb,
                                      const rocsparse_mat_descr descr,
                                      float*                    bsr_val,
                                      const int*                bsr_row_ptr,
                                      const int*                bsr_col_ind,
                                      int                       bsr_dim,
                                      rocsparse_mat_info        info,
                                      rocsparse_solve_policy    policy,
                                      void*                     temp_buffer)
    {
        return rocsparse_sbsric0(handle,
                                 dir,
                                 mb,
                                 nnzb,
                                 descr,
                                 bsr_val,
                                 bsr_row_ptr,
                                 bsr_col_ind,
                                 bsr_dim,
                                 info,
                                 policy,
                                 temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTbsric0(rocsparse_handle          handle,
                                      rocsparse_direction       dir,
                                      int                       mb,
                                      int                       nnzb,
                                      const rocsparse_mat_descr descr,
                                      double*                   bsr_val,
                                      const int*                bsr_row_ptr,
                                      const int*                bsr_col_ind,
                                      int                       bsr_dim,
                                      rocsparse_mat_info        info,
                                      rocsparse_solve_policy    policy,
                                      void*                     temp_buffer)
    {
        return rocsparse_dbsric0(handle,
                                 dir,
                                 mb,
                                 nnzb,
                                 descr,
                                 bsr_val,
                                 bsr_row_ptr,
                                 bsr_col_ind,
                                 bsr_dim,
                                 info,
                                 policy,
                                 temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTbsric0(rocsparse_handle          handle,
                                      rocsparse_direction       dir,
                                      int                       mb,
                                      int                       nnzb,
                                      const rocsparse_mat_descr descr,
                                      std::complex<float>*      bsr_val,
                                      const int*                bsr_row_ptr,
                                      const int*                bsr_col_ind,
                                      int                       bsr_dim,
                                      rocsparse_mat_info        info,
                                      rocsparse_solve_policy    policy,
                                      void*                     temp_buffer)
    {
        return rocsparse_cbsric0(handle,
                                 dir,
                                 mb,
                                 nnzb,
                                 descr,
                                 (rocsparse_float_complex*)bsr_val,
                                 bsr_row_ptr,
                                 bsr_col_ind,
                                 bsr_dim,
                                 info,
                                 policy,
                                 temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTbsric0(rocsparse_handle          handle,
                                      rocsparse_direction       dir,
                                      int                       mb,
                                      int                       nnzb,
                                      const rocsparse_mat_descr descr,
                                      std::complex<double>*     bsr_val,
                                      const int*                bsr_row_ptr,
                                      const int*                bsr_col_ind,
                                      int                       bsr_dim,
                                      rocsparse_mat_info        info,
                                      rocsparse_solve_policy    policy,
                                      void*                     temp_buffer)
    {
        return rocsparse_zbsric0(handle,
                                 dir,
                                 mb,
                                 nnzb,
                                 descr,
                                 (rocsparse_double_complex*)bsr_val,
                                 bsr_row_ptr,
                                 bsr_col_ind,
                                 bsr_dim,
                                 info,
                                 policy,
                                 temp_buffer);
    }

    // rocsparse csrilu0 buffer size
    template <>
    rocsparse_status rocsparseTcsrilu0_buffer_size(rocsparse_handle          handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const rocsparse_mat_descr descr,
                                                   float*                    csr_val,
                                                   const int*                csr_row_ptr,
                                                   const int*                csr_col_ind,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size)
    {
        return rocsparse_scsrilu0_buffer_size(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsrilu0_buffer_size(rocsparse_handle          handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const rocsparse_mat_descr descr,
                                                   double*                   csr_val,
                                                   const int*                csr_row_ptr,
                                                   const int*                csr_col_ind,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size)
    {
        return rocsparse_dcsrilu0_buffer_size(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsrilu0_buffer_size(rocsparse_handle          handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const rocsparse_mat_descr descr,
                                                   std::complex<float>*      csr_val,
                                                   const int*                csr_row_ptr,
                                                   const int*                csr_col_ind,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size)
    {
        return rocsparse_ccsrilu0_buffer_size(handle,
                                              m,
                                              nnz,
                                              descr,
                                              (rocsparse_float_complex*)csr_val,
                                              csr_row_ptr,
                                              csr_col_ind,
                                              info,
                                              buffer_size);
    }

    template <>
    rocsparse_status rocsparseTcsrilu0_buffer_size(rocsparse_handle          handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const rocsparse_mat_descr descr,
                                                   std::complex<double>*     csr_val,
                                                   const int*                csr_row_ptr,
                                                   const int*                csr_col_ind,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size)
    {
        return rocsparse_zcsrilu0_buffer_size(handle,
                                              m,
                                              nnz,
                                              descr,
                                              (rocsparse_double_complex*)csr_val,
                                              csr_row_ptr,
                                              csr_col_ind,
                                              info,
                                              buffer_size);
    }

    // rocsparse csrilu0 analysis
    template <>
    rocsparse_status rocsparseTcsrilu0_analysis(rocsparse_handle          handle,
                                                int                       m,
                                                int                       nnz,
                                                const rocsparse_mat_descr descr,
                                                float*                    csr_val,
                                                const int*                csr_row_ptr,
                                                const int*                csr_col_ind,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                void*                     temp_buffer)
    {
        return rocsparse_scsrilu0_analysis(handle,
                                           m,
                                           nnz,
                                           descr,
                                           csr_val,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           info,
                                           analysis,
                                           solve,
                                           temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrilu0_analysis(rocsparse_handle          handle,
                                                int                       m,
                                                int                       nnz,
                                                const rocsparse_mat_descr descr,
                                                double*                   csr_val,
                                                const int*                csr_row_ptr,
                                                const int*                csr_col_ind,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                void*                     temp_buffer)
    {
        return rocsparse_dcsrilu0_analysis(handle,
                                           m,
                                           nnz,
                                           descr,
                                           csr_val,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           info,
                                           analysis,
                                           solve,
                                           temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrilu0_analysis(rocsparse_handle          handle,
                                                int                       m,
                                                int                       nnz,
                                                const rocsparse_mat_descr descr,
                                                std::complex<float>*      csr_val,
                                                const int*                csr_row_ptr,
                                                const int*                csr_col_ind,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                void*                     temp_buffer)
    {
        return rocsparse_ccsrilu0_analysis(handle,
                                           m,
                                           nnz,
                                           descr,
                                           (rocsparse_float_complex*)csr_val,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           info,
                                           analysis,
                                           solve,
                                           temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrilu0_analysis(rocsparse_handle          handle,
                                                int                       m,
                                                int                       nnz,
                                                const rocsparse_mat_descr descr,
                                                std::complex<double>*     csr_val,
                                                const int*                csr_row_ptr,
                                                const int*                csr_col_ind,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                void*                     temp_buffer)
    {
        return rocsparse_zcsrilu0_analysis(handle,
                                           m,
                                           nnz,
                                           descr,
                                           (rocsparse_double_complex*)csr_val,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           info,
                                           analysis,
                                           solve,
                                           temp_buffer);
    }

    // rocsparse csrilu0
    template <>
    rocsparse_status rocsparseTcsrilu0(rocsparse_handle          handle,
                                       int                       m,
                                       int                       nnz,
                                       const rocsparse_mat_descr descr,
                                       float*                    csr_val,
                                       const int*                csr_row_ptr,
                                       const int*                csr_col_ind,
                                       rocsparse_mat_info        info,
                                       rocsparse_solve_policy    policy,
                                       void*                     temp_buffer)
    {
        return rocsparse_scsrilu0(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrilu0(rocsparse_handle          handle,
                                       int                       m,
                                       int                       nnz,
                                       const rocsparse_mat_descr descr,
                                       double*                   csr_val,
                                       const int*                csr_row_ptr,
                                       const int*                csr_col_ind,
                                       rocsparse_mat_info        info,
                                       rocsparse_solve_policy    policy,
                                       void*                     temp_buffer)
    {
        return rocsparse_dcsrilu0(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrilu0(rocsparse_handle          handle,
                                       int                       m,
                                       int                       nnz,
                                       const rocsparse_mat_descr descr,
                                       std::complex<float>*      csr_val,
                                       const int*                csr_row_ptr,
                                       const int*                csr_col_ind,
                                       rocsparse_mat_info        info,
                                       rocsparse_solve_policy    policy,
                                       void*                     temp_buffer)
    {
        return rocsparse_ccsrilu0(handle,
                                  m,
                                  nnz,
                                  descr,
                                  (rocsparse_float_complex*)csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  info,
                                  policy,
                                  temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsrilu0(rocsparse_handle          handle,
                                       int                       m,
                                       int                       nnz,
                                       const rocsparse_mat_descr descr,
                                       std::complex<double>*     csr_val,
                                       const int*                csr_row_ptr,
                                       const int*                csr_col_ind,
                                       rocsparse_mat_info        info,
                                       rocsparse_solve_policy    policy,
                                       void*                     temp_buffer)
    {
        return rocsparse_zcsrilu0(handle,
                                  m,
                                  nnz,
                                  descr,
                                  (rocsparse_double_complex*)csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  info,
                                  policy,
                                  temp_buffer);
    }

    // rocsparse bsrilu0 buffer size
    template <>
    rocsparse_status rocsparseTbsrilu0_buffer_size(rocsparse_handle          handle,
                                                   rocsparse_direction       dir,
                                                   int                       mb,
                                                   int                       nnzb,
                                                   const rocsparse_mat_descr descr,
                                                   float*                    bsr_val,
                                                   const int*                bsr_row_ptr,
                                                   const int*                bsr_col_ind,
                                                   int                       bsr_dim,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size)
    {
        return rocsparse_sbsrilu0_buffer_size(handle,
                                              dir,
                                              mb,
                                              nnzb,
                                              descr,
                                              bsr_val,
                                              bsr_row_ptr,
                                              bsr_col_ind,
                                              bsr_dim,
                                              info,
                                              buffer_size);
    }

    template <>
    rocsparse_status rocsparseTbsrilu0_buffer_size(rocsparse_handle          handle,
                                                   rocsparse_direction       dir,
                                                   int                       mb,
                                                   int                       nnzb,
                                                   const rocsparse_mat_descr descr,
                                                   double*                   bsr_val,
                                                   const int*                bsr_row_ptr,
                                                   const int*                bsr_col_ind,
                                                   int                       bsr_dim,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size)
    {
        return rocsparse_dbsrilu0_buffer_size(handle,
                                              dir,
                                              mb,
                                              nnzb,
                                              descr,
                                              bsr_val,
                                              bsr_row_ptr,
                                              bsr_col_ind,
                                              bsr_dim,
                                              info,
                                              buffer_size);
    }

    template <>
    rocsparse_status rocsparseTbsrilu0_buffer_size(rocsparse_handle          handle,
                                                   rocsparse_direction       dir,
                                                   int                       mb,
                                                   int                       nnzb,
                                                   const rocsparse_mat_descr descr,
                                                   std::complex<float>*      bsr_val,
                                                   const int*                bsr_row_ptr,
                                                   const int*                bsr_col_ind,
                                                   int                       bsr_dim,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size)
    {
        return rocsparse_cbsrilu0_buffer_size(handle,
                                              dir,
                                              mb,
                                              nnzb,
                                              descr,
                                              (rocsparse_float_complex*)bsr_val,
                                              bsr_row_ptr,
                                              bsr_col_ind,
                                              bsr_dim,
                                              info,
                                              buffer_size);
    }

    template <>
    rocsparse_status rocsparseTbsrilu0_buffer_size(rocsparse_handle          handle,
                                                   rocsparse_direction       dir,
                                                   int                       mb,
                                                   int                       nnzb,
                                                   const rocsparse_mat_descr descr,
                                                   std::complex<double>*     bsr_val,
                                                   const int*                bsr_row_ptr,
                                                   const int*                bsr_col_ind,
                                                   int                       bsr_dim,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size)
    {
        return rocsparse_zbsrilu0_buffer_size(handle,
                                              dir,
                                              mb,
                                              nnzb,
                                              descr,
                                              (rocsparse_double_complex*)bsr_val,
                                              bsr_row_ptr,
                                              bsr_col_ind,
                                              bsr_dim,
                                              info,
                                              buffer_size);
    }

    // rocsparse bsrilu0 analysis
    template <>
    rocsparse_status rocsparseTbsrilu0_analysis(rocsparse_handle          handle,
                                                rocsparse_direction       dir,
                                                int                       mb,
                                                int                       nnzb,
                                                const rocsparse_mat_descr descr,
                                                float*                    bsr_val,
                                                const int*                bsr_row_ptr,
                                                const int*                bsr_col_ind,
                                                int                       bsr_dim,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                void*                     temp_buffer)
    {
        return rocsparse_sbsrilu0_analysis(handle,
                                           dir,
                                           mb,
                                           nnzb,
                                           descr,
                                           bsr_val,
                                           bsr_row_ptr,
                                           bsr_col_ind,
                                           bsr_dim,
                                           info,
                                           analysis,
                                           solve,
                                           temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTbsrilu0_analysis(rocsparse_handle          handle,
                                                rocsparse_direction       dir,
                                                int                       mb,
                                                int                       nnzb,
                                                const rocsparse_mat_descr descr,
                                                double*                   bsr_val,
                                                const int*                bsr_row_ptr,
                                                const int*                bsr_col_ind,
                                                int                       bsr_dim,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                void*                     temp_buffer)
    {
        return rocsparse_dbsrilu0_analysis(handle,
                                           dir,
                                           mb,
                                           nnzb,
                                           descr,
                                           bsr_val,
                                           bsr_row_ptr,
                                           bsr_col_ind,
                                           bsr_dim,
                                           info,
                                           analysis,
                                           solve,
                                           temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTbsrilu0_analysis(rocsparse_handle          handle,
                                                rocsparse_direction       dir,
                                                int                       mb,
                                                int                       nnzb,
                                                const rocsparse_mat_descr descr,
                                                std::complex<float>*      bsr_val,
                                                const int*                bsr_row_ptr,
                                                const int*                bsr_col_ind,
                                                int                       bsr_dim,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                void*                     temp_buffer)
    {
        return rocsparse_cbsrilu0_analysis(handle,
                                           dir,
                                           mb,
                                           nnzb,
                                           descr,
                                           (rocsparse_float_complex*)bsr_val,
                                           bsr_row_ptr,
                                           bsr_col_ind,
                                           bsr_dim,
                                           info,
                                           analysis,
                                           solve,
                                           temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTbsrilu0_analysis(rocsparse_handle          handle,
                                                rocsparse_direction       dir,
                                                int                       mb,
                                                int                       nnzb,
                                                const rocsparse_mat_descr descr,
                                                std::complex<double>*     bsr_val,
                                                const int*                bsr_row_ptr,
                                                const int*                bsr_col_ind,
                                                int                       bsr_dim,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                void*                     temp_buffer)
    {
        return rocsparse_zbsrilu0_analysis(handle,
                                           dir,
                                           mb,
                                           nnzb,
                                           descr,
                                           (rocsparse_double_complex*)bsr_val,
                                           bsr_row_ptr,
                                           bsr_col_ind,
                                           bsr_dim,
                                           info,
                                           analysis,
                                           solve,
                                           temp_buffer);
    }

    // rocsparse bsrilu0
    template <>
    rocsparse_status rocsparseTbsrilu0(rocsparse_handle          handle,
                                       rocsparse_direction       dir,
                                       int                       mb,
                                       int                       nnzb,
                                       const rocsparse_mat_descr descr,
                                       float*                    bsr_val,
                                       const int*                bsr_row_ptr,
                                       const int*                bsr_col_ind,
                                       int                       bsr_dim,
                                       rocsparse_mat_info        info,
                                       rocsparse_solve_policy    policy,
                                       void*                     temp_buffer)
    {
        return rocsparse_sbsrilu0(handle,
                                  dir,
                                  mb,
                                  nnzb,
                                  descr,
                                  bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind,
                                  bsr_dim,
                                  info,
                                  policy,
                                  temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTbsrilu0(rocsparse_handle          handle,
                                       rocsparse_direction       dir,
                                       int                       mb,
                                       int                       nnzb,
                                       const rocsparse_mat_descr descr,
                                       double*                   bsr_val,
                                       const int*                bsr_row_ptr,
                                       const int*                bsr_col_ind,
                                       int                       bsr_dim,
                                       rocsparse_mat_info        info,
                                       rocsparse_solve_policy    policy,
                                       void*                     temp_buffer)
    {
        return rocsparse_dbsrilu0(handle,
                                  dir,
                                  mb,
                                  nnzb,
                                  descr,
                                  bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind,
                                  bsr_dim,
                                  info,
                                  policy,
                                  temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTbsrilu0(rocsparse_handle          handle,
                                       rocsparse_direction       dir,
                                       int                       mb,
                                       int                       nnzb,
                                       const rocsparse_mat_descr descr,
                                       std::complex<float>*      bsr_val,
                                       const int*                bsr_row_ptr,
                                       const int*                bsr_col_ind,
                                       int                       bsr_dim,
                                       rocsparse_mat_info        info,
                                       rocsparse_solve_policy    policy,
                                       void*                     temp_buffer)
    {
        return rocsparse_cbsrilu0(handle,
                                  dir,
                                  mb,
                                  nnzb,
                                  descr,
                                  (rocsparse_float_complex*)bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind,
                                  bsr_dim,
                                  info,
                                  policy,
                                  temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTbsrilu0(rocsparse_handle          handle,
                                       rocsparse_direction       dir,
                                       int                       mb,
                                       int                       nnzb,
                                       const rocsparse_mat_descr descr,
                                       std::complex<double>*     bsr_val,
                                       const int*                bsr_row_ptr,
                                       const int*                bsr_col_ind,
                                       int                       bsr_dim,
                                       rocsparse_mat_info        info,
                                       rocsparse_solve_policy    policy,
                                       void*                     temp_buffer)
    {
        return rocsparse_zbsrilu0(handle,
                                  dir,
                                  mb,
                                  nnzb,
                                  descr,
                                  (rocsparse_double_complex*)bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind,
                                  bsr_dim,
                                  info,
                                  policy,
                                  temp_buffer);
    }

    // rocsparse_csr2bsr
    template <>
    rocsparse_status rocsparseTcsr2bsr(rocsparse_handle          handle,
                                       rocsparse_direction       dir,
                                       int                       m,
                                       int                       n,
                                       const rocsparse_mat_descr csr_descr,
                                       const float*              csr_val,
                                       const int*                csr_row_ptr,
                                       const int*                csr_col_ind,
                                       int                       block_dim,
                                       const rocsparse_mat_descr bsr_descr,
                                       float*                    bsr_val,
                                       int*                      bsr_row_ptr,
                                       int*                      bsr_col_ind)
    {
        return rocsparse_scsr2bsr(handle,
                                  dir,
                                  m,
                                  n,
                                  csr_descr,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  block_dim,
                                  bsr_descr,
                                  bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind);
    }

    template <>
    rocsparse_status rocsparseTcsr2bsr(rocsparse_handle          handle,
                                       rocsparse_direction       dir,
                                       int                       m,
                                       int                       n,
                                       const rocsparse_mat_descr csr_descr,
                                       const double*             csr_val,
                                       const int*                csr_row_ptr,
                                       const int*                csr_col_ind,
                                       int                       block_dim,
                                       const rocsparse_mat_descr bsr_descr,
                                       double*                   bsr_val,
                                       int*                      bsr_row_ptr,
                                       int*                      bsr_col_ind)
    {
        return rocsparse_dcsr2bsr(handle,
                                  dir,
                                  m,
                                  n,
                                  csr_descr,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  block_dim,
                                  bsr_descr,
                                  bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind);
    }

    template <>
    rocsparse_status rocsparseTcsr2bsr(rocsparse_handle           handle,
                                       rocsparse_direction        dir,
                                       int                        m,
                                       int                        n,
                                       const rocsparse_mat_descr  csr_descr,
                                       const std::complex<float>* csr_val,
                                       const int*                 csr_row_ptr,
                                       const int*                 csr_col_ind,
                                       int                        block_dim,
                                       const rocsparse_mat_descr  bsr_descr,
                                       std::complex<float>*       bsr_val,
                                       int*                       bsr_row_ptr,
                                       int*                       bsr_col_ind)
    {
        return rocsparse_ccsr2bsr(handle,
                                  dir,
                                  m,
                                  n,
                                  csr_descr,
                                  (const rocsparse_float_complex*)csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  block_dim,
                                  bsr_descr,
                                  (rocsparse_float_complex*)bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind);
    }

    template <>
    rocsparse_status rocsparseTcsr2bsr(rocsparse_handle            handle,
                                       rocsparse_direction         dir,
                                       int                         m,
                                       int                         n,
                                       const rocsparse_mat_descr   csr_descr,
                                       const std::complex<double>* csr_val,
                                       const int*                  csr_row_ptr,
                                       const int*                  csr_col_ind,
                                       int                         block_dim,
                                       const rocsparse_mat_descr   bsr_descr,
                                       std::complex<double>*       bsr_val,
                                       int*                        bsr_row_ptr,
                                       int*                        bsr_col_ind)
    {
        return rocsparse_zcsr2bsr(handle,
                                  dir,
                                  m,
                                  n,
                                  csr_descr,
                                  (const rocsparse_double_complex*)csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  block_dim,
                                  bsr_descr,
                                  (rocsparse_double_complex*)bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind);
    }

    // rocsparse_bsr2csr
    template <>
    rocsparse_status rocsparseTbsr2csr(rocsparse_handle          handle,
                                       rocsparse_direction       dir,
                                       int                       mb,
                                       int                       nb,
                                       const rocsparse_mat_descr bsr_descr,
                                       const float*              bsr_val,
                                       const int*                bsr_row_ptr,
                                       const int*                bsr_col_ind,
                                       int                       block_dim,
                                       const rocsparse_mat_descr csr_descr,
                                       float*                    csr_val,
                                       int*                      csr_row_ptr,
                                       int*                      csr_col_ind)
    {
        return rocsparse_sbsr2csr(handle,
                                  dir,
                                  mb,
                                  nb,
                                  bsr_descr,
                                  bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind,
                                  block_dim,
                                  csr_descr,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind);
    }

    template <>
    rocsparse_status rocsparseTbsr2csr(rocsparse_handle          handle,
                                       rocsparse_direction       dir,
                                       int                       mb,
                                       int                       nb,
                                       const rocsparse_mat_descr bsr_descr,
                                       const double*             bsr_val,
                                       const int*                bsr_row_ptr,
                                       const int*                bsr_col_ind,
                                       int                       block_dim,
                                       const rocsparse_mat_descr csr_descr,
                                       double*                   csr_val,
                                       int*                      csr_row_ptr,
                                       int*                      csr_col_ind)
    {
        return rocsparse_dbsr2csr(handle,
                                  dir,
                                  mb,
                                  nb,
                                  bsr_descr,
                                  bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind,
                                  block_dim,
                                  csr_descr,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind);
    }

    template <>
    rocsparse_status rocsparseTbsr2csr(rocsparse_handle           handle,
                                       rocsparse_direction        dir,
                                       int                        mb,
                                       int                        nb,
                                       const rocsparse_mat_descr  bsr_descr,
                                       const std::complex<float>* bsr_val,
                                       const int*                 bsr_row_ptr,
                                       const int*                 bsr_col_ind,
                                       int                        block_dim,
                                       const rocsparse_mat_descr  csr_descr,
                                       std::complex<float>*       csr_val,
                                       int*                       csr_row_ptr,
                                       int*                       csr_col_ind)
    {
        return rocsparse_cbsr2csr(handle,
                                  dir,
                                  mb,
                                  nb,
                                  bsr_descr,
                                  (const rocsparse_float_complex*)bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind,
                                  block_dim,
                                  csr_descr,
                                  (rocsparse_float_complex*)csr_val,
                                  csr_row_ptr,
                                  csr_col_ind);
    }

    template <>
    rocsparse_status rocsparseTbsr2csr(rocsparse_handle            handle,
                                       rocsparse_direction         dir,
                                       int                         mb,
                                       int                         nb,
                                       const rocsparse_mat_descr   bsr_descr,
                                       const std::complex<double>* bsr_val,
                                       const int*                  bsr_row_ptr,
                                       const int*                  bsr_col_ind,
                                       int                         block_dim,
                                       const rocsparse_mat_descr   csr_descr,
                                       std::complex<double>*       csr_val,
                                       int*                        csr_row_ptr,
                                       int*                        csr_col_ind)
    {
        return rocsparse_zbsr2csr(handle,
                                  dir,
                                  mb,
                                  nb,
                                  bsr_descr,
                                  (const rocsparse_double_complex*)bsr_val,
                                  bsr_row_ptr,
                                  bsr_col_ind,
                                  block_dim,
                                  csr_descr,
                                  (rocsparse_double_complex*)csr_val,
                                  csr_row_ptr,
                                  csr_col_ind);
    }

    // rocsparse csr2csc
    template <>
    rocsparse_status rocsparseTcsr2csc(rocsparse_handle     handle,
                                       int                  m,
                                       int                  n,
                                       int                  nnz,
                                       const float*         csr_val,
                                       const int*           csr_row_ptr,
                                       const int*           csr_col_ind,
                                       float*               csc_val,
                                       int*                 csc_row_ind,
                                       int*                 csc_col_ptr,
                                       rocsparse_action     copy_values,
                                       rocsparse_index_base idx_base,
                                       void*                temp_buffer)
    {
        return rocsparse_scsr2csc(handle,
                                  m,
                                  n,
                                  nnz,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  csc_val,
                                  csc_row_ind,
                                  csc_col_ptr,
                                  copy_values,
                                  idx_base,
                                  temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsr2csc(rocsparse_handle     handle,
                                       int                  m,
                                       int                  n,
                                       int                  nnz,
                                       const double*        csr_val,
                                       const int*           csr_row_ptr,
                                       const int*           csr_col_ind,
                                       double*              csc_val,
                                       int*                 csc_row_ind,
                                       int*                 csc_col_ptr,
                                       rocsparse_action     copy_values,
                                       rocsparse_index_base idx_base,
                                       void*                temp_buffer)
    {
        return rocsparse_dcsr2csc(handle,
                                  m,
                                  n,
                                  nnz,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  csc_val,
                                  csc_row_ind,
                                  csc_col_ptr,
                                  copy_values,
                                  idx_base,
                                  temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsr2csc(rocsparse_handle           handle,
                                       int                        m,
                                       int                        n,
                                       int                        nnz,
                                       const std::complex<float>* csr_val,
                                       const int*                 csr_row_ptr,
                                       const int*                 csr_col_ind,
                                       std::complex<float>*       csc_val,
                                       int*                       csc_row_ind,
                                       int*                       csc_col_ptr,
                                       rocsparse_action           copy_values,
                                       rocsparse_index_base       idx_base,
                                       void*                      temp_buffer)
    {
        return rocsparse_ccsr2csc(handle,
                                  m,
                                  n,
                                  nnz,
                                  (const rocsparse_float_complex*)csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  (rocsparse_float_complex*)csc_val,
                                  csc_row_ind,
                                  csc_col_ptr,
                                  copy_values,
                                  idx_base,
                                  temp_buffer);
    }

    template <>
    rocsparse_status rocsparseTcsr2csc(rocsparse_handle            handle,
                                       int                         m,
                                       int                         n,
                                       int                         nnz,
                                       const std::complex<double>* csr_val,
                                       const int*                  csr_row_ptr,
                                       const int*                  csr_col_ind,
                                       std::complex<double>*       csc_val,
                                       int*                        csc_row_ind,
                                       int*                        csc_col_ptr,
                                       rocsparse_action            copy_values,
                                       rocsparse_index_base        idx_base,
                                       void*                       temp_buffer)
    {
        return rocsparse_zcsr2csc(handle,
                                  m,
                                  n,
                                  nnz,
                                  (const rocsparse_double_complex*)csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  (rocsparse_double_complex*)csc_val,
                                  csc_row_ind,
                                  csc_col_ptr,
                                  copy_values,
                                  idx_base,
                                  temp_buffer);
    }

    // rocsparse csr2ell
    template <>
    rocsparse_status rocsparseTcsr2ell(rocsparse_handle          handle,
                                       int                       m,
                                       const rocsparse_mat_descr csr_descr,
                                       const float*              csr_val,
                                       const int*                csr_row_ptr,
                                       const int*                csr_col_ind,
                                       const rocsparse_mat_descr ell_descr,
                                       int                       ell_width,
                                       float*                    ell_val,
                                       int*                      ell_col_ind)
    {
        return rocsparse_scsr2ell(handle,
                                  m,
                                  csr_descr,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  ell_descr,
                                  ell_width,
                                  ell_val,
                                  ell_col_ind);
    }

    template <>
    rocsparse_status rocsparseTcsr2ell(rocsparse_handle          handle,
                                       int                       m,
                                       const rocsparse_mat_descr csr_descr,
                                       const double*             csr_val,
                                       const int*                csr_row_ptr,
                                       const int*                csr_col_ind,
                                       const rocsparse_mat_descr ell_descr,
                                       int                       ell_width,
                                       double*                   ell_val,
                                       int*                      ell_col_ind)
    {
        return rocsparse_dcsr2ell(handle,
                                  m,
                                  csr_descr,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  ell_descr,
                                  ell_width,
                                  ell_val,
                                  ell_col_ind);
    }

    template <>
    rocsparse_status rocsparseTcsr2ell(rocsparse_handle           handle,
                                       int                        m,
                                       const rocsparse_mat_descr  csr_descr,
                                       const std::complex<float>* csr_val,
                                       const int*                 csr_row_ptr,
                                       const int*                 csr_col_ind,
                                       const rocsparse_mat_descr  ell_descr,
                                       int                        ell_width,
                                       std::complex<float>*       ell_val,
                                       int*                       ell_col_ind)
    {
        return rocsparse_ccsr2ell(handle,
                                  m,
                                  csr_descr,
                                  (const rocsparse_float_complex*)csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  ell_descr,
                                  ell_width,
                                  (rocsparse_float_complex*)ell_val,
                                  ell_col_ind);
    }

    template <>
    rocsparse_status rocsparseTcsr2ell(rocsparse_handle            handle,
                                       int                         m,
                                       const rocsparse_mat_descr   csr_descr,
                                       const std::complex<double>* csr_val,
                                       const int*                  csr_row_ptr,
                                       const int*                  csr_col_ind,
                                       const rocsparse_mat_descr   ell_descr,
                                       int                         ell_width,
                                       std::complex<double>*       ell_val,
                                       int*                        ell_col_ind)
    {
        return rocsparse_zcsr2ell(handle,
                                  m,
                                  csr_descr,
                                  (const rocsparse_double_complex*)csr_val,
                                  csr_row_ptr,
                                  csr_col_ind,
                                  ell_descr,
                                  ell_width,
                                  (rocsparse_double_complex*)ell_val,
                                  ell_col_ind);
    }

    // rocsparse ell2csr
    template <>
    rocsparse_status rocsparseTell2csr(rocsparse_handle          handle,
                                       int                       m,
                                       int                       n,
                                       const rocsparse_mat_descr ell_descr,
                                       int                       ell_width,
                                       const float*              ell_val,
                                       const int*                ell_col_ind,
                                       const rocsparse_mat_descr csr_descr,
                                       float*                    csr_val,
                                       const int*                csr_row_ptr,
                                       int*                      csr_col_ind)
    {
        return rocsparse_sell2csr(handle,
                                  m,
                                  n,
                                  ell_descr,
                                  ell_width,
                                  ell_val,
                                  ell_col_ind,
                                  csr_descr,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind);
    }

    template <>
    rocsparse_status rocsparseTell2csr(rocsparse_handle          handle,
                                       int                       m,
                                       int                       n,
                                       const rocsparse_mat_descr ell_descr,
                                       int                       ell_width,
                                       const double*             ell_val,
                                       const int*                ell_col_ind,
                                       const rocsparse_mat_descr csr_descr,
                                       double*                   csr_val,
                                       const int*                csr_row_ptr,
                                       int*                      csr_col_ind)
    {
        return rocsparse_dell2csr(handle,
                                  m,
                                  n,
                                  ell_descr,
                                  ell_width,
                                  ell_val,
                                  ell_col_ind,
                                  csr_descr,
                                  csr_val,
                                  csr_row_ptr,
                                  csr_col_ind);
    }

    template <>
    rocsparse_status rocsparseTell2csr(rocsparse_handle           handle,
                                       int                        m,
                                       int                        n,
                                       const rocsparse_mat_descr  ell_descr,
                                       int                        ell_width,
                                       const std::complex<float>* ell_val,
                                       const int*                 ell_col_ind,
                                       const rocsparse_mat_descr  csr_descr,
                                       std::complex<float>*       csr_val,
                                       const int*                 csr_row_ptr,
                                       int*                       csr_col_ind)
    {
        return rocsparse_cell2csr(handle,
                                  m,
                                  n,
                                  ell_descr,
                                  ell_width,
                                  (const rocsparse_float_complex*)ell_val,
                                  ell_col_ind,
                                  csr_descr,
                                  (rocsparse_float_complex*)csr_val,
                                  csr_row_ptr,
                                  csr_col_ind);
    }

    template <>
    rocsparse_status rocsparseTell2csr(rocsparse_handle            handle,
                                       int                         m,
                                       int                         n,
                                       const rocsparse_mat_descr   ell_descr,
                                       int                         ell_width,
                                       const std::complex<double>* ell_val,
                                       const int*                  ell_col_ind,
                                       const rocsparse_mat_descr   csr_descr,
                                       std::complex<double>*       csr_val,
                                       const int*                  csr_row_ptr,
                                       int*                        csr_col_ind)
    {
        return rocsparse_zell2csr(handle,
                                  m,
                                  n,
                                  ell_descr,
                                  ell_width,
                                  (const rocsparse_double_complex*)ell_val,
                                  ell_col_ind,
                                  csr_descr,
                                  (rocsparse_double_complex*)csr_val,
                                  csr_row_ptr,
                                  csr_col_ind);
    }

    // rocsparse csr2dense
    template <>
    rocsparse_status rocsparseTcsr2dense(rocsparse_handle          handle,
                                         int                       m,
                                         int                       n,
                                         const rocsparse_mat_descr csr_descr,
                                         const float*              csr_val,
                                         const int*                csr_row_ptr,
                                         const int*                csr_col_ind,
                                         float*                    A,
                                         int                       ld)
    {
        return rocsparse_scsr2dense(
            handle, m, n, csr_descr, csr_val, csr_row_ptr, csr_col_ind, A, ld);
    }

    template <>
    rocsparse_status rocsparseTcsr2dense(rocsparse_handle          handle,
                                         int                       m,
                                         int                       n,
                                         const rocsparse_mat_descr csr_descr,
                                         const double*             csr_val,
                                         const int*                csr_row_ptr,
                                         const int*                csr_col_ind,
                                         double*                   A,
                                         int                       ld)
    {
        return rocsparse_dcsr2dense(
            handle, m, n, csr_descr, csr_val, csr_row_ptr, csr_col_ind, A, ld);
    }

    template <>
    rocsparse_status rocsparseTcsr2dense(rocsparse_handle           handle,
                                         int                        m,
                                         int                        n,
                                         const rocsparse_mat_descr  csr_descr,
                                         const std::complex<float>* csr_val,
                                         const int*                 csr_row_ptr,
                                         const int*                 csr_col_ind,
                                         std::complex<float>*       A,
                                         int                        ld)
    {
        return rocsparse_ccsr2dense(handle,
                                    m,
                                    n,
                                    csr_descr,
                                    (const rocsparse_float_complex*)csr_val,
                                    csr_row_ptr,
                                    csr_col_ind,
                                    (rocsparse_float_complex*)A,
                                    ld);
    }

    template <>
    rocsparse_status rocsparseTcsr2dense(rocsparse_handle            handle,
                                         int                         m,
                                         int                         n,
                                         const rocsparse_mat_descr   csr_descr,
                                         const std::complex<double>* csr_val,
                                         const int*                  csr_row_ptr,
                                         const int*                  csr_col_ind,
                                         std::complex<double>*       A,
                                         int                         ld)
    {
        return rocsparse_zcsr2dense(handle,
                                    m,
                                    n,
                                    csr_descr,
                                    (const rocsparse_double_complex*)csr_val,
                                    csr_row_ptr,
                                    csr_col_ind,
                                    (rocsparse_double_complex*)A,
                                    ld);
    }

    // rocsparse dense2csr
    template <>
    rocsparse_status rocsparseTdense2csr(rocsparse_handle          handle,
                                         int                       m,
                                         int                       n,
                                         const rocsparse_mat_descr descr_A,
                                         const float*              A,
                                         int                       lda,
                                         const int*                nnz_per_row,
                                         float*                    csr_val,
                                         int*                      csr_row_ptr,
                                         int*                      csr_col_ind)
    {
        return rocsparse_sdense2csr(
            handle, m, n, descr_A, A, lda, nnz_per_row, csr_val, csr_row_ptr, csr_col_ind);
    }

    template <>
    rocsparse_status rocsparseTdense2csr(rocsparse_handle          handle,
                                         int                       m,
                                         int                       n,
                                         const rocsparse_mat_descr descr_A,
                                         const double*             A,
                                         int                       lda,
                                         const int*                nnz_per_row,
                                         double*                   csr_val,
                                         int*                      csr_row_ptr,
                                         int*                      csr_col_ind)
    {
        return rocsparse_ddense2csr(
            handle, m, n, descr_A, A, lda, nnz_per_row, csr_val, csr_row_ptr, csr_col_ind);
    }

    template <>
    rocsparse_status rocsparseTdense2csr(rocsparse_handle           handle,
                                         int                        m,
                                         int                        n,
                                         const rocsparse_mat_descr  descr_A,
                                         const std::complex<float>* A,
                                         int                        lda,
                                         const int*                 nnz_per_row,
                                         std::complex<float>*       csr_val,
                                         int*                       csr_row_ptr,
                                         int*                       csr_col_ind)
    {
        return rocsparse_cdense2csr(handle,
                                    m,
                                    n,
                                    descr_A,
                                    (const rocsparse_float_complex*)A,
                                    lda,
                                    nnz_per_row,
                                    (rocsparse_float_complex*)csr_val,
                                    csr_row_ptr,
                                    csr_col_ind);
    }

    template <>
    rocsparse_status rocsparseTdense2csr(rocsparse_handle            handle,
                                         int                         m,
                                         int                         n,
                                         const rocsparse_mat_descr   descr_A,
                                         const std::complex<double>* A,
                                         int                         lda,
                                         const int*                  nnz_per_row,
                                         std::complex<double>*       csr_val,
                                         int*                        csr_row_ptr,
                                         int*                        csr_col_ind)
    {
        return rocsparse_zdense2csr(handle,
                                    m,
                                    n,
                                    descr_A,
                                    (const rocsparse_double_complex*)A,
                                    lda,
                                    nnz_per_row,
                                    (rocsparse_double_complex*)csr_val,
                                    csr_row_ptr,
                                    csr_col_ind);
    }

    // rocsparse nnz
    template <>
    rocsparse_status rocsparseTnnz(rocsparse_handle          handle,
                                   rocsparse_direction       dir_A,
                                   int                       m,
                                   int                       n,
                                   const rocsparse_mat_descr descr_A,
                                   const float*              A,
                                   int                       lda,
                                   int*                      nnz_per_row_column,
                                   int*                      nnz_total)
    {
        return rocsparse_snnz(handle, dir_A, m, n, descr_A, A, lda, nnz_per_row_column, nnz_total);
    }

    template <>
    rocsparse_status rocsparseTnnz(rocsparse_handle          handle,
                                   rocsparse_direction       dir_A,
                                   int                       m,
                                   int                       n,
                                   const rocsparse_mat_descr descr_A,
                                   const double*             A,
                                   int                       lda,
                                   int*                      nnz_per_row_column,
                                   int*                      nnz_total)
    {
        return rocsparse_dnnz(handle, dir_A, m, n, descr_A, A, lda, nnz_per_row_column, nnz_total);
    }

    template <>
    rocsparse_status rocsparseTnnz(rocsparse_handle           handle,
                                   rocsparse_direction        dir_A,
                                   int                        m,
                                   int                        n,
                                   const rocsparse_mat_descr  descr_A,
                                   const std::complex<float>* A,
                                   int                        lda,
                                   int*                       nnz_per_row_column,
                                   int*                       nnz_total)
    {
        return rocsparse_cnnz(handle,
                              dir_A,
                              m,
                              n,
                              descr_A,
                              (const rocsparse_float_complex*)A,
                              lda,
                              nnz_per_row_column,
                              nnz_total);
    }

    template <>
    rocsparse_status rocsparseTnnz(rocsparse_handle            handle,
                                   rocsparse_direction         dir_A,
                                   int                         m,
                                   int                         n,
                                   const rocsparse_mat_descr   descr_A,
                                   const std::complex<double>* A,
                                   int                         lda,
                                   int*                        nnz_per_row_column,
                                   int*                        nnz_total)
    {
        return rocsparse_znnz(handle,
                              dir_A,
                              m,
                              n,
                              descr_A,
                              (const rocsparse_double_complex*)A,
                              lda,
                              nnz_per_row_column,
                              nnz_total);
    }

    template <>
    rocsparse_status rocsparseTgthr(rocsparse_handle     handle,
                                    int                  nnz,
                                    float*               y,
                                    float*               x_val,
                                    int*                 x_ind,
                                    rocsparse_index_base idx_base)
    {
        return rocsparse_sgthr(handle, nnz, y, x_val, x_ind, idx_base);
    }

    template <>
    rocsparse_status rocsparseTgthr(rocsparse_handle     handle,
                                    int                  nnz,
                                    double*              y,
                                    double*              x_val,
                                    int*                 x_ind,
                                    rocsparse_index_base idx_base)
    {
        return rocsparse_dgthr(handle, nnz, y, x_val, x_ind, idx_base);
    }

    template <>
    rocsparse_status rocsparseTgthr(rocsparse_handle     handle,
                                    int                  nnz,
                                    std::complex<float>* y,
                                    std::complex<float>* x_val,
                                    int*                 x_ind,
                                    rocsparse_index_base idx_base)
    {
        return rocsparse_cgthr(handle,
                               nnz,
                               (rocsparse_float_complex*)y,
                               (rocsparse_float_complex*)x_val,
                               x_ind,
                               idx_base);
    }

    template <>
    rocsparse_status rocsparseTgthr(rocsparse_handle      handle,
                                    int                   nnz,
                                    std::complex<double>* y,
                                    std::complex<double>* x_val,
                                    int*                  x_ind,
                                    rocsparse_index_base  idx_base)
    {
        return rocsparse_zgthr(handle,
                               nnz,
                               (rocsparse_double_complex*)y,
                               (rocsparse_double_complex*)x_val,
                               x_ind,
                               idx_base);
    }

    // rocsparse nnz compress
    template <>
    rocsparse_status rocsparseTnnz_compress(rocsparse_handle          handle,
                                            rocsparse_int             m,
                                            const rocsparse_mat_descr descr_A,
                                            const float*              csr_val_A,
                                            const rocsparse_int*      csr_row_ptr_A,
                                            rocsparse_int*            nnz_per_row,
                                            rocsparse_int*            nnz_C,
                                            float                     tol)
    {
        return rocsparse_snnz_compress(
            handle, m, descr_A, csr_val_A, csr_row_ptr_A, nnz_per_row, nnz_C, tol);
    }

    template <>
    rocsparse_status rocsparseTnnz_compress(rocsparse_handle          handle,
                                            rocsparse_int             m,
                                            const rocsparse_mat_descr descr_A,
                                            const double*             csr_val_A,
                                            const rocsparse_int*      csr_row_ptr_A,
                                            rocsparse_int*            nnz_per_row,
                                            rocsparse_int*            nnz_C,
                                            double                    tol)
    {
        return rocsparse_dnnz_compress(
            handle, m, descr_A, csr_val_A, csr_row_ptr_A, nnz_per_row, nnz_C, tol);
    }

    template <>
    rocsparse_status rocsparseTnnz_compress(rocsparse_handle           handle,
                                            rocsparse_int              m,
                                            const rocsparse_mat_descr  descr_A,
                                            const std::complex<float>* csr_val_A,
                                            const rocsparse_int*       csr_row_ptr_A,
                                            rocsparse_int*             nnz_per_row,
                                            rocsparse_int*             nnz_C,
                                            std::complex<float>        tol)
    {
        return rocsparse_cnnz_compress(handle,
                                       m,
                                       descr_A,
                                       (const rocsparse_float_complex*)csr_val_A,
                                       csr_row_ptr_A,
                                       nnz_per_row,
                                       nnz_C,
                                       rocsparse_float_complex(std::real(tol), std::imag(tol)));
    }

    template <>
    rocsparse_status rocsparseTnnz_compress(rocsparse_handle            handle,
                                            rocsparse_int               m,
                                            const rocsparse_mat_descr   descr_A,
                                            const std::complex<double>* csr_val_A,
                                            const rocsparse_int*        csr_row_ptr_A,
                                            rocsparse_int*              nnz_per_row,
                                            rocsparse_int*              nnz_C,
                                            std::complex<double>        tol)
    {
        return rocsparse_znnz_compress(handle,
                                       m,
                                       descr_A,
                                       (const rocsparse_double_complex*)csr_val_A,
                                       csr_row_ptr_A,
                                       nnz_per_row,
                                       nnz_C,
                                       rocsparse_double_complex(std::real(tol), std::imag(tol)));
    }

    // rocsparse csr2csr compress
    template <>
    rocsparse_status rocsparseTcsr2csr_compress(rocsparse_handle          handle,
                                                rocsparse_int             m,
                                                rocsparse_int             n,
                                                const rocsparse_mat_descr descr_A,
                                                const float*              csr_val_A,
                                                const rocsparse_int*      csr_row_ptr_A,
                                                const rocsparse_int*      csr_col_ind_A,
                                                rocsparse_int             nnz_A,
                                                const rocsparse_int*      nnz_per_row,
                                                float*                    csr_val_C,
                                                rocsparse_int*            csr_row_ptr_C,
                                                rocsparse_int*            csr_col_ind_C,
                                                float                     tol)
    {
        return rocsparse_scsr2csr_compress(handle,
                                           m,
                                           n,
                                           descr_A,
                                           csr_val_A,
                                           csr_row_ptr_A,
                                           csr_col_ind_A,
                                           nnz_A,
                                           nnz_per_row,
                                           csr_val_C,
                                           csr_row_ptr_C,
                                           csr_col_ind_C,
                                           tol);
    }

    template <>
    rocsparse_status rocsparseTcsr2csr_compress(rocsparse_handle          handle,
                                                rocsparse_int             m,
                                                rocsparse_int             n,
                                                const rocsparse_mat_descr descr_A,
                                                const double*             csr_val_A,
                                                const rocsparse_int*      csr_row_ptr_A,
                                                const rocsparse_int*      csr_col_ind_A,
                                                rocsparse_int             nnz_A,
                                                const rocsparse_int*      nnz_per_row,
                                                double*                   csr_val_C,
                                                rocsparse_int*            csr_row_ptr_C,
                                                rocsparse_int*            csr_col_ind_C,
                                                double                    tol)
    {
        return rocsparse_dcsr2csr_compress(handle,
                                           m,
                                           n,
                                           descr_A,
                                           csr_val_A,
                                           csr_row_ptr_A,
                                           csr_col_ind_A,
                                           nnz_A,
                                           nnz_per_row,
                                           csr_val_C,
                                           csr_row_ptr_C,
                                           csr_col_ind_C,
                                           tol);
    }

    template <>
    rocsparse_status rocsparseTcsr2csr_compress(rocsparse_handle           handle,
                                                rocsparse_int              m,
                                                rocsparse_int              n,
                                                const rocsparse_mat_descr  descr_A,
                                                const std::complex<float>* csr_val_A,
                                                const rocsparse_int*       csr_row_ptr_A,
                                                const rocsparse_int*       csr_col_ind_A,
                                                rocsparse_int              nnz_A,
                                                const rocsparse_int*       nnz_per_row,
                                                std::complex<float>*       csr_val_C,
                                                rocsparse_int*             csr_row_ptr_C,
                                                rocsparse_int*             csr_col_ind_C,
                                                std::complex<float>        tol)
    {
        return rocsparse_ccsr2csr_compress(handle,
                                           m,
                                           n,
                                           descr_A,
                                           (const rocsparse_float_complex*)csr_val_A,
                                           csr_row_ptr_A,
                                           csr_col_ind_A,
                                           nnz_A,
                                           nnz_per_row,
                                           (rocsparse_float_complex*)csr_val_C,
                                           csr_row_ptr_C,
                                           csr_col_ind_C,
                                           rocsparse_float_complex(std::real(tol), std::imag(tol)));
    }

    template <>
    rocsparse_status rocsparseTcsr2csr_compress(rocsparse_handle            handle,
                                                rocsparse_int               m,
                                                rocsparse_int               n,
                                                const rocsparse_mat_descr   descr_A,
                                                const std::complex<double>* csr_val_A,
                                                const rocsparse_int*        csr_row_ptr_A,
                                                const rocsparse_int*        csr_col_ind_A,
                                                rocsparse_int               nnz_A,
                                                const rocsparse_int*        nnz_per_row,
                                                std::complex<double>*       csr_val_C,
                                                rocsparse_int*              csr_row_ptr_C,
                                                rocsparse_int*              csr_col_ind_C,
                                                std::complex<double>        tol)
    {
        return rocsparse_zcsr2csr_compress(
            handle,
            m,
            n,
            descr_A,
            (const rocsparse_double_complex*)csr_val_A,
            csr_row_ptr_A,
            csr_col_ind_A,
            nnz_A,
            nnz_per_row,
            (rocsparse_double_complex*)csr_val_C,
            csr_row_ptr_C,
            csr_col_ind_C,
            rocsparse_double_complex(std::real(tol), std::imag(tol)));
    }

    template <>
    rocsparse_status rocsparseTcsritilu0_compute(rocsparse_handle     handle,
                                                 rocsparse_itilu0_alg alg,
                                                 rocsparse_int        option,
                                                 rocsparse_int*       nmaxiter,
                                                 float                tol,
                                                 rocsparse_int        m,
                                                 rocsparse_int        nnz,
                                                 const rocsparse_int* csr_row_ptr,
                                                 const rocsparse_int* csr_col_ind,
                                                 const float*         csr_val,
                                                 float*               ilu0,
                                                 rocsparse_index_base idx_base,
                                                 size_t               buffer_size,
                                                 void*                buffer)
    {
        return rocsparse_scsritilu0_compute(handle,
                                            alg,
                                            option,
                                            nmaxiter,
                                            tol,
                                            m,
                                            nnz,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            csr_val,
                                            ilu0,
                                            idx_base,
                                            buffer_size,
                                            buffer);
    }

    template <>
    rocsparse_status rocsparseTcsritilu0_compute(rocsparse_handle     handle,
                                                 rocsparse_itilu0_alg alg,
                                                 rocsparse_int        option,
                                                 rocsparse_int*       nmaxiter,
                                                 double               tol,
                                                 rocsparse_int        m,
                                                 rocsparse_int        nnz,
                                                 const rocsparse_int* csr_row_ptr,
                                                 const rocsparse_int* csr_col_ind,
                                                 const double*        csr_val,
                                                 double*              ilu0,
                                                 rocsparse_index_base idx_base,
                                                 size_t               buffer_size,
                                                 void*                buffer)
    {
        return rocsparse_dcsritilu0_compute(handle,
                                            alg,
                                            option,
                                            nmaxiter,
                                            tol,
                                            m,
                                            nnz,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            csr_val,
                                            ilu0,
                                            idx_base,
                                            buffer_size,
                                            buffer);
    }

    template <>
    rocsparse_status rocsparseTcsritilu0_compute(rocsparse_handle           handle,
                                                 rocsparse_itilu0_alg       alg,
                                                 rocsparse_int              option,
                                                 rocsparse_int*             nmaxiter,
                                                 float                      tol,
                                                 rocsparse_int              m,
                                                 rocsparse_int              nnz,
                                                 const rocsparse_int*       csr_row_ptr,
                                                 const rocsparse_int*       csr_col_ind,
                                                 const std::complex<float>* csr_val,
                                                 std::complex<float>*       ilu0,
                                                 rocsparse_index_base       idx_base,
                                                 size_t                     buffer_size,
                                                 void*                      buffer)
    {
        return rocsparse_ccsritilu0_compute(handle,
                                            alg,
                                            option,
                                            nmaxiter,
                                            tol,
                                            m,
                                            nnz,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            (const rocsparse_float_complex*)csr_val,
                                            (rocsparse_float_complex*)ilu0,
                                            idx_base,
                                            buffer_size,
                                            buffer);
    }

    template <>
    rocsparse_status rocsparseTcsritilu0_compute(rocsparse_handle            handle,
                                                 rocsparse_itilu0_alg        alg,
                                                 rocsparse_int               option,
                                                 rocsparse_int*              nmaxiter,
                                                 double                      tol,
                                                 rocsparse_int               m,
                                                 rocsparse_int               nnz,
                                                 const rocsparse_int*        csr_row_ptr,
                                                 const rocsparse_int*        csr_col_ind,
                                                 const std::complex<double>* csr_val,
                                                 std::complex<double>*       ilu0,
                                                 rocsparse_index_base        idx_base,
                                                 size_t                      buffer_size,
                                                 void*                       buffer)
    {
        return rocsparse_zcsritilu0_compute(handle,
                                            alg,
                                            option,
                                            nmaxiter,
                                            tol,
                                            m,
                                            nnz,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            (const rocsparse_double_complex*)csr_val,
                                            (rocsparse_double_complex*)ilu0,
                                            idx_base,
                                            buffer_size,
                                            buffer);
    }

    template <>
    rocsparse_status rocsparseTcsritilu0_history<float>(rocsparse_handle     handle,
                                                        rocsparse_itilu0_alg alg,
                                                        rocsparse_int*       niter,
                                                        float*               data,
                                                        size_t               buffer_size,
                                                        void*                buffer)
    {
        return rocsparse_scsritilu0_history(handle, alg, niter, data, buffer_size, buffer);
    }

    template <>
    rocsparse_status rocsparseTcsritilu0_history<std::complex<float>>(rocsparse_handle     handle,
                                                                      rocsparse_itilu0_alg alg,
                                                                      rocsparse_int*       niter,
                                                                      float*               data,
                                                                      size_t buffer_size,
                                                                      void*  buffer)
    {
        return rocsparse_ccsritilu0_history(handle, alg, niter, data, buffer_size, buffer);
    }

    template <>
    rocsparse_status rocsparseTcsritilu0_history<double>(rocsparse_handle     handle,
                                                         rocsparse_itilu0_alg alg,
                                                         rocsparse_int*       niter,
                                                         double*              data,
                                                         size_t               buffer_size,
                                                         void*                buffer)
    {
        return rocsparse_dcsritilu0_history(handle, alg, niter, data, buffer_size, buffer);
    }

    template <>
    rocsparse_status rocsparseTcsritilu0_history<std::complex<double>>(rocsparse_handle     handle,
                                                                       rocsparse_itilu0_alg alg,
                                                                       rocsparse_int*       niter,
                                                                       double*              data,
                                                                       size_t buffer_size,
                                                                       void*  buffer)
    {
        return rocsparse_zcsritilu0_history(handle, alg, niter, data, buffer_size, buffer);
    }

} // namespace rocalution

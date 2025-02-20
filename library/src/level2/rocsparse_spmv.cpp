/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
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

#include "internal/generic/rocsparse_spmv.h"
#include "control.h"
#include "handle.h"
#include "utility.h"

#include "rocsparse_bsrmv.hpp"
#include "rocsparse_coomv.hpp"
#include "rocsparse_coomv_aos.hpp"
#include "rocsparse_cscmv.hpp"
#include "rocsparse_csrmv.hpp"
#include "rocsparse_ellmv.hpp"

namespace rocsparse
{
    rocsparse_indextype determine_I_index_type(rocsparse_const_spmat_descr mat);
    rocsparse_indextype determine_J_index_type(rocsparse_const_spmat_descr mat);

    static rocsparse_status check_spmv_alg(rocsparse_format format, rocsparse_spmv_alg alg)
    {
        switch(format)
        {
        case rocsparse_format_csr:
        case rocsparse_format_csc:
        {
            switch(alg)
            {
            case rocsparse_spmv_alg_default:
            case rocsparse_spmv_alg_csr_rowsplit:
            case rocsparse_spmv_alg_csr_adaptive:
            case rocsparse_spmv_alg_csr_lrb:
            {
                return rocsparse_status_success;
            }
            case rocsparse_spmv_alg_coo:
            case rocsparse_spmv_alg_ell:
            case rocsparse_spmv_alg_bsr:
            case rocsparse_spmv_alg_coo_atomic:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            }
            }

            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        }
        case rocsparse_format_coo:
        case rocsparse_format_coo_aos:
        {
            switch(alg)
            {
            case rocsparse_spmv_alg_default:
            case rocsparse_spmv_alg_coo:
            case rocsparse_spmv_alg_coo_atomic:
            {
                return rocsparse_status_success;
            }
            case rocsparse_spmv_alg_csr_rowsplit:
            case rocsparse_spmv_alg_csr_adaptive:
            case rocsparse_spmv_alg_bsr:
            case rocsparse_spmv_alg_ell:
            case rocsparse_spmv_alg_csr_lrb:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            }
            }

            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        }
        case rocsparse_format_ell:
        {
            switch(alg)
            {
            case rocsparse_spmv_alg_default:
            case rocsparse_spmv_alg_ell:
            {
                return rocsparse_status_success;
            }
            case rocsparse_spmv_alg_csr_rowsplit:
            case rocsparse_spmv_alg_csr_adaptive:
            case rocsparse_spmv_alg_bsr:
            case rocsparse_spmv_alg_coo:
            case rocsparse_spmv_alg_coo_atomic:
            case rocsparse_spmv_alg_csr_lrb:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            }
            }

            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        }
        case rocsparse_format_bell:
        {
            switch(alg)
            {
            case rocsparse_spmv_alg_default:
            case rocsparse_spmv_alg_coo:
            case rocsparse_spmv_alg_csr_rowsplit:
            case rocsparse_spmv_alg_csr_adaptive:
            case rocsparse_spmv_alg_ell:
            case rocsparse_spmv_alg_bsr:
            case rocsparse_spmv_alg_coo_atomic:
            case rocsparse_spmv_alg_csr_lrb:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            }
            }

            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        }

        case rocsparse_format_bsr:
        {
            switch(alg)
            {
            case rocsparse_spmv_alg_default:
            case rocsparse_spmv_alg_bsr:
            {
                return rocsparse_status_success;
            }
            case rocsparse_spmv_alg_ell:
            case rocsparse_spmv_alg_csr_rowsplit:
            case rocsparse_spmv_alg_csr_adaptive:
            case rocsparse_spmv_alg_coo:
            case rocsparse_spmv_alg_coo_atomic:
            case rocsparse_spmv_alg_csr_lrb:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            }
            }

            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        }
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    }

    static rocsparse_status spmv_alg2csrmv_alg(rocsparse_spmv_alg    spmv_alg,
                                               rocsparse::csrmv_alg& target)
    {
        switch(spmv_alg)
        {
        case rocsparse_spmv_alg_csr_rowsplit:
        {
            target = rocsparse::csrmv_alg_rowsplit;
            return rocsparse_status_success;
        }

        case rocsparse_spmv_alg_default:
        case rocsparse_spmv_alg_csr_adaptive:
        {
            target = rocsparse::csrmv_alg_adaptive;
            return rocsparse_status_success;
        }

        case rocsparse_spmv_alg_csr_lrb:
        {
            target = rocsparse::csrmv_alg_lrb;
            return rocsparse_status_success;
        }

        case rocsparse_spmv_alg_coo:
        case rocsparse_spmv_alg_coo_atomic:
        case rocsparse_spmv_alg_bsr:
        case rocsparse_spmv_alg_ell:
        {
            return rocsparse_status_invalid_value;
        }
        }
        return rocsparse_status_invalid_value;
    }

    static rocsparse_status spmv_alg2coomv_alg(rocsparse_spmv_alg   spmv_alg,
                                               rocsparse_coomv_alg& coomv_alg)
    {
        switch(spmv_alg)
        {
        case rocsparse_spmv_alg_default:
        {
            coomv_alg = rocsparse_coomv_alg_default;
            return rocsparse_status_success;
        }

        case rocsparse_spmv_alg_coo:
        {
            coomv_alg = rocsparse_coomv_alg_segmented;
            return rocsparse_status_success;
        }

        case rocsparse_spmv_alg_coo_atomic:
        {
            coomv_alg = rocsparse_coomv_alg_atomic;
            return rocsparse_status_success;
        }

        case rocsparse_spmv_alg_csr_adaptive:
        case rocsparse_spmv_alg_csr_rowsplit:
        case rocsparse_spmv_alg_bsr:
        case rocsparse_spmv_alg_ell:
        case rocsparse_spmv_alg_csr_lrb:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        }
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    }

    static rocsparse_status spmv_alg2coomv_aos_alg(rocsparse_spmv_alg       spmv_alg,
                                                   rocsparse_coomv_aos_alg& coomv_aos_alg)
    {
        switch(spmv_alg)
        {
        case rocsparse_spmv_alg_default:
        {
            coomv_aos_alg = rocsparse_coomv_aos_alg_default;
            return rocsparse_status_success;
        }

        case rocsparse_spmv_alg_coo:
        {
            coomv_aos_alg = rocsparse_coomv_aos_alg_segmented;
            return rocsparse_status_success;
        }

        case rocsparse_spmv_alg_coo_atomic:
        {
            coomv_aos_alg = rocsparse_coomv_aos_alg_atomic;
            return rocsparse_status_success;
        }

        case rocsparse_spmv_alg_csr_adaptive:
        case rocsparse_spmv_alg_csr_rowsplit:
        case rocsparse_spmv_alg_bsr:
        case rocsparse_spmv_alg_ell:
        case rocsparse_spmv_alg_csr_lrb:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        }
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    }
}

namespace rocsparse
{
    template <typename T, typename I, typename J, typename A, typename X, typename Y>
    rocsparse_status spmv_template(rocsparse_handle            handle,
                                   rocsparse_operation         trans,
                                   const void*                 alpha,
                                   rocsparse_const_spmat_descr mat,
                                   rocsparse_const_dnvec_descr x,
                                   const void*                 beta,
                                   const rocsparse_dnvec_descr y,
                                   rocsparse_spmv_alg          alg,
                                   rocsparse_spmv_stage        stage,
                                   size_t*                     buffer_size,
                                   void*                       temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        RETURN_IF_ROCSPARSE_ERROR((rocsparse::check_spmv_alg(mat->format, alg)));

        switch(mat->format)
        {
        case rocsparse_format_coo:
        {
            rocsparse_coomv_alg coomv_alg;
            RETURN_IF_ROCSPARSE_ERROR((rocsparse::spmv_alg2coomv_alg(alg, coomv_alg)));

            switch(stage)
            {
            case rocsparse_spmv_stage_buffer_size:
            {
                *buffer_size = 0;
                return rocsparse_status_success;
            }
            case rocsparse_spmv_stage_preprocess:
            {
                if(alg == rocsparse_spmv_alg_coo_atomic && mat->analysed == false)
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::coomv_analysis_template(handle,
                                                            trans,
                                                            coomv_alg,
                                                            (I)mat->rows,
                                                            (I)mat->cols,
                                                            mat->nnz,
                                                            mat->descr,
                                                            (const A*)mat->const_val_data,
                                                            (const I*)mat->const_row_data,
                                                            (const I*)mat->const_col_data)));

                    mat->analysed = true;
                }
                return rocsparse_status_success;
            }
            case rocsparse_spmv_stage_compute:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::coomv_template(handle,
                                                                    trans,
                                                                    coomv_alg,
                                                                    (I)mat->rows,
                                                                    (I)mat->cols,
                                                                    mat->nnz,
                                                                    (const T*)alpha,
                                                                    mat->descr,
                                                                    (const A*)mat->const_val_data,
                                                                    (const I*)mat->const_row_data,
                                                                    (const I*)mat->const_col_data,
                                                                    (const X*)x->const_values,
                                                                    (const T*)beta,
                                                                    (Y*)y->values));
                return rocsparse_status_success;
            }
            }
        }

        case rocsparse_format_coo_aos:
        {
            rocsparse_coomv_aos_alg coomv_aos_alg;
            RETURN_IF_ROCSPARSE_ERROR((rocsparse::spmv_alg2coomv_aos_alg(alg, coomv_aos_alg)));

            switch(stage)
            {
            case rocsparse_spmv_stage_buffer_size:
            {
                *buffer_size = 0;
                return rocsparse_status_success;
            }
            case rocsparse_spmv_stage_preprocess:
            {
                return rocsparse_status_success;
            }
            case rocsparse_spmv_stage_compute:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::coomv_aos_template(handle,
                                                  trans,
                                                  coomv_aos_alg,
                                                  (I)mat->rows,
                                                  (I)mat->cols,
                                                  mat->nnz,
                                                  (const T*)alpha,
                                                  mat->descr,
                                                  (const A*)mat->const_val_data,
                                                  (const I*)mat->const_ind_data,
                                                  (const X*)x->const_values,
                                                  (const T*)beta,
                                                  (Y*)y->values));
                return rocsparse_status_success;
            }
            }
        }

        case rocsparse_format_bsr:
        {
            switch(stage)
            {
            case rocsparse_spmv_stage_buffer_size:
            {
                *buffer_size = 0;
                return rocsparse_status_success;
            }

            case rocsparse_spmv_stage_preprocess:
            {
                //
                // If algorithm 1 or default is selected and analysis step is required
                //
                if(alg == rocsparse_spmv_alg_default && mat->analysed == false)
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        rocsparse::bsrmv_analysis_template(handle,
                                                           mat->block_dir,
                                                           trans,
                                                           (J)mat->rows,
                                                           (J)mat->cols,
                                                           (I)mat->nnz,
                                                           mat->descr,
                                                           (const A*)mat->const_val_data,
                                                           (const I*)mat->const_row_data,
                                                           (const J*)mat->const_col_data,
                                                           (J)mat->block_dim,
                                                           mat->info));
                    mat->analysed = true;
                }

                return rocsparse_status_success;
            }

            case rocsparse_spmv_stage_compute:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrmv_template(handle,
                                                                    mat->block_dir,
                                                                    trans,
                                                                    (J)mat->rows,
                                                                    (J)mat->cols,
                                                                    (I)mat->nnz,
                                                                    (const T*)alpha,
                                                                    mat->descr,
                                                                    (const A*)mat->const_val_data,
                                                                    (const I*)mat->const_row_data,
                                                                    (const J*)mat->const_col_data,
                                                                    (J)mat->block_dim,
                                                                    mat->info,
                                                                    (const X*)x->const_values,
                                                                    (const T*)beta,
                                                                    (Y*)y->values));
                return rocsparse_status_success;
            }
            }
        }

        case rocsparse_format_csr:
        {
            rocsparse::csrmv_alg alg_csrmv;
            RETURN_IF_ROCSPARSE_ERROR((rocsparse::spmv_alg2csrmv_alg(alg, alg_csrmv)));

            switch(stage)
            {
            case rocsparse_spmv_stage_buffer_size:
            {
                *buffer_size = 0;
                return rocsparse_status_success;
            }

            case rocsparse_spmv_stage_preprocess:
            {

                //
                // If algorithm 1 or default is selected and analysis step is required
                //
                if((alg == rocsparse_spmv_alg_default || alg == rocsparse_spmv_alg_csr_adaptive
                    || alg == rocsparse_spmv_alg_csr_lrb)
                   && mat->analysed == false)
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        rocsparse::csrmv_analysis_template(handle,
                                                           trans,
                                                           alg_csrmv,
                                                           (J)mat->rows,
                                                           (J)mat->cols,
                                                           (I)mat->nnz,
                                                           mat->descr,
                                                           (const A*)mat->const_val_data,
                                                           (const I*)mat->const_row_data,
                                                           (const J*)mat->const_col_data,
                                                           mat->info));

                    mat->analysed = true;
                }

                return rocsparse_status_success;
            }

            case rocsparse_spmv_stage_compute:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrmv_template(
                    handle,
                    trans,
                    alg_csrmv,
                    (J)mat->rows,
                    (J)mat->cols,
                    (I)mat->nnz,
                    (const T*)alpha,
                    mat->descr,
                    (const A*)mat->const_val_data,
                    (const I*)mat->const_row_data,
                    ((const I*)mat->const_row_data) + 1,
                    (const J*)mat->const_col_data,
                    (alg == rocsparse_spmv_alg_csr_rowsplit) ? nullptr : mat->info,
                    (const X*)x->const_values,
                    (const T*)beta,
                    (Y*)y->values,
                    false));
                return rocsparse_status_success;
            }
            }
        }

        case rocsparse_format_csc:
        {
            rocsparse::csrmv_alg alg_csrmv;
            RETURN_IF_ROCSPARSE_ERROR((rocsparse::spmv_alg2csrmv_alg(alg, alg_csrmv)));

            switch(stage)
            {
            case rocsparse_spmv_stage_buffer_size:
            {
                *buffer_size = 0;
                return rocsparse_status_success;
            }

            case rocsparse_spmv_stage_preprocess:
            {
                //
                // If algorithm 1 or default is selected and analysis step is required
                //
                if((alg == rocsparse_spmv_alg_default || alg == rocsparse_spmv_alg_csr_adaptive
                    || alg == rocsparse_spmv_alg_csr_lrb)
                   && mat->analysed == false)
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        rocsparse::cscmv_analysis_template(handle,
                                                           trans,
                                                           alg_csrmv,
                                                           (J)mat->rows,
                                                           (J)mat->cols,
                                                           (I)mat->nnz,
                                                           mat->descr,
                                                           (const A*)mat->const_val_data,
                                                           (const I*)mat->const_col_data,
                                                           (const J*)mat->const_row_data,
                                                           mat->info));

                    mat->analysed = true;
                }
                return rocsparse_status_success;
            }

            case rocsparse_spmv_stage_compute:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::cscmv_template(
                    handle,
                    trans,
                    alg_csrmv,
                    (J)mat->rows,
                    (J)mat->cols,
                    (I)mat->nnz,
                    (const T*)alpha,
                    mat->descr,
                    (const A*)mat->const_val_data,
                    (const I*)mat->const_col_data,
                    (const J*)mat->const_row_data,
                    (alg == rocsparse_spmv_alg_csr_rowsplit) ? nullptr : mat->info,
                    (const X*)x->const_values,
                    (const T*)beta,
                    (Y*)y->values));
                return rocsparse_status_success;
            }
            }
        }

        case rocsparse_format_ell:
        {
            switch(stage)
            {
            case rocsparse_spmv_stage_buffer_size:
            {
                *buffer_size = 0;
                return rocsparse_status_success;
            }

            case rocsparse_spmv_stage_preprocess:
            {
                return rocsparse_status_success;
            }

            case rocsparse_spmv_stage_compute:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::ellmv_template(handle,
                                                                    trans,
                                                                    (I)mat->rows,
                                                                    (I)mat->cols,
                                                                    (const T*)alpha,
                                                                    mat->descr,
                                                                    (const A*)mat->const_val_data,
                                                                    (const I*)mat->const_col_data,
                                                                    (I)mat->ell_width,
                                                                    (const X*)x->const_values,
                                                                    (const T*)beta,
                                                                    (Y*)y->values));
                return rocsparse_status_success;
            }
            }
        }

        case rocsparse_format_bell:
        {
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            // LCOV_EXCL_STOP
        }
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    }

    template <typename... Ts>
    rocsparse_status spmv_dynamic_dispatch(rocsparse_indextype itype,
                                           rocsparse_indextype jtype,
                                           rocsparse_datatype  atype,
                                           rocsparse_datatype  xtype,
                                           rocsparse_datatype  ytype,
                                           rocsparse_datatype  ctype,
                                           Ts&&... ts)
    {
        ROCSPARSE_ROUTINE_TRACE;

#define DISPATCH_COMPUTE_TYPE_I32R(ITYPE, JTYPE, CTYPE, atype, xtype, ytype)                  \
    if(atype == rocsparse_datatype_i8_r && xtype == rocsparse_datatype_i8_r                   \
       && ytype == rocsparse_datatype_i32_r)                                                  \
    {                                                                                         \
        RETURN_IF_ROCSPARSE_ERROR(                                                            \
            (rocsparse::spmv_template<CTYPE, ITYPE, JTYPE, int8_t, int8_t, int32_t>(ts...))); \
        return rocsparse_status_success;                                                      \
    }                                                                                         \
    else                                                                                      \
    {                                                                                         \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);                          \
    }

#define DISPATCH_COMPUTE_TYPE_F32R(ITYPE, JTYPE, CTYPE, atype, xtype, ytype)                \
    if(atype == rocsparse_datatype_f32_r && atype == xtype && atype == ytype)               \
    {                                                                                       \
        RETURN_IF_ROCSPARSE_ERROR(                                                          \
            (rocsparse::spmv_template<CTYPE, ITYPE, JTYPE, float, float, float>(ts...)));   \
        return rocsparse_status_success;                                                    \
    }                                                                                       \
    else if(atype == rocsparse_datatype_i8_r && xtype == rocsparse_datatype_i8_r            \
            && ytype == rocsparse_datatype_f32_r)                                           \
    {                                                                                       \
        RETURN_IF_ROCSPARSE_ERROR(                                                          \
            (rocsparse::spmv_template<CTYPE, ITYPE, JTYPE, int8_t, int8_t, float>(ts...))); \
        return rocsparse_status_success;                                                    \
    }                                                                                       \
    else                                                                                    \
    {                                                                                       \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);                        \
    }

#define DISPATCH_COMPUTE_TYPE_F64R(ITYPE, JTYPE, CTYPE, atype, xtype, ytype)                 \
    if(atype == rocsparse_datatype_f64_r && atype == xtype && atype == ytype)                \
    {                                                                                        \
        RETURN_IF_ROCSPARSE_ERROR(                                                           \
            (rocsparse::spmv_template<CTYPE, ITYPE, JTYPE, double, double, double>(ts...))); \
        return rocsparse_status_success;                                                     \
    }                                                                                        \
    else if(atype == rocsparse_datatype_f32_r && xtype == rocsparse_datatype_f64_r           \
            && xtype == ytype)                                                               \
    {                                                                                        \
        RETURN_IF_ROCSPARSE_ERROR(                                                           \
            (rocsparse::spmv_template<CTYPE, ITYPE, JTYPE, float, double, double>(ts...)));  \
        return rocsparse_status_success;                                                     \
    }                                                                                        \
    else                                                                                     \
    {                                                                                        \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);                         \
    }

#define DISPATCH_COMPUTE_TYPE_F32C(ITYPE, JTYPE, CTYPE, atype, xtype, ytype)                   \
    if(atype == rocsparse_datatype_f32_c && atype == xtype && atype == ytype)                  \
    {                                                                                          \
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::spmv_template<CTYPE,                             \
                                                            ITYPE,                             \
                                                            JTYPE,                             \
                                                            rocsparse_float_complex,           \
                                                            rocsparse_float_complex,           \
                                                            rocsparse_float_complex>(ts...))); \
        return rocsparse_status_success;                                                       \
    }                                                                                          \
    else if(atype == rocsparse_datatype_f32_r && xtype == rocsparse_datatype_f32_c             \
            && ytype == rocsparse_datatype_f32_c)                                              \
    {                                                                                          \
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::spmv_template<CTYPE,                             \
                                                            ITYPE,                             \
                                                            JTYPE,                             \
                                                            float,                             \
                                                            rocsparse_float_complex,           \
                                                            rocsparse_float_complex>(ts...))); \
        return rocsparse_status_success;                                                       \
    }                                                                                          \
    else                                                                                       \
    {                                                                                          \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);                           \
    }

#define DISPATCH_COMPUTE_TYPE_F64C(ITYPE, JTYPE, CTYPE, atype, xtype, ytype)                    \
    if(atype == rocsparse_datatype_f64_c && atype == xtype && atype == ytype)                   \
    {                                                                                           \
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::spmv_template<CTYPE,                              \
                                                            ITYPE,                              \
                                                            JTYPE,                              \
                                                            rocsparse_double_complex,           \
                                                            rocsparse_double_complex,           \
                                                            rocsparse_double_complex>(ts...))); \
        return rocsparse_status_success;                                                        \
    }                                                                                           \
    else if(atype == rocsparse_datatype_f64_r && xtype == rocsparse_datatype_f64_c              \
            && ytype == rocsparse_datatype_f64_c)                                               \
    {                                                                                           \
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::spmv_template<CTYPE,                              \
                                                            ITYPE,                              \
                                                            JTYPE,                              \
                                                            double,                             \
                                                            rocsparse_double_complex,           \
                                                            rocsparse_double_complex>(ts...))); \
        return rocsparse_status_success;                                                        \
    }                                                                                           \
    else if(atype == rocsparse_datatype_f32_c && xtype == rocsparse_datatype_f64_c              \
            && xtype == ytype)                                                                  \
    {                                                                                           \
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::spmv_template<CTYPE,                              \
                                                            ITYPE,                              \
                                                            JTYPE,                              \
                                                            rocsparse_float_complex,            \
                                                            rocsparse_double_complex,           \
                                                            rocsparse_double_complex>(ts...))); \
        return rocsparse_status_success;                                                        \
    }                                                                                           \
    else                                                                                        \
    {                                                                                           \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);                            \
    }

#define DISPATCH_COMPUTE_TYPE(ITYPE, JTYPE, atype, xtype, ytype, ctype)                         \
    switch(ctype)                                                                               \
    {                                                                                           \
    case rocsparse_datatype_i32_r:                                                              \
    {                                                                                           \
        DISPATCH_COMPUTE_TYPE_I32R(ITYPE, JTYPE, int32_t, atype, xtype, ytype)                  \
    }                                                                                           \
    case rocsparse_datatype_f32_r:                                                              \
    {                                                                                           \
        DISPATCH_COMPUTE_TYPE_F32R(ITYPE, JTYPE, float, atype, xtype, ytype)                    \
    }                                                                                           \
    case rocsparse_datatype_f64_r:                                                              \
    {                                                                                           \
        DISPATCH_COMPUTE_TYPE_F64R(ITYPE, JTYPE, double, atype, xtype, ytype)                   \
    }                                                                                           \
    case rocsparse_datatype_f32_c:                                                              \
    {                                                                                           \
        DISPATCH_COMPUTE_TYPE_F32C(ITYPE, JTYPE, rocsparse_float_complex, atype, xtype, ytype)  \
    }                                                                                           \
    case rocsparse_datatype_f64_c:                                                              \
    {                                                                                           \
        DISPATCH_COMPUTE_TYPE_F64C(ITYPE, JTYPE, rocsparse_double_complex, atype, xtype, ytype) \
    }                                                                                           \
    case rocsparse_datatype_i8_r:                                                               \
    case rocsparse_datatype_u8_r:                                                               \
    case rocsparse_datatype_u32_r:                                                              \
    case rocsparse_datatype_f16_r:                                                              \
    {                                                                                           \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);                            \
    }                                                                                           \
    }

        switch(itype)
        {
        case rocsparse_indextype_u16:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        case rocsparse_indextype_i32:
        {
            switch(jtype)
            {
            case rocsparse_indextype_u16:
            case rocsparse_indextype_i64:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            case rocsparse_indextype_i32:
            {
                DISPATCH_COMPUTE_TYPE(int32_t, int32_t, atype, xtype, ytype, ctype);
            }
            }
        }
        case rocsparse_indextype_i64:
        {
            switch(jtype)
            {
            case rocsparse_indextype_u16:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            case rocsparse_indextype_i32:
            {
                DISPATCH_COMPUTE_TYPE(int64_t, int32_t, atype, xtype, ytype, ctype);
            }
            case rocsparse_indextype_i64:
            {
                DISPATCH_COMPUTE_TYPE(int64_t, int64_t, atype, xtype, ytype, ctype);
            }
            }
        }
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_spmv(rocsparse_handle            handle, //0
                                           rocsparse_operation         trans, //1
                                           const void*                 alpha, //2
                                           rocsparse_const_spmat_descr mat, //3
                                           rocsparse_const_dnvec_descr x, //4
                                           const void*                 beta, //5
                                           const rocsparse_dnvec_descr y, //6
                                           rocsparse_datatype          compute_type, //7
                                           rocsparse_spmv_alg          alg, //8
                                           rocsparse_spmv_stage        stage, //9
                                           size_t*                     buffer_size, //10
                                           void*                       temp_buffer) //11
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_spmv",
                         trans,
                         (const void*&)alpha,
                         (const void*&)mat,
                         (const void*&)x,
                         (const void*&)beta,
                         (const void*&)y,
                         compute_type,
                         alg,
                         stage,
                         (const void*&)buffer_size,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, trans);
    ROCSPARSE_CHECKARG_POINTER(2, alpha);
    ROCSPARSE_CHECKARG_POINTER(3, mat);
    ROCSPARSE_CHECKARG_POINTER(4, x);
    ROCSPARSE_CHECKARG_POINTER(5, beta);
    ROCSPARSE_CHECKARG_POINTER(6, y);
    ROCSPARSE_CHECKARG_ENUM(7, compute_type);
    ROCSPARSE_CHECKARG_ENUM(8, alg);
    ROCSPARSE_CHECKARG_ENUM(9, stage);

    // Check for valid buffer_size pointer only if temp_buffer is nullptr
    ROCSPARSE_CHECKARG(10,
                       buffer_size,
                       (temp_buffer == nullptr && buffer_size == nullptr),
                       rocsparse_status_invalid_pointer);

    // Check if descriptors are initialized
    // Basically this never happens, but I let it here.
    // LCOV_EXCL_START
    ROCSPARSE_CHECKARG(3, mat, (mat->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(4, x, (x->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(6, y, (y->init == false), rocsparse_status_not_initialized);
    // LCOV_EXCL_STOP

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::spmv_dynamic_dispatch(rocsparse::determine_I_index_type(mat),
                                         rocsparse::determine_J_index_type(mat),
                                         mat->data_type,
                                         x->data_type,
                                         y->data_type,
                                         compute_type,
                                         handle,
                                         trans,
                                         alpha,
                                         mat,
                                         x,
                                         beta,
                                         y,
                                         alg,
                                         stage,
                                         buffer_size,
                                         temp_buffer));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

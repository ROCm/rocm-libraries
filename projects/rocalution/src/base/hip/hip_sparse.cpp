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

#include "hip_sparse.hpp"
#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "hip_allocate_free.hpp"
#include "hip_utils.hpp"

#include <cassert>
#include <complex>
#include <rocsparse/rocsparse.h>

namespace rocalution
{
    template <typename ValueType>
    HIPSpMV<ValueType>::HIPSpMV(void)
        : descr_(NULL)
        , buffer_size_(0)
        , buffer_(NULL)
    {
    }

    template <typename ValueType>
    HIPSpMV<ValueType>::~HIPSpMV(void)
    {
        this->Clear();
    }

    template <typename ValueType>
    bool HIPSpMV<ValueType>::IsAnalysed(void) const
    {
        return this->descr_ != NULL;
    }

    template <typename ValueType>
    void HIPSpMV<ValueType>::Analyse(rocsparse_handle            handle,
                                     rocsparse_spmv_alg          alg,
                                     rocsparse_const_spmat_descr mat,
                                     ValueType                   alpha,
                                     rocsparse_const_dnvec_descr x,
                                     ValueType                   beta,
                                     rocsparse_dnvec_descr       y)
    {
        assert(this->descr_ == NULL);
        assert(mat != NULL);

        const rocsparse_operation operation = rocsparse_operation_none;
        const rocsparse_datatype  datatype  = rocalution_datatype_traits<ValueType>::value;

        rocsparse_status status = rocsparse_create_spmv_descr(&this->descr_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_spmv_set_input(
            handle, this->descr_, rocsparse_spmv_input_alg, &alg, sizeof(alg), NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_spmv_set_input(handle,
                                          this->descr_,
                                          rocsparse_spmv_input_operation,
                                          &operation,
                                          sizeof(operation),
                                          NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_spmv_set_input(handle,
                                          this->descr_,
                                          rocsparse_spmv_input_scalar_datatype,
                                          &datatype,
                                          sizeof(datatype),
                                          NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_spmv_set_input(handle,
                                          this->descr_,
                                          rocsparse_spmv_input_compute_datatype,
                                          &datatype,
                                          sizeof(datatype),
                                          NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        size_t analysis_buffer_size = 0;
        status                      = rocsparse_v2_spmv_buffer_size(handle,
                                               this->descr_,
                                               mat,
                                               x,
                                               y,
                                               rocsparse_v2_spmv_stage_analysis,
                                               &analysis_buffer_size,
                                               NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        char* analysis_buffer = NULL;
        allocate_hip(analysis_buffer_size, &analysis_buffer);

        status = rocsparse_v2_spmv(handle,
                                   this->descr_,
                                   &alpha,
                                   mat,
                                   x,
                                   &beta,
                                   y,
                                   rocsparse_v2_spmv_stage_analysis,
                                   analysis_buffer_size,
                                   analysis_buffer,
                                   NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        free_hip(&analysis_buffer);

        status = rocsparse_v2_spmv_buffer_size(handle,
                                               this->descr_,
                                               mat,
                                               x,
                                               y,
                                               rocsparse_v2_spmv_stage_compute,
                                               &this->buffer_size_,
                                               NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        allocate_hip(this->buffer_size_, &this->buffer_);
    }

    template <typename ValueType>
    void HIPSpMV<ValueType>::Analyse(rocsparse_handle            handle,
                                     rocsparse_spmv_alg          alg,
                                     rocsparse_const_spmat_descr mat)
    {
        // The analysis only depends on the matrix, empty vectors are sufficient
        rocsparse_dnvec_descr x;
        rocsparse_dnvec_descr y;

        rocsparse_status status = rocsparse_create_dnvec_descr(
            &x, 0, NULL, rocalution_datatype_traits<ValueType>::value);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_create_dnvec_descr(
            &y, 0, NULL, rocalution_datatype_traits<ValueType>::value);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        this->Analyse(handle, alg, mat, static_cast<ValueType>(1), x, static_cast<ValueType>(0), y);

        status = rocsparse_destroy_dnvec_descr(x);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_destroy_dnvec_descr(y);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    template <typename ValueType>
    void HIPSpMV<ValueType>::Compute(rocsparse_handle            handle,
                                     ValueType                   alpha,
                                     rocsparse_const_spmat_descr mat,
                                     rocsparse_const_dnvec_descr x,
                                     ValueType                   beta,
                                     rocsparse_dnvec_descr       y) const
    {
        assert(this->descr_ != NULL);

        rocsparse_status status = rocsparse_v2_spmv(handle,
                                                    this->descr_,
                                                    &alpha,
                                                    mat,
                                                    x,
                                                    &beta,
                                                    y,
                                                    rocsparse_v2_spmv_stage_compute,
                                                    this->buffer_size_,
                                                    this->buffer_,
                                                    NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    template <typename ValueType>
    void HIPSpMV<ValueType>::Clear(void)
    {
        if(this->descr_ != NULL)
        {
            rocsparse_status status = rocsparse_destroy_spmv_descr(this->descr_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            this->descr_ = NULL;
        }

        free_hip(&this->buffer_);
        this->buffer_size_ = 0;
    }

    template class HIPSpMV<float>;
    template class HIPSpMV<double>;
#ifdef SUPPORT_COMPLEX
    template class HIPSpMV<std::complex<float>>;
    template class HIPSpMV<std::complex<double>>;
#endif

    // Creates a sparse matrix descriptor on the arrays of mat, whose fill mode and diagonal
    // type attributes select the triangle that is used by the triangular solvers
    static void create_triangle_descr(rocsparse_const_spmat_descr mat,
                                      rocsparse_fill_mode         fill_mode,
                                      rocsparse_diag_type         diag_type,
                                      rocsparse_spmat_descr*      triangle)
    {
        assert(mat != NULL);
        assert(triangle != NULL);

        rocsparse_format format;

        rocsparse_status status = rocsparse_spmat_get_format(mat, &format);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        int64_t              nrow;
        int64_t              ncol;
        int64_t              nnz;
        const void*          row;
        const void*          col;
        const void*          val;
        rocsparse_indextype  row_type;
        rocsparse_indextype  col_type;
        rocsparse_index_base base;
        rocsparse_datatype   data_type;

        switch(format)
        {
        case rocsparse_format_csr:
        {
            status = rocsparse_const_csr_get(
                mat, &nrow, &ncol, &nnz, &row, &col, &val, &row_type, &col_type, &base, &data_type);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            status = rocsparse_create_csr_descr(triangle,
                                                nrow,
                                                ncol,
                                                nnz,
                                                const_cast<void*>(row),
                                                const_cast<void*>(col),
                                                const_cast<void*>(val),
                                                row_type,
                                                col_type,
                                                base,
                                                data_type);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
            break;
        }
        default:
        {
            LOG_INFO("Triangular solves do not support this matrix format");
            FATAL_ERROR(__FILE__, __LINE__);
        }
        }

        status = rocsparse_spmat_set_attribute(
            *triangle, rocsparse_spmat_fill_mode, &fill_mode, sizeof(fill_mode));
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_spmat_set_attribute(
            *triangle, rocsparse_spmat_diag_type, &diag_type, sizeof(diag_type));
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    template <typename ValueType>
    HIPSpTRSV<ValueType>::HIPSpTRSV(void)
        : triangle_(NULL)
        , fill_mode_(rocsparse_fill_mode_lower)
        , diag_type_(rocsparse_diag_type_non_unit)
        , descr_(NULL)
        , buffer_size_(0)
        , buffer_(NULL)
    {
    }

    template <typename ValueType>
    HIPSpTRSV<ValueType>::~HIPSpTRSV(void)
    {
        this->Clear();
    }

    template <typename ValueType>
    bool HIPSpTRSV<ValueType>::IsAnalysed(void) const
    {
        return this->descr_ != NULL;
    }

    template <typename ValueType>
    void HIPSpTRSV<ValueType>::Analyse(rocsparse_handle            handle,
                                       rocsparse_const_spmat_descr mat,
                                       rocsparse_operation         trans,
                                       rocsparse_fill_mode         fill_mode,
                                       rocsparse_diag_type         diag_type)
    {
        this->Clear();

        this->fill_mode_ = fill_mode;
        this->diag_type_ = diag_type;

        create_triangle_descr(mat, fill_mode, diag_type, &this->triangle_);

        const rocsparse_sptrsv_alg      alg      = rocsparse_sptrsv_alg_default;
        const rocsparse_analysis_policy policy   = rocsparse_analysis_policy_force;
        const rocsparse_datatype        datatype = rocalution_datatype_traits<ValueType>::value;

        rocsparse_status status = rocsparse_create_sptrsv_descr(&this->descr_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_sptrsv_set_input(
            handle, this->descr_, rocsparse_sptrsv_input_alg, &alg, sizeof(alg), NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_sptrsv_set_input(
            handle, this->descr_, rocsparse_sptrsv_input_operation, &trans, sizeof(trans), NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_sptrsv_set_input(handle,
                                            this->descr_,
                                            rocsparse_sptrsv_input_scalar_datatype,
                                            &datatype,
                                            sizeof(datatype),
                                            NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_sptrsv_set_input(handle,
                                            this->descr_,
                                            rocsparse_sptrsv_input_compute_datatype,
                                            &datatype,
                                            sizeof(datatype),
                                            NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_sptrsv_set_input(handle,
                                            this->descr_,
                                            rocsparse_sptrsv_input_analysis_policy,
                                            &policy,
                                            sizeof(policy),
                                            NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        // The analysis only depends on the matrix, empty vectors are sufficient
        rocsparse_dnvec_descr x;
        rocsparse_dnvec_descr y;

        status = rocsparse_create_dnvec_descr(&x, 0, NULL, datatype);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_create_dnvec_descr(&y, 0, NULL, datatype);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        size_t analysis_buffer_size = 0;
        status                      = rocsparse_sptrsv_buffer_size(handle,
                                              this->descr_,
                                              this->triangle_,
                                              x,
                                              y,
                                              rocsparse_sptrsv_stage_analysis,
                                              &analysis_buffer_size,
                                              NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        char* analysis_buffer = NULL;
        allocate_hip(analysis_buffer_size, &analysis_buffer);

        status = rocsparse_sptrsv(handle,
                                  this->descr_,
                                  this->triangle_,
                                  x,
                                  y,
                                  rocsparse_sptrsv_stage_analysis,
                                  analysis_buffer_size,
                                  analysis_buffer,
                                  NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        free_hip(&analysis_buffer);

        status = rocsparse_sptrsv_buffer_size(handle,
                                              this->descr_,
                                              this->triangle_,
                                              x,
                                              y,
                                              rocsparse_sptrsv_stage_compute,
                                              &this->buffer_size_,
                                              NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        allocate_hip(this->buffer_size_, &this->buffer_);

        status = rocsparse_destroy_dnvec_descr(x);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_destroy_dnvec_descr(y);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    template <typename ValueType>
    void HIPSpTRSV<ValueType>::Update(rocsparse_const_spmat_descr mat)
    {
        if(this->triangle_ == NULL)
        {
            return;
        }

        rocsparse_format format;
        rocsparse_format format_T;
        int64_t          nrow, ncol, nnz;
        int64_t          nrow_T, ncol_T, nnz_T;

        rocsparse_status status = rocsparse_spmat_get_format(mat, &format);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_spmat_get_format(this->triangle_, &format_T);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_spmat_get_size(mat, &nrow, &ncol, &nnz);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_spmat_get_size(this->triangle_, &nrow_T, &ncol_T, &nnz_T);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        if(format != format_T || nrow != nrow_T || ncol != ncol_T || nnz != nnz_T)
        {
            this->Clear();
            return;
        }

        status = rocsparse_destroy_spmat_descr(this->triangle_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        this->triangle_ = NULL;

        create_triangle_descr(mat, this->fill_mode_, this->diag_type_, &this->triangle_);
    }

    template <typename ValueType>
    void HIPSpTRSV<ValueType>::Solve(rocsparse_handle            handle,
                                     ValueType                   alpha,
                                     rocsparse_const_dnvec_descr x,
                                     rocsparse_dnvec_descr       y) const
    {
        assert(this->descr_ != NULL);

        // rocsparse keeps the pointer to alpha, not its value
        rocsparse_status status = rocsparse_sptrsv_set_input(handle,
                                                             this->descr_,
                                                             rocsparse_sptrsv_input_scalar_alpha,
                                                             &alpha,
                                                             sizeof(&alpha),
                                                             NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_sptrsv(handle,
                                  this->descr_,
                                  this->triangle_,
                                  x,
                                  y,
                                  rocsparse_sptrsv_stage_compute,
                                  this->buffer_size_,
                                  this->buffer_,
                                  NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    template <typename ValueType>
    void HIPSpTRSV<ValueType>::Clear(void)
    {
        if(this->descr_ != NULL)
        {
            rocsparse_status status = rocsparse_destroy_sptrsv_descr(this->descr_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            this->descr_ = NULL;
        }

        if(this->triangle_ != NULL)
        {
            rocsparse_status status = rocsparse_destroy_spmat_descr(this->triangle_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            this->triangle_ = NULL;
        }

        free_hip(&this->buffer_);
        this->buffer_size_ = 0;
    }

    template class HIPSpTRSV<float>;
    template class HIPSpTRSV<double>;
#ifdef SUPPORT_COMPLEX
    template class HIPSpTRSV<std::complex<float>>;
    template class HIPSpTRSV<std::complex<double>>;
#endif

    template <typename ValueType>
    HIPSpITSV<ValueType>::HIPSpITSV(void)
        : triangle_(NULL)
        , trans_(rocsparse_operation_none)
        , buffer_size_(0)
        , buffer_(NULL)
    {
    }

    template <typename ValueType>
    HIPSpITSV<ValueType>::~HIPSpITSV(void)
    {
        this->Clear();
    }

    template <typename ValueType>
    bool HIPSpITSV<ValueType>::IsAnalysed(void) const
    {
        return this->triangle_ != NULL;
    }

    template <typename ValueType>
    void HIPSpITSV<ValueType>::Analyse(rocsparse_handle            handle,
                                       rocsparse_const_spmat_descr mat,
                                       rocsparse_operation         trans,
                                       rocsparse_fill_mode         fill_mode,
                                       rocsparse_diag_type         diag_type)
    {
        assert(mat != NULL);

        this->Clear();

        rocsparse_format format;

        rocsparse_status status = rocsparse_spmat_get_format(mat, &format);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        if(format != rocsparse_format_csr)
        {
            LOG_INFO("HIPSpITSV: rocsparse_spitsv does not support this matrix format");
            FATAL_ERROR(__FILE__, __LINE__);
        }

        create_triangle_descr(mat, fill_mode, diag_type, &this->triangle_);

        this->trans_ = trans;

        const rocsparse_datatype datatype = rocalution_datatype_traits<ValueType>::value;

        // The analysis only depends on the matrix, empty vectors are sufficient
        rocsparse_dnvec_descr x;
        rocsparse_dnvec_descr y;

        status = rocsparse_create_dnvec_descr(&x, 0, NULL, datatype);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_create_dnvec_descr(&y, 0, NULL, datatype);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        const ValueType alpha    = static_cast<ValueType>(1);
        rocsparse_int   max_iter = 0;

        status = rocsparse_spitsv(handle,
                                  &max_iter,
                                  NULL,
                                  NULL,
                                  this->trans_,
                                  &alpha,
                                  this->triangle_,
                                  x,
                                  y,
                                  datatype,
                                  rocsparse_spitsv_alg_default,
                                  rocsparse_spitsv_stage_buffer_size,
                                  &this->buffer_size_,
                                  NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        // The buffer is required by the solves as well
        allocate_hip(this->buffer_size_, &this->buffer_);

        status = rocsparse_spitsv(handle,
                                  &max_iter,
                                  NULL,
                                  NULL,
                                  this->trans_,
                                  &alpha,
                                  this->triangle_,
                                  x,
                                  y,
                                  datatype,
                                  rocsparse_spitsv_alg_default,
                                  rocsparse_spitsv_stage_preprocess,
                                  &this->buffer_size_,
                                  this->buffer_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_destroy_dnvec_descr(x);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_destroy_dnvec_descr(y);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    template <typename ValueType>
    void HIPSpITSV<ValueType>::Update(rocsparse_const_spmat_descr mat)
    {
        if(this->triangle_ == NULL)
        {
            return;
        }

        rocsparse_format format;
        int64_t          nrow, ncol, nnz;
        int64_t          nrow_T, ncol_T, nnz_T;

        rocsparse_status status = rocsparse_spmat_get_format(mat, &format);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_spmat_get_size(mat, &nrow, &ncol, &nnz);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_spmat_get_size(this->triangle_, &nrow_T, &ncol_T, &nnz_T);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        if(format != rocsparse_format_csr || nrow != nrow_T || ncol != ncol_T || nnz != nnz_T)
        {
            this->Clear();
            return;
        }

        const void*          row;
        const void*          col;
        const void*          val;
        rocsparse_indextype  row_type;
        rocsparse_indextype  col_type;
        rocsparse_index_base base;
        rocsparse_datatype   data_type;

        status = rocsparse_const_csr_get(
            mat, &nrow, &ncol, &nnz, &row, &col, &val, &row_type, &col_type, &base, &data_type);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        // Recreating the triangle would discard the analysis
        status = rocsparse_csr_set_pointers(this->triangle_,
                                            const_cast<void*>(row),
                                            const_cast<void*>(col),
                                            const_cast<void*>(val));
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    template <typename ValueType>
    void HIPSpITSV<ValueType>::Solve(rocsparse_handle      handle,
                                     int                   max_iter,
                                     double                tolerance,
                                     bool                  use_tol,
                                     ValueType             alpha,
                                     rocsparse_dnvec_descr x,
                                     rocsparse_dnvec_descr y) const
    {
        assert(this->triangle_ != NULL);

        const numeric_traits_t<ValueType> tol = static_cast<numeric_traits_t<ValueType>>(tolerance);

        rocsparse_int niter = max_iter;

        rocsparse_status status = rocsparse_spitsv(handle,
                                                   &niter,
                                                   use_tol ? &tol : NULL,
                                                   NULL,
                                                   this->trans_,
                                                   &alpha,
                                                   this->triangle_,
                                                   x,
                                                   y,
                                                   rocalution_datatype_traits<ValueType>::value,
                                                   rocsparse_spitsv_alg_default,
                                                   rocsparse_spitsv_stage_compute,
                                                   NULL,
                                                   this->buffer_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    template <typename ValueType>
    void HIPSpITSV<ValueType>::Clear(void)
    {
        if(this->triangle_ != NULL)
        {
            rocsparse_status status = rocsparse_destroy_spmat_descr(this->triangle_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            this->triangle_ = NULL;
        }

        free_hip(&this->buffer_);
        this->buffer_size_ = 0;
    }

    template class HIPSpITSV<float>;
    template class HIPSpITSV<double>;
#ifdef SUPPORT_COMPLEX
    template class HIPSpITSV<std::complex<float>>;
    template class HIPSpITSV<std::complex<double>>;
#endif

    template <typename ValueType>
    void HIPSpILU0(rocsparse_handle handle, rocsparse_spmat_descr mat)
    {
        assert(mat != NULL);

        const rocsparse_spilu0_alg      alg      = rocsparse_spilu0_alg_default;
        const rocsparse_analysis_policy policy   = rocsparse_analysis_policy_force;
        const rocsparse_datatype        datatype = rocalution_datatype_traits<ValueType>::value;

        rocsparse_spilu0_descr descr;

        rocsparse_status status = rocsparse_spilu0_descr_create(handle, &descr, NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_spilu0_set_input(
            handle, descr, rocsparse_spilu0_input_alg, &alg, sizeof(alg), NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_spilu0_set_input(
            handle, descr, rocsparse_spilu0_input_analysis_policy, &policy, sizeof(policy), NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_spilu0_set_input(handle,
                                            descr,
                                            rocsparse_spilu0_input_compute_datatype,
                                            &datatype,
                                            sizeof(datatype),
                                            NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        size_t buffer_size = 0;
        char*  buffer      = NULL;

        status = rocsparse_spilu0_buffer_size(
            handle, descr, mat, mat, rocsparse_spilu0_stage_analysis, &buffer_size, NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        allocate_hip(buffer_size, &buffer);

        status = rocsparse_spilu0(
            handle, descr, mat, mat, rocsparse_spilu0_stage_analysis, buffer_size, buffer, NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        free_hip(&buffer);

        status = rocsparse_spilu0_buffer_size(
            handle, descr, mat, mat, rocsparse_spilu0_stage_compute, &buffer_size, NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        allocate_hip(buffer_size, &buffer);

        status = rocsparse_spilu0(
            handle, descr, mat, mat, rocsparse_spilu0_stage_compute, buffer_size, buffer, NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        free_hip(&buffer);

        status = rocsparse_spilu0_descr_destroy(handle, descr, NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    template <typename ValueType>
    void HIPSpIC0(rocsparse_handle handle, rocsparse_spmat_descr mat)
    {
        assert(mat != NULL);

        const rocsparse_spic0_alg       alg      = rocsparse_spic0_alg_default;
        const rocsparse_analysis_policy policy   = rocsparse_analysis_policy_force;
        const rocsparse_datatype        datatype = rocalution_datatype_traits<ValueType>::value;

        rocsparse_spic0_descr descr;

        rocsparse_status status = rocsparse_spic0_descr_create(handle, &descr, NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_spic0_set_input(
            handle, descr, rocsparse_spic0_input_alg, &alg, sizeof(alg), NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_spic0_set_input(
            handle, descr, rocsparse_spic0_input_analysis_policy, &policy, sizeof(policy), NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_spic0_set_input(handle,
                                           descr,
                                           rocsparse_spic0_input_compute_datatype,
                                           &datatype,
                                           sizeof(datatype),
                                           NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        size_t buffer_size = 0;
        char*  buffer      = NULL;

        status = rocsparse_spic0_buffer_size(
            handle, descr, mat, mat, rocsparse_spic0_stage_analysis, &buffer_size, NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        allocate_hip(buffer_size, &buffer);

        status = rocsparse_spic0(
            handle, descr, mat, mat, rocsparse_spic0_stage_analysis, buffer_size, buffer, NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        free_hip(&buffer);

        status = rocsparse_spic0_buffer_size(
            handle, descr, mat, mat, rocsparse_spic0_stage_compute, &buffer_size, NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        allocate_hip(buffer_size, &buffer);

        status = rocsparse_spic0(
            handle, descr, mat, mat, rocsparse_spic0_stage_compute, buffer_size, buffer, NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        free_hip(&buffer);

        status = rocsparse_spic0_descr_destroy(handle, descr, NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    template void HIPSpILU0<float>(rocsparse_handle handle, rocsparse_spmat_descr mat);
    template void HIPSpILU0<double>(rocsparse_handle handle, rocsparse_spmat_descr mat);
    template void HIPSpIC0<float>(rocsparse_handle handle, rocsparse_spmat_descr mat);
    template void HIPSpIC0<double>(rocsparse_handle handle, rocsparse_spmat_descr mat);
#ifdef SUPPORT_COMPLEX
    template void HIPSpILU0<std::complex<float>>(rocsparse_handle      handle,
                                                 rocsparse_spmat_descr mat);
    template void HIPSpILU0<std::complex<double>>(rocsparse_handle      handle,
                                                  rocsparse_spmat_descr mat);
    template void HIPSpIC0<std::complex<float>>(rocsparse_handle handle, rocsparse_spmat_descr mat);
    template void HIPSpIC0<std::complex<double>>(rocsparse_handle      handle,
                                                 rocsparse_spmat_descr mat);
#endif

    void HIPSpSort(rocsparse_handle handle, rocsparse_spmat_descr mat)
    {
        assert(mat != NULL);

        const rocsparse_spsort_alg alg = rocsparse_spsort_alg_default;
        const rocsparse_direction  dir = rocsparse_direction_row;

        rocsparse_spsort_descr descr;

        rocsparse_status status = rocsparse_spsort_descr_create(handle, &descr, NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_spsort_set_input(
            handle, descr, rocsparse_spsort_input_alg, &alg, sizeof(alg), NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        // The direction is only used for COO matrices
        status = rocsparse_spsort_set_input(
            handle, descr, rocsparse_spsort_input_direction, &dir, sizeof(dir), NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        size_t buffer_size = 0;
        char*  buffer      = NULL;

        status = rocsparse_spsort_buffer_size(
            handle, descr, mat, mat, rocsparse_spsort_stage_analysis, &buffer_size, NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        allocate_hip(buffer_size, &buffer);

        status = rocsparse_spsort(
            handle, descr, mat, mat, rocsparse_spsort_stage_analysis, buffer_size, buffer, NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        free_hip(&buffer);

        status = rocsparse_spsort_buffer_size(
            handle, descr, mat, mat, rocsparse_spsort_stage_compute, &buffer_size, NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        allocate_hip(buffer_size, &buffer);

        status = rocsparse_spsort(
            handle, descr, mat, mat, rocsparse_spsort_stage_compute, buffer_size, buffer, NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        free_hip(&buffer);

        status = rocsparse_spsort_descr_destroy(handle, descr, NULL);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
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

    template <>
    rocsparse_status rocsparseTcsritilu0_compute_ex(rocsparse_handle     handle,
                                                    rocsparse_itilu0_alg alg,
                                                    rocsparse_int        option,
                                                    rocsparse_int*       nmaxiter,
                                                    rocsparse_int        nfreeiter,
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
        return rocsparse_scsritilu0_compute_ex(handle,
                                               alg,
                                               option,
                                               nmaxiter,
                                               nfreeiter,
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
    rocsparse_status rocsparseTcsritilu0_compute_ex(rocsparse_handle     handle,
                                                    rocsparse_itilu0_alg alg,
                                                    rocsparse_int        option,
                                                    rocsparse_int*       nmaxiter,
                                                    rocsparse_int        nfreeiter,
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
        return rocsparse_dcsritilu0_compute_ex(handle,
                                               alg,
                                               option,
                                               nmaxiter,
                                               nfreeiter,
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
    rocsparse_status rocsparseTcsritilu0_compute_ex(rocsparse_handle           handle,
                                                    rocsparse_itilu0_alg       alg,
                                                    rocsparse_int              option,
                                                    rocsparse_int*             nmaxiter,
                                                    rocsparse_int              nfreeiter,
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
        return rocsparse_ccsritilu0_compute_ex(handle,
                                               alg,
                                               option,
                                               nmaxiter,
                                               nfreeiter,
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
    rocsparse_status rocsparseTcsritilu0_compute_ex(rocsparse_handle            handle,
                                                    rocsparse_itilu0_alg        alg,
                                                    rocsparse_int               option,
                                                    rocsparse_int*              nmaxiter,
                                                    rocsparse_int               nfreeiter,
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
        return rocsparse_zcsritilu0_compute_ex(handle,
                                               alg,
                                               option,
                                               nmaxiter,
                                               nfreeiter,
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

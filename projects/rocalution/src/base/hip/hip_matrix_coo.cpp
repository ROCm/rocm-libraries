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

#include "hip_matrix_coo.hpp"
#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "../backend_manager.hpp"
#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../host/host_matrix_coo.hpp"
#include "../matrix_formats_ind.hpp"
#include "hip_allocate_free.hpp"
#include "hip_conversion.hpp"
#include "hip_kernels_coo.hpp"
#include "hip_kernels_general.hpp"
#include "hip_matrix_csr.hpp"
#include "hip_sparse.hpp"
#include "hip_utils.hpp"
#include "hip_vector.hpp"

#include <algorithm>
#include <hip/hip_runtime.h>
#include <rocsparse/rocsparse.h>

namespace rocalution
{

    template <typename ValueType>
    HIPAcceleratorMatrixCOO<ValueType>::HIPAcceleratorMatrixCOO()
    {
        // no default constructors
        LOG_INFO("no default constructor");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    HIPAcceleratorMatrixCOO<ValueType>::HIPAcceleratorMatrixCOO(
        const Rocalution_Backend_Descriptor& local_backend)
    {
        log_debug(this,
                  "HIPAcceleratorMatrixCOO::HIPAcceleratorMatrixCOO()",
                  "constructor with local_backend");

        this->mat_.row = NULL;
        this->mat_.col = NULL;
        this->mat_.val = NULL;
        this->set_backend(local_backend);

        this->spmat_descr_ = 0;

        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCOO<ValueType>::CreateSpMatDescr_(void)
    {
        this->DestroySpMatDescr_();

        rocsparse_status status
            = rocsparse_create_coo_descr(&this->spmat_descr_,
                                         this->nrow_,
                                         this->ncol_,
                                         this->nnz_,
                                         this->mat_.row,
                                         this->mat_.col,
                                         this->mat_.val,
                                         rocsparse_indextype_i32,
                                         rocsparse_index_base_zero,
                                         rocalution_datatype_traits<ValueType>::value);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCOO<ValueType>::DestroySpMatDescr_(void)
    {
        if(this->spmat_descr_ != NULL)
        {
            rocsparse_status status = rocsparse_destroy_spmat_descr(this->spmat_descr_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            this->spmat_descr_ = NULL;
        }

        this->ApplyAnalyseClear_();
    }

    template <typename ValueType>
    HIPAcceleratorMatrixCOO<ValueType>::~HIPAcceleratorMatrixCOO()
    {
        log_debug(this, "HIPAcceleratorMatrixCOO::~HIPAcceleratorMatrixCOO()", "destructor");

        this->Clear();
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCOO<ValueType>::Info(void) const
    {
        LOG_INFO("HIPAcceleratorMatrixCOO<ValueType>");
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCOO<ValueType>::AllocateCOO(int64_t nnz, int nrow, int ncol)
    {
        assert(nnz >= 0);
        assert(ncol >= 0);
        assert(nrow >= 0);

        this->Clear();

        allocate_hip(nnz, &this->mat_.row);
        allocate_hip(nnz, &this->mat_.col);
        allocate_hip(nnz, &this->mat_.val);

        set_to_zero_hip(this->local_backend_.HIP_block_size, nnz, this->mat_.row);
        set_to_zero_hip(this->local_backend_.HIP_block_size, nnz, this->mat_.col);
        set_to_zero_hip(this->local_backend_.HIP_block_size, nnz, this->mat_.val);

        this->nrow_ = nrow;
        this->ncol_ = ncol;
        this->nnz_  = nnz;

        this->CreateSpMatDescr_();
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCOO<ValueType>::SetDataPtrCOO(
        int** row, int** col, ValueType** val, int64_t nnz, int nrow, int ncol)
    {
        assert(nnz >= 0);
        assert(nrow >= 0);
        assert(ncol >= 0);

        if(nnz > 0)
        {
            assert(*row != NULL);
            assert(*col != NULL);
            assert(*val != NULL);
        }

        this->Clear();

        this->nrow_ = nrow;
        this->ncol_ = ncol;
        this->nnz_  = nnz;

        CHECK_HIP_STATUS(hipDeviceSynchronize());

        this->mat_.row = *row;
        this->mat_.col = *col;
        this->mat_.val = *val;

        this->CreateSpMatDescr_();
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCOO<ValueType>::LeaveDataPtrCOO(int** row, int** col, ValueType** val)
    {
        assert(this->nrow_ >= 0);
        assert(this->ncol_ >= 0);
        assert(this->nnz_ >= 0);

        CHECK_HIP_STATUS(hipDeviceSynchronize());

        this->DestroySpMatDescr_();

        // see free_host function for details
        *row = this->mat_.row;
        *col = this->mat_.col;
        *val = this->mat_.val;

        this->mat_.row = NULL;
        this->mat_.col = NULL;
        this->mat_.val = NULL;

        this->nrow_ = 0;
        this->ncol_ = 0;
        this->nnz_  = 0;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCOO<ValueType>::Clear()
    {
        this->DestroySpMatDescr_();

        free_hip(&this->mat_.row);
        free_hip(&this->mat_.col);
        free_hip(&this->mat_.val);

        this->nrow_ = 0;
        this->ncol_ = 0;
        this->nnz_  = 0;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCOO<ValueType>::CopyFromHost(const HostMatrix<ValueType>& src)
    {
        const HostMatrixCOO<ValueType>* cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == src.GetMatFormat());

        // CPU to HIP copy
        if((cast_mat = dynamic_cast<const HostMatrixCOO<ValueType>*>(&src)) != NULL)
        {
            if(this->nnz_ == 0)
            {
                this->AllocateCOO(cast_mat->nnz_, cast_mat->nrow_, cast_mat->ncol_);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

            copy_h2d(this->nnz_, cast_mat->mat_.row, this->mat_.row);
            copy_h2d(this->nnz_, cast_mat->mat_.col, this->mat_.col);
            copy_h2d(this->nnz_, cast_mat->mat_.val, this->mat_.val);

            this->ApplyAnalyseClear_();
        }
        else
        {
            LOG_INFO("Error unsupported HIP matrix type");
            this->Info();
            src.Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCOO<ValueType>::CopyToHost(HostMatrix<ValueType>* dst) const
    {
        HostMatrixCOO<ValueType>* cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == dst->GetMatFormat());

        // HIP to CPU copy
        if((cast_mat = dynamic_cast<HostMatrixCOO<ValueType>*>(dst)) != NULL)
        {
            cast_mat->set_backend(this->local_backend_);

            if(cast_mat->nnz_ == 0)
            {
                cast_mat->AllocateCOO(this->nnz_, this->nrow_, this->ncol_);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

            copy_d2h(this->nnz_, this->mat_.row, cast_mat->mat_.row);
            copy_d2h(this->nnz_, this->mat_.col, cast_mat->mat_.col);
            copy_d2h(this->nnz_, this->mat_.val, cast_mat->mat_.val);
        }
        else
        {
            LOG_INFO("Error unsupported HIP matrix type");
            this->Info();
            dst->Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCOO<ValueType>::CopyFrom(const BaseMatrix<ValueType>& src)
    {
        const HIPAcceleratorMatrixCOO<ValueType>* hip_cast_mat;
        const HostMatrix<ValueType>*              host_cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == src.GetMatFormat());

        // HIP to HIP copy
        if((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixCOO<ValueType>*>(&src)) != NULL)
        {
            if(this->nnz_ == 0)
            {
                this->AllocateCOO(hip_cast_mat->nnz_, hip_cast_mat->nrow_, hip_cast_mat->ncol_);
            }

            assert(this->nnz_ == hip_cast_mat->nnz_);
            assert(this->nrow_ == hip_cast_mat->nrow_);
            assert(this->ncol_ == hip_cast_mat->ncol_);

            copy_d2d(this->nnz_, hip_cast_mat->mat_.row, this->mat_.row);
            copy_d2d(this->nnz_, hip_cast_mat->mat_.col, this->mat_.col);
            copy_d2d(this->nnz_, hip_cast_mat->mat_.val, this->mat_.val);

            this->ApplyAnalyseClear_();
        }
        else
        {
            // CPU to HIP
            if((host_cast_mat = dynamic_cast<const HostMatrix<ValueType>*>(&src)) != NULL)
            {
                this->CopyFromHost(*host_cast_mat);
            }
            else
            {
                LOG_INFO("Error unsupported HIP matrix type");
                this->Info();
                src.Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }
        }
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCOO<ValueType>::CopyTo(BaseMatrix<ValueType>* dst) const
    {
        HIPAcceleratorMatrixCOO<ValueType>* hip_cast_mat;
        HostMatrix<ValueType>*              host_cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == dst->GetMatFormat());

        // HIP to HIP copy
        if((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixCOO<ValueType>*>(dst)) != NULL)
        {
            hip_cast_mat->set_backend(this->local_backend_);

            if(hip_cast_mat->nnz_ == 0)
            {
                hip_cast_mat->AllocateCOO(this->nnz_, this->nrow_, this->ncol_);
            }

            assert(this->nnz_ == hip_cast_mat->nnz_);
            assert(this->nrow_ == hip_cast_mat->nrow_);
            assert(this->ncol_ == hip_cast_mat->ncol_);

            copy_d2d(this->nnz_, this->mat_.row, hip_cast_mat->mat_.row);
            copy_d2d(this->nnz_, this->mat_.col, hip_cast_mat->mat_.col);
            copy_d2d(this->nnz_, this->mat_.val, hip_cast_mat->mat_.val);

            hip_cast_mat->ApplyAnalyseClear_();
        }
        else
        {
            // HIP to CPU
            if((host_cast_mat = dynamic_cast<HostMatrix<ValueType>*>(dst)) != NULL)
            {
                this->CopyToHost(host_cast_mat);
            }
            else
            {
                LOG_INFO("Error unsupported HIP matrix type");
                this->Info();
                dst->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }
        }
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCOO<ValueType>::CopyFromHostAsync(const HostMatrix<ValueType>& src)
    {
        const HostMatrixCOO<ValueType>* cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == src.GetMatFormat());

        // CPU to HIP copy
        if((cast_mat = dynamic_cast<const HostMatrixCOO<ValueType>*>(&src)) != NULL)
        {
            if(this->nnz_ == 0)
            {
                this->AllocateCOO(cast_mat->nnz_, cast_mat->nrow_, cast_mat->ncol_);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

            copy_h2d(this->nnz_,
                     cast_mat->mat_.row,
                     this->mat_.row,
                     true,
                     HIPSTREAM(_get_backend_descriptor()->HIP_stream_current));
            copy_h2d(this->nnz_,
                     cast_mat->mat_.col,
                     this->mat_.col,
                     true,
                     HIPSTREAM(_get_backend_descriptor()->HIP_stream_current));
            copy_h2d(this->nnz_,
                     cast_mat->mat_.val,
                     this->mat_.val,
                     true,
                     HIPSTREAM(_get_backend_descriptor()->HIP_stream_current));

            this->ApplyAnalyseClear_();
        }
        else
        {
            LOG_INFO("Error unsupported HIP matrix type");
            this->Info();
            src.Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCOO<ValueType>::CopyToHostAsync(HostMatrix<ValueType>* dst) const
    {
        HostMatrixCOO<ValueType>* cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == dst->GetMatFormat());

        // HIP to CPU copy
        if((cast_mat = dynamic_cast<HostMatrixCOO<ValueType>*>(dst)) != NULL)
        {
            cast_mat->set_backend(this->local_backend_);

            if(cast_mat->nnz_ == 0)
            {
                cast_mat->AllocateCOO(this->nnz_, this->nrow_, this->ncol_);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

            copy_d2h(this->nnz_,
                     this->mat_.row,
                     cast_mat->mat_.row,
                     true,
                     HIPSTREAM(_get_backend_descriptor()->HIP_stream_current));
            copy_d2h(this->nnz_,
                     this->mat_.col,
                     cast_mat->mat_.col,
                     true,
                     HIPSTREAM(_get_backend_descriptor()->HIP_stream_current));
            copy_d2h(this->nnz_,
                     this->mat_.val,
                     cast_mat->mat_.val,
                     true,
                     HIPSTREAM(_get_backend_descriptor()->HIP_stream_current));
        }
        else
        {
            LOG_INFO("Error unsupported HIP matrix type");
            this->Info();
            dst->Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCOO<ValueType>::CopyFromAsync(const BaseMatrix<ValueType>& src)
    {
        const HIPAcceleratorMatrixCOO<ValueType>* hip_cast_mat;
        const HostMatrix<ValueType>*              host_cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == src.GetMatFormat());

        // HIP to HIP copy
        if((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixCOO<ValueType>*>(&src)) != NULL)
        {
            if(this->nnz_ == 0)
            {
                this->AllocateCOO(hip_cast_mat->nnz_, hip_cast_mat->nrow_, hip_cast_mat->ncol_);
            }

            assert(this->nnz_ == hip_cast_mat->nnz_);
            assert(this->nrow_ == hip_cast_mat->nrow_);
            assert(this->ncol_ == hip_cast_mat->ncol_);

            copy_d2d(this->nnz_,
                     hip_cast_mat->mat_.row,
                     this->mat_.row,
                     true,
                     HIPSTREAM(_get_backend_descriptor()->HIP_stream_current));
            copy_d2d(this->nnz_,
                     hip_cast_mat->mat_.col,
                     this->mat_.col,
                     true,
                     HIPSTREAM(_get_backend_descriptor()->HIP_stream_current));
            copy_d2d(this->nnz_,
                     hip_cast_mat->mat_.val,
                     this->mat_.val,
                     true,
                     HIPSTREAM(_get_backend_descriptor()->HIP_stream_current));

            this->ApplyAnalyseClear_();
        }
        else
        {
            // CPU to HIP
            if((host_cast_mat = dynamic_cast<const HostMatrix<ValueType>*>(&src)) != NULL)
            {
                this->CopyFromHostAsync(*host_cast_mat);
            }
            else
            {
                LOG_INFO("Error unsupported HIP matrix type");
                this->Info();
                src.Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }
        }
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCOO<ValueType>::CopyToAsync(BaseMatrix<ValueType>* dst) const
    {
        HIPAcceleratorMatrixCOO<ValueType>* hip_cast_mat;
        HostMatrix<ValueType>*              host_cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == dst->GetMatFormat());

        // HIP to HIP copy
        if((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixCOO<ValueType>*>(dst)) != NULL)
        {
            hip_cast_mat->set_backend(this->local_backend_);

            if(hip_cast_mat->nnz_ == 0)
            {
                hip_cast_mat->AllocateCOO(this->nnz_, this->nrow_, this->ncol_);
            }

            assert(this->nnz_ == hip_cast_mat->nnz_);
            assert(this->nrow_ == hip_cast_mat->nrow_);
            assert(this->ncol_ == hip_cast_mat->ncol_);

            copy_d2h(this->nnz_,
                     this->mat_.row,
                     hip_cast_mat->mat_.row,
                     true,
                     HIPSTREAM(_get_backend_descriptor()->HIP_stream_current));
            copy_d2h(this->nnz_,
                     this->mat_.col,
                     hip_cast_mat->mat_.col,
                     true,
                     HIPSTREAM(_get_backend_descriptor()->HIP_stream_current));
            copy_d2h(this->nnz_,
                     this->mat_.val,
                     hip_cast_mat->mat_.val,
                     true,
                     HIPSTREAM(_get_backend_descriptor()->HIP_stream_current));

            hip_cast_mat->ApplyAnalyseClear_();
        }
        else
        {
            // HIP to CPU
            if((host_cast_mat = dynamic_cast<HostMatrix<ValueType>*>(dst)) != NULL)
            {
                this->CopyToHostAsync(host_cast_mat);
            }
            else
            {
                LOG_INFO("Error unsupported HIP matrix type");
                this->Info();
                dst->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }
        }
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCOO<ValueType>::CopyFromCOO(const int*       row,
                                                         const int*       col,
                                                         const ValueType* val)
    {
        copy_d2d(this->nnz_, row, this->mat_.row);
        copy_d2d(this->nnz_, col, this->mat_.col);
        copy_d2d(this->nnz_, val, this->mat_.val);

        this->ApplyAnalyseClear_();
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCOO<ValueType>::CopyToCOO(int* row, int* col, ValueType* val) const
    {
        copy_d2d(this->nnz_, this->mat_.row, row);
        copy_d2d(this->nnz_, this->mat_.col, col);
        copy_d2d(this->nnz_, this->mat_.val, val);
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCOO<ValueType>::ConvertFrom(const BaseMatrix<ValueType>& mat)
    {
        this->Clear();

        // empty matrix is empty matrix
        if(mat.GetNnz() == 0)
        {
            this->AllocateCOO(mat.GetNnz(), mat.GetM(), mat.GetN());

            return true;
        }

        const HIPAcceleratorMatrixCOO<ValueType>* cast_mat_coo;

        if((cast_mat_coo = dynamic_cast<const HIPAcceleratorMatrixCOO<ValueType>*>(&mat)) != NULL)
        {
            this->CopyFrom(*cast_mat_coo);
            return true;
        }

        const HIPAcceleratorMatrixCSR<ValueType>* cast_mat_csr;

        if((cast_mat_csr = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*>(&mat)) != NULL)
        {
            this->Clear();

            if(csr_to_coo_hip(&this->local_backend_,
                              cast_mat_csr->nnz_,
                              cast_mat_csr->nrow_,
                              cast_mat_csr->ncol_,
                              cast_mat_csr->mat_,
                              &this->mat_)
               == true)
            {
                this->nrow_ = cast_mat_csr->nrow_;
                this->ncol_ = cast_mat_csr->ncol_;
                this->nnz_  = cast_mat_csr->nnz_;

                this->CreateSpMatDescr_();

                return true;
            }
        }

        return false;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCOO<ValueType>::ApplyAnalyse(void) const
    {
        if(this->nnz_ > 0 && this->spmv_.IsAnalysed() == false)
        {
            this->spmv_.Analyse(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                rocsparse_spmv_alg_coo,
                                this->spmat_descr_);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCOO<ValueType>::ApplyAnalyse_(ValueType                   alpha,
                                                           rocsparse_const_dnvec_descr x,
                                                           ValueType                   beta,
                                                           rocsparse_dnvec_descr       y) const
    {
        if(this->spmv_.IsAnalysed() == false)
        {
            LOG_VERBOSE_INFO(2,
                             "*** warning: HIPAcceleratorMatrixCOO performs the SpMV analysis "
                             "lazily, call ApplyAnalyse() beforehand to avoid this");

            this->spmv_.Analyse(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                rocsparse_spmv_alg_coo,
                                this->spmat_descr_,
                                alpha,
                                x,
                                beta,
                                y);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCOO<ValueType>::ApplyAnalyseClear_(void)
    {
        this->spmv_.Clear();
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCOO<ValueType>::Apply(const BaseVector<ValueType>& in,
                                                   BaseVector<ValueType>*       out) const
    {
        if(this->nnz_ > 0)
        {
            assert(in.GetSize() >= 0);
            assert(out->GetSize() >= 0);
            assert(in.GetSize() == this->ncol_);
            assert(out->GetSize() == this->nrow_);

            const HIPAcceleratorVector<ValueType>* cast_in
                = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&in);
            HIPAcceleratorVector<ValueType>* cast_out
                = dynamic_cast<HIPAcceleratorVector<ValueType>*>(out);

            assert(cast_in != NULL);
            assert(cast_out != NULL);

            ValueType alpha = static_cast<ValueType>(1);
            ValueType beta  = static_cast<ValueType>(0);

            // Lazy matrix analyse
            this->ApplyAnalyse_(alpha, cast_in->dnvec_descr_, beta, cast_out->dnvec_descr_);

            this->spmv_.Compute(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                alpha,
                                this->spmat_descr_,
                                cast_in->dnvec_descr_,
                                beta,
                                cast_out->dnvec_descr_);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixCOO<ValueType>::ApplyAdd(const BaseVector<ValueType>& in,
                                                      ValueType                    scalar,
                                                      BaseVector<ValueType>*       out) const
    {
        if(this->nnz_ > 0)
        {
            assert(in.GetSize() >= 0);
            assert(out->GetSize() >= 0);
            assert(in.GetSize() == this->ncol_);
            assert(out->GetSize() == this->nrow_);

            const HIPAcceleratorVector<ValueType>* cast_in
                = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&in);
            HIPAcceleratorVector<ValueType>* cast_out
                = dynamic_cast<HIPAcceleratorVector<ValueType>*>(out);

            assert(cast_in != NULL);
            assert(cast_out != NULL);

            ValueType beta = static_cast<ValueType>(1);

            // Lazy matrix analyse
            this->ApplyAnalyse_(scalar, cast_in->dnvec_descr_, beta, cast_out->dnvec_descr_);

            this->spmv_.Compute(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                scalar,
                                this->spmat_descr_,
                                cast_in->dnvec_descr_,
                                beta,
                                cast_out->dnvec_descr_);
        }
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCOO<ValueType>::Permute(const BaseVector<int>& permutation)
    {
        // symmetric permutation only
        assert(permutation.GetSize() == this->nrow_);
        assert(permutation.GetSize() == this->ncol_);

        if(this->nnz_ > 0)
        {
            const HIPAcceleratorVector<int>* cast_perm
                = dynamic_cast<const HIPAcceleratorVector<int>*>(&permutation);
            assert(cast_perm != NULL);

            HIPAcceleratorMatrixCOO<ValueType> src(this->local_backend_);
            src.AllocateCOO(this->nnz_, this->nrow_, this->ncol_);
            src.CopyFrom(*this);

            int64_t nnz = this->nnz_;
            int64_t s   = nnz;
            int64_t k
                = (nnz / this->local_backend_.HIP_block_size) / this->local_backend_.HIP_max_threads
                  + 1;
            if(k > 1)
            {
                s = nnz / k;
            }

            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(s / this->local_backend_.HIP_block_size + 1);

            kernel_coo_permute<ValueType, int>
                <<<GridSize,
                   BlockSize,
                   0,
                   HIPSTREAM(_get_backend_descriptor()->HIP_stream_current)>>>(nnz,
                                                                               src.mat_.row,
                                                                               src.mat_.col,
                                                                               cast_perm->vec_,
                                                                               this->mat_.row,
                                                                               this->mat_.col);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            this->ApplyAnalyseClear_();
        }

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCOO<ValueType>::PermuteBackward(const BaseVector<int>& permutation)
    {
        // symmetric permutation only
        assert(permutation.GetSize() == this->nrow_);
        assert(permutation.GetSize() == this->ncol_);

        if(this->nnz_ > 0)
        {
            const HIPAcceleratorVector<int>* cast_perm
                = dynamic_cast<const HIPAcceleratorVector<int>*>(&permutation);

            assert(cast_perm != NULL);

            int* pb = NULL;
            allocate_hip(this->nrow_, &pb);

            int  n = this->nrow_;
            dim3 BlockSize1(this->local_backend_.HIP_block_size);
            dim3 GridSize1(n / this->local_backend_.HIP_block_size + 1);

            kernel_reverse_index<<<GridSize1,
                                   BlockSize1,
                                   0,
                                   HIPSTREAM(_get_backend_descriptor()->HIP_stream_current)>>>(
                n, cast_perm->vec_, pb);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            HIPAcceleratorMatrixCOO<ValueType> src(this->local_backend_);
            src.AllocateCOO(this->nnz_, this->nrow_, this->ncol_);
            src.CopyFrom(*this);

            int64_t nnz = this->nnz_;
            int64_t s   = nnz;
            int64_t k
                = (nnz / this->local_backend_.HIP_block_size) / this->local_backend_.HIP_max_threads
                  + 1;
            if(k > 1)
            {
                s = nnz / k;
            }

            dim3 BlockSize2(this->local_backend_.HIP_block_size);
            dim3 GridSize2(s / this->local_backend_.HIP_block_size + 1);

            kernel_coo_permute<ValueType, int>
                <<<GridSize2,
                   BlockSize2,
                   0,
                   HIPSTREAM(_get_backend_descriptor()->HIP_stream_current)>>>(
                    nnz, src.mat_.row, src.mat_.col, pb, this->mat_.row, this->mat_.col);
            CHECK_HIP_ERROR(__FILE__, __LINE__);

            free_hip(&pb);

            this->ApplyAnalyseClear_();
        }

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixCOO<ValueType>::Sort(void)
    {
        if(this->nnz_ > 0)
        {
            HIPSpSort(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle), this->spmat_descr_);

            this->ApplyAnalyseClear_();
        }

        return true;
    }

    template class HIPAcceleratorMatrixCOO<double>;
    template class HIPAcceleratorMatrixCOO<float>;
#ifdef SUPPORT_COMPLEX
    template class HIPAcceleratorMatrixCOO<std::complex<double>>;
    template class HIPAcceleratorMatrixCOO<std::complex<float>>;
#endif

} // namespace rocalution

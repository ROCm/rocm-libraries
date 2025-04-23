/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "hip_matrix_ell.hpp"
#include "../../utils/allocate_free.hpp"
#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "../backend_manager.hpp"
#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../host/host_matrix_ell.hpp"
#include "../matrix_formats_ind.hpp"
#include "hip_allocate_free.hpp"
#include "hip_conversion.hpp"
#include "hip_matrix_csr.hpp"
#include "hip_sparse.hpp"
#include "hip_utils.hpp"
#include "hip_vector.hpp"

#include <hip/hip_runtime.h>

namespace rocalution
{

    template <typename ValueType>
    HIPAcceleratorMatrixELL<ValueType>::HIPAcceleratorMatrixELL()
    {
        // no default constructors
        LOG_INFO("no default constructor");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    HIPAcceleratorMatrixELL<ValueType>::HIPAcceleratorMatrixELL(
        const Rocalution_Backend_Descriptor& local_backend)
    {
        log_debug(this,
                  "HIPAcceleratorMatrixELL::HIPAcceleratorMatrixELL()",
                  "constructor with local_backend");

        this->mat_.val     = NULL;
        this->mat_.col     = NULL;
        this->mat_.max_row = 0;
        this->set_backend(local_backend);

        this->mat_descr_ = 0;

        CHECK_HIP_ERROR(__FILE__, __LINE__);

        rocsparse_status status;

        status = rocsparse_create_mat_descr(&this->mat_descr_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_index_base(this->mat_descr_, rocsparse_index_base_zero);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_type(this->mat_descr_, rocsparse_matrix_type_general);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    template <typename ValueType>
    HIPAcceleratorMatrixELL<ValueType>::~HIPAcceleratorMatrixELL()
    {
        log_debug(this, "HIPAcceleratorMatrixELL::~HIPAcceleratorMatrixELL()", "destructor");

        this->Clear();

        rocsparse_status status;

        status = rocsparse_destroy_mat_descr(this->mat_descr_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixELL<ValueType>::Info(void) const
    {
        LOG_INFO("HIPAcceleratorMatrixELL<ValueType>");
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixELL<ValueType>::AllocateELL(int64_t nnz,
                                                         int     nrow,
                                                         int     ncol,
                                                         int     max_row)
    {
        assert(nnz >= 0);
        assert(ncol >= 0);
        assert(nrow >= 0);
        assert(max_row >= 0);

        this->Clear();

        if(nnz > 0)
        {
            assert(nnz == max_row * nrow);
        }

        allocate_hip(nnz, &this->mat_.val);
        allocate_hip(nnz, &this->mat_.col);

        set_to_zero_hip(this->local_backend_.HIP_block_size, nnz, this->mat_.val);
        set_to_zero_hip(this->local_backend_.HIP_block_size, nnz, this->mat_.col);

        this->mat_.max_row = max_row;
        this->nrow_        = nrow;
        this->ncol_        = ncol;
        this->nnz_         = nnz;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixELL<ValueType>::Clear()
    {
        free_hip(&this->mat_.val);
        free_hip(&this->mat_.col);

        this->nrow_ = 0;
        this->ncol_ = 0;
        this->nnz_  = 0;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixELL<ValueType>::SetDataPtrELL(
        int** col, ValueType** val, int64_t nnz, int nrow, int ncol, int max_row)
    {
        assert(nnz >= 0);
        assert(nrow >= 0);
        assert(ncol >= 0);
        assert(max_row >= 0);
        assert(max_row * nrow == nnz);

        if(nnz > 0)
        {
            assert(*col != NULL);
            assert(*val != NULL);
        }

        this->Clear();

        hipDeviceSynchronize();

        this->mat_.max_row = max_row;
        this->nrow_        = nrow;
        this->ncol_        = ncol;
        this->nnz_         = nnz;

        this->mat_.col = *col;
        this->mat_.val = *val;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixELL<ValueType>::LeaveDataPtrELL(int**       col,
                                                             ValueType** val,
                                                             int&        max_row)
    {
        assert(this->nrow_ >= 0);
        assert(this->ncol_ >= 0);
        assert(this->nnz_ >= 0);
        assert(this->mat_.max_row >= 0);
        assert(this->mat_.max_row * this->nrow_ == this->nnz_);

        hipDeviceSynchronize();

        // see free_host function for details
        *col = this->mat_.col;
        *val = this->mat_.val;

        this->mat_.col = NULL;
        this->mat_.val = NULL;

        max_row = this->mat_.max_row;

        this->mat_.max_row = 0;
        this->nrow_        = 0;
        this->ncol_        = 0;
        this->nnz_         = 0;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixELL<ValueType>::CopyFromHost(const HostMatrix<ValueType>& src)
    {
        const HostMatrixELL<ValueType>* cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == src.GetMatFormat());

        // CPU to HIP copy
        if((cast_mat = dynamic_cast<const HostMatrixELL<ValueType>*>(&src)) != NULL)
        {
            if(this->nnz_ == 0)
            {
                this->AllocateELL(
                    cast_mat->nnz_, cast_mat->nrow_, cast_mat->ncol_, cast_mat->mat_.max_row);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

            copy_h2d(this->nnz_, cast_mat->mat_.col, this->mat_.col);
            copy_h2d(this->nnz_, cast_mat->mat_.val, this->mat_.val);
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
    void HIPAcceleratorMatrixELL<ValueType>::CopyToHost(HostMatrix<ValueType>* dst) const
    {
        HostMatrixELL<ValueType>* cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == dst->GetMatFormat());

        // HIP to CPU copy
        if((cast_mat = dynamic_cast<HostMatrixELL<ValueType>*>(dst)) != NULL)
        {
            cast_mat->set_backend(this->local_backend_);

            if(cast_mat->nnz_ == 0)
            {
                cast_mat->AllocateELL(this->nnz_, this->nrow_, this->ncol_, this->mat_.max_row);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

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
    void HIPAcceleratorMatrixELL<ValueType>::CopyFrom(const BaseMatrix<ValueType>& src)
    {
        const HIPAcceleratorMatrixELL<ValueType>* hip_cast_mat;
        const HostMatrix<ValueType>*              host_cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == src.GetMatFormat());

        // HIP to HIP copy
        if((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixELL<ValueType>*>(&src)) != NULL)
        {
            if(this->nnz_ == 0)
            {
                this->AllocateELL(hip_cast_mat->nnz_,
                                  hip_cast_mat->nrow_,
                                  hip_cast_mat->ncol_,
                                  hip_cast_mat->mat_.max_row);
            }

            assert(this->nnz_ == hip_cast_mat->nnz_);
            assert(this->nrow_ == hip_cast_mat->nrow_);
            assert(this->ncol_ == hip_cast_mat->ncol_);

            copy_d2d(this->nnz_, hip_cast_mat->mat_.col, this->mat_.col);
            copy_d2d(this->nnz_, hip_cast_mat->mat_.val, this->mat_.val);
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
    void HIPAcceleratorMatrixELL<ValueType>::CopyTo(BaseMatrix<ValueType>* dst) const
    {
        HIPAcceleratorMatrixELL<ValueType>* hip_cast_mat;
        HostMatrix<ValueType>*              host_cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == dst->GetMatFormat());

        // HIP to HIP copy
        if((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixELL<ValueType>*>(dst)) != NULL)
        {
            hip_cast_mat->set_backend(this->local_backend_);

            if(hip_cast_mat->nnz_ == 0)
            {
                hip_cast_mat->AllocateELL(this->nnz_, this->nrow_, this->ncol_, this->mat_.max_row);
            }

            assert(this->nnz_ == hip_cast_mat->nnz_);
            assert(this->nrow_ == hip_cast_mat->nrow_);
            assert(this->ncol_ == hip_cast_mat->ncol_);
            assert(this->mat_.max_row == hip_cast_mat->mat_.max_row);

            copy_d2d(this->nnz_, this->mat_.col, hip_cast_mat->mat_.col);
            copy_d2d(this->nnz_, this->mat_.val, hip_cast_mat->mat_.val);
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
    void HIPAcceleratorMatrixELL<ValueType>::CopyFromHostAsync(const HostMatrix<ValueType>& src)
    {
        const HostMatrixELL<ValueType>* cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == src.GetMatFormat());

        // CPU to HIP copy
        if((cast_mat = dynamic_cast<const HostMatrixELL<ValueType>*>(&src)) != NULL)
        {
            if(this->nnz_ == 0)
            {
                this->AllocateELL(
                    cast_mat->nnz_, cast_mat->nrow_, cast_mat->ncol_, cast_mat->mat_.max_row);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

            copy_h2d(this->nnz_,
                     cast_mat->mat_.col,
                     this->mat_.col,
                     true,
                     HIPSTREAM(this->local_backend_.HIP_stream_current));
            copy_h2d(this->nnz_,
                     cast_mat->mat_.val,
                     this->mat_.val,
                     true,
                     HIPSTREAM(this->local_backend_.HIP_stream_current));
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
    void HIPAcceleratorMatrixELL<ValueType>::CopyToHostAsync(HostMatrix<ValueType>* dst) const
    {
        HostMatrixELL<ValueType>* cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == dst->GetMatFormat());

        // HIP to CPU copy
        if((cast_mat = dynamic_cast<HostMatrixELL<ValueType>*>(dst)) != NULL)
        {
            cast_mat->set_backend(this->local_backend_);

            if(cast_mat->nnz_ == 0)
            {
                cast_mat->AllocateELL(this->nnz_, this->nrow_, this->ncol_, this->mat_.max_row);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

            copy_d2h(this->nnz_,
                     this->mat_.col,
                     cast_mat->mat_.col,
                     true,
                     HIPSTREAM(this->local_backend_.HIP_stream_current));
            copy_d2h(this->nnz_,
                     this->mat_.val,
                     cast_mat->mat_.val,
                     true,
                     HIPSTREAM(this->local_backend_.HIP_stream_current));
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
    void HIPAcceleratorMatrixELL<ValueType>::CopyFromAsync(const BaseMatrix<ValueType>& src)
    {
        const HIPAcceleratorMatrixELL<ValueType>* hip_cast_mat;
        const HostMatrix<ValueType>*              host_cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == src.GetMatFormat());

        // HIP to HIP copy
        if((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixELL<ValueType>*>(&src)) != NULL)
        {
            if(this->nnz_ == 0)
            {
                this->AllocateELL(hip_cast_mat->nnz_,
                                  hip_cast_mat->nrow_,
                                  hip_cast_mat->ncol_,
                                  hip_cast_mat->mat_.max_row);
            }

            assert(this->nnz_ == hip_cast_mat->nnz_);
            assert(this->nrow_ == hip_cast_mat->nrow_);
            assert(this->ncol_ == hip_cast_mat->ncol_);

            copy_d2d(this->nnz_,
                     hip_cast_mat->mat_.col,
                     this->mat_.col,
                     true,
                     HIPSTREAM(this->local_backend_.HIP_stream_current));
            copy_d2d(this->nnz_,
                     hip_cast_mat->mat_.val,
                     this->mat_.val,
                     true,
                     HIPSTREAM(this->local_backend_.HIP_stream_current));
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
    void HIPAcceleratorMatrixELL<ValueType>::CopyToAsync(BaseMatrix<ValueType>* dst) const
    {
        HIPAcceleratorMatrixELL<ValueType>* hip_cast_mat;
        HostMatrix<ValueType>*              host_cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == dst->GetMatFormat());

        // HIP to HIP copy
        if((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixELL<ValueType>*>(dst)) != NULL)
        {
            hip_cast_mat->set_backend(this->local_backend_);

            if(this->nnz_ == 0)
            {
                hip_cast_mat->AllocateELL(this->nnz_, this->nrow_, this->ncol_, this->mat_.max_row);
            }

            assert(this->nnz_ == hip_cast_mat->nnz_);
            assert(this->nrow_ == hip_cast_mat->nrow_);
            assert(this->ncol_ == hip_cast_mat->ncol_);
            assert(this->mat_.max_row == hip_cast_mat->mat_.max_row);

            copy_d2d(this->nnz_,
                     this->mat_.col,
                     hip_cast_mat->mat_.col,
                     true,
                     HIPSTREAM(this->local_backend_.HIP_stream_current));
            copy_d2d(this->nnz_,
                     this->mat_.val,
                     hip_cast_mat->mat_.val,
                     true,
                     HIPSTREAM(this->local_backend_.HIP_stream_current));
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
    bool HIPAcceleratorMatrixELL<ValueType>::ConvertFrom(const BaseMatrix<ValueType>& mat)
    {
        this->Clear();

        // Empty matrix
        if(mat.GetNnz() == 0)
        {
            this->AllocateELL(0, mat.GetM(), mat.GetN(), 0);
            return true;
        }

        const HIPAcceleratorMatrixELL<ValueType>* cast_mat_ell;

        if((cast_mat_ell = dynamic_cast<const HIPAcceleratorMatrixELL<ValueType>*>(&mat)) != NULL)
        {
            this->CopyFrom(*cast_mat_ell);
            return true;
        }

        const HIPAcceleratorMatrixCSR<ValueType>* cast_mat_csr;
        if((cast_mat_csr = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*>(&mat)) != NULL)
        {
            this->Clear();

            int64_t ell_nnz;

            if(csr_to_ell_hip(&this->local_backend_,
                              cast_mat_csr->nnz_,
                              cast_mat_csr->nrow_,
                              cast_mat_csr->ncol_,
                              cast_mat_csr->mat_,
                              cast_mat_csr->mat_descr_,
                              &this->mat_,
                              this->mat_descr_,
                              &ell_nnz)
               == true)
            {
                this->nrow_ = cast_mat_csr->nrow_;
                this->ncol_ = cast_mat_csr->ncol_;
                this->nnz_  = ell_nnz;

                return true;
            }
        }

        return false;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixELL<ValueType>::Apply(const BaseVector<ValueType>& in,
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

            ValueType alpha = 1.0;
            ValueType beta  = 0.0;

            rocsparse_status status;
            status = rocsparseTellmv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                     rocsparse_operation_none,
                                     this->nrow_,
                                     this->ncol_,
                                     &alpha,
                                     this->mat_descr_,
                                     this->mat_.val,
                                     this->mat_.col,
                                     this->mat_.max_row,
                                     cast_in->vec_,
                                     &beta,
                                     cast_out->vec_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixELL<ValueType>::ApplyAdd(const BaseVector<ValueType>& in,
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

            ValueType beta = 1.0;

            rocsparse_status status;
            status = rocsparseTellmv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                     rocsparse_operation_none,
                                     this->nrow_,
                                     this->ncol_,
                                     &scalar,
                                     this->mat_descr_,
                                     this->mat_.val,
                                     this->mat_.col,
                                     this->mat_.max_row,
                                     cast_in->vec_,
                                     &beta,
                                     cast_out->vec_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }
    }

    template class HIPAcceleratorMatrixELL<double>;
    template class HIPAcceleratorMatrixELL<float>;
#ifdef SUPPORT_COMPLEX
    template class HIPAcceleratorMatrixELL<std::complex<double>>;
    template class HIPAcceleratorMatrixELL<std::complex<float>>;
#endif

} // namespace rocalution

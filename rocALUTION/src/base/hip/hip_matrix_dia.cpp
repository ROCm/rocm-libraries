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

#include "hip_matrix_dia.hpp"
#include "../../utils/allocate_free.hpp"
#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "../backend_manager.hpp"
#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../host/host_matrix_dia.hpp"
#include "../matrix_formats_ind.hpp"
#include "hip_allocate_free.hpp"
#include "hip_conversion.hpp"
#include "hip_kernels_dia.hpp"
#include "hip_kernels_general.hpp"
#include "hip_kernels_vector.hpp"
#include "hip_matrix_csr.hpp"
#include "hip_utils.hpp"
#include "hip_vector.hpp"

#include <hip/hip_runtime.h>

namespace rocalution
{

    template <typename ValueType>
    HIPAcceleratorMatrixDIA<ValueType>::HIPAcceleratorMatrixDIA()
    {
        // no default constructors
        LOG_INFO("no default constructor");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    HIPAcceleratorMatrixDIA<ValueType>::HIPAcceleratorMatrixDIA(
        const Rocalution_Backend_Descriptor& local_backend)
    {
        log_debug(this,
                  "HIPAcceleratorMatrixDIA::HIPAcceleratorMatrixDIA()",
                  "constructor with local_backend");

        this->mat_.val      = NULL;
        this->mat_.offset   = NULL;
        this->mat_.num_diag = 0;
        this->set_backend(local_backend);

        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    HIPAcceleratorMatrixDIA<ValueType>::~HIPAcceleratorMatrixDIA()
    {
        log_debug(this, "HIPAcceleratorMatrixDIA::HIPAcceleratorMatrixDIA()", "destructor");

        this->Clear();
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixDIA<ValueType>::Info(void) const
    {
        LOG_INFO("HIPAcceleratorMatrixDIA<ValueType> diag=" << this->mat_.num_diag
                                                            << " nnz=" << this->nnz_);
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixDIA<ValueType>::AllocateDIA(int64_t nnz, int nrow, int ncol, int ndiag)
    {
        assert(nnz >= 0);
        assert(ncol >= 0);
        assert(nrow >= 0);
        assert(ndiag >= 0);

        this->Clear();

        allocate_hip(nnz, &this->mat_.val);
        allocate_hip(ndiag, &this->mat_.offset);

        set_to_zero_hip(this->local_backend_.HIP_block_size, nnz, mat_.val);

        set_to_zero_hip(this->local_backend_.HIP_block_size, ndiag, mat_.offset);

        this->nrow_         = nrow;
        this->ncol_         = ncol;
        this->nnz_          = nnz;
        this->mat_.num_diag = ndiag;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixDIA<ValueType>::SetDataPtrDIA(
        int** offset, ValueType** val, int64_t nnz, int nrow, int ncol, int num_diag)
    {
        assert(nnz >= 0);
        assert(nrow >= 0);
        assert(ncol >= 0);
        assert(num_diag >= 0);

        if(num_diag > 0)
        {
            assert(*offset != NULL);
        }

        if(nnz > 0)
        {
            assert(*val != NULL);
        }

        if(nrow < ncol)
        {
            assert(nnz == ncol * num_diag);
        }
        else
        {
            assert(nnz == nrow * num_diag);
        }

        this->Clear();

        hipDeviceSynchronize();

        this->mat_.num_diag = num_diag;
        this->nrow_         = nrow;
        this->ncol_         = ncol;
        this->nnz_          = nnz;

        this->mat_.offset = *offset;
        this->mat_.val    = *val;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixDIA<ValueType>::LeaveDataPtrDIA(int**       offset,
                                                             ValueType** val,
                                                             int&        num_diag)
    {
        assert(this->nrow_ >= 0);
        assert(this->ncol_ >= 0);
        assert(this->nnz_ >= 0);
        assert(this->mat_.num_diag >= 0);

        if(this->nrow_ < this->ncol_)
        {
            assert(this->nnz_ == this->ncol_ * this->mat_.num_diag);
        }
        else
        {
            assert(this->nnz_ == this->nrow_ * this->mat_.num_diag);
        }

        hipDeviceSynchronize();

        *offset = this->mat_.offset;
        *val    = this->mat_.val;

        this->mat_.offset = NULL;
        this->mat_.val    = NULL;

        num_diag = this->mat_.num_diag;

        this->mat_.num_diag = 0;
        this->nrow_         = 0;
        this->ncol_         = 0;
        this->nnz_          = 0;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixDIA<ValueType>::Clear()
    {
        free_hip(&this->mat_.val);
        free_hip(&this->mat_.offset);

        this->nrow_         = 0;
        this->ncol_         = 0;
        this->nnz_          = 0;
        this->mat_.num_diag = 0;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixDIA<ValueType>::CopyFromHost(const HostMatrix<ValueType>& src)
    {
        const HostMatrixDIA<ValueType>* cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == src.GetMatFormat());

        // CPU to HIP copy
        if((cast_mat = dynamic_cast<const HostMatrixDIA<ValueType>*>(&src)) != NULL)
        {
            if(this->nnz_ == 0)
            {
                this->AllocateDIA(
                    cast_mat->nnz_, cast_mat->nrow_, cast_mat->ncol_, cast_mat->mat_.num_diag);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

            copy_h2d(this->mat_.num_diag, cast_mat->mat_.offset, this->mat_.offset);
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
    void HIPAcceleratorMatrixDIA<ValueType>::CopyToHost(HostMatrix<ValueType>* dst) const
    {
        HostMatrixDIA<ValueType>* cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == dst->GetMatFormat());

        // HIP to CPU copy
        if((cast_mat = dynamic_cast<HostMatrixDIA<ValueType>*>(dst)) != NULL)
        {
            cast_mat->set_backend(this->local_backend_);

            if(cast_mat->nnz_ == 0)
            {
                cast_mat->AllocateDIA(this->nnz_, this->nrow_, this->ncol_, this->mat_.num_diag);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

            copy_d2h(this->mat_.num_diag, this->mat_.offset, cast_mat->mat_.offset);
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
    void HIPAcceleratorMatrixDIA<ValueType>::CopyFrom(const BaseMatrix<ValueType>& src)
    {
        const HIPAcceleratorMatrixDIA<ValueType>* hip_cast_mat;
        const HostMatrix<ValueType>*              host_cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == src.GetMatFormat());

        // HIP to HIP copy
        if((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixDIA<ValueType>*>(&src)) != NULL)
        {
            if(this->nnz_ == 0)
            {
                this->AllocateDIA(hip_cast_mat->nnz_,
                                  hip_cast_mat->nrow_,
                                  hip_cast_mat->ncol_,
                                  hip_cast_mat->mat_.num_diag);
            }

            assert(this->nnz_ == hip_cast_mat->nnz_);
            assert(this->nrow_ == hip_cast_mat->nrow_);
            assert(this->ncol_ == hip_cast_mat->ncol_);

            copy_d2d(this->mat_.num_diag, hip_cast_mat->mat_.offset, this->mat_.offset);
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
    void HIPAcceleratorMatrixDIA<ValueType>::CopyTo(BaseMatrix<ValueType>* dst) const
    {
        HIPAcceleratorMatrixDIA<ValueType>* hip_cast_mat;
        HostMatrix<ValueType>*              host_cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == dst->GetMatFormat());

        // HIP to HIP copy
        if((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixDIA<ValueType>*>(dst)) != NULL)
        {
            hip_cast_mat->set_backend(this->local_backend_);

            if(hip_cast_mat->nnz_ == 0)
            {
                hip_cast_mat->AllocateDIA(
                    this->nnz_, this->nrow_, this->ncol_, this->mat_.num_diag);
            }

            assert(this->nnz_ == hip_cast_mat->nnz_);
            assert(this->nrow_ == hip_cast_mat->nrow_);
            assert(this->ncol_ == hip_cast_mat->ncol_);

            copy_d2d(this->mat_.num_diag, this->mat_.offset, hip_cast_mat->mat_.offset);
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
    void HIPAcceleratorMatrixDIA<ValueType>::CopyFromHostAsync(const HostMatrix<ValueType>& src)
    {
        const HostMatrixDIA<ValueType>* cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == src.GetMatFormat());

        // CPU to HIP copy
        if((cast_mat = dynamic_cast<const HostMatrixDIA<ValueType>*>(&src)) != NULL)
        {
            if(this->nnz_ == 0)
            {
                this->AllocateDIA(
                    cast_mat->nnz_, cast_mat->nrow_, cast_mat->ncol_, cast_mat->mat_.num_diag);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

            copy_h2d(this->mat_.num_diag,
                     cast_mat->mat_.offset,
                     this->mat_.offset,
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
    void HIPAcceleratorMatrixDIA<ValueType>::CopyToHostAsync(HostMatrix<ValueType>* dst) const
    {
        HostMatrixDIA<ValueType>* cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == dst->GetMatFormat());

        // HIP to CPU copy
        if((cast_mat = dynamic_cast<HostMatrixDIA<ValueType>*>(dst)) != NULL)
        {
            cast_mat->set_backend(this->local_backend_);

            if(cast_mat->nnz_ == 0)
            {
                cast_mat->AllocateDIA(this->nnz_, this->nrow_, this->ncol_, this->mat_.num_diag);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

            copy_d2h(this->mat_.num_diag,
                     this->mat_.offset,
                     cast_mat->mat_.offset,
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
    void HIPAcceleratorMatrixDIA<ValueType>::CopyFromAsync(const BaseMatrix<ValueType>& src)
    {
        const HIPAcceleratorMatrixDIA<ValueType>* hip_cast_mat;
        const HostMatrix<ValueType>*              host_cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == src.GetMatFormat());

        // HIP to HIP copy
        if((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixDIA<ValueType>*>(&src)) != NULL)
        {
            if(this->nnz_ == 0)
            {
                this->AllocateDIA(hip_cast_mat->nnz_,
                                  hip_cast_mat->nrow_,
                                  hip_cast_mat->ncol_,
                                  hip_cast_mat->mat_.num_diag);
            }

            assert(this->nnz_ == hip_cast_mat->nnz_);
            assert(this->nrow_ == hip_cast_mat->nrow_);
            assert(this->ncol_ == hip_cast_mat->ncol_);

            copy_d2d(this->mat_.num_diag,
                     hip_cast_mat->mat_.offset,
                     this->mat_.offset,
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
    void HIPAcceleratorMatrixDIA<ValueType>::CopyToAsync(BaseMatrix<ValueType>* dst) const
    {
        HIPAcceleratorMatrixDIA<ValueType>* hip_cast_mat;
        HostMatrix<ValueType>*              host_cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == dst->GetMatFormat());

        // HIP to HIP copy
        if((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixDIA<ValueType>*>(dst)) != NULL)
        {
            hip_cast_mat->set_backend(this->local_backend_);

            if(hip_cast_mat->nnz_ == 0)
            {
                hip_cast_mat->AllocateDIA(
                    this->nnz_, this->nrow_, this->ncol_, this->mat_.num_diag);
            }

            assert(this->nnz_ == hip_cast_mat->nnz_);
            assert(this->nrow_ == hip_cast_mat->nrow_);
            assert(this->ncol_ == hip_cast_mat->ncol_);

            copy_d2h(this->mat_.num_diag,
                     this->mat_.offset,
                     hip_cast_mat->mat_.offset,
                     true,
                     HIPSTREAM(this->local_backend_.HIP_stream_current));
            copy_d2h(this->nnz_,
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
    bool HIPAcceleratorMatrixDIA<ValueType>::ConvertFrom(const BaseMatrix<ValueType>& mat)
    {
        this->Clear();

        // Empty matrix
        if(mat.GetNnz() == 0)
        {
            this->AllocateDIA(0, mat.GetM(), mat.GetN(), 0);

            return true;
        }

        const HIPAcceleratorMatrixDIA<ValueType>* cast_mat_dia;

        if((cast_mat_dia = dynamic_cast<const HIPAcceleratorMatrixDIA<ValueType>*>(&mat)) != NULL)
        {
            this->CopyFrom(*cast_mat_dia);
            return true;
        }

        const HIPAcceleratorMatrixCSR<ValueType>* cast_mat_csr;
        if((cast_mat_csr = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*>(&mat)) != NULL)
        {
            this->Clear();

            int64_t nnz_dia;
            int     num_diag;

            if(csr_to_dia_hip(&this->local_backend_,
                              cast_mat_csr->nnz_,
                              cast_mat_csr->nrow_,
                              cast_mat_csr->ncol_,
                              cast_mat_csr->mat_,
                              &this->mat_,
                              &nnz_dia,
                              &num_diag)
               == true)
            {
                this->nrow_         = cast_mat_csr->nrow_;
                this->ncol_         = cast_mat_csr->ncol_;
                this->nnz_          = nnz_dia;
                this->mat_.num_diag = num_diag;

                return true;
            }
        }

        return false;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixDIA<ValueType>::Apply(const BaseVector<ValueType>& in,
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

            int  nrow     = this->nrow_;
            int  ncol     = this->ncol_;
            int  num_diag = this->mat_.num_diag;
            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

            kernel_dia_spmv<<<GridSize,
                              BlockSize,
                              0,
                              HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                nrow,
                ncol,
                num_diag,
                this->mat_.offset,
                this->mat_.val,
                cast_in->vec_,
                cast_out->vec_);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixDIA<ValueType>::ApplyAdd(const BaseVector<ValueType>& in,
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

            int  nrow     = this->nrow_;
            int  ncol     = this->ncol_;
            int  num_diag = this->mat_.num_diag;
            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(nrow / this->local_backend_.HIP_block_size + 1);

            kernel_dia_add_spmv<<<GridSize,
                                  BlockSize,
                                  0,
                                  HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                nrow,
                ncol,
                num_diag,
                this->mat_.offset,
                this->mat_.val,
                scalar,
                cast_in->vec_,
                cast_out->vec_);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }

    template class HIPAcceleratorMatrixDIA<double>;
    template class HIPAcceleratorMatrixDIA<float>;
#ifdef SUPPORT_COMPLEX
    template class HIPAcceleratorMatrixDIA<std::complex<double>>;
    template class HIPAcceleratorMatrixDIA<std::complex<float>>;
#endif

} // namespace rocalution

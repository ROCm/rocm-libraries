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

#include "hip_matrix_bcsr.hpp"
#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "../backend_manager.hpp"
#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../host/host_matrix_bcsr.hpp"
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
    HIPAcceleratorMatrixBCSR<ValueType>::HIPAcceleratorMatrixBCSR()
    {
        // no default constructors
        LOG_INFO("no default constructor");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    HIPAcceleratorMatrixBCSR<ValueType>::HIPAcceleratorMatrixBCSR(
        const Rocalution_Backend_Descriptor& local_backend, int blockdim)
    {
        log_debug(this,
                  "HIPAcceleratorMatrixBCSR::HIPAcceleratorMatrixBCSR()",
                  "constructor with local_backend");

        this->mat_.row_offset = NULL;
        this->mat_.col        = NULL;
        this->mat_.val        = NULL;
        this->mat_.blockdim   = blockdim;
        this->set_backend(local_backend);

        this->L_mat_descr_ = 0;
        this->U_mat_descr_ = 0;

        this->mat_descr_ = 0;
        this->mat_info_  = 0;

        this->mat_buffer_size_ = 0;
        this->mat_buffer_      = NULL;

        this->tmp_vec_ = NULL;

        CHECK_HIP_ERROR(__FILE__, __LINE__);

        rocsparse_status status;

        status = rocsparse_create_mat_descr(&this->mat_descr_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_index_base(this->mat_descr_, rocsparse_index_base_zero);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_type(this->mat_descr_, rocsparse_matrix_type_general);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_create_mat_info(&this->mat_info_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    template <typename ValueType>
    HIPAcceleratorMatrixBCSR<ValueType>::~HIPAcceleratorMatrixBCSR()
    {
        log_debug(this, "HIPAcceleratorMatrixBCSR::~HIPAcceleratorMatrixBCSR()", "destructor");

        this->Clear();

        rocsparse_status status;

        status = rocsparse_destroy_mat_descr(this->mat_descr_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_destroy_mat_info(this->mat_info_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixBCSR<ValueType>::Info(void) const
    {
        LOG_INFO("HIPAcceleratorMatrixBCSR<ValueType>");
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixBCSR<ValueType>::AllocateBCSR(int64_t nnzb,
                                                           int     nrowb,
                                                           int     ncolb,
                                                           int     blockdim)
    {
        assert(nnzb >= 0);
        assert(ncolb >= 0);
        assert(nrowb >= 0);
        assert(blockdim > 1);

        this->Clear();

        allocate_hip(nrowb + 1, &this->mat_.row_offset);
        allocate_hip(nnzb, &this->mat_.col);
        allocate_hip(nnzb * blockdim * blockdim, &this->mat_.val);

        set_to_zero_hip(this->local_backend_.HIP_block_size, nrowb + 1, this->mat_.row_offset);
        set_to_zero_hip(this->local_backend_.HIP_block_size, nnzb, this->mat_.col);
        set_to_zero_hip(
            this->local_backend_.HIP_block_size, nnzb * blockdim * blockdim, this->mat_.val);

        this->nrow_ = nrowb * blockdim;
        this->ncol_ = ncolb * blockdim;
        this->nnz_  = nnzb * blockdim * blockdim;

        this->mat_.nrowb    = nrowb;
        this->mat_.ncolb    = ncolb;
        this->mat_.nnzb     = nnzb;
        this->mat_.blockdim = blockdim;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixBCSR<ValueType>::Clear()
    {
        free_hip(&this->mat_.row_offset);
        free_hip(&this->mat_.col);
        free_hip(&this->mat_.val);

        this->nrow_ = 0;
        this->ncol_ = 0;
        this->nnz_  = 0;

        this->LAnalyseClear();
        this->UAnalyseClear();
        this->LUAnalyseClear();
        this->LLAnalyseClear();
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixBCSR<ValueType>::SetDataPtrBCSR(int**       row_offset,
                                                             int**       col,
                                                             ValueType** val,
                                                             int64_t     nnzb,
                                                             int         nrowb,
                                                             int         ncolb,
                                                             int         blockdim)
    {
        assert(nnzb >= 0);
        assert(nrowb >= 0);
        assert(ncolb >= 0);
        assert(blockdim > 1);
        assert(*row_offset != NULL);

        if(nnzb > 0)
        {
            assert(*col != NULL);
            assert(*val != NULL);
        }

        this->Clear();

        hipDeviceSynchronize();

        this->nrow_ = nrowb * blockdim;
        this->ncol_ = ncolb * blockdim;
        this->nnz_  = nnzb * blockdim * blockdim;

        this->mat_.nrowb    = nrowb;
        this->mat_.ncolb    = ncolb;
        this->mat_.nnzb     = nnzb;
        this->mat_.blockdim = blockdim;

        this->mat_.row_offset = *row_offset;
        this->mat_.col        = *col;
        this->mat_.val        = *val;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixBCSR<ValueType>::LeaveDataPtrBCSR(int**       row_offset,
                                                               int**       col,
                                                               ValueType** val,
                                                               int&        blockdim)
    {
        assert(this->nrow_ >= 0);
        assert(this->ncol_ >= 0);
        assert(this->nnz_ >= 0);
        assert(this->mat_.blockdim > 1);

        hipDeviceSynchronize();

        *row_offset = this->mat_.row_offset;
        *col        = this->mat_.col;
        *val        = this->mat_.val;

        this->mat_.row_offset = NULL;
        this->mat_.col        = NULL;
        this->mat_.val        = NULL;

        blockdim = this->mat_.blockdim;

        this->mat_.blockdim = 0;

        this->nrow_ = 0;
        this->ncol_ = 0;
        this->nnz_  = 0;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixBCSR<ValueType>::CopyFromHost(const HostMatrix<ValueType>& src)
    {
        const HostMatrixBCSR<ValueType>* cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == src.GetMatFormat());

        // CPU to HIP copy
        if((cast_mat = dynamic_cast<const HostMatrixBCSR<ValueType>*>(&src)) != NULL)
        {
            if(this->nnz_ == 0)
            {
                this->AllocateBCSR(cast_mat->mat_.nnzb,
                                   cast_mat->mat_.nrowb,
                                   cast_mat->mat_.ncolb,
                                   cast_mat->mat_.blockdim);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);
            assert(this->mat_.nrowb == cast_mat->mat_.nrowb);
            assert(this->mat_.ncolb == cast_mat->mat_.ncolb);
            assert(this->mat_.nnzb == cast_mat->mat_.nnzb);
            assert(this->mat_.blockdim == cast_mat->mat_.blockdim);

            // Copy only if initialized
            if(cast_mat->mat_.row_offset != NULL)
            {
                copy_h2d(this->mat_.nrowb + 1, cast_mat->mat_.row_offset, this->mat_.row_offset);
            }

            copy_h2d(this->mat_.nnzb, cast_mat->mat_.col, this->mat_.col);
            copy_h2d(this->mat_.nnzb * this->mat_.blockdim * this->mat_.blockdim,
                     cast_mat->mat_.val,
                     this->mat_.val);
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
    void HIPAcceleratorMatrixBCSR<ValueType>::CopyToHost(HostMatrix<ValueType>* dst) const
    {
        HostMatrixBCSR<ValueType>* cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == dst->GetMatFormat());

        // HIP to CPU copy
        if((cast_mat = dynamic_cast<HostMatrixBCSR<ValueType>*>(dst)) != NULL)
        {
            cast_mat->set_backend(this->local_backend_);

            if(cast_mat->nnz_ == 0)
            {
                cast_mat->AllocateBCSR(
                    this->mat_.nnzb, this->mat_.nrowb, this->mat_.ncolb, this->mat_.blockdim);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);
            assert(this->mat_.nrowb == cast_mat->mat_.nrowb);
            assert(this->mat_.ncolb == cast_mat->mat_.ncolb);
            assert(this->mat_.nnzb == cast_mat->mat_.nnzb);
            assert(this->mat_.blockdim == cast_mat->mat_.blockdim);

            if(this->mat_.row_offset != NULL)
            {
                copy_d2h(this->mat_.nrowb + 1, this->mat_.row_offset, cast_mat->mat_.row_offset);
            }

            copy_d2h(this->mat_.nnzb, this->mat_.col, cast_mat->mat_.col);
            copy_d2h(this->mat_.nnzb * this->mat_.blockdim * this->mat_.blockdim,
                     this->mat_.val,
                     cast_mat->mat_.val);
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
    void HIPAcceleratorMatrixBCSR<ValueType>::CopyFrom(const BaseMatrix<ValueType>& src)
    {
        const HIPAcceleratorMatrixBCSR<ValueType>* hip_cast_mat;
        const HostMatrix<ValueType>*               host_cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == src.GetMatFormat());

        // HIP to HIP copy
        if((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixBCSR<ValueType>*>(&src)) != NULL)
        {
            if(this->nnz_ == 0)
            {
                this->AllocateBCSR(hip_cast_mat->mat_.nnzb,
                                   hip_cast_mat->mat_.nrowb,
                                   hip_cast_mat->mat_.ncolb,
                                   hip_cast_mat->mat_.blockdim);
            }

            assert(this->nnz_ == hip_cast_mat->nnz_);
            assert(this->nrow_ == hip_cast_mat->nrow_);
            assert(this->ncol_ == hip_cast_mat->ncol_);
            assert(this->mat_.nrowb == hip_cast_mat->mat_.nrowb);
            assert(this->mat_.ncolb == hip_cast_mat->mat_.ncolb);
            assert(this->mat_.nnzb == hip_cast_mat->mat_.nnzb);
            assert(this->mat_.blockdim == hip_cast_mat->mat_.blockdim);

            // Copy only if initialized
            if(hip_cast_mat->mat_.row_offset != NULL)
            {
                copy_h2d(
                    this->mat_.nrowb + 1, hip_cast_mat->mat_.row_offset, this->mat_.row_offset);
            }

            copy_h2d(this->mat_.nnzb, hip_cast_mat->mat_.col, this->mat_.col);
            copy_h2d(this->mat_.nnzb * this->mat_.blockdim * this->mat_.blockdim,
                     hip_cast_mat->mat_.val,
                     this->mat_.val);
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
    void HIPAcceleratorMatrixBCSR<ValueType>::CopyTo(BaseMatrix<ValueType>* dst) const
    {
        HIPAcceleratorMatrixBCSR<ValueType>* hip_cast_mat;
        HostMatrix<ValueType>*               host_cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == dst->GetMatFormat());

        // HIP to HIP copy
        if((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixBCSR<ValueType>*>(dst)) != NULL)
        {
            hip_cast_mat->set_backend(this->local_backend_);

            if(hip_cast_mat->nnz_ == 0)
            {
                hip_cast_mat->AllocateBCSR(
                    this->mat_.nnzb, this->mat_.nrowb, this->mat_.ncolb, this->mat_.blockdim);
            }

            assert(this->nnz_ == hip_cast_mat->nnz_);
            assert(this->nrow_ == hip_cast_mat->nrow_);
            assert(this->ncol_ == hip_cast_mat->ncol_);
            assert(this->mat_.nrowb == hip_cast_mat->mat_.nrowb);
            assert(this->mat_.ncolb == hip_cast_mat->mat_.ncolb);
            assert(this->mat_.nnzb == hip_cast_mat->mat_.nnzb);
            assert(this->mat_.blockdim == hip_cast_mat->mat_.blockdim);

            if(this->mat_.row_offset != NULL)
            {
                copy_d2d(
                    this->mat_.nrowb + 1, this->mat_.row_offset, hip_cast_mat->mat_.row_offset);
            }

            copy_d2d(this->mat_.nnzb, this->mat_.col, hip_cast_mat->mat_.col);
            copy_d2d(this->mat_.nnzb * this->mat_.blockdim * this->mat_.blockdim,
                     this->mat_.val,
                     hip_cast_mat->mat_.val);
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
    void HIPAcceleratorMatrixBCSR<ValueType>::CopyFromHostAsync(const HostMatrix<ValueType>& src)
    {
        const HostMatrixBCSR<ValueType>* cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == src.GetMatFormat());

        // CPU to HIP copy
        if((cast_mat = dynamic_cast<const HostMatrixBCSR<ValueType>*>(&src)) != NULL)
        {
            if(this->nnz_ == 0)
            {
                this->AllocateBCSR(cast_mat->mat_.nnzb,
                                   cast_mat->mat_.nrowb,
                                   cast_mat->mat_.ncolb,
                                   cast_mat->mat_.blockdim);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);
            assert(this->mat_.nrowb == cast_mat->mat_.nrowb);
            assert(this->mat_.ncolb == cast_mat->mat_.ncolb);
            assert(this->mat_.nnzb == cast_mat->mat_.nnzb);
            assert(this->mat_.blockdim == cast_mat->mat_.blockdim);

            // Copy only if initialized
            if(cast_mat->mat_.row_offset != NULL)
            {
                copy_h2d(this->mat_.nrowb + 1,
                         cast_mat->mat_.row_offset,
                         this->mat_.row_offset,
                         true,
                         HIPSTREAM(this->local_backend_.HIP_stream_current));
            }

            copy_h2d(this->mat_.nnzb,
                     cast_mat->mat_.col,
                     this->mat_.col,
                     true,
                     HIPSTREAM(this->local_backend_.HIP_stream_current));
            copy_h2d(this->mat_.nnzb * this->mat_.blockdim * this->mat_.blockdim,
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
    void HIPAcceleratorMatrixBCSR<ValueType>::CopyToHostAsync(HostMatrix<ValueType>* dst) const
    {
        HostMatrixBCSR<ValueType>* cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == dst->GetMatFormat());

        // HIP to CPU copy
        if((cast_mat = dynamic_cast<HostMatrixBCSR<ValueType>*>(dst)) != NULL)
        {
            cast_mat->set_backend(this->local_backend_);

            if(cast_mat->nnz_ == 0)
            {
                cast_mat->AllocateBCSR(
                    this->mat_.nnzb, this->mat_.nrowb, this->mat_.ncolb, this->mat_.blockdim);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);
            assert(this->mat_.nrowb == cast_mat->mat_.nrowb);
            assert(this->mat_.ncolb == cast_mat->mat_.ncolb);
            assert(this->mat_.nnzb == cast_mat->mat_.nnzb);
            assert(this->mat_.blockdim == cast_mat->mat_.blockdim);

            if(this->mat_.row_offset != NULL)
            {
                copy_d2h(this->mat_.nrowb + 1,
                         this->mat_.row_offset,
                         cast_mat->mat_.row_offset,
                         true,
                         HIPSTREAM(this->local_backend_.HIP_stream_current));
            }

            copy_d2h(this->mat_.nnzb,
                     this->mat_.col,
                     cast_mat->mat_.col,
                     true,
                     HIPSTREAM(this->local_backend_.HIP_stream_current));
            copy_d2h(this->mat_.nnzb * this->mat_.blockdim * this->mat_.blockdim,
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
    void HIPAcceleratorMatrixBCSR<ValueType>::CopyFromAsync(const BaseMatrix<ValueType>& src)
    {
        const HIPAcceleratorMatrixBCSR<ValueType>* hip_cast_mat;
        const HostMatrix<ValueType>*               host_cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == src.GetMatFormat());

        // HIP to HIP copy
        if((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixBCSR<ValueType>*>(&src)) != NULL)
        {
            if(this->nnz_ == 0)
            {
                this->AllocateBCSR(hip_cast_mat->mat_.nnzb,
                                   hip_cast_mat->mat_.nrowb,
                                   hip_cast_mat->mat_.ncolb,
                                   hip_cast_mat->mat_.blockdim);
            }

            assert(this->nnz_ == hip_cast_mat->nnz_);
            assert(this->nrow_ == hip_cast_mat->nrow_);
            assert(this->ncol_ == hip_cast_mat->ncol_);
            assert(this->mat_.nrowb == hip_cast_mat->mat_.nrowb);
            assert(this->mat_.ncolb == hip_cast_mat->mat_.ncolb);
            assert(this->mat_.nnzb == hip_cast_mat->mat_.nnzb);
            assert(this->mat_.blockdim == hip_cast_mat->mat_.blockdim);

            // Copy only if initialized
            if(hip_cast_mat->mat_.row_offset != NULL)
            {
                copy_d2d(this->mat_.nrowb + 1,
                         hip_cast_mat->mat_.row_offset,
                         this->mat_.row_offset,
                         true,
                         HIPSTREAM(this->local_backend_.HIP_stream_current));
            }

            copy_d2d(this->mat_.nnzb,
                     hip_cast_mat->mat_.col,
                     this->mat_.col,
                     true,
                     HIPSTREAM(this->local_backend_.HIP_stream_current));
            copy_d2d(this->mat_.nnzb * this->mat_.blockdim * this->mat_.blockdim,
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
    void HIPAcceleratorMatrixBCSR<ValueType>::CopyToAsync(BaseMatrix<ValueType>* dst) const
    {
        HIPAcceleratorMatrixBCSR<ValueType>* hip_cast_mat;
        HostMatrix<ValueType>*               host_cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == dst->GetMatFormat());

        // HIP to HIP copy
        if((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixBCSR<ValueType>*>(dst)) != NULL)
        {
            hip_cast_mat->set_backend(this->local_backend_);

            if(this->nnz_ == 0)
            {
                hip_cast_mat->AllocateBCSR(
                    this->mat_.nnzb, this->mat_.nrowb, this->mat_.ncolb, this->mat_.blockdim);
            }

            assert(this->nnz_ == hip_cast_mat->nnz_);
            assert(this->nrow_ == hip_cast_mat->nrow_);
            assert(this->ncol_ == hip_cast_mat->ncol_);
            assert(this->mat_.nrowb == hip_cast_mat->mat_.nrowb);
            assert(this->mat_.ncolb == hip_cast_mat->mat_.ncolb);
            assert(this->mat_.nnzb == hip_cast_mat->mat_.nnzb);
            assert(this->mat_.blockdim == hip_cast_mat->mat_.blockdim);

            if(this->mat_.row_offset != NULL)
            {
                copy_d2d(this->mat_.nrowb + 1,
                         this->mat_.row_offset,
                         hip_cast_mat->mat_.row_offset,
                         true,
                         HIPSTREAM(this->local_backend_.HIP_stream_current));
            }

            copy_d2d(this->mat_.nnzb,
                     this->mat_.col,
                     hip_cast_mat->mat_.col,
                     true,
                     HIPSTREAM(this->local_backend_.HIP_stream_current));
            copy_d2d(this->mat_.nnzb * this->mat_.blockdim * this->mat_.blockdim,
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
    bool HIPAcceleratorMatrixBCSR<ValueType>::ConvertFrom(const BaseMatrix<ValueType>& mat)
    {
        this->Clear();

        // Empty matrix
        if(mat.GetNnz() == 0)
        {
            int mb = (mat.GetM() + 1) / 2;
            int nb = (mat.GetN() + 1) / 2;

            this->AllocateBCSR(0, mb, nb, 2);

            return true;
        }

        const HIPAcceleratorMatrixBCSR<ValueType>* cast_mat_bcsr;

        if((cast_mat_bcsr = dynamic_cast<const HIPAcceleratorMatrixBCSR<ValueType>*>(&mat)) != NULL)
        {
            this->CopyFrom(*cast_mat_bcsr);
            return true;
        }

        const HIPAcceleratorMatrixCSR<ValueType>* cast_mat_csr;
        if((cast_mat_csr = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*>(&mat)) != NULL)
        {
            this->Clear();

            if(csr_to_bcsr_hip(&this->local_backend_,
                               cast_mat_csr->nnz_,
                               cast_mat_csr->nrow_,
                               cast_mat_csr->ncol_,
                               cast_mat_csr->mat_,
                               cast_mat_csr->mat_descr_,
                               &this->mat_,
                               this->mat_descr_)
               == true)
            {
                this->nrow_ = this->mat_.nrowb * this->mat_.blockdim;
                this->ncol_ = this->mat_.ncolb * this->mat_.blockdim;
                this->nnz_  = this->mat_.nnzb * this->mat_.blockdim * this->mat_.blockdim;

                return true;
            }
        }

        return false;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixBCSR<ValueType>::ILU0Factorize(void)
    {
        if(this->nnz_ > 0)
        {
            rocsparse_status status;

            // Determine whether we are using row or column major for the blocks
            rocsparse_direction dir
                = BCSR_IND_BASE ? rocsparse_direction_row : rocsparse_direction_column;

            // Create buffer, if not already available
            size_t buffer_size = 0;
            status             = rocsparseTbsrilu0_buffer_size(
                ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                dir,
                this->mat_.nrowb,
                this->mat_.nnzb,
                this->mat_descr_,
                this->mat_.val,
                this->mat_.row_offset,
                this->mat_.col,
                this->mat_.blockdim,
                this->mat_info_,
                &buffer_size);

            // Buffer is shared with ILU0 and other solve functions
            if(this->mat_buffer_ == NULL)
            {
                this->mat_buffer_size_ = buffer_size;
                allocate_hip(buffer_size, &this->mat_buffer_);
            }

            assert(this->mat_buffer_size_ >= buffer_size);
            assert(this->mat_buffer_ != NULL);

            status = rocsparseTbsrilu0_analysis(
                ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                dir,
                this->mat_.nrowb,
                this->mat_.nnzb,
                this->mat_descr_,
                this->mat_.val,
                this->mat_.row_offset,
                this->mat_.col,
                this->mat_.blockdim,
                this->mat_info_,
                rocsparse_analysis_policy_reuse,
                rocsparse_solve_policy_auto,
                this->mat_buffer_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            status = rocsparseTbsrilu0(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                       dir,
                                       this->mat_.nrowb,
                                       this->mat_.nnzb,
                                       this->mat_descr_,
                                       this->mat_.val,
                                       this->mat_.row_offset,
                                       this->mat_.col,
                                       this->mat_.blockdim,
                                       this->mat_info_,
                                       rocsparse_solve_policy_auto,
                                       this->mat_buffer_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            status = rocsparse_bsrilu0_clear(
                ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle), this->mat_info_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        return true;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixBCSR<ValueType>::LUAnalyse(void)
    {
        assert(this->ncol_ == this->nrow_);
        assert(this->tmp_vec_ == NULL);

        this->tmp_vec_ = new HIPAcceleratorVector<ValueType>(this->local_backend_);

        assert(this->tmp_vec_ != NULL);

        rocsparse_status status;

        // L part
        status = rocsparse_create_mat_descr(&this->L_mat_descr_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_type(this->L_mat_descr_, rocsparse_matrix_type_general);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_index_base(this->L_mat_descr_, rocsparse_index_base_zero);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_fill_mode(this->L_mat_descr_, rocsparse_fill_mode_lower);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_diag_type(this->L_mat_descr_, rocsparse_diag_type_unit);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        // U part
        status = rocsparse_create_mat_descr(&this->U_mat_descr_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_type(this->U_mat_descr_, rocsparse_matrix_type_general);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_index_base(this->U_mat_descr_, rocsparse_index_base_zero);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_fill_mode(this->U_mat_descr_, rocsparse_fill_mode_upper);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_diag_type(this->U_mat_descr_, rocsparse_diag_type_non_unit);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        // Determine whether we are using row or column major for the blocks
        rocsparse_direction dir
            = BCSR_IND_BASE ? rocsparse_direction_row : rocsparse_direction_column;

        // Create buffer, if not already available
        size_t buffer_size = 0;
        status
            = rocsparseTbsrsv_buffer_size(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                          dir,
                                          rocsparse_operation_none,
                                          this->mat_.nrowb,
                                          this->mat_.nnzb,
                                          this->L_mat_descr_,
                                          this->mat_.val,
                                          this->mat_.row_offset,
                                          this->mat_.col,
                                          this->mat_.blockdim,
                                          this->mat_info_,
                                          &buffer_size);

        // Buffer is shared with ILU0 and other solve functions
        if(this->mat_buffer_ == NULL)
        {
            this->mat_buffer_size_ = buffer_size;
            allocate_hip(buffer_size, &this->mat_buffer_);
        }

        assert(this->mat_buffer_size_ >= buffer_size);
        assert(this->mat_buffer_ != NULL);

        // L part analysis
        status = rocsparseTbsrsv_analysis(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                          dir,
                                          rocsparse_operation_none,
                                          this->mat_.nrowb,
                                          this->mat_.nnzb,
                                          this->L_mat_descr_,
                                          this->mat_.val,
                                          this->mat_.row_offset,
                                          this->mat_.col,
                                          this->mat_.blockdim,
                                          this->mat_info_,
                                          rocsparse_analysis_policy_reuse,
                                          rocsparse_solve_policy_auto,
                                          this->mat_buffer_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        // U part analysis
        status = rocsparseTbsrsv_analysis(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                          dir,
                                          rocsparse_operation_none,
                                          this->mat_.nrowb,
                                          this->mat_.nnzb,
                                          this->U_mat_descr_,
                                          this->mat_.val,
                                          this->mat_.row_offset,
                                          this->mat_.col,
                                          this->mat_.blockdim,
                                          this->mat_info_,
                                          rocsparse_analysis_policy_reuse,
                                          rocsparse_solve_policy_auto,
                                          this->mat_buffer_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        // Allocate temporary vector
        this->tmp_vec_->Allocate(this->nrow_);
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixBCSR<ValueType>::LUAnalyseClear(void)
    {
        rocsparse_status status;

        // Clear analysis info
        if(this->L_mat_descr_ != NULL)
        {
            status = rocsparse_bsrsv_clear(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                           this->mat_info_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        if(this->U_mat_descr_ != NULL)
        {
            status = rocsparse_bsrsv_clear(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                           this->mat_info_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        // Clear matrix descriptor
        if(this->L_mat_descr_ != NULL)
        {
            status = rocsparse_destroy_mat_descr(this->L_mat_descr_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        if(this->U_mat_descr_ != NULL)
        {
            status = rocsparse_destroy_mat_descr(this->U_mat_descr_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        this->L_mat_descr_ = 0;
        this->U_mat_descr_ = 0;

        // Clear buffer
        if(this->mat_buffer_ != NULL)
        {
            free_hip(&this->mat_buffer_);
            this->mat_buffer_ = NULL;
        }

        this->mat_buffer_size_ = 0;

        // Clear temporary vector
        if(this->tmp_vec_ != NULL)
        {
            delete this->tmp_vec_;
            this->tmp_vec_ = NULL;
        }
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixBCSR<ValueType>::LUSolve(const BaseVector<ValueType>& in,
                                                      BaseVector<ValueType>*       out) const
    {
        if(this->nnz_ > 0)
        {
            assert(this->L_mat_descr_ != 0);
            assert(this->U_mat_descr_ != 0);
            assert(this->mat_info_ != 0);

            assert(in.GetSize() >= 0);
            assert(out->GetSize() >= 0);
            assert(in.GetSize() == this->ncol_);
            assert(out->GetSize() == this->nrow_);
            assert(this->ncol_ == this->nrow_);

            assert(this->tmp_vec_ != NULL);

            const HIPAcceleratorVector<ValueType>* cast_in
                = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&in);
            HIPAcceleratorVector<ValueType>* cast_out
                = dynamic_cast<HIPAcceleratorVector<ValueType>*>(out);

            assert(cast_in != NULL);
            assert(cast_out != NULL);

            rocsparse_status status;

            ValueType alpha = static_cast<ValueType>(1);

            // Determine whether we are using row or column major for the blocks
            rocsparse_direction dir
                = BCSR_IND_BASE ? rocsparse_direction_row : rocsparse_direction_column;

            // Solve L
            status = rocsparseTbsrsv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                     dir,
                                     rocsparse_operation_none,
                                     this->mat_.nrowb,
                                     this->mat_.nnzb,
                                     &alpha,
                                     this->L_mat_descr_,
                                     this->mat_.val,
                                     this->mat_.row_offset,
                                     this->mat_.col,
                                     this->mat_.blockdim,
                                     this->mat_info_,
                                     cast_in->vec_,
                                     this->tmp_vec_->vec_,
                                     rocsparse_solve_policy_auto,
                                     this->mat_buffer_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            // Solve U
            status = rocsparseTbsrsv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                     dir,
                                     rocsparse_operation_none,
                                     this->mat_.nrowb,
                                     this->mat_.nnzb,
                                     &alpha,
                                     this->U_mat_descr_,
                                     this->mat_.val,
                                     this->mat_.row_offset,
                                     this->mat_.col,
                                     this->mat_.blockdim,
                                     this->mat_info_,
                                     this->tmp_vec_->vec_,
                                     cast_out->vec_,
                                     rocsparse_solve_policy_auto,
                                     this->mat_buffer_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        return true;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixBCSR<ValueType>::LLAnalyse(void)
    {
        assert(this->ncol_ == this->nrow_);
        assert(this->tmp_vec_ == NULL);

        this->tmp_vec_ = new HIPAcceleratorVector<ValueType>(this->local_backend_);

        assert(this->tmp_vec_ != NULL);

        rocsparse_status status;

        status = rocsparse_create_mat_descr(&this->L_mat_descr_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_type(this->L_mat_descr_, rocsparse_matrix_type_general);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_index_base(this->L_mat_descr_, rocsparse_index_base_zero);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_fill_mode(this->L_mat_descr_, rocsparse_fill_mode_lower);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_diag_type(this->L_mat_descr_, rocsparse_diag_type_non_unit);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        // Determine whether we are using row or column major for the blocks
        rocsparse_direction dir
            = BCSR_IND_BASE ? rocsparse_direction_row : rocsparse_direction_column;

        // Create buffer, if not already available
        size_t buffer_size_L  = 0;
        size_t buffer_size_Lt = 0;

        status
            = rocsparseTbsrsv_buffer_size(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                          dir,
                                          rocsparse_operation_none,
                                          this->mat_.nrowb,
                                          this->mat_.nnzb,
                                          this->L_mat_descr_,
                                          this->mat_.val,
                                          this->mat_.row_offset,
                                          this->mat_.col,
                                          this->mat_.blockdim,
                                          this->mat_info_,
                                          &buffer_size_L);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status
            = rocsparseTbsrsv_buffer_size(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                          dir,
                                          rocsparse_operation_transpose,
                                          this->mat_.nrowb,
                                          this->mat_.nnzb,
                                          this->L_mat_descr_,
                                          this->mat_.val,
                                          this->mat_.row_offset,
                                          this->mat_.col,
                                          this->mat_.blockdim,
                                          this->mat_info_,
                                          &buffer_size_Lt);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        size_t buffer_size = std::max(buffer_size_L, buffer_size_Lt);

        // Buffer is shared with ILU0, IC0 and other solve functions
        if(this->mat_buffer_ == NULL)
        {
            this->mat_buffer_size_ = buffer_size;
            allocate_hip(buffer_size, &this->mat_buffer_);
        }
        else if(this->mat_buffer_size_ < buffer_size)
        {
            this->mat_buffer_size_ = buffer_size;
            free_hip(&this->mat_buffer_);
            allocate_hip(buffer_size, &this->mat_buffer_);
        }

        assert(this->mat_buffer_size_ >= buffer_size);
        assert(this->mat_buffer_ != NULL);

        // L part analysis
        status = rocsparseTbsrsv_analysis(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                          dir,
                                          rocsparse_operation_none,
                                          this->mat_.nrowb,
                                          this->mat_.nnzb,
                                          this->L_mat_descr_,
                                          this->mat_.val,
                                          this->mat_.row_offset,
                                          this->mat_.col,
                                          this->mat_.blockdim,
                                          this->mat_info_,
                                          rocsparse_analysis_policy_reuse,
                                          rocsparse_solve_policy_auto,
                                          this->mat_buffer_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        // L^T part analysis
        status = rocsparseTbsrsv_analysis(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                          dir,
                                          rocsparse_operation_transpose,
                                          this->mat_.nrowb,
                                          this->mat_.nnzb,
                                          this->L_mat_descr_,
                                          this->mat_.val,
                                          this->mat_.row_offset,
                                          this->mat_.col,
                                          this->mat_.blockdim,
                                          this->mat_info_,
                                          rocsparse_analysis_policy_reuse,
                                          rocsparse_solve_policy_auto,
                                          this->mat_buffer_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        // Allocate temporary vector
        this->tmp_vec_->Allocate(this->nrow_);
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixBCSR<ValueType>::LLAnalyseClear(void)
    {
        rocsparse_status status;

        // Clear analysis info
        if(this->L_mat_descr_ != 0)
        {
            status = rocsparse_bsrsv_clear(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                           this->mat_info_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        // Clear matrix descriptor
        if(this->L_mat_descr_ != NULL)
        {
            status = rocsparse_destroy_mat_descr(this->L_mat_descr_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        this->L_mat_descr_ = 0;

        // Clear buffer
        if(this->mat_buffer_ != NULL)
        {
            free_hip(&this->mat_buffer_);
            this->mat_buffer_ = NULL;
        }

        this->mat_buffer_size_ = 0;

        // Clear temporary vector
        if(this->tmp_vec_ != NULL)
        {
            delete this->tmp_vec_;
            this->tmp_vec_ = NULL;
        }
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixBCSR<ValueType>::LLSolve(const BaseVector<ValueType>& in,
                                                      BaseVector<ValueType>*       out) const
    {
        if(this->nnz_ > 0)
        {
            assert(this->L_mat_descr_ != 0);
            assert(this->mat_info_ != 0);

            assert(in.GetSize() >= 0);
            assert(out->GetSize() >= 0);
            assert(in.GetSize() == this->ncol_);
            assert(out->GetSize() == this->nrow_);
            assert(this->ncol_ == this->nrow_);

            assert(this->tmp_vec_ != NULL);
            assert(this->mat_buffer_ != NULL);

            const HIPAcceleratorVector<ValueType>* cast_in
                = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&in);
            HIPAcceleratorVector<ValueType>* cast_out
                = dynamic_cast<HIPAcceleratorVector<ValueType>*>(out);

            assert(cast_in != NULL);
            assert(cast_out != NULL);

            rocsparse_status status;

            ValueType alpha = static_cast<ValueType>(1);

            // Determine whether we are using row or column major for the blocks
            rocsparse_direction dir
                = BCSR_IND_BASE ? rocsparse_direction_row : rocsparse_direction_column;

            // Solve L
            status = rocsparseTbsrsv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                     dir,
                                     rocsparse_operation_none,
                                     this->mat_.nrowb,
                                     this->mat_.nnzb,
                                     &alpha,
                                     this->L_mat_descr_,
                                     this->mat_.val,
                                     this->mat_.row_offset,
                                     this->mat_.col,
                                     this->mat_.blockdim,
                                     this->mat_info_,
                                     cast_in->vec_,
                                     this->tmp_vec_->vec_,
                                     rocsparse_solve_policy_auto,
                                     this->mat_buffer_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

            // Solve L^T
            status = rocsparseTbsrsv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                     dir,
                                     rocsparse_operation_transpose,
                                     this->mat_.nrowb,
                                     this->mat_.nnzb,
                                     &alpha,
                                     this->L_mat_descr_,
                                     this->mat_.val,
                                     this->mat_.row_offset,
                                     this->mat_.col,
                                     this->mat_.blockdim,
                                     this->mat_info_,
                                     this->tmp_vec_->vec_,
                                     cast_out->vec_,
                                     rocsparse_solve_policy_auto,
                                     this->mat_buffer_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixBCSR<ValueType>::LLSolve(const BaseVector<ValueType>& in,
                                                      const BaseVector<ValueType>& inv_diag,
                                                      BaseVector<ValueType>*       out) const
    {
        return LLSolve(in, out);
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixBCSR<ValueType>::LAnalyse(bool diag_unit)
    {
        rocsparse_status status;

        // L part
        status = rocsparse_create_mat_descr(&this->L_mat_descr_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_type(this->L_mat_descr_, rocsparse_matrix_type_general);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_index_base(this->L_mat_descr_, rocsparse_index_base_zero);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_fill_mode(this->L_mat_descr_, rocsparse_fill_mode_lower);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        if(diag_unit == true)
        {
            status = rocsparse_set_mat_diag_type(this->L_mat_descr_, rocsparse_diag_type_unit);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }
        else
        {
            status = rocsparse_set_mat_diag_type(this->L_mat_descr_, rocsparse_diag_type_non_unit);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        // Determine whether we are using row or column major for the blocks
        rocsparse_direction dir
            = BCSR_IND_BASE ? rocsparse_direction_row : rocsparse_direction_column;

        // Create buffer, if not already available
        size_t buffer_size = 0;
        status
            = rocsparseTbsrsv_buffer_size(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                          dir,
                                          rocsparse_operation_none,
                                          this->mat_.nrowb,
                                          this->mat_.nnzb,
                                          this->L_mat_descr_,
                                          this->mat_.val,
                                          this->mat_.row_offset,
                                          this->mat_.col,
                                          this->mat_.blockdim,
                                          this->mat_info_,
                                          &buffer_size);

        // Buffer is shared with ILU0 and other solve functions
        if(this->mat_buffer_ == NULL)
        {
            this->mat_buffer_size_ = buffer_size;
            allocate_hip(buffer_size, &this->mat_buffer_);
        }

        assert(this->mat_buffer_size_ >= buffer_size);
        assert(this->mat_buffer_ != NULL);

        status = rocsparseTbsrsv_analysis(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                          dir,
                                          rocsparse_operation_none,
                                          this->mat_.nrowb,
                                          this->mat_.nnzb,
                                          this->L_mat_descr_,
                                          this->mat_.val,
                                          this->mat_.row_offset,
                                          this->mat_.col,
                                          this->mat_.blockdim,
                                          this->mat_info_,
                                          rocsparse_analysis_policy_reuse,
                                          rocsparse_solve_policy_auto,
                                          this->mat_buffer_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixBCSR<ValueType>::UAnalyse(bool diag_unit)
    {
        rocsparse_status status;

        // U part
        status = rocsparse_create_mat_descr(&this->U_mat_descr_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_type(this->U_mat_descr_, rocsparse_matrix_type_general);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_index_base(this->U_mat_descr_, rocsparse_index_base_zero);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_fill_mode(this->U_mat_descr_, rocsparse_fill_mode_upper);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        if(diag_unit == true)
        {
            status = rocsparse_set_mat_diag_type(this->U_mat_descr_, rocsparse_diag_type_unit);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }
        else
        {
            status = rocsparse_set_mat_diag_type(this->U_mat_descr_, rocsparse_diag_type_non_unit);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        // Determine whether we are using row or column major for the blocks
        rocsparse_direction dir
            = BCSR_IND_BASE ? rocsparse_direction_row : rocsparse_direction_column;

        // Create buffer, if not already available
        size_t buffer_size = 0;
        status
            = rocsparseTbsrsv_buffer_size(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                          dir,
                                          rocsparse_operation_none,
                                          this->mat_.nrowb,
                                          this->mat_.nnzb,
                                          this->U_mat_descr_,
                                          this->mat_.val,
                                          this->mat_.row_offset,
                                          this->mat_.col,
                                          this->mat_.blockdim,
                                          this->mat_info_,
                                          &buffer_size);

        // Buffer is shared with ILU0 and other solve functions
        if(this->mat_buffer_ == NULL)
        {
            this->mat_buffer_size_ = buffer_size;
            allocate_hip(buffer_size, &this->mat_buffer_);
        }

        assert(this->mat_buffer_size_ >= buffer_size);
        assert(this->mat_buffer_ != NULL);

        status = rocsparseTbsrsv_analysis(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                          dir,
                                          rocsparse_operation_none,
                                          this->mat_.nrowb,
                                          this->mat_.nnzb,
                                          this->U_mat_descr_,
                                          this->mat_.val,
                                          this->mat_.row_offset,
                                          this->mat_.col,
                                          this->mat_.blockdim,
                                          this->mat_info_,
                                          rocsparse_analysis_policy_reuse,
                                          rocsparse_solve_policy_auto,
                                          this->mat_buffer_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixBCSR<ValueType>::LAnalyseClear(void)
    {
        rocsparse_status status;

        // Clear analysis info
        if(this->L_mat_descr_ != NULL)
        {
            status = rocsparse_bsrsv_clear(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                           this->mat_info_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        // Clear buffer
        if(this->mat_buffer_ != NULL)
        {
            free_hip(&this->mat_buffer_);
            this->mat_buffer_ = NULL;
        }

        this->mat_buffer_size_ = 0;

        // Clear matrix descriptor
        if(this->L_mat_descr_ != NULL)
        {
            status = rocsparse_destroy_mat_descr(this->L_mat_descr_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        this->L_mat_descr_ = 0;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixBCSR<ValueType>::UAnalyseClear(void)
    {
        rocsparse_status status;

        // Clear analysis info
        if(this->U_mat_descr_ != NULL)
        {
            status = rocsparse_bsrsv_clear(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                           this->mat_info_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        // Clear buffer
        if(this->mat_buffer_ != NULL)
        {
            free_hip(&this->mat_buffer_);
            this->mat_buffer_ = NULL;
        }

        this->mat_buffer_size_ = 0;

        // Clear matrix descriptor
        if(this->U_mat_descr_ != NULL)
        {
            status = rocsparse_destroy_mat_descr(this->U_mat_descr_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        this->U_mat_descr_ = 0;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixBCSR<ValueType>::LSolve(const BaseVector<ValueType>& in,
                                                     BaseVector<ValueType>*       out) const
    {
        if(this->nnz_ > 0)
        {
            assert(this->L_mat_descr_ != 0);
            assert(this->mat_info_ != 0);

            assert(in.GetSize() >= 0);
            assert(out->GetSize() >= 0);
            assert(in.GetSize() == this->ncol_);
            assert(out->GetSize() == this->nrow_);
            assert(this->ncol_ == this->nrow_);
            assert(this->mat_buffer_size_ > 0);
            assert(this->mat_buffer_ != NULL);

            const HIPAcceleratorVector<ValueType>* cast_in
                = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&in);
            HIPAcceleratorVector<ValueType>* cast_out
                = dynamic_cast<HIPAcceleratorVector<ValueType>*>(out);

            assert(cast_in != NULL);
            assert(cast_out != NULL);

            rocsparse_status status;

            ValueType alpha = static_cast<ValueType>(1);

            // Determine whether we are using row or column major for the blocks
            rocsparse_direction dir
                = BCSR_IND_BASE ? rocsparse_direction_row : rocsparse_direction_column;

            // Solve L
            status = rocsparseTbsrsv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                     dir,
                                     rocsparse_operation_none,
                                     this->mat_.nrowb,
                                     this->mat_.nnzb,
                                     &alpha,
                                     this->L_mat_descr_,
                                     this->mat_.val,
                                     this->mat_.row_offset,
                                     this->mat_.col,
                                     this->mat_.blockdim,
                                     this->mat_info_,
                                     cast_in->vec_,
                                     cast_out->vec_,
                                     rocsparse_solve_policy_auto,
                                     this->mat_buffer_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixBCSR<ValueType>::USolve(const BaseVector<ValueType>& in,
                                                     BaseVector<ValueType>*       out) const
    {
        if(this->nnz_ > 0)
        {
            assert(this->U_mat_descr_ != 0);
            assert(this->mat_info_ != 0);

            assert(in.GetSize() >= 0);
            assert(out->GetSize() >= 0);
            assert(in.GetSize() == this->ncol_);
            assert(out->GetSize() == this->nrow_);
            assert(this->ncol_ == this->nrow_);
            assert(this->mat_buffer_size_ > 0);
            assert(this->mat_buffer_ != NULL);

            const HIPAcceleratorVector<ValueType>* cast_in
                = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&in);
            HIPAcceleratorVector<ValueType>* cast_out
                = dynamic_cast<HIPAcceleratorVector<ValueType>*>(out);

            assert(cast_in != NULL);
            assert(cast_out != NULL);

            rocsparse_status status;

            ValueType alpha = static_cast<ValueType>(1);

            // Determine whether we are using row or column major for the blocks
            rocsparse_direction dir
                = BCSR_IND_BASE ? rocsparse_direction_row : rocsparse_direction_column;

            // Solve U
            status = rocsparseTbsrsv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                     dir,
                                     rocsparse_operation_none,
                                     this->mat_.nrowb,
                                     this->mat_.nnzb,
                                     &alpha,
                                     this->U_mat_descr_,
                                     this->mat_.val,
                                     this->mat_.row_offset,
                                     this->mat_.col,
                                     this->mat_.blockdim,
                                     this->mat_info_,
                                     cast_in->vec_,
                                     cast_out->vec_,
                                     rocsparse_solve_policy_auto,
                                     this->mat_buffer_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }

        return true;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixBCSR<ValueType>::Apply(const BaseVector<ValueType>& in,
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

            // Determine whether we are using row or column major for the blocks
            rocsparse_direction dir
                = BCSR_IND_BASE ? rocsparse_direction_row : rocsparse_direction_column;

            rocsparse_status status;
            status = rocsparseTbsrmv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                     dir,
                                     rocsparse_operation_none,
                                     this->mat_.nrowb,
                                     this->mat_.ncolb,
                                     this->mat_.nnzb,
                                     &alpha,
                                     this->mat_descr_,
                                     this->mat_.val,
                                     this->mat_.row_offset,
                                     this->mat_.col,
                                     this->mat_.blockdim,
                                     cast_in->vec_,
                                     &beta,
                                     cast_out->vec_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixBCSR<ValueType>::ApplyAdd(const BaseVector<ValueType>& in,
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

            // Determine whether we are using row or column major for the blocks
            rocsparse_direction dir
                = BCSR_IND_BASE ? rocsparse_direction_row : rocsparse_direction_column;

            rocsparse_status status;
            status = rocsparseTbsrmv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                     dir,
                                     rocsparse_operation_none,
                                     this->mat_.nrowb,
                                     this->mat_.ncolb,
                                     this->mat_.nnzb,
                                     &scalar,
                                     this->mat_descr_,
                                     this->mat_.val,
                                     this->mat_.row_offset,
                                     this->mat_.col,
                                     this->mat_.blockdim,
                                     cast_in->vec_,
                                     &beta,
                                     cast_out->vec_);
            CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
        }
    }

    template class HIPAcceleratorMatrixBCSR<double>;
    template class HIPAcceleratorMatrixBCSR<float>;
#ifdef SUPPORT_COMPLEX
    template class HIPAcceleratorMatrixBCSR<std::complex<double>>;
    template class HIPAcceleratorMatrixBCSR<std::complex<float>>;
#endif

} // namespace rocalution

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

#include "../../utils/def.hpp"

#include "hip_matrix_csr.hpp"
#include "hip_matrix_mcsr.hpp"
#include "hip_vector.hpp"

#include "../host/host_matrix_mcsr.hpp"

#include "../base_matrix.hpp"
#include "../base_vector.hpp"

#include "../backend_manager.hpp"

#include "../../utils/log.hpp"

#include "hip_utils.hpp"

#include "hip_kernels_general.hpp"
#include "hip_kernels_mcsr.hpp"

#include "hip_allocate_free.hpp"

#include "../matrix_formats_ind.hpp"

#include <hip/hip_runtime.h>

namespace rocalution
{

    template <typename ValueType>
    HIPAcceleratorMatrixMCSR<ValueType>::HIPAcceleratorMatrixMCSR()
    {
        // no default constructors
        LOG_INFO("no default constructor");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    HIPAcceleratorMatrixMCSR<ValueType>::HIPAcceleratorMatrixMCSR(
        const Rocalution_Backend_Descriptor& local_backend)
    {
        log_debug(this,
                  "HIPAcceleratorMatrixMCSR::HIPAcceleratorMatrixMCSR()",
                  "constructor with local_backend");

        this->mat_.row_offset = NULL;
        this->mat_.col        = NULL;
        this->mat_.val        = NULL;
        this->set_backend(local_backend);

        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    HIPAcceleratorMatrixMCSR<ValueType>::~HIPAcceleratorMatrixMCSR()
    {
        log_debug(this, "HIPAcceleratorMatrixMCSR::~HIPAcceleratorMatrixMCSR()", "destructor");

        this->Clear();
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixMCSR<ValueType>::Info(void) const
    {
        LOG_INFO("HIPAcceleratorMatrixMCSR<ValueType>");
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixMCSR<ValueType>::AllocateMCSR(int64_t nnz, int nrow, int ncol)
    {
        assert(nnz >= 0);
        assert(ncol >= 0);
        assert(nrow >= 0);

        this->Clear();

        allocate_hip(nrow + 1, &this->mat_.row_offset);
        allocate_hip(nnz, &this->mat_.col);
        allocate_hip(nnz, &this->mat_.val);

        set_to_zero_hip(this->local_backend_.HIP_block_size, nrow + 1, mat_.row_offset);
        set_to_zero_hip(this->local_backend_.HIP_block_size, nnz, mat_.col);
        set_to_zero_hip(this->local_backend_.HIP_block_size, nnz, mat_.val);

        this->nrow_ = nrow;
        this->ncol_ = ncol;
        this->nnz_  = nnz;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixMCSR<ValueType>::SetDataPtrMCSR(
        int** row_offset, int** col, ValueType** val, int64_t nnz, int nrow, int ncol)
    {
        assert(nnz >= 0);
        assert(nrow >= 0);
        assert(ncol >= 0);
        assert(*row_offset != NULL);

        if(nnz > 0)
        {
            assert(*col != NULL);
            assert(*val != NULL);
        }

        this->Clear();

        this->nrow_ = nrow;
        this->ncol_ = ncol;
        this->nnz_  = nnz;

        hipDeviceSynchronize();

        this->mat_.row_offset = *row_offset;
        this->mat_.col        = *col;
        this->mat_.val        = *val;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixMCSR<ValueType>::LeaveDataPtrMCSR(int**       row_offset,
                                                               int**       col,
                                                               ValueType** val)
    {
        assert(this->nrow_ >= 0);
        assert(this->ncol_ >= 0);
        assert(this->nnz_ >= 0);

        hipDeviceSynchronize();

        *row_offset = this->mat_.row_offset;
        *col        = this->mat_.col;
        *val        = this->mat_.val;

        this->mat_.row_offset = NULL;
        this->mat_.col        = NULL;
        this->mat_.val        = NULL;

        this->nrow_ = 0;
        this->ncol_ = 0;
        this->nnz_  = 0;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixMCSR<ValueType>::Clear()
    {
        free_hip(&this->mat_.row_offset);
        free_hip(&this->mat_.col);
        free_hip(&this->mat_.val);

        this->nrow_ = 0;
        this->ncol_ = 0;
        this->nnz_  = 0;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixMCSR<ValueType>::CopyFromHost(const HostMatrix<ValueType>& src)
    {
        const HostMatrixMCSR<ValueType>* cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == src.GetMatFormat());

        // CPU to HIP copy
        if((cast_mat = dynamic_cast<const HostMatrixMCSR<ValueType>*>(&src)) != NULL)
        {
            if(this->nnz_ == 0)
            {
                this->AllocateMCSR(cast_mat->nnz_, cast_mat->nrow_, cast_mat->ncol_);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

            copy_h2d(this->nrow_ + 1, cast_mat->mat_.row_offset, this->mat_.row_offset);
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
    void HIPAcceleratorMatrixMCSR<ValueType>::CopyToHost(HostMatrix<ValueType>* dst) const
    {
        HostMatrixMCSR<ValueType>* cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == dst->GetMatFormat());

        // HIP to CPU copy
        if((cast_mat = dynamic_cast<HostMatrixMCSR<ValueType>*>(dst)) != NULL)
        {
            cast_mat->set_backend(this->local_backend_);

            if(cast_mat->nnz_ == 0)
            {
                cast_mat->AllocateMCSR(this->nnz_, this->nrow_, this->ncol_);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

            copy_d2h(this->nrow_ + 1, this->mat_.row_offset, cast_mat->mat_.row_offset);
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
    void HIPAcceleratorMatrixMCSR<ValueType>::CopyFrom(const BaseMatrix<ValueType>& src)
    {
        const HIPAcceleratorMatrixMCSR<ValueType>* hip_cast_mat;
        const HostMatrix<ValueType>*               host_cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == src.GetMatFormat());

        // HIP to HIP copy
        if((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixMCSR<ValueType>*>(&src)) != NULL)
        {
            if(this->nnz_ == 0)
            {
                this->AllocateMCSR(hip_cast_mat->nnz_, hip_cast_mat->nrow_, hip_cast_mat->ncol_);
            }

            assert(this->nnz_ == hip_cast_mat->nnz_);
            assert(this->nrow_ == hip_cast_mat->nrow_);
            assert(this->ncol_ == hip_cast_mat->ncol_);

            copy_d2d(this->nrow_ + 1, hip_cast_mat->mat_.row_offset, this->mat_.row_offset);
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
    void HIPAcceleratorMatrixMCSR<ValueType>::CopyTo(BaseMatrix<ValueType>* dst) const
    {
        HIPAcceleratorMatrixMCSR<ValueType>* hip_cast_mat;
        HostMatrix<ValueType>*               host_cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == dst->GetMatFormat());

        // HIP to HIP copy
        if((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixMCSR<ValueType>*>(dst)) != NULL)
        {
            hip_cast_mat->set_backend(this->local_backend_);

            if(hip_cast_mat->nnz_ == 0)
            {
                hip_cast_mat->AllocateMCSR(this->nnz_, this->nrow_, this->ncol_);
            }

            assert(this->nnz_ == hip_cast_mat->nnz_);
            assert(this->nrow_ == hip_cast_mat->nrow_);
            assert(this->ncol_ == hip_cast_mat->ncol_);

            copy_d2d(this->nrow_ + 1, this->mat_.row_offset, hip_cast_mat->mat_.row_offset);
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
    void HIPAcceleratorMatrixMCSR<ValueType>::CopyFromHostAsync(const HostMatrix<ValueType>& src)
    {
        const HostMatrixMCSR<ValueType>* cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == src.GetMatFormat());

        // CPU to HIP copy
        if((cast_mat = dynamic_cast<const HostMatrixMCSR<ValueType>*>(&src)) != NULL)
        {
            if(this->nnz_ == 0)
            {
                this->AllocateMCSR(cast_mat->nnz_, cast_mat->nrow_, cast_mat->ncol_);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

            copy_h2d(this->nrow_ + 1,
                     cast_mat->mat_.row_offset,
                     this->mat_.row_offset,
                     true,
                     HIPSTREAM(this->local_backend_.HIP_stream_current));
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
    void HIPAcceleratorMatrixMCSR<ValueType>::CopyToHostAsync(HostMatrix<ValueType>* dst) const
    {
        HostMatrixMCSR<ValueType>* cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == dst->GetMatFormat());

        // HIP to CPU copy
        if((cast_mat = dynamic_cast<HostMatrixMCSR<ValueType>*>(dst)) != NULL)
        {
            cast_mat->set_backend(this->local_backend_);

            if(cast_mat->nnz_ == 0)
            {
                cast_mat->AllocateMCSR(this->nnz_, this->nrow_, this->ncol_);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

            copy_d2h(this->nrow_ + 1,
                     this->mat_.row_offset,
                     cast_mat->mat_.row_offset,
                     true,
                     HIPSTREAM(this->local_backend_.HIP_stream_current));
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
    void HIPAcceleratorMatrixMCSR<ValueType>::CopyFromAsync(const BaseMatrix<ValueType>& src)
    {
        const HIPAcceleratorMatrixMCSR<ValueType>* hip_cast_mat;
        const HostMatrix<ValueType>*               host_cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == src.GetMatFormat());

        // HIP to HIP copy
        if((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixMCSR<ValueType>*>(&src)) != NULL)
        {
            if(this->nnz_ == 0)
            {
                this->AllocateMCSR(hip_cast_mat->nnz_, hip_cast_mat->nrow_, hip_cast_mat->ncol_);
            }

            assert(this->nnz_ == hip_cast_mat->nnz_);
            assert(this->nrow_ == hip_cast_mat->nrow_);
            assert(this->ncol_ == hip_cast_mat->ncol_);

            copy_d2d(this->nrow_ + 1,
                     hip_cast_mat->mat_.row_offset,
                     this->mat_.row_offset,
                     true,
                     HIPSTREAM(this->local_backend_.HIP_stream_current));
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
    void HIPAcceleratorMatrixMCSR<ValueType>::CopyToAsync(BaseMatrix<ValueType>* dst) const
    {
        HIPAcceleratorMatrixMCSR<ValueType>* hip_cast_mat;
        HostMatrix<ValueType>*               host_cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == dst->GetMatFormat());

        // HIP to HIP copy
        if((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixMCSR<ValueType>*>(dst)) != NULL)
        {
            hip_cast_mat->set_backend(this->local_backend_);

            if(hip_cast_mat->nnz_ == 0)
            {
                hip_cast_mat->AllocateMCSR(this->nnz_, this->nrow_, this->ncol_);
            }

            assert(this->nnz_ == hip_cast_mat->nnz_);
            assert(this->nrow_ == hip_cast_mat->nrow_);
            assert(this->ncol_ == hip_cast_mat->ncol_);

            copy_d2h(this->nrow_ + 1,
                     this->mat_.row_offset,
                     hip_cast_mat->mat_.row_offset,
                     true,
                     HIPSTREAM(this->local_backend_.HIP_stream_current));
            copy_d2h(this->nnz_,
                     this->mat_.col,
                     hip_cast_mat->mat_.col,
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
    bool HIPAcceleratorMatrixMCSR<ValueType>::ConvertFrom(const BaseMatrix<ValueType>& mat)
    {
        this->Clear();

        // Empty matrix
        if(mat.GetNnz() == 0)
        {
            this->AllocateMCSR(0, mat.GetM(), mat.GetN());

            return true;
        }

        const HIPAcceleratorMatrixMCSR<ValueType>* cast_mat_mcsr;

        if((cast_mat_mcsr = dynamic_cast<const HIPAcceleratorMatrixMCSR<ValueType>*>(&mat)) != NULL)
        {
            this->CopyFrom(*cast_mat_mcsr);
            return true;
        }

        return false;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixMCSR<ValueType>::Apply(const BaseVector<ValueType>& in,
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

            dim3 BlockSize(512);
            dim3 GridSize((this->nrow_ - 1) / 512 + 1);

            int nnz_per_row = this->nnz_ / this->nrow_;

            if(this->local_backend_.HIP_warp == 32)
            {
                if(nnz_per_row < 4)
                {
                    kernel_mcsr_spmv<2><<<GridSize,
                                          BlockSize,
                                          0,
                                          HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->mat_.row_offset,
                        this->mat_.col,
                        this->mat_.val,
                        cast_in->vec_,
                        cast_out->vec_);
                }
                else if(nnz_per_row < 8)
                {
                    kernel_mcsr_spmv<4><<<GridSize,
                                          BlockSize,
                                          0,
                                          HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->mat_.row_offset,
                        this->mat_.col,
                        this->mat_.val,
                        cast_in->vec_,
                        cast_out->vec_);
                }
                else if(nnz_per_row < 16)
                {
                    kernel_mcsr_spmv<8><<<GridSize,
                                          BlockSize,
                                          0,
                                          HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->mat_.row_offset,
                        this->mat_.col,
                        this->mat_.val,
                        cast_in->vec_,
                        cast_out->vec_);
                }
                else if(nnz_per_row < 32)
                {
                    kernel_mcsr_spmv<16><<<GridSize,
                                           BlockSize,
                                           0,
                                           HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->mat_.row_offset,
                        this->mat_.col,
                        this->mat_.val,
                        cast_in->vec_,
                        cast_out->vec_);
                }
                else
                {
                    kernel_mcsr_spmv<32><<<GridSize,
                                           BlockSize,
                                           0,
                                           HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->mat_.row_offset,
                        this->mat_.col,
                        this->mat_.val,
                        cast_in->vec_,
                        cast_out->vec_);
                }
            }
            else if(this->local_backend_.HIP_warp == 64)
            {
                if(nnz_per_row < 4)
                {
                    kernel_mcsr_spmv<2><<<GridSize,
                                          BlockSize,
                                          0,
                                          HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->mat_.row_offset,
                        this->mat_.col,
                        this->mat_.val,
                        cast_in->vec_,
                        cast_out->vec_);
                }
                else if(nnz_per_row < 8)
                {
                    kernel_mcsr_spmv<4><<<GridSize,
                                          BlockSize,
                                          0,
                                          HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->mat_.row_offset,
                        this->mat_.col,
                        this->mat_.val,
                        cast_in->vec_,
                        cast_out->vec_);
                }
                else if(nnz_per_row < 16)
                {
                    kernel_mcsr_spmv<8><<<GridSize,
                                          BlockSize,
                                          0,
                                          HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->mat_.row_offset,
                        this->mat_.col,
                        this->mat_.val,
                        cast_in->vec_,
                        cast_out->vec_);
                }
                else if(nnz_per_row < 32)
                {
                    kernel_mcsr_spmv<16><<<GridSize,
                                           BlockSize,
                                           0,
                                           HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->mat_.row_offset,
                        this->mat_.col,
                        this->mat_.val,
                        cast_in->vec_,
                        cast_out->vec_);
                }
                else if(nnz_per_row < 64)
                {
                    kernel_mcsr_spmv<32><<<GridSize,
                                           BlockSize,
                                           0,
                                           HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->mat_.row_offset,
                        this->mat_.col,
                        this->mat_.val,
                        cast_in->vec_,
                        cast_out->vec_);
                }
                else
                {
                    kernel_mcsr_spmv<64><<<GridSize,
                                           BlockSize,
                                           0,
                                           HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->mat_.row_offset,
                        this->mat_.col,
                        this->mat_.val,
                        cast_in->vec_,
                        cast_out->vec_);
                }
            }
            else
            {
                LOG_INFO("Unsupported HIP warp size of " << this->local_backend_.HIP_warp);
                FATAL_ERROR(__FILE__, __LINE__);
            }

            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixMCSR<ValueType>::ApplyAdd(const BaseVector<ValueType>& in,
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

            dim3 BlockSize(512);
            dim3 GridSize((this->nrow_ - 1) / 512 + 1);

            int nnz_per_row = this->nnz_ / this->nrow_;

            if(this->local_backend_.HIP_warp == 32)
            {
                if(nnz_per_row < 4)
                {
                    kernel_mcsr_add_spmv<2><<<GridSize,
                                              BlockSize,
                                              0,
                                              HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->mat_.row_offset,
                        this->mat_.col,
                        this->mat_.val,
                        scalar,
                        cast_in->vec_,
                        cast_out->vec_);
                }
                else if(nnz_per_row < 8)
                {
                    kernel_mcsr_add_spmv<4><<<GridSize,
                                              BlockSize,
                                              0,
                                              HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->mat_.row_offset,
                        this->mat_.col,
                        this->mat_.val,
                        scalar,
                        cast_in->vec_,
                        cast_out->vec_);
                }
                else if(nnz_per_row < 16)
                {
                    kernel_mcsr_add_spmv<8><<<GridSize,
                                              BlockSize,
                                              0,
                                              HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->mat_.row_offset,
                        this->mat_.col,
                        this->mat_.val,
                        scalar,
                        cast_in->vec_,
                        cast_out->vec_);
                }
                else if(nnz_per_row < 32)
                {
                    kernel_mcsr_add_spmv<16>
                        <<<GridSize,
                           BlockSize,
                           0,
                           HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                            this->nrow_,
                            this->mat_.row_offset,
                            this->mat_.col,
                            this->mat_.val,
                            scalar,
                            cast_in->vec_,
                            cast_out->vec_);
                }
                else
                {
                    kernel_mcsr_add_spmv<32>
                        <<<GridSize,
                           BlockSize,
                           0,
                           HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                            this->nrow_,
                            this->mat_.row_offset,
                            this->mat_.col,
                            this->mat_.val,
                            scalar,
                            cast_in->vec_,
                            cast_out->vec_);
                }
            }
            else if(this->local_backend_.HIP_warp == 64)
            {
                if(nnz_per_row < 4)
                {
                    kernel_mcsr_add_spmv<2><<<GridSize,
                                              BlockSize,
                                              0,
                                              HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->mat_.row_offset,
                        this->mat_.col,
                        this->mat_.val,
                        scalar,
                        cast_in->vec_,
                        cast_out->vec_);
                }
                else if(nnz_per_row < 8)
                {
                    kernel_mcsr_add_spmv<4><<<GridSize,
                                              BlockSize,
                                              0,
                                              HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->mat_.row_offset,
                        this->mat_.col,
                        this->mat_.val,
                        scalar,
                        cast_in->vec_,
                        cast_out->vec_);
                }
                else if(nnz_per_row < 16)
                {
                    kernel_mcsr_add_spmv<8><<<GridSize,
                                              BlockSize,
                                              0,
                                              HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                        this->nrow_,
                        this->mat_.row_offset,
                        this->mat_.col,
                        this->mat_.val,
                        scalar,
                        cast_in->vec_,
                        cast_out->vec_);
                }
                else if(nnz_per_row < 32)
                {
                    kernel_mcsr_add_spmv<16>
                        <<<GridSize,
                           BlockSize,
                           0,
                           HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                            this->nrow_,
                            this->mat_.row_offset,
                            this->mat_.col,
                            this->mat_.val,
                            scalar,
                            cast_in->vec_,
                            cast_out->vec_);
                }
                else if(nnz_per_row < 64)
                {
                    kernel_mcsr_add_spmv<32>
                        <<<GridSize,
                           BlockSize,
                           0,
                           HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                            this->nrow_,
                            this->mat_.row_offset,
                            this->mat_.col,
                            this->mat_.val,
                            scalar,
                            cast_in->vec_,
                            cast_out->vec_);
                }
                else
                {
                    kernel_mcsr_add_spmv<64>
                        <<<GridSize,
                           BlockSize,
                           0,
                           HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                            this->nrow_,
                            this->mat_.row_offset,
                            this->mat_.col,
                            this->mat_.val,
                            scalar,
                            cast_in->vec_,
                            cast_out->vec_);
                }
            }
            else
            {
                LOG_INFO("Unsupported HIP warp size of " << this->local_backend_.HIP_warp);
                FATAL_ERROR(__FILE__, __LINE__);
            }

            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }
    }

    template class HIPAcceleratorMatrixMCSR<double>;
    template class HIPAcceleratorMatrixMCSR<float>;
#ifdef SUPPORT_COMPLEX
    template class HIPAcceleratorMatrixMCSR<std::complex<double>>;
    template class HIPAcceleratorMatrixMCSR<std::complex<float>>;
#endif

} // namespace rocalution

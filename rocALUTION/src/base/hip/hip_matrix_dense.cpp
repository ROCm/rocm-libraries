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

#include "hip_matrix_dense.hpp"
#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "../backend_manager.hpp"
#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../host/host_matrix_dense.hpp"
#include "../matrix_formats_ind.hpp"
#include "hip_allocate_free.hpp"
#include "hip_blas.hpp"
#include "hip_kernels_dense.hpp"
#include "hip_kernels_general.hpp"
#include "hip_matrix_bcsr.hpp"
#include "hip_matrix_coo.hpp"
#include "hip_matrix_csr.hpp"
#include "hip_matrix_dia.hpp"
#include "hip_matrix_ell.hpp"
#include "hip_matrix_hyb.hpp"
#include "hip_matrix_mcsr.hpp"
#include "hip_utils.hpp"
#include "hip_vector.hpp"

#include "hip_conversion.hpp"

#include <hip/hip_runtime.h>

namespace rocalution
{

    template <typename ValueType>
    HIPAcceleratorMatrixDENSE<ValueType>::HIPAcceleratorMatrixDENSE()
    {
        // no default constructors
        LOG_INFO("no default constructor");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    HIPAcceleratorMatrixDENSE<ValueType>::HIPAcceleratorMatrixDENSE(
        const Rocalution_Backend_Descriptor& local_backend)
    {
        log_debug(this,
                  "HIPAcceleratorMatrixDENSE::HIPAcceleratorMatrixDENSE()",
                  "constructor with local_backend");

        this->mat_.val = NULL;
        this->set_backend(local_backend);

        CHECK_HIP_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    HIPAcceleratorMatrixDENSE<ValueType>::~HIPAcceleratorMatrixDENSE()
    {
        log_debug(this, "HIPAcceleratorMatrixDENSE::~HIPAcceleratorMatrixDENSE()", "destructor");

        this->Clear();
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixDENSE<ValueType>::Info(void) const
    {
        LOG_INFO("HIPAcceleratorMatrixDENSE<ValueType>");
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixDENSE<ValueType>::AllocateDENSE(int nrow, int ncol)
    {
        assert(ncol >= 0);
        assert(nrow >= 0);

        this->Clear();

        allocate_hip(nrow * ncol, &this->mat_.val);

        set_to_zero_hip(this->local_backend_.HIP_block_size, nrow * ncol, mat_.val);

        this->nrow_ = nrow;
        this->ncol_ = ncol;
        this->nnz_  = nrow * ncol;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixDENSE<ValueType>::Clear()
    {
        free_hip(&this->mat_.val);

        this->nrow_ = 0;
        this->ncol_ = 0;
        this->nnz_  = 0;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixDENSE<ValueType>::SetDataPtrDENSE(ValueType** val, int nrow, int ncol)
    {
        assert(nrow >= 0);
        assert(ncol >= 0);

        if(nrow * ncol > 0)
        {
            assert(*val != NULL);
        }

        this->Clear();

        hipDeviceSynchronize();

        this->nrow_ = nrow;
        this->ncol_ = ncol;
        this->nnz_  = nrow * ncol;

        this->mat_.val = *val;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixDENSE<ValueType>::LeaveDataPtrDENSE(ValueType** val)
    {
        assert(this->nrow_ >= 0);
        assert(this->ncol_ >= 0);
        assert(this->nnz_ >= 0);
        assert(this->nnz_ == this->nrow_ * this->ncol_);

        hipDeviceSynchronize();

        *val = this->mat_.val;

        this->mat_.val = NULL;

        this->nrow_ = 0;
        this->ncol_ = 0;
        this->nnz_  = 0;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixDENSE<ValueType>::CopyFromHost(const HostMatrix<ValueType>& src)
    {
        const HostMatrixDENSE<ValueType>* cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == src.GetMatFormat());

        // CPU to HIP copy
        if((cast_mat = dynamic_cast<const HostMatrixDENSE<ValueType>*>(&src)) != NULL)
        {
            if(this->nnz_ == 0)
            {
                this->AllocateDENSE(cast_mat->nrow_, cast_mat->ncol_);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

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
    void HIPAcceleratorMatrixDENSE<ValueType>::CopyToHost(HostMatrix<ValueType>* dst) const
    {
        HostMatrixDENSE<ValueType>* cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == dst->GetMatFormat());

        // HIP to CPU copy
        if((cast_mat = dynamic_cast<HostMatrixDENSE<ValueType>*>(dst)) != NULL)
        {
            cast_mat->set_backend(this->local_backend_);

            if(cast_mat->nnz_ == 0)
            {
                cast_mat->AllocateDENSE(this->nrow_, this->ncol_);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

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
    void HIPAcceleratorMatrixDENSE<ValueType>::CopyFrom(const BaseMatrix<ValueType>& src)
    {
        const HIPAcceleratorMatrixDENSE<ValueType>* hip_cast_mat;
        const HostMatrix<ValueType>*                host_cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == src.GetMatFormat());

        // HIP to HIP copy
        if((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixDENSE<ValueType>*>(&src)) != NULL)
        {
            if(this->nnz_ == 0)
            {
                this->AllocateDENSE(hip_cast_mat->nrow_, hip_cast_mat->ncol_);
            }

            assert(this->nnz_ == hip_cast_mat->nnz_);
            assert(this->nrow_ == hip_cast_mat->nrow_);
            assert(this->ncol_ == hip_cast_mat->ncol_);

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
    void HIPAcceleratorMatrixDENSE<ValueType>::CopyTo(BaseMatrix<ValueType>* dst) const
    {
        HIPAcceleratorMatrixDENSE<ValueType>* hip_cast_mat;
        HostMatrix<ValueType>*                host_cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == dst->GetMatFormat());

        // HIP to HIP copy
        if((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixDENSE<ValueType>*>(dst)) != NULL)
        {
            hip_cast_mat->set_backend(this->local_backend_);

            if(hip_cast_mat->nnz_ == 0)
            {
                hip_cast_mat->AllocateDENSE(this->nrow_, this->ncol_);
            }

            assert(this->nnz_ == hip_cast_mat->nnz_);
            assert(this->nrow_ == hip_cast_mat->nrow_);
            assert(this->ncol_ == hip_cast_mat->ncol_);

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
    void HIPAcceleratorMatrixDENSE<ValueType>::CopyFromHostAsync(const HostMatrix<ValueType>& src)
    {
        const HostMatrixDENSE<ValueType>* cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == src.GetMatFormat());

        // CPU to HIP copy
        if((cast_mat = dynamic_cast<const HostMatrixDENSE<ValueType>*>(&src)) != NULL)
        {
            if(this->nnz_ == 0)
            {
                this->AllocateDENSE(cast_mat->nrow_, cast_mat->ncol_);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

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
    void HIPAcceleratorMatrixDENSE<ValueType>::CopyToHostAsync(HostMatrix<ValueType>* dst) const
    {
        HostMatrixDENSE<ValueType>* cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == dst->GetMatFormat());

        // HIP to CPU copy
        if((cast_mat = dynamic_cast<HostMatrixDENSE<ValueType>*>(dst)) != NULL)
        {
            cast_mat->set_backend(this->local_backend_);

            if(cast_mat->nnz_ == 0)
            {
                cast_mat->AllocateDENSE(this->nrow_, this->ncol_);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

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
    void HIPAcceleratorMatrixDENSE<ValueType>::CopyFromAsync(const BaseMatrix<ValueType>& src)
    {
        const HIPAcceleratorMatrixDENSE<ValueType>* hip_cast_mat;
        const HostMatrix<ValueType>*                host_cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == src.GetMatFormat());

        // HIP to HIP copy
        if((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixDENSE<ValueType>*>(&src)) != NULL)
        {
            if(this->nnz_ == 0)
            {
                this->AllocateDENSE(hip_cast_mat->nrow_, hip_cast_mat->ncol_);
            }

            assert(this->nnz_ == hip_cast_mat->nnz_);
            assert(this->nrow_ == hip_cast_mat->nrow_);
            assert(this->ncol_ == hip_cast_mat->ncol_);

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
    void HIPAcceleratorMatrixDENSE<ValueType>::CopyToAsync(BaseMatrix<ValueType>* dst) const
    {
        HIPAcceleratorMatrixDENSE<ValueType>* hip_cast_mat;
        HostMatrix<ValueType>*                host_cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == dst->GetMatFormat());

        // HIP to HIP copy
        if((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixDENSE<ValueType>*>(dst)) != NULL)
        {
            hip_cast_mat->set_backend(this->local_backend_);

            if(hip_cast_mat->nnz_ == 0)
            {
                hip_cast_mat->AllocateDENSE(this->nrow_, this->ncol_);
            }

            assert(this->nnz_ == hip_cast_mat->nnz_);
            assert(this->nrow_ == hip_cast_mat->nrow_);
            assert(this->ncol_ == hip_cast_mat->ncol_);

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
    bool HIPAcceleratorMatrixDENSE<ValueType>::ConvertFrom(const BaseMatrix<ValueType>& mat)
    {
        this->Clear();

        // Empty matrix
        if(mat.GetNnz() == 0)
        {
            this->AllocateDENSE(mat.GetM(), mat.GetN());

            return true;
        }

        const HIPAcceleratorMatrixDENSE<ValueType>* cast_mat_dense;

        if((cast_mat_dense = dynamic_cast<const HIPAcceleratorMatrixDENSE<ValueType>*>(&mat))
           != NULL)
        {
            this->CopyFrom(*cast_mat_dense);
            return true;
        }

        const HIPAcceleratorMatrixCSR<ValueType>* cast_mat_csr;
        if((cast_mat_csr = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*>(&mat)) != NULL)
        {
            this->Clear();

            if(csr_to_dense_hip(&this->local_backend_,
                                cast_mat_csr->nrow_,
                                cast_mat_csr->ncol_,
                                cast_mat_csr->mat_,
                                cast_mat_csr->mat_descr_,
                                &this->mat_)
               == true)
            {
                this->nrow_ = cast_mat_csr->nrow_;
                this->ncol_ = cast_mat_csr->ncol_;
                this->nnz_  = this->nrow_ * this->ncol_;

                return true;
            }
        }

        return false;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixDENSE<ValueType>::Apply(const BaseVector<ValueType>& in,
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

            rocblas_status status;

            ValueType alpha = static_cast<ValueType>(1);
            ValueType beta  = static_cast<ValueType>(0);

            if(DENSE_IND_BASE == 0)
            {
                status = rocblasTgemv(ROCBLAS_HANDLE(this->local_backend_.ROC_blas_handle),
                                      rocblas_operation_none,
                                      this->nrow_,
                                      this->ncol_,
                                      &alpha,
                                      this->mat_.val,
                                      this->nrow_,
                                      cast_in->vec_,
                                      1,
                                      &beta,
                                      cast_out->vec_,
                                      1);
                CHECK_ROCBLAS_ERROR(status, __FILE__, __LINE__);
            }
            else
            {
                status = rocblasTgemv(ROCBLAS_HANDLE(this->local_backend_.ROC_blas_handle),
                                      rocblas_operation_transpose,
                                      this->ncol_,
                                      this->nrow_,
                                      &alpha,
                                      this->mat_.val,
                                      this->ncol_,
                                      cast_in->vec_,
                                      1,
                                      &beta,
                                      cast_out->vec_,
                                      1);
                CHECK_ROCBLAS_ERROR(status, __FILE__, __LINE__);
            }
        }
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixDENSE<ValueType>::ApplyAdd(const BaseVector<ValueType>& in,
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

            rocblas_status status;

            ValueType beta = static_cast<ValueType>(0);

            if(DENSE_IND_BASE == 0)
            {
                status = rocblasTgemv(ROCBLAS_HANDLE(this->local_backend_.ROC_blas_handle),
                                      rocblas_operation_none,
                                      this->nrow_,
                                      this->ncol_,
                                      &scalar,
                                      this->mat_.val,
                                      this->nrow_,
                                      cast_in->vec_,
                                      1,
                                      &beta,
                                      cast_out->vec_,
                                      1);
                CHECK_ROCBLAS_ERROR(status, __FILE__, __LINE__);
            }
            else
            {
                status = rocblasTgemv(ROCBLAS_HANDLE(this->local_backend_.ROC_blas_handle),
                                      rocblas_operation_transpose,
                                      this->ncol_,
                                      this->nrow_,
                                      &scalar,
                                      this->mat_.val,
                                      this->ncol_,
                                      cast_in->vec_,
                                      1,
                                      &beta,
                                      cast_out->vec_,
                                      1);
                CHECK_ROCBLAS_ERROR(status, __FILE__, __LINE__);
            }
        }
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixDENSE<ValueType>::MatMatMult(const BaseMatrix<ValueType>& A,
                                                          const BaseMatrix<ValueType>& B)
    {
        assert((this != &A) && (this != &B));

        const HIPAcceleratorMatrixDENSE<ValueType>* cast_mat_A
            = dynamic_cast<const HIPAcceleratorMatrixDENSE<ValueType>*>(&A);
        const HIPAcceleratorMatrixDENSE<ValueType>* cast_mat_B
            = dynamic_cast<const HIPAcceleratorMatrixDENSE<ValueType>*>(&B);

        assert(cast_mat_A != NULL);
        assert(cast_mat_B != NULL);
        assert(cast_mat_A->ncol_ == cast_mat_B->nrow_);

        rocblas_status status;

        ValueType alpha = static_cast<ValueType>(1);
        ValueType beta  = static_cast<ValueType>(0);

        if(DENSE_IND_BASE == 0)
        {
            status = rocblasTgemm(ROCBLAS_HANDLE(this->local_backend_.ROC_blas_handle),
                                  rocblas_operation_none,
                                  rocblas_operation_none,
                                  cast_mat_A->nrow_,
                                  cast_mat_B->ncol_,
                                  cast_mat_A->ncol_,
                                  &alpha,
                                  cast_mat_A->mat_.val,
                                  cast_mat_A->nrow_,
                                  cast_mat_B->mat_.val,
                                  cast_mat_A->ncol_,
                                  &beta,
                                  this->mat_.val,
                                  cast_mat_A->nrow_);
            CHECK_ROCBLAS_ERROR(status, __FILE__, __LINE__);
        }
        else
        {
            status = rocblasTgemm(ROCBLAS_HANDLE(this->local_backend_.ROC_blas_handle),
                                  rocblas_operation_transpose,
                                  rocblas_operation_transpose,
                                  cast_mat_A->nrow_,
                                  cast_mat_B->ncol_,
                                  cast_mat_A->ncol_,
                                  &alpha,
                                  cast_mat_A->mat_.val,
                                  cast_mat_A->ncol_,
                                  cast_mat_B->mat_.val,
                                  cast_mat_B->ncol_,
                                  &beta,
                                  this->mat_.val,
                                  cast_mat_A->nrow_);
            CHECK_ROCBLAS_ERROR(status, __FILE__, __LINE__);
        }

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixDENSE<ValueType>::ReplaceColumnVector(int                          idx,
                                                                   const BaseVector<ValueType>& vec)
    {
        assert(vec.GetSize() == this->nrow_);

        if(this->nnz_ > 0)
        {
            const HIPAcceleratorVector<ValueType>* cast_vec
                = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&vec);
            assert(cast_vec != NULL);

            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(this->nrow_ / this->local_backend_.HIP_block_size + 1);

            kernel_dense_replace_column_vector<<<GridSize,
                                                 BlockSize,
                                                 0,
                                                 HIPSTREAM(
                                                     this->local_backend_.HIP_stream_current)>>>(
                cast_vec->vec_, idx, this->nrow_, this->ncol_, this->mat_.val);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixDENSE<ValueType>::ReplaceRowVector(int                          idx,
                                                                const BaseVector<ValueType>& vec)
    {
        assert(vec.GetSize() == this->ncol_);

        if(this->nnz_ > 0)
        {
            const HIPAcceleratorVector<ValueType>* cast_vec
                = dynamic_cast<const HIPAcceleratorVector<ValueType>*>(&vec);
            assert(cast_vec != NULL);

            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(this->ncol_ / this->local_backend_.HIP_block_size + 1);

            kernel_dense_replace_row_vector<<<GridSize,
                                              BlockSize,
                                              0,
                                              HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                cast_vec->vec_, idx, this->nrow_, this->ncol_, this->mat_.val);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixDENSE<ValueType>::ExtractColumnVector(int                    idx,
                                                                   BaseVector<ValueType>* vec) const
    {
        assert(vec != NULL);
        assert(vec->GetSize() == this->nrow_);

        if(this->nnz_ > 0)
        {
            HIPAcceleratorVector<ValueType>* cast_vec
                = dynamic_cast<HIPAcceleratorVector<ValueType>*>(vec);
            assert(cast_vec != NULL);

            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(this->nrow_ / this->local_backend_.HIP_block_size + 1);

            kernel_dense_extract_column_vector<<<GridSize,
                                                 BlockSize,
                                                 0,
                                                 HIPSTREAM(
                                                     this->local_backend_.HIP_stream_current)>>>(
                cast_vec->vec_, idx, this->nrow_, this->ncol_, this->mat_.val);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }

        return true;
    }

    template <typename ValueType>
    bool HIPAcceleratorMatrixDENSE<ValueType>::ExtractRowVector(int                    idx,
                                                                BaseVector<ValueType>* vec) const
    {
        assert(vec != NULL);
        assert(vec->GetSize() == this->ncol_);

        if(this->nnz_ > 0)
        {
            HIPAcceleratorVector<ValueType>* cast_vec
                = dynamic_cast<HIPAcceleratorVector<ValueType>*>(vec);
            assert(cast_vec != NULL);

            dim3 BlockSize(this->local_backend_.HIP_block_size);
            dim3 GridSize(this->ncol_ / this->local_backend_.HIP_block_size + 1);

            kernel_dense_extract_row_vector<<<GridSize,
                                              BlockSize,
                                              0,
                                              HIPSTREAM(this->local_backend_.HIP_stream_current)>>>(
                cast_vec->vec_, idx, this->nrow_, this->ncol_, this->mat_.val);
            CHECK_HIP_ERROR(__FILE__, __LINE__);
        }

        return true;
    }

    template class HIPAcceleratorMatrixDENSE<double>;
    template class HIPAcceleratorMatrixDENSE<float>;
#ifdef SUPPORT_COMPLEX
    template class HIPAcceleratorMatrixDENSE<std::complex<double>>;
    template class HIPAcceleratorMatrixDENSE<std::complex<float>>;
#endif

} // namespace rocalution

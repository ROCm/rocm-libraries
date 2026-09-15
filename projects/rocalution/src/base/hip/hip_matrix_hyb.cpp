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

#include "hip_matrix_hyb.hpp"
#include "../../utils/allocate_free.hpp"
#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "../backend_manager.hpp"
#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../host/host_matrix_hyb.hpp"
#include "../matrix_formats_ind.hpp"
#include "hip_allocate_free.hpp"
#include "hip_conversion.hpp"
#include "hip_kernels_general.hpp"
#include "hip_kernels_vector.hpp"
#include "hip_matrix_coo.hpp"
#include "hip_matrix_csr.hpp"
#include "hip_matrix_ell.hpp"
#include "hip_sparse.hpp"
#include "hip_utils.hpp"
#include "hip_vector.hpp"

#include <algorithm>
#include <hip/hip_runtime.h>

namespace rocalution
{

    template <typename ValueType>
    HIPAcceleratorMatrixHYB<ValueType>::HIPAcceleratorMatrixHYB()
    {
        // no default constructors
        LOG_INFO("no default constructor");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <typename ValueType>
    HIPAcceleratorMatrixHYB<ValueType>::HIPAcceleratorMatrixHYB(
        const Rocalution_Backend_Descriptor& local_backend)
    {
        log_debug(this,
                  "HIPAcceleratorMatrixHYB::HIPAcceleratorMatrixHYB()",
                  "constructor with local_backend");

        this->mat_.ELL.val     = NULL;
        this->mat_.ELL.col     = NULL;
        this->mat_.ELL.max_row = 0;

        this->mat_.COO.row = NULL;
        this->mat_.COO.col = NULL;
        this->mat_.COO.val = NULL;

        this->ell_nnz_ = 0;
        this->coo_nnz_ = 0;

        this->set_backend(local_backend);

        this->ell_mat_descr_ = 0;
        this->coo_mat_descr_ = 0;

        CHECK_HIP_ERROR(__FILE__, __LINE__);

        rocsparse_status status;

        status = rocsparse_create_mat_descr(&this->ell_mat_descr_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_index_base(this->ell_mat_descr_, rocsparse_index_base_zero);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_type(this->ell_mat_descr_, rocsparse_matrix_type_general);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_create_mat_descr(&this->coo_mat_descr_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_index_base(this->coo_mat_descr_, rocsparse_index_base_zero);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_set_mat_type(this->coo_mat_descr_, rocsparse_matrix_type_general);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    template <typename ValueType>
    HIPAcceleratorMatrixHYB<ValueType>::~HIPAcceleratorMatrixHYB()
    {
        log_debug(this, "HIPAcceleratorMatrixHYB::~HIPAcceleratorMatrixHYB()", "destructor");

        this->Clear();

        rocsparse_status status;

        status = rocsparse_destroy_mat_descr(this->ell_mat_descr_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);

        status = rocsparse_destroy_mat_descr(this->coo_mat_descr_);
        CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixHYB<ValueType>::Info(void) const
    {
        LOG_INFO("HIPAcceleratorMatrixHYB<ValueType>");
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixHYB<ValueType>::AllocateHYB(
        int64_t ell_nnz, int64_t coo_nnz, int ell_max_row, int nrow, int ncol)
    {
        assert(ell_nnz >= 0);
        assert(coo_nnz >= 0);
        assert(ell_max_row >= 0);

        assert(ncol >= 0);
        assert(nrow >= 0);

        this->Clear();

        this->nnz_  = 0;
        this->nrow_ = nrow;
        this->ncol_ = ncol;

        // ELL
        assert(ell_nnz == ell_max_row * nrow);

        allocate_hip(ell_nnz, &this->mat_.ELL.val);
        allocate_hip(ell_nnz, &this->mat_.ELL.col);

        set_to_zero_hip(this->local_backend_.HIP_block_size, ell_nnz, this->mat_.ELL.val);
        set_to_zero_hip(this->local_backend_.HIP_block_size, ell_nnz, this->mat_.ELL.col);

        this->mat_.ELL.max_row = ell_max_row;
        this->ell_nnz_         = ell_nnz;
        this->nnz_ += ell_nnz;

        // COO
        allocate_hip(coo_nnz, &this->mat_.COO.row);
        allocate_hip(coo_nnz, &this->mat_.COO.col);
        allocate_hip(coo_nnz, &this->mat_.COO.val);

        set_to_zero_hip(this->local_backend_.HIP_block_size, coo_nnz, this->mat_.COO.row);
        set_to_zero_hip(this->local_backend_.HIP_block_size, coo_nnz, this->mat_.COO.col);
        set_to_zero_hip(this->local_backend_.HIP_block_size, coo_nnz, this->mat_.COO.val);
        this->coo_nnz_ = coo_nnz;

        this->nnz_ += coo_nnz;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixHYB<ValueType>::Clear()
    {
        free_hip(&this->mat_.ELL.val);
        free_hip(&this->mat_.ELL.col);
        free_hip(&this->mat_.COO.row);
        free_hip(&this->mat_.COO.col);
        free_hip(&this->mat_.COO.val);

        this->ell_nnz_         = 0;
        this->mat_.ELL.max_row = 0;
        this->coo_nnz_         = 0;

        this->nrow_ = 0;
        this->ncol_ = 0;
        this->nnz_  = 0;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixHYB<ValueType>::CopyFromHost(const HostMatrix<ValueType>& src)
    {
        const HostMatrixHYB<ValueType>* cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == src.GetMatFormat());

        // CPU to HIP copy
        if((cast_mat = dynamic_cast<const HostMatrixHYB<ValueType>*>(&src)) != NULL)
        {
            if(this->nnz_ == 0)
            {
                this->AllocateHYB(cast_mat->ell_nnz_,
                                  cast_mat->coo_nnz_,
                                  cast_mat->mat_.ELL.max_row,
                                  cast_mat->nrow_,
                                  cast_mat->ncol_);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->ell_nnz_ == cast_mat->ell_nnz_);
            assert(this->coo_nnz_ == cast_mat->coo_nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

            copy_h2d(this->ell_nnz_, cast_mat->mat_.ELL.col, this->mat_.ELL.col);
            copy_h2d(this->ell_nnz_, cast_mat->mat_.ELL.val, this->mat_.ELL.val);
            copy_h2d(this->coo_nnz_, cast_mat->mat_.COO.row, this->mat_.COO.row);
            copy_h2d(this->coo_nnz_, cast_mat->mat_.COO.col, this->mat_.COO.col);
            copy_h2d(this->coo_nnz_, cast_mat->mat_.COO.val, this->mat_.COO.val);
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
    void HIPAcceleratorMatrixHYB<ValueType>::CopyToHost(HostMatrix<ValueType>* dst) const
    {
        HostMatrixHYB<ValueType>* cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == dst->GetMatFormat());

        // HIP to CPU copy
        if((cast_mat = dynamic_cast<HostMatrixHYB<ValueType>*>(dst)) != NULL)
        {
            cast_mat->set_backend(this->local_backend_);

            if(cast_mat->nnz_ == 0)
            {
                cast_mat->AllocateHYB(this->ell_nnz_,
                                      this->coo_nnz_,
                                      this->mat_.ELL.max_row,
                                      this->nrow_,
                                      this->ncol_);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->ell_nnz_ == cast_mat->ell_nnz_);
            assert(this->coo_nnz_ == cast_mat->coo_nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

            copy_d2h(this->ell_nnz_, this->mat_.ELL.col, cast_mat->mat_.ELL.col);
            copy_d2h(this->ell_nnz_, this->mat_.ELL.val, cast_mat->mat_.ELL.val);
            copy_d2h(this->coo_nnz_, this->mat_.COO.row, cast_mat->mat_.COO.row);
            copy_d2h(this->coo_nnz_, this->mat_.COO.col, cast_mat->mat_.COO.col);
            copy_d2h(this->coo_nnz_, this->mat_.COO.val, cast_mat->mat_.COO.val);
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
    void HIPAcceleratorMatrixHYB<ValueType>::CopyFrom(const BaseMatrix<ValueType>& src)
    {
        const HIPAcceleratorMatrixHYB<ValueType>* hip_cast_mat;
        const HostMatrix<ValueType>*              host_cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == src.GetMatFormat());

        // HIP to HIP copy
        if((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixHYB<ValueType>*>(&src)) != NULL)
        {
            if(this->nnz_ == 0)
            {
                this->AllocateHYB(hip_cast_mat->ell_nnz_,
                                  hip_cast_mat->coo_nnz_,
                                  hip_cast_mat->mat_.ELL.max_row,
                                  hip_cast_mat->nrow_,
                                  hip_cast_mat->ncol_);
            }

            assert(this->nnz_ == hip_cast_mat->nnz_);
            assert(this->ell_nnz_ == hip_cast_mat->ell_nnz_);
            assert(this->coo_nnz_ == hip_cast_mat->coo_nnz_);
            assert(this->nrow_ == hip_cast_mat->nrow_);
            assert(this->ncol_ == hip_cast_mat->ncol_);

            copy_d2d(this->ell_nnz_, hip_cast_mat->mat_.ELL.col, this->mat_.ELL.col);
            copy_d2d(this->ell_nnz_, hip_cast_mat->mat_.ELL.val, this->mat_.ELL.val);
            copy_d2d(this->coo_nnz_, hip_cast_mat->mat_.COO.row, this->mat_.COO.row);
            copy_d2d(this->coo_nnz_, hip_cast_mat->mat_.COO.col, this->mat_.COO.col);
            copy_d2d(this->coo_nnz_, hip_cast_mat->mat_.COO.val, this->mat_.COO.val);
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
    void HIPAcceleratorMatrixHYB<ValueType>::CopyTo(BaseMatrix<ValueType>* dst) const
    {
        HIPAcceleratorMatrixHYB<ValueType>* hip_cast_mat;
        HostMatrix<ValueType>*              host_cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == dst->GetMatFormat());

        // HIP to HIP copy
        if((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixHYB<ValueType>*>(dst)))
        {
            hip_cast_mat->set_backend(this->local_backend_);

            if(hip_cast_mat->nnz_ == 0)
            {
                hip_cast_mat->AllocateHYB(this->ell_nnz_,
                                          this->coo_nnz_,
                                          this->mat_.ELL.max_row,
                                          this->nrow_,
                                          this->ncol_);
            }

            assert(this->nnz_ == hip_cast_mat->nnz_);
            assert(this->ell_nnz_ == hip_cast_mat->ell_nnz_);
            assert(this->coo_nnz_ == hip_cast_mat->coo_nnz_);
            assert(this->nrow_ == hip_cast_mat->nrow_);
            assert(this->ncol_ == hip_cast_mat->ncol_);

            copy_d2d(this->ell_nnz_, this->mat_.ELL.col, hip_cast_mat->mat_.ELL.col);
            copy_d2d(this->ell_nnz_, this->mat_.ELL.val, hip_cast_mat->mat_.ELL.val);
            copy_d2d(this->coo_nnz_, this->mat_.COO.row, hip_cast_mat->mat_.COO.row);
            copy_d2d(this->coo_nnz_, this->mat_.COO.col, hip_cast_mat->mat_.COO.col);
            copy_d2d(this->coo_nnz_, this->mat_.COO.val, hip_cast_mat->mat_.COO.val);
        }
        else
        {
            // HIP to CPU
            if((host_cast_mat = dynamic_cast<HostMatrix<ValueType>*>(dst)))
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
    void HIPAcceleratorMatrixHYB<ValueType>::CopyFromHostAsync(const HostMatrix<ValueType>& src)
    {
        const HostMatrixHYB<ValueType>* cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == src.GetMatFormat());

        // CPU to HIP copy
        if((cast_mat = dynamic_cast<const HostMatrixHYB<ValueType>*>(&src)) != NULL)
        {
            if(this->nnz_ == 0)
            {
                this->AllocateHYB(cast_mat->ell_nnz_,
                                  cast_mat->coo_nnz_,
                                  cast_mat->mat_.ELL.max_row,
                                  cast_mat->nrow_,
                                  cast_mat->ncol_);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->ell_nnz_ == cast_mat->ell_nnz_);
            assert(this->coo_nnz_ == cast_mat->coo_nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

            copy_h2d(this->ell_nnz_,
                     cast_mat->mat_.ELL.col,
                     this->mat_.ELL.col,
                     true,
                     HIPSTREAM(_get_backend_descriptor()->HIP_stream_current));
            copy_h2d(this->ell_nnz_,
                     cast_mat->mat_.ELL.val,
                     this->mat_.ELL.val,
                     true,
                     HIPSTREAM(_get_backend_descriptor()->HIP_stream_current));
            copy_h2d(this->coo_nnz_,
                     cast_mat->mat_.COO.row,
                     this->mat_.COO.row,
                     true,
                     HIPSTREAM(_get_backend_descriptor()->HIP_stream_current));
            copy_h2d(this->coo_nnz_,
                     cast_mat->mat_.COO.col,
                     this->mat_.COO.col,
                     true,
                     HIPSTREAM(_get_backend_descriptor()->HIP_stream_current));
            copy_h2d(this->coo_nnz_,
                     cast_mat->mat_.COO.val,
                     this->mat_.COO.val,
                     true,
                     HIPSTREAM(_get_backend_descriptor()->HIP_stream_current));
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
    void HIPAcceleratorMatrixHYB<ValueType>::CopyToHostAsync(HostMatrix<ValueType>* dst) const
    {
        HostMatrixHYB<ValueType>* cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == dst->GetMatFormat());

        // HIP to CPU copy
        if((cast_mat = dynamic_cast<HostMatrixHYB<ValueType>*>(dst)) != NULL)
        {
            cast_mat->set_backend(this->local_backend_);

            if(cast_mat->nnz_ == 0)
            {
                cast_mat->AllocateHYB(this->ell_nnz_,
                                      this->coo_nnz_,
                                      this->mat_.ELL.max_row,
                                      this->nrow_,
                                      this->ncol_);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->ell_nnz_ == cast_mat->ell_nnz_);
            assert(this->coo_nnz_ == cast_mat->coo_nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

            copy_d2h(this->ell_nnz_,
                     this->mat_.ELL.col,
                     cast_mat->mat_.ELL.col,
                     true,
                     HIPSTREAM(_get_backend_descriptor()->HIP_stream_current));
            copy_d2h(this->ell_nnz_,
                     this->mat_.ELL.val,
                     cast_mat->mat_.ELL.val,
                     true,
                     HIPSTREAM(_get_backend_descriptor()->HIP_stream_current));
            copy_d2h(this->coo_nnz_,
                     this->mat_.COO.row,
                     cast_mat->mat_.COO.row,
                     true,
                     HIPSTREAM(_get_backend_descriptor()->HIP_stream_current));
            copy_d2h(this->coo_nnz_,
                     this->mat_.COO.col,
                     cast_mat->mat_.COO.col,
                     true,
                     HIPSTREAM(_get_backend_descriptor()->HIP_stream_current));
            copy_d2h(this->coo_nnz_,
                     this->mat_.COO.val,
                     cast_mat->mat_.COO.val,
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
    void HIPAcceleratorMatrixHYB<ValueType>::CopyFromAsync(const BaseMatrix<ValueType>& src)
    {
        const HIPAcceleratorMatrixHYB<ValueType>* hip_cast_mat;
        const HostMatrix<ValueType>*              host_cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == src.GetMatFormat());

        // HIP to HIP copy
        if((hip_cast_mat = dynamic_cast<const HIPAcceleratorMatrixHYB<ValueType>*>(&src)) != NULL)
        {
            if(this->nnz_ == 0)
            {
                this->AllocateHYB(hip_cast_mat->ell_nnz_,
                                  hip_cast_mat->coo_nnz_,
                                  hip_cast_mat->mat_.ELL.max_row,
                                  hip_cast_mat->nrow_,
                                  hip_cast_mat->ncol_);
            }

            assert(this->nnz_ == hip_cast_mat->nnz_);
            assert(this->ell_nnz_ == hip_cast_mat->ell_nnz_);
            assert(this->coo_nnz_ == hip_cast_mat->coo_nnz_);
            assert(this->nrow_ == hip_cast_mat->nrow_);
            assert(this->ncol_ == hip_cast_mat->ncol_);

            copy_d2d(this->ell_nnz_,
                     hip_cast_mat->mat_.ELL.col,
                     this->mat_.ELL.col,
                     true,
                     HIPSTREAM(_get_backend_descriptor()->HIP_stream_current));
            copy_d2d(this->ell_nnz_,
                     hip_cast_mat->mat_.ELL.val,
                     this->mat_.ELL.val,
                     true,
                     HIPSTREAM(_get_backend_descriptor()->HIP_stream_current));
            copy_d2d(this->coo_nnz_,
                     hip_cast_mat->mat_.COO.row,
                     this->mat_.COO.row,
                     true,
                     HIPSTREAM(_get_backend_descriptor()->HIP_stream_current));
            copy_d2d(this->coo_nnz_,
                     hip_cast_mat->mat_.COO.col,
                     this->mat_.COO.col,
                     true,
                     HIPSTREAM(_get_backend_descriptor()->HIP_stream_current));
            copy_d2d(this->coo_nnz_,
                     hip_cast_mat->mat_.COO.val,
                     this->mat_.COO.val,
                     true,
                     HIPSTREAM(_get_backend_descriptor()->HIP_stream_current));
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
    void HIPAcceleratorMatrixHYB<ValueType>::CopyToAsync(BaseMatrix<ValueType>* dst) const
    {
        HIPAcceleratorMatrixHYB<ValueType>* hip_cast_mat;
        HostMatrix<ValueType>*              host_cast_mat;

        // copy only in the same format
        assert(this->GetMatFormat() == dst->GetMatFormat());

        // HIP to HIP copy
        if((hip_cast_mat = dynamic_cast<HIPAcceleratorMatrixHYB<ValueType>*>(dst)))
        {
            hip_cast_mat->set_backend(this->local_backend_);

            if(hip_cast_mat->nnz_ == 0)
            {
                hip_cast_mat->AllocateHYB(this->ell_nnz_,
                                          this->coo_nnz_,
                                          this->mat_.ELL.max_row,
                                          this->nrow_,
                                          this->ncol_);
            }

            assert(this->nnz_ == hip_cast_mat->nnz_);
            assert(this->ell_nnz_ == hip_cast_mat->ell_nnz_);
            assert(this->coo_nnz_ == hip_cast_mat->coo_nnz_);
            assert(this->nrow_ == hip_cast_mat->nrow_);
            assert(this->ncol_ == hip_cast_mat->ncol_);

            copy_d2d(this->ell_nnz_,
                     this->mat_.ELL.col,
                     hip_cast_mat->mat_.ELL.col,
                     true,
                     HIPSTREAM(_get_backend_descriptor()->HIP_stream_current));
            copy_d2d(this->ell_nnz_,
                     this->mat_.ELL.val,
                     hip_cast_mat->mat_.ELL.val,
                     true,
                     HIPSTREAM(_get_backend_descriptor()->HIP_stream_current));
            copy_d2d(this->coo_nnz_,
                     this->mat_.COO.row,
                     hip_cast_mat->mat_.COO.row,
                     true,
                     HIPSTREAM(_get_backend_descriptor()->HIP_stream_current));
            copy_d2d(this->coo_nnz_,
                     this->mat_.COO.col,
                     hip_cast_mat->mat_.COO.col,
                     true,
                     HIPSTREAM(_get_backend_descriptor()->HIP_stream_current));
            copy_d2d(this->coo_nnz_,
                     this->mat_.COO.val,
                     hip_cast_mat->mat_.COO.val,
                     true,
                     HIPSTREAM(_get_backend_descriptor()->HIP_stream_current));
        }
        else
        {
            // HIP to CPU
            if((host_cast_mat = dynamic_cast<HostMatrix<ValueType>*>(dst)))
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
    bool HIPAcceleratorMatrixHYB<ValueType>::ConvertFrom(const BaseMatrix<ValueType>& mat)
    {
        this->Clear();

        // Empty matrix
        if(mat.GetNnz() == 0)
        {
            this->AllocateHYB(0, 0, 0, mat.GetM(), mat.GetN());

            return true;
        }

        const HIPAcceleratorMatrixHYB<ValueType>* cast_mat_hyb;

        if((cast_mat_hyb = dynamic_cast<const HIPAcceleratorMatrixHYB<ValueType>*>(&mat)) != NULL)
        {
            this->CopyFrom(*cast_mat_hyb);
            return true;
        }

        const HIPAcceleratorMatrixCSR<ValueType>* cast_mat_csr;

        if((cast_mat_csr = dynamic_cast<const HIPAcceleratorMatrixCSR<ValueType>*>(&mat)) != NULL)
        {
            this->Clear();

            int64_t nnz_hyb;
            int64_t nnz_ell;
            int64_t nnz_coo;

            if(csr_to_hyb_hip(&this->local_backend_,
                              cast_mat_csr->nnz_,
                              cast_mat_csr->nrow_,
                              cast_mat_csr->ncol_,
                              cast_mat_csr->mat_,
                              &this->mat_,
                              &nnz_hyb,
                              &nnz_ell,
                              &nnz_coo)
               == true)
            {
                this->nrow_    = cast_mat_csr->nrow_;
                this->ncol_    = cast_mat_csr->ncol_;
                this->nnz_     = nnz_hyb;
                this->ell_nnz_ = nnz_ell;
                this->coo_nnz_ = nnz_coo;

                return true;
            }
        }

        return false;
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixHYB<ValueType>::Apply(const BaseVector<ValueType>& in,
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

            // ELL
            if(this->ell_nnz_ > 0)
            {
                ValueType beta = 0.0;

                rocsparse_status status;
                status = rocsparseTellmv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                         rocsparse_operation_none,
                                         this->nrow_,
                                         this->ncol_,
                                         &alpha,
                                         this->ell_mat_descr_,
                                         this->mat_.ELL.val,
                                         this->mat_.ELL.col,
                                         this->mat_.ELL.max_row,
                                         cast_in->vec_,
                                         &beta,
                                         cast_out->vec_);
                CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
            }

            // COO
            if(this->coo_nnz_ > 0)
            {
                // Add to y from ELL part
                ValueType beta = 1.0;

                rocsparse_status status;
                status = rocsparseTcoomv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                         rocsparse_operation_none,
                                         this->nrow_,
                                         this->ncol_,
                                         this->coo_nnz_,
                                         &alpha,
                                         this->coo_mat_descr_,
                                         this->mat_.COO.val,
                                         this->mat_.COO.row,
                                         this->mat_.COO.col,
                                         cast_in->vec_,
                                         &beta,
                                         cast_out->vec_);
                CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
            }
        }
    }

    template <typename ValueType>
    void HIPAcceleratorMatrixHYB<ValueType>::ApplyAdd(const BaseVector<ValueType>& in,
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

            // ELL
            if(this->ell_nnz_ > 0)
            {
                ValueType beta = 0.0;

                rocsparse_status status;
                status = rocsparseTellmv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                         rocsparse_operation_none,
                                         this->nrow_,
                                         this->ncol_,
                                         &scalar,
                                         this->ell_mat_descr_,
                                         this->mat_.ELL.val,
                                         this->mat_.ELL.col,
                                         this->mat_.ELL.max_row,
                                         cast_in->vec_,
                                         &beta,
                                         cast_out->vec_);
                CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
            }

            // COO
            if(this->coo_nnz_ > 0)
            {
                // Add to y from ELL part
                ValueType beta = 1.0;

                rocsparse_status status;
                status = rocsparseTcoomv(ROCSPARSE_HANDLE(this->local_backend_.ROC_sparse_handle),
                                         rocsparse_operation_none,
                                         this->nrow_,
                                         this->ncol_,
                                         this->coo_nnz_,
                                         &scalar,
                                         this->coo_mat_descr_,
                                         this->mat_.COO.val,
                                         this->mat_.COO.row,
                                         this->mat_.COO.col,
                                         cast_in->vec_,
                                         &beta,
                                         cast_out->vec_);
                CHECK_ROCSPARSE_ERROR(status, __FILE__, __LINE__);
            }
        }
    }

    template class HIPAcceleratorMatrixHYB<double>;
    template class HIPAcceleratorMatrixHYB<float>;
#ifdef SUPPORT_COMPLEX
    template class HIPAcceleratorMatrixHYB<std::complex<double>>;
    template class HIPAcceleratorMatrixHYB<std::complex<float>>;
#endif

} // namespace rocalution

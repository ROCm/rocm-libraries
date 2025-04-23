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

#include "local_matrix.hpp"
#include "../utils/allocate_free.hpp"
#include "../utils/def.hpp"
#include "../utils/log.hpp"
#include "../utils/math_functions.hpp"
#include "../utils/rocsparseio.h"
#include "backend_manager.hpp"
#include "base_matrix.hpp"
#include "base_vector.hpp"
#include "host/host_matrix_coo.hpp"
#include "host/host_matrix_csr.hpp"
#include "host/host_vector.hpp"
#include "local_vector.hpp"

#include <algorithm>
#include <complex>
#include <limits>
#include <sstream>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace rocalution
{

    template <typename ValueType>
    LocalMatrix<ValueType>::LocalMatrix()
    {
        log_debug(this, "LocalMatrix::LocalMatrix()");

        this->object_name_ = "";

        // Create empty matrix on the host
        // CSR is the default format
        this->matrix_host_ = new HostMatrixCSR<ValueType>(this->local_backend_);

        this->matrix_accel_ = NULL;
        this->matrix_       = this->matrix_host_;
    }

    template <typename ValueType>
    LocalMatrix<ValueType>::~LocalMatrix()
    {
        log_debug(this, "LocalMatrix::~LocalMatrix()");

        this->Clear();
        delete this->matrix_;
    }

    template <typename ValueType>
    int64_t LocalMatrix<ValueType>::GetM(void) const
    {
        return static_cast<int64_t>(this->matrix_->GetM());
    }

    template <typename ValueType>
    int64_t LocalMatrix<ValueType>::GetN(void) const
    {
        return static_cast<int64_t>(this->matrix_->GetN());
    }

    template <typename ValueType>
    int64_t LocalMatrix<ValueType>::GetNnz(void) const
    {
        return this->matrix_->GetNnz();
    }

    template <typename ValueType>
    unsigned int LocalMatrix<ValueType>::GetFormat(void) const
    {
        return this->matrix_->GetMatFormat();
    }

    template <typename ValueType>
    int LocalMatrix<ValueType>::GetBlockDimension(void) const
    {
        return this->matrix_->GetMatBlockDimension();
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::Clear(void)
    {
        log_debug(this, "LocalMatrix::Clear()", "");

        this->matrix_->Clear();
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::Zeros(void)
    {
        log_debug(this, "LocalMatrix::Zeros()", "");

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->Zeros();

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::Zeros() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                // Move to host
                bool is_accel = this->is_accel_();
                this->MoveToHost();

                // Convert to CSR
                unsigned int format   = this->GetFormat();
                int          blockdim = this->GetBlockDimension();
                this->ConvertToCSR();

                if(this->matrix_->Zeros() == false)
                {
                    LOG_INFO("Computation of LocalMatrix::Zeros() failed");
                    this->Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(format != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::Zeros() is performed in CSR format");

                    this->ConvertTo(format, blockdim);
                }

                if(is_accel == true)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::Zeros() is performed on the host");

                    this->MoveToAccelerator();
                }
            }
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::AllocateCSR(const std::string& name,
                                             int64_t            nnz,
                                             int64_t            nrow,
                                             int64_t            ncol)
    {
        log_debug(this, "LocalMatrix::AllocateCSR()", name, nnz, nrow, ncol);

        assert(nnz >= 0);
        assert(nrow >= 0);
        assert(ncol >= 0);

        this->Clear();
        this->object_name_ = name;
        this->ConvertToCSR();

        if(nnz > 0)
        {
            assert(nrow > 0);
            assert(ncol > 0);
        }

        Rocalution_Backend_Descriptor backend       = this->local_backend_;
        unsigned int                  matrix_format = this->GetFormat();

        // init host matrix
        if(this->matrix_ == this->matrix_host_)
        {
            delete this->matrix_host_;
            this->matrix_host_
                = _rocalution_init_base_host_matrix<ValueType>(backend, matrix_format);
            this->matrix_ = this->matrix_host_;
        }
        else
        {
            // init accel matrix
            assert(this->matrix_ == this->matrix_accel_);

            delete this->matrix_accel_;
            this->matrix_accel_
                = _rocalution_init_base_backend_matrix<ValueType>(backend, matrix_format);
            this->matrix_ = this->matrix_accel_;
        }

        // Internally, we use 32 bits for now
        assert(nrow <= std::numeric_limits<int>::max());
        assert(ncol <= std::numeric_limits<int>::max());

        this->matrix_->AllocateCSR(nnz, static_cast<int>(nrow), static_cast<int>(ncol));

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::AllocateBCSR(
        const std::string& name, int64_t nnzb, int64_t nrowb, int64_t ncolb, int blockdim)
    {
        log_debug(this, "LocalMatrix::AllocateBCSR()", name, nnzb, nrowb, ncolb, blockdim);

        assert(nnzb >= 0);
        assert(nrowb >= 0);
        assert(ncolb >= 0);
        assert(blockdim > 1);

        this->Clear();
        this->object_name_ = name;
        this->ConvertToBCSR(blockdim);

        if(nnzb > 0)
        {
            assert(nrowb > 0);
            assert(ncolb > 0);

            Rocalution_Backend_Descriptor backend       = this->local_backend_;
            unsigned int                  matrix_format = this->GetFormat();
            int                           blockdim      = this->GetBlockDimension();

            // init host matrix
            if(this->matrix_ == this->matrix_host_)
            {
                delete this->matrix_host_;
                this->matrix_host_ = _rocalution_init_base_host_matrix<ValueType>(
                    backend, matrix_format, blockdim);
                this->matrix_ = this->matrix_host_;
            }
            else
            {
                // init accel matrix
                assert(this->matrix_ == this->matrix_accel_);

                delete this->matrix_accel_;
                this->matrix_accel_ = _rocalution_init_base_backend_matrix<ValueType>(
                    backend, matrix_format, blockdim);
                this->matrix_ = this->matrix_accel_;
            }

            assert(nrowb <= std::numeric_limits<int>::max());
            assert(ncolb <= std::numeric_limits<int>::max());

            this->matrix_->AllocateBCSR(
                nnzb, static_cast<int>(nrowb), static_cast<int>(ncolb), blockdim);
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::AllocateCOO(const std::string& name,
                                             int64_t            nnz,
                                             int64_t            nrow,
                                             int64_t            ncol)
    {
        log_debug(this, "LocalMatrix::AllocateCOO()", name, nnz, nrow, ncol);

        assert(nnz >= 0);
        assert(nrow >= 0);
        assert(ncol >= 0);

        this->Clear();
        this->object_name_ = name;
        this->ConvertToCOO();

        if(nnz > 0)
        {
            assert(nrow > 0);
            assert(ncol > 0);

            Rocalution_Backend_Descriptor backend       = this->local_backend_;
            unsigned int                  matrix_format = this->GetFormat();

            // init host matrix
            if(this->matrix_ == this->matrix_host_)
            {
                delete this->matrix_host_;
                this->matrix_host_
                    = _rocalution_init_base_host_matrix<ValueType>(backend, matrix_format);
                this->matrix_ = this->matrix_host_;
            }
            else
            {
                // init accel matrix
                assert(this->matrix_ == this->matrix_accel_);

                delete this->matrix_accel_;
                this->matrix_accel_
                    = _rocalution_init_base_backend_matrix<ValueType>(backend, matrix_format);
                this->matrix_ = this->matrix_accel_;
            }

            assert(nrow <= std::numeric_limits<int>::max());
            assert(ncol <= std::numeric_limits<int>::max());

            this->matrix_->AllocateCOO(nnz, static_cast<int>(nrow), static_cast<int>(ncol));
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::AllocateDIA(
        const std::string& name, int64_t nnz, int64_t nrow, int64_t ncol, int ndiag)
    {
        log_debug(this, "LocalMatrix::AllocateDIA()", name, nnz, nrow, ncol, ndiag);

        assert(nnz >= 0);
        assert(nrow >= 0);
        assert(ncol >= 0);

        this->Clear();
        this->object_name_ = name;
        this->ConvertToDIA();

        if(nnz > 0)
        {
            assert(nrow > 0);
            assert(ncol > 0);

            Rocalution_Backend_Descriptor backend       = this->local_backend_;
            unsigned int                  matrix_format = this->GetFormat();

            // init host matrix
            if(this->matrix_ == this->matrix_host_)
            {
                delete this->matrix_host_;
                this->matrix_host_
                    = _rocalution_init_base_host_matrix<ValueType>(backend, matrix_format);
                this->matrix_ = this->matrix_host_;
            }
            else
            {
                // init accel matrix
                assert(this->matrix_ == this->matrix_accel_);

                delete this->matrix_accel_;
                this->matrix_accel_
                    = _rocalution_init_base_backend_matrix<ValueType>(backend, matrix_format);
                this->matrix_ = this->matrix_accel_;
            }

            assert(nrow <= std::numeric_limits<int>::max());
            assert(ncol <= std::numeric_limits<int>::max());

            this->matrix_->AllocateDIA(nnz, static_cast<int>(nrow), static_cast<int>(ncol), ndiag);
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::AllocateMCSR(const std::string& name,
                                              int64_t            nnz,
                                              int64_t            nrow,
                                              int64_t            ncol)
    {
        log_debug(this, "LocalMatrix::AllocateMCSR()", name, nnz, nrow, ncol);

        assert(nnz >= 0);
        assert(nrow >= 0);
        assert(ncol >= 0);

        this->Clear();
        this->object_name_ = name;
        this->ConvertToMCSR();

        if(nnz > 0)
        {
            assert(nrow > 0);
            assert(ncol > 0);

            Rocalution_Backend_Descriptor backend       = this->local_backend_;
            unsigned int                  matrix_format = this->GetFormat();

            // init host matrix
            if(this->matrix_ == this->matrix_host_)
            {
                delete this->matrix_host_;
                this->matrix_host_
                    = _rocalution_init_base_host_matrix<ValueType>(backend, matrix_format);
                this->matrix_ = this->matrix_host_;
            }
            else
            {
                // init accel matrix
                assert(this->matrix_ == this->matrix_accel_);

                delete this->matrix_accel_;
                this->matrix_accel_
                    = _rocalution_init_base_backend_matrix<ValueType>(backend, matrix_format);
                this->matrix_ = this->matrix_accel_;
            }

            assert(nrow <= std::numeric_limits<int>::max());
            assert(ncol <= std::numeric_limits<int>::max());

            this->matrix_->AllocateMCSR(nnz, static_cast<int>(nrow), static_cast<int>(ncol));
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::AllocateELL(
        const std::string& name, int64_t nnz, int64_t nrow, int64_t ncol, int max_row)
    {
        log_debug(this, "LocalMatrix::AllocateELL()", name, nnz, nrow, ncol, max_row);

        assert(nnz >= 0);
        assert(nrow >= 0);
        assert(ncol >= 0);

        this->Clear();
        this->object_name_ = name;
        this->ConvertToELL();

        if(nnz > 0)
        {
            assert(nrow > 0);
            assert(ncol > 0);

            Rocalution_Backend_Descriptor backend       = this->local_backend_;
            unsigned int                  matrix_format = this->GetFormat();

            // init host matrix
            if(this->matrix_ == this->matrix_host_)
            {
                delete this->matrix_host_;
                this->matrix_host_
                    = _rocalution_init_base_host_matrix<ValueType>(backend, matrix_format);
                this->matrix_ = this->matrix_host_;
            }
            else
            {
                // init accel matrix
                assert(this->matrix_ == this->matrix_accel_);

                delete this->matrix_accel_;
                this->matrix_accel_
                    = _rocalution_init_base_backend_matrix<ValueType>(backend, matrix_format);
                this->matrix_ = this->matrix_accel_;
            }

            assert(nrow <= std::numeric_limits<int>::max());
            assert(ncol <= std::numeric_limits<int>::max());

            this->matrix_->AllocateELL(
                nnz, static_cast<int>(nrow), static_cast<int>(ncol), max_row);
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::AllocateHYB(const std::string& name,
                                             int64_t            ell_nnz,
                                             int64_t            coo_nnz,
                                             int                ell_max_row,
                                             int64_t            nrow,
                                             int64_t            ncol)
    {
        log_debug(
            this, "LocalMatrix::AllocateHYB()", name, ell_nnz, coo_nnz, ell_max_row, nrow, ncol);

        assert(ell_nnz >= 0);
        assert(coo_nnz >= 0);
        assert(nrow >= 0);
        assert(ncol >= 0);

        this->Clear();
        this->object_name_ = name;
        this->ConvertToHYB();

        if(ell_nnz + coo_nnz > 0)
        {
            assert(nrow > 0);
            assert(ncol > 0);

            Rocalution_Backend_Descriptor backend       = this->local_backend_;
            unsigned int                  matrix_format = this->GetFormat();

            // init host matrix
            if(this->matrix_ == this->matrix_host_)
            {
                delete this->matrix_host_;
                this->matrix_host_
                    = _rocalution_init_base_host_matrix<ValueType>(backend, matrix_format);
                this->matrix_ = this->matrix_host_;
            }
            else
            {
                // init accel matrix
                assert(this->matrix_ == this->matrix_accel_);

                delete this->matrix_accel_;
                this->matrix_accel_
                    = _rocalution_init_base_backend_matrix<ValueType>(backend, matrix_format);
                this->matrix_ = this->matrix_accel_;
            }

            assert(nrow <= std::numeric_limits<int>::max());
            assert(ncol <= std::numeric_limits<int>::max());

            this->matrix_->AllocateHYB(
                ell_nnz, coo_nnz, ell_max_row, static_cast<int>(nrow), static_cast<int>(ncol));
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::AllocateDENSE(const std::string& name, int64_t nrow, int64_t ncol)
    {
        log_debug(this, "LocalMatrix::AllocateDENSE()", name, nrow, ncol);

        assert(nrow >= 0);
        assert(ncol >= 0);

        this->Clear();
        this->object_name_ = name;
        this->ConvertToDENSE();

        if(nrow * ncol > 0)
        {
            Rocalution_Backend_Descriptor backend       = this->local_backend_;
            unsigned int                  matrix_format = this->GetFormat();

            // init host matrix
            if(this->matrix_ == this->matrix_host_)
            {
                delete this->matrix_host_;
                this->matrix_host_
                    = _rocalution_init_base_host_matrix<ValueType>(backend, matrix_format);
                this->matrix_ = this->matrix_host_;
            }
            else
            {
                // init accel matrix
                assert(this->matrix_ == this->matrix_accel_);

                delete this->matrix_accel_;
                this->matrix_accel_
                    = _rocalution_init_base_backend_matrix<ValueType>(backend, matrix_format);
                this->matrix_ = this->matrix_accel_;
            }

            assert(nrow <= std::numeric_limits<int>::max());
            assert(ncol <= std::numeric_limits<int>::max());

            this->matrix_->AllocateDENSE(static_cast<int>(nrow), static_cast<int>(ncol));
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    bool LocalMatrix<ValueType>::Check(void) const
    {
        log_debug(this, "LocalMatrix::Check()", "");

        bool check = false;

        if(this->is_accel_() == true)
        {
            LocalMatrix<ValueType> mat_host;
            mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
            mat_host.CopyFrom(*this);

            // Convert to CSR
            mat_host.ConvertToCSR();

            check = mat_host.matrix_->Check();

            if(this->GetFormat() != CSR)
            {
                LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::Check() is performed in CSR format");
            }

            LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::Check() is performed on the host");
        }
        else
        {
            if(this->GetFormat() != CSR)
            {
                LocalMatrix<ValueType> mat_csr;
                mat_csr.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_csr.CopyFrom(*this);

                // Convert to CSR
                mat_csr.ConvertToCSR();

                check = mat_csr.matrix_->Check();

                LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::Check() is performed in CSR format");
            }
            else
            {
                check = this->matrix_->Check();
            }
        }

        return check;
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::SetDataPtrCOO(int**       row,
                                               int**       col,
                                               ValueType** val,
                                               std::string name,
                                               int64_t     nnz,
                                               int64_t     nrow,
                                               int64_t     ncol)
    {
        log_debug(this, "LocalMatrix::SetDataPtrCOO()", row, col, val, name, nnz, nrow, ncol);

        assert(row != NULL);
        assert(col != NULL);
        assert(val != NULL);
        assert(*row != NULL);
        assert(*col != NULL);
        assert(*val != NULL);
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);

        this->Clear();

        this->object_name_ = name;

        //  this->MoveToHost();
        this->ConvertToCOO();

        assert(nrow <= std::numeric_limits<int>::max());
        assert(ncol <= std::numeric_limits<int>::max());

        this->matrix_->SetDataPtrCOO(
            row, col, val, nnz, static_cast<int>(nrow), static_cast<int>(ncol));

        *row = NULL;
        *col = NULL;
        *val = NULL;

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::LeaveDataPtrCOO(int** row, int** col, ValueType** val)
    {
        log_debug(this, "LocalMatrix::LeaveDataPtrCOO()", row, col, val);

        assert(*row == NULL);
        assert(*col == NULL);
        assert(*val == NULL);
        assert(this->GetM() > 0);
        assert(this->GetN() > 0);
        assert(this->GetNnz() > 0);

#ifdef DEBUG_MODE
        this->Check();
#endif

        //  this->MoveToHost();
        this->ConvertToCOO();

        this->matrix_->LeaveDataPtrCOO(row, col, val);
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::SetDataPtrCSR(PtrType**   row_offset,
                                               int**       col,
                                               ValueType** val,
                                               std::string name,
                                               int64_t     nnz,
                                               int64_t     nrow,
                                               int64_t     ncol)
    {
        log_debug(
            this, "LocalMatrix::SetDataPtrCSR()", row_offset, col, val, name, nnz, nrow, ncol);

        assert(nnz >= 0);
        assert(nrow >= 0);
        assert(ncol >= 0);
        assert(row_offset != NULL);
        assert(*row_offset != NULL);

        if(nnz > 0)
        {
            assert(col != NULL);
            assert(val != NULL);
            assert(*col != NULL);
            assert(*val != NULL);
        }

        this->Clear();

        this->object_name_ = name;

        //  this->MoveToHost();
        this->ConvertToCSR();

        assert(nrow <= std::numeric_limits<int>::max());
        assert(ncol <= std::numeric_limits<int>::max());

        this->matrix_->SetDataPtrCSR(
            row_offset, col, val, nnz, static_cast<int>(nrow), static_cast<int>(ncol));

        *row_offset = NULL;
        *col        = NULL;
        *val        = NULL;

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::LeaveDataPtrCSR(PtrType** row_offset, int** col, ValueType** val)
    {
        log_debug(this, "LocalMatrix::LeaveDataPtrCSR()", row_offset, col, val);

        assert(*row_offset == NULL);
        assert(*col == NULL);
        assert(*val == NULL);
        assert(this->GetM() >= 0);
        assert(this->GetN() >= 0);
        assert(this->GetNnz() >= 0);

#ifdef DEBUG_MODE
        this->Check();
#endif

        //  this->MoveToHost();
        this->ConvertToCSR();

        this->matrix_->LeaveDataPtrCSR(row_offset, col, val);
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::SetDataPtrBCSR(int**       row_offset,
                                                int**       col,
                                                ValueType** val,
                                                std::string name,
                                                int64_t     nnzb,
                                                int64_t     nrowb,
                                                int64_t     ncolb,
                                                int         blockdim)
    {
        log_debug(this,
                  "LocalMatrix::SetDataPtrBCSR()",
                  row_offset,
                  col,
                  val,
                  name,
                  nnzb,
                  nrowb,
                  ncolb,
                  blockdim);

        assert(row_offset != NULL);
        assert(col != NULL);
        assert(val != NULL);
        assert(*row_offset != NULL);
        assert(*col != NULL);
        assert(*val != NULL);
        assert(nnzb > 0);
        assert(nrowb > 0);
        assert(ncolb > 0);
        assert(blockdim > 1);

        this->Clear();

        this->object_name_ = name;

        //  this->MoveToHost();
        this->ConvertToBCSR(blockdim);

        assert(nrowb <= std::numeric_limits<int>::max());
        assert(ncolb <= std::numeric_limits<int>::max());

        this->matrix_->SetDataPtrBCSR(
            row_offset, col, val, nnzb, static_cast<int>(nrowb), static_cast<int>(ncolb), blockdim);

        *row_offset = NULL;
        *col        = NULL;
        *val        = NULL;

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::LeaveDataPtrBCSR(int**       row_offset,
                                                  int**       col,
                                                  ValueType** val,
                                                  int&        blockdim)
    {
        log_debug(this, "LocalMatrix::LeaveDataPtrBCSR()", row_offset, col, val, blockdim);

        assert(*row_offset == NULL);
        assert(*col == NULL);
        assert(*val == NULL);
        assert(this->GetM() > 0);
        assert(this->GetN() > 0);
        assert(this->GetNnz() > 0);

#ifdef DEBUG_MODE
        this->Check();
#endif
        // If matrix is not yet a BCSR matrix, blockdim need to be valid
        if(this->GetFormat() != BCSR)
        {
            assert(blockdim > 1);
        }

        //  this->MoveToHost();
        this->ConvertToBCSR(blockdim);

        this->matrix_->LeaveDataPtrBCSR(row_offset, col, val, blockdim);
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::SetDataPtrMCSR(int**       row_offset,
                                                int**       col,
                                                ValueType** val,
                                                std::string name,
                                                int64_t     nnz,
                                                int64_t     nrow,
                                                int64_t     ncol)
    {
        log_debug(
            this, "LocalMatrix::SetDataPtrMCSR()", row_offset, col, val, name, nnz, nrow, ncol);

        assert(row_offset != NULL);
        assert(col != NULL);
        assert(val != NULL);
        assert(*row_offset != NULL);
        assert(*col != NULL);
        assert(*val != NULL);
        assert(nnz > 0);
        assert(nrow > 0);
        assert(ncol > 0);

        this->Clear();

        this->object_name_ = name;

        //  this->MoveToHost();
        this->ConvertToMCSR();

        assert(nrow <= std::numeric_limits<int>::max());
        assert(ncol <= std::numeric_limits<int>::max());

        this->matrix_->SetDataPtrMCSR(
            row_offset, col, val, nnz, static_cast<int>(nrow), static_cast<int>(ncol));

        *row_offset = NULL;
        *col        = NULL;
        *val        = NULL;

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::LeaveDataPtrMCSR(int** row_offset, int** col, ValueType** val)
    {
        log_debug(this, "LocalMatrix::LeaveDataPtrMCSR()", row_offset, col, val);

        assert(*row_offset == NULL);
        assert(*col == NULL);
        assert(*val == NULL);
        assert(this->GetM() > 0);
        assert(this->GetN() > 0);
        assert(this->GetNnz() > 0);

#ifdef DEBUG_MODE
        this->Check();
#endif

        //  this->MoveToHost();
        this->ConvertToMCSR();

        this->matrix_->LeaveDataPtrMCSR(row_offset, col, val);
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::SetDataPtrELL(int**       col,
                                               ValueType** val,
                                               std::string name,
                                               int64_t     nnz,
                                               int64_t     nrow,
                                               int64_t     ncol,
                                               int         max_row)
    {
        log_debug(this, "LocalMatrix::SetDataPtrELL()", col, val, name, nnz, nrow, ncol, max_row);

        assert(col != NULL);
        assert(val != NULL);
        assert(*col != NULL);
        assert(*val != NULL);
        assert(nnz > 0);
        assert(nrow > 0);
        assert(max_row > 0);
        assert(max_row * nrow == nnz);

        this->Clear();

        this->object_name_ = name;

        //  this->MoveToHost();
        this->ConvertToELL();

        assert(nrow <= std::numeric_limits<int>::max());
        assert(ncol <= std::numeric_limits<int>::max());

        this->matrix_->SetDataPtrELL(
            col, val, nnz, static_cast<int>(nrow), static_cast<int>(ncol), max_row);

        *col = NULL;
        *val = NULL;

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::LeaveDataPtrELL(int** col, ValueType** val, int& max_row)
    {
        log_debug(this, "LocalMatrix::LeaveDataPtrELL()", col, val, max_row);

        assert(*col == NULL);
        assert(*val == NULL);
        assert(this->GetM() > 0);
        assert(this->GetN() > 0);
        assert(this->GetNnz() > 0);

#ifdef DEBUG_MODE
        this->Check();
#endif

        //  this->MoveToHost();
        this->ConvertToELL();

        this->matrix_->LeaveDataPtrELL(col, val, max_row);
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::SetDataPtrDIA(int**       offset,
                                               ValueType** val,
                                               std::string name,
                                               int64_t     nnz,
                                               int64_t     nrow,
                                               int64_t     ncol,
                                               int         num_diag)
    {
        log_debug(
            this, "LocalMatrix::SetDataPtrDIA()", offset, val, name, nnz, nrow, ncol, num_diag);

        assert(offset != NULL);
        assert(val != NULL);
        assert(*offset != NULL);
        assert(*val != NULL);
        assert(nnz > 0);
        assert(nrow > 0);
        assert(num_diag > 0);

        if(nrow < ncol)
        {
            assert(nnz == ncol * num_diag);
        }
        else
        {
            assert(nnz == nrow * num_diag);
        }

        this->Clear();

        this->object_name_ = name;

        //  this->MoveToHost();
        this->ConvertToDIA();

        assert(nrow <= std::numeric_limits<int>::max());
        assert(ncol <= std::numeric_limits<int>::max());

        this->matrix_->SetDataPtrDIA(
            offset, val, nnz, static_cast<int>(nrow), static_cast<int>(ncol), num_diag);

        *offset = NULL;
        *val    = NULL;

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::LeaveDataPtrDIA(int** offset, ValueType** val, int& num_diag)
    {
        log_debug(this, "LocalMatrix::LeaveDataPtrDIA()", offset, val, num_diag);

        assert(*offset == NULL);
        assert(*val == NULL);
        assert(this->GetM() > 0);
        assert(this->GetN() > 0);
        assert(this->GetNnz() > 0);

#ifdef DEBUG_MODE
        this->Check();
#endif

        //  this->MoveToHost();
        this->ConvertToDIA();

        this->matrix_->LeaveDataPtrDIA(offset, val, num_diag);
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::SetDataPtrDENSE(ValueType** val,
                                                 std::string name,
                                                 int64_t     nrow,
                                                 int64_t     ncol)
    {
        log_debug(this, "LocalMatrix::SetDataPtrDENSE()", val, name, nrow, ncol);

        assert(val != NULL);
        assert(*val != NULL);
        assert(nrow > 0);
        assert(ncol > 0);

        this->Clear();

        this->object_name_ = name;

        //  this->MoveToHost();
        this->ConvertToDENSE();

        assert(nrow <= std::numeric_limits<int>::max());
        assert(ncol <= std::numeric_limits<int>::max());

        this->matrix_->SetDataPtrDENSE(val, static_cast<int>(nrow), static_cast<int>(ncol));

        *val = NULL;

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::LeaveDataPtrDENSE(ValueType** val)
    {
        log_debug(this, "LocalMatrix::LeaveDataPtrDENSE()", val);

        assert(*val == NULL);
        assert(this->GetM() > 0);
        assert(this->GetN() > 0);
        assert(this->GetNnz() > 0);

#ifdef DEBUG_MODE
        this->Check();
#endif

        //  this->MoveToHost();
        this->ConvertToDENSE();

        this->matrix_->LeaveDataPtrDENSE(val);
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::CopyFromCSR(const PtrType*   row_offsets,
                                             const int*       col,
                                             const ValueType* val)
    {
        log_debug(this, "LocalMatrix::CopyFromCSR()", row_offsets, col, val);

        assert(row_offsets != NULL);
        assert(this->GetFormat() == CSR);

        if(this->GetNnz() > 0)
        {
            assert(col != NULL);
            assert(val != NULL);

            this->matrix_->CopyFromCSR(row_offsets, col, val);
        }

        this->object_name_ = "Imported from CSR matrix";

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::CopyToCSR(PtrType* row_offsets, int* col, ValueType* val) const
    {
        log_debug(this, "LocalMatrix::CopyToCSR()", row_offsets, col, val);

        assert(row_offsets != NULL);
        assert(this->GetFormat() == CSR);

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            assert(col != NULL);
            assert(val != NULL);

            this->matrix_->CopyToCSR(row_offsets, col, val);
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::CopyFromCOO(const int* row, const int* col, const ValueType* val)
    {
        log_debug(this, "LocalMatrix::CopyFromCOO()", row, col, val);

        assert(row != NULL);
        assert(col != NULL);
        assert(val != NULL);
        assert(this->GetFormat() == COO);

        if(this->GetNnz() > 0)
        {
            this->matrix_->CopyFromCOO(row, col, val);
        }

        this->object_name_ = "Imported from COO matrix";

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::CopyToCOO(int* row, int* col, ValueType* val) const
    {
        log_debug(this, "LocalMatrix::CopyToCOO()", row, col, val);

        assert(row != NULL);
        assert(col != NULL);
        assert(val != NULL);
        assert(this->GetFormat() == COO);

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            this->matrix_->CopyToCOO(row, col, val);
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::CopyFromHostCSR(const PtrType*     row_offset,
                                                 const int*         col,
                                                 const ValueType*   val,
                                                 const std::string& name,
                                                 int64_t            nnz,
                                                 int64_t            nrow,
                                                 int64_t            ncol)
    {
        log_debug(
            this, "LocalMatrix::CopyFromHostCSR()", row_offset, col, val, name, nnz, nrow, ncol);

        assert(nnz >= 0);
        assert(nrow >= 0);
        assert(ncol >= 0);
        assert(row_offset != NULL);

        this->Clear();
        this->object_name_ = name;
        this->ConvertToCSR();

        if(nnz > 0)
        {
            assert(nrow > 0);
            assert(ncol > 0);
            assert(col != NULL);
            assert(val != NULL);
        }

        Rocalution_Backend_Descriptor backend       = this->local_backend_;
        unsigned int                  matrix_format = this->GetFormat();

        // init host matrix
        if(this->matrix_ == this->matrix_host_)
        {
            delete this->matrix_host_;
            this->matrix_host_
                = _rocalution_init_base_host_matrix<ValueType>(backend, matrix_format);
            this->matrix_ = this->matrix_host_;
        }
        else
        {
            // init accel matrix
            assert(this->matrix_ == this->matrix_accel_);

            delete this->matrix_accel_;
            this->matrix_accel_
                = _rocalution_init_base_backend_matrix<ValueType>(backend, matrix_format);
            this->matrix_ = this->matrix_accel_;
        }

        assert(nrow <= std::numeric_limits<int>::max());
        assert(ncol <= std::numeric_limits<int>::max());

        this->matrix_->CopyFromHostCSR(
            row_offset, col, val, nnz, static_cast<int>(nrow), static_cast<int>(ncol));

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ReadFileMTX(const std::string& filename)
    {
        log_debug(this, "LocalMatrix::ReadFileMTX()", filename);

        LOG_INFO("ReadFileMTX: filename=" << filename << "; reading...");

        this->Clear();

        bool err = this->matrix_->ReadFileMTX(filename);

        if((err == false) && (this->is_host_() == true) && (this->GetFormat() == COO))
        {
            LOG_INFO("Execution of LocalMatrix::ReadFileMTX() failed");
            this->Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }

        if(err == false)
        {
            // Move to host
            bool is_accel = this->is_accel_();
            this->MoveToHost();

            // Convert to COO
            unsigned int format   = this->GetFormat();
            int          blockdim = this->GetBlockDimension();
            this->ConvertToCOO();

            if(this->matrix_->ReadFileMTX(filename) == false)
            {
                LOG_INFO("ReadFileMTX: failed to read matrix " << filename);
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(is_accel == true)
            {
                this->MoveToAccelerator();
            }

            this->Sort();

            this->ConvertTo(format, blockdim);
        }
        else
        {
            this->Sort();
        }

        this->object_name_ = filename;

#ifdef DEBUG_MODE
        this->Check();
#endif

        LOG_INFO("ReadFileMTX: filename=" << filename << "; done");
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::WriteFileMTX(const std::string& filename) const
    {
        log_debug(this, "LocalMatrix::WriteFileMTX()", filename);

        LOG_INFO("WriteFileMTX: filename=" << filename << "; writing...");

#ifdef DEBUG_MODE
        this->Check();
#endif

        bool err = this->matrix_->WriteFileMTX(filename);

        if((err == false) && (this->is_host_() == true) && (this->GetFormat() == COO))
        {
            LOG_INFO("Execution of LocalMatrix::WriteFileMTX() failed");
            this->Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }

        if(err == false)
        {
            // Move to host
            LocalMatrix<ValueType> mat_host;
            mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
            mat_host.CopyFrom(*this);

            // Convert to COO
            mat_host.ConvertToCOO();

            if(mat_host.matrix_->WriteFileMTX(filename) == false)
            {
                LOG_INFO("Execution of LocalMatrix::WriteFileMTX() failed");
                mat_host.Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }
        }

        LOG_INFO("WriteFileMTX: filename=" << filename << "; done");
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ReadFileCSR(const std::string& filename)
    {
        log_debug(this, "LocalMatrix::ReadFileCSR()", filename);

        LOG_INFO("ReadFileCSR: filename=" << filename << "; reading...");

        this->Clear();

        bool err = this->matrix_->ReadFileCSR(filename);

        if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
        {
            LOG_INFO("Execution of LocalMatrix::ReadFileCSR() failed");
            this->Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }

        if(err == false)
        {
            // Move to host
            bool is_accel = this->is_accel_();
            this->MoveToHost();

            // Convert to CSR
            unsigned int format   = this->GetFormat();
            int          blockdim = this->GetBlockDimension();
            this->ConvertToCSR();

            if(this->matrix_->ReadFileCSR(filename) == false)
            {
                LOG_INFO("Execution of LocalMatrix::ReadFileCSR() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(is_accel == true)
            {
                this->MoveToAccelerator();
            }

            this->ConvertTo(format, blockdim);
        }

        this->object_name_ = filename;

#ifdef DEBUG_MODE
        this->Check();
#endif

        LOG_INFO("ReadFileCSR: filename=" << filename << "; done");
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ReadFileRSIO(const std::string& filename,
                                              bool               maintain_initial_format)
    {
        log_debug(this, "LocalMatrix::ReadFileRSIO()", filename);

        LOG_INFO("ReadFileRSIO: filename=" << filename << "; reading...");

        // Clear matrix object before reading
        this->Clear();

        // For reading, we need a host matrix
        bool is_accel = this->is_accel_();
        this->MoveToHost();

        // Store initial format
        unsigned int format   = this->GetFormat();
        int          blockdim = this->GetBlockDimension();

        // rocsparse I/O handle
        rocsparseio_handle handle;
        rocsparseio_status status;

        // Create handle
        status = rocsparseio_open(&handle, rocsparseio_rwmode_read, filename.c_str());

        if(status != rocsparseio_status_success)
        {
            LOG_INFO("Execution of LocalMatrix::ReadFileRSIO() failed: cannot open file");
            FATAL_ERROR(__FILE__, __LINE__);
        }

        // Get format
        rocsparseio_format rsio_format;
        status = rocsparseio_read_format(handle, &rsio_format);

        if(status != rocsparseio_status_success)
        {
            LOG_INFO("Execution of LocalMatrix::ReadFileRSIO() failed: cannot read format");
            FATAL_ERROR(__FILE__, __LINE__);
        }

        // Close file
        status = rocsparseio_close(handle);

        if(status != rocsparseio_status_success)
        {
            LOG_INFO("Execution of LocalMatrix::ReadFileRSIO() failed: cannot close file");
            FATAL_ERROR(__FILE__, __LINE__);
        }

        switch(rsio_format)
        {
        case rocsparseio_format_dense_matrix:
        {
            LOG_INFO("ReadFileRSIO: filename=" << filename << "; format=DENSE");
            this->ConvertToDENSE();
            break;
        }
        case rocsparseio_format_sparse_csx:
        {
            LOG_INFO("ReadFileRSIO: filename=" << filename << "; format=CSR");
            this->ConvertToCSR();
            break;
        }
        case rocsparseio_format_sparse_gebsx:
        {
            LOG_INFO("ReadFileRSIO: filename=" << filename << "; format=BCSR");
            this->ConvertToBCSR(blockdim);
            break;
        }
        case rocsparseio_format_sparse_coo:
        {
            LOG_INFO("ReadFileRSIO: filename=" << filename << "; format=COO");
            this->ConvertToCOO();
            break;
        }
        case rocsparseio_format_sparse_ell:
        {
            LOG_INFO("ReadFileRSIO: filename=" << filename << "; format=ELL");
            this->ConvertToELL();
            break;
        }
        case rocsparseio_format_sparse_dia:
        {
            LOG_INFO("ReadFileRSIO: filename=" << filename << "; format=DIA");
            this->ConvertToDIA();
            break;
        }
        case rocsparseio_format_sparse_hyb:
        {
            LOG_INFO("ReadFileRSIO: filename=" << filename << "; format=HYB");
            this->ConvertToHYB();
            break;
        }
        case rocsparseio_format_sparse_mcsx:
        {
            LOG_INFO("ReadFileRSIO: filename=" << filename << "; format=MCSR");
            this->ConvertToMCSR();
            break;
        }
        case rocsparseio_format_dense_vector:
        {
            LOG_INFO("Execution of LocalMatrix::ReadFileRSIO() failed: unknown format");
            FATAL_ERROR(__FILE__, __LINE__);
        }
        }

        // Read in matrix
        if(!this->matrix_->ReadFileRSIO(filename))
        {
            LOG_INFO("Execution of LocalMatrix::ReadFileRSIO() failed");
            this->Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }

        // Move back to initial backend
        if(is_accel == true)
        {
            this->MoveToAccelerator();
        }

        // Warning on implicit conversion
        if(maintain_initial_format == false && format != this->GetFormat())
        {
            LOG_VERBOSE_INFO(2,
                             "*** warning: LocalMatrix::ReadFileRSIO() conversion from "
                                 << _matrix_format_names[format] << " to "
                                 << _matrix_format_names[this->GetFormat()]);
        }

        // Move back to initial format, if requested
        if(maintain_initial_format == true)
        {
            this->ConvertTo(format, blockdim);
        }

        this->object_name_ = filename;

#ifdef DEBUG_MODE
        this->Check();
#endif

        LOG_INFO("ReadFileRSIO: filename=" << filename << "; done");
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::WriteFileCSR(const std::string& filename) const
    {
        log_debug(this, "LocalMatrix::WriteFileCSR()", filename);

        LOG_INFO("WriteFileCSR: filename=" << filename << "; writing...");

#ifdef DEBUG_MODE
        this->Check();
#endif

        bool err = this->matrix_->WriteFileCSR(filename);

        if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
        {
            LOG_INFO("Execution of LocalMatrix::WriteFileCSR() failed");
            this->Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }

        if(err == false)
        {
            // Move to host
            LocalMatrix<ValueType> mat_host;
            mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
            mat_host.CopyFrom(*this);

            // Convert to CSR
            mat_host.ConvertToCSR();

            if(mat_host.matrix_->WriteFileCSR(filename) == false)
            {
                LOG_INFO("Execution of LocalMatrix::WriteFileCSR() failed");
                mat_host.Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }
        }

        LOG_INFO("WriteFileCSR: filename=" << filename << "; done");
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::WriteFileRSIO(const std::string& filename) const
    {
        log_debug(this, "LocalMatrix::WriteFileRSIO()", filename);

        LOG_INFO("WriteFileRSIO: filename=" << filename << "; writing...");

#ifdef DEBUG_MODE
        this->Check();
#endif

        bool err = this->matrix_->WriteFileRSIO(filename);

        if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
        {
            LOG_INFO("Execution of LocalMatrix::WriteFileRSIO() failed");
            this->Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }

        if(err == false)
        {
            // Move to host
            LocalMatrix<ValueType> mat_host;
            mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
            mat_host.CopyFrom(*this);

            // Try again on host
            err = mat_host.matrix_->WriteFileRSIO(filename);

            if((err == false) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Execution of LocalMatrix::WriteFileRSIO() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            // Convert to CSR
            mat_host.ConvertToCSR();

            if(mat_host.matrix_->WriteFileRSIO(filename) == false)
            {
                LOG_INFO("Execution of LocalMatrix::WriteFileRSIO() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }
        }

        LOG_INFO("WriteFileRSIO: filename=" << filename << "; done");
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::CopyFrom(const LocalMatrix<ValueType>& src)
    {
        log_debug(this, "LocalMatrix::CopyFrom()", (const void*&)src);

        assert(this != &src);

        this->matrix_->CopyFrom(*src.matrix_);
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::CopyFromAsync(const LocalMatrix<ValueType>& src)
    {
        log_debug(this, "LocalMatrix::CopyFromAsync()", (const void*&)src);

        assert(this->asyncf_ == false);
        assert(this != &src);

        this->matrix_->CopyFromAsync(*src.matrix_);

        this->asyncf_ = true;
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::CloneFrom(const LocalMatrix<ValueType>& src)
    {
        log_debug(this, "LocalMatrix::CloneFrom()", (const void*&)src);

        assert(this != &src);

#ifdef DEBUG_MODE
        src.Check();
#endif

        this->object_name_ = "Cloned from (";
        this->object_name_ += src.object_name_ + ")";
        this->local_backend_ = src.local_backend_;

        Rocalution_Backend_Descriptor backend = this->local_backend_;

        // delete current matrix
        if(this->matrix_ == this->matrix_host_)
        {
            delete this->matrix_host_;
            this->matrix_host_ = NULL;
        }
        else
        {
            delete this->matrix_accel_;
            this->matrix_accel_ = NULL;
        }

        if(src.matrix_ == src.matrix_host_)
        {
            // host
            this->matrix_host_ = _rocalution_init_base_host_matrix<ValueType>(
                backend, src.GetFormat(), src.GetBlockDimension());
            this->matrix_ = this->matrix_host_;
        }
        else
        {
            // accel
            this->matrix_accel_ = _rocalution_init_base_backend_matrix<ValueType>(
                backend, src.GetFormat(), src.GetBlockDimension());
            this->matrix_ = this->matrix_accel_;
        }

        this->matrix_->CopyFrom(*src.matrix_);

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::UpdateValuesCSR(ValueType* val)
    {
        log_debug(this, "LocalMatrix::UpdateValues()", val);

        assert(val != NULL);
        assert(this->GetNnz() > 0);
        assert(this->GetM() > 0);
        assert(this->GetN() > 0);
        assert(this->GetFormat() == CSR);

#ifdef DEBUG_MODE
        this->Check();
#endif

        // Move to host
        bool is_accel = this->is_accel_();
        this->MoveToHost();

        PtrType*   mat_row_offset = NULL;
        int*       mat_col        = NULL;
        ValueType* mat_val        = NULL;

        int64_t nrow = this->GetLocalM();
        int64_t ncol = this->GetLocalN();
        int64_t nnz  = this->GetLocalNnz();

        // Extract matrix pointers
        this->matrix_->LeaveDataPtrCSR(&mat_row_offset, &mat_col, &mat_val);

        LocalVector<ValueType> vec1;
        vec1.MoveToHost();
        vec1.SetDataPtr(&mat_val, "dummy1", nnz);
        vec1.CloneBackend(*this);

        // Dummy vector to follow the correct backend
        LocalVector<ValueType> vec2;
        vec2.MoveToHost();
        vec2.Allocate("dummy2", nnz);
        vec2.CopyFromData(val);
        vec2.CloneBackend(*this);

        vec2.LeaveDataPtr(&mat_val);

        assert(nrow <= std::numeric_limits<int>::max());
        assert(ncol <= std::numeric_limits<int>::max());

        // Set matrix pointers
        this->matrix_->SetDataPtrCSR(&mat_row_offset,
                                     &mat_col,
                                     &mat_val,
                                     nnz,
                                     static_cast<int>(nrow),
                                     static_cast<int>(ncol));

        if(is_accel)
        {
            LOG_VERBOSE_INFO(
                2, "*** warning: LocalMatrix::UpdateValuesCSR() is performed on the host");
            this->MoveToAccelerator();
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    bool LocalMatrix<ValueType>::is_host_(void) const
    {
        return (this->matrix_ == this->matrix_host_);
    }

    template <typename ValueType>
    bool LocalMatrix<ValueType>::is_accel_(void) const
    {
        return (this->matrix_ == this->matrix_accel_);
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::Info(void) const
    {
        std::string current_backend_name;

        if(this->matrix_ == this->matrix_host_)
        {
            current_backend_name = _rocalution_host_name[0];
        }
        else
        {
            assert(this->matrix_ == this->matrix_accel_);
            current_backend_name = _rocalution_backend_name[this->local_backend_.backend];
        }

        std::string format = _matrix_format_names[this->GetFormat()];

        if(this->GetFormat() == CSR)
        {
            std::stringstream sstr;
            sstr << "(" << 8 * sizeof(PtrType) << "," << 8 * sizeof(int) << ")";
            format += sstr.str();
        }

        LOG_INFO("LocalMatrix"
                 << " name=" << this->object_name_ << ";"
                 << " rows=" << this->GetM() << ";"
                 << " cols=" << this->GetN() << ";"
                 << " nnz=" << this->GetNnz() << ";"
                 << " prec=" << 8 * sizeof(ValueType) << "bit;"
                 << " format=" << format << ";"
                 << " host backend={" << _rocalution_host_name[0] << "};"
                 << " accelerator backend={"
                 << _rocalution_backend_name[this->local_backend_.backend] << "};"
                 << " current=" << current_backend_name);

        // this->matrix_->Info();
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::MoveToAccelerator(void)
    {
        log_debug(this, "LocalMatrix::MoveToAccelerator()");

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(_rocalution_available_accelerator() == false)
        {
            LOG_VERBOSE_INFO(4,
                             "*** info: LocalMatrix::MoveToAccelerator() no accelerator available "
                             "- doing nothing");
        }

        if((_rocalution_available_accelerator()) && (this->matrix_ == this->matrix_host_))
        {
            this->matrix_accel_ = _rocalution_init_base_backend_matrix<ValueType>(
                this->local_backend_, this->GetFormat(), this->GetBlockDimension());
            this->matrix_accel_->CopyFrom(*this->matrix_host_);

            this->matrix_ = this->matrix_accel_;
            delete this->matrix_host_;
            this->matrix_host_ = NULL;

            LOG_VERBOSE_INFO(
                4, "*** info: LocalMatrix::MoveToAccelerator() host to accelerator transfer");
        }

        // if on accelerator - do nothing
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::MoveToHost(void)
    {
        log_debug(this, "LocalMatrix::MoveToHost()");

        if((_rocalution_available_accelerator()) && (this->matrix_ == this->matrix_accel_))
        {
            this->matrix_host_ = _rocalution_init_base_host_matrix<ValueType>(
                this->local_backend_, this->GetFormat(), this->GetBlockDimension());
            this->matrix_host_->CopyFrom(*this->matrix_accel_);

            this->matrix_ = this->matrix_host_;
            delete this->matrix_accel_;
            this->matrix_accel_ = NULL;

            LOG_VERBOSE_INFO(4, "*** info: LocalMatrix::MoveToHost() accelerator to host transfer");
        }

        // if on host - do nothing

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::MoveToAcceleratorAsync(void)
    {
        log_debug(this, "LocalMatrix::MoveToAcceleratorAsync()");

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(_rocalution_available_accelerator() == false)
        {
            LOG_VERBOSE_INFO(4,
                             "*** info: LocalMatrix::MoveToAcceleratorAsync() no accelerator "
                             "available - doing nothing");
        }

        if((_rocalution_available_accelerator()) && (this->matrix_ == this->matrix_host_))
        {
            this->matrix_accel_ = _rocalution_init_base_backend_matrix<ValueType>(
                this->local_backend_, this->GetFormat(), this->GetBlockDimension());
            this->matrix_accel_->CopyFromAsync(*this->matrix_host_);
            this->asyncf_ = true;

            LOG_VERBOSE_INFO(4,
                             "*** info: LocalMatrix::MoveToAcceleratorAsync() host to accelerator "
                             "transfer (started)");
        }

        // if on accelerator - do nothing
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::MoveToHostAsync(void)
    {
        log_debug(this, "LocalMatrix::MoveToHostAsync()");

        if((_rocalution_available_accelerator()) && (this->matrix_ == this->matrix_accel_))
        {
            this->matrix_host_ = _rocalution_init_base_host_matrix<ValueType>(
                this->local_backend_, this->GetFormat(), this->GetBlockDimension());
            this->matrix_host_->CopyFromAsync(*this->matrix_accel_);
            this->asyncf_ = true;

            LOG_VERBOSE_INFO(
                4,
                "*** info: LocalMatrix::MoveToHostAsync() accelerator to host transfer (started)");
        }

        // if on host - do nothing

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::Sync(void)
    {
        log_debug(this, "LocalMatrix::Sync()");

        // check for active async transfer
        if(this->asyncf_ == true)
        {
            // The Move*Async function is active
            if((this->matrix_accel_ != NULL) && (this->matrix_host_ != NULL))
            {
                // MoveToHostAsync();
                if((_rocalution_available_accelerator() == true)
                   && (this->matrix_ == this->matrix_accel_))
                {
                    _rocalution_sync();

                    this->matrix_ = this->matrix_host_;
                    delete this->matrix_accel_;
                    this->matrix_accel_ = NULL;

                    LOG_VERBOSE_INFO(4,
                                     "*** info: LocalMatrix::MoveToHostAsync() accelerator to host "
                                     "transfer (synced)");
                }

                // MoveToAcceleratorAsync();
                if((_rocalution_available_accelerator() == true)
                   && (this->matrix_ == this->matrix_host_))
                {
                    _rocalution_sync();

                    this->matrix_ = this->matrix_accel_;
                    delete this->matrix_host_;
                    this->matrix_host_ = NULL;
                    LOG_VERBOSE_INFO(4,
                                     "*** info: LocalMatrix::MoveToAcceleratorAsync() host to "
                                     "accelerator transfer (synced)");
                }
            }
            else
            {
                // The Copy*Async function is active
                _rocalution_sync();
                LOG_VERBOSE_INFO(4, "*** info: LocalMatrix::Copy*Async() transfer (synced)");
            }
        }

        this->asyncf_ = false;
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ConvertToCSR(void)
    {
        this->ConvertTo(CSR);
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ConvertToMCSR(void)
    {
        this->ConvertTo(MCSR);
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ConvertToBCSR(int blockdim)
    {
        this->ConvertTo(BCSR, blockdim);
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ConvertToCOO(void)
    {
        this->ConvertTo(COO);
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ConvertToELL(void)
    {
        this->ConvertTo(ELL);
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ConvertToDIA(void)
    {
        this->ConvertTo(DIA);
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ConvertToHYB(void)
    {
        this->ConvertTo(HYB);
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ConvertToDENSE(void)
    {
        this->ConvertTo(DENSE);
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ConvertTo(unsigned int matrix_format, int blockdim)
    {
        log_debug(this, "LocalMatrix::ConvertTo()", matrix_format, blockdim);

        assert((matrix_format == DENSE) || (matrix_format == CSR) || (matrix_format == MCSR)
               || (matrix_format == BCSR) || (matrix_format == COO) || (matrix_format == DIA)
               || (matrix_format == ELL) || (matrix_format == HYB));

        LOG_VERBOSE_INFO(5,
                         "Converting " << _matrix_format_names[matrix_format] << " <- "
                                       << _matrix_format_names[this->GetFormat()]);

        if(this->GetFormat() != matrix_format)
        {
            if((this->GetFormat() != CSR) && (matrix_format != CSR))
            {
                this->ConvertToCSR();
            }

            // CPU matrix
            if(this->matrix_ == this->matrix_host_)
            {
                assert(this->matrix_host_ != NULL);

                HostMatrix<ValueType>* new_mat;
                new_mat = _rocalution_init_base_host_matrix<ValueType>(
                    this->local_backend_, matrix_format, blockdim);
                assert(new_mat != NULL);

                // If conversion fails, try CSR before we give up
                if(new_mat->ConvertFrom(*this->matrix_host_) == false)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: Matrix conversion to "
                                         << _matrix_format_names[matrix_format]
                                         << " failed, falling back to CSR format");
                    delete new_mat;
                    new_mat
                        = _rocalution_init_base_host_matrix<ValueType>(this->local_backend_, CSR);
                    assert(new_mat != NULL);

                    // If CSR conversion fails too, exit with error
                    if(new_mat->ConvertFrom(*this->matrix_host_) == false)
                    {
                        LOG_INFO("Unsupported (on host) conversion to CSR");
                        this->Info();
                        FATAL_ERROR(__FILE__, __LINE__);
                    }
                }

                delete this->matrix_host_;

                this->matrix_host_ = new_mat;
                this->matrix_      = this->matrix_host_;
            }
            else
            {
                // Accelerator Matrix
                assert(this->matrix_accel_ != NULL);

                AcceleratorMatrix<ValueType>* new_mat;
                new_mat = _rocalution_init_base_backend_matrix<ValueType>(
                    this->local_backend_, matrix_format, blockdim);
                assert(new_mat != NULL);

                if(new_mat->ConvertFrom(*this->matrix_accel_) == false)
                {
                    delete new_mat;

                    this->MoveToHost();
                    this->ConvertTo(matrix_format, blockdim);
                    this->MoveToAccelerator();

                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::ConvertTo() is performed on the host");
                }
                else
                {
                    delete this->matrix_accel_;

                    this->matrix_accel_ = new_mat;
                    this->matrix_       = this->matrix_accel_;
                }
            }

            assert(this->GetFormat() == matrix_format || this->GetFormat() == CSR);
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::Apply(const LocalVector<ValueType>& in,
                                       LocalVector<ValueType>*       out) const
    {
        log_debug(this, "LocalMatrix::Apply()", (const void*&)in, out);

        assert(out != NULL);

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            assert(in.GetSize() == this->GetN());
            assert(out->GetSize() == this->GetM());

            assert(((this->matrix_ == this->matrix_host_) && (in.vector_ == in.vector_host_)
                    && (out->vector_ == out->vector_host_))
                   || ((this->matrix_ == this->matrix_accel_) && (in.vector_ == in.vector_accel_)
                       && (out->vector_ == out->vector_accel_)));

            this->matrix_->Apply(*in.vector_, out->vector_);
        }
        else
        {
            // If matrix is empty, but not a 0x0 matrix, output vector need to be set to zero
            out->vector_->Zeros();
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ApplyAdd(const LocalVector<ValueType>& in,
                                          ValueType                     scalar,
                                          LocalVector<ValueType>*       out) const
    {
        log_debug(this, "LocalMatrix::ApplyAdd()", (const void*&)in, scalar, out);

        assert(out != NULL);

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            assert(in.GetSize() == this->GetN());
            assert(out->GetSize() == this->GetM());

            assert(((this->matrix_ == this->matrix_host_) && (in.vector_ == in.vector_host_)
                    && (out->vector_ == out->vector_host_))
                   || ((this->matrix_ == this->matrix_accel_) && (in.vector_ == in.vector_accel_)
                       && (out->vector_ == out->vector_accel_)));

            this->matrix_->ApplyAdd(*in.vector_, scalar, out->vector_);
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ExtractDiagonal(LocalVector<ValueType>* vec_diag) const
    {
        log_debug(this, "LocalMatrix::ExtractDiagonal()", vec_diag);

        assert(vec_diag != NULL);

        assert(
            ((this->matrix_ == this->matrix_host_) && (vec_diag->vector_ == vec_diag->vector_host_))
            || ((this->matrix_ == this->matrix_accel_)
                && (vec_diag->vector_ == vec_diag->vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            std::string vec_diag_name = "Diagonal elements of " + this->object_name_;
            vec_diag->Allocate(vec_diag_name, std::min(this->GetLocalM(), this->GetLocalN()));

            bool err = this->matrix_->ExtractDiagonal(vec_diag->vector_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::ExtractDiagonal() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalMatrix<ValueType> mat_host;
                mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_host.CopyFrom(*this);

                vec_diag->MoveToHost();

                mat_host.ConvertToCSR();

                if(mat_host.matrix_->ExtractDiagonal(vec_diag->vector_) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::ExtractDiagonal() failed");
                    this->Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2,
                        "*** warning: LocalMatrix::ExtractDiagonal() is performed in CSR format");
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::ExtractDiagonal() is performed on the host");

                    vec_diag->MoveToAccelerator();
                }
            }
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ExtractInverseDiagonal(LocalVector<ValueType>* vec_inv_diag) const
    {
        log_debug(this, "LocalMatrix::ExtractInverseDiagonal()", vec_inv_diag);

        assert(vec_inv_diag != NULL);

        assert(((this->matrix_ == this->matrix_host_)
                && (vec_inv_diag->vector_ == vec_inv_diag->vector_host_))
               || ((this->matrix_ == this->matrix_accel_)
                   && (vec_inv_diag->vector_ == vec_inv_diag->vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            std::string vec_inv_diag_name
                = "Inverse of the diagonal elements of " + this->object_name_;
            vec_inv_diag->Allocate(vec_inv_diag_name,
                                   std::min(this->GetLocalM(), this->GetLocalN()));

            bool err = this->matrix_->ExtractInverseDiagonal(vec_inv_diag->vector_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::ExtractInverseDiagonal() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalMatrix<ValueType> mat_host;
                mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_host.CopyFrom(*this);

                vec_inv_diag->MoveToHost();

                mat_host.ConvertToCSR();

                if(mat_host.matrix_->ExtractInverseDiagonal(vec_inv_diag->vector_) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::ExtractInverseDiagonal() failed");
                    this->Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::ExtractInverseDiagonal() is "
                                     "performed in CSR format");
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::ExtractInverseDiagonal() is "
                                     "performed on the host");

                    vec_inv_diag->MoveToAccelerator();
                }
            }
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ExtractSubMatrix(int64_t                 row_offset,
                                                  int64_t                 col_offset,
                                                  int64_t                 row_size,
                                                  int64_t                 col_size,
                                                  LocalMatrix<ValueType>* mat) const
    {
        log_debug(this,
                  "LocalMatrix::ExtractSubMatrix()",
                  row_offset,
                  col_offset,
                  row_size,
                  col_size,
                  mat);

        assert(this != mat);
        assert(mat != NULL);
        assert(row_size >= 0);
        assert(col_size >= 0);
        assert(static_cast<int64_t>(row_offset) <= this->GetM());
        assert(static_cast<int64_t>(col_offset) <= this->GetN());
        assert(this->is_host_() == mat->is_host_());

#ifdef DEBUG_MODE
        this->Check();
#endif

        // offsets and sizes are expected to be within 32 bits
        assert(row_offset <= std::numeric_limits<int>::max());
        assert(col_offset <= std::numeric_limits<int>::max());
        assert(row_size <= std::numeric_limits<int>::max());
        assert(col_size <= std::numeric_limits<int>::max());

        mat->Clear();

        // Submatrix should be same format as full matrix
        mat->ConvertTo(this->GetFormat(), this->GetBlockDimension());

        bool err = false;

        // if the sub matrix has only 1 row
        // it is computed on the host
        if((this->is_host_() == true) || (row_size > 1))
        {
            err = this->matrix_->ExtractSubMatrix(static_cast<int>(row_offset),
                                                  static_cast<int>(col_offset),
                                                  static_cast<int>(row_size),
                                                  static_cast<int>(col_size),
                                                  mat->matrix_);
        }

        if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
        {
            LOG_INFO("Computation of LocalMatrix::ExtractSubMatrix() failed");
            this->Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }

        if(err == false)
        {
            LocalMatrix<ValueType> mat_host;
            mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
            mat_host.CopyFrom(*this);

            mat->MoveToHost();

            mat_host.ConvertToCSR();
            mat->ConvertToCSR();

            if(mat_host.matrix_->ExtractSubMatrix(static_cast<int>(row_offset),
                                                  static_cast<int>(col_offset),
                                                  static_cast<int>(row_size),
                                                  static_cast<int>(col_size),
                                                  mat->matrix_)
               == false)
            {
                LOG_INFO("Computation of LocalMatrix::ExtractSubMatrix() failed");
                mat_host.Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(this->GetFormat() != CSR)
            {
                if(row_size > 1)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::ExtractSubMatrix() is "
                                     "performed in CSR format");
                }

                mat->ConvertTo(this->GetFormat(), this->GetBlockDimension());
            }

            if(this->is_accel_() == true)
            {
                if(row_size > 1)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::ExtractSubMatrix() is "
                                     "performed on the host");
                }

                mat->MoveToAccelerator();
            }

            if(row_size <= 1)
            {
                LOG_VERBOSE_INFO(2,
                                 "*** warning: LocalMatrix::ExtractSubMatrix() is performed on "
                                 "the host due to size = 1");
            }
        }

        std::ostringstream row_begin;
        std::ostringstream row_end;
        std::ostringstream col_begin;
        std::ostringstream col_end;

        row_begin << row_offset;
        row_end << row_offset + row_size - 1;
        col_begin << col_offset;
        col_end << col_offset + row_size - 1;

        std::string mat_name
            = "Submatrix of " + this->object_name_ + " " + "["
              + row_begin
                    .str() //static_cast<std::ostringstream*>(&(std::ostringstream() << row_offset))->str()
              + ","
              + col_begin
                    .str() //static_cast<std::ostringstream*>(&(std::ostringstream() << col_offset))->str()
              + "]-" + "["
              + row_end.str() //static_cast<std::ostringstream*>(
              //    &(std::ostringstream() << row_offset + row_size - 1))
              //    ->str()
              + ","
              + col_end.str() //static_cast<std::ostringstream*>(
              //&(std::ostringstream() << col_offset + row_size - 1))
              //->str()
              + "]";

        mat->object_name_ = mat_name;

#ifdef DEBUG_MODE
        mat->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ExtractSubMatrices(int                       row_num_blocks,
                                                    int                       col_num_blocks,
                                                    const int*                row_offset,
                                                    const int*                col_offset,
                                                    LocalMatrix<ValueType>*** mat) const
    {
        log_debug(this,
                  "LocalMatrix::ExtractSubMatrices()",
                  row_num_blocks,
                  col_num_blocks,
                  row_offset,
                  col_offset,
                  mat);

        assert(row_num_blocks > 0);
        assert(col_num_blocks > 0);
        assert(row_offset != NULL);
        assert(col_offset != NULL);
        assert(mat != NULL);
        assert(*mat != NULL);

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            // implementation via ExtractSubMatrix() calls
            // TODO OMP
            //#pragma omp parallel for schedule(dynamic,1) collapse(2)
            for(int i = 0; i < row_num_blocks; ++i)
            {
                for(int j = 0; j < col_num_blocks; ++j)
                {
                    this->ExtractSubMatrix(row_offset[i],
                                           col_offset[j],
                                           row_offset[i + 1] - row_offset[i],
                                           col_offset[j + 1] - col_offset[j],
                                           mat[i][j]);
                }
            }
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ExtractU(LocalMatrix<ValueType>* U, bool diag) const
    {
        log_debug(this, "LocalMatrix::ExtractU()", U, diag);

        assert(U != NULL);
        assert(U != this);

        assert(((this->matrix_ == this->matrix_host_) && (U->matrix_ == U->matrix_host_))
               || ((this->matrix_ == this->matrix_accel_) && (U->matrix_ == U->matrix_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = false;

            if(diag == true)
            {
                err = this->matrix_->ExtractUDiagonal(U->matrix_);
            }
            else
            {
                err = this->matrix_->ExtractU(U->matrix_);
            }

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::ExtractU() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalMatrix<ValueType> mat_host;
                mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_host.CopyFrom(*this);

                // Move to host
                U->MoveToHost();

                // Convert to CSR
                mat_host.ConvertToCSR();

                if(diag == true)
                {
                    err = mat_host.matrix_->ExtractUDiagonal(U->matrix_);
                }
                else
                {
                    err = mat_host.matrix_->ExtractU(U->matrix_);
                }

                if(err == false)
                {
                    LOG_INFO("Computation of LocalMatrix::ExtractU() failed");
                    mat_host.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::ExtractU() is performed in CSR format");

                    U->ConvertTo(this->GetFormat(), this->GetBlockDimension());
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::ExtractU() is performed on the host");

                    U->MoveToAccelerator();
                }
            }
        }

#ifdef DEBUG_MODE
        U->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ExtractL(LocalMatrix<ValueType>* L, bool diag) const
    {
        log_debug(this, "LocalMatrix::ExtractL()", L, diag);

        assert(L != NULL);
        assert(L != this);

        assert(((this->matrix_ == this->matrix_host_) && (L->matrix_ == L->matrix_host_))
               || ((this->matrix_ == this->matrix_accel_) && (L->matrix_ == L->matrix_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = false;

            if(diag == true)
            {
                err = this->matrix_->ExtractLDiagonal(L->matrix_);
            }
            else
            {
                err = this->matrix_->ExtractL(L->matrix_);
            }

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::ExtractL() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalMatrix<ValueType> mat_host;
                mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_host.CopyFrom(*this);

                // Move to host
                L->MoveToHost();

                // Convert to CSR
                mat_host.ConvertToCSR();

                if(diag == true)
                {
                    err = mat_host.matrix_->ExtractLDiagonal(L->matrix_);
                }
                else
                {
                    err = mat_host.matrix_->ExtractL(L->matrix_);
                }

                if(err == false)
                {
                    LOG_INFO("Computation of LocalMatrix::ExtractL() failed");
                    mat_host.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::ExtractL() is performed in CSR format");

                    L->ConvertTo(this->GetFormat(), this->GetBlockDimension());
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::ExtractL() is performed on the host");

                    L->MoveToAccelerator();
                }
            }
        }

#ifdef DEBUG_MODE
        L->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::LUAnalyse(void)
    {
        log_debug(this, "LocalMatrix::LUAnalyse()");

        if(this->GetNnz() > 0)
        {
            this->matrix_->LUAnalyse();
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::LUAnalyseClear(void)
    {
        log_debug(this, "LocalMatrix::LUAnalyseClear()");

        if(this->GetNnz() > 0)
        {
            this->matrix_->LUAnalyseClear();
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::LUSolve(const LocalVector<ValueType>& in,
                                         LocalVector<ValueType>*       out) const
    {
        log_debug(this, "LocalMatrix::LUSolve()", (const void*&)in, out);

        assert(out != NULL);
        assert(in.GetSize() == this->GetN());
        assert(out->GetSize() == this->GetM());

        assert(((this->matrix_ == this->matrix_host_) && (in.vector_ == in.vector_host_)
                && (out->vector_ == out->vector_host_))
               || ((this->matrix_ == this->matrix_accel_) && (in.vector_ == in.vector_accel_)
                   && (out->vector_ == out->vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->LUSolve(*in.vector_, out->vector_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::LUSolve() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalMatrix<ValueType> mat_host;
                mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_host.CopyFrom(*this);

                LocalVector<ValueType> vec_host;
                vec_host.CopyFrom(in);

                out->MoveToHost();

                // Try again
                err = mat_host.matrix_->LUSolve(*vec_host.vector_, out->vector_);

                if(err == false)
                {
                    mat_host.ConvertToCSR();

                    if(mat_host.matrix_->LUSolve(*vec_host.vector_, out->vector_) == false)
                    {
                        LOG_INFO("Computation of LocalMatrix::LUSolve() failed");
                        mat_host.Info();
                        FATAL_ERROR(__FILE__, __LINE__);
                    }

                    if(this->GetFormat() != CSR)
                    {
                        LOG_VERBOSE_INFO(
                            2, "*** warning: LocalMatrix::LUSolve() is performed in CSR format");
                    }
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::LUSolve() is performed on the host");

                    out->MoveToAccelerator();
                }
            }
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::LLAnalyse(void)
    {
        log_debug(this, "LocalMatrix::LLAnalyse()");

        if(this->GetNnz() > 0)
        {
            this->matrix_->LLAnalyse();
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::LLAnalyseClear(void)
    {
        log_debug(this, "LocalMatrix::LLAnalyseClear()");

        if(this->GetNnz() > 0)
        {
            this->matrix_->LLAnalyseClear();
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::LLSolve(const LocalVector<ValueType>& in,
                                         LocalVector<ValueType>*       out) const
    {
        log_debug(this, "LocalMatrix::LLSolve()", (const void*&)in, out);

        assert(out != NULL);
        assert(in.GetSize() == this->GetN());
        assert(out->GetSize() == this->GetM());

        assert(((this->matrix_ == this->matrix_host_) && (in.vector_ == in.vector_host_)
                && (out->vector_ == out->vector_host_))
               || ((this->matrix_ == this->matrix_accel_) && (in.vector_ == in.vector_accel_)
                   && (out->vector_ == out->vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->LLSolve(*in.vector_, out->vector_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::LLSolve() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalMatrix<ValueType> mat_host;
                mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_host.CopyFrom(*this);

                LocalVector<ValueType> vec_host;
                vec_host.CopyFrom(in);

                out->MoveToHost();

                mat_host.ConvertToCSR();

                if(mat_host.matrix_->LLSolve(*vec_host.vector_, out->vector_) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::LLSolve() failed");
                    mat_host.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::LLSolve() is performed in CSR format");
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::LLSolve() is performed on the host");

                    out->MoveToAccelerator();
                }
            }
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::LLSolve(const LocalVector<ValueType>& in,
                                         const LocalVector<ValueType>& inv_diag,
                                         LocalVector<ValueType>*       out) const
    {
        log_debug(this, "LocalMatrix::LLSolve()", (const void*&)in, (const void*&)inv_diag, out);

        assert(out != NULL);
        assert(in.GetSize() == this->GetN());
        assert(out->GetSize() == this->GetM());

        assert(((this->matrix_ == this->matrix_host_) && (in.vector_ == in.vector_host_)
                && (out->vector_ == out->vector_host_)
                && (inv_diag.vector_ == inv_diag.vector_host_))
               || ((this->matrix_ == this->matrix_accel_) && (in.vector_ == in.vector_accel_)
                   && (out->vector_ == out->vector_accel_)
                   && (inv_diag.vector_ == inv_diag.vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->LLSolve(*in.vector_, *inv_diag.vector_, out->vector_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::LLSolve() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalMatrix<ValueType> mat_host;
                mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_host.CopyFrom(*this);

                LocalVector<ValueType> vec_host;
                vec_host.CopyFrom(in);

                LocalVector<ValueType> inv_diag_host;
                inv_diag_host.CopyFrom(inv_diag);

                out->MoveToHost();

                mat_host.ConvertToCSR();

                if(mat_host.matrix_->LLSolve(
                       *vec_host.vector_, *inv_diag_host.vector_, out->vector_)
                   == false)
                {
                    LOG_INFO("Computation of LocalMatrix::LLSolve() failed");
                    mat_host.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::LLSolve() is performed in CSR format");
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::LLSolve() is performed on the host");

                    out->MoveToAccelerator();
                }
            }
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::LAnalyse(bool diag_unit)
    {
        log_debug(this, "LocalMatrix::LAnalyse()", diag_unit);

        if(this->GetNnz() > 0)
        {
            this->matrix_->LAnalyse(diag_unit);
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::LAnalyseClear(void)
    {
        log_debug(this, "LocalMatrix::LAnalyseClear()");

        if(this->GetNnz() > 0)
        {
            this->matrix_->LAnalyseClear();
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::LSolve(const LocalVector<ValueType>& in,
                                        LocalVector<ValueType>*       out) const
    {
        log_debug(this, "LocalMatrix::LSolve()", (const void*&)in, out);

        assert(out != NULL);
        assert(in.GetSize() == this->GetN());
        assert(out->GetSize() == this->GetM());

        assert(((this->matrix_ == this->matrix_host_) && (in.vector_ == in.vector_host_)
                && (out->vector_ == out->vector_host_))
               || ((this->matrix_ == this->matrix_accel_) && (in.vector_ == in.vector_accel_)
                   && (out->vector_ == out->vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->LSolve(*in.vector_, out->vector_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::LSolve() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalMatrix<ValueType> mat_host;
                mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_host.CopyFrom(*this);

                LocalVector<ValueType> vec_host;
                vec_host.CopyFrom(in);

                out->MoveToHost();

                mat_host.ConvertToCSR();

                if(mat_host.matrix_->LSolve(*vec_host.vector_, out->vector_) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::LSolve() failed");
                    mat_host.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::LSolve() is performed in CSR format");
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::LSolve() is performed on the host");

                    out->MoveToAccelerator();
                }
            }
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::UAnalyse(bool diag_unit)
    {
        log_debug(this, "LocalMatrix::UAnalyse()", diag_unit);

        if(this->GetNnz() > 0)
        {
            this->matrix_->UAnalyse(diag_unit);
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::UAnalyseClear(void)
    {
        log_debug(this, "LocalMatrix::UAnalyseClear()");

        if(this->GetNnz() > 0)
        {
            this->matrix_->UAnalyseClear();
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::USolve(const LocalVector<ValueType>& in,
                                        LocalVector<ValueType>*       out) const
    {
        log_debug(this, "LocalMatrix::USolve()", (const void*&)in, out);

        assert(out != NULL);
        assert(in.GetSize() == this->GetN());
        assert(out->GetSize() == this->GetM());

        assert(((this->matrix_ == this->matrix_host_) && (in.vector_ == in.vector_host_)
                && (out->vector_ == out->vector_host_))
               || ((this->matrix_ == this->matrix_accel_) && (in.vector_ == in.vector_accel_)
                   && (out->vector_ == out->vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->USolve(*in.vector_, out->vector_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::USolve() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalMatrix<ValueType> mat_host;
                mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_host.CopyFrom(*this);

                LocalVector<ValueType> vec_host;
                vec_host.CopyFrom(in);

                out->MoveToHost();

                mat_host.ConvertToCSR();

                if(mat_host.matrix_->USolve(*vec_host.vector_, out->vector_) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::USolve() failed");
                    mat_host.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::USolve() is performed in CSR format");
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::USolve() is performed on the host");

                    out->MoveToAccelerator();
                }
            }
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ItLUAnalyse(void)
    {
        log_debug(this, "LocalMatrix::ItLUAnalyse()");

        if(this->GetNnz() > 0)
        {
            this->matrix_->ItLUAnalyse();
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ItLUAnalyseClear(void)
    {
        log_debug(this, "LocalMatrix::ItLUAnalyseClear()");

        if(this->GetNnz() > 0)
        {
            this->matrix_->ItLUAnalyseClear();
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ItLUSolve(int                           max_iter,
                                           double                        tolerance,
                                           bool                          use_tol,
                                           const LocalVector<ValueType>& in,
                                           LocalVector<ValueType>*       out) const
    {
        log_debug(
            this, "LocalMatrix::ItLUSolve()", max_iter, tolerance, use_tol, (const void*&)in, out);

        assert(out != NULL);
        assert(in.GetSize() == this->GetN());
        assert(out->GetSize() == this->GetM());
        assert(max_iter > 0);

        if(use_tol == true)
        {
            assert(tolerance >= 0.0);
        }

        assert(((this->matrix_ == this->matrix_host_) && (in.vector_ == in.vector_host_)
                && (out->vector_ == out->vector_host_))
               || ((this->matrix_ == this->matrix_accel_) && (in.vector_ == in.vector_accel_)
                   && (out->vector_ == out->vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err
                = this->matrix_->ItLUSolve(max_iter, tolerance, use_tol, *in.vector_, out->vector_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::ItLUSolve() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                // Convert to CSR
                LocalMatrix<ValueType> mat_csr;
                mat_csr.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_csr.CopyFrom(*this);

                mat_csr.ConvertToCSR();

                // Try CSR format first
                err = mat_csr.matrix_->ItLUSolve(
                    max_iter, tolerance, use_tol, *in.vector_, out->vector_);
                if((err == false) && (this->is_host_() == true))
                {
                    LOG_INFO("Computation of LocalMatrix::ItLUSolve() failed");
                    mat_csr.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                // Try Host
                LocalVector<ValueType> vec_host;
                vec_host.CopyFrom(in);

                vec_host.MoveToHost();
                out->MoveToHost();
                mat_csr.MoveToHost();

                err = mat_csr.matrix_->ItLUSolve(
                    max_iter, tolerance, use_tol, *in.vector_, out->vector_);
                if((err == false))
                {
                    LOG_INFO("Computation of LocalMatrix::ItLUSolve() failed");
                    mat_csr.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::ItLUSolve() is performed in CSR format");
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::ItLUSolve() is performed on the host");

                    out->MoveToAccelerator();
                }
            }
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ItLLAnalyse(void)
    {
        log_debug(this, "LocalMatrix::ItLLAnalyse()");

        if(this->GetNnz() > 0)
        {
            this->matrix_->ItLLAnalyse();
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ItLLAnalyseClear(void)
    {
        log_debug(this, "LocalMatrix::ItLLAnalyseClear()");

        if(this->GetNnz() > 0)
        {
            this->matrix_->ItLLAnalyseClear();
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ItLLSolve(int                           max_iter,
                                           double                        tolerance,
                                           bool                          use_tol,
                                           const LocalVector<ValueType>& in,
                                           LocalVector<ValueType>*       out) const
    {
        log_debug(
            this, "LocalMatrix::ItLLSolve()", max_iter, tolerance, use_tol, (const void*&)in, out);

        assert(out != NULL);
        assert(in.GetSize() == this->GetN());
        assert(out->GetSize() == this->GetM());
        assert(max_iter > 0);

        if(use_tol == true)
        {
            assert(tolerance >= 0.0);
        }

        assert(((this->matrix_ == this->matrix_host_) && (in.vector_ == in.vector_host_)
                && (out->vector_ == out->vector_host_))
               || ((this->matrix_ == this->matrix_accel_) && (in.vector_ == in.vector_accel_)
                   && (out->vector_ == out->vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err
                = this->matrix_->ItLLSolve(max_iter, tolerance, use_tol, *in.vector_, out->vector_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::ItLLSolve() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                // Convert to CSR
                LocalMatrix<ValueType> mat_csr;
                mat_csr.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_csr.CopyFrom(*this);

                mat_csr.ConvertToCSR();

                // Try CSR format first
                err = mat_csr.matrix_->ItLLSolve(
                    max_iter, tolerance, use_tol, *in.vector_, out->vector_);
                if((err == false) && (this->is_host_() == true))
                {
                    LOG_INFO("Computation of LocalMatrix::ItLLSolve() failed");
                    mat_csr.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                // Try Host
                LocalVector<ValueType> vec_host;
                vec_host.CopyFrom(in);

                vec_host.MoveToHost();
                out->MoveToHost();
                mat_csr.MoveToHost();

                err = mat_csr.matrix_->ItLLSolve(
                    max_iter, tolerance, use_tol, *in.vector_, out->vector_);
                if((err == false))
                {
                    LOG_INFO("Computation of LocalMatrix::ItLLSolve() failed");
                    mat_csr.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::ItLLSolve() is performed in CSR format");
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::ItLLSolve() is performed on the host");

                    out->MoveToAccelerator();
                }
            }
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ItLLSolve(int                           max_iter,
                                           double                        tolerance,
                                           bool                          use_tol,
                                           const LocalVector<ValueType>& in,
                                           const LocalVector<ValueType>& inv_diag,
                                           LocalVector<ValueType>*       out) const
    {
        log_debug(this,
                  "LocalMatrix::ItLLSolve()",
                  max_iter,
                  tolerance,
                  use_tol,
                  (const void*&)in,
                  (const void*&)inv_diag,
                  out);

        assert(out != NULL);
        assert(in.GetSize() == this->GetN());
        assert(out->GetSize() == this->GetM());

        assert(max_iter > 0);

        if(use_tol == true)
        {
            assert(tolerance >= 0.0);
        }

        assert(((this->matrix_ == this->matrix_host_) && (in.vector_ == in.vector_host_)
                && (out->vector_ == out->vector_host_)
                && (inv_diag.vector_ == inv_diag.vector_host_))
               || ((this->matrix_ == this->matrix_accel_) && (in.vector_ == in.vector_accel_)
                   && (out->vector_ == out->vector_accel_)
                   && (inv_diag.vector_ == inv_diag.vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->ItLLSolve(
                max_iter, tolerance, use_tol, *in.vector_, *inv_diag.vector_, out->vector_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::ItLLSolve() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                // Convert to CSR
                LocalMatrix<ValueType> mat_csr;
                mat_csr.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_csr.CopyFrom(*this);

                mat_csr.ConvertToCSR();

                err = mat_csr.matrix_->ItLLSolve(
                    max_iter, tolerance, use_tol, *in.vector_, *inv_diag.vector_, out->vector_);
                if((err == false) && (this->is_host_() == true))
                {
                    LOG_INFO("Computation of LocalMatrix::ItLLSolve() failed");
                    mat_csr.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                // Try Host
                LocalVector<ValueType> vec_host;
                vec_host.CopyFrom(in);

                vec_host.MoveToHost();
                out->MoveToHost();
                mat_csr.MoveToHost();

                err = mat_csr.matrix_->ItLLSolve(
                    max_iter, tolerance, use_tol, *in.vector_, *inv_diag.vector_, out->vector_);
                if((err == false))
                {
                    LOG_INFO("Computation of LocalMatrix::ItLLSolve() failed");
                    mat_csr.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::ItLLSolve() is performed in CSR format");
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::ItLLSolve() is performed on the host");

                    out->MoveToAccelerator();
                }
            }
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ItLAnalyse(bool diag_unit)
    {
        log_debug(this, "LocalMatrix::ItLAnalyse()", diag_unit);

        if(this->GetNnz() > 0)
        {
            this->matrix_->ItLAnalyse(diag_unit);
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ItLAnalyseClear(void)
    {
        log_debug(this, "LocalMatrix::ItLAnalyseClear()");

        if(this->GetNnz() > 0)
        {
            this->matrix_->ItLAnalyseClear();
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ItLSolve(int                           max_iter,
                                          double                        tolerance,
                                          bool                          use_tol,
                                          const LocalVector<ValueType>& in,
                                          LocalVector<ValueType>*       out) const
    {
        log_debug(
            this, "LocalMatrix::ItLSolve()", max_iter, tolerance, use_tol, (const void*&)in, out);

        assert(out != NULL);
        assert(in.GetSize() == this->GetN());
        assert(out->GetSize() == this->GetM());

        assert(max_iter > 0);

        if(use_tol == true)
        {
            assert(tolerance >= 0.0);
        }

        assert(((this->matrix_ == this->matrix_host_) && (in.vector_ == in.vector_host_)
                && (out->vector_ == out->vector_host_))
               || ((this->matrix_ == this->matrix_accel_) && (in.vector_ == in.vector_accel_)
                   && (out->vector_ == out->vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err
                = this->matrix_->ItLSolve(max_iter, tolerance, use_tol, *in.vector_, out->vector_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::ItLSolve() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                // Convert to CSR
                LocalMatrix<ValueType> mat_csr;
                mat_csr.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_csr.CopyFrom(*this);

                mat_csr.ConvertToCSR();

                err = mat_csr.matrix_->ItLSolve(
                    max_iter, tolerance, use_tol, *in.vector_, out->vector_);
                if((err == false) && (this->is_host_() == true))
                {
                    LOG_INFO("Computation of LocalMatrix::ItLSolve() failed");
                    mat_csr.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                // Try Host
                LocalVector<ValueType> vec_host;
                vec_host.CopyFrom(in);

                vec_host.MoveToHost();
                out->MoveToHost();
                mat_csr.MoveToHost();

                err = mat_csr.matrix_->ItLSolve(
                    max_iter, tolerance, use_tol, *in.vector_, out->vector_);
                if((err == false))
                {
                    LOG_INFO("Computation of LocalMatrix::ItLSolve() failed");
                    mat_csr.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::ItLSolve() is performed in CSR format");
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::ItLSolve() is performed on the host");

                    out->MoveToAccelerator();
                }
            }
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ItUAnalyse(bool diag_unit)
    {
        log_debug(this, "LocalMatrix::ItUAnalyse()", diag_unit);

        if(this->GetNnz() > 0)
        {
            this->matrix_->ItUAnalyse(diag_unit);
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ItUAnalyseClear(void)
    {
        log_debug(this, "LocalMatrix::ItUAnalyseClear()");

        if(this->GetNnz() > 0)
        {
            this->matrix_->ItUAnalyseClear();
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ItUSolve(int                           max_iter,
                                          double                        tolerance,
                                          bool                          use_tol,
                                          const LocalVector<ValueType>& in,
                                          LocalVector<ValueType>*       out) const
    {
        log_debug(
            this, "LocalMatrix::ItUSolve()", max_iter, tolerance, use_tol, (const void*&)in, out);

        assert(out != NULL);
        assert(in.GetSize() == this->GetN());
        assert(out->GetSize() == this->GetM());

        assert(max_iter > 0);

        if(use_tol == true)
        {
            assert(tolerance >= 0.0);
        }

        assert(((this->matrix_ == this->matrix_host_) && (in.vector_ == in.vector_host_)
                && (out->vector_ == out->vector_host_))
               || ((this->matrix_ == this->matrix_accel_) && (in.vector_ == in.vector_accel_)
                   && (out->vector_ == out->vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err
                = this->matrix_->ItUSolve(max_iter, tolerance, use_tol, *in.vector_, out->vector_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::ItUSolve() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                // Convert to CSR
                LocalMatrix<ValueType> mat_csr;
                mat_csr.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_csr.CopyFrom(*this);

                err = mat_csr.matrix_->ItUSolve(
                    max_iter, tolerance, use_tol, *in.vector_, out->vector_);
                if((err == false) && (this->is_host_() == true))
                {
                    LOG_INFO("Computation of LocalMatrix::ItUSolve() failed");
                    mat_csr.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                // Try Host
                LocalVector<ValueType> vec_host;
                vec_host.CopyFrom(in);

                vec_host.MoveToHost();
                out->MoveToHost();
                mat_csr.MoveToHost();

                err = mat_csr.matrix_->ItUSolve(
                    max_iter, tolerance, use_tol, *in.vector_, out->vector_);
                if((err == false))
                {
                    LOG_INFO("Computation of LocalMatrix::ItUSolve() failed");
                    mat_csr.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::ItUSolve() is performed in CSR format");
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::ItUSolve() is performed on the host");

                    out->MoveToAccelerator();
                }
            }
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ILU0Factorize(void)
    {
        log_debug(this, "LocalMatrix::ILU0Factorize()");

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->ILU0Factorize();

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::ILU0Factorize() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                // Move to host
                bool is_accel = this->is_accel_();
                this->MoveToHost();

                // Convert to CSR
                unsigned int format   = this->GetFormat();
                int          blockdim = this->GetBlockDimension();
                this->ConvertToCSR();

                if(this->matrix_->ILU0Factorize() == false)
                {
                    LOG_INFO("Computation of LocalMatrix::ILU0Factorize() failed");
                    this->Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(format != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::ILU0Factorize() is performed in CSR format");

                    this->ConvertTo(format, blockdim);
                }

                if(is_accel == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::ILU0Factorize() is performed on the host");

                    this->MoveToAccelerator();
                }
            }
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ItILU0Factorize(ItILU0Algorithm alg,
                                                 int             option,
                                                 int             max_iter,
                                                 double          tolerance,
                                                 int*            niter,
                                                 double*         history)
    {
        log_debug(this, "LocalMatrix::ItILU0Factorize()", alg, option, max_iter, tolerance);

        assert(option >= 0);
        assert(max_iter > 0);
        assert(tolerance > 0);

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err
                = this->matrix_->ItILU0Factorize(alg, option, max_iter, tolerance, niter, history);

            if((err == false) && (this->GetFormat() == CSR) && (this->is_host_() == true))
            {
                LOG_INFO("Computation of LocalMatrix::ItILU0Factorize() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                // Move to host
                bool is_accel = this->is_accel_();
                this->MoveToHost();

                // Convert to CSR
                unsigned int format   = this->GetFormat();
                int          blockdim = this->GetBlockDimension();
                this->ConvertToCSR();

                // We fall back on standard ILU factorization (? assumption we know the implementation)
                if(this->matrix_->ItILU0Factorize(alg, option, max_iter, tolerance, niter, history)
                   == false)
                {
                    LOG_INFO("Computation of LocalMatrix::ItILU0Factorize() failed");
                    this->Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(format != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2,
                        "*** warning: LocalMatrix::ItILU0Factorize() is performed in CSR format");

                    this->ConvertTo(format, blockdim);
                }

                if(is_accel == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::ItILU0Factorize() is performed on the host");

                    this->MoveToAccelerator();
                }
            }
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ILUTFactorize(double t, int maxrow)
    {
        log_debug(this, "LocalMatrix::ILUTFactorize()", t, maxrow);

#ifdef DEBUG_MODE
        this->Check();
#endif

        assert(maxrow > 0);
        assert(t >= 0.0);

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->ILUTFactorize(t, maxrow);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::ILUTFactorize() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                // Move to host
                bool is_accel = this->is_accel_();
                this->MoveToHost();

                // Convert to CSR
                unsigned int format   = this->GetFormat();
                int          blockdim = this->GetBlockDimension();
                this->ConvertToCSR();

                if(this->matrix_->ILUTFactorize(t, maxrow) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::ILUTFactorize() failed");
                    this->Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(format != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::ILUTFactorize() is performed in CSR format");

                    this->ConvertTo(format, blockdim);
                }

                if(is_accel == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::ILUTFactorize() is performed on the host");

                    this->MoveToAccelerator();
                }
            }
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ILUpFactorize(int p, bool level)
    {
        log_debug(this, "LocalMatrix::ILUpFactorize()", p, level);

        assert(p >= 0);

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(p == 0)
        {
            this->ILU0Factorize();
        }
        else
        {
            if(this->GetNnz() > 0)
            {
                // with control levels
                if(level == true)
                {
                    LocalMatrix structure;
                    structure.CloneFrom(*this);
                    structure.SymbolicPower(p + 1);

                    bool err = this->matrix_->ILUpFactorizeNumeric(p, *structure.matrix_);

                    if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
                    {
                        LOG_INFO("Computation of LocalMatrix::ILUpFactorize() failed");
                        this->Info();
                        FATAL_ERROR(__FILE__, __LINE__);
                    }

                    if(err == false)
                    {
                        // Move to host
                        bool is_accel = this->is_accel_();
                        this->MoveToHost();
                        structure.MoveToHost();

                        // Convert to CSR
                        unsigned int format   = this->GetFormat();
                        int          blockdim = this->GetBlockDimension();
                        this->ConvertToCSR();
                        structure.ConvertToCSR();

                        if(this->matrix_->ILUpFactorizeNumeric(p, *structure.matrix_) == false)
                        {
                            LOG_INFO("Computation of LocalMatrix::ILUpFactorize() failed");
                            this->Info();
                            FATAL_ERROR(__FILE__, __LINE__);
                        }

                        if(format != CSR)
                        {
                            LOG_VERBOSE_INFO(2,
                                             "*** warning: LocalMatrix::ILUpFactorize() is "
                                             "performed in CSR format");

                            this->ConvertTo(format, blockdim);
                        }

                        if(is_accel == true)
                        {
                            LOG_VERBOSE_INFO(2,
                                             "*** warning: LocalMatrix::ILUpFactorize() is "
                                             "performed on the host");

                            this->MoveToAccelerator();
                        }
                    }

                    // without control levels
                }
                else
                {
                    LocalMatrix values;
                    values.CloneFrom(*this);

                    this->SymbolicPower(p + 1);
                    this->MatrixAdd(values);

                    bool err = this->matrix_->ILU0Factorize();

                    if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
                    {
                        LOG_INFO("Computation of LocalMatrix::ILUpFactorize() failed");
                        this->Info();
                        FATAL_ERROR(__FILE__, __LINE__);
                    }

                    if(err == false)
                    {
                        // Move to host
                        bool is_accel = this->is_accel_();
                        this->MoveToHost();

                        // Convert to CSR
                        unsigned int format   = this->GetFormat();
                        int          blockdim = this->GetBlockDimension();
                        this->ConvertToCSR();

                        if(this->matrix_->ILU0Factorize() == false)
                        {
                            LOG_INFO("Computation of LocalMatrix::ILUpFactorize() failed");
                            this->Info();
                            FATAL_ERROR(__FILE__, __LINE__);
                        }

                        if(format != CSR)
                        {
                            LOG_VERBOSE_INFO(2,
                                             "*** warning: LocalMatrix::ILUpFactorize() is "
                                             "performed in CSR format");

                            this->ConvertTo(format, blockdim);
                        }

                        if(is_accel == true)
                        {
                            LOG_VERBOSE_INFO(2,
                                             "*** warning: LocalMatrix::ILUpFactorize() is "
                                             "performed on the host");

                            this->MoveToAccelerator();
                        }
                    }
                }
            }
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ICFactorize(LocalVector<ValueType>* inv_diag)
    {
        log_debug(this, "LocalMatrix::ICFactorize()", inv_diag);

        assert(inv_diag != NULL);

        assert(
            ((this->matrix_ == this->matrix_host_) && (inv_diag->vector_ == inv_diag->vector_host_))
            || ((this->matrix_ == this->matrix_accel_)
                && (inv_diag->vector_ == inv_diag->vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->ICFactorize(inv_diag->vector_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::ICFactorize() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                // Move to host
                bool is_accel = this->is_accel_();
                this->MoveToHost();
                inv_diag->MoveToHost();

                // Convert to CSR
                unsigned int format   = this->GetFormat();
                int          blockdim = this->GetBlockDimension();
                this->ConvertToCSR();

                if(this->matrix_->ICFactorize(inv_diag->vector_) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::ICFactorize() failed");
                    this->Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(format != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::ICFactorize() is performed in CSR format");

                    this->ConvertTo(format, blockdim);
                }

                if(is_accel == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::ICFactorize() is performed on the host");

                    this->MoveToAccelerator();
                    inv_diag->MoveToAccelerator();
                }
            }
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::MultiColoring(int&              num_colors,
                                               int**             size_colors,
                                               LocalVector<int>* permutation) const
    {
        log_debug(this, "LocalMatrix::MultiColoring()", num_colors, size_colors, permutation);

        assert(*size_colors == NULL);
        assert(permutation != NULL);
        assert(this->GetM() == this->GetN());

        assert(((this->matrix_ == this->matrix_host_)
                && (permutation->vector_ == permutation->vector_host_))
               || ((this->matrix_ == this->matrix_accel_)
                   && (permutation->vector_ == permutation->vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            std::string vec_perm_name = "MultiColoring permutation of " + this->object_name_;
            permutation->Allocate(vec_perm_name, 0);
            permutation->CloneBackend(*this);

            bool err = this->matrix_->MultiColoring(num_colors, size_colors, permutation->vector_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::MultiColoring() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalMatrix<ValueType> mat_host;
                mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_host.CopyFrom(*this);

                // Move to host
                permutation->MoveToHost();

                // Convert to CSR
                mat_host.ConvertToCSR();

                if(mat_host.matrix_->MultiColoring(num_colors, size_colors, permutation->vector_)
                   == false)
                {
                    LOG_INFO("Computation of LocalMatrix::MultiColoring() failed");
                    this->Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::MultiColoring() is performed in CSR format");
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::MultiColoring() is performed on the host");

                    permutation->MoveToAccelerator();
                }
            }
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::MaximalIndependentSet(int&              size,
                                                       LocalVector<int>* permutation) const
    {
        log_debug(this, "LocalMatrix::MaximalIndependentSet()", size, permutation);

        assert(permutation != NULL);
        assert(this->GetM() == this->GetN());

        assert(((this->matrix_ == this->matrix_host_)
                && (permutation->vector_ == permutation->vector_host_))
               || ((this->matrix_ == this->matrix_accel_)
                   && (permutation->vector_ == permutation->vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            std::string vec_perm_name
                = "MaximalIndependentSet permutation of " + this->object_name_;
            permutation->Allocate(vec_perm_name, 0);
            permutation->CloneBackend(*this);

            bool err = this->matrix_->MaximalIndependentSet(size, permutation->vector_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::MaximalIndependentSet() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalMatrix<ValueType> mat_host;
                mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_host.CopyFrom(*this);

                // Move to host
                permutation->MoveToHost();

                // Convert to CSR
                mat_host.ConvertToCSR();

                if(mat_host.matrix_->MaximalIndependentSet(size, permutation->vector_) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::MaximalIndependentSet() failed");
                    this->Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::MaximalIndependentSet() is "
                                     "performed in CSR format");
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::MaximalIndependentSet() is "
                                     "performed on the host");

                    permutation->MoveToAccelerator();
                }
            }
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ZeroBlockPermutation(int&              size,
                                                      LocalVector<int>* permutation) const
    {
        log_debug(this, "LocalMatrix::ZeroBlockPermutation()", size, permutation);

        assert(permutation != NULL);
        assert(this->GetM() == this->GetN());

        assert(((this->matrix_ == this->matrix_host_)
                && (permutation->vector_ == permutation->vector_host_))
               || ((this->matrix_ == this->matrix_accel_)
                   && (permutation->vector_ == permutation->vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            std::string vec_perm_name = "ZeroBlockPermutation permutation of " + this->object_name_;
            permutation->Allocate(vec_perm_name, this->GetLocalM());

            bool err = this->matrix_->ZeroBlockPermutation(size, permutation->vector_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::ZeroBlockPermutation() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalMatrix<ValueType> mat_host;
                mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_host.CopyFrom(*this);

                // Move to host
                permutation->MoveToHost();

                // Convert to CSR
                mat_host.ConvertToCSR();

                if(mat_host.matrix_->ZeroBlockPermutation(size, permutation->vector_) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::ZeroBlockPermutation() failed");
                    mat_host.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::ZeroBlockPermutation() is "
                                     "performed in CSR format");
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::ZeroBlockPermutation() is "
                                     "performed on the host");

                    permutation->MoveToAccelerator();
                }
            }
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::Householder(int                     idx,
                                             ValueType&              beta,
                                             LocalVector<ValueType>* vec) const
    {
        log_debug(this, "LocalMatrix::Householder()", idx, beta, vec);

        assert(idx >= 0);
        assert(vec != NULL);

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->Householder(idx, beta, vec->vector_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == DENSE))
            {
                LOG_INFO("Computation of LocalMatrix::Householder() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalMatrix<ValueType> mat_host;
                mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_host.CopyFrom(*this);

                vec->MoveToHost();

                // Convert to DENSE
                mat_host.ConvertToDENSE();

                if(mat_host.matrix_->Householder(idx, beta, vec->vector_) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::Householder() failed");
                    mat_host.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != DENSE)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::Householder() is performed in DENSE format");
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::Householder() is performed on the host");

                    vec->MoveToAccelerator();
                }
            }
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::QRDecompose(void)
    {
        log_debug(this, "LocalMatrix::QRDecompose()");

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->QRDecompose();

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == DENSE))
            {
                LOG_INFO("Computation of LocalMatrix::QRDecompose() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                // Move to host
                bool is_accel = this->is_accel_();
                this->MoveToHost();

                // Convert to DENSE
                unsigned int format   = this->GetFormat();
                int          blockdim = this->GetBlockDimension();
                this->ConvertToDENSE();

                if(this->matrix_->QRDecompose() == false)
                {
                    LOG_INFO("Computation of LocalMatrix::QRDecompose() failed");
                    this->Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(format != DENSE)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::QRDecompose() is performed in DENSE format");

                    this->ConvertTo(format, blockdim);
                }

                if(is_accel == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::QRDecompose() is performed on the host");

                    this->MoveToAccelerator();
                }
            }
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::QRSolve(const LocalVector<ValueType>& in,
                                         LocalVector<ValueType>*       out) const
    {
        log_debug(this, "LocalMatrix::QRSolve()", (const void*&)in, out);

        assert(out != NULL);
        assert(in.GetSize() == this->GetN());
        assert(out->GetSize() == this->GetM());

        assert(((this->matrix_ == this->matrix_host_) && (in.vector_ == in.vector_host_)
                && (out->vector_ == out->vector_host_))
               || ((this->matrix_ == this->matrix_accel_) && (in.vector_ == in.vector_accel_)
                   && (out->vector_ == out->vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->QRSolve(*in.vector_, out->vector_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == DENSE))
            {
                LOG_INFO("Computation of LocalMatrix::QRSolve() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalMatrix<ValueType> mat_host;
                mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_host.CopyFrom(*this);

                LocalVector<ValueType> vec_host;
                vec_host.CopyFrom(in);

                mat_host.MoveToHost();
                vec_host.MoveToHost();
                out->MoveToHost();

                mat_host.ConvertToDENSE();

                if(mat_host.matrix_->QRSolve(*vec_host.vector_, out->vector_) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::QRSolve() failed");
                    mat_host.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != DENSE)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::QRSolve() is performed in DENSE format");
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::QRSolve() is performed on the host");

                    out->MoveToAccelerator();
                }
            }
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::Permute(const LocalVector<int>& permutation)
    {
        log_debug(this, "LocalMatrix::Permute()", (const void*&)permutation);

        assert((permutation.GetSize() == this->GetM()) || (permutation.GetSize() == this->GetN()));
        assert(permutation.GetSize() > 0);

        assert(((this->matrix_ == this->matrix_host_)
                && (permutation.vector_ == permutation.vector_host_))
               || ((this->matrix_ == this->matrix_accel_)
                   && (permutation.vector_ == permutation.vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->Permute(*permutation.vector_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::Permute() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalVector<int> perm_host;
                perm_host.CopyFrom(permutation);

                // Move to host
                this->MoveToHost();

                // Convert to CSR
                unsigned int format   = this->GetFormat();
                int          blockdim = this->GetBlockDimension();
                this->ConvertToCSR();

                if(this->matrix_->Permute(*perm_host.vector_) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::Permute() failed");
                    this->Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(format != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::Permute() is performed in CSR format");

                    this->ConvertTo(format, blockdim);
                }

                if(permutation.is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::Permute() is performed on the host");

                    this->MoveToAccelerator();
                }
            }
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::PermuteBackward(const LocalVector<int>& permutation)
    {
        log_debug(this, "LocalMatrix::PermuteBackward()", (const void*&)permutation);

        assert((permutation.GetSize() == this->GetM()) || (permutation.GetSize() == this->GetN()));
        assert(permutation.GetSize() > 0);

        assert(((this->matrix_ == this->matrix_host_)
                && (permutation.vector_ == permutation.vector_host_))
               || ((this->matrix_ == this->matrix_accel_)
                   && (permutation.vector_ == permutation.vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->PermuteBackward(*permutation.vector_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == COO))
            {
                LOG_INFO("Computation of LocalMatrix::PermuteBackward() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalVector<int> perm_host;
                perm_host.CopyFrom(permutation);

                // Move to host
                this->MoveToHost();

                // Convert to COO
                unsigned int format   = this->GetFormat();
                int          blockdim = this->GetBlockDimension();
                this->ConvertToCOO();

                if(this->matrix_->PermuteBackward(*perm_host.vector_) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::PermuteBackward() failed");
                    this->Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(format != COO)
                {
                    LOG_VERBOSE_INFO(
                        2,
                        "*** warning: LocalMatrix::PermuteBackward() is performed in COO format");

                    this->ConvertTo(format, blockdim);
                }

                if(permutation.is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::PermuteBackward() is performed on the host");

                    this->MoveToAccelerator();
                }
            }
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::CMK(LocalVector<int>* permutation) const
    {
        log_debug(this, "LocalMatrix::CMK()", permutation);

        assert(permutation != NULL);

        assert(((this->matrix_ == this->matrix_host_)
                && (permutation->vector_ == permutation->vector_host_))
               || ((this->matrix_ == this->matrix_accel_)
                   && (permutation->vector_ == permutation->vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->CMK(permutation->vector_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::CMK() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalMatrix<ValueType> mat_host;
                mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_host.CopyFrom(*this);

                // Move to host
                permutation->MoveToHost();

                // Convert to CSR
                mat_host.ConvertToCSR();

                if(mat_host.matrix_->CMK(permutation->vector_) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::CMK() failed");
                    mat_host.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::CMK() is performed in CSR format");
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::CMK() is performed on the host");

                    permutation->MoveToAccelerator();
                }
            }
        }

        std::string vec_name      = "CMK permutation of " + this->object_name_;
        permutation->object_name_ = vec_name;

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::RCMK(LocalVector<int>* permutation) const
    {
        log_debug(this, "LocalMatrix::RCMK()", permutation);

        assert(permutation != NULL);

        assert(((this->matrix_ == this->matrix_host_)
                && (permutation->vector_ == permutation->vector_host_))
               || ((this->matrix_ == this->matrix_accel_)
                   && (permutation->vector_ == permutation->vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->RCMK(permutation->vector_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::RCMK() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalMatrix<ValueType> mat_host;
                mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_host.CopyFrom(*this);

                // Move to host
                permutation->MoveToHost();

                // Convert to CSR
                mat_host.ConvertToCSR();

                if(mat_host.matrix_->RCMK(permutation->vector_) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::RCMK() failed");
                    mat_host.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::RCMK() is performed in CSR format");
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::RCMK() is performed on the host");

                    permutation->MoveToAccelerator();
                }
            }
        }

        std::string vec_name      = "RCMK permutation of " + this->object_name_;
        permutation->object_name_ = vec_name;

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ConnectivityOrder(LocalVector<int>* permutation) const
    {
        log_debug(this, "LocalMatrix::ConnectivityOrder()", permutation);

        assert(permutation != NULL);

        assert(((this->matrix_ == this->matrix_host_)
                && (permutation->vector_ == permutation->vector_host_))
               || ((this->matrix_ == this->matrix_accel_)
                   && (permutation->vector_ == permutation->vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->ConnectivityOrder(permutation->vector_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::ConnectivityOrder() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalMatrix<ValueType> mat_host;
                mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_host.CopyFrom(*this);

                // Move to host
                permutation->MoveToHost();

                // Convert to CSR
                mat_host.ConvertToCSR();

                if(mat_host.matrix_->ConnectivityOrder(permutation->vector_) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::ConnectivityOrder() failed");
                    mat_host.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2,
                        "*** warning: LocalMatrix::ConnectivityOrder() is performed in CSR format");
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2,
                        "*** warning: LocalMatrix::ConnectivityOrder() is performed on the host");

                    permutation->MoveToAccelerator();
                }
            }
        }

        std::string vec_name      = "ConnectivityOrder permutation of " + this->object_name_;
        permutation->object_name_ = vec_name;
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::SymbolicPower(int p)
    {
        log_debug(this, "LocalMatrix::SymbolicPower()", p);

        assert(p >= 1);

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->SymbolicPower(p);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::SymbolicPower() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                // Move to host
                bool is_accel = this->is_accel_();
                this->MoveToHost();

                // Convert to CSR
                unsigned int format   = this->GetFormat();
                int          blockdim = this->GetBlockDimension();
                this->ConvertToCSR();

                if(this->matrix_->SymbolicPower(p) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::SymbolicPower() failed");
                    this->Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(format != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::SymbolicPower() is performed in CSR format");

                    this->ConvertTo(format, blockdim);
                }

                if(is_accel == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::SymbolicPower() is performed on the host");

                    this->MoveToAccelerator();
                }
            }
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::MatrixAdd(const LocalMatrix<ValueType>& mat,
                                           ValueType                     alpha,
                                           ValueType                     beta,
                                           bool                          structure)
    {
        log_debug(this, "LocalMatrix::MatrixAdd()", (const void*&)mat, alpha, beta, structure);

        assert(&mat != this);
        assert(this->GetFormat() == mat.GetFormat());
        assert(this->GetM() == mat.GetM());
        assert(this->GetN() == mat.GetN());

        assert(((this->matrix_ == this->matrix_host_) && (mat.matrix_ == mat.matrix_host_))
               || ((this->matrix_ == this->matrix_accel_) && (mat.matrix_ == mat.matrix_accel_)));

#ifdef DEBUG_MODE
        this->Check();
        mat.Check();
#endif

        bool err = this->matrix_->MatrixAdd(*mat.matrix_, alpha, beta, structure);

        if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
        {
            LOG_INFO("Computation of LocalMatrix::MatrixAdd() failed");
            this->Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }

        if(err == false)
        {
            LocalMatrix<ValueType> mat_host;
            mat_host.ConvertTo(mat.GetFormat(), this->GetBlockDimension());
            mat_host.CopyFrom(mat);

            this->MoveToHost();

            this->ConvertToCSR();
            mat_host.ConvertToCSR();

            if(this->matrix_->MatrixAdd(*mat_host.matrix_, alpha, beta, structure) == false)
            {
                LOG_INFO("Computation of LocalMatrix::MatrixAdd() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(mat.GetFormat() != CSR)
            {
                LOG_VERBOSE_INFO(
                    2, "*** warning: LocalMatrix::MatrixAdd() is performed in CSR format");

                this->ConvertTo(mat.GetFormat(), this->GetBlockDimension());
            }

            if(mat.is_accel_() == true)
            {
                LOG_VERBOSE_INFO(2,
                                 "*** warning: LocalMatrix::MatrixAdd() is performed on the host");

                this->MoveToAccelerator();
            }
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::Gershgorin(ValueType& lambda_min, ValueType& lambda_max) const
    {
        log_debug(this, "LocalMatrix::Gershgorin()");

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->Gershgorin(lambda_min, lambda_max);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::Gershgorin() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalMatrix<ValueType> mat_host;
                mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_host.CopyFrom(*this);

                // Convert to CSR
                mat_host.ConvertToCSR();

                if(mat_host.matrix_->Gershgorin(lambda_min, lambda_max) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::Gershgorin() failed");
                    mat_host.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::Gershgorin() is performed in CSR format");
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::Gershgorin() is performed on the host");
                }
            }
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::Scale(ValueType alpha)
    {
        log_debug(this, "LocalMatrix::Scale()", alpha);

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->Scale(alpha);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::Scale() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                // Move to host
                bool is_accel = this->is_accel_();
                this->MoveToHost();

                // Convert to CSR
                unsigned int format   = this->GetFormat();
                int          blockdim = this->GetBlockDimension();
                this->ConvertToCSR();

                if(this->matrix_->Scale(alpha) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::Scale() failed");
                    this->Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(format != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::Scale() is performed in CSR format");

                    this->ConvertTo(format, blockdim);
                }

                if(is_accel == true)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::Scale() is performed on the host");

                    this->MoveToAccelerator();
                }
            }
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ScaleDiagonal(ValueType alpha)
    {
        log_debug(this, "LocalMatrix::ScaleDiagonal()", alpha);

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->ScaleDiagonal(alpha);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::ScaleDiagonal() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                // Move to host
                bool is_accel = this->is_accel_();
                this->MoveToHost();

                // Convert to CSR
                unsigned int format   = this->GetFormat();
                int          blockdim = this->GetBlockDimension();
                this->ConvertToCSR();

                if(this->matrix_->ScaleDiagonal(alpha) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::ScaleDiagonal() failed");
                    this->Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(format != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::ScaleDiagonal() is performed in CSR format");

                    this->ConvertTo(format, blockdim);
                }

                if(is_accel == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::ScaleDiagonal() is performed on the host");

                    this->MoveToAccelerator();
                }
            }
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ScaleOffDiagonal(ValueType alpha)
    {
        log_debug(this, "LocalMatrix::ScaleOffDiagonal()", alpha);

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->ScaleOffDiagonal(alpha);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::ScaleOffDiagonal() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                // Move to host
                bool is_accel = this->is_accel_();
                this->MoveToHost();

                // Convert to CSR
                unsigned int format   = this->GetFormat();
                int          blockdim = this->GetBlockDimension();
                this->ConvertToCSR();

                if(this->matrix_->ScaleOffDiagonal(alpha) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::ScaleOffDiagonal() failed");
                    this->Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(format != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2,
                        "*** warning: LocalMatrix::ScaleOffDiagonal() is performed in CSR format");

                    this->ConvertTo(format, blockdim);
                }

                if(is_accel == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::ScaleOffDiagonal() is performed on the host");

                    this->MoveToAccelerator();
                }
            }
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::AddScalar(ValueType alpha)
    {
        log_debug(this, "LocalMatrix::AddScalar()", alpha);

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->AddScalar(alpha);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::AddScalar() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                // Move to host
                bool is_accel = this->is_accel_();
                this->MoveToHost();

                // Convert to CSR
                unsigned int format   = this->GetFormat();
                int          blockdim = this->GetBlockDimension();
                this->ConvertToCSR();

                if(this->matrix_->AddScalar(alpha) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::AddScalar() failed");
                    this->Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(format != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::AddScalar() is performed in CSR format");

                    this->ConvertTo(format, blockdim);
                }

                if(is_accel == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::AddScalar() is performed on the host");

                    this->MoveToAccelerator();
                }
            }
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::AddScalarDiagonal(ValueType alpha)
    {
        log_debug(this, "LocalMatrix::AddScalarDiagonal()", alpha);

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->AddScalarDiagonal(alpha);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::AddScalarDiagonal() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                // Move to host
                bool is_accel = this->is_accel_();
                this->MoveToHost();

                // Convert to CSR
                unsigned int format   = this->GetFormat();
                int          blockdim = this->GetBlockDimension();
                this->ConvertToCSR();

                if(this->matrix_->AddScalarDiagonal(alpha) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::AddScalarDiagonal() failed");
                    this->Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(format != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2,
                        "*** warning: LocalMatrix::AddScalarDiagonal() is performed in CSR format");

                    this->ConvertTo(format, blockdim);
                }

                if(is_accel == true)
                {
                    LOG_VERBOSE_INFO(
                        2,
                        "*** warning: LocalMatrix::AddScalarDiagonal() is performed on the host");

                    this->MoveToAccelerator();
                }
            }
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::AddScalarOffDiagonal(ValueType alpha)
    {
        log_debug(this, "LocalMatrix::AddScalarOffDiagonal()", alpha);

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->AddScalarOffDiagonal(alpha);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::AddScalarOffDiagonal() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                // Move to host
                bool is_accel = this->is_accel_();
                this->MoveToHost();

                // Convert to CSR
                unsigned int format   = this->GetFormat();
                int          blockdim = this->GetBlockDimension();
                this->ConvertToCSR();

                if(this->matrix_->AddScalarOffDiagonal(alpha) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::AddScalarOffDiagonal() failed");
                    this->Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(format != CSR)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::AddScalarOffDiagonal() is "
                                     "performed in CSR format");

                    this->ConvertTo(format, blockdim);
                }

                if(is_accel == true)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::AddScalarOffDiagonal() is "
                                     "performed on the host");

                    this->MoveToAccelerator();
                }
            }
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::MatrixMult(const LocalMatrix<ValueType>& A,
                                            const LocalMatrix<ValueType>& B)
    {
        log_debug(this, "LocalMatrix::MatrixMult()", (const void*&)A, (const void*&)B);

        assert(&A != this);
        assert(&B != this);
        assert(A.GetN() == B.GetM());

        assert(A.GetFormat() == B.GetFormat());

        assert(((this->matrix_ == this->matrix_host_) && (A.matrix_ == A.matrix_host_)
                && (B.matrix_ == B.matrix_host_))
               || ((this->matrix_ == this->matrix_accel_) && (A.matrix_ == A.matrix_accel_)
                   && (B.matrix_ == B.matrix_accel_)));

#ifdef DEBUG_MODE
        this->Check();
        A.Check();
        B.Check();
#endif

        if(this->GetFormat() == DENSE)
        {
            if(this->GetNnz() != A.GetNnz())
            {
                this->Clear();
                this->AllocateDENSE("", A.GetLocalM(), B.GetLocalN());
            }
        }
        else
        {
            this->Clear();
        }

        this->object_name_ = A.object_name_ + " x " + B.object_name_;
        this->ConvertTo(A.GetFormat(), A.GetBlockDimension());

        bool err = this->matrix_->MatMatMult(*A.matrix_, *B.matrix_);

        if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
        {
            LOG_INFO("Computation of LocalMatrix::MatMatMult() failed");
            this->Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }

        if(err == false)
        {
            LocalMatrix<ValueType> A_host;
            LocalMatrix<ValueType> B_host;
            A_host.ConvertTo(A.GetFormat(), A.GetBlockDimension());
            B_host.ConvertTo(B.GetFormat(), B.GetBlockDimension());
            A_host.CopyFrom(A);
            B_host.CopyFrom(B);

            this->MoveToHost();

            A_host.ConvertToCSR();
            B_host.ConvertToCSR();
            this->ConvertToCSR();

            if(this->matrix_->MatMatMult(*A_host.matrix_, *B_host.matrix_) == false)
            {
                LOG_INFO("Computation of LocalMatrix::MatMatMult() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(A.GetFormat() != CSR)
            {
                LOG_VERBOSE_INFO(
                    2, "*** warning: LocalMatrix::MatMatMult() is performed in CSR format");

                this->ConvertTo(A.GetFormat(), A.GetBlockDimension());
            }

            if(A.is_accel_() == true)
            {
                LOG_VERBOSE_INFO(2,
                                 "*** warning: LocalMatrix::MatMatMult() is performed on the host");

                this->MoveToAccelerator();
            }
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::TripleMatrixProduct(const LocalMatrix<ValueType>& R,
                                                     const LocalMatrix<ValueType>& A,
                                                     const LocalMatrix<ValueType>& P)
    {
        log_debug(this,
                  "LocalMatrix::TripleMatrixProduct()",
                  (const void*&)R,
                  (const void*&)A,
                  (const void*&)P);

        assert(&R != this);
        assert(&A != this);
        assert(&P != this);

        assert(R.GetN() == A.GetM());
        assert(A.GetN() == P.GetM());
        assert(this->is_host_() == R.is_host_());
        assert(this->is_host_() == A.is_host_());
        assert(this->is_host_() == P.is_host_());

        // Only CSR matrices are supported
        LocalMatrix<ValueType> csr_R;
        LocalMatrix<ValueType> csr_A;
        LocalMatrix<ValueType> csr_P;

        const LocalMatrix<ValueType>* R_ptr = &R;
        const LocalMatrix<ValueType>* A_ptr = &A;
        const LocalMatrix<ValueType>* P_ptr = &P;

        // Check R
        if(R.GetFormat() != CSR)
        {
            csr_R.CloneFrom(R);
            csr_R.ConvertToCSR();
            R_ptr = &csr_R;
        }

        // Check A
        if(A.GetFormat() != CSR)
        {
            csr_A.CloneFrom(A);
            csr_A.ConvertToCSR();
            A_ptr = &csr_A;
        }

        // Check P
        if(P.GetFormat() != CSR)
        {
            csr_P.CloneFrom(P);
            csr_P.ConvertToCSR();
            P_ptr = &csr_P;
        }

        // Convert this to CSR
        unsigned int format   = this->GetFormat();
        int          blockdim = this->GetBlockDimension();

        this->ConvertToCSR();

        LocalMatrix<ValueType> tmp;
        tmp.CloneBackend(*this);

        tmp.MatrixMult(*R_ptr, *A_ptr);
        this->MatrixMult(tmp, *P_ptr);

        if(format != CSR || R.GetFormat() != CSR || A.GetFormat() != CSR || P.GetFormat() != CSR)
        {
            LOG_VERBOSE_INFO(
                2, "*** warning: LocalMatrix::TripleMatrixProduct() is performed in CSR format");

            if(format != CSR)
            {
                this->ConvertTo(format, blockdim);
            }
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::DiagonalMatrixMultR(const LocalVector<ValueType>& diag)
    {
        log_debug(this, "LocalMatrix::DiagonalMatrixMultR()", (const void*&)diag);

        assert((diag.GetSize() == this->GetM()) || (diag.GetSize() == this->GetN()));

        assert(((this->matrix_ == this->matrix_host_) && (diag.vector_ == diag.vector_host_))
               || ((this->matrix_ == this->matrix_accel_) && (diag.vector_ == diag.vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->DiagonalMatrixMultR(*diag.vector_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::DiagonalMatrixMultR() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalVector<ValueType> diag_host;
                diag_host.CopyFrom(diag);

                // Move to host
                this->MoveToHost();

                // Convert to CSR
                unsigned int format   = this->GetFormat();
                int          blockdim = this->GetBlockDimension();
                this->ConvertToCSR();

                if(this->matrix_->DiagonalMatrixMultR(*diag_host.vector_) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::DiagonalMatrixMultR() failed");
                    this->Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(format != CSR)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::DiagonalMatrixMultR() is performed "
                                     "in CSR format");

                    this->ConvertTo(format, blockdim);
                }

                if(diag.is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2,
                        "*** warning: LocalMatrix::DiagonalMatrixMultR() is performed on the host");

                    this->MoveToAccelerator();
                }
            }
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::DiagonalMatrixMult(const LocalVector<ValueType>& diag)
    {
        this->DiagonalMatrixMultR(diag);
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::DiagonalMatrixMultL(const LocalVector<ValueType>& diag)
    {
        log_debug(this, "LocalMatrix::DiagonalMatrixMultL()", (const void*&)diag);

        assert((diag.GetSize() == this->GetM()) || (diag.GetSize() == this->GetN()));

        assert(((this->matrix_ == this->matrix_host_) && (diag.vector_ == diag.vector_host_))
               || ((this->matrix_ == this->matrix_accel_) && (diag.vector_ == diag.vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->DiagonalMatrixMultL(*diag.vector_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::DiagonalMatrixMultL() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalVector<ValueType> diag_host;
                diag_host.CopyFrom(diag);

                // Move to host
                this->MoveToHost();

                // Convert to CSR
                unsigned int format   = this->GetFormat();
                int          blockdim = this->GetBlockDimension();
                this->ConvertToCSR();

                if(this->matrix_->DiagonalMatrixMultL(*diag_host.vector_) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::DiagonalMatrixMultL() failed");
                    this->Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(format != CSR)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::DiagonalMatrixMultL() is performed "
                                     "in CSR format");

                    this->ConvertTo(format, blockdim);
                }

                if(diag.is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2,
                        "*** warning: LocalMatrix::DiagonalMatrixMultL() is performed on the host");

                    this->MoveToAccelerator();
                }
            }
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::Compress(double drop_off)
    {
        log_debug(this, "LocalMatrix::Compress()", drop_off);

        assert(std::abs(drop_off) >= 0.0);

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->Compress(drop_off);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::Compress() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                // Move to host
                bool is_accel = this->is_accel_();
                this->MoveToHost();

                // Convert to CSR
                unsigned int format   = this->GetFormat();
                int          blockdim = this->GetBlockDimension();
                this->ConvertToCSR();

                if(this->matrix_->Compress(drop_off) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::Compress() failed");
                    this->Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(format != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::Compress() is performed in CSR format");

                    this->ConvertTo(format, blockdim);
                }

                if(is_accel == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::Compress() is performed on the host");

                    this->MoveToAccelerator();
                }
            }
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::Transpose(void)
    {
        log_debug(this, "LocalMatrix::Transpose()");

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->Transpose();

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::Transpose() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                // Move to host
                bool is_accel = this->is_accel_();
                this->MoveToHost();

                // Convert to CSR
                unsigned int format   = this->GetFormat();
                int          blockdim = this->GetBlockDimension();
                this->ConvertToCSR();

                if(this->matrix_->Transpose() == false)
                {
                    LOG_INFO("Computation of LocalMatrix::Transpose() failed");
                    this->Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(format != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::Transpose() is performed in CSR format");

                    this->ConvertTo(format, blockdim);
                }

                if(is_accel == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::Transpose() is performed on the host");

                    this->MoveToAccelerator();
                }
            }
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::Transpose(LocalMatrix<ValueType>* T) const
    {
        log_debug(this, "LocalMatrix::Transpose()", T);

        assert(T != NULL);
        assert(T != this);

        assert(((this->matrix_ == this->matrix_host_) && (T->matrix_ == T->matrix_host_))
               || ((this->matrix_ == this->matrix_accel_) && (T->matrix_ == T->matrix_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->Transpose(T->matrix_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::Transpose() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                // Move to host
                LocalMatrix<ValueType> mat_host;
                mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_host.CopyFrom(*this);

                T->MoveToHost();

                // Convert to CSR
                mat_host.ConvertToCSR();

                if(mat_host.matrix_->Transpose(T->matrix_) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::Transpose() failed");
                    mat_host.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::Transpose() is performed in CSR format");

                    T->ConvertTo(this->GetFormat(), this->GetBlockDimension());
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::Transpose() is performed on the host");

                    T->MoveToAccelerator();
                }
            }
        }

#ifdef DEBUG_MODE
        T->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::Sort(void)
    {
        log_debug(this, "LocalMatrix::Sort()");

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->Sort();

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::Sort() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                // Move to host
                bool is_accel = this->is_accel_();
                this->MoveToHost();

                // Try sorting on host
                if(this->matrix_->Sort() == false)
                {
                    // Convert to CSR
                    unsigned int format   = this->GetFormat();
                    int          blockdim = this->GetBlockDimension();
                    this->ConvertToCSR();

                    if(this->matrix_->Sort() == false)
                    {
                        LOG_INFO("Computation of LocalMatrix::Sort() failed");
                        this->Info();
                        FATAL_ERROR(__FILE__, __LINE__);
                    }

                    if(format != CSR)
                    {
                        LOG_VERBOSE_INFO(
                            2, "*** warning: LocalMatrix::Sort() is performed in CSR format");
                        this->ConvertTo(format, blockdim);
                    }
                }

                if(is_accel == true)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::Sort() is performed on the host");
                    this->MoveToAccelerator();
                }
            }
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::Key(long int& row_key, long int& col_key, long int& val_key) const
    {
        log_debug(this, "LocalMatrix::Key()", row_key, col_key, val_key);

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->Key(row_key, col_key, val_key);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::Key() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                // Move to host
                LocalMatrix<ValueType> mat_host;
                mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_host.CopyFrom(*this);

                // Convert to CSR
                mat_host.ConvertToCSR();

                if(mat_host.matrix_->Key(row_key, col_key, val_key) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::Key() failed");
                    mat_host.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::Key() is performed in CSR format");
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::Key() is performed on the host");
                }
            }
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::AMGConnect(ValueType eps, LocalVector<int>* connections) const
    {
        log_debug(this, "LocalMatrix::AMGConnect()", eps, connections);

        assert(eps > static_cast<ValueType>(0));
        assert(connections != NULL);

        assert(((this->matrix_ == this->matrix_host_)
                && (connections->vector_ == connections->vector_host_))
               || ((this->matrix_ == this->matrix_accel_)
                   && (connections->vector_ == connections->vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->AMGConnect(eps, connections->vector_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::AMGConnect() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalMatrix<ValueType> mat_host;
                mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_host.CopyFrom(*this);

                // Move to host
                connections->MoveToHost();

                // Convert to CSR
                mat_host.ConvertToCSR();

                if(mat_host.matrix_->AMGConnect(eps, connections->vector_) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::AMGConnect() failed");
                    mat_host.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::AMGConnect() is performed in CSR format");
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::AMGConnect() is performed on the host");

                    connections->MoveToAccelerator();
                }
            }
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::AMGAggregate(const LocalVector<int>& connections,
                                              LocalVector<int>*       aggregates) const
    {
        log_debug(this, "LocalMatrix::AMGAggregate()", (const void*&)connections, aggregates);

        assert(aggregates != NULL);

        assert(((this->matrix_ == this->matrix_host_)
                && (connections.vector_ == connections.vector_host_)
                && (aggregates->vector_ == aggregates->vector_host_))
               || ((this->matrix_ == this->matrix_accel_)
                   && (connections.vector_ == connections.vector_accel_)
                   && (aggregates->vector_ == aggregates->vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->AMGAggregate(*connections.vector_, aggregates->vector_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::AMGAggregate() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalMatrix<ValueType> mat_host;
                LocalVector<int>       conn_host;
                mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_host.CopyFrom(*this);
                conn_host.CopyFrom(connections);

                // Move to host
                aggregates->MoveToHost();

                // Convert to CSR
                mat_host.ConvertToCSR();

                if(mat_host.matrix_->AMGAggregate(*conn_host.vector_, aggregates->vector_) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::AMGAggregate() failed");
                    mat_host.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::AMGAggregate() is performed in CSR format");
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::AMGAggregate() is performed on the host");

                    aggregates->MoveToAccelerator();
                }
            }
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::AMGPMISAggregate(const LocalVector<int>& connections,
                                                  LocalVector<int>*       aggregates) const
    {
        log_debug(this, "LocalMatrix::AMGPMISAggregate()", (const void*&)connections, aggregates);

        assert(aggregates != NULL);

        assert(((this->matrix_ == this->matrix_host_)
                && (connections.vector_ == connections.vector_host_)
                && (aggregates->vector_ == aggregates->vector_host_))
               || ((this->matrix_ == this->matrix_accel_)
                   && (connections.vector_ == connections.vector_accel_)
                   && (aggregates->vector_ == aggregates->vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->AMGPMISAggregate(*connections.vector_, aggregates->vector_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::AMGPMISAggregate() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalMatrix<ValueType> mat_host;
                LocalVector<int>       conn_host;

                mat_host.ConvertTo(this->GetFormat());
                mat_host.CopyFrom(*this);
                conn_host.CopyFrom(connections);

                // Move to host
                aggregates->MoveToHost();

                // Convert to CSR
                mat_host.ConvertToCSR();

                if(mat_host.matrix_->AMGPMISAggregate(*conn_host.vector_, aggregates->vector_)
                   == false)
                {
                    LOG_INFO("Computation of LocalMatrix::AMGPMISAggregate() failed");
                    mat_host.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2,
                        "*** warning: LocalMatrix::AMGPMISAggregate() is performed in CSR format");
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::AMGPMISAggregate() is performed on the host");

                    aggregates->MoveToAccelerator();
                }
            }
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::AMGSmoothedAggregation(ValueType               relax,
                                                        const LocalVector<int>& aggregates,
                                                        const LocalVector<int>& connections,
                                                        LocalMatrix<ValueType>* prolong,
                                                        int                     lumping_strat) const
    {
        log_debug(this,
                  "LocalMatrix::AMGSmoothedAggregation()",
                  relax,
                  (const void*&)aggregates,
                  (const void*&)connections,
                  prolong);

        assert(relax > static_cast<ValueType>(0));
        assert(prolong != NULL);
        assert(this != prolong);
        assert(this->is_host_() == aggregates.is_host_());
        assert(this->is_host_() == connections.is_host_());
        assert(this->is_host_() == prolong->is_host_());

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->AMGSmoothedAggregation(
                relax, *aggregates.vector_, *connections.vector_, prolong->matrix_, lumping_strat);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::AMGSmoothedAggregation() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalMatrix<ValueType> mat_host;
                LocalVector<int>       conn_host;
                LocalVector<int>       aggr_host;
                mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_host.CopyFrom(*this);
                conn_host.CopyFrom(connections);
                aggr_host.CopyFrom(aggregates);

                // Move to host
                prolong->MoveToHost();

                // Convert to CSR
                mat_host.ConvertToCSR();

                if(mat_host.matrix_->AMGSmoothedAggregation(relax,
                                                            *aggr_host.vector_,
                                                            *conn_host.vector_,
                                                            prolong->matrix_,
                                                            lumping_strat)
                   == false)
                {
                    LOG_INFO("Computation of LocalMatrix::AMGSmoothedAggregation() failed");
                    mat_host.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::AMGSmoothedAggregation() is "
                                     "performed in CSR format");

                    prolong->ConvertTo(this->GetFormat(), this->GetBlockDimension());
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::AMGSmoothedAggregation() is "
                                     "performed on the host");

                    prolong->MoveToAccelerator();
                }
            }
        }

#ifdef DEBUG_MODE
        prolong->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::AMGAggregation(const LocalVector<int>& aggregates,
                                                LocalMatrix<ValueType>* prolong) const
    {
        log_debug(this, "LocalMatrix::AMGAggregation()", (const void*&)aggregates, prolong);

        assert(prolong != NULL);
        assert(this != prolong);
        assert(this->is_host_() == aggregates.is_host_());
        assert(this->is_host_() == prolong->is_host_());

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->AMGAggregation(*aggregates.vector_, prolong->matrix_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::AMGAggregation() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalMatrix<ValueType> mat_host;
                LocalVector<int>       aggr_host;
                mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_host.CopyFrom(*this);
                aggr_host.CopyFrom(aggregates);

                // Move to host
                prolong->MoveToHost();

                // Convert to CSR
                mat_host.ConvertToCSR();

                if(mat_host.matrix_->AMGAggregation(*aggr_host.vector_, prolong->matrix_) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::AMGAggregation() failed");
                    mat_host.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::AMGAggregation() is performed in CSR format");

                    prolong->ConvertTo(this->GetFormat(), this->GetBlockDimension());
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::AMGAggregation() is performed on the host");

                    prolong->MoveToAccelerator();
                }
            }
        }

#ifdef DEBUG_MODE
        prolong->Check();
#endif
    }

    template <typename ValueType>
    void
        LocalMatrix<ValueType>::AMGGreedyAggregate(ValueType             eps,
                                                   LocalVector<bool>*    connections,
                                                   LocalVector<int64_t>* aggregates,
                                                   LocalVector<int64_t>* aggregate_root_nodes) const
    {
        log_debug(this,
                  "LocalMatrix::AMGGreedyAggregate()",
                  connections,
                  aggregates,
                  aggregate_root_nodes);

        assert(connections != NULL);
        assert(aggregates != NULL);
        assert(aggregate_root_nodes != NULL);

        assert(this->is_host_() == connections->is_host_());
        assert(this->is_host_() == aggregates->is_host_());
        assert(this->is_host_() == aggregate_root_nodes->is_host_());

#ifdef DEBUG_MODE
        this->Check();
#endif
        // Only CSR matrices are supported
        LocalMatrix<ValueType>        csr_mat;
        const LocalMatrix<ValueType>* csr_ptr = this;

        if(csr_ptr->GetFormat() != CSR)
        {
            csr_mat.CloneFrom(*csr_ptr);
            csr_mat.ConvertToCSR();
            csr_ptr = &csr_mat;
        }

        if(this->GetNnz() > 0)
        {
            // Zero structures
            LocalMatrix<ValueType> zero_mat;
            zero_mat.CloneBackend(*this);

            LocalVector<int64_t> zero_vec;
            zero_vec.CloneBackend(*this);

            LocalVector<ValueType> diag;
            diag.CloneBackend(*this);
            diag.Allocate("diag", csr_ptr->GetM());

            csr_ptr->ExtractDiagonal(&diag);

            connections->Allocate("Connections", csr_ptr->GetNnz());
            aggregates->Allocate("Aggregates", csr_ptr->GetM());
            aggregate_root_nodes->Allocate("Aggregate root nodes", csr_ptr->GetM());

            csr_ptr->matrix_->AMGComputeStrongConnections(
                eps, *diag.vector_, *zero_vec.vector_, connections->vector_, *zero_mat.matrix_);

            bool status = csr_ptr->matrix_->AMGGreedyAggregate(
                *connections->vector_, aggregates->vector_, aggregate_root_nodes->vector_);

            if((status == false) && (this->is_host_() == true))
            {
                LOG_INFO("Computation of LocalMatrix::AMGGreedyAggregate() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(status == false)
            {
                LocalMatrix<ValueType> mat_host;
                mat_host.CopyFrom(*this);

                // Move to host
                aggregates->MoveToHost();
                aggregate_root_nodes->MoveToHost();
                connections->MoveToHost();

                if(mat_host.matrix_->AMGGreedyAggregate(
                       *connections->vector_, aggregates->vector_, aggregate_root_nodes->vector_)
                   == false)
                {
                    LOG_INFO("Computation of LocalMatrix::AMGGreedyAggregate() failed");
                    mat_host.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2,
                        "*** warning: LocalMatrix::AMGGreedyAggregate() is performed on the host");

                    aggregates->MoveToAccelerator();
                    aggregate_root_nodes->MoveToAccelerator();
                    connections->MoveToAccelerator();
                }
            }
        }

        if(this->GetFormat() != CSR)
        {
            LOG_VERBOSE_INFO(
                2, "*** warning: LocalMatrix::AMGGreedyAggregate() is performed in CSR format");
        }
    }
    template <typename ValueType>
    void LocalMatrix<ValueType>::AMGPMISAggregate(ValueType             eps,
                                                  LocalVector<bool>*    connections,
                                                  LocalVector<int64_t>* aggregates,
                                                  LocalVector<int64_t>* aggregate_root_nodes) const
    {
        log_debug(
            this, "LocalMatrix::AMGPMISAggregate()", connections, aggregates, aggregate_root_nodes);

        assert(connections != NULL);
        assert(aggregates != NULL);
        assert(aggregate_root_nodes != NULL);

        assert(this->is_host_() == connections->is_host_());
        assert(this->is_host_() == aggregates->is_host_());
        assert(this->is_host_() == aggregate_root_nodes->is_host_());

#ifdef DEBUG_MODE
        this->Check();
#endif
        // Only CSR matrices are supported
        LocalMatrix<ValueType>        csr_mat;
        const LocalMatrix<ValueType>* csr_ptr = this;
        if(csr_ptr->GetFormat() != CSR)
        {
            csr_mat.CloneFrom(*csr_ptr);
            csr_mat.ConvertToCSR();
            csr_ptr = &csr_mat;
        }
        if(this->GetNnz() > 0)
        {
            // Zero structures
            LocalMatrix<ValueType> zero_mat;
            zero_mat.CloneBackend(*this);

            LocalVector<int> izero_vec;
            izero_vec.CloneBackend(*this);

            LocalVector<int64_t> i64zero_vec;
            i64zero_vec.CloneBackend(*this);

            LocalVector<PtrType> pzero_vec;
            pzero_vec.CloneBackend(*this);

            connections->Allocate("Connections", csr_ptr->GetNnz());
            aggregates->Allocate("Aggregates", csr_ptr->GetM());
            aggregate_root_nodes->Allocate("Aggregate root nodes", csr_ptr->GetM());

            LocalVector<int> hash;
            hash.CloneBackend(*this);
            hash.Allocate("hash", csr_ptr->GetM());

            LocalVector<int> state;
            state.CloneBackend(*this);
            state.Allocate("state", csr_ptr->GetM());

            LocalVector<int> max_state;
            max_state.CloneBackend(*this);
            max_state.Allocate("max_state", csr_ptr->GetM());

            LocalVector<ValueType> diag;
            diag.CloneBackend(*this);
            diag.Allocate("diag", csr_ptr->GetM());

            int64_t global_col_begin = 0;
            int64_t global_col_end   = this->GetN();

            csr_ptr->ExtractDiagonal(&diag);

            // Compute connections
            csr_ptr->matrix_->AMGComputeStrongConnections(
                eps, *diag.vector_, *i64zero_vec.vector_, connections->vector_, *zero_mat.matrix_);

            // Initialize max state
            csr_ptr->matrix_->AMGPMISInitializeState(global_col_begin,
                                                     *connections->vector_,
                                                     max_state.vector_,
                                                     hash.vector_,
                                                     *zero_mat.matrix_);

            int iter = 0;
            while(true)
            {
                state.CopyFrom(max_state);

                // Find maximum distance 2 node
                bool undecided = false;
                csr_ptr->matrix_->AMGPMISFindMaxNeighbourNode(global_col_begin,
                                                              global_col_end,
                                                              undecided,
                                                              *connections->vector_,
                                                              *state.vector_,
                                                              *hash.vector_,
                                                              *pzero_vec.vector_,
                                                              *i64zero_vec.vector_,
                                                              *izero_vec.vector_,
                                                              *izero_vec.vector_,
                                                              max_state.vector_,
                                                              aggregates->vector_,
                                                              *zero_mat.matrix_);

                // If no more undecided nodes are left, we are done
                if(undecided == false)
                {
                    break;
                }

                ++iter;

                // Print some warning if number of iteration is getting huge
                if(iter > 20)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::AMGPMISAggregate() Current "
                                     "number of iterations: "
                                         << iter);
                }
            }

            aggregate_root_nodes->SetValues(-1);

            csr_ptr->matrix_->AMGPMISInitializeAggregateGlobalIndices(
                global_col_begin, *aggregates->vector_, aggregate_root_nodes->vector_);

            aggregates->ExclusiveSum();

            // Add any unassigned nodes to existing aggregations
            for(int k = 0; k < 2; k++)
            {
                state.CopyFrom(max_state);

                csr_ptr->matrix_->AMGPMISAddUnassignedNodesToAggregations(
                    global_col_begin,
                    *connections->vector_,
                    *state.vector_,
                    *i64zero_vec.vector_,
                    max_state.vector_,
                    aggregates->vector_,
                    aggregate_root_nodes->vector_,
                    *zero_mat.matrix_);
            }
        }

        if(this->GetFormat() != CSR)
        {
            LOG_VERBOSE_INFO(
                2, "*** warning: LocalMatrix::AMGPMISAggregate() is performed in CSR format");
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::AMGSmoothedAggregation(
        ValueType                   relax,
        const LocalVector<bool>&    connections,
        const LocalVector<int64_t>& aggregates,
        const LocalVector<int64_t>& aggregate_root_nodes,
        LocalMatrix<ValueType>*     prolong,
        int                         lumping_strat) const
    {
        log_debug(this,
                  "LocalMatrix::AMGSmoothedAggregation()",
                  relax,
                  (const void*&)connections,
                  (const void*&)aggregates,
                  (const void*&)aggregate_root_nodes,
                  prolong);

        assert(relax > static_cast<ValueType>(0));
        assert(prolong != NULL);
        assert(this != prolong);
        assert(this->is_host_() == connections.is_host_());
        assert(this->is_host_() == aggregates.is_host_());
        assert(this->is_host_() == aggregate_root_nodes.is_host_());
        assert(this->is_host_() == prolong->is_host_());

#ifdef DEBUG_MODE
        this->Check();
#endif
        // Only CSR matrices are supported
        LocalMatrix<ValueType>        csr_mat;
        const LocalMatrix<ValueType>* csr_ptr = this;

        if(csr_ptr->GetFormat() != CSR)
        {
            csr_mat.CloneFrom(*csr_ptr);
            csr_mat.ConvertToCSR();
            csr_ptr = &csr_mat;
        }

        //csr_ptr->matrix_->AMGSmoothedAggregation(relax, *connections.vector_, *aggregates.vector_, prolong->matrix_, lumping_strat);

        // Zero structures
        LocalMatrix<ValueType> zero_mat;
        zero_mat.CloneBackend(*this);

        LocalVector<int> izero_vec;
        izero_vec.CloneBackend(*this);

        LocalVector<int64_t> i64zero_vec;
        i64zero_vec.CloneBackend(*this);

        int64_t global_col_begin = 0;
        int64_t global_col_end   = this->GetN();

        // fine to coarse map
        LocalVector<int> f2c_map;
        f2c_map.CloneBackend(*this);
        f2c_map.Allocate("f2c map", csr_ptr->GetM() + 1);
        f2c_map.Zeros();

        // Determine number of non-zeros for P
        bool err = csr_ptr->matrix_->AMGSmoothedAggregationProlongNnz(global_col_begin,
                                                                      global_col_end,
                                                                      *connections.vector_,
                                                                      *aggregates.vector_,
                                                                      *aggregate_root_nodes.vector_,
                                                                      *zero_mat.matrix_,
                                                                      f2c_map.vector_,
                                                                      prolong->matrix_,
                                                                      NULL);

        if((err == false) && (csr_ptr->is_host_() == true) && (csr_ptr->GetFormat() == CSR))
        {
            LOG_INFO("Computation of LocalMatrix::ILU0Factorize() failed");
            csr_ptr->Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }

        if(err == false)
        {
            // Move to host
            LocalMatrix<ValueType> mat_host;
            mat_host.ConvertTo(csr_ptr->GetFormat());
            mat_host.CopyFrom(*csr_ptr);

            LocalVector<bool> connections_host;
            connections_host.CopyFrom(connections);

            LocalVector<int64_t> aggregates_host;
            aggregates_host.CopyFrom(aggregates);

            LocalVector<int64_t> aggregate_root_nodes_host;
            aggregate_root_nodes_host.CopyFrom(aggregate_root_nodes);

            zero_mat.MoveToHost();
            i64zero_vec.MoveToHost();
            f2c_map.MoveToHost();
            prolong->MoveToHost();

            // Convert to CSR
            unsigned int format   = mat_host.GetFormat();
            int          blockdim = mat_host.GetBlockDimension();
            mat_host.ConvertToCSR();

            if(mat_host.matrix_->AMGSmoothedAggregationProlongNnz(
                   global_col_begin,
                   global_col_end,
                   *connections_host.vector_,
                   *aggregates_host.vector_,
                   *aggregate_root_nodes_host.vector_,
                   *zero_mat.matrix_,
                   f2c_map.vector_,
                   prolong->matrix_,
                   NULL)
               == false)
            {
                LOG_INFO("Computation of LocalMatrix::AMGSmoothedAggregation() failed");
                mat_host.Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(mat_host.matrix_->AMGSmoothedAggregationProlongFill(
                   global_col_begin,
                   global_col_end,
                   lumping_strat,
                   relax,
                   *connections_host.vector_,
                   *aggregates_host.vector_,
                   *aggregate_root_nodes_host.vector_,
                   *i64zero_vec.vector_,
                   *f2c_map.vector_,
                   *zero_mat.matrix_,
                   prolong->matrix_,
                   NULL,
                   NULL)
               == false)
            {
                LOG_INFO("Computation of LocalMatrix::AMGSmoothedAggregation() failed");
                mat_host.Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(format != CSR)
            {
                LOG_VERBOSE_INFO(2,
                                 "*** warning: LocalMatrix::AMGSmoothedAggregation() is performed "
                                 "in CSR format");
            }

            if(csr_ptr->is_accel_() == true)
            {
                LOG_VERBOSE_INFO(
                    2,
                    "*** warning: LocalMatrix::AMGSmoothedAggregation() is performed on the host");

                zero_mat.MoveToAccelerator();
                i64zero_vec.MoveToAccelerator();
                f2c_map.MoveToAccelerator();
                prolong->MoveToAccelerator();
            }
        }
        else
        {
            // Fill P
            csr_ptr->matrix_->AMGSmoothedAggregationProlongFill(global_col_begin,
                                                                global_col_end,
                                                                lumping_strat,
                                                                relax,
                                                                *connections.vector_,
                                                                *aggregates.vector_,
                                                                *aggregate_root_nodes.vector_,
                                                                *i64zero_vec.vector_,
                                                                *f2c_map.vector_,
                                                                *zero_mat.matrix_,
                                                                prolong->matrix_,
                                                                NULL,
                                                                NULL);
        }

#ifdef DEBUG_MODE
        prolong->Check();
#endif
        if(this->GetFormat() != CSR)
        {
            LOG_VERBOSE_INFO(
                2, "*** warning: LocalMatrix::AMGSmoothedAggregation() is performed in CSR format");
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::AMGUnsmoothedAggregation(
        const LocalVector<int64_t>& aggregates,
        const LocalVector<int64_t>& aggregate_root_nodes,
        LocalMatrix<ValueType>*     prolong) const
    {
        log_debug(this,
                  "LocalMatrix::AMGUnsmoothedAggregation()",
                  (const void*&)aggregates,
                  (const void*&)aggregate_root_nodes,
                  prolong);

        assert(prolong != NULL);
        assert(this != prolong);
        assert(this->is_host_() == aggregates.is_host_());
        assert(this->is_host_() == aggregate_root_nodes.is_host_());
        assert(this->is_host_() == prolong->is_host_());

#ifdef DEBUG_MODE
        this->Check();
#endif

        // Only CSR matrices are supported
        LocalMatrix<ValueType>        csr_mat;
        const LocalMatrix<ValueType>* csr_ptr = this;

        if(csr_ptr->GetFormat() != CSR)
        {
            csr_mat.CloneFrom(*csr_ptr);
            csr_mat.ConvertToCSR();
            csr_ptr = &csr_mat;
        }

        if(this->GetNnz() > 0)
        {
            // csr_ptr->matrix_->AMGUnsmoothedAggregation(*aggregates.vector_, prolong->matrix_);

            // Zero structures
            LocalMatrix<ValueType> zero_mat;
            zero_mat.CloneBackend(*this);

            LocalVector<int> izero_vec;
            izero_vec.CloneBackend(*this);

            int64_t global_col_begin = 0;
            int64_t global_col_end   = this->GetN();

            // fine to coarse map
            LocalVector<int> f2c_map;
            f2c_map.CloneBackend(*this);
            f2c_map.Allocate("f2c map", csr_ptr->GetM() + 1);
            f2c_map.Zeros();

            // Determine number of non-zeros for P
            csr_ptr->matrix_->AMGUnsmoothedAggregationProlongNnz(global_col_begin,
                                                                 global_col_end,
                                                                 *aggregates.vector_,
                                                                 *aggregate_root_nodes.vector_,
                                                                 *zero_mat.matrix_,
                                                                 f2c_map.vector_,
                                                                 prolong->matrix_,
                                                                 NULL);

            // Fill P
            csr_ptr->matrix_->AMGUnsmoothedAggregationProlongFill(global_col_begin,
                                                                  global_col_end,
                                                                  *aggregates.vector_,
                                                                  *aggregate_root_nodes.vector_,
                                                                  *f2c_map.vector_,
                                                                  *zero_mat.matrix_,
                                                                  prolong->matrix_,
                                                                  NULL,
                                                                  NULL);
        }

#ifdef DEBUG_MODE
        prolong->Check();
#endif

        if(this->GetFormat() != CSR)
        {
            LOG_VERBOSE_INFO(
                2,
                "*** warning: LocalMatrix::AMGUnsmoothedAggregation() is performed in CSR format");
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::RSCoarsening(float              eps,
                                              LocalVector<int>*  CFmap,
                                              LocalVector<bool>* S) const
    {
        log_debug(this, "LocalMatrix::RSCoarsening()", eps, CFmap, S);

        assert(eps < 1.0f);
        assert(eps > 0.0f);
        assert(CFmap != NULL);
        assert(S != NULL);

        assert(((this->matrix_ == this->matrix_host_) && (CFmap->vector_ == CFmap->vector_host_)
                && (S->vector_ == S->vector_host_))
               || ((this->matrix_ == this->matrix_accel_)
                   && (CFmap->vector_ == CFmap->vector_accel_)
                   && (S->vector_ == S->vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->RSCoarsening(eps, CFmap->vector_, S->vector_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::RSCoarsening() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalMatrix<ValueType> mat_host;
                mat_host.ConvertTo(this->GetFormat());
                mat_host.CopyFrom(*this);

                // Move to host
                CFmap->MoveToHost();
                S->MoveToHost();

                // Convert to CSR
                mat_host.ConvertToCSR();

                if(mat_host.matrix_->RSCoarsening(eps, CFmap->vector_, S->vector_) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::RSCoarsening() failed");
                    mat_host.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::RSCoarsening() is performed in CSR format");
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::RSCoarsening() is performed on the host");

                    CFmap->MoveToAccelerator();
                    S->MoveToAccelerator();
                }
            }
        }

        std::string CFmap_name = "CF map of " + this->object_name_;
        std::string S_name     = "S of " + this->object_name_;

        CFmap->object_name_ = CFmap_name;
        S->object_name_     = S_name;
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::RSPMISCoarsening(float              eps,
                                                  LocalVector<int>*  CFmap,
                                                  LocalVector<bool>* S) const
    {
        log_debug(this, "LocalMatrix::RSPMISCoarsening()", eps, CFmap, S);

        assert(eps < 1.0f);
        assert(eps > 0.0f);
        assert(CFmap != NULL);
        assert(S != NULL);
        assert(this->is_host_() == CFmap->is_host_());
        assert(this->is_host_() == S->is_host_());

#ifdef DEBUG_MODE
        this->Check();
#endif

        // Only CSR matrices are supported
        LocalMatrix<ValueType>        csr_mat;
        const LocalMatrix<ValueType>* csr_ptr = this;

        if(csr_ptr->GetFormat() != CSR)
        {
            csr_mat.CloneFrom(*csr_ptr);
            csr_mat.ConvertToCSR();
            csr_ptr = &csr_mat;
        }

        if(this->GetNnz() > 0)
        {
            // Zero structures
            LocalMatrix<ValueType> zero;
            zero.CloneBackend(*this);

            // Allocate S
            // S is the auxiliary strength matrix such that
            //
            // S_ij = { 1   if i != j and -a_ij > eps * max(-a_ik) for k != i }
            //        { 0   otherwise                                         }
            //
            // This means, S_ij is 1, only if i strongly depends on j.
            // S has been extended to also work with matrices that do not have
            // fully non-positive off-diagonal entries.
            S->Allocate("S", csr_ptr->GetNnz());

            // Sample rng
            LocalVector<float> omega;
            omega.CloneBackend(*this);
            omega.Allocate("omega", csr_ptr->GetM());

            // Determine strong influences in the matrix
            csr_ptr->matrix_->RSPMISStrongInfluences(
                eps, S->vector_, omega.vector_, 0, *zero.matrix_);

            // Allocate CF mapping
            CFmap->Allocate("CF map", csr_ptr->GetM());

            // Mark all vertices as undecided
            CFmap->Zeros();

            LocalVector<bool> marked;
            marked.CloneBackend(*this);
            marked.Allocate("marked coarse", csr_ptr->GetM());

            // Iteratively find coarse and fine vertices until all undecided vertices have
            // been marked (JPL approach)
            int iter = 0;

            while(true)
            {
                // First, mark all vertices that have not been assigned yet, as coarse
                csr_ptr->matrix_->RSPMISUnassignedToCoarse(
                    CFmap->vector_, marked.vector_, *omega.vector_);

                // Now, correct previously marked vertices with respect to omega
                csr_ptr->matrix_->RSPMISCorrectCoarse(
                    CFmap->vector_, *S->vector_, *marked.vector_, *omega.vector_, *zero.matrix_);

                // Mark remaining edges of a coarse point to fine
                csr_ptr->matrix_->RSPMISCoarseEdgesToFine(
                    CFmap->vector_, *S->vector_, *zero.matrix_);

                // Now, we need to check whether we have vertices left that are marked
                // undecided, in order to restart the loop
                bool undecided;
                csr_ptr->matrix_->RSPMISCheckUndecided(undecided, *CFmap->vector_);

                // If no more undecided vertices are left, we are done
                if(undecided == false)
                {
                    break;
                }

                ++iter;

                // Print some warning if number of iteration is getting huge
                if(iter > 20)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::RSPMISCoarsening() Current "
                                     "number of iterations: "
                                         << iter);
                }
            }

            omega.Clear();
        }

        std::string CFmap_name = "CF map of " + this->object_name_;
        std::string S_name     = "S of " + this->object_name_;

        CFmap->object_name_ = CFmap_name;
        S->object_name_     = S_name;

        if(this->GetFormat() != CSR)
        {
            LOG_VERBOSE_INFO(
                2, "*** warning: LocalMatrix::RSPMISCoarsening() is performed in CSR format");
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::RSDirectInterpolation(const LocalVector<int>&  CFmap,
                                                       const LocalVector<bool>& S,
                                                       LocalMatrix<ValueType>*  prolong) const
    {
        log_debug(this,
                  "LocalMatrix::RSDirectInterpolation()",
                  (const void*&)CFmap,
                  (const void*&)S,
                  prolong);

        assert(prolong != NULL);
        assert(this != prolong);
        assert(this->is_host_() == CFmap.is_host_());
        assert(this->is_host_() == S.is_host_());
        assert(this->is_host_() == prolong->is_host_());

#ifdef DEBUG_MODE
        this->Check();
#endif

        // Only CSR matrices are supported
        LocalMatrix<ValueType>        csr_mat;
        const LocalMatrix<ValueType>* mat = this;

        if(this->GetFormat() != CSR)
        {
            csr_mat.CloneFrom(*this);
            csr_mat.ConvertToCSR();
            mat = &csr_mat;

            LOG_VERBOSE_INFO(
                2, "*** warning: LocalMatrix::RSDirectInterpolation() is performed in CSR format");
        }

        // Zero vector
        LocalVector<int64_t> zero;
        zero.CloneBackend(*this);

        // Zero matrix
        LocalMatrix<ValueType> zerom;
        zerom.CloneBackend(*this);

        // Fine to coarse map
        LocalVector<int> f2c;
        f2c.CloneBackend(*mat);
        f2c.Allocate("fine to coarse map", mat->GetM() + 1);

        // Amin/max
        LocalVector<ValueType> Amin;
        LocalVector<ValueType> Amax;

        Amin.CloneBackend(*mat);
        Amax.CloneBackend(*mat);

        Amin.Allocate("A min", mat->GetM());
        Amax.Allocate("A max", mat->GetM());

        // Determine number of non-zeros of P
        mat->matrix_->RSDirectProlongNnz(*CFmap.vector_,
                                         *S.vector_,
                                         *zerom.matrix_,
                                         Amin.vector_,
                                         Amax.vector_,
                                         f2c.vector_,
                                         prolong->matrix_,
                                         NULL);

        // Fill P
        mat->matrix_->RSDirectProlongFill(*zero.vector_,
                                          *f2c.vector_,
                                          *CFmap.vector_,
                                          *S.vector_,
                                          *zerom.matrix_,
                                          *Amin.vector_,
                                          *Amax.vector_,
                                          prolong->matrix_,
                                          NULL,
                                          NULL);

        // Sort P
        prolong->Sort();

        std::string prolong_name = "Prolongation Operator of " + mat->object_name_;
        prolong->object_name_    = prolong_name;

#ifdef DEBUG_MODE
        prolong->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::RSExtPIInterpolation(const LocalVector<int>&  CFmap,
                                                      const LocalVector<bool>& S,
                                                      bool                     FF1,
                                                      LocalMatrix<ValueType>*  prolong) const
    {
        log_debug(this,
                  "LocalMatrix::RSExtPIInterpolation()",
                  (const void*&)CFmap,
                  (const void*&)S,
                  FF1,
                  prolong);

        assert(prolong != NULL);
        assert(this != prolong);
        assert(this->is_host_() == CFmap.is_host_());
        assert(this->is_host_() == S.is_host_());
        assert(this->is_host_() == prolong->is_host_());

#ifdef DEBUG_MODE
        this->Check();
#endif

        // Only CSR matrices are supported
        LocalMatrix<ValueType>        csr_mat;
        const LocalMatrix<ValueType>* mat = this;

        if(this->GetFormat() != CSR)
        {
            csr_mat.CloneFrom(*this);
            csr_mat.ConvertToCSR();
            mat = &csr_mat;

            LOG_VERBOSE_INFO(
                2, "*** warning: LocalMatrix::RSExtPIInterpolation() is performed in CSR format");
        }

        // Zero vector
        LocalVector<PtrType>   zero;
        LocalVector<int64_t>   zerog;
        LocalVector<ValueType> zerov;
        LocalMatrix<ValueType> zerom;

        zero.CloneBackend(*this);
        zerog.CloneBackend(*this);
        zerov.CloneBackend(*this);
        zerom.CloneBackend(*this);

        // Fine to coarse map
        LocalVector<int> f2c;
        f2c.CloneBackend(*mat);
        f2c.Allocate("f2c map", mat->GetM() + 1);

        // Determine row nnz of P
        mat->RSExtPIProlongNnz(
            0, mat->GetN(), FF1, zerog, CFmap, S, zerom, zero, zerog, &f2c, prolong, NULL);

        // Fill column indices and values of P
        mat->RSExtPIProlongFill(0,
                                mat->GetN(),
                                FF1,
                                zerog,
                                f2c,
                                CFmap,
                                S,
                                zerom,
                                zero,
                                zerog,
                                zero,
                                zerog,
                                zerov,
                                prolong,
                                NULL,
                                NULL);

        std::string prolong_name = "Prolongation Operator of " + mat->object_name_;
        prolong->object_name_    = prolong_name;

#ifdef DEBUG_MODE
        prolong->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::InitialPairwiseAggregation(ValueType         beta,
                                                            int&              nc,
                                                            LocalVector<int>* G,
                                                            int&              Gsize,
                                                            int**             rG,
                                                            int&              rGsize,
                                                            int               ordering) const
    {
        log_debug(this,
                  "LocalMatrix::InitialPairwiseAggregation()",
                  beta,
                  nc,
                  G,
                  Gsize,
                  rG,
                  rGsize,
                  ordering);

        assert(*rG == NULL);
        assert(beta > static_cast<ValueType>(0));
        assert(G != NULL);

        assert(((this->matrix_ == this->matrix_host_) && (G->vector_ == G->vector_host_))
               || ((this->matrix_ == this->matrix_accel_) && (G->vector_ == G->vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->InitialPairwiseAggregation(
                beta, nc, G->vector_, Gsize, rG, rGsize, ordering);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::InitialPairwiseAggregation() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalMatrix<ValueType> mat_host;
                mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_host.CopyFrom(*this);

                // Move to host
                G->MoveToHost();

                // Convert to CSR
                mat_host.ConvertToCSR();

                if(mat_host.matrix_->InitialPairwiseAggregation(
                       beta, nc, G->vector_, Gsize, rG, rGsize, ordering)
                   == false)
                {
                    LOG_INFO("Computation of LocalMatrix::InitialPairwiseAggregation() failed");
                    mat_host.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::InitialPairwiseAggregation() is "
                                     "performed in CSR format");
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::InitialPairwiseAggregation() is "
                                     "performed on the host");

                    G->MoveToAccelerator();
                }
            }
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::InitialPairwiseAggregation(const LocalMatrix<ValueType>& mat,
                                                            ValueType                     beta,
                                                            int&                          nc,
                                                            LocalVector<int>*             G,
                                                            int&                          Gsize,
                                                            int**                         rG,
                                                            int&                          rGsize,
                                                            int ordering) const
    {
        log_debug(this,
                  "LocalMatrix::InitialPairwiseAggregation()",
                  (const void*&)mat,
                  beta,
                  nc,
                  G,
                  Gsize,
                  rG,
                  rGsize,
                  ordering);

        assert(*rG == NULL);
        assert(&mat != this);
        assert(beta > static_cast<ValueType>(0));
        assert(G != NULL);

        assert(((this->matrix_ == this->matrix_host_) && (mat.matrix_ == mat.matrix_host_)
                && (G->vector_ == G->vector_host_))
               || ((this->matrix_ == this->matrix_accel_) && (mat.matrix_ == mat.matrix_accel_)
                   && (G->vector_ == G->vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
        mat.Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->InitialPairwiseAggregation(
                *mat.matrix_, beta, nc, G->vector_, Gsize, rG, rGsize, ordering);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::InitialPairwiseAggregation() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalMatrix<ValueType> mat_host;
                LocalMatrix<ValueType> mat2_host;
                mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat2_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_host.CopyFrom(*this);
                mat2_host.CopyFrom(mat);

                // Move to host
                G->MoveToHost();

                // Convert to CSR
                mat_host.ConvertToCSR();
                mat2_host.ConvertToCSR();

                if(mat_host.matrix_->InitialPairwiseAggregation(
                       *mat2_host.matrix_, beta, nc, G->vector_, Gsize, rG, rGsize, ordering)
                   == false)
                {
                    LOG_INFO("Computation of LocalMatrix::InitialPairwiseAggregation() failed");
                    mat_host.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::InitialPairwiseAggregation() is "
                                     "performed in CSR format");
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::InitialPairwiseAggregation() is "
                                     "performed on the host");

                    G->MoveToAccelerator();
                }
            }
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::FurtherPairwiseAggregation(ValueType         beta,
                                                            int&              nc,
                                                            LocalVector<int>* G,
                                                            int&              Gsize,
                                                            int**             rG,
                                                            int&              rGsize,
                                                            int               ordering) const
    {
        log_debug(this,
                  "LocalMatrix::FurtherPairwiseAggregation()",
                  beta,
                  nc,
                  G,
                  Gsize,
                  rG,
                  rGsize,
                  ordering);

        assert(*rG != NULL);
        assert(beta > static_cast<ValueType>(0));
        assert(G != NULL);

        assert(((this->matrix_ == this->matrix_host_) && (G->vector_ == G->vector_host_))
               || ((this->matrix_ == this->matrix_accel_) && (G->vector_ == G->vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->FurtherPairwiseAggregation(
                beta, nc, G->vector_, Gsize, rG, rGsize, ordering);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::FurtherPairwiseAggregation() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalMatrix<ValueType> mat_host;
                mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_host.CopyFrom(*this);

                // Move to host
                G->MoveToHost();

                // Convert to CSR
                mat_host.ConvertToCSR();

                if(mat_host.matrix_->FurtherPairwiseAggregation(
                       beta, nc, G->vector_, Gsize, rG, rGsize, ordering)
                   == false)
                {
                    LOG_INFO("Computation of LocalMatrix::FurtherPairwiseAggregation() failed");
                    mat_host.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::FurtherPairwiseAggregation() is "
                                     "performed in CSR format");
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::FurtherPairwiseAggregation() is "
                                     "performed on the host");

                    G->MoveToAccelerator();
                }
            }
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::FurtherPairwiseAggregation(const LocalMatrix<ValueType>& mat,
                                                            ValueType                     beta,
                                                            int&                          nc,
                                                            LocalVector<int>*             G,
                                                            int&                          Gsize,
                                                            int**                         rG,
                                                            int&                          rGsize,
                                                            int ordering) const
    {
        log_debug(this,
                  "LocalMatrix::FurtherPairwiseAggregation()",
                  (const void*&)mat,
                  beta,
                  nc,
                  G,
                  Gsize,
                  rG,
                  rGsize,
                  ordering);

        assert(*rG != NULL);
        assert(&mat != this);
        assert(beta > static_cast<ValueType>(0));
        assert(G != NULL);

        assert(((this->matrix_ == this->matrix_host_) && (mat.matrix_ == mat.matrix_host_)
                && (G->vector_ == G->vector_host_))
               || ((this->matrix_ == this->matrix_accel_) && (mat.matrix_ == mat.matrix_accel_)
                   && (G->vector_ == G->vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
        mat.Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->FurtherPairwiseAggregation(
                *mat.matrix_, beta, nc, G->vector_, Gsize, rG, rGsize, ordering);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::FurtherPairwiseAggregation() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalMatrix<ValueType> mat_host;
                LocalMatrix<ValueType> mat2_host;
                mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat2_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_host.CopyFrom(*this);
                mat2_host.CopyFrom(mat);

                // Move to host
                G->MoveToHost();

                // Convert to CSR
                mat_host.ConvertToCSR();

                if(mat_host.matrix_->FurtherPairwiseAggregation(
                       *mat2_host.matrix_, beta, nc, G->vector_, Gsize, rG, rGsize, ordering)
                   == false)
                {
                    LOG_INFO("Computation of LocalMatrix::FurtherPairwiseAggregation() failed");
                    mat_host.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::FurtherPairwiseAggregation() is "
                                     "performed in CSR format");
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::FurtherPairwiseAggregation() is "
                                     "performed on the host");

                    G->MoveToAccelerator();
                }
            }
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::CoarsenOperator(LocalMatrix<ValueType>* Ac,
                                                 int                     nrow,
                                                 int                     ncol,
                                                 const LocalVector<int>& G,
                                                 int                     Gsize,
                                                 const int*              rG,
                                                 int                     rGsize) const
    {
        log_debug(this,
                  "LocalMatrix::CoarsenOperator()",
                  Ac,
                  nrow,
                  ncol,
                  (const void*&)G,
                  Gsize,
                  rG,
                  rGsize);

        assert(Ac != NULL);
        assert(Ac != this);
        assert(nrow > 0);
        assert(ncol > 0);
        assert(rG != NULL);
        assert(Gsize > 0);
        assert(rGsize > 0);

        assert(((this->matrix_ == this->matrix_host_) && (Ac->matrix_ == Ac->matrix_host_)
                && (G.vector_ == G.vector_host_))
               || ((this->matrix_ == this->matrix_accel_) && (Ac->matrix_ == Ac->matrix_accel_)
                   && (G.vector_ == G.vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->CoarsenOperator(
                Ac->matrix_, nrow, ncol, *G.vector_, Gsize, rG, rGsize);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::CoarsenOperator() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalMatrix<ValueType> mat_host;
                mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_host.CopyFrom(*this);

                LocalVector<int> vec_host;
                vec_host.CopyFrom(G);

                // Move to host
                Ac->MoveToHost();

                // Convert to CSR
                mat_host.ConvertToCSR();
                Ac->ConvertToCSR();

                if(mat_host.matrix_->CoarsenOperator(
                       Ac->matrix_, nrow, ncol, *vec_host.vector_, Gsize, rG, rGsize)
                   == false)
                {
                    LOG_INFO("Computation of LocalMatrix::CoarsenOperator() failed");
                    mat_host.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    // Adding COO due to MPI using COO ghost matrix
                    if(this->GetFormat() != COO)
                    {
                        LOG_VERBOSE_INFO(2,
                                         "*** warning: LocalMatrix::CoarsenOperator() is performed "
                                         "in CSR format");
                    }

                    Ac->ConvertTo(this->GetFormat(), this->GetBlockDimension());
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::CoarsenOperator() is performed on the host");

                    Ac->MoveToAccelerator();
                }
            }
        }

#ifdef DEBUG_MODE
        Ac->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::CreateFromMap(const LocalVector<int>& map, int64_t n, int64_t m)
    {
        log_debug(this, "LocalMatrix::CreateFromMap()", (const void*&)map, n, m);

        assert(map.GetSize() == n);
        assert(m > 0);
        assert(m <= std::numeric_limits<int>::max());
        assert(n <= std::numeric_limits<int>::max());

        assert(((this->matrix_ == this->matrix_host_) && (map.vector_ == map.vector_host_))
               || ((this->matrix_ == this->matrix_accel_) && (map.vector_ == map.vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->CreateFromMap(
                *map.vector_, static_cast<int>(n), static_cast<int>(m));

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::CreateFromMap() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalVector<int> map_host;
                map_host.CopyFrom(map);

                // Move to host
                this->MoveToHost();

                // Convert to CSR
                unsigned int format   = this->GetFormat();
                int          blockdim = this->GetBlockDimension();
                this->ConvertToCSR();

                if(this->matrix_->CreateFromMap(
                       *map_host.vector_, static_cast<int>(n), static_cast<int>(m))
                   == false)
                {
                    LOG_INFO("Computation of LocalMatrix::CreateFromMap() failed");
                    this->Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(format != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::CreateFromMap() is performed in CSR format");

                    this->ConvertTo(format, blockdim);
                }

                if(map.is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::CreateFromMap() is performed on the host");

                    this->MoveToAccelerator();
                }
            }
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::CreateFromMap(const LocalVector<int>& map,
                                               int64_t                 n,
                                               int64_t                 m,
                                               LocalMatrix<ValueType>* pro)
    {
        log_debug(this, "LocalMatrix::CreateFromMap()", (const void*&)map, n, m, pro);

        assert(pro != NULL);
        assert(this != pro);
        assert(map.GetSize() == n);
        assert(m > 0);
        assert(m <= std::numeric_limits<int>::max());
        assert(n <= std::numeric_limits<int>::max());

        assert(((this->matrix_ == this->matrix_host_) && (map.vector_ == map.vector_host_)
                && (pro->matrix_ == pro->matrix_host_))
               || ((this->matrix_ == this->matrix_accel_) && (map.vector_ == map.vector_accel_)
                   && (pro->matrix_ == pro->matrix_accel_)));

        this->Clear();
        pro->Clear();

        bool err = this->matrix_->CreateFromMap(
            *map.vector_, static_cast<int>(n), static_cast<int>(m), pro->matrix_);

        if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
        {
            LOG_INFO("Computation of LocalMatrix::CreateFromMap() failed");
            this->Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }

        if(err == false)
        {
            LocalVector<int> map_host;
            map_host.CopyFrom(map);

            // Move to host
            this->MoveToHost();
            pro->MoveToHost();

            // Convert to CSR
            unsigned int format   = this->GetFormat();
            int          blockdim = this->GetBlockDimension();
            this->ConvertToCSR();

            if(this->matrix_->CreateFromMap(
                   *map_host.vector_, static_cast<int>(n), static_cast<int>(m), pro->matrix_)
               == false)
            {
                LOG_INFO("Computation of LocalMatrix::CreateFromMap() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(format != CSR)
            {
                LOG_VERBOSE_INFO(
                    2, "*** warning: LocalMatrix::CreateFromMap() is performed in CSR format");

                this->ConvertTo(format, blockdim);
                pro->ConvertTo(format, blockdim);
            }

            if(map.is_accel_() == true)
            {
                LOG_VERBOSE_INFO(
                    2, "*** warning: LocalMatrix::CreateFromMap() is performed on the host");

                this->MoveToAccelerator();
                pro->MoveToAccelerator();
            }
        }

#ifdef DEBUG_MODE
        this->Check();
        pro->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::LUFactorize(void)
    {
        log_debug(this, "LocalMatrix::LUFactorize()");

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->LUFactorize();

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == DENSE))
            {
                LOG_INFO("Computation of LocalMatrix::LUFactorize() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                // Move to host
                bool is_accel = this->is_accel_();
                this->MoveToHost();

                // Convert to DENSE
                unsigned int format   = this->GetFormat();
                int          blockdim = this->GetBlockDimension();
                this->ConvertToDENSE();

                if(this->matrix_->LUFactorize() == false)
                {
                    LOG_INFO("Computation of LocalMatrix::LUFactorize() failed");
                    this->Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(format != DENSE)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::LUFactorize() is performed in DENSE format");

                    this->ConvertTo(format, blockdim);
                }

                if(is_accel == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::LUFactorize() is performed on the host");

                    this->MoveToAccelerator();
                }
            }
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::FSAI(int power, const LocalMatrix<ValueType>* pattern)
    {
        log_debug(this, "LocalMatrix::FSAI()", power, pattern);

        assert(power > 0);
        assert(pattern != this);
        assert(this->GetM() == this->GetN());

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err;

            if(pattern != NULL)
            {
                assert(((this->matrix_ == this->matrix_host_)
                        && (pattern->matrix_ == pattern->matrix_host_))
                       || ((this->matrix_ == this->matrix_accel_)
                           && (pattern->matrix_ == pattern->matrix_accel_)));
                err = this->matrix_->FSAI(power, pattern->matrix_);
            }
            else
            {
                err = this->matrix_->FSAI(power, NULL);
            }

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::FSAI() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                // Move to host
                bool is_accel = this->is_accel_();
                this->MoveToHost();

                // Convert to CSR
                unsigned int format   = this->GetFormat();
                int          blockdim = this->GetBlockDimension();
                this->ConvertToCSR();

                if(pattern != NULL)
                {
                    LocalMatrix<ValueType> pattern_host;
                    pattern_host.CopyFrom(*pattern);

                    if(this->matrix_->FSAI(power, pattern_host.matrix_) == false)
                    {
                        LOG_INFO("Computation of LocalMatrix::FSAI() failed");
                        this->Info();
                        FATAL_ERROR(__FILE__, __LINE__);
                    }
                }
                else
                {
                    if(this->matrix_->FSAI(power, NULL) == false)
                    {
                        LOG_INFO("Computation of LocalMatrix::FSAI() failed");
                        this->Info();
                        FATAL_ERROR(__FILE__, __LINE__);
                    }
                }

                if(format != CSR)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::FSAI() is performed in CSR format");

                    this->ConvertTo(format, blockdim);
                }

                if(is_accel == true)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::FSAI() is performed on the host");

                    this->MoveToAccelerator();
                }
            }
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::SPAI(void)
    {
        log_debug(this, "LocalMatrix::SPAI()");

        assert(this->GetM() == this->GetN());

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->SPAI();

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::SPAI() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                // Move to host
                bool is_accel = this->is_accel_();
                this->MoveToHost();

                // Convert to CSR
                unsigned int format   = this->GetFormat();
                int          blockdim = this->GetBlockDimension();
                this->ConvertToCSR();

                if(this->matrix_->SPAI() == false)
                {
                    LOG_INFO("Computation of LocalMatrix::SPAI() failed");
                    this->Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(format != CSR)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::SPAI() is performed in CSR format");

                    this->ConvertTo(format, blockdim);
                }

                if(is_accel == true)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::SPAI() is performed on the host");

                    this->MoveToAccelerator();
                }
            }
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::Invert(void)
    {
        log_debug(this, "LocalMatrix::Invert()");

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->Invert();

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == DENSE))
            {
                LOG_INFO("Computation of LocalMatrix::Invert() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                // Move to host
                bool is_accel = this->is_accel_();
                this->MoveToHost();

                // Convert to DENSE
                unsigned int format   = this->GetFormat();
                int          blockdim = this->GetBlockDimension();
                this->ConvertToDENSE();

                if(this->matrix_->Invert() == false)
                {
                    LOG_INFO("Computation of LocalMatrix::Invert() failed");
                    this->Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(format != DENSE)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::Invert() is performed in DENSE format");

                    this->ConvertTo(format, blockdim);
                }

                if(is_accel == true)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::Invert() is performed on the host");

                    this->MoveToAccelerator();
                }
            }
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ReplaceColumnVector(int idx, const LocalVector<ValueType>& vec)
    {
        log_debug(this, "LocalMatrix::ReplaceColumnVector()", idx, (const void*&)vec);

        assert(vec.GetSize() == this->GetM());
        assert(idx >= 0);

        assert(((this->matrix_ == this->matrix_host_) && (vec.vector_ == vec.vector_host_))
               || ((this->matrix_ == this->matrix_accel_) && (vec.vector_ == vec.vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->ReplaceColumnVector(idx, *vec.vector_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::ReplaceColumnVector() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalVector<ValueType> vec_host;
                vec_host.CopyFrom(vec);

                // Move to host
                this->MoveToHost();

                // try again
                err = this->matrix_->ReplaceColumnVector(idx, *vec_host.vector_);

                if(err == false)
                {
                    // Convert to CSR
                    unsigned int format   = this->GetFormat();
                    int          blockdim = this->GetBlockDimension();
                    this->ConvertToCSR();

                    if(this->matrix_->ReplaceColumnVector(idx, *vec_host.vector_) == false)
                    {
                        LOG_INFO("Computation of LocalMatrix::ReplaceColumnVector() failed");
                        this->Info();
                        FATAL_ERROR(__FILE__, __LINE__);
                    }

                    if(format != CSR)
                    {
                        LOG_VERBOSE_INFO(2,
                                         "*** warning: LocalMatrix::ReplaceColumnVector() is "
                                         "performed in CSR format");

                        this->ConvertTo(format, blockdim);
                    }
                }

                if(vec.is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2,
                        "*** warning: LocalMatrix::ReplaceColumnVector() is performed on the host");

                    this->MoveToAccelerator();
                }
            }
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ExtractColumnVector(int idx, LocalVector<ValueType>* vec) const
    {
        log_debug(this, "LocalMatrix::ExtractColumnVector()", idx, vec);

        assert(vec != NULL);
        assert(vec->GetSize() == this->GetM());
        assert(idx >= 0);

        assert(((this->matrix_ == this->matrix_host_) && (vec->vector_ == vec->vector_host_))
               || ((this->matrix_ == this->matrix_accel_) && (vec->vector_ == vec->vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->ExtractColumnVector(idx, vec->vector_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::ExtractColumnVector() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalMatrix<ValueType> mat_host;
                mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_host.CopyFrom(*this);

                // Move to host
                vec->MoveToHost();

                // Convert to CSR
                mat_host.ConvertToCSR();

                if(mat_host.matrix_->ExtractColumnVector(idx, vec->vector_) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::ExtractColumnVector() failed");
                    mat_host.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::ExtractColumnVector() is performed "
                                     "in CSR format");
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2,
                        "*** warning: LocalMatrix::ExtractColumnVector() is performed on the host");

                    vec->MoveToAccelerator();
                }
            }
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ReplaceRowVector(int idx, const LocalVector<ValueType>& vec)
    {
        log_debug(this, "LocalMatrix::ReplaceRowVector()", idx, (const void*&)vec);

        assert(vec.GetSize() == this->GetN());
        assert(idx >= 0);

        assert(((this->matrix_ == this->matrix_host_) && (vec.vector_ == vec.vector_host_))
               || ((this->matrix_ == this->matrix_accel_) && (vec.vector_ == vec.vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->ReplaceRowVector(idx, *vec.vector_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::ReplaceRowVector() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalVector<ValueType> vec_host;
                vec_host.CopyFrom(vec);

                // Move to host
                this->MoveToHost();

                // try again
                err = this->matrix_->ReplaceRowVector(idx, *vec_host.vector_);

                if(err == false)
                {
                    // Convert to CSR
                    unsigned int format   = this->GetFormat();
                    int          blockdim = this->GetBlockDimension();
                    this->ConvertToCSR();

                    if(this->matrix_->ReplaceRowVector(idx, *vec_host.vector_) == false)
                    {
                        LOG_INFO("Computation of LocalMatrix::ReplaceRowVector() failed");
                        this->Info();
                        FATAL_ERROR(__FILE__, __LINE__);
                    }

                    if(format != CSR)
                    {
                        LOG_VERBOSE_INFO(2,
                                         "*** warning: LocalMatrix::ReplaceRowVector() is "
                                         "performed in CSR format");

                        this->ConvertTo(format, blockdim);
                    }
                }

                if(vec.is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::ReplaceRowVector() is performed on the host");

                    this->MoveToAccelerator();
                }
            }
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::ExtractRowVector(int idx, LocalVector<ValueType>* vec) const
    {
        log_debug(this, "LocalMatrix::ExtractRowVector()", idx, vec);

        assert(vec != NULL);
        assert(vec->GetSize() == this->GetN());
        assert(idx >= 0);

        assert(((this->matrix_ == this->matrix_host_) && (vec->vector_ == vec->vector_host_))
               || ((this->matrix_ == this->matrix_accel_) && (vec->vector_ == vec->vector_accel_)));

#ifdef DEBUG_MODE
        this->Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->ExtractRowVector(idx, vec->vector_);

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::ExtractRowVector() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalMatrix<ValueType> mat_host;
                mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_host.CopyFrom(*this);

                // Move to host
                vec->MoveToHost();

                // Convert to CSR
                mat_host.ConvertToCSR();

                if(mat_host.matrix_->ExtractRowVector(idx, vec->vector_) == false)
                {
                    LOG_INFO("Computation of LocalMatrix::ExtractRowVector() failed");
                    mat_host.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2,
                        "*** warning: LocalMatrix::ExtractRowVector() is performed in CSR format");
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2, "*** warning: LocalMatrix::ExtractRowVector() is performed on the host");

                    vec->MoveToAccelerator();
                }
            }
        }
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::CompressAdd(const LocalVector<int64_t>&   l2g,
                                             const LocalVector<int64_t>&   global_ghost_col,
                                             const LocalMatrix<ValueType>& ext,
                                             LocalVector<int64_t>*         global_col)
    {
        log_debug(this,
                  "LocalMatrix::CompressAdd()",
                  (const void*&)l2g,
                  (const void*&)global_ghost_col,
                  (const void*&)ext,
                  global_col);

        assert(l2g.is_host_() == this->is_host_());
        assert(global_ghost_col.is_host_() == this->is_host_());
        assert(ext.is_host_() == this->is_host_());

        if(global_col != NULL)
        {
            assert(global_col->is_host_() == this->is_host_());
        }

#ifdef DEBUG_MODE
        ext.Check();
        this->Check();
#endif

        bool err = this->matrix_->CompressAdd(*l2g.vector_,
                                              *global_ghost_col.vector_,
                                              *ext.matrix_,
                                              (global_col != NULL ? global_col->vector_ : NULL));

        if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
        {
            LOG_INFO("Computation of LocalMatrix::CompressAdd() failed");
            this->Info();
            FATAL_ERROR(__FILE__, __LINE__);
        }

        if(err == false)
        {
            LocalVector<int64_t>   l2g_host;
            LocalVector<int64_t>   ggc_host;
            LocalMatrix<ValueType> ext_host;
            l2g_host.CopyFrom(l2g);
            ggc_host.CopyFrom(global_ghost_col);
            ext_host.ConvertTo(ext.GetFormat(), ext.GetBlockDimension());
            ext_host.CopyFrom(ext);

            this->MoveToHost();

            if(global_col != NULL)
            {
                global_col->MoveToHost();
            }

            if(this->matrix_->CompressAdd(*l2g_host.vector_,
                                          *ggc_host.vector_,
                                          *ext_host.matrix_,
                                          (global_col != NULL ? global_col->vector_ : NULL))
               == false)
            {
                LOG_INFO("Computation of LocalMatrix::CompressAdd() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(ext.GetFormat() != CSR)
            {
                LOG_VERBOSE_INFO(
                    2, "*** warning: LocalMatrix::CompressAdd() is performed in CSR format");

                this->ConvertTo(ext.GetFormat(), ext.GetBlockDimension());
            }

            if(ext.is_accel_() == true)
            {
                LOG_VERBOSE_INFO(
                    2, "*** warning: LocalMatrix::CompressAdd() is performed on the host");

                this->MoveToAccelerator();

                if(global_col != NULL)
                {
                    global_col->MoveToAccelerator();
                }
            }
        }

#ifdef DEBUG_MODE
        this->Check();
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::RSExtPIProlongNnz(int64_t                     global_column_begin,
                                                   int64_t                     global_column_end,
                                                   bool                        FF1,
                                                   const LocalVector<int64_t>& l2g,
                                                   const LocalVector<int>&     CFmap,
                                                   const LocalVector<bool>&    S,
                                                   const LocalMatrix<ValueType>& ghost,
                                                   const LocalVector<PtrType>&   bnd_csr_row_ptr,
                                                   const LocalVector<int64_t>&   bnd_csr_col_ind,
                                                   LocalVector<int>*             f2c,
                                                   LocalMatrix<ValueType>*       prolong_int,
                                                   LocalMatrix<ValueType>*       prolong_gst) const
    {
        log_debug(this,
                  "LocalMatrix::RSExtPIProlongNnz()",
                  global_column_begin,
                  global_column_end,
                  FF1,
                  (const void*&)l2g,
                  (const void*&)CFmap,
                  (const void*&)S,
                  (const void*&)ghost,
                  (const void*&)bnd_csr_row_ptr,
                  (const void*&)bnd_csr_col_ind,
                  f2c,
                  prolong_int,
                  prolong_gst);

        assert(f2c != NULL);
        assert(prolong_int != NULL);
        assert(prolong_int != prolong_gst);
        assert(this != prolong_int);
        assert(this != prolong_gst);

        assert(this->is_host_() == l2g.is_host_());
        assert(this->is_host_() == CFmap.is_host_());
        assert(this->is_host_() == S.is_host_());
        assert(this->is_host_() == ghost.is_host_());
        assert(this->is_host_() == bnd_csr_row_ptr.is_host_());
        assert(this->is_host_() == bnd_csr_col_ind.is_host_());
        assert(this->is_host_() == f2c->is_host_());
        assert(this->is_host_() == prolong_int->is_host_());
        assert(prolong_gst != NULL ? this->is_host_() == prolong_gst->is_host_() : true);

#ifdef DEBUG_MODE
        this->Check();
        ghost.Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->RSExtPIProlongNnz(
                global_column_begin,
                global_column_end,
                FF1,
                *l2g.vector_,
                *CFmap.vector_,
                *S.vector_,
                *ghost.matrix_,
                *bnd_csr_row_ptr.vector_,
                *bnd_csr_col_ind.vector_,
                f2c->vector_,
                prolong_int->matrix_,
                (prolong_gst != NULL ? prolong_gst->matrix_ : NULL));

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::RSExtPIProlongNnz() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalMatrix<ValueType> mat_host;
                LocalVector<int64_t>   l2g_host;
                LocalVector<int>       cf_host;
                LocalVector<bool>      S_host;
                LocalMatrix<ValueType> gst_host;
                LocalVector<PtrType>   row_host;
                LocalVector<int64_t>   col_host;
                mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                mat_host.CopyFrom(*this);
                l2g_host.CopyFrom(l2g);
                cf_host.CopyFrom(CFmap);
                S_host.CopyFrom(S);
                gst_host.CopyFrom(ghost);
                row_host.CopyFrom(bnd_csr_row_ptr);
                col_host.CopyFrom(bnd_csr_col_ind);

                // Move to host
                f2c->MoveToHost();
                prolong_int->MoveToHost();

                if(prolong_gst != NULL)
                {
                    prolong_gst->MoveToHost();
                }

                // Convert to CSR
                mat_host.ConvertToCSR();
                gst_host.ConvertToCSR();
                prolong_int->ConvertToCSR();

                if(prolong_gst != NULL)
                {
                    prolong_gst->ConvertToCSR();
                }

                if(mat_host.matrix_->RSExtPIProlongNnz(
                       global_column_begin,
                       global_column_end,
                       FF1,
                       *l2g_host.vector_,
                       *cf_host.vector_,
                       *S_host.vector_,
                       *gst_host.matrix_,
                       *row_host.vector_,
                       *col_host.vector_,
                       f2c->vector_,
                       prolong_int->matrix_,
                       (prolong_gst != NULL ? prolong_gst->matrix_ : NULL))
                   == false)
                {
                    LOG_INFO("Computation of LocalMatrix::RSExtPIProlongNnz() failed");
                    mat_host.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(
                        2,
                        "*** warning: LocalMatrix::RSExtPIProlongNnz() is performed in CSR format");

                    prolong_int->ConvertTo(this->GetFormat(), this->GetBlockDimension());

                    if(prolong_gst != NULL)
                    {
                        prolong_gst->ConvertTo(this->GetFormat(), this->GetBlockDimension());
                    }
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2,
                        "*** warning: LocalMatrix::RSExtPIProlongNnz() is performed on the host");

                    f2c->MoveToAccelerator();
                    prolong_int->MoveToAccelerator();

                    if(prolong_gst != NULL)
                    {
                        prolong_gst->MoveToAccelerator();
                    }
                }
            }
        }

#ifdef DEBUG_MODE
        prolong_int->Check();

        if(prolong_gst != NULL)
        {
            prolong_gst->Check();
        }
#endif
    }

    template <typename ValueType>
    void LocalMatrix<ValueType>::RSExtPIProlongFill(int64_t                     global_column_begin,
                                                    int64_t                     global_column_end,
                                                    bool                        FF1,
                                                    const LocalVector<int64_t>& l2g,
                                                    const LocalVector<int>&     f2c,
                                                    const LocalVector<int>&     CFmap,
                                                    const LocalVector<bool>&    S,
                                                    const LocalMatrix<ValueType>& ghost,
                                                    const LocalVector<PtrType>&   bnd_csr_row_ptr,
                                                    const LocalVector<int64_t>&   bnd_csr_col_ind,
                                                    const LocalVector<PtrType>&   ext_csr_row_ptr,
                                                    const LocalVector<int64_t>&   ext_csr_col_ind,
                                                    const LocalVector<ValueType>& ext_csr_val,
                                                    LocalMatrix<ValueType>*       prolong_int,
                                                    LocalMatrix<ValueType>*       prolong_gst,
                                                    LocalVector<int64_t>* global_ghost_col) const
    {
        log_debug(this,
                  "LocalMatrix::RSExtPIProlongFill()",
                  global_column_begin,
                  global_column_end,
                  FF1,
                  (const void*&)l2g,
                  (const void*&)f2c,
                  (const void*&)CFmap,
                  (const void*&)S,
                  (const void*&)ghost,
                  (const void*&)bnd_csr_row_ptr,
                  (const void*&)bnd_csr_col_ind,
                  (const void*&)ext_csr_row_ptr,
                  (const void*&)ext_csr_col_ind,
                  (const void*&)ext_csr_val,
                  prolong_int,
                  prolong_gst,
                  global_ghost_col);

        assert(global_column_begin >= 0);
        assert(global_column_end >= global_column_begin);
        assert(prolong_int != NULL);
        assert(prolong_int != prolong_gst);
        assert(this != prolong_int);
        assert(this != prolong_gst);

        assert(this->is_host_() == l2g.is_host_());
        assert(this->is_host_() == f2c.is_host_());
        assert(this->is_host_() == CFmap.is_host_());
        assert(this->is_host_() == S.is_host_());
        assert(this->is_host_() == ghost.is_host_());
        assert(this->is_host_() == bnd_csr_row_ptr.is_host_());
        assert(this->is_host_() == bnd_csr_col_ind.is_host_());
        assert(this->is_host_() == ext_csr_row_ptr.is_host_());
        assert(this->is_host_() == ext_csr_col_ind.is_host_());
        assert(this->is_host_() == ext_csr_val.is_host_());
        assert(this->is_host_() == prolong_int->is_host_());
        assert(prolong_gst != NULL ? this->is_host_() == prolong_gst->is_host_() : true);
        assert(global_ghost_col != NULL ? this->is_host_() == global_ghost_col->is_host_() : true);

#ifdef DEBUG_MODE
        this->Check();
        ghost.Check();
#endif

        if(this->GetNnz() > 0)
        {
            bool err = this->matrix_->RSExtPIProlongFill(
                global_column_begin,
                global_column_end,
                FF1,
                *l2g.vector_,
                *f2c.vector_,
                *CFmap.vector_,
                *S.vector_,
                *ghost.matrix_,
                *bnd_csr_row_ptr.vector_,
                *bnd_csr_col_ind.vector_,
                *ext_csr_row_ptr.vector_,
                *ext_csr_col_ind.vector_,
                *ext_csr_val.vector_,
                prolong_int->matrix_,
                (prolong_gst != NULL ? prolong_gst->matrix_ : NULL),
                (global_ghost_col != NULL ? global_ghost_col->vector_ : NULL));

            if((err == false) && (this->is_host_() == true) && (this->GetFormat() == CSR))
            {
                LOG_INFO("Computation of LocalMatrix::RSExtPIProlongFill() failed");
                this->Info();
                FATAL_ERROR(__FILE__, __LINE__);
            }

            if(err == false)
            {
                LocalMatrix<ValueType> mat_host;
                LocalVector<int64_t>   l2g_host;
                LocalVector<int>       f2c_host;
                LocalVector<int>       cf_host;
                LocalVector<bool>      S_host;
                LocalMatrix<ValueType> gst_host;
                LocalVector<PtrType>   bndr_host;
                LocalVector<int64_t>   bndc_host;
                LocalVector<PtrType>   extr_host;
                LocalVector<int64_t>   extc_host;
                LocalVector<ValueType> extv_host;
                mat_host.ConvertTo(this->GetFormat(), this->GetBlockDimension());
                gst_host.ConvertTo(ghost.GetFormat(), ghost.GetBlockDimension());
                mat_host.CopyFrom(*this);
                gst_host.CopyFrom(ghost);
                l2g_host.CopyFrom(l2g);
                f2c_host.CopyFrom(f2c);
                cf_host.CopyFrom(CFmap);
                S_host.CopyFrom(S);
                bndr_host.CopyFrom(bnd_csr_row_ptr);
                bndc_host.CopyFrom(bnd_csr_col_ind);
                extr_host.CopyFrom(ext_csr_row_ptr);
                extc_host.CopyFrom(ext_csr_col_ind);
                extv_host.CopyFrom(ext_csr_val);

                // Move to host
                prolong_int->MoveToHost();

                if(prolong_gst != NULL)
                {
                    prolong_gst->MoveToHost();
                }

                if(global_ghost_col != NULL)
                {
                    global_ghost_col->MoveToHost();
                }

                // Convert to CSR
                mat_host.ConvertToCSR();
                gst_host.ConvertToCSR();
                prolong_int->ConvertToCSR();

                if(prolong_gst != NULL)
                {
                    prolong_gst->ConvertToCSR();
                }

                if(mat_host.matrix_->RSExtPIProlongFill(
                       global_column_begin,
                       global_column_end,
                       FF1,
                       *l2g_host.vector_,
                       *f2c_host.vector_,
                       *cf_host.vector_,
                       *S_host.vector_,
                       *gst_host.matrix_,
                       *bndr_host.vector_,
                       *bndc_host.vector_,
                       *extr_host.vector_,
                       *extc_host.vector_,
                       *extv_host.vector_,
                       prolong_int->matrix_,
                       (prolong_gst != NULL ? prolong_gst->matrix_ : NULL),
                       (global_ghost_col != NULL ? global_ghost_col->vector_ : NULL))
                   == false)
                {
                    LOG_INFO("Computation of LocalMatrix::RSExtPIProlongFill() failed");
                    mat_host.Info();
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                if(this->GetFormat() != CSR)
                {
                    LOG_VERBOSE_INFO(2,
                                     "*** warning: LocalMatrix::RSExtPIProlongFill() is performed "
                                     "in CSR format");

                    prolong_int->ConvertTo(this->GetFormat(), this->GetBlockDimension());

                    if(prolong_gst != NULL)
                    {
                        prolong_gst->ConvertTo(this->GetFormat(), this->GetBlockDimension());
                    }
                }

                if(this->is_accel_() == true)
                {
                    LOG_VERBOSE_INFO(
                        2,
                        "*** warning: LocalMatrix::RSExtPIProlongFill() is performed on the host");

                    prolong_int->MoveToAccelerator();

                    if(prolong_gst != NULL)
                    {
                        prolong_gst->MoveToAccelerator();
                    }

                    if(global_ghost_col != NULL)
                    {
                        global_ghost_col->MoveToAccelerator();
                    }
                }
            }
        }

#ifdef DEBUG_MODE
        prolong_int->Check();

        if(prolong_gst != NULL)
        {
            prolong_gst->Check();
        }
#endif
    }

    template class LocalMatrix<float>;
    template class LocalMatrix<double>;
#ifdef SUPPORT_COMPLEX
    template class LocalMatrix<std::complex<float>>;
    template class LocalMatrix<std::complex<double>>;
#endif

} // namespace rocalution

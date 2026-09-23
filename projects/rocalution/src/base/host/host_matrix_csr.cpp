/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "host_matrix_csr.hpp"
#include "../../utils/allocate_free.hpp"
#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "../../utils/math_functions.hpp"
#include "../matrix_formats_ind.hpp"
#include "host_conversion.hpp"
#include "host_io.hpp"
#include "host_matrix_bcsr.hpp"
#include "host_matrix_coo.hpp"
#include "host_matrix_dense.hpp"
#include "host_matrix_dia.hpp"
#include "host_matrix_ell.hpp"
#include "host_matrix_hyb.hpp"
#include "host_matrix_mcsr.hpp"
#include "host_sparse.hpp"
#include "host_vector.hpp"
#include "rocalution/utils/types.hpp"

#include "host_ilut_driver_csr.hpp"

#include <algorithm>
#include <complex>
#include <limits>
#include <map>
#include <math.h>
#include <numeric>
#include <string.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_set_num_threads(num) ;
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#define omp_set_nested(num) ;
#endif

namespace rocalution
{

    // LCOV_EXCL_START
    template <typename ValueType>
    HostMatrixCSR<ValueType>::HostMatrixCSR()
    {
        // no default constructors
        LOG_INFO("no default constructor");
        FATAL_ERROR(__FILE__, __LINE__);
    }
    // LCOV_EXCL_STOP

    template <typename ValueType>
    HostMatrixCSR<ValueType>::HostMatrixCSR(const Rocalution_Backend_Descriptor& local_backend)
    {
        log_debug(this, "HostMatrixCSR::HostMatrixCSR()", "constructor with local_backend");

        this->mat_.row_offset = NULL;
        this->mat_.col        = NULL;
        this->mat_.val        = NULL;
        this->set_backend(local_backend);

        this->L_diag_unit_ = false;
        this->U_diag_unit_ = false;

        this->mat_buffer_size_ = 0;
        this->mat_buffer_      = NULL;

        this->tmp_vec_ = NULL;
    }

    template <typename ValueType>
    HostMatrixCSR<ValueType>::~HostMatrixCSR()
    {
        log_debug(this, "HostMatrixCSR::~HostMatrixCSR()", "destructor");

        this->Clear();
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::Clear(void)
    {
        free_host(&this->mat_.row_offset);
        free_host(&this->mat_.col);
        free_host(&this->mat_.val);

        this->nrow_ = 0;
        this->ncol_ = 0;
        this->nnz_  = 0;

        this->ItLAnalyseClear();
        this->ItUAnalyseClear();
        this->ItLUAnalyseClear();
        this->ItLLAnalyseClear();
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::Zeros(void)
    {
        set_to_zero_host(this->nnz_, mat_.val);

        return true;
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::Info(void) const
    {
        LOG_INFO(
            "HostMatrixCSR<ValueType>, OpenMP threads: " << this->local_backend_.OpenMP_threads);
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::Check(void) const
    {
        bool sorted = true;

        if(this->nnz_ > 0)
        {
            assert(this->nrow_ > 0);
            assert(this->ncol_ > 0);

            assert(this->mat_.row_offset != NULL);
            assert(this->mat_.val != NULL);
            assert(this->mat_.col != NULL);

            // check nnz
            if((std::abs(this->nnz_) == std::numeric_limits<int64_t>::infinity()) || // inf
               (this->nnz_ != this->nnz_))
            { // NaN
                // LCOV_EXCL_START
                LOG_VERBOSE_INFO(2, "*** error: Matrix CSR:Check - problems with matrix nnz");
                return false;
                // LCOV_EXCL_STOP
            }

            // nrow
            if((std::abs(this->nrow_) == std::numeric_limits<int>::infinity()) || // inf
               (this->nrow_ != this->nrow_))
            { // NaN
                // LCOV_EXCL_START
                LOG_VERBOSE_INFO(2, "*** error: Matrix CSR:Check - problems with matrix nrow");
                return false;
                // LCOV_EXCL_STOP
            }

            // ncol
            if((std::abs(this->ncol_) == std::numeric_limits<int>::infinity()) || // inf
               (this->ncol_ != this->ncol_))
            { // NaN
                // LCOV_EXCL_START
                LOG_VERBOSE_INFO(2, "*** error: Matrix CSR:Check - problems with matrix ncol");
                return false;
                // LCOV_EXCL_STOP
            }

            for(int ai = 0; ai < this->nrow_ + 1; ++ai)
            {
                PtrType row = this->mat_.row_offset[ai];
                if((row < 0) || (row > this->nnz_))
                {
                    // LCOV_EXCL_START
                    LOG_VERBOSE_INFO(
                        2,
                        "*** error: Matrix CSR:Check - problems with matrix row offset pointers");
                    return false;
                    // LCOV_EXCL_STOP
                }
            }

            for(int ai = 0; ai < this->nrow_; ++ai)
            {
                int s = this->mat_.col[this->mat_.row_offset[ai]];

                for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1];
                    ++aj)
                {
                    int col      = this->mat_.col[aj];
                    int prev_col = (aj > this->mat_.row_offset[ai]) ? this->mat_.col[aj - 1] : -1;

                    if((col < 0) || (col > this->ncol_))
                    {
                        // LCOV_EXCL_START
                        LOG_VERBOSE_INFO(
                            2, "*** error: Matrix CSR:Check - problems with matrix col values");
                        return false;
                        // LCOV_EXCL_STOP
                    }

                    if(col == prev_col)
                    {
                        // LCOV_EXCL_START
                        LOG_VERBOSE_INFO(2,
                                         "*** error: Matrix CSR:Check - problems with matrix col "
                                         "values - the matrix has duplicated column entries");
                        return false;
                        // LCOV_EXCL_STOP
                    }

                    ValueType val = this->mat_.val[aj];
                    if((val == std::numeric_limits<ValueType>::infinity()) || (val != val))
                    {
                        // LCOV_EXCL_START
                        LOG_VERBOSE_INFO(
                            2, "*** error: Matrix CSR:Check - problems with matrix values");
                        return false;
                        // LCOV_EXCL_STOP
                    }

                    if((aj > this->mat_.row_offset[ai]) && (s >= col))
                    {
                        sorted = false;
                    }

                    s = this->mat_.col[aj];
                }
            }
        }
        else
        {
            assert(this->nnz_ == 0);
            assert(this->nrow_ >= 0);
            assert(this->ncol_ >= 0);

            if(this->nrow_ == 0 && this->ncol_ == 0)
            {
                assert(this->mat_.val == NULL);
                assert(this->mat_.col == NULL);
            }
        }

        if(sorted == false)
        {
            LOG_VERBOSE_INFO(2,
                             "*** warning: Matrix CSR:Check - the matrix has not sorted columns");
        }

        return true;
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::AllocateCSR(int64_t nnz, int nrow, int ncol)
    {
        assert(nnz >= 0);
        assert(ncol >= 0);
        assert(nrow >= 0);

        this->Clear();

        allocate_host(nrow + 1, &this->mat_.row_offset);
        allocate_host(nnz, &this->mat_.col);
        allocate_host(nnz, &this->mat_.val);

        set_to_zero_host(nrow + 1, mat_.row_offset);
        set_to_zero_host(nnz, mat_.col);
        set_to_zero_host(nnz, mat_.val);

        this->nrow_ = nrow;
        this->ncol_ = ncol;
        this->nnz_  = nnz;
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::SetDataPtrCSR(
        PtrType** row_offset, int** col, ValueType** val, int64_t nnz, int nrow, int ncol)
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

        this->mat_.row_offset = *row_offset;
        this->mat_.col        = *col;
        this->mat_.val        = *val;
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::LeaveDataPtrCSR(PtrType** row_offset, int** col, ValueType** val)
    {
        assert(this->nrow_ >= 0);
        assert(this->ncol_ >= 0);
        assert(this->nnz_ >= 0);

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
    void HostMatrixCSR<ValueType>::CopyFromCSR(const PtrType*   row_offsets,
                                               const int*       col,
                                               const ValueType* val)
    {
        assert(row_offsets != NULL);

        copy_h2h(this->nrow_ + 1, row_offsets, this->mat_.row_offset);

        if(this->nnz_ > 0)
        {
            assert(this->nrow_ > 0);
            assert(this->ncol_ > 0);
            assert(col != NULL);
            assert(val != NULL);

            copy_h2h(this->nnz_, col, this->mat_.col);
            copy_h2h(this->nnz_, val, this->mat_.val);
        }
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::CopyToCSR(PtrType* row_offsets, int* col, ValueType* val) const
    {
        assert(row_offsets != NULL);

        copy_h2h(this->nrow_ + 1, this->mat_.row_offset, row_offsets);

        if(this->nnz_ > 0)
        {
            assert(this->nrow_ > 0);
            assert(this->ncol_ > 0);
            assert(col != NULL);
            assert(val != NULL);

            copy_h2h(this->nnz_, this->mat_.col, col);
            copy_h2h(this->nnz_, this->mat_.val, val);
        }
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::CopyFrom(const BaseMatrix<ValueType>& mat)
    {
        // copy only in the same format
        assert(this->GetMatFormat() == mat.GetMatFormat());

        if(const HostMatrixCSR<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixCSR<ValueType>*>(&mat))
        {
            if(this->nnz_ == 0)
            {
                this->AllocateCSR(cast_mat->nnz_, cast_mat->nrow_, cast_mat->ncol_);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

            // Copy only if initialized
            if(cast_mat->mat_.row_offset != NULL)
            {
                copy_h2h(this->nrow_ + 1, cast_mat->mat_.row_offset, this->mat_.row_offset);
            }

            copy_h2h(this->nnz_, cast_mat->mat_.col, this->mat_.col);
            copy_h2h(this->nnz_, cast_mat->mat_.val, this->mat_.val);
        }
        else
        {
            // Host matrix knows only host matrices
            // -> dispatching
            mat.CopyTo(this);
        }
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::CopyTo(BaseMatrix<ValueType>* mat) const
    {
        mat->CopyFrom(*this);
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ReadFileRSIO(const std::string& filename)
    {
        int64_t nrow;
        int64_t ncol;
        int64_t nnz;

        PtrType*   ptr = NULL;
        int*       col = NULL;
        ValueType* val = NULL;

        if(read_matrix_csr_rocsparseio(nrow, ncol, nnz, &ptr, &col, &val, filename.c_str()) != true)
        {
            return false;
        }

        // Number of rows and columns are expected to be within 32 bits locally
        assert(nrow <= std::numeric_limits<int>::max());
        assert(ncol <= std::numeric_limits<int>::max());

        this->Clear();
        this->SetDataPtrCSR(&ptr, &col, &val, nnz, static_cast<int>(nrow), static_cast<int>(ncol));

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ReadFileCSR(const std::string& filename)
    {
        int64_t nrow;
        int64_t ncol;
        int64_t nnz;

        PtrType*   ptr = NULL;
        int*       col = NULL;
        ValueType* val = NULL;

        if(read_matrix_csr(nrow, ncol, nnz, &ptr, &col, &val, filename.c_str()) != true)
        {
            return false;
        }

        // Number of rows and columns are expected to be within 32 bits locally
        assert(nrow <= std::numeric_limits<int>::max());
        assert(ncol <= std::numeric_limits<int>::max());

        this->Clear();
        this->SetDataPtrCSR(&ptr, &col, &val, nnz, static_cast<int>(nrow), static_cast<int>(ncol));

        return true;
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::CopyFromHostCSR(const PtrType*   row_offset,
                                                   const int*       col,
                                                   const ValueType* val,
                                                   int64_t          nnz,
                                                   int              nrow,
                                                   int              ncol)
    {
        assert(nnz >= 0);
        assert(ncol >= 0);
        assert(nrow >= 0);
        assert(row_offset != NULL);

        // Allocate matrix
        this->Clear();

        this->nrow_ = nrow;
        this->ncol_ = ncol;
        this->nnz_  = nnz;

        allocate_host(nrow + 1, &this->mat_.row_offset);

        copy_h2h(this->nrow_ + 1, row_offset, this->mat_.row_offset);

        if(nnz > 0)
        {
            assert(col != NULL);
            assert(val != NULL);
        }

        allocate_host(nnz, &this->mat_.col);
        allocate_host(nnz, &this->mat_.val);

        copy_h2h(this->nnz_, col, this->mat_.col);
        copy_h2h(this->nnz_, val, this->mat_.val);
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::WriteFileRSIO(const std::string& filename) const
    {
        return write_matrix_csr_rocsparseio(this->nrow_,
                                            this->ncol_,
                                            this->nnz_,
                                            this->mat_.row_offset,
                                            this->mat_.col,
                                            this->mat_.val,
                                            filename.c_str());
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::WriteFileCSR(const std::string& filename) const
    {
        if(write_matrix_csr(this->nrow_,
                            this->ncol_,
                            this->nnz_,
                            this->mat_.row_offset,
                            this->mat_.col,
                            this->mat_.val,
                            filename.c_str())
           != true)
        {
            return false;
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ConvertFrom(const BaseMatrix<ValueType>& mat)
    {
        this->Clear();

        // Empty matrix
        if(mat.GetNnz() == 0)
        {
            this->AllocateCSR(mat.GetNnz(), mat.GetM(), mat.GetN());

            return true;
        }

        if(const HostMatrixCSR<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixCSR<ValueType>*>(&mat))
        {
            this->CopyFrom(*cast_mat);
            return true;
        }

        if(const HostMatrixBCSR<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixBCSR<ValueType>*>(&mat))
        {
            this->Clear();

            int     nrow = cast_mat->mat_.nrowb * cast_mat->mat_.blockdim;
            int     ncol = cast_mat->mat_.ncolb * cast_mat->mat_.blockdim;
            int64_t nnz  = cast_mat->mat_.nnzb * cast_mat->mat_.blockdim * cast_mat->mat_.blockdim;

            if(bcsr_to_csr(this->local_backend_.OpenMP_threads,
                           nnz,
                           nrow,
                           ncol,
                           cast_mat->mat_,
                           &this->mat_)
               == true)
            {
                this->nrow_ = nrow;
                this->ncol_ = ncol;
                this->nnz_  = nnz;

                return true;
            }
        }

        if(const HostMatrixCOO<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixCOO<ValueType>*>(&mat))
        {
            this->Clear();

            if(coo_to_csr(this->local_backend_.OpenMP_threads,
                          cast_mat->nnz_,
                          cast_mat->nrow_,
                          cast_mat->ncol_,
                          cast_mat->mat_,
                          &this->mat_)
               == true)
            {
                this->nrow_ = cast_mat->nrow_;
                this->ncol_ = cast_mat->ncol_;
                this->nnz_  = cast_mat->nnz_;

                return true;
            }
        }

        if(const HostMatrixDENSE<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixDENSE<ValueType>*>(&mat))
        {
            this->Clear();
            int64_t nnz = 0;

            if(dense_to_csr(this->local_backend_.OpenMP_threads,
                            cast_mat->nrow_,
                            cast_mat->ncol_,
                            cast_mat->mat_,
                            &this->mat_,
                            &nnz)
               == true)
            {
                this->nrow_ = cast_mat->nrow_;
                this->ncol_ = cast_mat->ncol_;
                this->nnz_  = nnz;

                return true;
            }
        }

        if(const HostMatrixDIA<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixDIA<ValueType>*>(&mat))
        {
            this->Clear();
            int64_t nnz;

            if(dia_to_csr(this->local_backend_.OpenMP_threads,
                          cast_mat->nnz_,
                          cast_mat->nrow_,
                          cast_mat->ncol_,
                          cast_mat->mat_,
                          &this->mat_,
                          &nnz)
               == true)
            {
                this->nrow_ = cast_mat->nrow_;
                this->ncol_ = cast_mat->ncol_;
                this->nnz_  = nnz;

                return true;
            }
        }

        if(const HostMatrixELL<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixELL<ValueType>*>(&mat))
        {
            this->Clear();
            int64_t nnz;

            if(ell_to_csr(this->local_backend_.OpenMP_threads,
                          cast_mat->nnz_,
                          cast_mat->nrow_,
                          cast_mat->ncol_,
                          cast_mat->mat_,
                          &this->mat_,
                          &nnz)
               == true)
            {
                this->nrow_ = cast_mat->nrow_;
                this->ncol_ = cast_mat->ncol_;
                this->nnz_  = nnz;

                return true;
            }
        }

        if(const HostMatrixMCSR<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixMCSR<ValueType>*>(&mat))
        {
            this->Clear();

            if(mcsr_to_csr(this->local_backend_.OpenMP_threads,
                           cast_mat->nnz_,
                           cast_mat->nrow_,
                           cast_mat->ncol_,
                           cast_mat->mat_,
                           &this->mat_)
               == true)
            {
                this->nrow_ = cast_mat->nrow_;
                this->ncol_ = cast_mat->ncol_;
                this->nnz_  = cast_mat->nnz_;

                return true;
            }
        }

        if(const HostMatrixHYB<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixHYB<ValueType>*>(&mat))
        {
            this->Clear();
            int64_t nnz;

            if(hyb_to_csr(this->local_backend_.OpenMP_threads,
                          cast_mat->nnz_,
                          cast_mat->nrow_,
                          cast_mat->ncol_,
                          cast_mat->ell_nnz_,
                          cast_mat->coo_nnz_,
                          cast_mat->mat_,
                          &this->mat_,
                          &nnz)
               == true)
            {
                this->nrow_ = cast_mat->nrow_;
                this->ncol_ = cast_mat->ncol_;
                this->nnz_  = nnz;

                return true;
            }
        }

        return false;
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::Apply(const BaseVector<ValueType>& in,
                                         BaseVector<ValueType>*       out) const
    {
        assert(in.GetSize() >= 0);
        assert(out->GetSize() >= 0);
        assert(in.GetSize() == this->ncol_);
        assert(out->GetSize() == this->nrow_);

        const HostVector<ValueType>* cast_in  = dynamic_cast<const HostVector<ValueType>*>(&in);
        HostVector<ValueType>*       cast_out = dynamic_cast<HostVector<ValueType>*>(out);

        assert(cast_in != NULL);
        assert(cast_out != NULL);

        _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            ValueType sum     = static_cast<ValueType>(0);
            PtrType   row_beg = this->mat_.row_offset[ai];
            PtrType   row_end = this->mat_.row_offset[ai + 1];

            for(PtrType aj = row_beg; aj < row_end; ++aj)
            {
                sum += this->mat_.val[aj] * cast_in->vec_[this->mat_.col[aj]];
            }

            cast_out->vec_[ai] = sum;
        }
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::ApplyAdd(const BaseVector<ValueType>& in,
                                            ValueType                    scalar,
                                            BaseVector<ValueType>*       out) const
    {
        if(this->nnz_ > 0)
        {
            assert(in.GetSize() >= 0);
            assert(out->GetSize() >= 0);
            assert(in.GetSize() == this->ncol_);
            assert(out->GetSize() == this->nrow_);

            const HostVector<ValueType>* cast_in  = dynamic_cast<const HostVector<ValueType>*>(&in);
            HostVector<ValueType>*       cast_out = dynamic_cast<HostVector<ValueType>*>(out);

            assert(cast_in != NULL);
            assert(cast_out != NULL);

            _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int ai = 0; ai < this->nrow_; ++ai)
            {
                for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1];
                    ++aj)
                {
                    cast_out->vec_[ai]
                        += scalar * this->mat_.val[aj] * cast_in->vec_[this->mat_.col[aj]];
                }
            }
        }
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ExtractDiagonal(BaseVector<ValueType>* vec_diag) const
    {
        assert(vec_diag != NULL);
        assert(vec_diag->GetSize() >= this->nrow_);

        HostVector<ValueType>* cast_vec_diag = dynamic_cast<HostVector<ValueType>*>(vec_diag);

        _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(ai == this->mat_.col[aj])
                {
                    cast_vec_diag->vec_[ai] = this->mat_.val[aj];
                    break;
                }
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ExtractInverseDiagonal(BaseVector<ValueType>* vec_inv_diag) const
    {
        assert(vec_inv_diag != NULL);
        assert(vec_inv_diag->GetSize() == this->nrow_);

        HostVector<ValueType>* cast_vec_inv_diag
            = dynamic_cast<HostVector<ValueType>*>(vec_inv_diag);

        int detect_zero_diag = 0;

        _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(ai == this->mat_.col[aj])
                {
                    if(this->mat_.val[aj] != static_cast<ValueType>(0))
                    {
                        cast_vec_inv_diag->vec_[ai]
                            = static_cast<ValueType>(1) / this->mat_.val[aj];
                    }
                    else
                    {
                        cast_vec_inv_diag->vec_[ai] = static_cast<ValueType>(1);
                        detect_zero_diag            = 1;
                    }

                    break;
                }
            }
        }

        if(detect_zero_diag == 1)
        {
            LOG_VERBOSE_INFO(
                2,
                "*** warning: in HostMatrixCSR::ExtractInverseDiagonal() a zero has been detected "
                "on the diagonal. It has been replaced with one to avoid inf");
        }
        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ExtractSubMatrix(int                    row_offset,
                                                    int                    col_offset,
                                                    int                    row_size,
                                                    int                    col_size,
                                                    BaseMatrix<ValueType>* mat) const
    {
        assert(mat != NULL);

        assert(row_offset >= 0);
        assert(col_offset >= 0);

        assert(this->nrow_ >= 0);
        assert(this->ncol_ >= 0);

        HostMatrixCSR<ValueType>* cast_mat = dynamic_cast<HostMatrixCSR<ValueType>*>(mat);
        assert(cast_mat != NULL);

        int64_t mat_nnz = 0;

        // use omp in local_matrix (higher level)

        //  _set_omp_backend_threads(this->local_backend_, this->nrow_);

        //#ifdef _OPENMP
        //#pragma omp parallel for reduction(+:mat_nnz)
        //#endif
        for(int ai = row_offset; ai < row_offset + row_size; ++ai)
        {
            for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if((this->mat_.col[aj] >= col_offset)
                   && (this->mat_.col[aj] < col_offset + col_size))
                {
                    ++mat_nnz;
                }
            }
        }

        cast_mat->AllocateCSR(mat_nnz, row_size, col_size);

        // not empty submatrix
        if(mat_nnz > 0)
        {
            PtrType mat_row_offset       = 0;
            cast_mat->mat_.row_offset[0] = mat_row_offset;

            for(int ai = row_offset; ai < row_offset + row_size; ++ai)
            {
                for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1];
                    ++aj)
                {
                    if((this->mat_.col[aj] >= col_offset)
                       && (this->mat_.col[aj] < col_offset + col_size))
                    {
                        cast_mat->mat_.col[mat_row_offset] = this->mat_.col[aj] - col_offset;
                        cast_mat->mat_.val[mat_row_offset] = this->mat_.val[aj];
                        ++mat_row_offset;
                    }
                }

                cast_mat->mat_.row_offset[ai - row_offset + 1] = mat_row_offset;
            }

            cast_mat->mat_.row_offset[row_size] = mat_row_offset;
            assert(mat_row_offset == mat_nnz);
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ExtractU(BaseMatrix<ValueType>* U) const
    {
        assert(U != NULL);

        assert(this->nrow_ > 0);
        assert(this->ncol_ > 0);

        HostMatrixCSR<ValueType>* cast_U = dynamic_cast<HostMatrixCSR<ValueType>*>(U);

        assert(cast_U != NULL);

        // count nnz of upper triangular part
        int64_t nnz_U = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : nnz_U)
#endif
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(this->mat_.col[aj] > ai)
                {
                    ++nnz_U;
                }
            }
        }

        // allocate upper triangular part structure
        PtrType*   row_offset = NULL;
        int*       col        = NULL;
        ValueType* val        = NULL;

        allocate_host(this->nrow_ + 1, &row_offset);
        allocate_host(nnz_U, &col);
        allocate_host(nnz_U, &val);

        // fill upper triangular part
        PtrType nnz   = 0;
        row_offset[0] = 0;
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(this->mat_.col[aj] > ai)
                {
                    col[nnz] = this->mat_.col[aj];
                    val[nnz] = this->mat_.val[aj];
                    ++nnz;
                }
            }

            row_offset[ai + 1] = nnz;
        }

        cast_U->Clear();
        cast_U->SetDataPtrCSR(&row_offset, &col, &val, nnz_U, this->nrow_, this->ncol_);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ExtractUDiagonal(BaseMatrix<ValueType>* U) const
    {
        assert(U != NULL);

        assert(this->nrow_ > 0);
        assert(this->ncol_ > 0);

        HostMatrixCSR<ValueType>* cast_U = dynamic_cast<HostMatrixCSR<ValueType>*>(U);

        assert(cast_U != NULL);

        // count nnz of upper triangular part
        int64_t nnz_U = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : nnz_U)
#endif
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(this->mat_.col[aj] >= ai)
                {
                    ++nnz_U;
                }
            }
        }

        // allocate upper triangular part structure
        PtrType*   row_offset = NULL;
        int*       col        = NULL;
        ValueType* val        = NULL;

        allocate_host(this->nrow_ + 1, &row_offset);
        allocate_host(nnz_U, &col);
        allocate_host(nnz_U, &val);

        // fill upper triangular part
        PtrType nnz   = 0;
        row_offset[0] = 0;
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(this->mat_.col[aj] >= ai)
                {
                    col[nnz] = this->mat_.col[aj];
                    val[nnz] = this->mat_.val[aj];
                    ++nnz;
                }
            }

            row_offset[ai + 1] = nnz;
        }

        cast_U->Clear();
        cast_U->SetDataPtrCSR(&row_offset, &col, &val, nnz_U, this->nrow_, this->ncol_);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ExtractL(BaseMatrix<ValueType>* L) const
    {
        assert(L != NULL);

        assert(this->nrow_ > 0);
        assert(this->ncol_ > 0);

        HostMatrixCSR<ValueType>* cast_L = dynamic_cast<HostMatrixCSR<ValueType>*>(L);

        assert(cast_L != NULL);

        // count nnz of lower triangular part
        int64_t nnz_L = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : nnz_L)
#endif
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(this->mat_.col[aj] < ai)
                {
                    ++nnz_L;
                }
            }
        }

        // allocate lower triangular part structure
        PtrType*   row_offset = NULL;
        int*       col        = NULL;
        ValueType* val        = NULL;

        allocate_host(this->nrow_ + 1, &row_offset);
        allocate_host(nnz_L, &col);
        allocate_host(nnz_L, &val);

        // fill lower triangular part
        PtrType nnz   = 0;
        row_offset[0] = 0;
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(this->mat_.col[aj] < ai)
                {
                    col[nnz] = this->mat_.col[aj];
                    val[nnz] = this->mat_.val[aj];
                    ++nnz;
                }
            }

            row_offset[ai + 1] = nnz;
        }

        cast_L->Clear();
        cast_L->SetDataPtrCSR(&row_offset, &col, &val, nnz_L, this->nrow_, this->ncol_);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ExtractLDiagonal(BaseMatrix<ValueType>* L) const
    {
        assert(L != NULL);

        assert(this->nrow_ > 0);
        assert(this->ncol_ > 0);

        HostMatrixCSR<ValueType>* cast_L = dynamic_cast<HostMatrixCSR<ValueType>*>(L);

        assert(cast_L != NULL);

        // count nnz of lower triangular part
        int64_t nnz_L = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : nnz_L)
#endif
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(this->mat_.col[aj] <= ai)
                {
                    ++nnz_L;
                }
            }
        }

        // allocate lower triangular part structure
        PtrType*   row_offset = NULL;
        int*       col        = NULL;
        ValueType* val        = NULL;

        allocate_host(this->nrow_ + 1, &row_offset);
        allocate_host(nnz_L, &col);
        allocate_host(nnz_L, &val);

        // fill lower triangular part
        PtrType nnz   = 0;
        row_offset[0] = 0;
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(this->mat_.col[aj] <= ai)
                {
                    col[nnz] = this->mat_.col[aj];
                    val[nnz] = this->mat_.val[aj];
                    ++nnz;
                }
            }

            row_offset[ai + 1] = nnz;
        }

        cast_L->Clear();
        cast_L->SetDataPtrCSR(&row_offset, &col, &val, nnz_L, this->nrow_, this->ncol_);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::LUSolve(const BaseVector<ValueType>& in,
                                           BaseVector<ValueType>*       out) const
    {
        assert(in.GetSize() >= 0);
        assert(out->GetSize() >= 0);
        assert(in.GetSize() == this->ncol_);
        assert(out->GetSize() == this->nrow_);

        const HostVector<ValueType>* cast_in  = dynamic_cast<const HostVector<ValueType>*>(&in);
        HostVector<ValueType>*       cast_out = dynamic_cast<HostVector<ValueType>*>(out);

        assert(cast_in != NULL);
        assert(cast_out != NULL);

        // Solve L
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            cast_out->vec_[ai] = cast_in->vec_[ai];

            for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(this->mat_.col[aj] < ai)
                {
                    // under the diagonal
                    cast_out->vec_[ai] -= this->mat_.val[aj] * cast_out->vec_[this->mat_.col[aj]];
                }
                else
                {
                    // CSR should be sorted
                    break;
                }
            }
        }

        // last elements should be the diagonal one (last)
        int64_t diag_aj = this->nnz_ - 1;

        // Solve U
        for(int ai = this->nrow_ - 1; ai >= 0; --ai)
        {
            for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(this->mat_.col[aj] > ai)
                {
                    // above the diagonal
                    cast_out->vec_[ai] -= this->mat_.val[aj] * cast_out->vec_[this->mat_.col[aj]];
                }

                if(this->mat_.col[aj] == ai)
                {
                    diag_aj = aj;
                }
            }

            cast_out->vec_[ai] /= this->mat_.val[diag_aj];
        }

        return true;
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::LLAnalyse(void)
    {
        // do nothing
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::LLAnalyseClear(void)
    {
        // do nothing
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::LUAnalyse(void)
    {
        // do nothing
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::LUAnalyseClear(void)
    {
        // do nothing
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::LLSolve(const BaseVector<ValueType>& in,
                                           BaseVector<ValueType>*       out) const
    {
        assert(in.GetSize() >= 0);
        assert(out->GetSize() >= 0);
        assert(in.GetSize() == this->ncol_);
        assert(out->GetSize() == this->nrow_);

        const HostVector<ValueType>* cast_in  = dynamic_cast<const HostVector<ValueType>*>(&in);
        HostVector<ValueType>*       cast_out = dynamic_cast<HostVector<ValueType>*>(out);

        assert(cast_in != NULL);
        assert(cast_out != NULL);

        // Solve L
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            ValueType value    = cast_in->vec_[ai];
            PtrType   diag_idx = this->mat_.row_offset[ai + 1] - 1;

            for(PtrType aj = this->mat_.row_offset[ai]; aj < diag_idx; ++aj)
            {
                value -= this->mat_.val[aj] * cast_out->vec_[this->mat_.col[aj]];
            }

            cast_out->vec_[ai] = value / this->mat_.val[diag_idx];
        }

        // Solve L^T
        for(int ai = this->nrow_ - 1; ai >= 0; --ai)
        {
            PtrType   diag_idx = this->mat_.row_offset[ai + 1] - 1;
            ValueType value    = cast_out->vec_[ai] / this->mat_.val[diag_idx];

            for(PtrType aj = this->mat_.row_offset[ai]; aj < diag_idx; ++aj)
            {
                cast_out->vec_[this->mat_.col[aj]] -= value * this->mat_.val[aj];
            }

            cast_out->vec_[ai] = value;
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::LLSolve(const BaseVector<ValueType>& in,
                                           const BaseVector<ValueType>& inv_diag,
                                           BaseVector<ValueType>*       out) const
    {
        assert(in.GetSize() >= 0);
        assert(out->GetSize() >= 0);
        assert(in.GetSize() == this->ncol_);
        assert(out->GetSize() == this->nrow_);
        assert(inv_diag.GetSize() == this->nrow_ || inv_diag.GetSize() == this->ncol_);

        const HostVector<ValueType>* cast_in = dynamic_cast<const HostVector<ValueType>*>(&in);
        const HostVector<ValueType>* cast_diag
            = dynamic_cast<const HostVector<ValueType>*>(&inv_diag);
        HostVector<ValueType>* cast_out = dynamic_cast<HostVector<ValueType>*>(out);

        assert(cast_in != NULL);
        assert(cast_out != NULL);

        // Solve L
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            ValueType value    = cast_in->vec_[ai];
            PtrType   diag_idx = this->mat_.row_offset[ai + 1] - 1;

            for(PtrType aj = this->mat_.row_offset[ai]; aj < diag_idx; ++aj)
            {
                value -= this->mat_.val[aj] * cast_out->vec_[this->mat_.col[aj]];
            }

            cast_out->vec_[ai] = value * cast_diag->vec_[ai];
        }

        // Solve L^T
        for(int ai = this->nrow_ - 1; ai >= 0; --ai)
        {
            PtrType   diag_idx = this->mat_.row_offset[ai + 1] - 1;
            ValueType value    = cast_out->vec_[ai] * cast_diag->vec_[ai];

            for(PtrType aj = this->mat_.row_offset[ai]; aj < diag_idx; ++aj)
            {
                cast_out->vec_[this->mat_.col[aj]] -= value * this->mat_.val[aj];
            }

            cast_out->vec_[ai] = value;
        }

        return true;
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::LAnalyse(bool diag_unit)
    {
        this->L_diag_unit_ = diag_unit;
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::LAnalyseClear(void)
    {
        // do nothing
        this->L_diag_unit_ = true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::LSolve(const BaseVector<ValueType>& in,
                                          BaseVector<ValueType>*       out) const
    {
        assert(in.GetSize() >= 0);
        assert(out->GetSize() >= 0);
        assert(in.GetSize() == this->ncol_);
        assert(out->GetSize() == this->nrow_);

        const HostVector<ValueType>* cast_in  = dynamic_cast<const HostVector<ValueType>*>(&in);
        HostVector<ValueType>*       cast_out = dynamic_cast<HostVector<ValueType>*>(out);

        assert(cast_in != NULL);
        assert(cast_out != NULL);

        PtrType diag_aj = 0;

        // Solve L
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            cast_out->vec_[ai] = cast_in->vec_[ai];

            for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(this->mat_.col[aj] < ai)
                {
                    // under the diagonal
                    cast_out->vec_[ai] -= this->mat_.val[aj] * cast_out->vec_[this->mat_.col[aj]];
                }
                else
                {
                    // CSR should be sorted
                    if(this->L_diag_unit_ == false)
                    {
                        assert(this->mat_.col[aj] == ai);
                        diag_aj = aj;
                    }
                    break;
                }
            }

            if(this->L_diag_unit_ == false)
            {
                cast_out->vec_[ai] /= this->mat_.val[diag_aj];
            }
        }

        return true;
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::UAnalyse(bool diag_unit)
    {
        this->U_diag_unit_ = diag_unit;
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::UAnalyseClear(void)
    {
        // do nothing
        this->U_diag_unit_ = false;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::USolve(const BaseVector<ValueType>& in,
                                          BaseVector<ValueType>*       out) const
    {
        assert(in.GetSize() >= 0);
        assert(out->GetSize() >= 0);
        assert(in.GetSize() == this->ncol_);
        assert(out->GetSize() == this->nrow_);

        const HostVector<ValueType>* cast_in  = dynamic_cast<const HostVector<ValueType>*>(&in);
        HostVector<ValueType>*       cast_out = dynamic_cast<HostVector<ValueType>*>(out);

        assert(cast_in != NULL);
        assert(cast_out != NULL);

        // last elements should the diagonal one (last)
        int64_t diag_aj = this->nnz_ - 1;

        // Solve U
        for(int ai = this->nrow_ - 1; ai >= 0; --ai)
        {
            cast_out->vec_[ai] = cast_in->vec_[ai];

            for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(this->mat_.col[aj] > ai)
                {
                    // above the diagonal
                    cast_out->vec_[ai] -= this->mat_.val[aj] * cast_out->vec_[this->mat_.col[aj]];
                }

                if(this->U_diag_unit_ == false)
                {
                    if(this->mat_.col[aj] == ai)
                    {
                        diag_aj = aj;
                    }
                }
            }

            if(this->U_diag_unit_ == false)
            {
                cast_out->vec_[ai] /= this->mat_.val[diag_aj];
            }
        }

        return true;
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::ItLUAnalyse(void)
    {
        assert(this->ncol_ == this->nrow_);
        assert(this->tmp_vec_ == NULL);

        this->tmp_vec_ = new HostVector<ValueType>(this->local_backend_);

        assert(this->tmp_vec_ != NULL);

        // Status
        bool status;

        assert(this->nnz_ <= std::numeric_limits<int>::max());

        // Create buffer, if not already available
        size_t buffer_size   = 0;
        size_t L_buffer_size = 0;
        size_t U_buffer_size = 0;

        status = host_csritsv_buffer_size(host_sparse_operation_none,
                                          this->nrow_,
                                          static_cast<PtrType>(this->nnz_),
                                          host_sparse_fill_mode_lower,
                                          host_sparse_diag_type_unit,
                                          host_sparse_matrix_type_general,
                                          this->mat_.val,
                                          this->mat_.row_offset,
                                          this->mat_.col,
                                          &L_buffer_size);

        if(!status)
        {
            // LCOV_EXCL_START
            LOG_INFO("ItLUAnalyse() failed");
            FATAL_ERROR(__FILE__, __LINE__);
            // LCOV_EXCL_STOP
        }

        status = host_csritsv_buffer_size(host_sparse_operation_none,
                                          this->nrow_,
                                          static_cast<PtrType>(this->nnz_),
                                          host_sparse_fill_mode_upper,
                                          host_sparse_diag_type_non_unit,
                                          host_sparse_matrix_type_general,
                                          this->mat_.val,
                                          this->mat_.row_offset,
                                          this->mat_.col,
                                          &U_buffer_size);

        if(!status)
        {
            // LCOV_EXCL_START
            LOG_INFO("ItLUAnalyse() failed");
            FATAL_ERROR(__FILE__, __LINE__);
            // LCOV_EXCL_STOP
        }

        buffer_size = std::max(L_buffer_size, U_buffer_size);

        // Check buffer size
        if(this->mat_buffer_ != NULL && buffer_size > this->mat_buffer_size_)
        {
            free_host(&this->mat_buffer_);
            this->mat_buffer_ = NULL;
        }

        if(this->mat_buffer_ == NULL)
        {
            this->mat_buffer_size_ = buffer_size;
            allocate_host(buffer_size, &this->mat_buffer_);
        }

        assert(this->mat_buffer_size_ >= buffer_size);
        assert(this->mat_buffer_ != NULL);

        // Allocate temporary vector
        this->tmp_vec_->Allocate(this->nrow_);
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::ItLUAnalyseClear(void)
    {
        // Clear buffer
        if(this->mat_buffer_ != NULL)
        {
            free_host(&this->mat_buffer_);
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
    bool HostMatrixCSR<ValueType>::ItLUSolve(int                          max_iter,
                                             double                       tolerance,
                                             bool                         use_tol,
                                             const BaseVector<ValueType>& in,
                                             BaseVector<ValueType>*       out) const
    {
        if(this->nnz_ > 0)
        {
            assert(out != NULL);
            assert(this->ncol_ == this->nrow_);
            assert(in.GetSize() == this->ncol_);
            assert(out->GetSize() == this->nrow_);

            assert(this->tmp_vec_ != NULL);

            const HostVector<ValueType>* cast_in  = dynamic_cast<const HostVector<ValueType>*>(&in);
            HostVector<ValueType>*       cast_out = dynamic_cast<HostVector<ValueType>*>(out);

            assert(cast_in != NULL);
            assert(cast_out != NULL);

            bool status;

            int                               zero_pivot;
            const ValueType                   alpha = static_cast<ValueType>(1);
            const numeric_traits_t<ValueType> temp_tol
                = static_cast<numeric_traits_t<ValueType>>(tolerance);

            const numeric_traits_t<ValueType>* tol_ptr = (use_tol == false) ? nullptr : &temp_tol;

            assert(this->nnz_ <= std::numeric_limits<int>::max());

            // Solve L
            status = host_csritsv_solve(&max_iter,
                                        tol_ptr,
                                        nullptr,
                                        host_sparse_operation_none,
                                        this->nrow_,
                                        static_cast<PtrType>(this->nnz_),
                                        &alpha,
                                        host_sparse_fill_mode_lower,
                                        host_sparse_diag_type_unit,
                                        host_sparse_matrix_type_general,
                                        this->mat_.val,
                                        this->mat_.row_offset,
                                        this->mat_.col,
                                        cast_in->vec_,
                                        this->tmp_vec_->vec_,
                                        this->mat_buffer_,
                                        &zero_pivot);

            if(!status)
            {
                // LCOV_EXCL_START
                LOG_INFO("ItLUSolve() failed to solve L");
                FATAL_ERROR(__FILE__, __LINE__);
                // LCOV_EXCL_STOP
            }

            // Solve U
            status = host_csritsv_solve(&max_iter,
                                        tol_ptr,
                                        nullptr,
                                        host_sparse_operation_none,
                                        this->nrow_,
                                        static_cast<PtrType>(this->nnz_),
                                        &alpha,
                                        host_sparse_fill_mode_upper,
                                        host_sparse_diag_type_non_unit,
                                        host_sparse_matrix_type_general,
                                        this->mat_.val,
                                        this->mat_.row_offset,
                                        this->mat_.col,
                                        this->tmp_vec_->vec_,
                                        cast_out->vec_,
                                        this->mat_buffer_,
                                        &zero_pivot);

            if(!status)
            {
                // LCOV_EXCL_START
                LOG_INFO("ItLUSolve() failed to solve U");
                FATAL_ERROR(__FILE__, __LINE__);
                // LCOV_EXCL_STOP
            }
        }

        return true;
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::ItLLAnalyse(void)
    {
        assert(this->ncol_ == this->nrow_);
        assert(this->tmp_vec_ == NULL);

        this->tmp_vec_ = new HostVector<ValueType>(this->local_backend_);

        assert(this->tmp_vec_ != NULL);

        // Status
        bool status;

        assert(this->nnz_ <= std::numeric_limits<int>::max());

        // Create buffer, if not already available
        size_t buffer_size    = 0;
        size_t L_buffer_size  = 0;
        size_t Lt_buffer_size = 0;

        status = host_csritsv_buffer_size(host_sparse_operation_none,
                                          this->nrow_,
                                          static_cast<PtrType>(this->nnz_),
                                          host_sparse_fill_mode_lower,
                                          host_sparse_diag_type_non_unit,
                                          host_sparse_matrix_type_general,
                                          this->mat_.val,
                                          this->mat_.row_offset,
                                          this->mat_.col,
                                          &L_buffer_size);

        if(!status)
        {
            // LCOV_EXCL_START
            LOG_INFO("ItLLAnalyse() failed");
            FATAL_ERROR(__FILE__, __LINE__);
            // LCOV_EXCL_STOP
        }

        status = host_csritsv_buffer_size(host_sparse_operation_transpose,
                                          this->nrow_,
                                          static_cast<PtrType>(this->nnz_),
                                          host_sparse_fill_mode_lower,
                                          host_sparse_diag_type_non_unit,
                                          host_sparse_matrix_type_general,
                                          this->mat_.val,
                                          this->mat_.row_offset,
                                          this->mat_.col,
                                          &Lt_buffer_size);

        if(!status)
        {
            // LCOV_EXCL_START
            LOG_INFO("ItLLAnalyse() failed");
            FATAL_ERROR(__FILE__, __LINE__);
            // LCOV_EXCL_STOP
        }

        buffer_size = std::max(L_buffer_size, Lt_buffer_size);

        // Check buffer size
        if(this->mat_buffer_ != NULL && buffer_size > this->mat_buffer_size_)
        {
            free_host(&this->mat_buffer_);
            this->mat_buffer_ = NULL;
        }

        if(this->mat_buffer_ == NULL)
        {
            this->mat_buffer_size_ = buffer_size;
            allocate_host(buffer_size, &this->mat_buffer_);
        }

        assert(this->mat_buffer_size_ >= buffer_size);
        assert(this->mat_buffer_ != NULL);

        // Allocate temporary vector
        this->tmp_vec_->Allocate(this->nrow_);
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::ItLLAnalyseClear(void)
    {
        // Clear buffer
        if(this->mat_buffer_ != NULL)
        {
            free_host(&this->mat_buffer_);
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
    bool HostMatrixCSR<ValueType>::ItLLSolve(int                          max_iter,
                                             double                       tolerance,
                                             bool                         use_tol,
                                             const BaseVector<ValueType>& in,
                                             BaseVector<ValueType>*       out) const
    {
        if(this->nnz_ > 0)
        {
            assert(out != NULL);
            assert(this->ncol_ == this->nrow_);
            assert(in.GetSize() == this->ncol_);
            assert(out->GetSize() == this->nrow_);

            assert(this->tmp_vec_ != NULL);

            const HostVector<ValueType>* cast_in  = dynamic_cast<const HostVector<ValueType>*>(&in);
            HostVector<ValueType>*       cast_out = dynamic_cast<HostVector<ValueType>*>(out);

            assert(cast_in != NULL);
            assert(cast_out != NULL);

            bool status;

            int                               zero_pivot;
            const ValueType                   alpha = static_cast<ValueType>(1);
            const numeric_traits_t<ValueType> temp_tol
                = static_cast<numeric_traits_t<ValueType>>(tolerance);

            const numeric_traits_t<ValueType>* tol_ptr = (use_tol == false) ? nullptr : &temp_tol;

            assert(this->nnz_ <= std::numeric_limits<int>::max());

            // Solve L
            status = host_csritsv_solve(&max_iter,
                                        tol_ptr,
                                        nullptr,
                                        host_sparse_operation_none,
                                        this->nrow_,
                                        static_cast<PtrType>(this->nnz_),
                                        &alpha,
                                        host_sparse_fill_mode_lower,
                                        host_sparse_diag_type_non_unit,
                                        host_sparse_matrix_type_general,
                                        this->mat_.val,
                                        this->mat_.row_offset,
                                        this->mat_.col,
                                        cast_in->vec_,
                                        this->tmp_vec_->vec_,
                                        this->mat_buffer_,
                                        &zero_pivot);

            if(!status)
            {
                // LCOV_EXCL_START
                LOG_INFO("ItLLSolve() failed");
                FATAL_ERROR(__FILE__, __LINE__);
                // LCOV_EXCL_STOP
            }

            // Solve Lt
            status = host_csritsv_solve(&max_iter,
                                        tol_ptr,
                                        nullptr,
                                        host_sparse_operation_transpose,
                                        this->nrow_,
                                        static_cast<PtrType>(this->nnz_),
                                        &alpha,
                                        host_sparse_fill_mode_lower,
                                        host_sparse_diag_type_non_unit,
                                        host_sparse_matrix_type_general,
                                        this->mat_.val,
                                        this->mat_.row_offset,
                                        this->mat_.col,
                                        this->tmp_vec_->vec_,
                                        cast_out->vec_,
                                        this->mat_buffer_,
                                        &zero_pivot);

            if(!status)
            {
                // LCOV_EXCL_START
                LOG_INFO("ItLLSolve() failed");
                FATAL_ERROR(__FILE__, __LINE__);
                // LCOV_EXCL_STOP
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ItLLSolve(int                          max_iter,
                                             double                       tolerance,
                                             bool                         use_tol,
                                             const BaseVector<ValueType>& in,
                                             const BaseVector<ValueType>& inv_diag,
                                             BaseVector<ValueType>*       out) const
    {
        return ItLLSolve(max_iter, tolerance, use_tol, in, out);
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::ItLAnalyse(bool diag_unit)
    {
        assert(this->ncol_ == this->nrow_);

        // Status
        bool status;

        assert(this->nnz_ <= std::numeric_limits<int>::max());

        this->L_diag_unit_ = diag_unit;

        // Create buffer, if not already available
        size_t buffer_size = 0;

        status = host_csritsv_buffer_size(
            host_sparse_operation_none,
            this->nrow_,
            static_cast<PtrType>(this->nnz_),
            host_sparse_fill_mode_lower,
            (this->L_diag_unit_ ? host_sparse_diag_type_unit : host_sparse_diag_type_non_unit),
            host_sparse_matrix_type_general,
            this->mat_.val,
            this->mat_.row_offset,
            this->mat_.col,
            &buffer_size);

        if(!status)
        {
            // LCOV_EXCL_START
            LOG_INFO("ItLAnalyse() failed");
            FATAL_ERROR(__FILE__, __LINE__);
            // LCOV_EXCL_STOP
        }

        // Check buffer size
        if(this->mat_buffer_ != NULL && buffer_size > this->mat_buffer_size_)
        {
            free_host(&this->mat_buffer_);
            this->mat_buffer_ = NULL;
        }

        if(this->mat_buffer_ == NULL)
        {
            this->mat_buffer_size_ = buffer_size;
            allocate_host(buffer_size, &this->mat_buffer_);
        }

        assert(this->mat_buffer_size_ >= buffer_size);
        assert(this->mat_buffer_ != NULL);
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::ItLAnalyseClear(void)
    {
        // Clear buffer
        if(this->mat_buffer_ != NULL)
        {
            free_host(&this->mat_buffer_);
            this->mat_buffer_ = NULL;
        }

        this->mat_buffer_size_ = 0;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ItLSolve(int                          max_iter,
                                            double                       tolerance,
                                            bool                         use_tol,
                                            const BaseVector<ValueType>& in,
                                            BaseVector<ValueType>*       out) const
    {
        if(this->nnz_ > 0)
        {
            assert(out != NULL);
            assert(this->ncol_ == this->nrow_);
            assert(in.GetSize() == this->ncol_);
            assert(out->GetSize() == this->nrow_);

            const HostVector<ValueType>* cast_in  = dynamic_cast<const HostVector<ValueType>*>(&in);
            HostVector<ValueType>*       cast_out = dynamic_cast<HostVector<ValueType>*>(out);

            assert(cast_in != NULL);
            assert(cast_out != NULL);

            bool status;

            int                               zero_pivot;
            const ValueType                   alpha = static_cast<ValueType>(1);
            const numeric_traits_t<ValueType> temp_tol
                = static_cast<numeric_traits_t<ValueType>>(tolerance);

            const numeric_traits_t<ValueType>* tol_ptr = (use_tol == false) ? nullptr : &temp_tol;

            assert(this->nnz_ <= std::numeric_limits<int>::max());

            // Solve L
            status = host_csritsv_solve(
                &max_iter,
                tol_ptr,
                nullptr,
                host_sparse_operation_none,
                this->nrow_,
                static_cast<PtrType>(this->nnz_),
                &alpha,
                host_sparse_fill_mode_lower,
                (this->L_diag_unit_ ? host_sparse_diag_type_unit : host_sparse_diag_type_non_unit),
                host_sparse_matrix_type_general,
                this->mat_.val,
                this->mat_.row_offset,
                this->mat_.col,
                cast_in->vec_,
                cast_out->vec_,
                this->mat_buffer_,
                &zero_pivot);

            if(!status)
            {
                // LCOV_EXCL_START
                LOG_INFO("ItLSolve() failed");
                FATAL_ERROR(__FILE__, __LINE__);
                // LCOV_EXCL_STOP
            }
        }

        return true;
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::ItUAnalyse(bool diag_unit)
    {
        assert(this->ncol_ == this->nrow_);

        // Status
        bool status;

        assert(this->nnz_ <= std::numeric_limits<int>::max());

        this->U_diag_unit_ = diag_unit;

        // Create buffer, if not already available
        size_t buffer_size = 0;

        status = host_csritsv_buffer_size(
            host_sparse_operation_none,
            this->nrow_,
            static_cast<PtrType>(this->nnz_),
            host_sparse_fill_mode_upper,
            (this->U_diag_unit_ ? host_sparse_diag_type_unit : host_sparse_diag_type_non_unit),
            host_sparse_matrix_type_general,
            this->mat_.val,
            this->mat_.row_offset,
            this->mat_.col,
            &buffer_size);

        if(!status)
        {
            // LCOV_EXCL_START
            LOG_INFO("ItUAnalyse() failed");
            FATAL_ERROR(__FILE__, __LINE__);
            // LCOV_EXCL_STOP
        }

        // Check buffer size
        if(this->mat_buffer_ != NULL && buffer_size > this->mat_buffer_size_)
        {
            free_host(&this->mat_buffer_);
            this->mat_buffer_ = NULL;
        }

        if(this->mat_buffer_ == NULL)
        {
            this->mat_buffer_size_ = buffer_size;
            allocate_host(buffer_size, &this->mat_buffer_);
        }

        assert(this->mat_buffer_size_ >= buffer_size);
        assert(this->mat_buffer_ != NULL);
    }

    template <typename ValueType>
    void HostMatrixCSR<ValueType>::ItUAnalyseClear(void)
    {
        // Clear buffer
        if(this->mat_buffer_ != NULL)
        {
            free_host(&this->mat_buffer_);
            this->mat_buffer_ = NULL;
        }

        this->mat_buffer_size_ = 0;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ItUSolve(int                          max_iter,
                                            double                       tolerance,
                                            bool                         use_tol,
                                            const BaseVector<ValueType>& in,
                                            BaseVector<ValueType>*       out) const
    {
        if(this->nnz_ > 0)
        {
            assert(out != NULL);
            assert(this->ncol_ == this->nrow_);
            assert(in.GetSize() == this->ncol_);
            assert(out->GetSize() == this->nrow_);

            const HostVector<ValueType>* cast_in  = dynamic_cast<const HostVector<ValueType>*>(&in);
            HostVector<ValueType>*       cast_out = dynamic_cast<HostVector<ValueType>*>(out);

            assert(cast_in != NULL);
            assert(cast_out != NULL);

            bool status;

            int                               zero_pivot;
            const ValueType                   alpha = static_cast<ValueType>(1);
            const numeric_traits_t<ValueType> temp_tol
                = static_cast<numeric_traits_t<ValueType>>(tolerance);

            const numeric_traits_t<ValueType>* tol_ptr = (use_tol == false) ? nullptr : &temp_tol;

            assert(this->nnz_ <= std::numeric_limits<int>::max());

            // Solve U
            status = host_csritsv_solve(
                &max_iter,
                tol_ptr,
                nullptr,
                host_sparse_operation_none,
                this->nrow_,
                static_cast<PtrType>(this->nnz_),
                &alpha,
                host_sparse_fill_mode_upper,
                (this->U_diag_unit_ ? host_sparse_diag_type_unit : host_sparse_diag_type_non_unit),
                host_sparse_matrix_type_general,
                this->mat_.val,
                this->mat_.row_offset,
                this->mat_.col,
                cast_in->vec_,
                cast_out->vec_,
                this->mat_buffer_,
                &zero_pivot);

            if(!status)
            {
                // LCOV_EXCL_START
                LOG_INFO("ItUSolve() failed");
                FATAL_ERROR(__FILE__, __LINE__);
                // LCOV_EXCL_STOP
            }
        }

        return true;
    }

    // Algorithm for ILU factorization is based on
    // Y. Saad, Iterative methods for sparse linear systems, 2nd edition, SIAM
    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ILU0Factorize(void)
    {
        assert(this->nrow_ == this->ncol_);
        assert(this->nnz_ > 0);

        // pointer of upper part of each row
        PtrType* diag_offset = NULL;
        PtrType* nnz_entries = NULL;

        allocate_host(this->nrow_, &diag_offset);
        allocate_host(this->nrow_, &nnz_entries);

        set_to_zero_host(this->nrow_, nnz_entries);

        // ai = 0 to N loop over all rows
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            // ai-th row entries
            PtrType row_start = this->mat_.row_offset[ai];
            PtrType row_end   = this->mat_.row_offset[ai + 1];
            PtrType j;

            // nnz position of ai-th row in mat_.val array
            for(j = row_start; j < row_end; ++j)
            {
                nnz_entries[this->mat_.col[j]] = j;
            }

            // loop over ai-th row nnz entries
            for(j = row_start; j < row_end; ++j)
            {
                // if nnz entry is in lower matrix
                if(this->mat_.col[j] < ai)
                {
                    int     col_j  = this->mat_.col[j];
                    PtrType diag_j = diag_offset[col_j];

                    if(this->mat_.val[diag_j] != static_cast<ValueType>(0))
                    {
                        // multiplication factor
                        this->mat_.val[j] = this->mat_.val[j] / this->mat_.val[diag_j];

                        // loop over upper offset pointer and do linear combination for nnz entry
                        for(PtrType k = diag_j + 1; k < this->mat_.row_offset[col_j + 1]; ++k)
                        {
                            // if nnz at this position do linear combination
                            if(nnz_entries[this->mat_.col[k]] != 0)
                            {
                                this->mat_.val[nnz_entries[this->mat_.col[k]]]
                                    -= this->mat_.val[j] * this->mat_.val[k];
                            }
                        }
                    }
                }
                else
                {
                    break;
                }
            }

            // set diagonal pointer to diagonal element
            diag_offset[ai] = j;

            // clear nnz entries
            for(j = row_start; j < row_end; ++j)
            {
                nnz_entries[this->mat_.col[j]] = 0;
            }
        }

        free_host(&diag_offset);
        free_host(&nnz_entries);

        return true;
    }

    // Algorithm for ILUT factorization is based on
    // Y. Saad, Iterative methods for sparse linear systems, 2nd edition, SIAM
    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ILUTFactorize(double t, int maxrow)
    {
        assert(this->nrow_ == this->ncol_);
        assert(this->nnz_ > 0);

        int nrow = this->nrow_;
        int ncol = this->ncol_;

        int64_t nnz = std::min(2 * maxrow, nrow) * nrow;

        int remaining_nnz = nnz;

        PtrType*   row_offset = NULL;
        int*       col        = NULL;
        ValueType* val        = NULL;

        allocate_host(nrow + 1, &row_offset);
        allocate_host(nnz, &col);
        allocate_host(nnz, &val);

        void*    buffer            = NULL;
        PtrType* temp_pivot_buffer = NULL;

        ILUTDriverCSR<ValueType, int> driver(nrow, nrow);

        buffer = (void*)malloc(driver.buffer_size());
        driver.set_buffer(buffer);

        allocate_host(nrow, &temp_pivot_buffer);
        memset(temp_pivot_buffer, 0, sizeof(PtrType) * nrow);

        // initialize row ptr B
        row_offset[0] = 0;

        // for each row
        for(int i = 0; i < nrow; i++)
        {
            PtrType row_begin = mat_.row_offset[i];
            PtrType row_end   = mat_.row_offset[i + 1];
            PtrType row_size  = row_end - row_begin;

            // Initialize working structure
            driver.initialize(&mat_.val[row_begin], &mat_.col[row_begin], row_size, 0, i);

            ValueType k_val;
            int       k_col;

            // for each column in row i under the diagonal
            while(driver.next_lower(k_col, k_val))
            {
                //find the index of element k, k in the upper matrix
                PtrType pivot_index;
                if((pivot_index = temp_pivot_buffer[k_col]) <= 0)
                {
                    // zero pivot
                    continue;
                }
                pivot_index--;

                // get the coefficient to perform row(i) = row(i) - factor * row(k)
                // factor = a(i, k) / a(k, k)
                ValueType factor = static_cast<ValueType>(1) / val[pivot_index];

                factor = factor * k_val;

                if(std::abs(factor) > std::abs(t))
                {
                    // put pivot in lower matrix; put in a(i, k)
                    driver.save_lower(factor);

                    // subtract row(k) from row(i); perform row(i) = row(i) - pivot * row(k)
                    // for each non-zero element in row k after column k
                    for(PtrType subtrahend_index = pivot_index + 1;
                        subtrahend_index < row_offset[k_col + 1];
                        subtrahend_index++)
                    {
                        // a(i, j) = a(i, j) - pivot * a(k, j)
                        int       column     = col[subtrahend_index];
                        ValueType val_to_add = -factor * val[subtrahend_index];

                        driver.add_to_element(column, val_to_add);
                    }
                }
            }

            // filter elements
            driver.trim(t, maxrow);

            // store elements
            int store_size = driver.row_size();

            if(remaining_nnz < store_size)
            {
                nnz += static_cast<size_t>(nnz * 1.5f);
                remaining_nnz += static_cast<size_t>(nnz * 1.5f);

                LOG_INFO("ILUTFactorize reallocating");

                int*       col_tmp = (int*)realloc(col, nnz * sizeof(int));
                ValueType* val_tmp = (ValueType*)realloc(val, nnz * sizeof(ValueType));

                if(col_tmp == NULL || val_tmp == NULL)
                {
                    // LCOV_EXCL_START
                    free(col);
                    free(val);

                    LOG_INFO("ILUTFactorize failed on realloc");
                    FATAL_ERROR(__FILE__, __LINE__);
                    // LCOV_EXCL_STOP
                }
                else
                {
                    col = col_tmp;
                    val = val_tmp;
                }
            }

            int diag_in;
            if(driver.store_row(&val[row_offset[i]], &col[row_offset[i]], diag_in))
            {
                temp_pivot_buffer[i] = row_offset[i] + diag_in + 1;
            }
            else
            {
                LOG_INFO("(ILUT) zero row");
            }

            row_offset[i + 1] = row_offset[i] + store_size;
            remaining_nnz -= store_size;
        }

        free(buffer);
        free(temp_pivot_buffer);

        // pinned memory
        int*       p_col = NULL;
        ValueType* p_val = NULL;

        nnz = row_offset[nrow];

        allocate_host(nnz, &p_col);
        allocate_host(nnz, &p_val);

        copy_h2h(nnz, col, p_col);
        copy_h2h(nnz, val, p_val);

        free(col);
        free(val);

        this->Clear();
        this->SetDataPtrCSR(&row_offset, &p_col, &p_val, nnz, nrow, ncol);

        this->Sort();

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ItILU0Factorize(ItILU0Algorithm alg,
                                                   int             option,
                                                   int             max_iter,
                                                   double          tolerance,
                                                   int*            niter,
                                                   double*         history)
    {
        // On host, we fall back to default ILU0
        return this->ILU0Factorize();
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ICFactorize(BaseVector<ValueType>* inv_diag)
    {
        assert(this->nrow_ == this->ncol_);
        assert(this->nnz_ > 0);

        assert(inv_diag != NULL);
        HostVector<ValueType>* cast_diag = dynamic_cast<HostVector<ValueType>*>(inv_diag);
        assert(cast_diag != NULL);

        cast_diag->Allocate(this->nrow_);

        PtrType* diag_offset = NULL;
        PtrType* nnz_entries = NULL;

        allocate_host(this->nrow_, &diag_offset);
        allocate_host(this->nrow_, &nnz_entries);

        set_to_zero_host(this->nrow_, nnz_entries);

        // i=0,..n
        for(int i = 0; i < this->nrow_; ++i)
        {
            PtrType row_begin = this->mat_.row_offset[i];
            PtrType row_end   = this->mat_.row_offset[i + 1];

            for(PtrType j = row_begin; j < row_end; ++j)
            {
                nnz_entries[this->mat_.col[j]] = j;
            }

            ValueType sum = static_cast<ValueType>(0);

            bool has_diag = false;

            // j=0,..i
            PtrType j;
            for(j = row_begin; j < row_end; ++j)
            {
                int       col_j = this->mat_.col[j];
                ValueType val_j = this->mat_.val[j];

                // Mark diagonal and skip row
                if(col_j == i)
                {
                    has_diag = true;
                    break;
                }

                // Skip upper triangular
                if(col_j > i)
                {
                    break;
                }

                PtrType row_begin_j = this->mat_.row_offset[col_j];
                PtrType row_diag_j  = diag_offset[col_j];

                ValueType local_sum = static_cast<ValueType>(0);
                ValueType inv_diag  = this->mat_.val[row_diag_j];

                // Check for numeric zero
                if(inv_diag == static_cast<ValueType>(0))
                {
                    // LCOV_EXCL_START
                    LOG_INFO("IC breakdown: division by zero");
                    FATAL_ERROR(__FILE__, __LINE__);
                    // LCOV_EXCL_STOP
                }

                inv_diag = static_cast<ValueType>(1) / inv_diag;

                for(PtrType k = row_begin_j; k < row_diag_j; ++k)
                {
                    int col_k = this->mat_.col[k];

                    if(nnz_entries[col_k] != 0)
                    {
                        PtrType idx = nnz_entries[col_k];
                        local_sum += this->mat_.val[k] * this->mat_.val[idx];
                    }
                }

                val_j = (val_j - local_sum) * inv_diag;
                sum += val_j * val_j;

                this->mat_.val[j] = val_j;
            }

            if(!has_diag)
            {
                // LCOV_EXCL_START
                // Structural zero
                LOG_INFO("IC breakdown: structural zero diagonal");
                FATAL_ERROR(__FILE__, __LINE__);
                // LCOV_EXCL_STOP
            }

            // Process diagonal entry
            ValueType diag_entry = std::sqrt(std::abs(this->mat_.val[j] - sum));
            this->mat_.val[j]    = diag_entry;

            // Check for numerical zero
            if(diag_entry == static_cast<ValueType>(0))
            {
                // LCOV_EXCL_START
                LOG_INFO("IC breakdown: division by zero");
                FATAL_ERROR(__FILE__, __LINE__);
                // LCOV_EXCL_STOP
            }

            // Store inverse diagonal entry
            cast_diag->vec_[i] = static_cast<ValueType>(1) / diag_entry;

            // Store diagonal offset
            diag_offset[i] = j;

            // Clear nnz entries
            for(j = row_begin; j < row_end; ++j)
            {
                nnz_entries[this->mat_.col[j]] = 0;
            }
        }

        // Free temporary storage
        free_host(&diag_offset);
        free_host(&nnz_entries);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::MultiColoring(int&             num_colors,
                                                 int**            size_colors,
                                                 BaseVector<int>* permutation) const
    {
        assert(*size_colors == NULL);
        assert(permutation != NULL);
        HostVector<int>* cast_perm = dynamic_cast<HostVector<int>*>(permutation);
        assert(cast_perm != NULL);

        /*
        *   Create CSC
        */
        PtrType* csc_ptr = NULL;
        int*     csc_ind = NULL;

        allocate_host(this->ncol_ + 1, &csc_ptr);
        allocate_host(this->nnz_, &csc_ind);

        set_to_zero_host(this->nrow_ + 1, csc_ptr);

        for(PtrType i = 0; i < this->nnz_; ++i)
        {
            csc_ptr[this->mat_.col[i] + 1] += 1;
        }

        for(int i = 1; i < this->nrow_ + 1; ++i)
        {
            csc_ptr[i] += csc_ptr[i - 1];
        }

        for(int i = 0; i < this->nrow_; ++i)
        {
            for(PtrType k = this->mat_.row_offset[i]; k < this->mat_.row_offset[i + 1]; ++k)
            {
                csc_ind[csc_ptr[this->mat_.col[k]]++] = i;
            }
        }
        for(int i = this->nrow_; i > 0; --i)
        {
            csc_ptr[i] = csc_ptr[i - 1];
        }
        csc_ptr[0] = 0;

        // node colors (init value = 0 i.e. no color)
        int* color = NULL;
        allocate_host(this->nrow_, &color);

        memset(color, 0, sizeof(int) * this->nrow_);
        num_colors = 0;
        std::vector<bool> row_col;

        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            color[ai] = 1;
            row_col.clear();
            row_col.reserve(num_colors + 2);
            row_col.assign(num_colors + 2, false);

            for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(ai != this->mat_.col[aj])
                {
                    row_col[color[this->mat_.col[aj]]] = true;
                }
            }

            for(PtrType aj = csc_ptr[ai]; aj < csc_ptr[ai + 1]; ++aj)
            {
                if(ai != csc_ind[aj])
                {
                    row_col[color[csc_ind[aj]]] = true;
                }
            }

            PtrType count = this->mat_.row_offset[ai + 1] - this->mat_.row_offset[ai]
                            + csc_ptr[ai + 1] - csc_ptr[ai];

            for(PtrType aj = 0; aj < count; ++aj)
            {
                if(row_col[color[ai]] == true)
                {
                    ++color[ai];
                }
                else
                {
                    break;
                }
            }

            if(color[ai] > num_colors)
            {
                num_colors = color[ai];
            }
        }

        free_host(&csc_ptr);
        free_host(&csc_ind);

        allocate_host(num_colors, size_colors);
        set_to_zero_host(num_colors, *size_colors);

        int* offsets_color = NULL;
        allocate_host(num_colors, &offsets_color);
        memset(offsets_color, 0, sizeof(int) * num_colors);

        for(int i = 0; i < this->nrow_; ++i)
        {
            ++(*size_colors)[color[i] - 1];
        }

        int total = 0;
        for(int i = 1; i < num_colors; ++i)
        {
            total += (*size_colors)[i - 1];
            offsets_color[i] = total;
            //   LOG_INFO("offsets = " << total);
        }

        cast_perm->Allocate(this->nrow_);

        for(int i = 0; i < permutation->GetSize(); ++i)
        {
            cast_perm->vec_[i] = offsets_color[color[i] - 1];
            ++offsets_color[color[i] - 1];
        }

        free_host(&color);
        free_host(&offsets_color);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::MaximalIndependentSet(int&             size,
                                                         BaseVector<int>* permutation) const
    {
        assert(permutation != NULL);
        assert(this->nrow_ == this->ncol_);

        HostVector<int>* cast_perm = dynamic_cast<HostVector<int>*>(permutation);
        assert(cast_perm != NULL);

        int* mis = NULL;
        allocate_host(this->nrow_, &mis);
        memset(mis, 0, sizeof(int) * this->nrow_);

        size = 0;

        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            if(mis[ai] == 0)
            {
                // set the node
                mis[ai] = 1;
                ++size;

                // remove all nbh nodes (without diagonal)
                for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1];
                    ++aj)
                {
                    if(ai != this->mat_.col[aj])
                    {
                        mis[this->mat_.col[aj]] = -1;
                    }
                }
            }
        }

        cast_perm->Allocate(this->nrow_);

        int pos = 0;
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            if(mis[ai] == 1)
            {
                cast_perm->vec_[ai] = pos;
                ++pos;
            }
            else
            {
                cast_perm->vec_[ai] = size + ai - pos;
            }
        }

        // Check the permutation
        //
        //  for (int ai=0; ai<this->nrow_; ++ai) {
        //    assert( cast_perm->vec_[ai] >= 0 );
        //    assert( cast_perm->vec_[ai] < this->nrow_ );
        //  }

        free_host(&mis);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ZeroBlockPermutation(int&             size,
                                                        BaseVector<int>* permutation) const
    {
        assert(permutation != NULL);
        assert(permutation->GetSize() == this->nrow_);
        assert(permutation->GetSize() == this->ncol_);

        HostVector<int>* cast_perm = dynamic_cast<HostVector<int>*>(permutation);
        assert(cast_perm != NULL);

        size = 0;

        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(ai == this->mat_.col[aj])
                {
                    ++size;
                }
            }
        }

        int k_z  = size;
        int k_nz = 0;

        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            bool hit = false;

            for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(ai == this->mat_.col[aj])
                {
                    cast_perm->vec_[ai] = k_nz;
                    ++k_nz;
                    hit = true;
                }
            }

            if(hit == false)
            {
                cast_perm->vec_[ai] = k_z;
                ++k_z;
            }
        }

        return true;
    }

    // following R.E.Bank and C.C.Douglas paper
    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::SymbolicMatMatMult(const BaseMatrix<ValueType>& src)
    {
        const HostMatrixCSR<ValueType>* cast_mat
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&src);

        assert(cast_mat != NULL);
        assert(this->ncol_ == cast_mat->nrow_);

        std::vector<PtrType> row_offset;
        std::vector<int>*    new_col = new std::vector<int>[this->nrow_];

        row_offset.resize(this->nrow_ + 1);

        row_offset[0] = 0;

        _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int i = 0; i < this->nrow_; ++i)
        {
            // loop over the row
            for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                int ii = this->mat_.col[j];

                // loop corresponding row
                for(PtrType k = cast_mat->mat_.row_offset[ii];
                    k < cast_mat->mat_.row_offset[ii + 1];
                    ++k)
                {
                    new_col[i].push_back(cast_mat->mat_.col[k]);
                }
            }

            std::sort(new_col[i].begin(), new_col[i].end());
            new_col[i].erase(std::unique(new_col[i].begin(), new_col[i].end()), new_col[i].end());

            row_offset[i + 1] = static_cast<PtrType>(new_col[i].size());
        }

        for(int i = 0; i < this->nrow_; ++i)
        {
            row_offset[i + 1] += row_offset[i];
        }

        this->AllocateCSR(row_offset[this->nrow_], this->nrow_, this->ncol_);

        copy_h2h(this->nrow_ + 1, row_offset.data(), this->mat_.row_offset);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int i = 0; i < this->nrow_; ++i)
        {
            PtrType jj = 0;
            for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                this->mat_.col[j] = new_col[i][jj];
                ++jj;
            }
        }

        //#ifdef _OPENMP
        //#pragma omp parallel for
        //#endif
        //  for (int64_t i=0; i<this->nnz_; ++i)
        //    this->mat_.val[i] = static_cast<ValueType>(1);

        delete[] new_col;

        return true;
    }

    // ----------------------------------------------------------
    // original function prod(const spmat1 &A, const spmat2 &B)
    // ----------------------------------------------------------
    // Modified and adapted from AMGCL,
    // https://github.com/ddemidov/amgcl
    // MIT License
    // ----------------------------------------------------------
    // CHANGELOG
    // - adapted interface
    // sorting is added
    // ----------------------------------------------------------
    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::MatMatMult(const BaseMatrix<ValueType>& A,
                                              const BaseMatrix<ValueType>& B)
    {
        assert((this != &A) && (this != &B));

        //  this->SymbolicMatMatMult(A, B);
        //  this->NumericMatMatMult (A, B);
        //
        //  return true;

        const HostMatrixCSR<ValueType>* cast_mat_A
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&A);
        const HostMatrixCSR<ValueType>* cast_mat_B
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&B);

        assert(cast_mat_A != NULL);
        assert(cast_mat_B != NULL);
        assert(cast_mat_A->ncol_ == cast_mat_B->nrow_);

        int n = cast_mat_A->nrow_;
        int m = cast_mat_B->ncol_;

        PtrType* row_offset = NULL;
        allocate_host(n + 1, &row_offset);
        int*       col = NULL;
        ValueType* val = NULL;

        set_to_zero_host(n + 1, row_offset);

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            std::vector<PtrType> marker(m, -1);

#ifdef _OPENMP
            int nt  = omp_get_num_threads();
            int tid = omp_get_thread_num();

            int chunk_size  = (n + nt - 1) / nt;
            int chunk_start = tid * chunk_size;
            int chunk_end   = std::min(n, chunk_start + chunk_size);
#else
            int chunk_start = 0;
            int chunk_end   = n;
#endif

            for(int ia = chunk_start; ia < chunk_end; ++ia)
            {
                for(PtrType ja = cast_mat_A->mat_.row_offset[ia],
                            ea = cast_mat_A->mat_.row_offset[ia + 1];
                    ja < ea;
                    ++ja)
                {
                    int ca = cast_mat_A->mat_.col[ja];
                    for(PtrType jb = cast_mat_B->mat_.row_offset[ca],
                                eb = cast_mat_B->mat_.row_offset[ca + 1];
                        jb < eb;
                        ++jb)
                    {
                        int cb = cast_mat_B->mat_.col[jb];

                        if(marker[cb] != ia)
                        {
                            marker[cb] = ia;
                            ++row_offset[ia + 1];
                        }
                    }
                }
            }

            std::fill(marker.begin(), marker.end(), -1);

#ifdef _OPENMP
#pragma omp barrier
#endif
#ifdef _OPENMP
#pragma omp single
#endif
            {
                for(int i = 1; i < n + 1; ++i)
                {
                    row_offset[i] += row_offset[i - 1];
                }

                allocate_host(row_offset[n], &col);
                allocate_host(row_offset[n], &val);
            }

            for(int ia = chunk_start; ia < chunk_end; ++ia)
            {
                PtrType row_begin = row_offset[ia];
                PtrType row_end   = row_begin;

                for(PtrType ja = cast_mat_A->mat_.row_offset[ia],
                            ea = cast_mat_A->mat_.row_offset[ia + 1];
                    ja < ea;
                    ++ja)
                {
                    int       ca = cast_mat_A->mat_.col[ja];
                    ValueType va = cast_mat_A->mat_.val[ja];

                    for(PtrType jb = cast_mat_B->mat_.row_offset[ca],
                                eb = cast_mat_B->mat_.row_offset[ca + 1];
                        jb < eb;
                        ++jb)
                    {
                        int       cb = cast_mat_B->mat_.col[jb];
                        ValueType vb = cast_mat_B->mat_.val[jb];

                        if(marker[cb] < row_begin)
                        {
                            marker[cb]   = row_end;
                            col[row_end] = cb;
                            val[row_end] = va * vb;
                            ++row_end;
                        }
                        else
                        {
                            val[marker[cb]] += va * vb;
                        }
                    }
                }
            }
        }

        this->SetDataPtrCSR(
            &row_offset, &col, &val, row_offset[n], cast_mat_A->nrow_, cast_mat_B->ncol_);

        // Sorting the col (per row)
        this->Sort();

        return true;
    }

    // following R.E.Bank and C.C.Douglas paper
    // this = A * B
    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::SymbolicMatMatMult(const BaseMatrix<ValueType>& A,
                                                      const BaseMatrix<ValueType>& B)
    {
        const HostMatrixCSR<ValueType>* cast_mat_A
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&A);
        const HostMatrixCSR<ValueType>* cast_mat_B
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&B);

        assert(cast_mat_A != NULL);
        assert(cast_mat_B != NULL);
        assert(cast_mat_A->ncol_ == cast_mat_B->nrow_);

        std::vector<PtrType> row_offset;
        std::vector<int>*    new_col = new std::vector<int>[cast_mat_A->nrow_];

        row_offset.resize(cast_mat_A->nrow_ + 1);

        row_offset[0] = 0;

        _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int i = 0; i < cast_mat_A->nrow_; ++i)
        {
            // loop over the row
            for(PtrType j = cast_mat_A->mat_.row_offset[i]; j < cast_mat_A->mat_.row_offset[i + 1];
                ++j)
            {
                int ii = cast_mat_A->mat_.col[j];
                //      new_col[i].push_back(ii);

                // loop corresponding row
                for(PtrType k = cast_mat_B->mat_.row_offset[ii];
                    k < cast_mat_B->mat_.row_offset[ii + 1];
                    ++k)
                {
                    new_col[i].push_back(cast_mat_B->mat_.col[k]);
                }
            }

            std::sort(new_col[i].begin(), new_col[i].end());
            new_col[i].erase(std::unique(new_col[i].begin(), new_col[i].end()), new_col[i].end());

            row_offset[i + 1] = static_cast<PtrType>(new_col[i].size());
        }

        for(int i = 0; i < cast_mat_A->nrow_; ++i)
        {
            row_offset[i + 1] += row_offset[i];
        }

        this->AllocateCSR(row_offset[cast_mat_A->nrow_], cast_mat_A->nrow_, cast_mat_B->ncol_);

        copy_h2h(cast_mat_A->nrow_ + 1, row_offset.data(), this->mat_.row_offset);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int i = 0; i < cast_mat_A->nrow_; ++i)
        {
            PtrType jj = 0;
            for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                this->mat_.col[j] = new_col[i][jj];
                ++jj;
            }
        }

        //#ifdef _OPENMP
        //#pragma omp parallel for
        //#endif
        //      for (int64_t i=0; i<this->nnz_; ++i)
        //      this->mat_.val[i] = static_cast<ValueType>(1);

        delete[] new_col;

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::NumericMatMatMult(const BaseMatrix<ValueType>& A,
                                                     const BaseMatrix<ValueType>& B)
    {
        const HostMatrixCSR<ValueType>* cast_mat_A
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&A);
        const HostMatrixCSR<ValueType>* cast_mat_B
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&B);

        assert(cast_mat_A != NULL);
        assert(cast_mat_B != NULL);
        assert(cast_mat_A->ncol_ == cast_mat_B->nrow_);
        assert(this->nrow_ == cast_mat_A->nrow_);
        assert(this->ncol_ == cast_mat_B->ncol_);

        _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int i = 0; i < cast_mat_A->nrow_; ++i)
        {
            // loop over the row
            for(PtrType j = cast_mat_A->mat_.row_offset[i]; j < cast_mat_A->mat_.row_offset[i + 1];
                ++j)
            {
                int ii = cast_mat_A->mat_.col[j];

                // loop corresponding row
                for(PtrType k = cast_mat_B->mat_.row_offset[ii];
                    k < cast_mat_B->mat_.row_offset[ii + 1];
                    ++k)
                {
                    for(PtrType p = this->mat_.row_offset[i]; p < this->mat_.row_offset[i + 1]; ++p)
                    {
                        if(cast_mat_B->mat_.col[k] == this->mat_.col[p])
                        {
                            this->mat_.val[p] += cast_mat_B->mat_.val[k] * cast_mat_A->mat_.val[j];
                            break;
                        }
                    }
                }
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::SymbolicPower(int p)
    {
        assert(p > 1);

        // The optimal values for the firsts values

        if(p == 2)
        {
            this->SymbolicMatMatMult(*this);
        }

        if(p == 3)
        {
            HostMatrixCSR<ValueType> tmp(this->local_backend_);
            tmp.CopyFrom(*this);

            this->SymbolicPower(2);
            this->SymbolicMatMatMult(tmp);
        }

        if(p == 4)
        {
            this->SymbolicPower(2);
            this->SymbolicPower(2);
        }

        if(p == 5)
        {
            HostMatrixCSR<ValueType> tmp(this->local_backend_);
            tmp.CopyFrom(*this);

            this->SymbolicPower(4);
            this->SymbolicMatMatMult(tmp);
        }

        if(p == 6)
        {
            this->SymbolicPower(2);
            this->SymbolicPower(3);
        }

        if(p == 7)
        {
            HostMatrixCSR<ValueType> tmp(this->local_backend_);
            tmp.CopyFrom(*this);

            this->SymbolicPower(6);
            this->SymbolicMatMatMult(tmp);
        }

        if(p == 8)
        {
            HostMatrixCSR<ValueType> tmp(this->local_backend_);
            tmp.CopyFrom(*this);

            this->SymbolicPower(6);
            tmp.SymbolicPower(2);

            this->SymbolicMatMatMult(tmp);
        }

        if(p > 8)
        {
            HostMatrixCSR<ValueType> tmp(this->local_backend_);
            tmp.CopyFrom(*this);

            for(int i = 0; i < p; ++i)
            {
                this->SymbolicMatMatMult(tmp);
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ILUpFactorizeNumeric(int p, const BaseMatrix<ValueType>& mat)
    {
        const HostMatrixCSR<ValueType>* cast_mat
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&mat);

        assert(cast_mat != NULL);
        assert(cast_mat->nrow_ == this->nrow_);
        assert(cast_mat->ncol_ == this->ncol_);
        assert(this->nnz_ > 0);
        assert(cast_mat->nnz_ > 0);

        PtrType*   row_offset = NULL;
        PtrType*   ind_diag   = NULL;
        int*       levels     = NULL;
        ValueType* val        = NULL;

        allocate_host(cast_mat->nrow_ + 1, &row_offset);
        allocate_host(cast_mat->nrow_, &ind_diag);
        allocate_host(cast_mat->nnz_, &levels);
        allocate_host(cast_mat->nnz_, &val);

        int     inf_level = 99999;
        int64_t nnz       = 0;

        _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        // find diagonals
        for(int ai = 0; ai < cast_mat->nrow_; ++ai)
        {
            for(PtrType aj = cast_mat->mat_.row_offset[ai]; aj < cast_mat->mat_.row_offset[ai + 1];
                ++aj)
            {
                if(ai == cast_mat->mat_.col[aj])
                {
                    ind_diag[ai] = aj;
                    break;
                }
            }
        }

        // init row_offset
        set_to_zero_host(cast_mat->nrow_ + 1, row_offset);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        // init inf levels
        for(int64_t i = 0; i < cast_mat->nnz_; ++i)
        {
            levels[i] = inf_level;
        }

        set_to_zero_host(cast_mat->nnz_, val);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        // fill levels and values
        for(int ai = 0; ai < cast_mat->nrow_; ++ai)
        {
            for(PtrType aj = cast_mat->mat_.row_offset[ai]; aj < cast_mat->mat_.row_offset[ai + 1];
                ++aj)
            {
                for(PtrType ajj = this->mat_.row_offset[ai]; ajj < this->mat_.row_offset[ai + 1];
                    ++ajj)
                {
                    if(cast_mat->mat_.col[aj] == this->mat_.col[ajj])
                    {
                        val[aj]    = this->mat_.val[ajj];
                        levels[aj] = 0;
                        break;
                    }
                }
            }
        }

        // ai = 1 to N
        for(int ai = 1; ai < cast_mat->nrow_; ++ai)
        {
            // ak = 1 to ai-1
            for(PtrType ak = cast_mat->mat_.row_offset[ai]; ai > cast_mat->mat_.col[ak]; ++ak)
            {
                if(levels[ak] <= p)
                {
                    val[ak] /= val[ind_diag[cast_mat->mat_.col[ak]]];

                    // aj = ak+1 to N
                    for(PtrType aj = ak + 1; aj < cast_mat->mat_.row_offset[ai + 1]; ++aj)
                    {
                        ValueType val_kj   = static_cast<ValueType>(0);
                        int       level_kj = inf_level;

                        // find a_k,j
                        for(PtrType kj = cast_mat->mat_.row_offset[cast_mat->mat_.col[ak]];
                            kj < cast_mat->mat_.row_offset[cast_mat->mat_.col[ak] + 1];
                            ++kj)
                        {
                            if(cast_mat->mat_.col[aj] == cast_mat->mat_.col[kj])
                            {
                                level_kj = levels[kj];
                                val_kj   = val[kj];
                                break;
                            }
                        }

                        int lev = level_kj + levels[ak] + 1;

                        if(levels[aj] > lev)
                        {
                            levels[aj] = lev;
                        }

                        // a_i,j = a_i,j - a_i,k * a_k,j
                        val[aj] -= val[ak] * val_kj;
                    }
                }
            }

            for(PtrType ak = cast_mat->mat_.row_offset[ai]; ak < cast_mat->mat_.row_offset[ai + 1];
                ++ak)
            {
                if(levels[ak] > p)
                {
                    levels[ak] = inf_level;
                    val[ak]    = static_cast<ValueType>(0);
                }
                else
                {
                    ++row_offset[ai + 1];
                }
            }
        }

        row_offset[0] = this->mat_.row_offset[0];
        row_offset[1] = this->mat_.row_offset[1];

        for(int i = 0; i < cast_mat->nrow_; ++i)
        {
            row_offset[i + 1] += row_offset[i];
        }

        nnz = row_offset[cast_mat->nrow_];

        this->AllocateCSR(nnz, cast_mat->nrow_, cast_mat->ncol_);

        PtrType jj = 0;
        for(int i = 0; i < cast_mat->nrow_; ++i)
        {
            for(PtrType j = cast_mat->mat_.row_offset[i]; j < cast_mat->mat_.row_offset[i + 1]; ++j)
            {
                if(levels[j] <= p)
                {
                    this->mat_.col[jj] = cast_mat->mat_.col[j];
                    this->mat_.val[jj] = val[j];
                    ++jj;
                }
            }
        }

        assert(jj == nnz);

        copy_h2h(this->nrow_ + 1, row_offset, this->mat_.row_offset);

        free_host(&row_offset);
        free_host(&ind_diag);
        free_host(&levels);
        free_host(&val);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::MatrixAdd(const BaseMatrix<ValueType>& mat,
                                             ValueType                    alpha,
                                             ValueType                    beta,
                                             bool                         structure)
    {
        const HostMatrixCSR<ValueType>* cast_mat
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&mat);

        assert(cast_mat != NULL);
        assert(cast_mat->nrow_ == this->nrow_);
        assert(cast_mat->ncol_ == this->ncol_);
        assert(this->nnz_ >= 0);
        assert(cast_mat->nnz_ >= 0);

        _set_omp_backend_threads(this->local_backend_, this->nrow_);

        // the structure is sub-set
        if(structure == false)
        {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            // CSR should be sorted
            for(int ai = 0; ai < cast_mat->nrow_; ++ai)
            {
                PtrType first_col = cast_mat->mat_.row_offset[ai];

                for(PtrType ajj = this->mat_.row_offset[ai]; ajj < this->mat_.row_offset[ai + 1];
                    ++ajj)
                {
                    for(PtrType aj = first_col; aj < cast_mat->mat_.row_offset[ai + 1]; ++aj)
                    {
                        if(cast_mat->mat_.col[aj] == this->mat_.col[ajj])
                        {
                            this->mat_.val[ajj]
                                = alpha * this->mat_.val[ajj] + beta * cast_mat->mat_.val[aj];
                            ++first_col;
                            break;
                        }
                    }
                }
            }
        }
        else
        {
            std::vector<PtrType> row_offset;
            std::vector<int>*    new_col = new std::vector<int>[this->nrow_];

            HostMatrixCSR<ValueType> tmp(this->local_backend_);

            tmp.CopyFrom(*this);

            row_offset.resize(this->nrow_ + 1);

            row_offset[0] = 0;

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 0; i < this->nrow_; ++i)
            {
                for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
                {
                    new_col[i].push_back(this->mat_.col[j]);
                }

                for(PtrType k = cast_mat->mat_.row_offset[i]; k < cast_mat->mat_.row_offset[i + 1];
                    ++k)
                {
                    new_col[i].push_back(cast_mat->mat_.col[k]);
                }

                std::sort(new_col[i].begin(), new_col[i].end());
                new_col[i].erase(std::unique(new_col[i].begin(), new_col[i].end()),
                                 new_col[i].end());

                row_offset[i + 1] = static_cast<PtrType>(new_col[i].size());
            }

            for(int i = 0; i < this->nrow_; ++i)
            {
                row_offset[i + 1] += row_offset[i];
            }

            this->AllocateCSR(row_offset[this->nrow_], this->nrow_, this->ncol_);

            // copy structure
            copy_h2h(this->nrow_ + 1, row_offset.data(), this->mat_.row_offset);

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 0; i < this->nrow_; ++i)
            {
                PtrType jj = 0;
                for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
                {
                    this->mat_.col[j] = new_col[i][jj];
                    ++jj;
                }
            }

#ifdef _OPENMP
#pragma omp parallel for
#endif
            // add values
            for(int i = 0; i < this->nrow_; ++i)
            {
                PtrType Aj = tmp.mat_.row_offset[i];
                PtrType Bj = cast_mat->mat_.row_offset[i];

                for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
                {
                    for(PtrType jj = Aj; jj < tmp.mat_.row_offset[i + 1]; ++jj)
                    {
                        if(this->mat_.col[j] == tmp.mat_.col[jj])
                        {
                            this->mat_.val[j] += alpha * tmp.mat_.val[jj];
                            ++Aj;
                            break;
                        }
                    }

                    for(PtrType jj = Bj; jj < cast_mat->mat_.row_offset[i + 1]; ++jj)
                    {
                        if(this->mat_.col[j] == cast_mat->mat_.col[jj])
                        {
                            this->mat_.val[j] += beta * cast_mat->mat_.val[jj];
                            ++Bj;
                            break;
                        }
                    }
                }
            }
            delete[] new_col;
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::Gershgorin(ValueType& lambda_min, ValueType& lambda_max) const
    {
        _set_omp_backend_threads(this->local_backend_, this->nrow_);

        lambda_min = static_cast<ValueType>(0);
        lambda_max = static_cast<ValueType>(0);

        // TODO (parallel max, min)
        //#ifdef _OPENMP
        // #pragma omp parallel for
        //#endif

        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            ValueType sum  = static_cast<ValueType>(0);
            ValueType diag = static_cast<ValueType>(0);

            for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(ai != this->mat_.col[aj])
                {
                    sum += std::abs(this->mat_.val[aj]);
                }
                else
                {
                    diag = this->mat_.val[aj];
                }
            }

            if(sum + diag > lambda_max)
            {
                lambda_max = sum + diag;
            }

            if(diag - sum < lambda_min)
            {
                lambda_min = diag - sum;
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::Scale(ValueType alpha)
    {
        _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int64_t ai = 0; ai < this->nnz_; ++ai)
        {
            this->mat_.val[ai] *= alpha;
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ScaleDiagonal(ValueType alpha)
    {
        _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(ai == this->mat_.col[aj])
                {
                    this->mat_.val[aj] *= alpha;
                    break;
                }
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ScaleOffDiagonal(ValueType alpha)
    {
        _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(ai != this->mat_.col[aj])
                {
                    this->mat_.val[aj] *= alpha;
                }
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::AddScalar(ValueType alpha)
    {
        _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int64_t ai = 0; ai < this->nnz_; ++ai)
        {
            this->mat_.val[ai] += alpha;
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::AddScalarDiagonal(ValueType alpha)
    {
        _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(ai == this->mat_.col[aj])
                {
                    this->mat_.val[aj] += alpha;
                    break;
                }
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::AddScalarOffDiagonal(ValueType alpha)
    {
        _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(ai != this->mat_.col[aj])
                {
                    this->mat_.val[aj] += alpha;
                }
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::DiagonalMatrixMultR(const BaseVector<ValueType>& diag)
    {
        assert(diag.GetSize() == this->ncol_);

        const HostVector<ValueType>* cast_diag = dynamic_cast<const HostVector<ValueType>*>(&diag);
        assert(cast_diag != NULL);

        _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                this->mat_.val[aj] *= cast_diag->vec_[this->mat_.col[aj]];
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::DiagonalMatrixMultL(const BaseVector<ValueType>& diag)
    {
        assert(diag.GetSize() == this->ncol_);

        const HostVector<ValueType>* cast_diag = dynamic_cast<const HostVector<ValueType>*>(&diag);
        assert(cast_diag != NULL);

        _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                this->mat_.val[aj] *= cast_diag->vec_[ai];
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::Compress(double drop_off)
    {
        if(this->nnz_ > 0)
        {
            std::vector<PtrType> row_offset;

            HostMatrixCSR<ValueType> tmp(this->local_backend_);

            tmp.CopyFrom(*this);

            row_offset.resize(this->nrow_ + 1);

            row_offset[0] = 0;

            _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 0; i < this->nrow_; ++i)
            {
                row_offset[i + 1] = 0;

                for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
                {
                    if((std::abs(this->mat_.val[j]) > drop_off) || (this->mat_.col[j] == i))
                    {
                        row_offset[i + 1] += 1;
                    }
                }
            }

            for(int i = 0; i < this->nrow_; ++i)
            {
                row_offset[i + 1] += row_offset[i];
            }

            this->AllocateCSR(row_offset[this->nrow_], this->nrow_, this->ncol_);

            copy_h2h(this->nrow_ + 1, row_offset.data(), this->mat_.row_offset);

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 0; i < this->nrow_; ++i)
            {
                PtrType jj = this->mat_.row_offset[i];

                for(PtrType j = tmp.mat_.row_offset[i]; j < tmp.mat_.row_offset[i + 1]; ++j)
                {
                    if((std::abs(tmp.mat_.val[j]) > drop_off) || (tmp.mat_.col[j] == i))
                    {
                        this->mat_.col[jj] = tmp.mat_.col[j];
                        this->mat_.val[jj] = tmp.mat_.val[j];
                        ++jj;
                    }
                }
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::Transpose(void)
    {
        if(this->nnz_ > 0)
        {
            HostMatrixCSR<ValueType> tmp(this->local_backend_);

            tmp.CopyFrom(*this);
            tmp.Transpose(this);
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::Transpose(BaseMatrix<ValueType>* T) const
    {
        assert(T != NULL);

        HostMatrixCSR<ValueType>* cast_T = dynamic_cast<HostMatrixCSR<ValueType>*>(T);

        assert(cast_T != NULL);

        if(this->nnz_ > 0)
        {
            cast_T->Clear();
            cast_T->AllocateCSR(this->nnz_, this->ncol_, this->nrow_);

            for(int64_t i = 0; i < cast_T->nnz_; ++i)
            {
                cast_T->mat_.row_offset[this->mat_.col[i] + 1] += 1;
            }

            for(int i = 0; i < cast_T->nrow_; ++i)
            {
                cast_T->mat_.row_offset[i + 1] += cast_T->mat_.row_offset[i];
            }

            for(int ai = 0; ai < cast_T->ncol_; ++ai)
            {
                for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1];
                    ++aj)
                {
                    int     ind_col = this->mat_.col[aj];
                    PtrType ind     = cast_T->mat_.row_offset[ind_col];

                    cast_T->mat_.col[ind] = ai;
                    cast_T->mat_.val[ind] = this->mat_.val[aj];

                    cast_T->mat_.row_offset[ind_col] += 1;
                }
            }

            PtrType shift = 0;
            for(int i = 0; i < cast_T->nrow_; ++i)
            {
                PtrType tmp                = cast_T->mat_.row_offset[i];
                cast_T->mat_.row_offset[i] = shift;
                shift                      = tmp;
            }

            cast_T->mat_.row_offset[cast_T->nrow_] = shift;

            assert(this->nnz_ == shift);
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::Sort(void)
    {
        if(this->nnz_ > 0)
        {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 0; i < this->nrow_; ++i)
            {
                for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
                {
                    for(PtrType jj = this->mat_.row_offset[i];
                        jj < this->mat_.row_offset[i + 1] - 1;
                        ++jj)
                    {
                        if(this->mat_.col[jj] > this->mat_.col[jj + 1])
                        {
                            // swap elements
                            int       ind = this->mat_.col[jj];
                            ValueType val = this->mat_.val[jj];

                            this->mat_.col[jj] = this->mat_.col[jj + 1];
                            this->mat_.val[jj] = this->mat_.val[jj + 1];

                            this->mat_.col[jj + 1] = ind;
                            this->mat_.val[jj + 1] = val;
                        }
                    }
                }
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::Permute(const BaseVector<int>& permutation)
    {
        assert((permutation.GetSize() == this->nrow_) && (permutation.GetSize() == this->ncol_));

        if(this->nnz_ > 0)
        {
            const HostVector<int>* cast_perm = dynamic_cast<const HostVector<int>*>(&permutation);
            assert(cast_perm != NULL);

            _set_omp_backend_threads(this->local_backend_, this->nrow_);

            // Calculate nnz per row
            int* row_nnz = NULL;
            allocate_host(this->nrow_, &row_nnz);

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 0; i < this->nrow_; ++i)
            {
                row_nnz[i]
                    = static_cast<int>(this->mat_.row_offset[i + 1] - this->mat_.row_offset[i]);
            }

            // Permute vector of nnz per row
            int* perm_row_nnz = NULL;
            allocate_host(this->nrow_, &perm_row_nnz);

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 0; i < this->nrow_; ++i)
            {
                perm_row_nnz[cast_perm->vec_[i]] = row_nnz[i];
            }

            // Calculate new nnz
            PtrType* perm_nnz = NULL;
            allocate_host(this->nrow_ + 1, &perm_nnz);
            PtrType sum = 0;

            for(int i = 0; i < this->nrow_; ++i)
            {
                perm_nnz[i] = sum;
                sum += perm_row_nnz[i];
            }

            perm_nnz[this->nrow_] = sum;

            // Permute rows
            int*       col = NULL;
            ValueType* val = NULL;
            allocate_host(this->nnz_, &col);
            allocate_host(this->nnz_, &val);

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 0; i < this->nrow_; ++i)
            {
                PtrType permIndex = perm_nnz[cast_perm->vec_[i]];
                PtrType prevIndex = this->mat_.row_offset[i];

                for(int j = 0; j < row_nnz[i]; ++j)
                {
                    col[permIndex + j] = this->mat_.col[prevIndex + j];
                    val[permIndex + j] = this->mat_.val[prevIndex + j];
                }
            }

// Permute columns
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 0; i < this->nrow_; ++i)
            {
                PtrType row_index = perm_nnz[i];

                for(int j = 0; j < perm_row_nnz[i]; ++j)
                {
                    int k     = j - 1;
                    int aComp = col[row_index + j];
                    int comp  = cast_perm->vec_[aComp];
                    for(; k >= 0; --k)
                    {
                        if(this->mat_.col[row_index + k] > comp)
                        {
                            this->mat_.val[row_index + k + 1] = this->mat_.val[row_index + k];
                            this->mat_.col[row_index + k + 1] = this->mat_.col[row_index + k];
                        }
                        else
                        {
                            break;
                        }
                    }

                    this->mat_.val[row_index + k + 1] = val[row_index + j];
                    this->mat_.col[row_index + k + 1] = comp;
                }
            }

            free_host(&this->mat_.row_offset);
            this->mat_.row_offset = perm_nnz;
            free_host(&col);
            free_host(&val);
            free_host(&row_nnz);
            free_host(&perm_row_nnz);
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::CMK(BaseVector<int>* permutation) const
    {
        assert(this->nnz_ > 0);
        assert(permutation != NULL);

        HostVector<int>* cast_perm = dynamic_cast<HostVector<int>*>(permutation);
        assert(cast_perm != NULL);

        cast_perm->Clear();
        cast_perm->Allocate(this->nrow_);

        int next   = 0;
        int head   = 0;
        int tmp    = 0;
        int test   = 1;
        int maxdeg = 0;

        int* nd         = NULL;
        int* marker     = NULL;
        int* levset     = NULL;
        int* nextlevset = NULL;

        allocate_host(this->nrow_, &nd);
        allocate_host(this->nrow_, &marker);
        allocate_host(this->nrow_, &levset);
        allocate_host(this->nrow_, &nextlevset);

        int qlength = 1;

        for(int k = 0; k < this->nrow_; ++k)
        {
            marker[k] = 0;
            nd[k] = static_cast<int>(this->mat_.row_offset[k + 1] - this->mat_.row_offset[k] - 1);

            if(nd[k] > maxdeg)
            {
                maxdeg = nd[k];
            }
        }

        head               = this->mat_.col[0];
        levset[0]          = head;
        cast_perm->vec_[0] = 0;
        ++next;
        marker[head] = 1;

        while(next < this->nrow_)
        {
            int position = 0;

            for(int h = 0; h < qlength; ++h)
            {
                head = levset[h];

                for(PtrType k = this->mat_.row_offset[head]; k < this->mat_.row_offset[head + 1];
                    ++k)
                {
                    tmp = this->mat_.col[k];

                    if((marker[tmp] == 0) && (tmp != head))
                    {
                        nextlevset[position] = tmp;
                        marker[tmp]          = 1;
                        cast_perm->vec_[tmp] = next;
                        ++next;
                        ++position;
                    }
                }
            }

            qlength = position;

            while(test == 1)
            {
                test = 0;

                for(int j = position - 1; j > 0; --j)
                {
                    if(nd[nextlevset[j]] < nd[nextlevset[j - 1]])
                    {
                        tmp               = nextlevset[j];
                        nextlevset[j]     = nextlevset[j - 1];
                        nextlevset[j - 1] = tmp;
                        test              = 1;
                    }
                }
            }

            for(int i = 0; i < position; ++i)
            {
                levset[i] = nextlevset[i];
            }

            if(qlength == 0)
            {
                for(int i = 0; i < this->nrow_; ++i)
                {
                    if(marker[i] == 0)
                    {
                        levset[0]          = i;
                        qlength            = 1;
                        cast_perm->vec_[i] = next;
                        marker[i]          = 1;
                        ++next;
                    }
                }
            }
        }

        free_host(&nd);
        free_host(&marker);
        free_host(&levset);
        free_host(&nextlevset);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::RCMK(BaseVector<int>* permutation) const
    {
        HostVector<int>* cast_perm = dynamic_cast<HostVector<int>*>(permutation);
        assert(cast_perm != NULL);

        cast_perm->Clear();
        cast_perm->Allocate(this->nrow_);

        HostVector<int> tmp_perm(this->local_backend_);

        this->CMK(&tmp_perm);

        for(int i = 0; i < this->nrow_; ++i)
        {
            cast_perm->vec_[i] = this->nrow_ - tmp_perm.vec_[i] - 1;
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ConnectivityOrder(BaseVector<int>* permutation) const
    {
        HostVector<int>* cast_perm = dynamic_cast<HostVector<int>*>(permutation);
        assert(cast_perm != NULL);

        cast_perm->Clear();
        cast_perm->Allocate(this->nrow_);

        std::multimap<int, int> map;

        for(int i = 0; i < this->nrow_; ++i)
        {
            map.insert(
                std::pair<int, int>(this->mat_.row_offset[i + 1] - this->mat_.row_offset[i], i));
        }

        std::multimap<int, int>::iterator it = map.begin();

        for(int i = 0; i < this->nrow_; ++i, ++it)
        {
            cast_perm->vec_[i] = it->second;
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::CreateFromMap(const BaseVector<int>& map, int n, int m)
    {
        assert(map.GetSize() == n);

        const HostVector<int>* cast_map = dynamic_cast<const HostVector<int>*>(&map);

        assert(cast_map != NULL);

        int*     row_nnz    = NULL;
        PtrType* row_buffer = NULL;
        allocate_host(m, &row_nnz);
        allocate_host(m + 1, &row_buffer);

        set_to_zero_host(m, row_nnz);

        int nnz = 0;

        for(int i = 0; i < n; ++i)
        {
            assert(cast_map->vec_[i] < m);

            if(cast_map->vec_[i] < 0)
            {
                continue;
            }

            ++row_nnz[cast_map->vec_[i]];
            ++nnz;
        }

        this->Clear();
        this->AllocateCSR(nnz, m, n);

        this->mat_.row_offset[0] = 0;
        row_buffer[0]            = 0;

        for(int i = 0; i < m; ++i)
        {
            this->mat_.row_offset[i + 1] = this->mat_.row_offset[i] + row_nnz[i];
            row_buffer[i + 1]            = this->mat_.row_offset[i + 1];
        }

        for(int i = 0; i < nnz; ++i)
        {
            if(cast_map->vec_[i] < 0)
            {
                continue;
            }

            this->mat_.col[row_buffer[cast_map->vec_[i]]] = i;
            this->mat_.val[i]                             = static_cast<ValueType>(1);
            row_buffer[cast_map->vec_[i]]++;
        }

        assert(this->mat_.row_offset[m] == nnz);

        free_host(&row_nnz);
        free_host(&row_buffer);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::CreateFromMap(const BaseVector<int>& map,
                                                 int                    n,
                                                 int                    m,
                                                 BaseMatrix<ValueType>* pro)
    {
        assert(map.GetSize() == n);
        assert(pro != NULL);

        const HostVector<int>*    cast_map = dynamic_cast<const HostVector<int>*>(&map);
        HostMatrixCSR<ValueType>* cast_pro = dynamic_cast<HostMatrixCSR<ValueType>*>(pro);

        assert(cast_pro != NULL);
        assert(cast_map != NULL);

        // Build restriction operator
        this->CreateFromMap(map, n, m);

        // Build prolongation operator
        cast_pro->Clear();
        cast_pro->AllocateCSR(this->nnz_, n, m);

        int k = 0;

        for(int i = 0; i < n; ++i)
        {
            cast_pro->mat_.row_offset[i + 1] = cast_pro->mat_.row_offset[i];

            // Check for entry in i-th row
            if(cast_map->vec_[i] < 0)
            {
                continue;
            }

            assert(cast_map->vec_[i] < m);

            ++cast_pro->mat_.row_offset[i + 1];
            cast_pro->mat_.col[k] = cast_map->vec_[i];
            cast_pro->mat_.val[k] = static_cast<ValueType>(1);
            ++k;
        }

        return true;
    }

    // ----------------------------------------------------------
    // original function connect(const spmat &A,
    //                           float eps_strong)
    // ----------------------------------------------------------
    // Modified and adapted from AMGCL,
    // https://github.com/ddemidov/amgcl
    // MIT License
    // ----------------------------------------------------------
    // CHANGELOG
    // - adapted interface
    // ----------------------------------------------------------
    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::AMGConnect(ValueType eps, BaseVector<int>* connections) const
    {
        assert(connections != NULL);

        HostVector<int>* cast_conn = dynamic_cast<HostVector<int>*>(connections);
        assert(cast_conn != NULL);

        cast_conn->Clear();
        cast_conn->Allocate(this->nnz_);

        ValueType eps2 = eps * eps;

        HostVector<ValueType> vec_diag(this->local_backend_);
        vec_diag.Allocate(this->nrow_);
        this->ExtractDiagonal(&vec_diag);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < this->nrow_; ++i)
        {
            ValueType eps_dia_i = eps2 * vec_diag.vec_[i];

            for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                int       c = this->mat_.col[j];
                ValueType v = this->mat_.val[j];

                cast_conn->vec_[j] = (c != i) && (v * v > eps_dia_i * vec_diag.vec_[c]);
            }
        }

        return true;
    }

    // ----------------------------------------------------------
    // original function aggregates( const spmat &A,
    //                               const std::vector<char> &S )
    // ----------------------------------------------------------
    // Modified and adapted from AMGCL,
    // https://github.com/ddemidov/amgcl
    // MIT License
    // ----------------------------------------------------------
    // CHANGELOG
    // - adapted interface
    // ----------------------------------------------------------
    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::AMGAggregate(const BaseVector<int>& connections,
                                                BaseVector<int>*       aggregates) const
    {
        assert(aggregates != NULL);

        HostVector<int>*       cast_agg  = dynamic_cast<HostVector<int>*>(aggregates);
        const HostVector<int>* cast_conn = dynamic_cast<const HostVector<int>*>(&connections);

        assert(cast_agg != NULL);
        assert(cast_conn != NULL);

        aggregates->Clear();
        aggregates->Allocate(this->nrow_);

        const int undefined = -1;
        const int removed   = -2;

        // Remove nodes without neighbours
        int max_neib = 0;
        for(int i = 0; i < this->nrow_; ++i)
        {
            PtrType j = this->mat_.row_offset[i];
            PtrType e = this->mat_.row_offset[i + 1];
            max_neib  = std::max(static_cast<int>(e - j), max_neib);
            int state = removed;
            for(; j < e; ++j)
            {
                if(cast_conn->vec_[j])
                {
                    state = undefined;
                    break;
                }
            }
            cast_agg->vec_[i] = state;
        }
        std::vector<int> neib;
        neib.reserve(max_neib);

        int last_g = -1;

        // Perform plain aggregation
        for(int i = 0; i < this->nrow_; ++i)
        {
            if(cast_agg->vec_[i] != undefined)
            {
                continue;
            }

            // The point is not adjacent to a core of any previous aggregate:
            // so its a seed of a new aggregate.
            cast_agg->vec_[i] = ++last_g;

            neib.clear();

            // Include its neighbors as well.
            for(PtrType j = this->mat_.row_offset[i], e = this->mat_.row_offset[i + 1]; j < e; ++j)
            {
                int c = this->mat_.col[j];
                if(cast_conn->vec_[j] && cast_agg->vec_[c] != removed)
                {
                    cast_agg->vec_[c] = last_g;
                    neib.push_back(c);
                }
            }
            // Temporarily mark undefined points adjacent to the new aggregate as
            // belonging to the aggregate. If nobody claims them later, they will
            // stay here.
            for(typename std::vector<int>::const_iterator nb = neib.begin(); nb != neib.end(); ++nb)
            {
                for(PtrType j = this->mat_.row_offset[*nb], e = this->mat_.row_offset[*nb + 1];
                    j < e;
                    ++j)
                {
                    if(cast_conn->vec_[j] && cast_agg->vec_[this->mat_.col[j]] == undefined)
                    {
                        cast_agg->vec_[this->mat_.col[j]] = last_g;
                    }
                }
            }
        }
        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::AMGPMISAggregate(const BaseVector<int>& connections,
                                                    BaseVector<int>*       aggregates) const
    {
        assert(aggregates != NULL);

        HostVector<int>*       cast_agg  = dynamic_cast<HostVector<int>*>(aggregates);
        const HostVector<int>* cast_conn = dynamic_cast<const HostVector<int>*>(&connections);

        assert(cast_agg != NULL);
        assert(cast_conn != NULL);

        aggregates->Clear();
        aggregates->Allocate(this->nrow_);

        std::vector<mis_tuple> tuples(this->nrow_);
        std::vector<mis_tuple> max_tuples(this->nrow_);

        // Initialize tuples
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < this->nrow_; ++i)
        {
            int state = -2;

            PtrType row_start = this->mat_.row_offset[i];
            PtrType row_end   = this->mat_.row_offset[i + 1];
            for(PtrType j = row_start; j < row_end; j++)
            {
                if(cast_conn->vec_[j] == 1)
                {
                    state = 0;
                    break;
                }
            }

            unsigned int hash = i;
            hash              = ((hash >> 16) ^ hash) * 0x45d9f3b;
            hash              = ((hash >> 16) ^ hash) * 0x45d9f3b;
            hash              = (hash >> 16) ^ hash;

            tuples[i].s = state;
            tuples[i].v = hash;
            tuples[i].i = i;
        }

        bool done = false;
        int  iter = 0;
        while(!done)
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(int i = 0; i < this->nrow_; ++i)
            {
                max_tuples[i] = tuples[i];
            }

            for(int k = 0; k < 2; k++)
            {
                // Find max tuples
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
                for(int i = 0; i < this->nrow_; ++i)
                {
                    mis_tuple t_max = max_tuples[i];

                    PtrType row_start = this->mat_.row_offset[t_max.i];
                    PtrType row_end   = this->mat_.row_offset[t_max.i + 1];
                    for(PtrType j = row_start; j < row_end; j++)
                    {
                        if(cast_conn->vec_[j] == 1)
                        {
                            int       c  = this->mat_.col[j];
                            mis_tuple tj = tuples[c];

                            // find lexographical maximum
                            if(tj.s > t_max.s)
                            {
                                t_max = tj;
                            }
                            else if(tj.s == t_max.s && (tj.v > t_max.v))
                            {
                                t_max = tj;
                            }
                        }
                    }

                    max_tuples[i] = t_max;
                }
            }

            done = true;

            // Update tuples
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(int i = 0; i < this->nrow_; ++i)
            {
                if(tuples[i].s == 0)
                {
                    mis_tuple t_max = max_tuples[i];

                    if(t_max.i == i)
                    {
                        tuples[i].s       = 1;
                        cast_agg->vec_[i] = 1;
                    }
                    else if(t_max.s == 1)
                    {
                        tuples[i].s       = -1;
                        cast_agg->vec_[i] = 0;
                    }
                    else
                    {
                        done = false;
                    }
                }
            }

            iter++;

            if(iter > 10)
            {
                LOG_VERBOSE_INFO(
                    2,
                    "*** warning: HostMatrixCSR::AMGPMISAggregate() Current number of iterations: "
                        << iter);
            }
        }

        // exclusive scan on aggregates array
        int sum = 0;
        for(int i = 0; i < this->nrow_; ++i)
        {
            int temp          = cast_agg->vec_[i];
            cast_agg->vec_[i] = sum;
            sum += temp;
        }

        for(int k = 0; k < 2; k++)
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(int i = 0; i < this->nrow_; ++i)
            {
                max_tuples[i] = tuples[i];
            }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
            for(int i = 0; i < this->nrow_; ++i)
            {
                mis_tuple t = max_tuples[i];

                assert(t.s != 0);

                if(t.s == -1)
                {
                    PtrType row_start = this->mat_.row_offset[i];
                    PtrType row_end   = this->mat_.row_offset[i + 1];

                    for(PtrType j = row_start; j < row_end; j++)
                    {
                        if(cast_conn->vec_[j] == 1)
                        {
                            int c = this->mat_.col[j];

                            if(max_tuples[c].s == 1)
                            {
                                cast_agg->vec_[i] = cast_agg->vec_[c];
                                tuples[i].s       = 1;
                                break;
                            }
                        }
                    }
                }
                else if(t.s == -2)
                {
                    cast_agg->vec_[i] = -2;
                }
            }
        }

        return true;
    }

    // ----------------------------------------------------------
    // original function interp(const sparse::matrix<value_t,
    //                          index_t> &A, const params &prm)
    // ----------------------------------------------------------
    // Modified and adapted from AMGCL,
    // https://github.com/ddemidov/amgcl
    // MIT License
    // ----------------------------------------------------------
    // CHANGELOG
    // - adapted interface
    // ----------------------------------------------------------
    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::AMGSmoothedAggregation(ValueType              relax,
                                                          const BaseVector<int>& aggregates,
                                                          const BaseVector<int>& connections,
                                                          BaseMatrix<ValueType>* prolong,
                                                          int lumping_strat) const
    {
        assert(prolong != NULL);

        const HostVector<int>*    cast_agg     = dynamic_cast<const HostVector<int>*>(&aggregates);
        const HostVector<int>*    cast_conn    = dynamic_cast<const HostVector<int>*>(&connections);
        HostMatrixCSR<ValueType>* cast_prolong = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong);

        assert(cast_agg != NULL);
        assert(cast_conn != NULL);
        assert(cast_prolong != NULL);

        // Allocate
        cast_prolong->Clear();
        cast_prolong->AllocateCSR(this->nnz_, this->nrow_, this->ncol_);

        int ncol = 0;

        for(int i = 0; i < cast_agg->GetSize(); ++i)
        {
            if(cast_agg->vec_[i] > ncol)
            {
                ncol = cast_agg->vec_[i];
            }
        }

        ++ncol;

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            std::vector<PtrType> marker(ncol, -1);

#ifdef _OPENMP
            int nt  = omp_get_num_threads();
            int tid = omp_get_thread_num();

            int chunk_size  = (this->nrow_ + nt - 1) / nt;
            int chunk_start = tid * chunk_size;
            int chunk_end   = std::min(this->nrow_, chunk_start + chunk_size);
#else
            int chunk_start = 0;
            int chunk_end   = this->nrow_;
#endif

            // Count number of entries in prolong
            for(int i = chunk_start; i < chunk_end; ++i)
            {
                for(PtrType j = this->mat_.row_offset[i], e = this->mat_.row_offset[i + 1]; j < e;
                    ++j)
                {
                    int c = this->mat_.col[j];

                    if(c != i && !cast_conn->vec_[j])
                    {
                        continue;
                    }

                    int g = cast_agg->vec_[c];

                    if(g >= 0 && marker[g] != i)
                    {
                        marker[g] = i;
                        ++cast_prolong->mat_.row_offset[i + 1];
                    }
                }
            }

            std::fill(marker.begin(), marker.end(), -1);

#ifdef _OPENMP
#pragma omp barrier
#pragma omp single
#endif
            {
                PtrType* row_offset = NULL;
                allocate_host(cast_prolong->nrow_ + 1, &row_offset);

                int*       col = NULL;
                ValueType* val = NULL;

                int64_t nnz   = 0;
                int     nrow  = cast_prolong->nrow_;
                row_offset[0] = 0;
                for(int i = 1; i < nrow + 1; ++i)
                {
                    row_offset[i] = cast_prolong->mat_.row_offset[i] + row_offset[i - 1];
                }

                nnz = row_offset[nrow];

                allocate_host(nnz, &col);
                allocate_host(nnz, &val);

                cast_prolong->Clear();
                cast_prolong->SetDataPtrCSR(&row_offset, &col, &val, nnz, nrow, ncol);
            }

            // Fill the interpolation matrix.
            for(int i = chunk_start; i < chunk_end; ++i)
            {
                // Diagonal of the filtered matrix is original matrix diagonal plus (lumping_strat = 0) or
                // minus (lumping_strat = 1) its weak connections.
                ValueType dia = static_cast<ValueType>(0);
                for(PtrType j = this->mat_.row_offset[i], e = this->mat_.row_offset[i + 1]; j < e;
                    ++j)
                {
                    if(this->mat_.col[j] == i)
                    {
                        dia += this->mat_.val[j];
                    }
                    else if(!cast_conn->vec_[j])
                    {
                        if(lumping_strat == 0)
                        {
                            dia += this->mat_.val[j];
                        }
                        else
                        {
                            dia -= this->mat_.val[j];
                        }
                    }
                }

                dia = static_cast<ValueType>(1) / dia;

                PtrType row_begin = cast_prolong->mat_.row_offset[i];
                PtrType row_end   = row_begin;

                for(PtrType j = this->mat_.row_offset[i], e = this->mat_.row_offset[i + 1]; j < e;
                    ++j)
                {
                    int c = this->mat_.col[j];

                    // Skip weak couplings, ...
                    if(c != i && !cast_conn->vec_[j])
                    {
                        continue;
                    }

                    // ... and the ones not in any aggregate.
                    int g = cast_agg->vec_[c];
                    if(g < 0)
                    {
                        continue;
                    }

                    ValueType v = (c == i) ? static_cast<ValueType>(1) - relax
                                           : -relax * dia * this->mat_.val[j];

                    if(marker[g] < row_begin)
                    {
                        marker[g]                       = row_end;
                        cast_prolong->mat_.col[row_end] = g;
                        cast_prolong->mat_.val[row_end] = v;
                        ++row_end;
                    }
                    else
                    {
                        cast_prolong->mat_.val[marker[g]] += v;
                    }
                }
            }
        }

        cast_prolong->Sort();

        return true;
    }

    // ----------------------------------------------------------
    // original function interp(const sparse::matrix<value_t,
    //                          index_t> &A, const params &prm)
    // ----------------------------------------------------------
    // Modified and adapted from AMGCL,
    // https://github.com/ddemidov/amgcl
    // MIT License
    // ----------------------------------------------------------
    // CHANGELOG
    // - adapted interface
    // ----------------------------------------------------------
    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::AMGAggregation(const BaseVector<int>& aggregates,
                                                  BaseMatrix<ValueType>* prolong) const
    {
        assert(prolong != NULL);

        const HostVector<int>*    cast_agg     = dynamic_cast<const HostVector<int>*>(&aggregates);
        HostMatrixCSR<ValueType>* cast_prolong = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong);

        assert(cast_agg != NULL);
        assert(cast_prolong != NULL);

        int ncol = 0;

        for(int i = 0; i < cast_agg->GetSize(); ++i)
        {
            if(cast_agg->vec_[i] > ncol)
            {
                ncol = cast_agg->vec_[i];
            }
        }

        ++ncol;

        PtrType* row_offset = NULL;
        allocate_host(this->nrow_ + 1, &row_offset);

        int*       col = NULL;
        ValueType* val = NULL;

        row_offset[0] = 0;
        for(int i = 0; i < this->nrow_; ++i)
        {
            if(cast_agg->vec_[i] >= 0)
            {
                row_offset[i + 1] = row_offset[i] + 1;
            }
            else
            {
                row_offset[i + 1] = row_offset[i];
            }
        }

        allocate_host(row_offset[this->nrow_], &col);
        allocate_host(row_offset[this->nrow_], &val);

        for(int i = 0, j = 0; i < this->nrow_; ++i)
        {
            if(cast_agg->vec_[i] >= 0)
            {
                col[j] = cast_agg->vec_[i];
                val[j] = 1.0;
                ++j;
            }
        }

        cast_prolong->Clear();
        cast_prolong->SetDataPtrCSR(
            &row_offset, &col, &val, row_offset[this->nrow_], this->nrow_, ncol);

        return true;
    }

    // ----------------------------------------------------------
    // original function aggregates( const spmat &A,
    //                               const std::vector<char> &S )
    // ----------------------------------------------------------
    // Modified and adapted from AMGCL,
    // https://github.com/ddemidov/amgcl
    // MIT License
    // ----------------------------------------------------------
    // CHANGELOG
    // - adapted interface
    // ----------------------------------------------------------
    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::AMGGreedyAggregate(
        const BaseVector<bool>& connections,
        BaseVector<int64_t>*    aggregates,
        BaseVector<int64_t>*    aggregate_root_nodes) const
    {
        assert(aggregate_root_nodes != NULL);
        assert(aggregates != NULL);

        HostVector<int64_t>* cast_agg_nodes
            = dynamic_cast<HostVector<int64_t>*>(aggregate_root_nodes);
        HostVector<int64_t>*    cast_agg  = dynamic_cast<HostVector<int64_t>*>(aggregates);
        const HostVector<bool>* cast_conn = dynamic_cast<const HostVector<bool>*>(&connections);

        assert(cast_agg_nodes != NULL);
        assert(cast_agg != NULL);
        assert(cast_conn != NULL);

        const int undefined = -1;
        const int removed   = -2;

        // Remove nodes without neighbours
        int max_neib = 0;
        for(int i = 0; i < this->nrow_; ++i)
        {
            PtrType j = this->mat_.row_offset[i];
            PtrType e = this->mat_.row_offset[i + 1];

            max_neib = std::max(static_cast<int>(e - j), max_neib);

            int state = removed;
            for(; j < e; ++j)
            {
                if(cast_conn->vec_[j])
                {
                    state = undefined;
                    break;
                }
            }

            cast_agg->vec_[i] = state;
        }

        std::vector<int> neib;
        neib.reserve(max_neib);

        int64_t last_g = -1;

        // Perform plain aggregation
        for(int i = 0; i < this->nrow_; ++i)
        {
            if(cast_agg->vec_[i] != undefined)
            {
                continue;
            }

            // The point is not adjacent to a core of any previous aggregate:
            // so its a seed of a new aggregate.
            cast_agg->vec_[i]       = ++last_g;
            cast_agg_nodes->vec_[i] = i;

            neib.clear();

            // Include its neighbors as well.
            for(PtrType j = this->mat_.row_offset[i], e = this->mat_.row_offset[i + 1]; j < e; ++j)
            {
                int c = this->mat_.col[j];

                assert(c >= 0);
                assert(c < this->nrow_);

                if(cast_conn->vec_[j] && cast_agg->vec_[c] != removed)
                {
                    cast_agg->vec_[c]       = last_g;
                    cast_agg_nodes->vec_[c] = i;
                    neib.push_back(c);
                }
            }

            // Temporarily mark undefined points adjacent to the new aggregate as
            // belonging to the aggregate. If nobody claims them later, they will
            // stay here.
            for(typename std::vector<int>::const_iterator nb = neib.begin(); nb != neib.end(); ++nb)
            {
                for(PtrType j = this->mat_.row_offset[*nb], e = this->mat_.row_offset[*nb + 1];
                    j < e;
                    ++j)
                {
                    if(cast_conn->vec_[j] && cast_agg->vec_[this->mat_.col[j]] == undefined)
                    {
                        cast_agg->vec_[this->mat_.col[j]]       = last_g;
                        cast_agg_nodes->vec_[this->mat_.col[j]] = i;
                    }
                }
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::AMGBoundaryNnz(const BaseVector<int>&       boundary,
                                                  const BaseVector<bool>&      connections,
                                                  const BaseMatrix<ValueType>& ghost,
                                                  BaseVector<PtrType>*         row_nnz) const
    {
        const HostVector<int>*  cast_bnd  = dynamic_cast<const HostVector<int>*>(&boundary);
        const HostVector<bool>* cast_conn = dynamic_cast<const HostVector<bool>*>(&connections);
        const HostMatrixCSR<ValueType>* cast_gst
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&ghost);
        HostVector<PtrType>* cast_row_nnz = dynamic_cast<HostVector<PtrType>*>(row_nnz);

        assert(cast_bnd != NULL);
        assert(cast_conn != NULL);
        assert(cast_gst != NULL);
        assert(cast_row_nnz != NULL);

        assert(cast_row_nnz->size_ >= cast_bnd->size_);

        bool global = cast_gst->nrow_ > 0;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int64_t i = 0; i < cast_bnd->size_; ++i)
        {
            // Get boundary row
            int row = cast_bnd->vec_[i];

            // Counter
            PtrType ext_nnz = 0;

            // Interior part
            PtrType row_begin = this->mat_.row_offset[row];
            PtrType row_end   = this->mat_.row_offset[row + 1];

            for(PtrType j = row_begin; j < row_end; ++j)
            {
                // Check whether this vertex is strongly connected
                if(cast_conn->vec_[j] == true)
                {
                    ++ext_nnz;
                }
            }

            if(global == true)
            {
                // Ghost part
                row_begin = cast_gst->mat_.row_offset[row];
                row_end   = cast_gst->mat_.row_offset[row + 1];

                for(PtrType j = row_begin; j < row_end; ++j)
                {
                    // Check whether this vertex is strongly connected
                    if(cast_conn->vec_[j + this->nnz_] == true)
                    {
                        ++ext_nnz;
                    }
                }
            }

            // Write total number of strongly connected coarse vertices to global memory
            cast_row_nnz->vec_[i] = ext_nnz;
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::AMGExtractBoundary(int64_t                global_column_begin,
                                                      const BaseVector<int>& boundary,
                                                      const BaseVector<int64_t>&   l2g,
                                                      const BaseVector<bool>&      connections,
                                                      const BaseMatrix<ValueType>& ghost,
                                                      const BaseVector<PtrType>&   bnd_csr_row_ptr,
                                                      BaseVector<int64_t>* bnd_csr_col_ind) const
    {
        const HostVector<int>*     cast_bnd  = dynamic_cast<const HostVector<int>*>(&boundary);
        const HostVector<int64_t>* cast_l2g  = dynamic_cast<const HostVector<int64_t>*>(&l2g);
        const HostVector<bool>*    cast_conn = dynamic_cast<const HostVector<bool>*>(&connections);
        const HostMatrixCSR<ValueType>* cast_gst
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&ghost);
        const HostVector<PtrType>* cast_bnd_ptr
            = dynamic_cast<const HostVector<PtrType>*>(&bnd_csr_row_ptr);
        HostVector<int64_t>* cast_bnd_col = dynamic_cast<HostVector<int64_t>*>(bnd_csr_col_ind);

        assert(cast_bnd != NULL);
        assert(cast_l2g != NULL);
        assert(cast_conn != NULL);
        assert(cast_gst != NULL);
        assert(cast_bnd_ptr != NULL);
        assert(cast_bnd_col != NULL);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int64_t i = 0; i < cast_bnd->size_; ++i)
        {
            // Get boundary row and index into A_ext
            int     row  = cast_bnd->vec_[i];
            PtrType idx  = cast_bnd_ptr->vec_[i];
            PtrType idx2 = cast_bnd_ptr->vec_[i + 1];

            // Extract interior part
            PtrType row_begin = this->mat_.row_offset[row];
            PtrType row_end   = this->mat_.row_offset[row + 1];

            for(PtrType j = row_begin; j < row_end; ++j)
            {
                // Check whether this vertex is strongly connected
                if(cast_conn->vec_[j] == true)
                {
                    // Get column index
                    int col = this->mat_.col[j];

                    assert(col >= 0);
                    assert(col < this->nrow_);

                    // Shift column by global column offset, to obtain the global column index
                    cast_bnd_col->vec_[idx++] = col + global_column_begin;
                }
            }

            // Extract ghost part
            row_begin = cast_gst->mat_.row_offset[row];
            row_end   = cast_gst->mat_.row_offset[row + 1];

            for(PtrType j = row_begin; j < row_end; ++j)
            {
                // Check whether this vertex is strongly connected
                if(cast_conn->vec_[j + this->nnz_] == true)
                {
                    // Get column index
                    int col = cast_gst->mat_.col[j];

                    // Transform local ghost index into global column index
                    cast_bnd_col->vec_[idx++] = cast_l2g->vec_[col];
                }
            }

            assert(idx2 == idx);
        }

        return true;
    }

    // ----------------------------------------------------------
    // original function connect(const spmat &A,
    //                           float eps_strong)
    // ----------------------------------------------------------
    // Modified and adapted from AMGCL,
    // https://github.com/ddemidov/amgcl
    // MIT License
    // ----------------------------------------------------------
    // CHANGELOG
    // - adapted interface
    // ----------------------------------------------------------
    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::AMGComputeStrongConnections(
        ValueType                    eps,
        const BaseVector<ValueType>& diag,
        const BaseVector<int64_t>&   l2g,
        BaseVector<bool>*            connections,
        const BaseMatrix<ValueType>& ghost) const
    {
        assert(connections != NULL);

        const HostVector<ValueType>* cast_diag = dynamic_cast<const HostVector<ValueType>*>(&diag);
        const HostVector<int64_t>*   cast_l2g  = dynamic_cast<const HostVector<int64_t>*>(&l2g);
        HostVector<bool>*            cast_conn = dynamic_cast<HostVector<bool>*>(connections);
        const HostMatrixCSR<ValueType>* cast_gst
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&ghost);

        assert(cast_diag != NULL);
        assert(cast_l2g != NULL);
        assert(cast_conn != NULL);
        assert(cast_gst != NULL);

        bool global = cast_gst->nrow_ > 0;

        ValueType eps2 = eps * eps;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < this->nrow_; ++i)
        {
            ValueType eps_dia_i = eps2 * cast_diag->vec_[i];

            PtrType row_begin = this->mat_.row_offset[i];
            PtrType row_end   = this->mat_.row_offset[i + 1];

            for(PtrType j = row_begin; j < row_end; ++j)
            {
                int       c = this->mat_.col[j];
                ValueType v = this->mat_.val[j];

                assert(c >= 0);
                assert(c < this->nrow_);

                cast_conn->vec_[j] = (c != i) && (v * v > eps_dia_i * cast_diag->vec_[c]);
            }

            if(global == true)
            {
                PtrType gst_row_begin = cast_gst->mat_.row_offset[i];
                PtrType gst_row_end   = cast_gst->mat_.row_offset[i + 1];

                for(PtrType j = gst_row_begin; j < gst_row_end; ++j)
                {
                    int       c = cast_gst->mat_.col[j];
                    ValueType v = cast_gst->mat_.val[j];

                    cast_conn->vec_[j + this->nnz_]
                        = (v * v > eps_dia_i * cast_diag->vec_[c + this->nrow_]);
                }
            }
        }

        return true;
    }

    unsigned int hash1(unsigned int x)
    {
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = (x >> 16) ^ x;
        return x / 2;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::AMGPMISInitializeState(int64_t global_column_begin,
                                                          const BaseVector<bool>&      connections,
                                                          BaseVector<int>*             state,
                                                          BaseVector<int>*             hash,
                                                          const BaseMatrix<ValueType>& ghost) const
    {
        assert(state != NULL);
        assert(hash != NULL);

        HostVector<int>*        cast_state = dynamic_cast<HostVector<int>*>(state);
        HostVector<int>*        cast_hash  = dynamic_cast<HostVector<int>*>(hash);
        const HostVector<bool>* cast_conn  = dynamic_cast<const HostVector<bool>*>(&connections);
        const HostMatrixCSR<ValueType>* cast_gst
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&ghost);

        assert(cast_state != NULL);
        assert(cast_hash != NULL);
        assert(cast_conn != NULL);
        assert(cast_gst != NULL);

        bool global = cast_gst->nrow_ > 0;

        // Initialize state
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < this->nrow_; ++i)
        {
            int s = -2;

            PtrType row_start = this->mat_.row_offset[i];
            PtrType row_end   = this->mat_.row_offset[i + 1];
            for(PtrType j = row_start; j < row_end; j++)
            {
                if(cast_conn->vec_[j] == true)
                {
                    s = 0;
                    break;
                }
            }

            if(global == true)
            {
                PtrType gst_row_begin = cast_gst->mat_.row_offset[i];
                PtrType gst_row_end   = cast_gst->mat_.row_offset[i + 1];

                for(PtrType j = gst_row_begin; j < gst_row_end; ++j)
                {
                    if(cast_conn->vec_[j + this->nnz_] == true)
                    {
                        s = 0;
                        break;
                    }
                }
            }

            cast_state->vec_[i] = s;
            cast_hash->vec_[i]  = hash1(i + global_column_begin);
        }

        return true;
    }

    mis_tuple lexographical_max(mis_tuple* ti, mis_tuple* tj)
    {
        // find lexographical maximum
        if(tj->s > ti->s)
        {
            return *tj;
        }
        else if(tj->s == ti->s)
        {
            if(tj->v > ti->v)
            {
                return *tj;
            }
        }

        return *ti;
    }

    // LCOV_EXCL_START
    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::AMGExtractBoundaryState(
        const BaseVector<PtrType>&   bnd_csr_row_ptr,
        const BaseVector<bool>&      connections,
        const BaseVector<int>&       max_state,
        const BaseVector<int>&       hash,
        BaseVector<int>*             bnd_max_state,
        BaseVector<int>*             bnd_hash,
        int64_t                      global_column_offset,
        const BaseVector<int>&       boundary_index,
        const BaseMatrix<ValueType>& gst) const
    {
        assert(bnd_max_state != NULL);
        assert(bnd_hash != NULL);

        HostVector<int>* cast_bnd_max_state = dynamic_cast<HostVector<int>*>(bnd_max_state);
        HostVector<int>* cast_bnd_hash      = dynamic_cast<HostVector<int>*>(bnd_hash);

        const HostVector<PtrType>* cast_bnd_ptr
            = dynamic_cast<const HostVector<PtrType>*>(&bnd_csr_row_ptr);
        const HostVector<bool>* cast_conn = dynamic_cast<const HostVector<bool>*>(&connections);
        const HostVector<int>*  cast_max_state = dynamic_cast<const HostVector<int>*>(&max_state);
        const HostVector<int>*  cast_hash      = dynamic_cast<const HostVector<int>*>(&hash);
        const HostVector<int>*  cast_bnd = dynamic_cast<const HostVector<int>*>(&boundary_index);
        const HostMatrixCSR<ValueType>* cast_gst
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&gst);

        assert(cast_bnd_ptr != NULL);
        assert(cast_conn != NULL);
        assert(cast_max_state != NULL);
        assert(cast_hash != NULL);
        assert(cast_bnd != NULL);
        assert(cast_gst != NULL);

        for(int64_t i = 0; i < cast_bnd->size_; ++i)
        {
            // This is a boundary row
            int row = cast_bnd->vec_[i];

            // Index into boundary structures
            PtrType idx  = cast_bnd_ptr->vec_[i];
            PtrType idx2 = cast_bnd_ptr->vec_[i + 1];

            // Entry points
            PtrType row_begin = this->mat_.row_offset[row];
            PtrType row_end   = this->mat_.row_offset[row + 1];

            // Interior
            for(PtrType j = row_begin; j < row_end; ++j)
            {
                if(cast_conn->vec_[j] == true)
                {
                    int col = this->mat_.col[j];

                    assert(col >= 0);
                    assert(col < this->nrow_);

                    cast_bnd_max_state->vec_[idx] = cast_max_state->vec_[col];
                    cast_bnd_hash->vec_[idx]      = cast_hash->vec_[col];

                    ++idx;
                }
            }

            // Ghost
            row_begin = cast_gst->mat_.row_offset[row];
            row_end   = cast_gst->mat_.row_offset[row + 1];

            for(PtrType j = row_begin; j < row_end; ++j)
            {
                if(cast_conn->vec_[j + this->nnz_] == true)
                {
                    int col = cast_gst->mat_.col[j];

                    cast_bnd_max_state->vec_[idx] = cast_max_state->vec_[col + this->nrow_];
                    cast_bnd_hash->vec_[idx]      = cast_hash->vec_[col + this->nrow_];

                    ++idx;
                }
            }

            assert(idx2 == idx);
        }

        return true;
    }
    // LCOV_EXCL_STOP

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::AMGPMISFindMaxNeighbourNode(
        int64_t                      global_column_begin,
        int64_t                      global_column_end,
        bool&                        undecided,
        const BaseVector<bool>&      connections,
        const BaseVector<int>&       state,
        const BaseVector<int>&       hash,
        const BaseVector<PtrType>&   bnd_csr_row_ptr,
        const BaseVector<int64_t>&   bnd_csr_col_ind,
        const BaseVector<int>&       bnd_state,
        const BaseVector<int>&       bnd_hash,
        BaseVector<int>*             max_state,
        BaseVector<int64_t>*         aggregates,
        const BaseMatrix<ValueType>& ghost) const
    {
        HostVector<int>*        cast_max_state = dynamic_cast<HostVector<int>*>(max_state);
        HostVector<int64_t>*    cast_agg       = dynamic_cast<HostVector<int64_t>*>(aggregates);
        const HostVector<int>*  cast_hash      = dynamic_cast<const HostVector<int>*>(&hash);
        const HostVector<int>*  cast_state     = dynamic_cast<const HostVector<int>*>(&state);
        const HostVector<bool>* cast_conn = dynamic_cast<const HostVector<bool>*>(&connections);
        const HostMatrixCSR<ValueType>* cast_gst
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&ghost);

        const HostVector<PtrType>* cast_bnd_ptr
            = dynamic_cast<const HostVector<PtrType>*>(&bnd_csr_row_ptr);
        const HostVector<int64_t>* cast_bnd_col
            = dynamic_cast<const HostVector<int64_t>*>(&bnd_csr_col_ind);
        const HostVector<int>* cast_bnd_state = dynamic_cast<const HostVector<int>*>(&bnd_state);
        const HostVector<int>* cast_bnd_hash  = dynamic_cast<const HostVector<int>*>(&bnd_hash);

        assert(cast_max_state != NULL);
        assert(cast_agg != NULL);
        assert(cast_bnd_ptr != NULL);
        assert(cast_bnd_col != NULL);
        assert(cast_bnd_state != NULL);
        assert(cast_bnd_hash != NULL);
        assert(cast_hash != NULL);
        assert(cast_state != NULL);
        assert(cast_conn != NULL);
        assert(cast_gst != NULL);

        bool global = cast_gst->nrow_ > 0;

        // Find distance one max node
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < this->nrow_; ++i)
        {
            mis_tuple t_max;
            t_max.s = cast_state->vec_[i];
            t_max.v = cast_hash->vec_[i];
            t_max.i = i;

            PtrType row_start = this->mat_.row_offset[i];
            PtrType row_end   = this->mat_.row_offset[i + 1];
            for(PtrType j = row_start; j < row_end; j++)
            {
                if(cast_conn->vec_[j] == true)
                {
                    int col = this->mat_.col[j];

                    assert(col >= 0);
                    assert(col < this->nrow_);

                    mis_tuple tj;
                    tj.s = cast_state->vec_[col];
                    tj.v = cast_hash->vec_[col];
                    tj.i = col;

                    t_max = lexographical_max(&tj, &t_max);
                }
            }

            if(global == true)
            {
                PtrType gst_row_begin = cast_gst->mat_.row_offset[i];
                PtrType gst_row_end   = cast_gst->mat_.row_offset[i + 1];

                for(PtrType j = gst_row_begin; j < gst_row_end; ++j)
                {
                    if(cast_conn->vec_[j + this->nnz_] == true)
                    {
                        int col = cast_gst->mat_.col[j];

                        mis_tuple tj;
                        tj.s = cast_state->vec_[this->nrow_ + col];
                        tj.v = cast_hash->vec_[this->nrow_ + col];
                        tj.i = this->nrow_ + col;

                        t_max = lexographical_max(&tj, &t_max);
                    }
                }
            }

            // Find distance two max node
            if(t_max.i < this->nrow_)
            {
                int row = t_max.i;

                assert(row >= 0);
                assert(row < this->nrow_);

                PtrType row_start = this->mat_.row_offset[row];
                PtrType row_end   = this->mat_.row_offset[row + 1];

                for(PtrType j = row_start; j < row_end; j++)
                {
                    if(cast_conn->vec_[j] == true)
                    {
                        int col = this->mat_.col[j];

                        assert(col >= 0);
                        assert(col < this->nrow_);

                        mis_tuple tj;
                        tj.s = cast_state->vec_[col];
                        tj.v = cast_hash->vec_[col];
                        tj.i = col;

                        t_max = lexographical_max(&tj, &t_max);
                    }
                }

                if(global == true)
                {
                    PtrType gst_row_begin = cast_gst->mat_.row_offset[row];
                    PtrType gst_row_end   = cast_gst->mat_.row_offset[row + 1];

                    for(PtrType j = gst_row_begin; j < gst_row_end; ++j)
                    {
                        if(cast_conn->vec_[j + this->nnz_] == true)
                        {
                            int col = cast_gst->mat_.col[j];

                            mis_tuple tj;
                            tj.s = cast_state->vec_[this->nrow_ + col];
                            tj.v = cast_hash->vec_[this->nrow_ + col];
                            tj.i = this->nrow_ + col;

                            t_max = lexographical_max(&tj, &t_max);
                        }
                    }
                }
            }
            else
            {
                PtrType bnd_row_start = cast_bnd_ptr->vec_[t_max.i - this->nrow_];
                PtrType bnd_row_end   = cast_bnd_ptr->vec_[t_max.i - this->nrow_ + 1];

                for(PtrType j = bnd_row_start; j < bnd_row_end; j++)
                {
                    int64_t gcol = cast_bnd_col->vec_[j];

                    // Differentiate between local and ghost column
                    int col = -1;
                    if(gcol >= global_column_begin && gcol < global_column_end)
                    {
                        col = static_cast<int>(gcol - global_column_begin);

                        assert(col >= 0);
                        assert(col < this->nrow_);
                    }

                    mis_tuple tj;
                    tj.s = cast_bnd_state->vec_[j];
                    tj.v = cast_bnd_hash->vec_[j];
                    tj.i = col;

                    t_max = lexographical_max(&tj, &t_max);
                }
            }

            if(cast_state->vec_[i] == 0)
            {
                if(t_max.i == i)
                {
                    cast_max_state->vec_[i] = 1;
                    cast_agg->vec_[i]       = 1;
                }
                else if(t_max.s == 1)
                {
                    cast_max_state->vec_[i] = -1;
                    cast_agg->vec_[i]       = 0;
                }
                else
                {
                    undecided = true;
                }
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::AMGPMISAddUnassignedNodesToAggregations(
        int64_t                      global_column_begin,
        const BaseVector<bool>&      connections,
        const BaseVector<int>&       state,
        const BaseVector<int64_t>&   l2g,
        BaseVector<int>*             max_state,
        BaseVector<int64_t>*         aggregates,
        BaseVector<int64_t>*         aggregate_root_nodes,
        const BaseMatrix<ValueType>& ghost) const
    {
        assert(max_state != NULL);
        assert(aggregates != NULL);

        HostVector<int64_t>* cast_agg = dynamic_cast<HostVector<int64_t>*>(aggregates);
        HostVector<int64_t>* cast_agg_nodes
            = dynamic_cast<HostVector<int64_t>*>(aggregate_root_nodes);
        HostVector<int>* cast_max_state = dynamic_cast<HostVector<int>*>(max_state);

        const HostVector<int64_t>* cast_l2g   = dynamic_cast<const HostVector<int64_t>*>(&l2g);
        const HostVector<int>*     cast_state = dynamic_cast<const HostVector<int>*>(&state);
        const HostVector<bool>*    cast_conn  = dynamic_cast<const HostVector<bool>*>(&connections);
        const HostMatrixCSR<ValueType>* cast_gst
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&ghost);

        assert(cast_agg != NULL);
        assert(cast_max_state != NULL);

        assert(cast_l2g != NULL);
        assert(cast_state != NULL);
        assert(cast_conn != NULL);
        assert(cast_gst != NULL);

        bool global = cast_gst->nrow_ > 0;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < this->nrow_; ++i)
        {
            int s = cast_state->vec_[i];

            if(s == -1)
            {
                PtrType row_begin = this->mat_.row_offset[i];
                PtrType row_end   = this->mat_.row_offset[i + 1];

                int64_t global_col = -1;
                for(PtrType j = row_begin; j < row_end; j++)
                {
                    if(cast_conn->vec_[j] == true)
                    {
                        int c = this->mat_.col[j];

                        assert(c >= 0);
                        assert(c < this->nrow_);

                        if(cast_state->vec_[c] == 1)
                        {
                            cast_agg->vec_[i]       = cast_agg->vec_[c];
                            cast_max_state->vec_[i] = 1;

                            cast_agg_nodes->vec_[i] = cast_agg_nodes->vec_[c];

                            global_col = global_column_begin + c;
                            break;
                        }
                    }
                }

                if(global)
                {
                    PtrType gst_row_begin = cast_gst->mat_.row_offset[i];
                    PtrType gst_row_end   = cast_gst->mat_.row_offset[i + 1];

                    for(PtrType j = gst_row_begin; j < gst_row_end; j++)
                    {
                        if(cast_conn->vec_[j + this->nnz_] == true)
                        {
                            int c = cast_gst->mat_.col[j];

                            if(cast_state->vec_[c + this->nrow_] == 1)
                            {
                                if(global_col == -1
                                   || (global_col >= 0 && cast_l2g->vec_[c] < global_col))
                                {
                                    cast_agg->vec_[i]       = cast_agg->vec_[c + this->nrow_];
                                    cast_max_state->vec_[i] = 1;

                                    cast_agg_nodes->vec_[i] = cast_agg_nodes->vec_[c + this->nrow_];
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            else if(s == -2)
            {
                cast_agg->vec_[i] = -2;
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::AMGPMISInitializeAggregateGlobalIndices(
        int64_t                    global_column_begin,
        const BaseVector<int64_t>& aggregates,
        BaseVector<int64_t>*       aggregate_root_nodes) const
    {
        assert(aggregate_root_nodes != NULL);

        const HostVector<int64_t>* cast_agg = dynamic_cast<const HostVector<int64_t>*>(&aggregates);
        HostVector<int64_t>*       cast_agg_nodes
            = dynamic_cast<HostVector<int64_t>*>(aggregate_root_nodes);

        assert(cast_agg != NULL);
        assert(cast_agg_nodes != NULL);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < this->nrow_; ++i)
        {
            cast_agg_nodes->vec_[i] = (cast_agg->vec_[i] == 1) ? global_column_begin + i : -1;
        }

        return true;
    }

    // ----------------------------------------------------------
    // original function interp(const sparse::matrix<value_t,
    //                          index_t> &A, const params &prm)
    // ----------------------------------------------------------
    // Modified and adapted from AMGCL,
    // https://github.com/ddemidov/amgcl
    // MIT License
    // ----------------------------------------------------------
    // CHANGELOG
    // - adapted interface
    // ----------------------------------------------------------
    /*template <typename ValueType>
    bool HostMatrixCSR<ValueType>::AMGSmoothedAggregation(ValueType                  relax,
                                                          const BaseVector<bool>&    connections,
                                                          const BaseVector<int64_t>& aggregates,
                                                          BaseMatrix<ValueType>*     prolong,
                                                          int lumping_strat) const
    {
        assert(prolong != NULL);

        const HostVector<bool>*    cast_conn = dynamic_cast<const HostVector<bool>*>(&connections);
        const HostVector<int64_t>* cast_agg = dynamic_cast<const HostVector<int64_t>*>(&aggregates);
        HostMatrixCSR<ValueType>*  cast_prolong = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong);

        assert(cast_agg != NULL);
        assert(cast_conn != NULL);
        assert(cast_prolong != NULL);

        // Allocate
        cast_prolong->Clear();
        cast_prolong->AllocateCSR(this->nnz_, this->nrow_, this->ncol_);

        int ncol = 0;

        for(int i = 0; i < cast_agg->GetSize(); ++i)
        {
            if(cast_agg->vec_[i] > ncol)
            {
                ncol = cast_agg->vec_[i];
            }
        }

        ++ncol;

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            std::vector<PtrType> marker(ncol, -1);

#ifdef _OPENMP
            int nt  = omp_get_num_threads();
            int tid = omp_get_thread_num();

            int chunk_size  = (this->nrow_ + nt - 1) / nt;
            int chunk_start = tid * chunk_size;
            int chunk_end   = std::min(this->nrow_, chunk_start + chunk_size);
#else
            int chunk_start = 0;
            int chunk_end   = this->nrow_;
#endif

            // Count number of entries in prolong
            for(int i = chunk_start; i < chunk_end; ++i)
            {
                for(PtrType j = this->mat_.row_offset[i], e = this->mat_.row_offset[i + 1]; j < e;
                    ++j)
                {
                    int c = this->mat_.col[j];

                    assert(c >= 0);
                    assert(c < this->nrow_);

                    if(c != i && !cast_conn->vec_[j])
                    {
                        continue;
                    }

                    int g = cast_agg->vec_[c];

                    if(g >= 0 && marker[g] != i)
                    {
                        marker[g] = i;
                        ++cast_prolong->mat_.row_offset[i + 1];
                    }
                }
            }

            std::fill(marker.begin(), marker.end(), -1);

#ifdef _OPENMP
#pragma omp barrier
#pragma omp single
#endif
            {
                PtrType* row_offset = NULL;
                allocate_host(cast_prolong->nrow_ + 1, &row_offset);

                int*       col = NULL;
                ValueType* val = NULL;

                int64_t nnz  = 0;
                int     nrow = cast_prolong->nrow_;

                row_offset[0] = 0;
                for(int i = 1; i < nrow + 1; ++i)
                {
                    row_offset[i] = cast_prolong->mat_.row_offset[i] + row_offset[i - 1];
                }

                nnz = row_offset[nrow];

                allocate_host(nnz, &col);
                allocate_host(nnz, &val);

                cast_prolong->Clear();
                cast_prolong->SetDataPtrCSR(&row_offset, &col, &val, nnz, nrow, ncol);
            }

            // Fill the interpolation matrix.
            for(int i = chunk_start; i < chunk_end; ++i)
            {
                // Diagonal of the filtered matrix is original matrix diagonal plus (lumping_strat = 0) or
                // minus (lumping_strat = 1) its weak connections.
                ValueType dia = static_cast<ValueType>(0);
                for(PtrType j = this->mat_.row_offset[i], e = this->mat_.row_offset[i + 1]; j < e;
                    ++j)
                {
                    if(this->mat_.col[j] == i)
                    {
                        dia += this->mat_.val[j];
                    }
                    else if(!cast_conn->vec_[j])
                    {
                        if(lumping_strat == 0)
                        {
                            dia += this->mat_.val[j];
                        }
                        else
                        {
                            dia -= this->mat_.val[j];
                        }
                    }
                }

                dia = static_cast<ValueType>(1) / dia;

                PtrType row_begin = cast_prolong->mat_.row_offset[i];
                PtrType row_end   = row_begin;

                for(PtrType j = this->mat_.row_offset[i], e = this->mat_.row_offset[i + 1]; j < e;
                    ++j)
                {
                    int c = this->mat_.col[j];

                    assert(c >= 0);
                    assert(c < this->nrow_);

                    // Skip weak couplings, ...
                    if(c != i && !cast_conn->vec_[j])
                    {
                        continue;
                    }

                    // ... and the ones not in any aggregate.
                    int g = cast_agg->vec_[c];
                    if(g < 0)
                    {
                        continue;
                    }

                    ValueType v = (c == i) ? static_cast<ValueType>(1) - relax
                                           : -relax * dia * this->mat_.val[j];

                    if(marker[g] < row_begin)
                    {
                        marker[g]                       = row_end;
                        cast_prolong->mat_.col[row_end] = g;
                        cast_prolong->mat_.val[row_end] = v;
                        ++row_end;
                    }
                    else
                    {
                        cast_prolong->mat_.val[marker[g]] += v;
                    }
                }
            }
        }

        cast_prolong->Sort();

        return true;
    }*/

    // ----------------------------------------------------------
    // original function interp(const sparse::matrix<value_t,
    //                          index_t> &A, const params &prm)
    // ----------------------------------------------------------
    // Modified and adapted from AMGCL,
    // https://github.com/ddemidov/amgcl
    // MIT License
    // ----------------------------------------------------------
    // CHANGELOG
    // - adapted interface
    // ----------------------------------------------------------
    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::AMGUnsmoothedAggregation(const BaseVector<int64_t>& aggregates,
                                                            BaseMatrix<ValueType>* prolong) const
    {
        assert(prolong != NULL);

        const HostVector<int64_t>* cast_agg = dynamic_cast<const HostVector<int64_t>*>(&aggregates);
        HostMatrixCSR<ValueType>*  cast_prolong = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong);

        assert(cast_agg != NULL);
        assert(cast_prolong != NULL);

        int64_t ncol = 0;

        for(int i = 0; i < cast_agg->GetSize(); ++i)
        {
            if(cast_agg->vec_[i] > ncol)
            {
                ncol = cast_agg->vec_[i];
            }
        }

        ++ncol;

        PtrType* row_offset = NULL;
        allocate_host(this->nrow_ + 1, &row_offset);

        int*       col = NULL;
        ValueType* val = NULL;

        row_offset[0] = 0;
        for(int i = 0; i < this->nrow_; ++i)
        {
            if(cast_agg->vec_[i] >= 0)
            {
                row_offset[i + 1] = row_offset[i] + 1;
            }
            else
            {
                row_offset[i + 1] = row_offset[i];
            }
        }

        allocate_host(row_offset[this->nrow_], &col);
        allocate_host(row_offset[this->nrow_], &val);

        for(int i = 0, j = 0; i < this->nrow_; ++i)
        {
            if(cast_agg->vec_[i] >= 0)
            {
                col[j] = cast_agg->vec_[i];
                val[j] = 1.0;
                ++j;
            }
        }

        cast_prolong->Clear();
        cast_prolong->SetDataPtrCSR(
            &row_offset, &col, &val, row_offset[this->nrow_], this->nrow_, ncol);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::AMGSmoothedAggregationProlongNnz(
        int64_t                      global_column_begin,
        int64_t                      global_column_end,
        const BaseVector<bool>&      connections,
        const BaseVector<int64_t>&   aggregates,
        const BaseVector<int64_t>&   aggregate_root_nodes,
        const BaseMatrix<ValueType>& ghost,
        BaseVector<int>*             f2c,
        BaseMatrix<ValueType>*       prolong_int,
        BaseMatrix<ValueType>*       prolong_gst) const
    {
        const HostVector<bool>*    cast_conn = dynamic_cast<const HostVector<bool>*>(&connections);
        const HostVector<int64_t>* cast_agg = dynamic_cast<const HostVector<int64_t>*>(&aggregates);
        const HostVector<int64_t>* cast_agg_nodes
            = dynamic_cast<const HostVector<int64_t>*>(&aggregate_root_nodes);
        const HostMatrixCSR<ValueType>* cast_gst
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&ghost);
        HostVector<int>*          cast_f2c = dynamic_cast<HostVector<int>*>(f2c);
        HostMatrixCSR<ValueType>* cast_pi  = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong_int);
        HostMatrixCSR<ValueType>* cast_pg  = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong_gst);

        assert(cast_conn != NULL);
        assert(cast_agg != NULL);
        assert(cast_agg_nodes != NULL);
        assert(cast_f2c != NULL);
        assert(cast_pi != NULL);

        // Do we need communication?
        bool global = prolong_gst != NULL;

        // Start with fresh P
        cast_pi->Clear();

        // Allocate P row pointer array
        allocate_host(this->nrow_ + 1, &cast_pi->mat_.row_offset);
        set_to_zero_host(this->nrow_ + 1, cast_pi->mat_.row_offset);

        // We already know the number of rows of P
        cast_pi->nrow_ = this->nrow_;

        // Ghost part
        if(global == true)
        {
            // LCOV_EXCL_START
            assert(cast_gst != NULL);
            assert(cast_pg != NULL);

            // Start with fresh P ghost
            cast_pg->Clear();

            // Allocate P ghost row pointer array
            allocate_host(this->nrow_ + 1, &cast_pg->mat_.row_offset);
            set_to_zero_host(this->nrow_ + 1, cast_pg->mat_.row_offset);

            // Number of ghost rows is identical to interior
            cast_pg->nrow_ = this->nrow_;
            // LCOV_EXCL_STOP
        }

        // Count number of entries in interior prolong
        for(int i = 0; i < this->nrow_; ++i)
        {
            std::unordered_set<int64_t> int_set;
            std::unordered_set<int64_t> gst_set;

            PtrType row_begin = this->mat_.row_offset[i];
            PtrType row_end   = this->mat_.row_offset[i + 1];

            for(PtrType j = row_begin; j < row_end; ++j)
            {
                int c = this->mat_.col[j];

                assert(c >= 0);
                assert(c < this->nrow_);

                if(c != i && !cast_conn->vec_[j])
                {
                    continue;
                }

                int64_t agg = cast_agg->vec_[c];

                if(agg >= 0)
                {
                    int64_t global_node = cast_agg_nodes->vec_[c];
                    assert(global_node != -1);

                    if(global_node >= global_column_begin && global_node < global_column_end)
                    {
                        int_set.insert(global_node);
                        cast_f2c->vec_[global_node - global_column_begin] = 1;
                    }
                    else
                    {
                        gst_set.insert(global_node);
                    }
                }
            }

            if(global == true)
            {
                // LCOV_EXCL_START
                PtrType gst_row_begin = cast_gst->mat_.row_offset[i];
                PtrType gst_row_end   = cast_gst->mat_.row_offset[i + 1];

                for(PtrType j = gst_row_begin; j < gst_row_end; ++j)
                {
                    int c = cast_gst->mat_.col[j];
                    if(!cast_conn->vec_[j + this->nnz_])
                    {
                        continue;
                    }

                    int64_t agg = cast_agg->vec_[c + this->nrow_];

                    if(agg >= 0)
                    {
                        int64_t global_node = cast_agg_nodes->vec_[c + this->nrow_];
                        assert(global_node != -1);

                        if(global_node >= global_column_begin && global_node < global_column_end)
                        {
                            int_set.insert(global_node);
                            cast_f2c->vec_[global_node - global_column_begin] = 1;
                        }
                        else
                        {
                            gst_set.insert(global_node);
                        }
                    }
                }

                cast_pg->mat_.row_offset[i + 1] = gst_set.size();
                // LCOV_EXCL_STOP
            }

            cast_pi->mat_.row_offset[i + 1] = int_set.size();
        }

        cast_f2c->ExclusiveSum(*cast_f2c);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::AMGSmoothedAggregationProlongFill(
        int64_t                      global_column_begin,
        int64_t                      global_column_end,
        int                          lumping_strat,
        ValueType                    relax,
        const BaseVector<bool>&      connections,
        const BaseVector<int64_t>&   aggregates,
        const BaseVector<int64_t>&   aggregate_root_nodes,
        const BaseVector<int64_t>&   l2g,
        const BaseVector<int>&       f2c,
        const BaseMatrix<ValueType>& ghost,
        BaseMatrix<ValueType>*       prolong_int,
        BaseMatrix<ValueType>*       prolong_gst,
        BaseVector<int64_t>*         global_ghost_col) const
    {
        const HostVector<bool>*    cast_conn = dynamic_cast<const HostVector<bool>*>(&connections);
        const HostVector<int64_t>* cast_agg = dynamic_cast<const HostVector<int64_t>*>(&aggregates);
        const HostVector<int64_t>* cast_agg_nodes
            = dynamic_cast<const HostVector<int64_t>*>(&aggregate_root_nodes);
        const HostVector<int64_t>*      cast_l2g = dynamic_cast<const HostVector<int64_t>*>(&l2g);
        const HostVector<int>*          cast_f2c = dynamic_cast<const HostVector<int>*>(&f2c);
        const HostMatrixCSR<ValueType>* cast_gst
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&ghost);
        HostMatrixCSR<ValueType>* cast_pi  = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong_int);
        HostMatrixCSR<ValueType>* cast_pg  = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong_gst);
        HostVector<int64_t>*      cast_glo = dynamic_cast<HostVector<int64_t>*>(global_ghost_col);

        assert(cast_conn != NULL);
        assert(cast_agg != NULL);
        assert(cast_agg_nodes != NULL);
        assert(cast_l2g != NULL);
        assert(cast_f2c != NULL);
        assert(cast_pi != NULL);

        // Do we need communication?
        bool global = prolong_gst != NULL;

        cast_pi->mat_.row_offset[0] = 0;
        for(int i = 0; i < this->nrow_; ++i)
        {
            cast_pi->mat_.row_offset[i + 1] += cast_pi->mat_.row_offset[i];
        }

        // Initialize nnz of P
        cast_pi->nnz_ = cast_pi->mat_.row_offset[this->nrow_];

        // Initialize ncol of P
        cast_pi->ncol_ = cast_f2c->vec_[this->nrow_];

        // Allocate column and value arrays
        allocate_host(cast_pi->nnz_, &cast_pi->mat_.col);
        allocate_host(cast_pi->nnz_, &cast_pi->mat_.val);
        set_to_zero_host(cast_pi->nnz_, cast_pi->mat_.col);
        set_to_zero_host(cast_pi->nnz_, cast_pi->mat_.val);

        // Once again for the ghost part
        if(global == true)
        {
            // LCOV_EXCL_START
            assert(cast_gst != NULL);
            assert(cast_pg != NULL);
            assert(cast_glo != NULL);

            cast_pg->mat_.row_offset[0] = 0;
            for(int i = 0; i < this->nrow_; ++i)
            {
                cast_pg->mat_.row_offset[i + 1] += cast_pg->mat_.row_offset[i];
            }

            // Initialize nnz of P
            cast_pg->nnz_ = cast_pg->mat_.row_offset[this->nrow_];

            // Initialize ncol of P
            cast_pg->ncol_ = this->nrow_; // Need to fix this!!!!

            allocate_host(cast_pg->nnz_, &cast_pg->mat_.col);
            allocate_host(cast_pg->nnz_, &cast_pg->mat_.val);
            set_to_zero_host(cast_pg->nnz_, cast_pg->mat_.col);
            set_to_zero_host(cast_pg->nnz_, cast_pg->mat_.val);

            cast_glo->Allocate(cast_pg->nnz_);
            // LCOV_EXCL_STOP
        }

        // Fill the interpolation matrix.
        for(int i = 0; i < this->nrow_; ++i)
        {
            // Diagonal of the filtered matrix is original matrix diagonal plus (lumping_strat = 0) or
            // minus (lumping_strat = 1) its weak connections.
            ValueType dia = static_cast<ValueType>(0);

            for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                if(this->mat_.col[j] == i)
                {
                    dia += this->mat_.val[j];
                }
                else if(!cast_conn->vec_[j])
                {
                    if(lumping_strat == 0)
                    {
                        dia += this->mat_.val[j];
                    }
                    else
                    {
                        dia -= this->mat_.val[j];
                    }
                }
            }

            if(global == true)
            {
                // LCOV_EXCL_START
                for(PtrType j = cast_gst->mat_.row_offset[i]; j < cast_gst->mat_.row_offset[i + 1];
                    ++j)
                {
                    if(!cast_conn->vec_[j + this->nnz_])
                    {
                        if(lumping_strat == 0)
                        {
                            dia += cast_gst->mat_.val[j];
                        }
                        else
                        {
                            dia -= cast_gst->mat_.val[j];
                        }
                    }
                }
                // LCOV_EXCL_STOP
            }

            dia = static_cast<ValueType>(1) / dia;

            PtrType pi_row_begin = cast_pi->mat_.row_offset[i];
            PtrType pi_row_end   = pi_row_begin;

            // Hash table
            std::map<int64_t, ValueType> int_table;
            std::map<int64_t, ValueType> gst_table;

            for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                int c = this->mat_.col[j];

                assert(c >= 0);
                assert(c < this->nrow_);

                // Skip weak couplings, ...
                if(c != i && !cast_conn->vec_[j])
                {
                    continue;
                }

                // ... and the ones not in any aggregate.
                int64_t agg = cast_agg->vec_[c];
                if(agg >= 0)
                {
                    ValueType v = (c == i) ? static_cast<ValueType>(1) - relax
                                           : -relax * dia * this->mat_.val[j];

                    int64_t global_node = cast_agg_nodes->vec_[c];
                    assert(global_node != -1);

                    if(global_node >= global_column_begin && global_node < global_column_end)
                    {
                        if(int_table.find(global_node) != int_table.end())
                        {
                            int_table[global_node] += v;
                        }
                        else
                        {
                            int_table[global_node] = v;
                        }
                    }
                    else
                    {
                        if(gst_table.find(global_node) != gst_table.end())
                        {
                            gst_table[global_node] += v;
                        }
                        else
                        {
                            gst_table[global_node] = v;
                        }
                    }
                }
            }
            if(global == true)
            {
                // LCOV_EXCL_START
                PtrType pg_row_begin = cast_pg->mat_.row_offset[i];
                PtrType pg_row_end   = pg_row_begin;

                for(PtrType j = cast_gst->mat_.row_offset[i]; j < cast_gst->mat_.row_offset[i + 1];
                    ++j)
                {
                    int c = cast_gst->mat_.col[j];

                    // Skip weak couplings, ...
                    if(!cast_conn->vec_[j + this->nnz_])
                    {
                        continue;
                    }

                    // ... and the ones not in any aggregate.
                    int64_t agg = cast_agg->vec_[c + this->nrow_];

                    if(agg >= 0)
                    {
                        ValueType v = -relax * dia * cast_gst->mat_.val[j];

                        int64_t global_node = cast_agg_nodes->vec_[c + this->nrow_];
                        assert(global_node != -1);

                        if(global_node >= global_column_begin && global_node < global_column_end)
                        {
                            if(int_table.find(global_node) != int_table.end())
                            {
                                int_table[global_node] += v;
                            }
                            else
                            {
                                int_table[global_node] = v;
                            }
                        }
                        else
                        {
                            if(gst_table.find(global_node) != gst_table.end())
                            {
                                gst_table[global_node] += v;
                            }
                            else
                            {
                                gst_table[global_node] = v;
                            }
                        }
                    }
                }

                for(auto it = gst_table.begin(); it != gst_table.end(); it++)
                {
                    cast_pg->mat_.val[pg_row_end] = it->second;
                    cast_glo->vec_[pg_row_end]    = it->first;
                    ++pg_row_end;
                }
                // LCOV_EXCL_STOP
            }

            for(auto it = int_table.begin(); it != int_table.end(); it++)
            {
                cast_pi->mat_.col[pi_row_end] = cast_f2c->vec_[it->first - global_column_begin];
                cast_pi->mat_.val[pi_row_end] = it->second;
                ++pi_row_end;
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::AMGUnsmoothedAggregationProlongNnz(
        int64_t                      global_column_begin,
        int64_t                      global_column_end,
        const BaseVector<int64_t>&   aggregates,
        const BaseVector<int64_t>&   aggregate_root_nodes,
        const BaseMatrix<ValueType>& ghost,
        BaseVector<int>*             f2c,
        BaseMatrix<ValueType>*       prolong_int,
        BaseMatrix<ValueType>*       prolong_gst) const
    {
        const HostVector<int64_t>* cast_agg = dynamic_cast<const HostVector<int64_t>*>(&aggregates);
        const HostVector<int64_t>* cast_agg_nodes
            = dynamic_cast<const HostVector<int64_t>*>(&aggregate_root_nodes);
        const HostMatrixCSR<ValueType>* cast_gst
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&ghost);
        HostVector<int>*          cast_f2c = dynamic_cast<HostVector<int>*>(f2c);
        HostMatrixCSR<ValueType>* cast_pi  = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong_int);
        HostMatrixCSR<ValueType>* cast_pg  = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong_gst);

        assert(cast_agg != NULL);
        assert(cast_agg_nodes != NULL);
        assert(cast_f2c != NULL);
        assert(cast_pi != NULL);

        // Do we need communication?
        bool global = prolong_gst != NULL;

        // Start with fresh P
        cast_pi->Clear();

        // Allocate P row pointer array
        allocate_host(this->nrow_ + 1, &cast_pi->mat_.row_offset);
        set_to_zero_host(this->nrow_ + 1, cast_pi->mat_.row_offset);

        // We already know the number of rows of P
        cast_pi->nrow_ = this->nrow_;

        // Ghost part
        if(global == true)
        {
            // LCOV_EXCL_START
            assert(cast_gst != NULL);
            assert(cast_pg != NULL);

            // Start with fresh P ghost
            cast_pg->Clear();

            // Allocate P ghost row pointer array
            allocate_host(this->nrow_ + 1, &cast_pg->mat_.row_offset);
            set_to_zero_host(this->nrow_ + 1, cast_pg->mat_.row_offset);

            // Number of ghost rows is identical to interior
            cast_pg->nrow_ = this->nrow_;
            // LCOV_EXCL_STOP
        }

        // Count number of entries in interior prolong
        for(int i = 0; i < this->nrow_; ++i)
        {
            int64_t agg = cast_agg->vec_[i];

            if(agg >= 0)
            {
                int64_t global_node = cast_agg_nodes->vec_[i];
                assert(global_node != -1);

                if(global_node >= global_column_begin && global_node < global_column_end)
                {
                    cast_pi->mat_.row_offset[i + 1]                   = 1;
                    cast_f2c->vec_[global_node - global_column_begin] = 1;
                }
                else
                {
                    cast_pg->mat_.row_offset[i + 1] = 1;
                }
            }
        }

        cast_f2c->ExclusiveSum(*cast_f2c);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::AMGUnsmoothedAggregationProlongFill(
        int64_t                      global_column_begin,
        int64_t                      global_column_end,
        const BaseVector<int64_t>&   aggregates,
        const BaseVector<int64_t>&   aggregate_root_nodes,
        const BaseVector<int>&       f2c,
        const BaseMatrix<ValueType>& ghost,
        BaseMatrix<ValueType>*       prolong_int,
        BaseMatrix<ValueType>*       prolong_gst,
        BaseVector<int64_t>*         global_ghost_col) const
    {
        const HostVector<int64_t>* cast_agg = dynamic_cast<const HostVector<int64_t>*>(&aggregates);
        const HostVector<int64_t>* cast_agg_nodes
            = dynamic_cast<const HostVector<int64_t>*>(&aggregate_root_nodes);
        const HostVector<int>*          cast_f2c = dynamic_cast<const HostVector<int>*>(&f2c);
        const HostMatrixCSR<ValueType>* cast_gst
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&ghost);
        HostMatrixCSR<ValueType>* cast_pi  = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong_int);
        HostMatrixCSR<ValueType>* cast_pg  = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong_gst);
        HostVector<int64_t>*      cast_glo = dynamic_cast<HostVector<int64_t>*>(global_ghost_col);

        assert(cast_agg != NULL);
        assert(cast_agg_nodes != NULL);
        assert(cast_f2c != NULL);
        assert(cast_pi != NULL);

        // Do we need communication?
        bool global = prolong_gst != NULL;

        cast_pi->mat_.row_offset[0] = 0;
        for(int i = 0; i < this->nrow_; ++i)
        {
            cast_pi->mat_.row_offset[i + 1] += cast_pi->mat_.row_offset[i];
        }

        // Initialize nnz of P
        cast_pi->nnz_ = cast_pi->mat_.row_offset[this->nrow_];

        // Initialize ncol of P
        cast_pi->ncol_ = cast_f2c->vec_[this->nrow_];

        // Allocate column and value arrays
        allocate_host(cast_pi->nnz_, &cast_pi->mat_.col);
        allocate_host(cast_pi->nnz_, &cast_pi->mat_.val);
        set_to_zero_host(cast_pi->nnz_, cast_pi->mat_.col);
        set_to_zero_host(cast_pi->nnz_, cast_pi->mat_.val);

        // Once again for the ghost part
        if(global == true)
        {
            // LCOV_EXCL_START
            assert(cast_gst != NULL);
            assert(cast_pg != NULL);
            assert(cast_glo != NULL);

            cast_pg->mat_.row_offset[0] = 0;
            for(int i = 0; i < this->nrow_; ++i)
            {
                cast_pg->mat_.row_offset[i + 1] += cast_pg->mat_.row_offset[i];
            }

            // Initialize nnz of P
            cast_pg->nnz_ = cast_pg->mat_.row_offset[this->nrow_];

            // Initialize ncol of P
            cast_pg->ncol_ = this->nrow_; // Need to fix this!!!!

            allocate_host(cast_pg->nnz_, &cast_pg->mat_.col);
            allocate_host(cast_pg->nnz_, &cast_pg->mat_.val);
            set_to_zero_host(cast_pg->nnz_, cast_pg->mat_.col);
            set_to_zero_host(cast_pg->nnz_, cast_pg->mat_.val);

            cast_glo->Allocate(cast_pg->nnz_);
            // LCOV_EXCL_STOP
        }

        // Count number of entries in interior prolong
        for(int i = 0; i < this->nrow_; ++i)
        {
            int64_t agg = cast_agg->vec_[i];

            if(agg >= 0)
            {
                int64_t global_node = cast_agg_nodes->vec_[i];
                assert(global_node != -1);

                if(global_node >= global_column_begin && global_node < global_column_end)
                {
                    PtrType pi_row_end = cast_pi->mat_.row_offset[i];

                    cast_pi->mat_.col[pi_row_end]
                        = cast_f2c->vec_[global_node - global_column_begin];
                    cast_pi->mat_.val[pi_row_end] = 1;
                }
                else
                {
                    PtrType pg_row_end = cast_pg->mat_.row_offset[i];

                    cast_pg->mat_.val[pg_row_end] = 1;
                    cast_glo->vec_[pg_row_end]    = global_node;
                }
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::FSAI(int power, const BaseMatrix<ValueType>* pattern)
    {
        // Extract lower triangular matrix of A
        HostMatrixCSR<ValueType> L(this->local_backend_);

        const HostMatrixCSR<ValueType>* cast_pattern = NULL;
        if(pattern != NULL)
        {
            cast_pattern = dynamic_cast<const HostMatrixCSR<ValueType>*>(pattern);
            assert(cast_pattern != NULL);

            cast_pattern->ExtractLDiagonal(&L);
        }
        else if(power > 1)
        {
            HostMatrixCSR<ValueType> structure(this->local_backend_);
            structure.CopyFrom(*this);
            structure.SymbolicPower(power);
            structure.ExtractLDiagonal(&L);
        }
        else
        {
            this->ExtractLDiagonal(&L);
        }

        int64_t    nnz        = L.nnz_;
        int        nrow       = L.nrow_;
        int        ncol       = L.ncol_;
        PtrType*   row_offset = NULL;
        int*       col        = NULL;
        ValueType* val        = NULL;

        L.LeaveDataPtrCSR(&row_offset, &col, &val);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            // entries of ai-th row
            PtrType nnz_row = row_offset[ai + 1] - row_offset[ai];

            if(nnz_row == 1)
            {
                PtrType aj = this->mat_.row_offset[ai];
                if(this->mat_.col[aj] == ai)
                {
                    val[row_offset[ai]] = static_cast<ValueType>(1) / this->mat_.val[aj];
                }
            }
            else
            {
                // create submatrix taking only the lower tridiagonal part into account
                std::vector<ValueType> Asub(nnz_row * nnz_row, static_cast<ValueType>(0));

                for(PtrType k = 0; k < nnz_row; ++k)
                {
                    PtrType row_begin = this->mat_.row_offset[col[row_offset[ai] + k]];
                    PtrType row_end   = this->mat_.row_offset[col[row_offset[ai] + k] + 1];

                    for(PtrType aj = row_begin; aj < row_end; ++aj)
                    {
                        for(PtrType j = 0; j < nnz_row; ++j)
                        {
                            int Asub_col = col[row_offset[ai] + j];

                            if(this->mat_.col[aj] < Asub_col)
                            {
                                break;
                            }

                            if(this->mat_.col[aj] == Asub_col)
                            {
                                Asub[j + k * nnz_row] = this->mat_.val[aj];
                                break;
                            }
                        }

                        if(this->mat_.col[aj] == ai)
                        {
                            break;
                        }
                    }
                }

                std::vector<ValueType> mk(nnz_row, static_cast<ValueType>(0));
                mk[nnz_row - 1] = static_cast<ValueType>(1);

                // compute inplace LU factorization of Asub
                for(PtrType i = 0; i < nnz_row - 1; ++i)
                {
                    for(PtrType k = i + 1; k < nnz_row; ++k)
                    {
                        Asub[i + k * nnz_row] /= Asub[i + i * nnz_row];

                        for(PtrType j = i + 1; j < nnz_row; ++j)
                        {
                            Asub[j + k * nnz_row] -= Asub[i + k * nnz_row] * Asub[j + i * nnz_row];
                        }
                    }
                }

                // backward sweeps
                for(PtrType i = nnz_row - 1; i >= 0; --i)
                {
                    mk[i] /= Asub[i + i * nnz_row];

                    for(PtrType j = 0; j < i; ++j)
                    {
                        mk[j] -= mk[i] * Asub[i + j * nnz_row];
                    }
                }

                // update the preconditioner matrix with mk
                for(PtrType aj = row_offset[ai], k = 0; aj < row_offset[ai + 1]; ++aj, ++k)
                {
                    val[aj] = mk[k];
                }
            }
        }

// Scaling
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int ai = 0; ai < nrow; ++ai)
        {
            ValueType fac = sqrt(static_cast<ValueType>(1) / std::abs(val[row_offset[ai + 1] - 1]));

            for(PtrType aj = row_offset[ai]; aj < row_offset[ai + 1]; ++aj)
            {
                val[aj] *= fac;
            }
        }

        this->Clear();
        this->SetDataPtrCSR(&row_offset, &col, &val, nnz, nrow, ncol);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::SPAI(void)
    {
        int     nrow = this->nrow_;
        int64_t nnz  = this->nnz_;

        ValueType* val = NULL;
        allocate_host(nnz, &val);

        HostMatrixCSR<ValueType> T(this->local_backend_);
        T.CopyFrom(*this);
        this->Transpose();

// Loop over each row to get J indexing vector
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int i = 0; i < nrow; ++i)
        {
            int* J     = NULL;
            int  Jsize = static_cast<int>(this->mat_.row_offset[i + 1] - this->mat_.row_offset[i]);
            allocate_host(Jsize, &J);
            std::vector<int> I;

            // Setup J = {j | m(j) != 0}
            for(PtrType j = this->mat_.row_offset[i], idx = 0; j < this->mat_.row_offset[i + 1];
                ++j, ++idx)
            {
                J[idx] = this->mat_.col[j];
            }

            // Setup I = {i | row A(i,J) != 0}
            for(int idx = 0; idx < Jsize; ++idx)
            {
                for(PtrType j = this->mat_.row_offset[J[idx]];
                    j < this->mat_.row_offset[J[idx] + 1];
                    ++j)
                {
                    if(std::find(I.begin(), I.end(), this->mat_.col[j]) == I.end())
                    {
                        I.push_back(this->mat_.col[j]);
                    }
                }
            }

            // Build dense matrix
            HostMatrixDENSE<ValueType> Asub(this->local_backend_);
            Asub.AllocateDENSE(int(I.size()), Jsize);

            for(int k = 0; k < Asub.nrow_; ++k)
            {
                for(PtrType aj = T.mat_.row_offset[I[k]]; aj < T.mat_.row_offset[I[k] + 1]; ++aj)
                {
                    for(int j = 0; j < Jsize; ++j)
                    {
                        if(T.mat_.col[aj] == J[j])
                        {
                            Asub.mat_.val[DENSE_IND(k, j, Asub.nrow_, Asub.ncol_)] = T.mat_.val[aj];
                        }
                    }
                }
            }

            // QR Decomposition of dense submatrix Ak
            Asub.QRDecompose();

            // Solve least squares
            HostVector<ValueType> ek(this->local_backend_);
            HostVector<ValueType> mk(this->local_backend_);

            ek.Allocate(Asub.nrow_);
            mk.Allocate(Asub.ncol_);

            for(int j = 0; j < ek.GetSize(); ++j)
            {
                if(I[j] == i)
                {
                    ek.vec_[j] = 1.0;
                }
            }

            Asub.QRSolve(ek, &mk);

            // Write m_k into preconditioner matrix
            for(int j = 0; j < Jsize; ++j)
            {
                val[this->mat_.row_offset[i] + j] = mk.vec_[j];
            }

            // Clear index vectors
            I.clear();
            ek.Clear();
            mk.Clear();
            Asub.Clear();
            free_host(&J);
        }

        // Only reset value array since we keep the sparsity pattern of A
        free_host(&this->mat_.val);
        this->mat_.val = val;

        this->Transpose();

        return true;
    }

    // ----------------------------------------------------------
    // original functions:
    //   transfer_operators(const Matrix &A, const params &prm)
    //   connect(backend::crs<Val, Col, Ptr> const &A,
    //           float eps_strong,
    //           backend::crs<char, Col, Ptr> &S,
    //           std::vector<char> &cf)
    // ----------------------------------------------------------
    // Modified and adopted from AMGCL,
    // https://github.com/ddemidov/amgcl
    // MIT License
    // ----------------------------------------------------------
    // CHANGELOG
    // - adopted interface
    // ----------------------------------------------------------
    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::RSCoarsening(float             eps,
                                                BaseVector<int>*  CFmap,
                                                BaseVector<bool>* S) const
    {
        assert(CFmap != NULL);
        assert(S != NULL);

        HostVector<int>*  cast_cf = dynamic_cast<HostVector<int>*>(CFmap);
        HostVector<bool>* cast_S  = dynamic_cast<HostVector<bool>*>(S);

        assert(cast_cf != NULL);
        assert(cast_S != NULL);

        // Allocate CF mapping
        cast_cf->Clear();
        cast_cf->Allocate(this->nrow_);

        // Mark all vertices as undecided
        cast_cf->Zeros();

        // Allocate S
        // S is the auxiliary strength matrix such that
        //
        // S_ij = { 1   if i != j and -a_ij > eps * max(-a_ik) for k != i
        //        { 0   otherwise
        //
        // This means, S_ij is 1, only if i strongly depends on j.
        // S has been extended to also work with matrices that do not have
        // fully non-positive off-diagonal entries.

        cast_S->Clear();
        cast_S->Allocate(this->nnz_);

        // Initialize S to false (no dependencies)
        cast_S->Zeros();

        // Array of strong couplings S
        PtrType* S_row_offset = NULL;
        int*     S_col        = NULL;

        allocate_host(this->nrow_ + 1, &S_row_offset);
        set_to_zero_host(this->nrow_ + 1, S_row_offset);

// Determine strong influences in matrix (Ruge Stuben approach)
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < this->nrow_; ++i)
        {
            // Determine minimum and maximum off-diagonal of the current row
            ValueType min_a_ik = static_cast<ValueType>(0);
            ValueType max_a_ik = static_cast<ValueType>(0);

            PtrType row_begin = this->mat_.row_offset[i];
            PtrType row_end   = this->mat_.row_offset[i + 1];

            // True, if the diagonal element is negative
            bool sign = false;

            // Determine diagonal sign and min/max
            for(PtrType j = row_begin; j < row_end; ++j)
            {
                int       col = this->mat_.col[j];
                ValueType val = this->mat_.val[j];

                if(col == i)
                {
                    // Get diagonal entry sign
                    sign = val < static_cast<ValueType>(0);
                }
                else
                {
                    // Get min / max entries
                    min_a_ik = (min_a_ik < val) ? min_a_ik : val;
                    max_a_ik = (max_a_ik > val) ? max_a_ik : val;
                }
            }

            // Threshold to check for strength of connection
            ValueType cond = (sign ? max_a_ik : min_a_ik) * static_cast<ValueType>(eps);

            // Fill S
            for(PtrType j = row_begin; j < row_end; ++j)
            {
                int       col = this->mat_.col[j];
                ValueType val = this->mat_.val[j];

                cast_S->vec_[j] = (col != i) && (val < cond);
            }

            // If cond is zero -> i is independent of other grid points
            if(cond == static_cast<ValueType>(0))
            {
                cast_cf->vec_[i] = 2;
            }
        }

        // Transpose S
        for(int64_t i = 0; i < this->nnz_; ++i)
        {
            if(cast_S->vec_[i])
            {
                S_row_offset[this->mat_.col[i] + 1]++;
            }
        }

        for(int i = 0; i < this->nrow_; ++i)
        {
            S_row_offset[i + 1] += S_row_offset[i];
        }

        allocate_host(S_row_offset[this->nrow_], &S_col);

        for(int i = 0; i < this->nrow_; ++i)
        {
            for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                if(cast_S->vec_[j])
                {
                    S_col[S_row_offset[this->mat_.col[j]]++] = i;
                }
            }
        }

        for(int i = this->nrow_; i > 0; --i)
        {
            S_row_offset[i] = S_row_offset[i - 1];
        }

        S_row_offset[0] = 0;

        // Split into C and F
        std::vector<int> lambda(this->nrow_);

        for(int i = 0; i < this->nrow_; ++i)
        {
            int temp = 0;
            for(PtrType j = S_row_offset[i]; j < S_row_offset[i + 1]; ++j)
            {
                temp += (cast_cf->vec_[S_col[j]] == 0 ? 1 : 2);
            }

            lambda[i] = temp;
        }

        std::vector<int> ptr(this->nrow_ + 1, static_cast<int>(0));
        std::vector<int> cnt(this->nrow_, static_cast<int>(0));
        std::vector<int> i2n(this->nrow_);
        std::vector<int> n2i(this->nrow_);

        for(int i = 0; i < this->nrow_; ++i)
        {
            ptr[lambda[i] + 1]++;
        }

        for(unsigned int i = 1; i < ptr.size(); ++i)
        {
            ptr[i] += ptr[i - 1];
        }

        for(int i = 0; i < this->nrow_; ++i)
        {
            int lam  = lambda[i];
            int idx  = ptr[lam] + cnt[lam]++;
            i2n[idx] = i;
            n2i[i]   = idx;
        }

        for(int top = this->nrow_ - 1; top >= 0; --top)
        {
            int i   = i2n[top];
            int lam = lambda[i];

            if(lam == 0)
            {
                for(int ai = 0; ai < this->nrow_; ++ai)
                {
                    if(cast_cf->vec_[ai] == 0)
                    {
                        cast_cf->vec_[ai] = 1;
                    }
                }

                break;
            }

            cnt[lam]--;

            if(cast_cf->vec_[i] == 2)
            {
                continue;
            }

            assert(cast_cf->vec_[i] == 0);

            cast_cf->vec_[i] = 1;

            for(PtrType j = S_row_offset[i]; j < S_row_offset[i + 1]; ++j)
            {
                int c = S_col[j];

                if(cast_cf->vec_[c] != 0)
                {
                    continue;
                }

                cast_cf->vec_[c] = 2;

                for(PtrType jj = this->mat_.row_offset[c]; jj < this->mat_.row_offset[c + 1]; ++jj)
                {
                    if(!cast_S->vec_[jj])
                    {
                        continue;
                    }

                    int cc     = this->mat_.col[jj];
                    int lam_cc = lambda[cc];

                    if(cast_cf->vec_[cc] != 0 || lam_cc >= this->nrow_ - 1)
                    {
                        continue;
                    }

                    int old_pos = n2i[cc];
                    int new_pos = ptr[lam_cc] + cnt[lam_cc] - 1;

                    n2i[i2n[old_pos]] = new_pos;
                    n2i[i2n[new_pos]] = old_pos;

                    std::swap(i2n[old_pos], i2n[new_pos]);

                    --cnt[lam_cc];
                    ++cnt[lam_cc + 1];
                    ptr[lam_cc + 1] = ptr[lam_cc] + cnt[lam_cc];

                    ++lambda[cc];
                }
            }

            for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                if(!cast_S->vec_[j])
                {
                    continue;
                }

                int c   = this->mat_.col[j];
                int lam = lambda[c];

                if(cast_cf->vec_[c] != 0 || lam == 0)
                {
                    continue;
                }

                int old_pos = n2i[c];
                int new_pos = ptr[lam];

                n2i[i2n[old_pos]] = new_pos;
                n2i[i2n[new_pos]] = old_pos;

                std::swap(i2n[old_pos], i2n[new_pos]);

                --cnt[lam];
                ++cnt[lam - 1];
                ++ptr[lam];
                --lambda[c];

                assert(ptr[lam - 1] == ptr[lam] - cnt[lam - 1]);
            }
        }

        // Clean up
        free_host(&S_row_offset);
        free_host(&S_col);

        return true;
    }

    static float hash(uint64_t key)
    {
        // 64 bit integer mix function
        key = (~key) + (key << 21); // key = (key << 21) - key - 1;
        key = key ^ (key >> 24);
        key = (key + (key << 3)) + (key << 8); // key * 265
        key = key ^ (key >> 14);
        key = (key + (key << 2)) + (key << 4); // key * 21
        key = key ^ (key >> 28);
        key = key + (key << 31);
        return static_cast<float>(key / (float)UINT64_MAX);
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::RSPMISStrongInfluences(float              eps,
                                                          BaseVector<bool>*  S,
                                                          BaseVector<float>* omega,
                                                          int64_t            global_row_offset,
                                                          const BaseMatrix<ValueType>& ghost) const
    {
        assert(S != NULL);
        assert(omega != NULL);

        HostVector<bool>*               cast_S = dynamic_cast<HostVector<bool>*>(S);
        HostVector<float>*              cast_w = dynamic_cast<HostVector<float>*>(omega);
        const HostMatrixCSR<ValueType>* cast_gst
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&ghost);

        assert(cast_S != NULL);
        assert(cast_w != NULL);
        assert(cast_gst != NULL);

        // Do we need communication?
        bool global = cast_gst->nrow_ > 0;

        // Initialize S to false (no dependencies)
        cast_S->Zeros();

        // Sample some numbers using hash function to initialize omega
        for(int64_t i = 0; i < this->nrow_; ++i)
        {
            cast_w->vec_[i] = hash(i + global_row_offset);
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        // Determine strong influences in the matrix
        for(int i = 0; i < this->nrow_; ++i)
        {
            // Determine minimum and maximum off-diagonal of the current row
            ValueType min_a_ik = static_cast<ValueType>(0);
            ValueType max_a_ik = static_cast<ValueType>(0);

            PtrType row_begin = this->mat_.row_offset[i];
            PtrType row_end   = this->mat_.row_offset[i + 1];

            // True, if the diagonal element is negative
            bool sign = false;

            // Determine diagonal sign and min/max
            for(PtrType j = row_begin; j < row_end; ++j)
            {
                int       col = this->mat_.col[j];
                ValueType val = this->mat_.val[j];

                if(col == i)
                {
                    // Get diagonal entry sign
                    sign = val < static_cast<ValueType>(0);
                }
                else
                {
                    // Get min / max entries
                    min_a_ik = (min_a_ik < val) ? min_a_ik : val;
                    max_a_ik = (max_a_ik > val) ? max_a_ik : val;
                }
            }

            if(global == true)
            {
                PtrType gst_row_begin = cast_gst->mat_.row_offset[i];
                PtrType gst_row_end   = cast_gst->mat_.row_offset[i + 1];

                for(PtrType j = gst_row_begin; j < gst_row_end; ++j)
                {
                    ValueType val = cast_gst->mat_.val[j];

                    min_a_ik = (min_a_ik < val) ? min_a_ik : val;
                    max_a_ik = (max_a_ik > val) ? max_a_ik : val;
                }
            }

            // Threshold to check for strength of connection
            ValueType cond = (sign ? max_a_ik : min_a_ik) * static_cast<ValueType>(eps);

            // Fill S
            for(PtrType j = row_begin; j < row_end; ++j)
            {
                int       col = this->mat_.col[j];
                ValueType val = this->mat_.val[j];

                if(col != i && val < cond)
                {
                    // col is strongly connected to i
                    cast_S->vec_[j] = true;

#ifdef _OPENMP
#pragma omp atomic
#endif
                    // Increment omega, as it holds all strongly connected edges
                    // of vertex col.
                    // Additionally, omega holds a random number between 0 and 1 to
                    // distinguish neighbor points with equal number of strong
                    // connections.
                    cast_w->vec_[col] += 1.0f;
                }
            }

            if(global == true)
            {
                PtrType gst_row_begin = cast_gst->mat_.row_offset[i];
                PtrType gst_row_end   = cast_gst->mat_.row_offset[i + 1];

                for(PtrType j = gst_row_begin; j < gst_row_end; ++j)
                {
                    int       col = cast_gst->mat_.col[j];
                    ValueType val = cast_gst->mat_.val[j];

                    if(val < cond)
                    {
                        // col is strongly connected to i
                        cast_S->vec_[j + this->nnz_] = true;

#ifdef _OPENMP
#pragma omp atomic
#endif
                        // Increment omega, as it holds all strongly connected edges
                        // of vertex col.
                        // Additionally, omega holds a random number between 0 and 1 to
                        // distinguish neighbor points with equal number of strong
                        // connections.
                        cast_w->vec_[col + this->nrow_] += 1.0f;
                    }
                }
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::RSPMISUnassignedToCoarse(BaseVector<int>*         CFmap,
                                                            BaseVector<bool>*        marked,
                                                            const BaseVector<float>& omega) const
    {
        assert(CFmap != NULL);
        assert(marked != NULL);

        HostVector<int>*         cast_cf = dynamic_cast<HostVector<int>*>(CFmap);
        HostVector<bool>*        cast_m  = dynamic_cast<HostVector<bool>*>(marked);
        const HostVector<float>* cast_w  = dynamic_cast<const HostVector<float>*>(&omega);

        assert(cast_cf != NULL);
        assert(cast_m != NULL);
        assert(cast_w != NULL);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        // First, mark all vertices that have not been assigned yet, as coarse
        for(int i = 0; i < cast_cf->size_; ++i)
        {
            // Marked vector keeps track, whether a vertex has been marked coarse
            // during the current iteration, or not.
            cast_m->vec_[i] = false;

            // Check only undecided vertices
            if(cast_cf->vec_[i] == 0)
            {
                // If this vertex has an edge, it might be a coarse one
                if(cast_w->vec_[i] >= 1.0f)
                {
                    cast_cf->vec_[i] = 1;

                    // Keep in mind, that this vertex has been marked coarse in the
                    // current iteration
                    cast_m->vec_[i] = true;
                }
                else
                {
                    // This point does not influence any other points and thus is a
                    // fine point
                    cast_cf->vec_[i] = 2;
                }
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::RSPMISCorrectCoarse(BaseVector<int>*             CFmap,
                                                       const BaseVector<bool>&      S,
                                                       const BaseVector<bool>&      marked,
                                                       const BaseVector<float>&     omega,
                                                       const BaseMatrix<ValueType>& ghost) const
    {
        assert(CFmap != NULL);

        HostVector<int>*                cast_cf = dynamic_cast<HostVector<int>*>(CFmap);
        const HostVector<bool>*         cast_S  = dynamic_cast<const HostVector<bool>*>(&S);
        const HostVector<bool>*         cast_m  = dynamic_cast<const HostVector<bool>*>(&marked);
        const HostVector<float>*        cast_w  = dynamic_cast<const HostVector<float>*>(&omega);
        const HostMatrixCSR<ValueType>* cast_gst
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&ghost);

        assert(cast_cf != NULL);
        assert(cast_S != NULL);
        assert(cast_m != NULL);
        assert(cast_w != NULL);
        assert(cast_gst != NULL);

        // Do we need communication?
        bool global = cast_gst->nrow_ > 0;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        // Now, correct previously marked vertices with respect to omega
        for(int i = 0; i < this->nrow_; ++i)
        {
            PtrType row_begin = this->mat_.row_offset[i];
            PtrType row_end   = this->mat_.row_offset[i + 1];

            // If this vertex has been marked coarse in the current iteration,
            // process it for further checks
            if(cast_m->vec_[i])
            {
                // Get the weight of the current row for comparison
                float omega_row = cast_w->vec_[i];

                // Loop over the full row to compare weights of other vertices that
                // have been marked coarse in the current iteration
                for(PtrType j = row_begin; j < row_end; ++j)
                {
                    // Process only vertices that are strongly connected
                    if(cast_S->vec_[j])
                    {
                        int col = this->mat_.col[j];

                        // If this vertex has been marked coarse in the current iteration,
                        // we need to check whether it is accepted as a coarse vertex or not.
                        if(cast_m->vec_[col] == true)
                        {
                            // Get the weight of the current vertex for comparison
                            float omega_col = cast_w->vec_[col];

                            if(omega_row > omega_col)
                            {
                                // The diagonal entry has more edges and will remain
                                // a coarse point, whereas this vertex gets reverted
                                // back to undecided, for further processing.
                                cast_cf->vec_[col] = 0;
                            }
                            else if(omega_row < omega_col)
                            {
                                // The diagonal entry has fewer edges and gets
                                // reverted back to undecided for further processing,
                                // whereas this vertex stays
                                // a coarse one.
                                cast_cf->vec_[i] = 0;
                            }
                        }
                    }
                }

                if(global == true)
                {
                    PtrType gst_row_begin = cast_gst->mat_.row_offset[i];
                    PtrType gst_row_end   = cast_gst->mat_.row_offset[i + 1];

                    // Loop over the full boundary row to compare weights of other vertices that
                    // have been marked coarse in the current iteration
                    for(PtrType j = gst_row_begin; j < gst_row_end; ++j)
                    {
                        // Process only vertices that are strongly connected
                        if(cast_S->vec_[j + this->nnz_])
                        {
                            int col = cast_gst->mat_.col[j];

                            // If this vertex has been marked coarse in the current iteration,
                            // we need to check whether it is accepted as a coarse vertex or not.
                            if(cast_m->vec_[col + this->nrow_])
                            {
                                // Get the weight of the current vertex for comparison
                                float omega_col = cast_w->vec_[col + this->nrow_];

                                if(omega_row > omega_col)
                                {
                                    // The diagonal entry has more edges and will remain
                                    // a coarse point, whereas this vertex gets reverted
                                    // back to undecided, for further processing.
                                    cast_cf->vec_[col + this->nrow_] = 0;
                                }
                                else if(omega_row < omega_col)
                                {
                                    // The diagonal entry has fewer edges and gets
                                    // reverted back to undecided for further processing,
                                    // whereas this vertex stays
                                    // a coarse one.
                                    cast_cf->vec_[i] = 0;
                                }
                            }
                        }
                    }
                }
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::RSPMISCoarseEdgesToFine(BaseVector<int>*             CFmap,
                                                           const BaseVector<bool>&      S,
                                                           const BaseMatrix<ValueType>& ghost) const
    {
        assert(CFmap != NULL);

        HostVector<int>*                cast_cf = dynamic_cast<HostVector<int>*>(CFmap);
        const HostVector<bool>*         cast_S  = dynamic_cast<const HostVector<bool>*>(&S);
        const HostMatrixCSR<ValueType>* cast_gst
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&ghost);

        assert(cast_cf != NULL);
        assert(cast_S != NULL);
        assert(cast_gst != NULL);

        // Do we need communication?
        bool global = cast_gst->nrow_ > 0;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        // Mark remaining edges of a coarse point to fine
        for(int i = 0; i < this->nrow_; ++i)
        {
            // Process only undecided vertices
            if(cast_cf->vec_[i] == 0)
            {
                PtrType row_begin = this->mat_.row_offset[i];
                PtrType row_end   = this->mat_.row_offset[i + 1];

                // Loop over all edges of this undecided vertex
                // and check, if there is a coarse point connected
                for(PtrType j = row_begin; j < row_end; ++j)
                {
                    // Check, whether this edge is strongly connected to the vertex
                    if(cast_S->vec_[j])
                    {
                        int col = this->mat_.col[j];

                        // If this edge is coarse, our vertex must be fine
                        if(cast_cf->vec_[col] == 1)
                        {
                            cast_cf->vec_[i] = 2;
                            break;
                        }
                    }
                }

                if(global == true)
                {
                    PtrType gst_row_begin = cast_gst->mat_.row_offset[i];
                    PtrType gst_row_end   = cast_gst->mat_.row_offset[i + 1];

                    // Loop over all ghost edges of this undecided vertex
                    // and check, if there is a coarse point connected
                    for(PtrType j = gst_row_begin; j < gst_row_end; ++j)
                    {
                        // Check, whether this edge is strongly connected to the vertex
                        if(cast_S->vec_[j + this->nnz_])
                        {
                            int col = cast_gst->mat_.col[j];

                            // If this edge is coarse, our vertex must be fine
                            if(cast_cf->vec_[col + this->nrow_] == 1)
                            {
                                cast_cf->vec_[i] = 2;
                                break;
                            }
                        }
                    }
                }
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::RSPMISCheckUndecided(bool&                  undecided,
                                                        const BaseVector<int>& CFmap) const
    {
        const HostVector<int>* cast_cf = dynamic_cast<const HostVector<int>*>(&CFmap);

        assert(cast_cf != NULL);

        undecided = false;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < this->nrow_; ++i)
        {
            // Check whether this vertex is undecided or not
            if(cast_cf->vec_[i] == 0)
            {
                undecided = true;
                i         = this->nrow_;
            }
        }

        return true;
    }

    // ----------------------------------------------------------
    // original functions:
    //   cfsplit(backend::crs<Val, Col, Ptr> const &A,
    //           backend::crs<char, Col, Ptr> const &S,
    //           std::vector<char> &cf)
    // ----------------------------------------------------------
    // Modified and adopted from AMGCL,
    // https://github.com/ddemidov/amgcl
    // MIT License
    // ----------------------------------------------------------
    // CHANGELOG
    // - adopted interface
    // ----------------------------------------------------------
    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::RSDirectProlongNnz(const BaseVector<int>&       CFmap,
                                                      const BaseVector<bool>&      S,
                                                      const BaseMatrix<ValueType>& ghost,
                                                      BaseVector<ValueType>*       Amin,
                                                      BaseVector<ValueType>*       Amax,
                                                      BaseVector<int>*             f2c,
                                                      BaseMatrix<ValueType>*       prolong_int,
                                                      BaseMatrix<ValueType>* prolong_gst) const
    {
        const HostVector<int>*          cast_cf = dynamic_cast<const HostVector<int>*>(&CFmap);
        const HostVector<bool>*         cast_S  = dynamic_cast<const HostVector<bool>*>(&S);
        const HostMatrixCSR<ValueType>* cast_gst
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&ghost);
        HostVector<ValueType>*    cast_Amin = dynamic_cast<HostVector<ValueType>*>(Amin);
        HostVector<ValueType>*    cast_Amax = dynamic_cast<HostVector<ValueType>*>(Amax);
        HostVector<int>*          cast_f2c  = dynamic_cast<HostVector<int>*>(f2c);
        HostMatrixCSR<ValueType>* cast_pi   = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong_int);
        HostMatrixCSR<ValueType>* cast_pg   = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong_gst);

        assert(cast_cf != NULL);
        assert(cast_S != NULL);
        assert(cast_f2c != NULL);
        assert(cast_pi != NULL);
        assert(cast_Amin != NULL);
        assert(cast_Amax != NULL);
        assert(cast_Amin->size_ == this->nrow_);
        assert(cast_Amax->size_ == this->nrow_);

        // Do we need communication?
        bool global = prolong_gst != NULL;

        // Start with fresh P
        cast_pi->Clear();

        // Allocate P row pointer array
        allocate_host(this->nrow_ + 1, &cast_pi->mat_.row_offset);

        // We already know the number of rows of P
        cast_pi->nrow_ = this->nrow_;

        // Ghost part
        if(global == true)
        {
            // LCOV_EXCL_START
            assert(cast_gst != NULL);
            assert(cast_pg != NULL);

            // Start with fresh P ghost
            cast_pg->Clear();

            // Allocate P ghost row pointer array
            allocate_host(this->nrow_ + 1, &cast_pg->mat_.row_offset);

            // Number of ghost rows is identical to interior
            cast_pg->nrow_ = this->nrow_;
            // LCOV_EXCL_STOP
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int row = 0; row < this->nrow_; ++row)
        {
            // Coarse points generate a single entry
            if(cast_cf->vec_[row] == 1)
            {
                // Set this points state to coarse
                cast_f2c->vec_[row]           = 1;
                cast_pi->mat_.row_offset[row] = 1;

                if(global == true)
                {
                    cast_pg->mat_.row_offset[row] = 0;
                }
            }
            else
            {
                cast_f2c->vec_[row] = 0;

                PtrType nnz = 0;

                ValueType amin = static_cast<ValueType>(0);
                ValueType amax = static_cast<ValueType>(0);

                PtrType row_begin = this->mat_.row_offset[row];
                PtrType row_end   = this->mat_.row_offset[row + 1];

                for(PtrType j = row_begin; j < row_end; ++j)
                {
                    if(!cast_S->vec_[j] || cast_cf->vec_[this->mat_.col[j]] != 1)
                    {
                        continue;
                    }

                    amin = (amin < this->mat_.val[j]) ? amin : this->mat_.val[j];
                    amax = (amax > this->mat_.val[j]) ? amax : this->mat_.val[j];
                }

                // Ghost part
                if(global == true)
                {
                    for(PtrType j = cast_gst->mat_.row_offset[row];
                        j < cast_gst->mat_.row_offset[row + 1];
                        ++j)
                    {
                        if(!cast_S->vec_[j + this->nnz_]
                           || cast_cf->vec_[cast_gst->mat_.col[j] + this->nrow_] != 1)
                        {
                            continue;
                        }

                        amin = (amin < cast_gst->mat_.val[j]) ? amin : cast_gst->mat_.val[j];
                        amax = (amax > cast_gst->mat_.val[j]) ? amax : cast_gst->mat_.val[j];
                    }
                }

                cast_Amin->vec_[row] = amin = amin * static_cast<ValueType>(0.2f);
                cast_Amax->vec_[row] = amax = amax * static_cast<ValueType>(0.2f);

                for(PtrType j = row_begin; j < row_end; ++j)
                {
                    int col = this->mat_.col[j];

                    if(cast_S->vec_[j] && cast_cf->vec_[col] == 1)
                    {
                        if(this->mat_.val[j] <= amin || this->mat_.val[j] >= amax)
                        {
                            ++nnz;
                        }
                    }
                }

                cast_pi->mat_.row_offset[row] = nnz;

                // Ghost part
                if(global == true)
                {
                    PtrType gst_nnz = 0;

                    row_begin = cast_gst->mat_.row_offset[row];
                    row_end   = cast_gst->mat_.row_offset[row + 1];

                    for(PtrType j = row_begin; j < row_end; ++j)
                    {
                        int col = cast_gst->mat_.col[j];

                        if(cast_S->vec_[j + this->nnz_] && cast_cf->vec_[col + this->nrow_] == 1)
                        {
                            if(cast_gst->mat_.val[j] <= amin || cast_gst->mat_.val[j] >= amax)
                            {
                                ++gst_nnz;
                            }
                        }
                    }

                    cast_pg->mat_.row_offset[row] = gst_nnz;
                }
            }
        }

        cast_f2c->ExclusiveSum(*cast_f2c);

        return true;
    }

    // ----------------------------------------------------------
    // original functions:
    //   cfsplit(backend::crs<Val, Col, Ptr> const &A,
    //           backend::crs<char, Col, Ptr> const &S,
    //           std::vector<char> &cf)
    // ----------------------------------------------------------
    // Modified and adopted from AMGCL,
    // https://github.com/ddemidov/amgcl
    // MIT License
    // ----------------------------------------------------------
    // CHANGELOG
    // - adopted interface
    // ----------------------------------------------------------
    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::RSDirectProlongFill(const BaseVector<int64_t>&   l2g,
                                                       const BaseVector<int>&       f2c,
                                                       const BaseVector<int>&       CFmap,
                                                       const BaseVector<bool>&      S,
                                                       const BaseMatrix<ValueType>& ghost,
                                                       const BaseVector<ValueType>& Amin,
                                                       const BaseVector<ValueType>& Amax,
                                                       BaseMatrix<ValueType>*       prolong_int,
                                                       BaseMatrix<ValueType>*       prolong_gst,
                                                       BaseVector<int64_t>* global_ghost_col) const
    {
        const HostVector<int64_t>*      cast_l2g = dynamic_cast<const HostVector<int64_t>*>(&l2g);
        const HostVector<int>*          cast_f2c = dynamic_cast<const HostVector<int>*>(&f2c);
        const HostVector<int>*          cast_cf  = dynamic_cast<const HostVector<int>*>(&CFmap);
        const HostVector<bool>*         cast_S   = dynamic_cast<const HostVector<bool>*>(&S);
        const HostMatrixCSR<ValueType>* cast_gst
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&ghost);
        const HostVector<ValueType>* cast_Amin = dynamic_cast<const HostVector<ValueType>*>(&Amin);
        const HostVector<ValueType>* cast_Amax = dynamic_cast<const HostVector<ValueType>*>(&Amax);
        HostMatrixCSR<ValueType>*    cast_pi = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong_int);
        HostMatrixCSR<ValueType>*    cast_pg = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong_gst);
        HostVector<int64_t>* cast_glo        = dynamic_cast<HostVector<int64_t>*>(global_ghost_col);

        assert(cast_f2c != NULL);
        assert(cast_cf != NULL);
        assert(cast_S != NULL);
        assert(cast_pi != NULL);
        assert(cast_Amin != NULL);
        assert(cast_Amax != NULL);
        assert(cast_Amin->size_ == this->nrow_);
        assert(cast_Amax->size_ == this->nrow_);

        // Do we need communication?
        bool global = prolong_gst != NULL;

        // Ghost part
        if(global == true)
        {
            // LCOV_EXCL_START
            assert(cast_l2g != NULL);
            assert(cast_gst != NULL);
            assert(cast_pg != NULL);
            assert(cast_glo != NULL);
            // LCOV_EXCL_STOP
        }

        // Exclusive sum to obtain row offset pointers of P
        // P contains only nnz per row, so far
        for(int i = this->nrow_; i > 0; --i)
        {
            cast_pi->mat_.row_offset[i] = cast_pi->mat_.row_offset[i - 1];
        }

        cast_pi->mat_.row_offset[0] = 0;

        for(int i = 0; i < this->nrow_; ++i)
        {
            cast_pi->mat_.row_offset[i + 1] += cast_pi->mat_.row_offset[i];
        }

        // Initialize nnz of P
        cast_pi->nnz_ = cast_pi->mat_.row_offset[this->nrow_];

        // Initialize ncol of P
        cast_pi->ncol_ = cast_f2c->vec_[this->nrow_];

        // Allocate column and value arrays
        allocate_host(cast_pi->nnz_, &cast_pi->mat_.col);
        allocate_host(cast_pi->nnz_, &cast_pi->mat_.val);

        // Once again for the ghost part
        if(global == true)
        {
            // LCOV_EXCL_START
            for(int i = this->nrow_; i > 0; --i)
            {
                cast_pg->mat_.row_offset[i] = cast_pg->mat_.row_offset[i - 1];
            }

            cast_pg->mat_.row_offset[0] = 0;

            for(int i = 0; i < this->nrow_; ++i)
            {
                cast_pg->mat_.row_offset[i + 1] += cast_pg->mat_.row_offset[i];
            }

            cast_pg->nnz_  = cast_pg->mat_.row_offset[this->nrow_];
            cast_pg->ncol_ = this->nrow_;

            allocate_host(cast_pg->nnz_, &cast_pg->mat_.col);
            allocate_host(cast_pg->nnz_, &cast_pg->mat_.val);

            cast_glo->Allocate(cast_pg->nnz_);
            // LCOV_EXCL_STOP
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int row = 0; row < this->nrow_; ++row)
        {
            PtrType row_P = cast_pi->mat_.row_offset[row];

            if(cast_cf->vec_[row] == 1)
            {
                cast_pi->mat_.col[row_P] = cast_f2c->vec_[row];
                cast_pi->mat_.val[row_P] = static_cast<ValueType>(1);

                continue;
            }

            ValueType diag  = static_cast<ValueType>(0);
            ValueType a_num = static_cast<ValueType>(0), a_den = static_cast<ValueType>(0);
            ValueType b_num = static_cast<ValueType>(0), b_den = static_cast<ValueType>(0);
            ValueType d_neg = static_cast<ValueType>(0), d_pos = static_cast<ValueType>(0);

            PtrType row_begin = this->mat_.row_offset[row];
            PtrType row_end   = this->mat_.row_offset[row + 1];

            for(PtrType j = row_begin; j < row_end; ++j)
            {
                int       col = this->mat_.col[j];
                ValueType val = this->mat_.val[j];

                if(col == row)
                {
                    diag = val;
                    continue;
                }

                if(val < static_cast<ValueType>(0))
                {
                    a_num += val;

                    if(cast_S->vec_[j] && cast_cf->vec_[col] == 1)
                    {
                        a_den += val;

                        if(val > cast_Amin->vec_[row])
                        {
                            d_neg += val;
                        }
                    }
                }
                else
                {
                    b_num += val;

                    if(cast_S->vec_[j] && cast_cf->vec_[col] == 1)
                    {
                        b_den += val;

                        if(val < cast_Amax->vec_[row])
                        {
                            d_pos += val;
                        }
                    }
                }
            }

            if(global == true)
            {
                PtrType ghost_row_begin = cast_gst->mat_.row_offset[row];
                PtrType ghost_row_end   = cast_gst->mat_.row_offset[row + 1];

                for(PtrType j = ghost_row_begin; j < ghost_row_end; ++j)
                {
                    int       col = cast_gst->mat_.col[j];
                    ValueType val = cast_gst->mat_.val[j];

                    if(val < static_cast<ValueType>(0))
                    {
                        a_num += val;

                        if(cast_S->vec_[j + this->nnz_] && cast_cf->vec_[col + this->nrow_] == 1)
                        {
                            a_den += val;

                            if(val > cast_Amin->vec_[row])
                            {
                                d_neg += val;
                            }
                        }
                    }
                    else
                    {
                        b_num += val;

                        if(cast_S->vec_[j + this->nnz_] && cast_cf->vec_[col + this->nrow_] == 1)
                        {
                            b_den += val;

                            if(val < cast_Amax->vec_[row])
                            {
                                d_pos += val;
                            }
                        }
                    }
                }
            }

            ValueType cf_neg = static_cast<ValueType>(1);
            ValueType cf_pos = static_cast<ValueType>(1);

            if(std::abs(a_den - d_neg) > 1e-32)
            {
                cf_neg = a_den / (a_den - d_neg);
            }

            if(std::abs(b_den - d_pos) > 1e-32)
            {
                cf_pos = b_den / (b_den - d_pos);
            }

            if(b_num > static_cast<ValueType>(0) && std::abs(b_den) < 1e-32)
            {
                diag += b_num;
            }

            ValueType alpha = std::abs(a_den) > 1e-32 ? -cf_neg * a_num / (diag * a_den)
                                                      : static_cast<ValueType>(0);
            ValueType beta  = std::abs(b_den) > 1e-32 ? -cf_pos * b_num / (diag * b_den)
                                                      : static_cast<ValueType>(0);

            for(PtrType j = row_begin; j < row_end; ++j)
            {
                int       col = this->mat_.col[j];
                ValueType val = this->mat_.val[j];

                if(cast_S->vec_[j] && cast_cf->vec_[col] == 1)
                {
                    if(val > cast_Amin->vec_[row] && val < cast_Amax->vec_[row])
                    {
                        continue;
                    }

                    cast_pi->mat_.col[row_P] = cast_f2c->vec_[col];
                    cast_pi->mat_.val[row_P]
                        = (val < static_cast<ValueType>(0) ? alpha : beta) * val;
                    ++row_P;
                }
            }

            if(global)
            {
                PtrType ghost_row_P     = cast_pg->mat_.row_offset[row];
                PtrType ghost_row_begin = cast_gst->mat_.row_offset[row];
                PtrType ghost_row_end   = cast_gst->mat_.row_offset[row + 1];

                for(PtrType j = ghost_row_begin; j < ghost_row_end; ++j)
                {
                    int       col = cast_gst->mat_.col[j];
                    ValueType val = cast_gst->mat_.val[j];

                    if(cast_S->vec_[j + this->nnz_] && cast_cf->vec_[col + this->nrow_] == 1)
                    {
                        if(val > cast_Amin->vec_[row] && val < cast_Amax->vec_[row])
                        {
                            continue;
                        }

                        cast_glo->vec_[ghost_row_P] = cast_l2g->vec_[col];
                        cast_pg->mat_.val[ghost_row_P]
                            = (val < static_cast<ValueType>(0) ? alpha : beta) * val;
                        ++ghost_row_P;
                    }
                }
            }
        }

        return true;
    }

    // LCOV_EXCL_START
    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::RSExtPIBoundaryNnz(const BaseVector<int>&       boundary,
                                                      const BaseVector<int>&       CFmap,
                                                      const BaseVector<bool>&      S,
                                                      const BaseMatrix<ValueType>& ghost,
                                                      BaseVector<PtrType>*         row_nnz) const
    {
        const HostVector<int>*          cast_bnd = dynamic_cast<const HostVector<int>*>(&boundary);
        const HostVector<int>*          cast_cf  = dynamic_cast<const HostVector<int>*>(&CFmap);
        const HostVector<bool>*         cast_S   = dynamic_cast<const HostVector<bool>*>(&S);
        const HostMatrixCSR<ValueType>* cast_gst
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&ghost);
        HostVector<PtrType>* cast_nnz = dynamic_cast<HostVector<PtrType>*>(row_nnz);

        assert(cast_bnd != NULL);
        assert(cast_cf != NULL);
        assert(cast_S != NULL);
        assert(cast_gst != NULL);
        assert(cast_nnz != NULL);

        assert(cast_nnz->size_ >= cast_bnd->size_);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int64_t i = 0; i < cast_bnd->size_; ++i)
        {
            // Get boundary row
            int row = cast_bnd->vec_[i];

            // Counter
            PtrType ext_nnz = 0;

            // Interior part
            PtrType row_begin = this->mat_.row_offset[row];
            PtrType row_end   = this->mat_.row_offset[row + 1];

            for(PtrType j = row_begin; j < row_end; ++j)
            {
                // Check whether this vertex is strongly connected
                if(cast_S->vec_[j] == false)
                {
                    continue;
                }

                // Get column index
                int col = this->mat_.col[j];

                // Only coarse points contribute
                if(cast_cf->vec_[col] != 2)
                {
                    ++ext_nnz;
                }
            }

            // Ghost part
            row_begin = cast_gst->mat_.row_offset[row];
            row_end   = cast_gst->mat_.row_offset[row + 1];

            for(PtrType j = row_begin; j < row_end; ++j)
            {
                // Check whether this vertex is strongly connected
                if(cast_S->vec_[j + this->nnz_] == false)
                {
                    continue;
                }

                // Get column index
                int col = cast_gst->mat_.col[j];

                // Only coarse points contribute
                if(cast_cf->vec_[col + this->nrow_] != 2)
                {
                    ++ext_nnz;
                }
            }

            // Write total number of strongly connected coarse vertices to global memory
            cast_nnz->vec_[i] = ext_nnz;
        }

        return true;
    }

    template <typename ValueType>
    bool
        HostMatrixCSR<ValueType>::RSExtPIExtractBoundary(int64_t                global_column_begin,
                                                         const BaseVector<int>& boundary,
                                                         const BaseVector<int64_t>&   l2g,
                                                         const BaseVector<int>&       CFmap,
                                                         const BaseVector<bool>&      S,
                                                         const BaseMatrix<ValueType>& ghost,
                                                         const BaseVector<PtrType>& bnd_csr_row_ptr,
                                                         BaseVector<int64_t>* bnd_csr_col_ind) const
    {
        const HostVector<int>*          cast_bnd = dynamic_cast<const HostVector<int>*>(&boundary);
        const HostVector<int64_t>*      cast_l2g = dynamic_cast<const HostVector<int64_t>*>(&l2g);
        const HostVector<int>*          cast_cf  = dynamic_cast<const HostVector<int>*>(&CFmap);
        const HostVector<bool>*         cast_S   = dynamic_cast<const HostVector<bool>*>(&S);
        const HostMatrixCSR<ValueType>* cast_gst
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&ghost);
        const HostVector<PtrType>* cast_ptr
            = dynamic_cast<const HostVector<PtrType>*>(&bnd_csr_row_ptr);
        HostVector<int64_t>* cast_col = dynamic_cast<HostVector<int64_t>*>(bnd_csr_col_ind);

        assert(cast_bnd != NULL);
        assert(cast_l2g != NULL);
        assert(cast_cf != NULL);
        assert(cast_S != NULL);
        assert(cast_gst != NULL);
        assert(cast_ptr != NULL);
        assert(cast_col != NULL);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int64_t i = 0; i < cast_bnd->size_; ++i)
        {
            // Get boundary row and index into A_ext
            int     row = cast_bnd->vec_[i];
            PtrType idx = cast_ptr->vec_[i];

            // Extract interior part
            PtrType row_begin = this->mat_.row_offset[row];
            PtrType row_end   = this->mat_.row_offset[row + 1];

            for(PtrType j = row_begin; j < row_end; ++j)
            {
                // Check whether this vertex is strongly connected
                if(cast_S->vec_[j] == false)
                {
                    continue;
                }

                // Get column index
                int col = this->mat_.col[j];

                // Only coarse points contribute
                if(cast_cf->vec_[col] != 2)
                {
                    // Shift column by global column offset, to obtain the global column index
                    cast_col->vec_[idx++] = col + global_column_begin;
                }
            }

            // Extract ghost part
            row_begin = cast_gst->mat_.row_offset[row];
            row_end   = cast_gst->mat_.row_offset[row + 1];

            for(PtrType j = row_begin; j < row_end; ++j)
            {
                // Check whether this vertex is strongly connected
                if(cast_S->vec_[j + this->nnz_] == false)
                {
                    continue;
                }

                // Get column index
                int col = cast_gst->mat_.col[j];

                // Only coarse points contribute
                if(cast_cf->vec_[col + this->nrow_] != 2)
                {
                    // Transform local ghost index into global column index
                    cast_col->vec_[idx++] = cast_l2g->vec_[col];
                }
            }
        }

        return true;
    }
    // LCOV_EXCL_STOP

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::RSExtPIProlongNnz(int64_t                    global_column_begin,
                                                     int64_t                    global_column_end,
                                                     bool                       FF1,
                                                     const BaseVector<int64_t>& l2g,
                                                     const BaseVector<int>&     CFmap,
                                                     const BaseVector<bool>&    S,
                                                     const BaseMatrix<ValueType>& ghost,
                                                     const BaseVector<PtrType>&   bnd_csr_row_ptr,
                                                     const BaseVector<int64_t>&   bnd_csr_col_ind,
                                                     BaseVector<int>*             f2c,
                                                     BaseMatrix<ValueType>*       prolong_int,
                                                     BaseMatrix<ValueType>*       prolong_gst) const
    {
        const HostVector<int64_t>*      cast_l2g = dynamic_cast<const HostVector<int64_t>*>(&l2g);
        const HostVector<int>*          cast_cf  = dynamic_cast<const HostVector<int>*>(&CFmap);
        const HostVector<bool>*         cast_S   = dynamic_cast<const HostVector<bool>*>(&S);
        const HostMatrixCSR<ValueType>* cast_gst
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&ghost);
        const HostVector<PtrType>* cast_ptr
            = dynamic_cast<const HostVector<PtrType>*>(&bnd_csr_row_ptr);
        const HostVector<int64_t>* cast_col
            = dynamic_cast<const HostVector<int64_t>*>(&bnd_csr_col_ind);
        HostVector<int>*          cast_f2c = dynamic_cast<HostVector<int>*>(f2c);
        HostMatrixCSR<ValueType>* cast_pi  = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong_int);
        HostMatrixCSR<ValueType>* cast_pg  = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong_gst);

        assert(cast_cf != NULL);
        assert(cast_S != NULL);
        assert(cast_f2c != NULL);
        assert(cast_pi != NULL);

        // Do we need communication?
        bool global = prolong_gst != NULL;

        // Start with fresh P
        cast_pi->Clear();

        // Allocate P row pointer array
        allocate_host(this->nrow_ + 1, &cast_pi->mat_.row_offset);

        // We already know the number of rows of P
        cast_pi->nrow_ = this->nrow_;

        // Ghost part
        if(global == true)
        {
            // LCOV_EXCL_START
            assert(cast_l2g != NULL);
            assert(cast_gst != NULL);
            assert(cast_ptr != NULL);
            assert(cast_col != NULL);
            assert(cast_pg != NULL);

            // Start with fresh P ghost
            cast_pg->Clear();

            // Allocate P ghost row pointer array
            allocate_host(this->nrow_ + 1, &cast_pg->mat_.row_offset);

            // Number of ghost rows is identical to interior
            cast_pg->nrow_ = this->nrow_;
            // LCOV_EXCL_STOP
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        // Determine number of non-zeros for P
        for(int row = 0; row < this->nrow_; ++row)
        {
            // Coarse points generate a single entry
            if(cast_cf->vec_[row] == 1)
            {
                // Set this points state to coarse
                cast_f2c->vec_[row] = 1;

                // Set row nnz
                cast_pi->mat_.row_offset[row] = 1;

                if(global == true)
                {
                    cast_pg->mat_.row_offset[row] = 0;
                }

                continue;
            }

            // Set, to discard duplicated column entries (no need to be ordered)
            std::unordered_set<int>     int_set;
            std::unordered_set<int64_t> gst_set;

            // Row entry and exit points
            PtrType row_begin = this->mat_.row_offset[row];
            PtrType row_end   = this->mat_.row_offset[row + 1];

            // Loop over all columns of the i-th row, whereas each lane processes a column
            for(PtrType j = row_begin; j < row_end; ++j)
            {
                // Skip points that do not influence the current point
                if(cast_S->vec_[j] == false)
                {
                    continue;
                }

                // Get the column index
                int col_j = this->mat_.col[j];

                // Skip diagonal entries (i does not influence itself)
                if(col_j == row)
                {
                    continue;
                }

                // Switch between coarse and fine points that influence the i-th point
                if(cast_cf->vec_[col_j] == 1)
                {
                    // This is a coarse point and thus contributes, count it for the row nnz
                    // We need to use a set here, to discard duplicates.
                    int_set.insert(col_j);
                }
                else
                {
                    // This is a fine point, check for strongly connected coarse points

                    bool skip_ghost = false;

                    // Row entry and exit of this fine point
                    PtrType row_begin_j = this->mat_.row_offset[col_j];
                    PtrType row_end_j   = this->mat_.row_offset[col_j + 1];

                    // Loop over all columns of the fine point
                    for(PtrType k = row_begin_j; k < row_end_j; ++k)
                    {
                        // Skip points that do not influence the fine point
                        if(cast_S->vec_[k] == false)
                        {
                            continue;
                        }

                        // Get the column index
                        int col_k = this->mat_.col[k];

                        // Skip diagonal entries (the fine point does not influence itself)
                        if(col_k == col_j)
                        {
                            continue;
                        }

                        // Check whether k is a coarse point
                        if(cast_cf->vec_[col_k] == 1)
                        {
                            // This is a coarse point, it contributes, count it for the row nnz
                            // We need to use a set here, to discard duplicates.
                            int_set.insert(col_k);

                            // Stop if FF interpolation is limited
                            if(FF1 == true)
                            {
                                skip_ghost = true;
                                break;
                            }
                        }
                    }

                    if(skip_ghost == false && global == true)
                    {
                        // Row entry and exit of this fine point
                        row_begin_j = cast_gst->mat_.row_offset[col_j];
                        row_end_j   = cast_gst->mat_.row_offset[col_j + 1];

                        // Ghost iterate over the range of columns of B.
                        for(PtrType k = row_begin_j; k < row_end_j; ++k)
                        {
                            // Skip points that do not influence the fine point
                            if(cast_S->vec_[k + this->nnz_] == false)
                            {
                                continue;
                            }

                            // Check whether k is a coarse point
                            int col_k = cast_gst->mat_.col[k];

                            // Check whether k is a coarse point
                            if(cast_cf->vec_[col_k + this->nrow_] == 1)
                            {
                                // Get (global) column index
                                int64_t gcol_k = cast_l2g->vec_[col_k] + global_column_end
                                                 - global_column_begin;

                                // This is a coarse point, it contributes, count it for the row int_nnz
                                // We need to use a set here, to discard duplicates.
                                gst_set.insert(gcol_k);
                            }
                        }
                    }
                }
            }

            if(global == true)
            {
                // Row entry and exit points
                row_begin = cast_gst->mat_.row_offset[row];
                row_end   = cast_gst->mat_.row_offset[row + 1];

                // Loop over all columns of the i-th row, whereas each lane processes a column
                for(PtrType j = row_begin; j < row_end; ++j)
                {
                    // Skip points that do not influence the current point
                    if(cast_S->vec_[j + this->nnz_] == false)
                    {
                        continue;
                    }

                    // Get the column index
                    int col_j = cast_gst->mat_.col[j];

                    // Switch between coarse and fine points that influence the i-th point
                    if(cast_cf->vec_[col_j + this->nrow_] == 1)
                    {
                        // Get (global) column index
                        int64_t gcol_j
                            = cast_l2g->vec_[col_j] + global_column_end - global_column_begin;

                        // This is a coarse point and thus contributes, count it for the row int_nnz
                        // We need to use a set here, to discard duplicates.
                        gst_set.insert(gcol_j);
                    }
                    else
                    {
                        // This is a fine point, check for strongly connected coarse points

                        // Row entry and exit of this fine point
                        PtrType row_begin_j = cast_ptr->vec_[col_j];
                        PtrType row_end_j   = cast_ptr->vec_[col_j + 1];

                        // Loop over all columns of the fine point
                        for(PtrType k = row_begin_j; k < row_end_j; ++k)
                        {
                            // Get the (global) column index
                            int64_t gcol_k = cast_col->vec_[k];

                            // Differentiate between local and ghost column
                            if(gcol_k >= global_column_begin && gcol_k < global_column_end)
                            {
                                // Get (local) column index
                                int col_k = static_cast<int>(gcol_k - global_column_begin);

                                // This is a coarse point, it contributes, count it for the row nnz
                                // We need to use a set here, to discard duplicates.
                                int_set.insert(col_k);

                                // Stop if FF interpolation is limited
                                if(FF1 == true)
                                {
                                    break;
                                }
                            }
                            else
                            {
                                // This is a coarse point, it contributes, count it for the row nnz
                                // We need to use a set here, to discard duplicates.
                                gst_set.insert(gcol_k + global_column_end - global_column_begin);

                                // Stop if FF interpolation is limited
                                if(FF1 == true)
                                {
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            // Write row nnz back to global memory
            cast_pi->mat_.row_offset[row] = int_set.size();

            if(global == true)
            {
                cast_pg->mat_.row_offset[row] = gst_set.size();
            }

            // Set this points state to fine
            cast_f2c->vec_[row] = 0;
        }

        cast_f2c->ExclusiveSum(*cast_f2c);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::RSExtPIProlongFill(int64_t global_column_begin,
                                                      int64_t global_column_end,
                                                      bool    FF1,
                                                      const BaseVector<int64_t>&   l2g,
                                                      const BaseVector<int>&       f2c,
                                                      const BaseVector<int>&       CFmap,
                                                      const BaseVector<bool>&      S,
                                                      const BaseMatrix<ValueType>& ghost,
                                                      const BaseVector<PtrType>&   bnd_csr_row_ptr,
                                                      const BaseVector<int64_t>&   bnd_csr_col_ind,
                                                      const BaseVector<PtrType>&   ext_csr_row_ptr,
                                                      const BaseVector<int64_t>&   ext_csr_col_ind,
                                                      const BaseVector<ValueType>& ext_csr_val,
                                                      BaseMatrix<ValueType>*       prolong_int,
                                                      BaseMatrix<ValueType>*       prolong_gst,
                                                      BaseVector<int64_t>* global_ghost_col) const
    {
        const HostVector<int64_t>*      cast_l2g = dynamic_cast<const HostVector<int64_t>*>(&l2g);
        const HostVector<int>*          cast_f2c = dynamic_cast<const HostVector<int>*>(&f2c);
        const HostVector<int>*          cast_cf  = dynamic_cast<const HostVector<int>*>(&CFmap);
        const HostVector<bool>*         cast_S   = dynamic_cast<const HostVector<bool>*>(&S);
        const HostMatrixCSR<ValueType>* cast_gst
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&ghost);
        const HostVector<PtrType>* cast_ptr
            = dynamic_cast<const HostVector<PtrType>*>(&bnd_csr_row_ptr);
        const HostVector<int64_t>* cast_col
            = dynamic_cast<const HostVector<int64_t>*>(&bnd_csr_col_ind);
        const HostVector<PtrType>* cast_ext_ptr
            = dynamic_cast<const HostVector<PtrType>*>(&ext_csr_row_ptr);
        const HostVector<int64_t>* cast_ext_col
            = dynamic_cast<const HostVector<int64_t>*>(&ext_csr_col_ind);
        const HostVector<ValueType>* cast_ext_val
            = dynamic_cast<const HostVector<ValueType>*>(&ext_csr_val);
        HostMatrixCSR<ValueType>* cast_pi  = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong_int);
        HostMatrixCSR<ValueType>* cast_pg  = dynamic_cast<HostMatrixCSR<ValueType>*>(prolong_gst);
        HostVector<int64_t>*      cast_glo = dynamic_cast<HostVector<int64_t>*>(global_ghost_col);

        assert(cast_f2c != NULL);
        assert(cast_cf != NULL);
        assert(cast_S != NULL);
        assert(cast_pi != NULL);

        // Do we need communication?
        bool global = prolong_gst != NULL;

        // Ghost part
        if(global == true)
        {
            // LCOV_EXCL_START
            assert(cast_l2g != NULL);
            assert(cast_gst != NULL);
            assert(cast_ptr != NULL);
            assert(cast_col != NULL);
            assert(cast_ext_ptr != NULL);
            assert(cast_ext_col != NULL);
            assert(cast_ext_val != NULL);
            assert(cast_pg != NULL);
            assert(cast_glo != NULL);
            // LCOV_EXCL_STOP
        }

        // Exclusive sum to obtain row offset pointers of P
        // P contains only nnz per row, so far
        for(int i = this->nrow_; i > 0; --i)
        {
            cast_pi->mat_.row_offset[i] = cast_pi->mat_.row_offset[i - 1];
        }

        cast_pi->mat_.row_offset[0] = 0;

        for(int i = 0; i < this->nrow_; ++i)
        {
            cast_pi->mat_.row_offset[i + 1] += cast_pi->mat_.row_offset[i];
        }

        // Initialize nnz of P
        cast_pi->nnz_ = cast_pi->mat_.row_offset[this->nrow_];

        // Initialize ncol of P
        cast_pi->ncol_ = cast_f2c->vec_[this->nrow_];

        // Allocate column and value arrays
        allocate_host(cast_pi->nnz_, &cast_pi->mat_.col);
        allocate_host(cast_pi->nnz_, &cast_pi->mat_.val);

        // Once again for the ghost part
        if(global == true)
        {
            // LCOV_EXCL_START
            for(int i = this->nrow_; i > 0; --i)
            {
                cast_pg->mat_.row_offset[i] = cast_pg->mat_.row_offset[i - 1];
            }

            cast_pg->mat_.row_offset[0] = 0;

            for(int i = 0; i < this->nrow_; ++i)
            {
                cast_pg->mat_.row_offset[i + 1] += cast_pg->mat_.row_offset[i];
            }

            cast_pg->nnz_  = cast_pg->mat_.row_offset[this->nrow_];
            cast_pg->ncol_ = this->nrow_;

            allocate_host(cast_pg->nnz_, &cast_pg->mat_.col);
            allocate_host(cast_pg->nnz_, &cast_pg->mat_.val);

            cast_glo->Allocate(cast_pg->nnz_);
            // LCOV_EXCL_STOP
        }

        // Extract diagonal matrix entries
        HostVector<ValueType> diag(this->local_backend_);
        diag.Allocate(this->nrow_);

        this->ExtractDiagonal(&diag);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        // Fill column indices and values of P
        for(int row = 0; row < this->nrow_; ++row)
        {
            // Some helpers for readability
            constexpr ValueType zero = static_cast<ValueType>(0);

            // Coarse points generate a single entry
            if(cast_cf->vec_[row] == 1)
            {
                // Get index into P
                PtrType idx = cast_pi->mat_.row_offset[row];

                // Single entry in this row (coarse point)
                cast_pi->mat_.col[idx] = cast_f2c->vec_[row];
                cast_pi->mat_.val[idx] = static_cast<ValueType>(1);

                continue;
            }

            // Hash table
            std::map<int, ValueType>     int_table;
            std::map<int64_t, ValueType> gst_table;

            // Fill the hash table according to the nnz pattern of P
            // This is identical to the nnz per row part

            // Row entry and exit points
            PtrType row_begin = this->mat_.row_offset[row];
            PtrType row_end   = this->mat_.row_offset[row + 1];

            // Loop over all columns of the i-th row
            for(PtrType k = row_begin; k < row_end; ++k)
            {
                // Skip points that do not influence the current point
                if(cast_S->vec_[k] == false)
                {
                    continue;
                }

                // Get the column index
                int col_ik = this->mat_.col[k];

                // Skip diagonal entries (i does not influence itself)
                if(col_ik == row)
                {
                    continue;
                }

                // Switch between coarse and fine points that influence the i-th point
                if(cast_cf->vec_[col_ik] == 1)
                {
                    // This is a coarse point and thus contributes
                    int_table[col_ik] = zero;
                }
                else
                {
                    // This is a fine point, check for strongly connected coarse points

                    bool skip_ghost = false;

                    // Row entry and exit of this fine point
                    PtrType row_begin_k = this->mat_.row_offset[col_ik];
                    PtrType row_end_k   = this->mat_.row_offset[col_ik + 1];

                    // Loop over all columns of the fine point
                    for(PtrType l = row_begin_k; l < row_end_k; ++l)
                    {
                        // Skip points that do not influence the fine point
                        if(cast_S->vec_[l] == false)
                        {
                            continue;
                        }

                        // Get the column index
                        int col_kl = this->mat_.col[l];

                        // Skip diagonal entries (the fine point does not influence itself)
                        if(col_kl == col_ik)
                        {
                            continue;
                        }

                        // Check whether l is a coarse point
                        if(cast_cf->vec_[col_kl] == 1)
                        {
                            // This is a coarse point, it contributes
                            int_table[col_kl] = zero;

                            // Stop if FF interpolation is limited
                            if(FF1 == true)
                            {
                                skip_ghost = true;
                                break;
                            }
                        }
                    }

                    if(skip_ghost == false && global == true)
                    {
                        // Loop over all ghost columns of the fine point
                        for(PtrType l = cast_gst->mat_.row_offset[col_ik];
                            l < cast_gst->mat_.row_offset[col_ik + 1];
                            ++l)
                        {
                            if(cast_S->vec_[l + this->nnz_] == false)
                            {
                                continue;
                            }

                            // Get the column index
                            int col_kl = cast_gst->mat_.col[l];

                            // Check whether l is a coarse point
                            if(cast_cf->vec_[col_kl + this->nrow_] == 1)
                            {
                                // This is a coarse point, it contributes

                                // Global column shifted by local columns
                                int64_t gcol_kl = cast_l2g->vec_[col_kl] + global_column_end
                                                  - global_column_begin;
                                gst_table[gcol_kl] = zero;
                            }
                        }
                    }
                }
            }

            if(global == true)
            {
                for(PtrType k = cast_gst->mat_.row_offset[row];
                    k < cast_gst->mat_.row_offset[row + 1];
                    ++k)
                {
                    // Skip points that do not influence the current point
                    if(cast_S->vec_[k + this->nnz_] == false)
                    {
                        continue;
                    }

                    // Get the column index
                    int col_ik = cast_gst->mat_.col[k];

                    // Switch between coarse and fine points that influence the i-th point
                    if(cast_cf->vec_[col_ik + this->nrow_] == 1)
                    {
                        // Explicitly create an entry in the hash table
                        gst_table[cast_l2g->vec_[col_ik] + global_column_end - global_column_begin]
                            = zero;
                    }
                    else
                    {
                        for(PtrType l = cast_ptr->vec_[col_ik]; l < cast_ptr->vec_[col_ik + 1]; ++l)
                        {
                            // Get the (global) column index
                            int64_t gcol_kl = cast_col->vec_[l];

                            // Differentiate between local and ghost column
                            if(gcol_kl >= global_column_begin && gcol_kl < global_column_end)
                            {
                                int col_kl = static_cast<int>(gcol_kl - global_column_begin);

                                // This is a coarse point, it contributes, count it for the row nnz
                                // We need to use a set here, to discard duplicates.
                                int_table[col_kl] = zero;

                                // Stop if FF interpolation is limited
                                if(FF1 == true)
                                {
                                    break;
                                }
                            }
                            else
                            {
                                // This is a coarse point, it contributes, count it for the row nnz
                                // We need to use a set here, to discard duplicates.
                                gst_table[gcol_kl + global_column_end - global_column_begin] = zero;

                                // Stop if FF interpolation is limited
                                if(FF1 == true)
                                {
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            // Now, we need to do the numerical part

            // Diagonal entry of i-th row
            ValueType val_ii = diag.vec_[row];

            // Sign of diagonal entry of i-th row
            bool pos_ii = val_ii >= zero;

            // Accumulators
            ValueType sum_k = zero;
            ValueType sum_n = zero;

            // Loop over all columns of the i-th row
            for(PtrType k = row_begin; k < row_end; ++k)
            {
                // Get the column index
                int col_ik = this->mat_.col[k];

                // Skip diagonal entries (i does not influence itself)
                if(col_ik == row)
                {
                    continue;
                }

                // Get the column value
                ValueType val_ik = this->mat_.val[k];

                // Check, whether the k-th entry of the row is a fine point and strongly
                // connected to the i-th point (e.g. k \in F^S_i)
                if(cast_S->vec_[k] == true && cast_cf->vec_[col_ik] == 2)
                {
                    // Accumulator for the sum over l
                    ValueType sum_l = zero;

                    // Diagonal entry of k-th row
                    ValueType val_kk = diag.vec_[col_ik];

                    // Store a_ki, if present
                    ValueType val_ki = zero;

                    // Row entry and exit of this fine point
                    PtrType row_begin_k = this->mat_.row_offset[col_ik];
                    PtrType row_end_k   = this->mat_.row_offset[col_ik + 1];

                    // Loop over all columns of the fine point
                    for(PtrType l = row_begin_k; l < row_end_k; ++l)
                    {
                        // Get the column index
                        int col_kl = this->mat_.col[l];

                        // Get the column value
                        ValueType val_kl = this->mat_.val[l];

                        // Sign of a_kl
                        bool pos_kl = val_kl >= zero;

                        // Differentiate between diagonal and off-diagonal
                        if(col_kl == row)
                        {
                            // Column that matches the i-th row
                            // Since we sum up all l in C^hat_i and i, the diagonal need to
                            // be added to the sum over l, e.g. a^bar_kl
                            // a^bar contributes only, if the sign is different to the
                            // i-th row diagonal sign.
                            if(pos_ii != pos_kl)
                            {
                                sum_l += val_kl;
                            }

                            // If a_ki exists, keep it for later
                            val_ki = val_kl;
                        }
                        else if(cast_cf->vec_[col_kl] == 1)
                        {
                            // Check if sign is different from i-th row diagonal
                            if(pos_ii != pos_kl)
                            {
                                // Entry contributes only, if it is a coarse point
                                // and part of C^hat (e.g. we need to check the hash table)
                                if(int_table.find(col_kl) != int_table.end())
                                {
                                    sum_l += val_kl;
                                }
                            }
                        }
                    }

                    if(global == true)
                    {
                        // Loop over all columns of the fine point
                        for(PtrType l = cast_gst->mat_.row_offset[col_ik];
                            l < cast_gst->mat_.row_offset[col_ik + 1];
                            ++l)
                        {
                            // Get the column index
                            int col_kl = cast_gst->mat_.col[l];

                            // Get the (global) column index
                            int64_t gcol_kl
                                = cast_l2g->vec_[col_kl] + global_column_end - global_column_begin;

                            // Get the column value
                            ValueType val_kl = cast_gst->mat_.val[l];

                            // Sign of a_kl
                            bool pos_kl = val_kl >= zero;

                            // Only coarse points contribute
                            if(cast_cf->vec_[col_kl + this->nrow_] == 1)
                            {
                                // Check if sign is different from i-th row diagonal
                                if(pos_ii != pos_kl)
                                {
                                    // Entry contributes only if it is part of C^hat
                                    // (e.g. we need to check the hash table)
                                    if(gst_table.find(gcol_kl) != gst_table.end())
                                    {
                                        sum_l += val_kl;
                                    }
                                }
                            }
                        }
                    }

                    // Update sum over l with a_ik
                    sum_l = val_ik / sum_l;

                    // Compute the sign of a_kk and a_ki, we need this for a_bar
                    bool pos_kk = val_kk >= zero;
                    bool pos_ki = val_ki >= zero;

                    // Additionally, for eq19 we need to add all coarse points in row k,
                    // if they have different sign than the diagonal a_kk
                    for(PtrType l = row_begin_k; l < row_end_k; ++l)
                    {
                        // Get the column index
                        int col_kl = this->mat_.col[l];

                        // Only coarse points contribute
                        if(cast_cf->vec_[col_kl] != 1)
                        {
                            continue;
                        }

                        // Get the column value
                        ValueType val_kl = this->mat_.val[l];

                        // Compute the sign of a_kl
                        bool pos_kl = val_kl >= zero;

                        // Check for different sign
                        if(pos_kk != pos_kl)
                        {
                            if(int_table.find(col_kl) != int_table.end())
                            {
                                int_table[col_kl] += val_kl * sum_l;
                            }
                        }
                    }

                    if(global == true)
                    {
                        // Row entry and exit of this fine point
                        row_begin_k = cast_gst->mat_.row_offset[col_ik];
                        row_end_k   = cast_gst->mat_.row_offset[col_ik + 1];

                        // Additionally, for eq19 we need to add all coarse points in row k,
                        // if they have different sign than the diagonal a_kk
                        for(PtrType l = row_begin_k; l < row_end_k; ++l)
                        {
                            // Get the column index
                            int col_kl = cast_gst->mat_.col[l];

                            // Get the (global) column index
                            int64_t gcol_kl
                                = cast_l2g->vec_[col_kl] + global_column_end - global_column_begin;

                            // Get the column value
                            ValueType val_kl = cast_gst->mat_.val[l];

                            // Compute the sign of a_kl
                            bool pos_kl = val_kl >= zero;

                            // Check for different sign
                            if(pos_kk != pos_kl)
                            {
                                if(gst_table.find(gcol_kl) != gst_table.end())
                                {
                                    gst_table[gcol_kl] += val_kl * sum_l;
                                }
                            }
                        }
                    }

                    // If sign of a_ki and a_kk are different, a_ki contributes to the
                    // sum over k in F^S_i
                    if(pos_kk != pos_ki)
                    {
                        sum_k += val_ki * sum_l;
                    }
                }

                // Boolean, to flag whether a_ik is in C hat or not
                // (we can query the hash table for it)
                bool in_C_hat = false;

                // a_ik can only be in C^hat if it is coarse
                if(cast_cf->vec_[col_ik] == 1)
                {
                    // Check, whether col_ik is in C hat or not
                    if(int_table.find(col_ik) != int_table.end())
                    {
                        // Append a_ik to the sum of eq19
                        int_table[col_ik] += val_ik;

                        in_C_hat = true;
                    }
                }

                // If a_ik is not in C^hat and does not strongly influence i, it contributes
                // to sum_n
                if(in_C_hat == false && cast_S->vec_[k] == false)
                {
                    sum_n += val_ik;
                }
            }

            if(global == true)
            {
                // Loop over all columns of the i-th row
                for(PtrType k = cast_gst->mat_.row_offset[row];
                    k < cast_gst->mat_.row_offset[row + 1];
                    ++k)
                {
                    // Get the column index
                    int col_ik = cast_gst->mat_.col[k];

                    // Get the column value
                    ValueType val_ik = cast_gst->mat_.val[k];

                    // Check, whether the k-th entry of the row is a fine point and strongly
                    // connected to the i-th point (e.g. k \in F^S_i)
                    if(cast_S->vec_[k + this->nnz_] == true
                       && cast_cf->vec_[col_ik + this->nrow_] == 2)
                    {
                        // Accumulator for the sum over l
                        ValueType sum_l = zero;

                        // Diagonal element of k-th row
                        ValueType val_kk = zero;

                        // Global column index
                        int64_t grow_k = cast_l2g->vec_[col_ik];

                        // Row entry and exit of this fine point
                        PtrType row_begin_k = cast_ext_ptr->vec_[col_ik];
                        PtrType row_end_k   = cast_ext_ptr->vec_[col_ik + 1];

                        // Loop over all columns of the fine point
                        for(PtrType l = row_begin_k; l < row_end_k; ++l)
                        {
                            // Get the (global) column index
                            int64_t gcol_kl = cast_ext_col->vec_[l];

                            // Get the column value
                            ValueType val_kl = cast_ext_val->vec_[l];

                            // Sign of a_kl
                            bool pos_kl = val_kl >= zero;

                            // Extract diagonal value
                            if(grow_k == gcol_kl)
                            {
                                val_kk = val_kl;
                            }

                            // Differentiate between local and ghost column
                            if(gcol_kl >= global_column_begin && gcol_kl < global_column_end)
                            {
                                // Get the (local) column index
                                int col_kl = static_cast<int>(gcol_kl - global_column_begin);

                                // Differentiate between diagonal and off-diagonal
                                if(col_kl == row)
                                {
                                    // Column that matches the i-th row
                                    // Since we sum up all l in C^hat_i and i, the diagonal need to
                                    // be added to the sum over l, e.g. a^bar_kl
                                    // a^bar contributes only, if the sign is different to the
                                    // i-th row diagonal sign.
                                    if(pos_ii != pos_kl)
                                    {
                                        sum_l += val_kl;
                                    }
                                }
                                else
                                {
                                    // Check if sign is different from i-th row diagonal
                                    if(pos_ii != pos_kl)
                                    {
                                        // Entry contributes only if it is part of C^hat
                                        // (e.g. we need to check the hash table)
                                        if(int_table.find(col_kl) != int_table.end())
                                        {
                                            sum_l += val_kl;
                                        }
                                    }
                                }
                            }
                            else
                            {
                                // Check if sign is different from i-th row diagonal
                                if(pos_ii != pos_kl)
                                {
                                    if(gst_table.find(gcol_kl + global_column_end
                                                      - global_column_begin)
                                       != gst_table.end())
                                    {
                                        sum_l += val_kl;
                                    }
                                }
                            }
                        }

                        // Store a_ki, if present
                        ValueType val_ki = zero;

                        // Compute the k-th sum
                        sum_l = val_ik / sum_l;

                        // Loop over all columns of A ext
                        for(PtrType l = row_begin_k; l < row_end_k; ++l)
                        {
                            // Get the (global) column index
                            int64_t gcol_kl = cast_ext_col->vec_[l];

                            // Get the column value
                            ValueType val_kl = cast_ext_val->vec_[l];

                            // Sign of a_kl
                            bool pos_kl = val_kl >= zero;

                            if((val_kk >= zero) == pos_kl)
                            {
                                val_kl = zero;
                            }

                            // Differentiate between local and ghost column
                            if(gcol_kl >= global_column_begin && gcol_kl < global_column_end)
                            {
                                // Get the (local) column index
                                int col_kl = static_cast<int>(gcol_kl - global_column_begin);

                                // Differentiate between diagonal and off-diagonal
                                if(row == col_kl)
                                {
                                    // If a_ki exists, keep it for later
                                    val_ki = val_kl;
                                }

                                // Entry contributes only if it is part of the hash table
                                if(int_table.find(col_kl) != int_table.end())
                                {
                                    int_table[col_kl] += val_kl * sum_l;
                                }
                            }
                            else
                            {
                                // Global column
                                if(gst_table.find(gcol_kl + global_column_end - global_column_begin)
                                   != gst_table.end())
                                {
                                    gst_table[gcol_kl + global_column_end - global_column_begin]
                                        += val_kl * sum_l;
                                }
                            }
                        }

                        sum_n += val_ki * sum_l;
                    }

                    // Map global id
                    int64_t gcol_ik
                        = cast_l2g->vec_[col_ik] + global_column_end - global_column_begin;

                    // Boolean, to flag whether a_ik is in C hat or not
                    // (we can query the hash table for it)
                    bool in_C_hat = false;

                    // Check, whether global col_ik is in C hat or not
                    if(gst_table.find(gcol_ik) != gst_table.end())
                    {
                        // Append a_ik to the sum of eq19
                        gst_table[gcol_ik] += val_ik;

                        in_C_hat = true;
                    }

                    // If a_ik is not in C^hat and does not strongly influence i, it contributes
                    // to the sum
                    if(cast_S->vec_[k + this->nnz_] == false && in_C_hat == false)
                    {
                        sum_k += val_ik;
                    }
                }
            }

            // Precompute a_ii_tilde
            ValueType a_ii_tilde = static_cast<ValueType>(-1) / (sum_n + sum_k + val_ii);

            // Entry point into P (interior)
            PtrType int_idx = cast_pi->mat_.row_offset[row];

            // Finally, extract the numerical values from the hash table and fill P such
            // that the resulting matrix is sorted by columns
            for(auto it = int_table.begin(); it != int_table.end(); ++it)
            {
                cast_pi->mat_.col[int_idx] = cast_f2c->vec_[it->first];
                cast_pi->mat_.val[int_idx] = a_ii_tilde * it->second;
                ++int_idx;
            }

            if(global == true)
            {
                // Entry point into P (ghost)
                PtrType gst_idx = cast_pg->mat_.row_offset[row];

                // Finally, extract the numerical values from the hash table and fill P such
                // that the resulting matrix is sorted by columns
                for(auto it = gst_table.begin(); it != gst_table.end(); ++it)
                {
                    cast_glo->vec_[gst_idx] = it->first - global_column_end + global_column_begin;
                    cast_pg->mat_.val[gst_idx] = a_ii_tilde * it->second;
                    ++gst_idx;
                }
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::InitialPairwiseAggregation(ValueType        beta,
                                                              int&             nc,
                                                              BaseVector<int>* G,
                                                              int&             Gsize,
                                                              int**            rG,
                                                              int&             rGsize,
                                                              int              ordering) const
    {
        assert(G != NULL);

        HostVector<int>* cast_G = dynamic_cast<HostVector<int>*>(G);

        assert(cast_G != NULL);

        // Initialize G
        for(int i = 0; i < cast_G->size_; ++i)
        {
            cast_G->vec_[i] = -2;
        }

        int      Usize    = 0;
        PtrType* ind_diag = NULL;
        allocate_host(this->nrow_, &ind_diag);

        // Build U
        for(int i = 0; i < this->nrow_; ++i)
        {
            ValueType sum = static_cast<ValueType>(0);

            for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                if(i != this->mat_.col[j])
                {
                    sum += std::abs(this->mat_.val[j]);
                }
                else
                {
                    ind_diag[i] = j;
                }
            }

            sum *= static_cast<ValueType>(5);

            if(this->mat_.val[ind_diag[i]] > sum)
            {
                cast_G->vec_[i] = -1;
                ++Usize;
            }
        }

        // Initialize rG and sizes
        Gsize  = 2;
        rGsize = this->nrow_ - Usize;
        allocate_host(Gsize * rGsize, rG);

        for(int i = 0; i < Gsize * rGsize; ++i)
        {
            (*rG)[i] = -1;
        }

        nc              = 0;
        ValueType betam = -beta;

        // Ordering
        HostVector<int> perm(this->local_backend_);

        switch(ordering)
        {
        case 0: // No ordering
            break;

        case 1: // Connectivity ordering
            this->ConnectivityOrder(&perm);
            break;

        case 2: // CMK
            this->CMK(&perm);
            break;

        case 3: // RCMK
            this->RCMK(&perm);
            break;

        case 4: // MIS
            int size;
            this->MaximalIndependentSet(size, &perm);
            break;

        case 5: // MultiColoring
            int  num_colors;
            int* size_colors = NULL;
            this->MultiColoring(num_colors, &size_colors, &perm);
            free_host(&size_colors);
            break;
        }

        // Main algorithm

        // While U != empty
        for(int k = 0; k < this->nrow_; ++k)
        {
            // Pick i according to the connectivity
            int i;
            if(ordering == 0)
            {
                i = k;
            }
            else
            {
                i = perm.vec_[k];
            }

            // Check if i in G
            if(cast_G->vec_[i] != -2)
            {
                continue;
            }

            // Mark i visited and fill G and rG
            cast_G->vec_[i] = nc;
            (*rG)[nc]       = i;

            // Determine minimum and maximum offdiagonal entry and j index
            ValueType min_a_ij = static_cast<ValueType>(0);
            ValueType max_a_ij = static_cast<ValueType>(0);
            PtrType   min_j    = -1;
            bool      neg      = false;
            if(this->mat_.val[ind_diag[i]] < static_cast<ValueType>(0))
            {
                neg = true;
            }

            for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                int       col_j = this->mat_.col[j];
                ValueType val_j = this->mat_.val[j];

                if(neg == true)
                {
                    val_j *= static_cast<ValueType>(-1);
                }

                if(i == col_j)
                {
                    continue;
                }

                if(min_j == -1)
                {
                    max_a_ij = val_j;
                    if(cast_G->vec_[col_j] == -2)
                    {
                        min_j    = j;
                        min_a_ij = val_j;
                    }
                }

                if(val_j < min_a_ij && cast_G->vec_[col_j] == -2)
                {
                    min_a_ij = val_j;
                    min_j    = j;
                }

                if(val_j > max_a_ij)
                {
                    max_a_ij = val_j;
                }
            }

            // Fill G
            if(min_j != -1)
            {
                max_a_ij *= betam;

                int       col_j = this->mat_.col[min_j];
                ValueType val_j = this->mat_.val[min_j];

                if(neg == true)
                {
                    val_j *= static_cast<ValueType>(-1);
                }

                // Add j if conditions are fulfilled
                if(val_j < max_a_ij)
                {
                    cast_G->vec_[col_j] = nc;
                    (*rG)[rGsize + nc]  = col_j;
                }
            }

            ++nc;
        }

        free_host(&ind_diag);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::InitialPairwiseAggregation(const BaseMatrix<ValueType>& mat,
                                                              ValueType                    beta,
                                                              int&                         nc,
                                                              BaseVector<int>*             G,
                                                              int&                         Gsize,
                                                              int**                        rG,
                                                              int&                         rGsize,
                                                              int ordering) const
    {
        assert(G != NULL);

        HostVector<int>*                cast_G = dynamic_cast<HostVector<int>*>(G);
        const HostMatrixCSR<ValueType>* cast_mat
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&mat);

        assert(cast_G != NULL);
        assert(cast_mat != NULL);

        // Initialize G
        for(int i = 0; i < cast_G->size_; ++i)
        {
            cast_G->vec_[i] = -2;
        }

        int      Usize    = 0;
        PtrType* ind_diag = NULL;
        allocate_host(this->nrow_, &ind_diag);

        // Build U
        for(int i = 0; i < this->nrow_; ++i)
        {
            ValueType sum = static_cast<ValueType>(0);

            for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                if(i != this->mat_.col[j])
                {
                    sum += std::abs(this->mat_.val[j]);
                }
                else
                {
                    ind_diag[i] = j;
                }
            }

            if(cast_mat->nnz_ > 0)
            {
                for(PtrType j = cast_mat->mat_.row_offset[i]; j < cast_mat->mat_.row_offset[i + 1];
                    ++j)
                {
                    sum += std::abs(cast_mat->mat_.val[j]);
                }
            }

            sum *= static_cast<ValueType>(5);

            if(this->mat_.val[ind_diag[i]] > sum)
            {
                cast_G->vec_[i] = -1;
                ++Usize;
            }
        }

        // Initialize rG and sizes
        Gsize  = 2;
        rGsize = this->nrow_ - Usize;
        allocate_host(Gsize * rGsize, rG);

        for(int i = 0; i < Gsize * rGsize; ++i)
        {
            (*rG)[i] = -1;
        }

        nc              = 0;
        ValueType betam = -beta;

        // Ordering
        HostVector<int> perm(this->local_backend_);

        switch(ordering)
        {
        case 0: // No ordering
            break;

        case 1: // Connectivity ordering
            this->ConnectivityOrder(&perm);
            break;

        case 2: // CMK
            this->CMK(&perm);
            break;

        case 3: // RCMK
            this->RCMK(&perm);
            break;

        case 4: // MIS
            int size;
            this->MaximalIndependentSet(size, &perm);
            break;

        case 5: // MultiColoring
            int  num_colors;
            int* size_colors = NULL;
            this->MultiColoring(num_colors, &size_colors, &perm);
            free_host(&size_colors);
            break;
        }

        // Main algorithm

        // While U != empty
        for(int k = 0; k < this->nrow_; ++k)
        {
            // Pick i according to the connectivity
            int i;
            if(ordering == 0)
            {
                i = k;
            }
            else
            {
                i = perm.vec_[k];
            }

            // Check if i in G
            if(cast_G->vec_[i] != -2)
            {
                continue;
            }

            // Mark i visited and fill G and rG
            cast_G->vec_[i] = nc;
            (*rG)[nc]       = i;

            // Determine minimum and maximum offdiagonal entry and j index
            ValueType min_a_ij = static_cast<ValueType>(0);
            ValueType max_a_ij = static_cast<ValueType>(0);
            int       min_j    = -1;
            bool      neg      = false;
            if(this->mat_.val[ind_diag[i]] < static_cast<ValueType>(0))
            {
                neg = true;
            }

            for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                int       col_j = this->mat_.col[j];
                ValueType val_j = this->mat_.val[j];

                if(neg == true)
                {
                    val_j *= static_cast<ValueType>(-1);
                }

                if(i == col_j)
                {
                    continue;
                }

                if(min_j == -1)
                {
                    max_a_ij = val_j;
                    if(cast_G->vec_[col_j] == -2)
                    {
                        min_j    = col_j;
                        min_a_ij = val_j;
                    }
                }

                if(val_j < min_a_ij && cast_G->vec_[col_j] == -2)
                {
                    min_a_ij = val_j;
                    min_j    = col_j;
                }

                if(val_j > max_a_ij)
                {
                    max_a_ij = val_j;
                }
            }

            if(cast_mat->nnz_ > 0)
            {
                for(PtrType j = cast_mat->mat_.row_offset[i]; j < cast_mat->mat_.row_offset[i + 1];
                    ++j)
                {
                    ValueType val_j = cast_mat->mat_.val[j];

                    if(neg == true)
                    {
                        val_j *= static_cast<ValueType>(-1);
                    }

                    if(val_j > max_a_ij)
                    {
                        max_a_ij = val_j;
                    }
                }
            }

            // Fill G
            if(min_j != -1)
            {
                max_a_ij *= betam;

                for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
                {
                    int       col_j = this->mat_.col[j];
                    ValueType val_j = this->mat_.val[j];

                    if(neg == true)
                    {
                        val_j *= static_cast<ValueType>(-1);
                    }

                    // Skip diagonal
                    if(i == col_j)
                    {
                        continue;
                    }

                    // Skip j which is not in U
                    if(cast_G->vec_[col_j] != -2)
                    {
                        continue;
                    }

                    // Add j if conditions are fulfilled
                    if(val_j < max_a_ij)
                    {
                        if(min_j == col_j)
                        {
                            cast_G->vec_[min_j] = nc;
                            (*rG)[rGsize + nc]  = min_j;
                            break;
                        }
                    }
                }
            }

            ++nc;
        }

        free_host(&ind_diag);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::FurtherPairwiseAggregation(ValueType        beta,
                                                              int&             nc,
                                                              BaseVector<int>* G,
                                                              int&             Gsize,
                                                              int**            rG,
                                                              int&             rGsize,
                                                              int              ordering) const
    {
        assert(G != NULL);
        HostVector<int>* cast_G = dynamic_cast<HostVector<int>*>(G);
        assert(cast_G != NULL);

        // Initialize G and inverse indexing for G
        Gsize *= 2;
        int  rGsizec = this->nrow_;
        int* rGc     = NULL;
        allocate_host(Gsize * rGsizec, &rGc);

        for(int i = 0; i < Gsize * rGsizec; ++i)
        {
            rGc[i] = -1;
        }

        for(int i = 0; i < cast_G->size_; ++i)
        {
            cast_G->vec_[i] = -1;
        }

        // Initialize U
        int* U = NULL;
        allocate_host(this->nrow_, &U);
        set_to_zero_host(this->nrow_, U);

        nc              = 0;
        ValueType betam = -beta;

        // Ordering
        HostVector<int> perm(this->local_backend_);

        switch(ordering)
        {
        case 0: // No ordering
            break;

        case 1: // Connectivity ordering
            this->ConnectivityOrder(&perm);
            break;

        case 2: // CMK
            this->CMK(&perm);
            break;

        case 3: // RCMK
            this->RCMK(&perm);
            break;

        case 4: // MIS
            int size;
            this->MaximalIndependentSet(size, &perm);
            break;

        case 5: // MultiColoring
            int  num_colors;
            int* size_colors = NULL;
            this->MultiColoring(num_colors, &size_colors, &perm);
            free_host(&size_colors);
            break;
        }

        // While U != empty
        for(int k = 0; k < this->nrow_; ++k)
        {
            // Pick i according to connectivity
            int i;
            if(ordering == 0)
            {
                i = k;
            }
            else
            {
                i = perm.vec_[k];
            }

            // Check if i in U
            if(U[i] == 1)
            {
                continue;
            }

            // Mark i visited
            U[i] = 1;

            // Fill G and rG
            for(int r = 0; r < Gsize / 2; ++r)
            {
                rGc[r * rGsizec + nc] = (*rG)[r * rGsize + i];

                if((*rG)[r * rGsize + i] >= 0)
                {
                    cast_G->vec_[(*rG)[r * rGsize + i]] = nc;
                }
            }

            // Get sign
            bool neg = false;
            for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                if(i == this->mat_.col[j])
                {
                    if(this->mat_.val[j] < static_cast<ValueType>(0))
                    {
                        neg = true;
                    }

                    break;
                }
            }

            // Determine minimum and maximum offdiagonal entry and j index
            ValueType min_a_ij = static_cast<ValueType>(0);
            ValueType max_a_ij = static_cast<ValueType>(0);
            PtrType   min_j    = -1;

            for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                int       col_j = this->mat_.col[j];
                ValueType val_j = this->mat_.val[j];

                if(neg == true)
                {
                    val_j *= static_cast<ValueType>(-1);
                }

                if(i == col_j)
                {
                    continue;
                }

                if(min_j == -1)
                {
                    max_a_ij = val_j;

                    if(U[col_j] == 0)
                    {
                        min_j    = j;
                        min_a_ij = max_a_ij;
                    }
                }

                if(val_j < min_a_ij && U[col_j] == 0)
                {
                    min_a_ij = val_j;
                    min_j    = j;
                }

                if(val_j < max_a_ij)
                {
                    max_a_ij = val_j;
                }
            }

            // Fill G
            if(min_j != -1)
            {
                max_a_ij *= betam;

                int       col_j = this->mat_.col[min_j];
                ValueType val_j = this->mat_.val[min_j];

                if(neg == true)
                {
                    val_j *= static_cast<ValueType>(-1);
                }

                // Add j if conditions are fulfilled
                if(val_j < max_a_ij)
                {
                    for(int r = 0; r < Gsize / 2; ++r)
                    {
                        rGc[(r + Gsize / 2) * rGsizec + nc] = (*rG)[r * rGsize + col_j];

                        if((*rG)[r * rGsize + col_j] >= 0)
                        {
                            cast_G->vec_[(*rG)[r * rGsize + col_j]] = nc;
                        }
                    }

                    U[col_j] = 1;
                }
            }

            ++nc;
        }

        free_host(&U);
        free_host(rG);

        (*rG)  = rGc;
        rGsize = rGsizec;

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::FurtherPairwiseAggregation(const BaseMatrix<ValueType>& mat,
                                                              ValueType                    beta,
                                                              int&                         nc,
                                                              BaseVector<int>*             G,
                                                              int&                         Gsize,
                                                              int**                        rG,
                                                              int&                         rGsize,
                                                              int ordering) const
    {
        assert(G != NULL);

        HostVector<int>*                cast_G = dynamic_cast<HostVector<int>*>(G);
        const HostMatrixCSR<ValueType>* cast_mat
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&mat);

        assert(cast_G != NULL);
        assert(cast_mat != NULL);

        // Initialize G and inverse indexing for G
        Gsize *= 2;
        int  rGsizec = this->nrow_;
        int* rGc     = NULL;
        allocate_host(Gsize * rGsizec, &rGc);

        for(int i = 0; i < Gsize * rGsizec; ++i)
        {
            rGc[i] = -1;
        }

        for(int i = 0; i < cast_G->size_; ++i)
        {
            cast_G->vec_[i] = -1;
        }

        // Initialize U
        int* U = NULL;
        allocate_host(this->nrow_, &U);
        set_to_zero_host(this->nrow_, U);

        nc              = 0;
        ValueType betam = -beta;

        // Ordering
        HostVector<int> perm(this->local_backend_);

        switch(ordering)
        {
        case 0: // No ordering
            break;

        case 1: // Connectivity ordering
            this->ConnectivityOrder(&perm);
            break;

        case 2: // CMK
            this->CMK(&perm);
            break;

        case 3: // RCMK
            this->RCMK(&perm);
            break;

        case 4: // MIS
            int size;
            this->MaximalIndependentSet(size, &perm);
            break;

        case 5: // MultiColoring
            int  num_colors;
            int* size_colors = NULL;
            this->MultiColoring(num_colors, &size_colors, &perm);
            free_host(&size_colors);
            break;
        }

        // While U != empty
        for(int k = 0; k < this->nrow_; ++k)
        {
            // Pick i according to connectivity
            int i;
            if(ordering == 0)
            {
                i = k;
            }
            else
            {
                i = perm.vec_[k];
            }

            // Check if i in U
            if(U[i] == 1)
            {
                continue;
            }

            // Mark i visited
            U[i] = 1;

            // Fill G and rG
            for(int r = 0; r < Gsize / 2; ++r)
            {
                rGc[r * rGsizec + nc] = (*rG)[r * rGsize + i];

                if((*rG)[r * rGsize + i] >= 0)
                {
                    cast_G->vec_[(*rG)[r * rGsize + i]] = nc;
                }
            }

            // Get sign
            bool neg = false;
            for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                if(i == this->mat_.col[j])
                {
                    if(this->mat_.val[j] < static_cast<ValueType>(0))
                    {
                        neg = true;
                    }

                    break;
                }
            }

            // Determine minimum and maximum offdiagonal entry and j index
            ValueType min_a_ij = static_cast<ValueType>(0);
            ValueType max_a_ij = static_cast<ValueType>(0);
            int       min_j    = -1;

            for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
            {
                int       col_j = this->mat_.col[j];
                ValueType val_j = this->mat_.val[j];

                if(neg == true)
                {
                    val_j *= static_cast<ValueType>(-1);
                }

                if(i == col_j)
                {
                    continue;
                }

                if(min_j == -1)
                {
                    max_a_ij = val_j;
                    if(U[col_j] == 0)
                    {
                        min_j    = col_j;
                        min_a_ij = max_a_ij;
                    }
                }

                if(val_j < min_a_ij && U[col_j] == 0)
                {
                    min_a_ij = val_j;
                    min_j    = col_j;
                }

                if(val_j < max_a_ij)
                {
                    max_a_ij = val_j;
                }
            }

            if(cast_mat->nnz_ > 0)
            {
                for(PtrType j = cast_mat->mat_.row_offset[i]; j < cast_mat->mat_.row_offset[i + 1];
                    ++j)
                {
                    ValueType val_j = cast_mat->mat_.val[j];

                    if(neg == true)
                    {
                        val_j *= static_cast<ValueType>(-1);
                    }

                    if(val_j > max_a_ij)
                    {
                        max_a_ij = val_j;
                    }
                }
            }

            // Fill G
            if(min_j != -1)
            {
                max_a_ij *= betam;

                for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
                {
                    int       col_j = this->mat_.col[j];
                    ValueType val_j = this->mat_.val[j];

                    if(neg == true)
                    {
                        val_j *= static_cast<ValueType>(-1);
                    }

                    // Skip diagonal
                    if(i == col_j)
                    {
                        continue;
                    }

                    // Skip j which is not in U
                    if(U[col_j] == 1)
                    {
                        continue;
                    }

                    // Add j if conditions are fulfilled
                    if(val_j < max_a_ij)
                    {
                        if(min_j == col_j)
                        {
                            for(int r = 0; r < Gsize / 2; ++r)
                            {
                                rGc[(r + Gsize / 2) * rGsizec + nc] = (*rG)[r * rGsize + min_j];

                                if((*rG)[r * rGsize + min_j] >= 0)
                                {
                                    cast_G->vec_[(*rG)[r * rGsize + min_j]] = nc;
                                }
                            }

                            U[min_j] = 1;
                            break;
                        }
                    }
                }
            }

            ++nc;
        }

        free_host(&U);
        free_host(rG);

        (*rG)  = rGc;
        rGsize = rGsizec;

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::CoarsenOperator(BaseMatrix<ValueType>* Ac,
                                                   int                    nrow,
                                                   int                    ncol,
                                                   const BaseVector<int>& G,
                                                   int                    Gsize,
                                                   const int*             rG,
                                                   int                    rGsize) const
    {
        assert(Ac != NULL);

        HostMatrixCSR<ValueType>* cast_Ac = dynamic_cast<HostMatrixCSR<ValueType>*>(Ac);
        const HostVector<int>*    cast_G  = dynamic_cast<const HostVector<int>*>(&G);

        assert(cast_Ac != NULL);
        assert(cast_G != NULL);

        // Allocate
        cast_Ac->Clear();

        PtrType*   row_offset = NULL;
        int*       col        = NULL;
        ValueType* val        = NULL;

        allocate_host(nrow + 1, &row_offset);
        allocate_host(this->nnz_, &col);
        allocate_host(this->nnz_, &val);

        // Create P_ij: if i in G_j -> 1, else 0
        // (Ac)_kl = sum i in G_k (sum j in G_l (a_ij))) with k,l=1,...,nrow

        PtrType* reverse_col = NULL;
        int*     Gl          = NULL;
        int*     erase       = NULL;

        int size = (nrow > ncol) ? nrow : ncol;

        allocate_host(size, &reverse_col);
        allocate_host(size, &Gl);
        allocate_host(size, &erase);

        for(int i = 0; i < size; ++i)
        {
            reverse_col[i] = -1;
        }

        set_to_zero_host(size, Gl);

        row_offset[0] = 0;

        for(int k = 0; k < nrow; ++k)
        {
            row_offset[k + 1] = row_offset[k];

            int m = 0;

            for(int r = 0; r < Gsize; ++r)
            {
                int i = rG[r * rGsize + k];

                if(i < 0)
                {
                    continue;
                }

                for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
                {
                    int l = cast_G->vec_[this->mat_.col[j]];

                    if(l < 0)
                    {
                        continue;
                    }

                    if(Gl[l] == 0)
                    {
                        Gl[l]      = 1;
                        erase[m++] = l;

                        col[row_offset[k + 1]] = l;
                        val[row_offset[k + 1]] = this->mat_.val[j];
                        reverse_col[l]         = row_offset[k + 1];

                        ++row_offset[k + 1];
                    }
                    else
                    {
                        val[reverse_col[l]] += this->mat_.val[j];
                    }
                }
            }

            for(int j = 0; j < m; ++j)
            {
                Gl[erase[j]] = 0;
            }
        }

        free_host(&reverse_col);
        free_host(&Gl);
        free_host(&erase);

        PtrType nnz = row_offset[nrow];

        int*       col_resized = NULL;
        ValueType* val_resized = NULL;

        allocate_host(nnz, &col_resized);
        allocate_host(nnz, &val_resized);

        // Resize
        copy_h2h(nnz, col, col_resized);
        copy_h2h(nnz, val, val_resized);

        free_host(&col);
        free_host(&val);

        // Allocate
        cast_Ac->Clear();
        cast_Ac->SetDataPtrCSR(&row_offset, &col_resized, &val_resized, nnz, nrow, nrow);

        return true;
    }

    template <typename T>
    int sgn(T val)
    {
        return (T(0) < val) - (val < T(0));
    }

    template <typename ValueType>
    bool
        HostMatrixCSR<ValueType>::Key(long int& row_key, long int& col_key, long int& val_key) const
    {
        row_key = 0;
        col_key = 0;
        val_key = 0;

        int row_sign = 1;
        int col_sign = 1;
        int val_sign = 1;

        int row_tmp = 0x12345678;
        int col_tmp = 0x23456789;
        int val_tmp = 0x34567890;

        int row_mask = 0x09876543;
        int col_mask = 0x98765432;
        int val_mask = 0x87654321;

        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            row_key += row_sign * row_tmp * (row_mask & this->mat_.row_offset[ai]);
            row_key  = row_key ^ (row_key >> 16);
            row_sign = sgn(row_tmp - (row_mask & this->mat_.row_offset[ai]));
            row_tmp  = row_mask & this->mat_.row_offset[ai];

            PtrType row_beg = this->mat_.row_offset[ai];
            PtrType row_end = this->mat_.row_offset[ai + 1];

            for(PtrType aj = row_beg; aj < row_end; ++aj)
            {
                col_key += col_sign * col_tmp * (col_mask | this->mat_.col[aj]);
                col_key  = col_key ^ (col_key >> 16);
                col_sign = sgn(row_tmp - (col_mask | this->mat_.col[aj]));
                col_tmp  = col_mask | this->mat_.col[aj];

                double   double_val = std::abs(this->mat_.val[aj]);
                long int val        = 0;

                assert(sizeof(long int) == sizeof(double));

                memcpy(&val, &double_val, sizeof(long int));

                val_key += val_sign * val_tmp * (long int)(val_mask | val);
                val_key = val_key ^ (val_key >> 16);

                if(sgn(this->mat_.val[aj]) > 0)
                {
                    val_key = val_key ^ val;
                }
                else
                {
                    val_key = val_key | val;
                }

                val_sign = sgn(val_tmp - (long int)(val_mask | val));
                val_tmp  = val_mask | val;
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ReplaceColumnVector(int idx, const BaseVector<ValueType>& vec)
    {
        assert(vec.GetSize() == this->nrow_);

        if(this->nnz_ > 0)
        {
            const HostVector<ValueType>* cast_vec
                = dynamic_cast<const HostVector<ValueType>*>(&vec);
            assert(cast_vec != NULL);

            PtrType*   row_offset = NULL;
            int*       col        = NULL;
            ValueType* val        = NULL;

            int nrow = this->nrow_;
            int ncol = this->ncol_;

            allocate_host(nrow + 1, &row_offset);
            row_offset[0] = 0;

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 0; i < nrow; ++i)
            {
                bool add = true;

                row_offset[i + 1] = this->mat_.row_offset[i + 1] - this->mat_.row_offset[i];

                for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
                {
                    if(this->mat_.col[j] == idx)
                    {
                        add = false;
                        break;
                    }
                }

                if(add == true && cast_vec->vec_[i] != static_cast<ValueType>(0))
                {
                    ++row_offset[i + 1];
                }

                if(add == false && cast_vec->vec_[i] == static_cast<ValueType>(0))
                {
                    --row_offset[i + 1];
                }
            }

            for(int i = 0; i < nrow; ++i)
            {
                row_offset[i + 1] += row_offset[i];
            }

            PtrType nnz = row_offset[nrow];

            allocate_host(nnz, &col);
            allocate_host(nnz, &val);

// Fill new CSR matrix
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 0; i < nrow; ++i)
            {
                PtrType k = row_offset[i];
                PtrType j = this->mat_.row_offset[i];

                for(; j < this->mat_.row_offset[i + 1]; ++j)
                {
                    if(this->mat_.col[j] < idx)
                    {
                        col[k] = this->mat_.col[j];
                        val[k] = this->mat_.val[j];
                        ++k;
                    }
                    else
                    {
                        break;
                    }
                }

                if(cast_vec->vec_[i] != static_cast<ValueType>(0))
                {
                    col[k] = idx;
                    val[k] = cast_vec->vec_[i];
                    ++k;
                    ++j;
                }

                for(; j < this->mat_.row_offset[i + 1]; ++j)
                {
                    if(this->mat_.col[j] > idx)
                    {
                        col[k] = this->mat_.col[j];
                        val[k] = this->mat_.val[j];
                        ++k;
                    }
                }
            }

            this->Clear();
            this->SetDataPtrCSR(&row_offset, &col, &val, row_offset[nrow], nrow, ncol);
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ExtractColumnVector(int idx, BaseVector<ValueType>* vec) const
    {
        assert(vec != NULL);
        assert(vec->GetSize() == this->nrow_);

        if(this->nnz_ > 0)
        {
            HostVector<ValueType>* cast_vec = dynamic_cast<HostVector<ValueType>*>(vec);
            assert(cast_vec != NULL);

            _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int ai = 0; ai < this->nrow_; ++ai)
            {
                // Initialize with zero
                cast_vec->vec_[ai] = static_cast<ValueType>(0);

                for(PtrType aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1];
                    ++aj)
                {
                    if(idx == this->mat_.col[aj])
                    {
                        cast_vec->vec_[ai] = this->mat_.val[aj];
                        break;
                    }
                }
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ReplaceRowVector(int idx, const BaseVector<ValueType>& vec)
    {
        assert(vec.GetSize() == this->ncol_);

        if(this->nnz_ > 0)
        {
            const HostVector<ValueType>* cast_vec
                = dynamic_cast<const HostVector<ValueType>*>(&vec);
            assert(cast_vec != NULL);

            PtrType*   row_offset = NULL;
            int*       col        = NULL;
            ValueType* val        = NULL;

            int nrow = this->nrow_;
            int ncol = this->ncol_;

            allocate_host(nrow + 1, &row_offset);
            row_offset[0] = 0;

            // Compute nnz of row idx
            int nnz_idx = 0;

            for(int i = 0; i < ncol; ++i)
            {
                if(cast_vec->vec_[i] != static_cast<ValueType>(0))
                {
                    ++nnz_idx;
                }
            }

            // Fill row_offset
            PtrType shift = nnz_idx - this->mat_.row_offset[idx + 1] + this->mat_.row_offset[idx];

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 0; i < nrow + 1; ++i)
            {
                if(i < idx + 1)
                {
                    row_offset[i] = this->mat_.row_offset[i];
                }
                else
                {
                    row_offset[i] = this->mat_.row_offset[i] + shift;
                }
            }

            PtrType nnz = row_offset[nrow];

            // Fill col and val
            allocate_host(nnz, &col);
            allocate_host(nnz, &val);

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 0; i < nrow; ++i)
            {
                // Rows before idx
                if(i < idx)
                {
                    for(PtrType j = row_offset[i]; j < row_offset[i + 1]; ++j)
                    {
                        col[j] = this->mat_.col[j];
                        val[j] = this->mat_.val[j];
                    }

                    // Row == idx
                }
                else if(i == idx)
                {
                    PtrType k = row_offset[i];

                    for(int j = 0; j < ncol; ++j)
                    {
                        if(cast_vec->vec_[j] != static_cast<ValueType>(0))
                        {
                            col[k] = j;
                            val[k] = cast_vec->vec_[j];
                            ++k;
                        }
                    }

                    // Rows after idx
                }
                else if(i > idx)
                {
                    PtrType k = row_offset[i];

                    for(PtrType j = this->mat_.row_offset[i]; j < this->mat_.row_offset[i + 1]; ++j)
                    {
                        col[k] = this->mat_.col[j];
                        val[k] = this->mat_.val[j];
                        ++k;
                    }
                }
            }

            this->Clear();
            this->SetDataPtrCSR(&row_offset, &col, &val, nnz, nrow, ncol);
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ExtractRowVector(int idx, BaseVector<ValueType>* vec) const
    {
        assert(vec != NULL);
        assert(vec->GetSize() == this->ncol_);

        if(this->nnz_ > 0)
        {
            HostVector<ValueType>* cast_vec = dynamic_cast<HostVector<ValueType>*>(vec);
            assert(cast_vec != NULL);

            _set_omp_backend_threads(this->local_backend_, this->nrow_);

            cast_vec->Zeros();

            for(PtrType aj = this->mat_.row_offset[idx]; aj < this->mat_.row_offset[idx + 1]; ++aj)
            {
                cast_vec->vec_[this->mat_.col[aj]] = this->mat_.val[aj];
            }
        }

        return true;
    }

    // LCOV_EXCL_START
    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ExtractGlobalColumnIndices(int     ncol,
                                                              int64_t global_offset,
                                                              const BaseVector<int64_t>& l2g,
                                                              BaseVector<int64_t>* global_col) const
    {
        if(this->nnz_ > 0)
        {
            const HostVector<int64_t>* cast_l2g = dynamic_cast<const HostVector<int64_t>*>(&l2g);
            HostVector<int64_t>*       cast_col = dynamic_cast<HostVector<int64_t>*>(global_col);

            assert(cast_col != NULL);
            assert(this->nnz_ == cast_col->size_);

            for(int64_t i = 0; i < this->nnz_; ++i)
            {
                int local_col = this->mat_.col[i];

                if(local_col >= ncol)
                {
                    // This is an ext column, map to global
                    cast_col->vec_[i] = cast_l2g->vec_[local_col - ncol];
                }
                else
                {
                    // This is a local column, shift by offset
                    cast_col->vec_[i] = local_col + global_offset;
                }
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ExtractExtRowNnz(int offset, BaseVector<PtrType>* row_nnz) const
    {
        assert(row_nnz != NULL);

        if(this->GetNnz() > 0)
        {
            HostVector<PtrType>* cast_vec = dynamic_cast<HostVector<PtrType>*>(row_nnz);

            assert(cast_vec != NULL);

            int size = this->nrow_ - offset;

            for(int i = 0; i < size; ++i)
            {
                cast_vec->vec_[i]
                    = this->mat_.row_offset[i + 1 + offset] - this->mat_.row_offset[i + offset];
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ExtractBoundaryRowNnz(BaseVector<PtrType>*   row_nnz,
                                                         const BaseVector<int>& boundary_index,
                                                         const BaseMatrix<ValueType>& gst) const
    {
        assert(row_nnz != NULL);

        HostVector<PtrType>*   cast_vec = dynamic_cast<HostVector<PtrType>*>(row_nnz);
        const HostVector<int>* cast_idx = dynamic_cast<const HostVector<int>*>(&boundary_index);
        const HostMatrixCSR<ValueType>* cast_gst
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&gst);

        assert(cast_vec != NULL);
        assert(cast_idx != NULL);
        assert(cast_gst != NULL);

        for(int64_t i = 0; i < cast_idx->size_; ++i)
        {
            // This is a boundary row
            int row = cast_idx->vec_[i];

            cast_vec->vec_[i] = this->mat_.row_offset[row + 1] - this->mat_.row_offset[row]
                                + cast_gst->mat_.row_offset[row + 1]
                                - cast_gst->mat_.row_offset[row];
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::ExtractBoundaryRows(const BaseVector<PtrType>& bnd_csr_row_ptr,
                                                       BaseVector<int64_t>*       bnd_csr_col_ind,
                                                       BaseVector<ValueType>*     bnd_csr_val,
                                                       int64_t                global_column_offset,
                                                       const BaseVector<int>& boundary_index,
                                                       const BaseVector<int64_t>&   ghost_mapping,
                                                       const BaseMatrix<ValueType>& gst) const
    {
        assert(bnd_csr_col_ind != NULL);
        assert(bnd_csr_val != NULL);

        const HostVector<PtrType>* cast_ptr
            = dynamic_cast<const HostVector<PtrType>*>(&bnd_csr_row_ptr);
        HostVector<int64_t>*       cast_col = dynamic_cast<HostVector<int64_t>*>(bnd_csr_col_ind);
        HostVector<ValueType>*     cast_val = dynamic_cast<HostVector<ValueType>*>(bnd_csr_val);
        const HostVector<int>*     cast_bnd = dynamic_cast<const HostVector<int>*>(&boundary_index);
        const HostVector<int64_t>* cast_l2g
            = dynamic_cast<const HostVector<int64_t>*>(&ghost_mapping);
        const HostMatrixCSR<ValueType>* cast_gst
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&gst);

        assert(cast_ptr != NULL);
        assert(cast_col != NULL);
        assert(cast_val != NULL);
        assert(cast_bnd != NULL);
        assert(cast_l2g != NULL);
        assert(cast_gst != NULL);

        for(int64_t i = 0; i < cast_bnd->size_; ++i)
        {
            // This is a boundary row
            int row = cast_bnd->vec_[i];

            // Index into boundary structures
            PtrType idx = cast_ptr->vec_[i];

            // Entry points
            PtrType row_begin = this->mat_.row_offset[row];
            PtrType row_end   = this->mat_.row_offset[row + 1];

            // Interior
            for(PtrType j = row_begin; j < row_end; ++j)
            {
                // Shift column by global column offset,
                // to obtain the global column index
                cast_col->vec_[idx] = this->mat_.col[j] + global_column_offset;
                cast_val->vec_[idx] = this->mat_.val[j];

                ++idx;
            }

            // Ghost
            row_begin = cast_gst->mat_.row_offset[row];
            row_end   = cast_gst->mat_.row_offset[row + 1];

            for(PtrType j = row_begin; j < row_end; ++j)
            {
                // Map the local ghost column to global column
                cast_col->vec_[idx] = cast_l2g->vec_[cast_gst->mat_.col[j]];
                cast_val->vec_[idx] = cast_gst->mat_.val[j];

                ++idx;
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::MergeToLocal(const BaseMatrix<ValueType>& mat_int,
                                                const BaseMatrix<ValueType>& mat_gst,
                                                const BaseMatrix<ValueType>& mat_ext,
                                                const BaseVector<int>&       vec_ext)
    {
        assert(this != &mat_int);
        assert(this != &mat_gst);
        assert(this != &mat_ext);
        assert(&mat_int != &mat_gst);
        assert(&mat_int != &mat_ext);
        assert(&mat_gst != &mat_ext);

        const HostMatrixCSR<ValueType>* cast_int
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&mat_int);
        const HostMatrixCSR<ValueType>* cast_gst
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&mat_gst);
        const HostMatrixCSR<ValueType>* cast_ext
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&mat_ext);
        const HostVector<int>* cast_vec = dynamic_cast<const HostVector<int>*>(&vec_ext);

        assert(cast_int != NULL);
        assert(cast_ext != NULL);
        assert(cast_vec != NULL);

        // Determine nnz per row
        for(int i = 0; i < this->nrow_; ++i)
        {
            if(i < cast_int->nrow_)
            {
                PtrType nnz = cast_int->mat_.row_offset[i + 1] - cast_int->mat_.row_offset[i];

                // If ghost is available
                if(cast_gst->nnz_ > 0)
                {
                    nnz += cast_gst->mat_.row_offset[i + 1] - cast_gst->mat_.row_offset[i];
                }

                this->mat_.row_offset[i + 1] = nnz;
            }
            else if(i - cast_int->nrow_ < cast_ext->nrow_)
            {
                this->mat_.row_offset[i + 1] = cast_ext->mat_.row_offset[i - cast_int->nrow_ + 1]
                                               - cast_ext->mat_.row_offset[i - cast_int->nrow_];
            }
        }

        // Exclusive sum to obtain pointers
        this->mat_.row_offset[0] = 0;
        for(int i = 0; i < this->nrow_; ++i)
        {
            this->mat_.row_offset[i + 1] += this->mat_.row_offset[i];
        }

        // Fill
        for(int i = 0; i < this->nrow_; ++i)
        {
            // Index into "this"
            PtrType idx = this->mat_.row_offset[i];

            if(i < cast_int->nrow_)
            {
                // Interior
                for(PtrType j = cast_int->mat_.row_offset[i]; j < cast_int->mat_.row_offset[i + 1];
                    ++j)
                {
                    this->mat_.col[idx] = cast_int->mat_.col[j];
                    this->mat_.val[idx] = cast_int->mat_.val[j];

                    ++idx;
                }

                // Ghost
                if(cast_gst->nnz_ > 0)
                {
                    for(PtrType j = cast_gst->mat_.row_offset[i];
                        j < cast_gst->mat_.row_offset[i + 1];
                        ++j)
                    {
                        this->mat_.col[idx]
                            = (cast_vec->size_ > 0 ? cast_vec->vec_[j] : cast_gst->mat_.col[j])
                              + cast_int->ncol_;
                        this->mat_.val[idx] = cast_gst->mat_.val[j];

                        ++idx;
                    }
                }
            }
            else if(i - cast_int->nrow_ < cast_ext->nrow_)
            {
                for(PtrType j = cast_ext->mat_.row_offset[i - cast_int->nrow_];
                    j < cast_ext->mat_.row_offset[i - cast_int->nrow_ + 1];
                    ++j)
                {
                    this->mat_.col[idx] = cast_ext->mat_.col[j];
                    this->mat_.val[idx] = cast_ext->mat_.val[j];

                    ++idx;
                }
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::CombineAndRenumber(int                        ncol,
                                                      int64_t                    ext_nnz,
                                                      int64_t                    global_col_begin,
                                                      int64_t                    global_col_end,
                                                      const BaseVector<int64_t>& l2g,
                                                      const BaseVector<int64_t>& ext,
                                                      BaseVector<int>*           merged,
                                                      BaseVector<int64_t>*       mapping,
                                                      BaseVector<int>*           local_col) const
    {
        assert(merged != NULL);
        assert(mapping != NULL);
        assert(local_col != NULL);

        const HostVector<int64_t>* cast_l2g = dynamic_cast<const HostVector<int64_t>*>(&l2g);
        const HostVector<int64_t>* cast_ext = dynamic_cast<const HostVector<int64_t>*>(&ext);
        HostVector<int>*           cast_cmb = dynamic_cast<HostVector<int>*>(merged);
        HostVector<int64_t>*       cast_map = dynamic_cast<HostVector<int64_t>*>(mapping);
        HostVector<int>*           cast_col = dynamic_cast<HostVector<int>*>(local_col);

        assert(cast_l2g != NULL);
        assert(cast_ext != NULL);
        assert(cast_cmb != NULL);
        assert(cast_map != NULL);
        assert(cast_col != NULL);

        // Ghost and ext should be 32 bit
        assert(this->nnz_ + ext_nnz < std::numeric_limits<int>::max());

        // Map columns from local to global
        int64_t* gcols = NULL;
        allocate_host(this->nnz_ + ext_nnz, &gcols);

        // Transfer local column indices to global ones
        for(int64_t i = 0; i < this->nnz_; ++i)
        {
            gcols[i] = cast_l2g->vec_[this->mat_.col[i]];
        }

        // Add additional global columns from ext
        int total_nnz = static_cast<int>(this->nnz_);

        for(int64_t i = 0; i < ext_nnz; ++i)
        {
            int64_t global_col = cast_ext->vec_[i];

            // Filter external source to match given column id interval
            if(global_col < global_col_begin || global_col >= global_col_end)
            {
                gcols[total_nnz] = global_col;
                ++total_nnz;
            }
        }

        // Sort global column array to eliminate duplicates later on
        int64_t* gcol_sorted = NULL;
        allocate_host(total_nnz + 1, &gcol_sorted);

        int* perm = NULL;
        allocate_host(total_nnz, &perm);

        // Create identity permutation
        std::iota(perm, perm + total_nnz, 0);

        // Sort with permutation
        std::sort(perm, perm + total_nnz, [&](const int64_t& a, const int64_t& b) {
            return (gcols[a] < gcols[b]);
        });

        for(int i = 0; i < total_nnz; ++i)
        {
            gcol_sorted[i] = gcols[perm[i]];
        }

        free_host(&gcols);

        // Run length encode to obtain counts for duplicated entries (we need this for renumbering)
        cast_map->Clear();
        cast_cmb->Clear();

        cast_map->Allocate(total_nnz);
        cast_cmb->Allocate(total_nnz);

        int nruns = 0;

        for(int i = 0; i < total_nnz; ++i)
        {
            // Count
            int cnt = 1;

            while(i < total_nnz - 1 && gcol_sorted[i] == gcol_sorted[i + 1])
            {
                ++cnt;
                ++i;
            }

            cast_map->vec_[nruns] = gcol_sorted[i];
            cast_cmb->vec_[nruns] = cnt;

            // Runs
            ++nruns;
        }

        // Update col_mapped size
        cast_map->size_ = nruns;

        // Prefix sum to get entry points to duplicates
        gcol_sorted[0] = 0;

        for(int i = 0; i < nruns; ++i)
        {
            gcol_sorted[i + 1] = gcol_sorted[i] + cast_cmb->vec_[i];
        }

        // Renumbered column ids
        for(int i = 0; i < nruns; ++i)
        {
            for(int64_t j = gcol_sorted[i]; j < gcol_sorted[i + 1]; ++j)
            {
                cast_cmb->vec_[perm[j]] = i;
            }
        }

        free_host(&perm);
        free_host(&gcol_sorted);

        // Column id transformation
        for(int64_t i = 0, k = 0; i < ext_nnz; ++i)
        {
            int64_t global_col = cast_ext->vec_[i];

            // Filter external source to match given column id interval
            if(global_col >= global_col_begin && global_col < global_col_end)
            {
                // Map column into interval
                cast_col->vec_[i] = static_cast<int>(global_col - global_col_begin);
            }
            else
            {
                cast_col->vec_[i] = cast_cmb->vec_[k + this->nnz_] + ncol;
                ++k;
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::SplitInteriorGhost(BaseMatrix<ValueType>* interior,
                                                      BaseMatrix<ValueType>* ghost) const
    {
        assert(interior != NULL);
        assert(ghost != NULL);
        assert(interior != ghost);

        HostMatrixCSR<ValueType>* cast_int = dynamic_cast<HostMatrixCSR<ValueType>*>(interior);
        HostMatrixCSR<ValueType>* cast_gst = dynamic_cast<HostMatrixCSR<ValueType>*>(ghost);

        assert(cast_int != NULL);
        assert(cast_gst != NULL);

        cast_int->Clear();
        cast_gst->Clear();

        // First, we need to determine the number of non-zeros for each matrix
        PtrType* int_csr_row_ptr = NULL;
        PtrType* gst_csr_row_ptr = NULL;

        allocate_host(this->nrow_ + 1, &int_csr_row_ptr);
        allocate_host(this->nrow_ + 1, &gst_csr_row_ptr);

        set_to_zero_host(this->nrow_ + 1, int_csr_row_ptr);
        set_to_zero_host(this->nrow_ + 1, gst_csr_row_ptr);

        for(int i = 0; i < this->nrow_; ++i)
        {
            int int_nnz = 0;
            int gst_nnz = 0;

            PtrType row_begin = this->mat_.row_offset[i];
            PtrType row_end   = this->mat_.row_offset[i + 1];

            for(PtrType j = row_begin; j < row_end; ++j)
            {
                int col = this->mat_.col[j];

                if(col >= this->nrow_)
                {
                    // This is a ghost column
                    ++gst_nnz;
                }
                else
                {
                    // This is an interior column
                    ++int_nnz;
                }
            }

            int_csr_row_ptr[i + 1] = int_nnz;
            gst_csr_row_ptr[i + 1] = gst_nnz;
        }

        // Exclusive sum to obtain offsets
        for(int i = 0; i < this->nrow_; ++i)
        {
            int_csr_row_ptr[i + 1] += int_csr_row_ptr[i];
            gst_csr_row_ptr[i + 1] += gst_csr_row_ptr[i];
        }

        PtrType int_nnz = int_csr_row_ptr[this->nrow_];
        PtrType gst_nnz = gst_csr_row_ptr[this->nrow_];

        // Allocate
        int*       int_csr_col_ind = NULL;
        int*       gst_csr_col_ind = NULL;
        ValueType* int_csr_val     = NULL;
        ValueType* gst_csr_val     = NULL;

        allocate_host(int_nnz, &int_csr_col_ind);
        allocate_host(int_nnz, &int_csr_val);
        allocate_host(gst_nnz, &gst_csr_col_ind);
        allocate_host(gst_nnz, &gst_csr_val);

        // Fill matrices
        for(int i = 0; i < this->nrow_; ++i)
        {
            PtrType row_begin = this->mat_.row_offset[i];
            PtrType row_end   = this->mat_.row_offset[i + 1];

            PtrType idx_int = int_csr_row_ptr[i];
            PtrType idx_gst = gst_csr_row_ptr[i];

            for(PtrType j = row_begin; j < row_end; ++j)
            {
                int col = this->mat_.col[j];

                if(col >= this->nrow_)
                {
                    // This is a ghost column
                    int gst_col = col - this->nrow_;

                    gst_csr_col_ind[idx_gst] = gst_col;
                    gst_csr_val[idx_gst]     = this->mat_.val[j];

                    ++idx_gst;
                }
                else
                {
                    // This is an interior column
                    int_csr_col_ind[idx_int] = col;
                    int_csr_val[idx_int]     = this->mat_.val[j];

                    ++idx_int;
                }
            }
        }

        cast_int->SetDataPtrCSR(
            &int_csr_row_ptr, &int_csr_col_ind, &int_csr_val, int_nnz, this->nrow_, this->nrow_);

        cast_gst->SetDataPtrCSR(
            &gst_csr_row_ptr, &gst_csr_col_ind, &gst_csr_val, gst_nnz, this->nrow_, this->nrow_);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::CopyGhostFromGlobalReceive(
        const BaseVector<int>&       boundary,
        const BaseVector<PtrType>&   recv_csr_row_ptr,
        const BaseVector<int64_t>&   recv_csr_col_ind,
        const BaseVector<ValueType>& recv_csr_val,
        BaseVector<int64_t>*         global_col)
    {
        const HostVector<int>*     cast_bnd = dynamic_cast<const HostVector<int>*>(&boundary);
        const HostVector<PtrType>* cast_ptr
            = dynamic_cast<const HostVector<PtrType>*>(&recv_csr_row_ptr);
        const HostVector<int64_t>* cast_col
            = dynamic_cast<const HostVector<int64_t>*>(&recv_csr_col_ind);
        const HostVector<ValueType>* cast_val
            = dynamic_cast<const HostVector<ValueType>*>(&recv_csr_val);
        HostVector<int64_t>* cast_glo = dynamic_cast<HostVector<int64_t>*>(global_col);

        assert(cast_bnd != NULL);
        assert(cast_ptr != NULL);
        assert(cast_col != NULL);
        assert(cast_val != NULL);

        // First, we need to determine the number of non-zeros
        for(int i = 0; i < cast_bnd->size_; ++i)
        {
            int     row = cast_bnd->vec_[i];
            PtrType nnz = cast_ptr->vec_[i + 1] - cast_ptr->vec_[i];

            this->mat_.row_offset[row + 1] += nnz;
        }

        // Exclusive sum to obtain offsets
        this->mat_.row_offset[0] = 0;
        for(int i = 0; i < this->nrow_; ++i)
        {
            this->mat_.row_offset[i + 1] += this->mat_.row_offset[i];
        }

        assert(this->mat_.row_offset[this->nrow_] == this->nnz_);

        cast_glo->Allocate(this->nnz_);

        // Fill matrices
        for(int i = 0; i < cast_bnd->size_; ++i)
        {
            int row = cast_bnd->vec_[i];

            PtrType row_begin = cast_ptr->vec_[i];
            PtrType row_end   = cast_ptr->vec_[i + 1];

            PtrType idx = this->mat_.row_offset[row];

            for(PtrType j = row_begin; j < row_end; ++j)
            {
                cast_glo->vec_[idx] = cast_col->vec_[j];
                this->mat_.val[idx] = cast_val->vec_[j];

                ++idx;
            }

            this->mat_.row_offset[row] = idx;
        }

        // Shift back
        for(int i = this->nrow_; i > 0; --i)
        {
            this->mat_.row_offset[i] = this->mat_.row_offset[i - 1];
        }

        this->mat_.row_offset[0] = 0;

        return true;
    }

    template <typename ValueType>
    bool
        HostMatrixCSR<ValueType>::CopyFromGlobalReceive(int                    nrow,
                                                        int64_t                global_column_begin,
                                                        int64_t                global_column_end,
                                                        const BaseVector<int>& boundary,
                                                        const BaseVector<PtrType>& recv_csr_row_ptr,
                                                        const BaseVector<int64_t>& recv_csr_col_ind,
                                                        const BaseVector<ValueType>& recv_csr_val,
                                                        BaseMatrix<ValueType>*       ghost,
                                                        BaseVector<int64_t>*         global_col)
    {
        assert(ghost != NULL);
        assert(global_col != NULL);

        const HostVector<int>*     cast_bnd = dynamic_cast<const HostVector<int>*>(&boundary);
        const HostVector<PtrType>* cast_ptr
            = dynamic_cast<const HostVector<PtrType>*>(&recv_csr_row_ptr);
        const HostVector<int64_t>* cast_col
            = dynamic_cast<const HostVector<int64_t>*>(&recv_csr_col_ind);
        const HostVector<ValueType>* cast_val
            = dynamic_cast<const HostVector<ValueType>*>(&recv_csr_val);
        HostMatrixCSR<ValueType>* cast_gst = dynamic_cast<HostMatrixCSR<ValueType>*>(ghost);
        HostVector<int64_t>*      cast_glo = dynamic_cast<HostVector<int64_t>*>(global_col);

        assert(cast_bnd != NULL);
        assert(cast_ptr != NULL);
        assert(cast_col != NULL);
        assert(cast_val != NULL);
        assert(cast_gst != NULL);

        PtrType* int_csr_row_ptr = NULL;
        PtrType* gst_csr_row_ptr = NULL;

        allocate_host(nrow + 1, &int_csr_row_ptr);
        allocate_host(nrow + 1, &gst_csr_row_ptr);

        set_to_zero_host(nrow + 1, int_csr_row_ptr);
        set_to_zero_host(nrow + 1, gst_csr_row_ptr);

        // First, we need to determine the number of non-zeros
        for(int i = 0; i < cast_bnd->size_; ++i)
        {
            int row = cast_bnd->vec_[i];

            int int_nnz = 0;
            int gst_nnz = 0;

            PtrType row_begin = cast_ptr->vec_[i];
            PtrType row_end   = cast_ptr->vec_[i + 1];

            for(PtrType j = row_begin; j < row_end; ++j)
            {
                int64_t col = cast_col->vec_[j];

                if(col >= global_column_begin && col < global_column_end)
                {
                    // This is an interior column
                    ++int_nnz;
                }
                else
                {
                    // This is a ghost column
                    ++gst_nnz;
                }
            }

            int_csr_row_ptr[row + 1] += int_nnz;
            gst_csr_row_ptr[row + 1] += gst_nnz;
        }

        // Exclusive sum to obtain offsets
        int_csr_row_ptr[0] = 0;
        gst_csr_row_ptr[0] = 0;

        for(int i = 0; i < nrow; ++i)
        {
            int_csr_row_ptr[i + 1] += int_csr_row_ptr[i];
            gst_csr_row_ptr[i + 1] += gst_csr_row_ptr[i];
        }

        PtrType int_nnz = int_csr_row_ptr[nrow];
        PtrType gst_nnz = gst_csr_row_ptr[nrow];

        cast_glo->Allocate(gst_nnz);

        int*       int_csr_col_ind = NULL;
        int*       gst_csr_col_ind = NULL;
        ValueType* int_csr_val     = NULL;
        ValueType* gst_csr_val     = NULL;

        allocate_host(int_nnz, &int_csr_col_ind);
        allocate_host(gst_nnz, &gst_csr_col_ind);
        allocate_host(int_nnz, &int_csr_val);
        allocate_host(gst_nnz, &gst_csr_val);

        // Fill matrices
        for(int i = 0; i < cast_bnd->size_; ++i)
        {
            int row = cast_bnd->vec_[i];

            PtrType row_begin = cast_ptr->vec_[i];
            PtrType row_end   = cast_ptr->vec_[i + 1];

            PtrType idx_int = int_csr_row_ptr[row];
            PtrType idx_gst = gst_csr_row_ptr[row];

            for(PtrType j = row_begin; j < row_end; ++j)
            {
                int64_t col = cast_col->vec_[j];

                if(col >= global_column_begin && col < global_column_end)
                {
                    // This is an interior column
                    int_csr_col_ind[idx_int] = static_cast<int>(col - global_column_begin);
                    int_csr_val[idx_int]     = cast_val->vec_[j];

                    ++idx_int;
                }
                else
                {
                    // This is a ghost column
                    cast_glo->vec_[idx_gst] = cast_col->vec_[j];
                    gst_csr_val[idx_gst]    = cast_val->vec_[j];

                    ++idx_gst;
                }
            }

            int_csr_row_ptr[row] = idx_int;
            gst_csr_row_ptr[row] = idx_gst;
        }

        // Shift back
        for(int i = nrow; i > 0; --i)
        {
            int_csr_row_ptr[i] = int_csr_row_ptr[i - 1];
            gst_csr_row_ptr[i] = gst_csr_row_ptr[i - 1];
        }

        int_csr_row_ptr[0] = 0;
        gst_csr_row_ptr[0] = 0;

        this->SetDataPtrCSR(&int_csr_row_ptr, &int_csr_col_ind, &int_csr_val, int_nnz, nrow, nrow);
        cast_gst->SetDataPtrCSR(
            &gst_csr_row_ptr, &gst_csr_col_ind, &gst_csr_val, gst_nnz, nrow, nrow);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::RenumberGlobalToLocal(const BaseVector<int64_t>& column_indices)
    {
        if(this->nnz_ > 0)
        {
            const HostVector<int64_t>* cast_col
                = dynamic_cast<const HostVector<int64_t>*>(&column_indices);

            assert(cast_col != NULL);

            // Allocate temporary arrays
            HostVector<int>     perm(this->local_backend_);
            HostVector<int64_t> sorted(this->local_backend_);
            HostVector<int>     workspace(this->local_backend_);

            perm.Allocate(this->nnz_);
            sorted.Allocate(this->nnz_);
            workspace.Allocate(this->nnz_);

            cast_col->Sort(&sorted, &perm);

            // Renumber columns consecutively, starting with 0

            // Loop over all non-zero entries
            for(int64_t i = 0; i < this->nnz_; ++i)
            {
                // First column entry cannot be a duplicate
                if(i == 0)
                {
                    workspace.vec_[0] = 1;

                    continue;
                }

                // Next column in line
                int64_t global_col = sorted.vec_[i];

                // Compare column indices
                if(global_col == sorted.vec_[i - 1])
                {
                    // Same index as previous column
                    workspace.vec_[i] = 0;
                }
                else
                {
                    // New column index
                    workspace.vec_[i] = 1;
                }
            }

            // Inclusive sum
            this->ncol_ = workspace.InclusiveSum(workspace);

            // Write new column indices into matrix
            for(int64_t i = 0; i < this->nnz_; ++i)
            {
                // Decrement by 1
                this->mat_.col[perm.vec_[i]] = workspace.vec_[i] - 1;
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCSR<ValueType>::CompressAdd(const BaseVector<int64_t>&   l2g,
                                               const BaseVector<int64_t>&   global_ghost_col,
                                               const BaseMatrix<ValueType>& ext,
                                               BaseVector<int64_t>*         global_col)
    {
        const HostVector<int64_t>* cast_l2g = dynamic_cast<const HostVector<int64_t>*>(&l2g);
        const HostVector<int64_t>* cast_col
            = dynamic_cast<const HostVector<int64_t>*>(&global_ghost_col);
        const HostMatrixCSR<ValueType>* cast_ext
            = dynamic_cast<const HostMatrixCSR<ValueType>*>(&ext);
        HostVector<int64_t>* cast_glo = dynamic_cast<HostVector<int64_t>*>(global_col);

        assert(cast_ext != NULL);
        assert(cast_ext->nrow_ == this->nrow_);

        // Sizes
        int nrow = this->nrow_;
        int ncol = this->ncol_;

        // Do we operate on interior? Then, we do not need to convert local to global indices
        // Additionally, we do not need to store global columns in an extra 64 bit vector
        bool interior = global_col == NULL;

        if(interior == false)
        {
            assert(cast_l2g != NULL);
            assert(cast_glo != NULL);

            cast_glo->Clear();
        }

        if(this->nnz_ > 0 || cast_ext->nnz_ > 0)
        {
            // Count non-zeros of merged matrix
            PtrType* csr_row_ptr = NULL;
            allocate_host(nrow + 1, &csr_row_ptr);

            for(int i = 0; i < nrow; ++i)
            {
                PtrType row_begin = this->mat_.row_offset[i];
                PtrType row_end   = this->mat_.row_offset[i + 1];

                // First, we need to count the unique column entries
                std::unordered_set<int64_t> columns;

                for(PtrType j = row_begin; j < row_end; ++j)
                {
                    // Get column index
                    auto col = (interior == true) ? this->mat_.col[j]
                                                  : cast_l2g->vec_[this->mat_.col[j]];

                    // Add column
                    columns.insert(col);
                }

                row_begin = cast_ext->mat_.row_offset[i];
                row_end   = cast_ext->mat_.row_offset[i + 1];

                for(PtrType j = row_begin; j < row_end; ++j)
                {
                    // Get column index
                    auto col = (interior == true) ? cast_ext->mat_.col[j] : cast_col->vec_[j];

                    // Add column
                    columns.insert(col);
                }

                csr_row_ptr[i + 1] = columns.size();
            }

            // Exclusive sum to obtain new row pointers
            csr_row_ptr[0] = 0;

            for(int i = 0; i < cast_ext->nrow_; ++i)
            {
                csr_row_ptr[i + 1] += csr_row_ptr[i];
            }

            // Obtain nnz
            PtrType nnz = csr_row_ptr[nrow];

            // Allocate
            int*       csr_col_ind = NULL;
            ValueType* csr_val     = NULL;

            allocate_host(nnz, &csr_col_ind);
            allocate_host(nnz, &csr_val);

            // If we are operating on ghost, we need a special vector to store column
            // indices in 64 bits
            if(interior == false)
            {
                cast_glo->Allocate(nnz);
            }

            // Fill
            for(int i = 0; i < nrow; ++i)
            {
                PtrType k = csr_row_ptr[i];

                PtrType row_begin = this->mat_.row_offset[i];
                PtrType row_end   = this->mat_.row_offset[i + 1];

                std::map<int64_t, ValueType> sorted_row;

                for(PtrType j = row_begin; j < row_end; ++j)
                {
                    // Get column index
                    auto col = (interior == true) ? this->mat_.col[j]
                                                  : cast_l2g->vec_[this->mat_.col[j]];

                    // Fill / merge
                    sorted_row[col] += this->mat_.val[j];
                }

                row_begin = cast_ext->mat_.row_offset[i];
                row_end   = cast_ext->mat_.row_offset[i + 1];

                for(PtrType j = row_begin; j < row_end; ++j)
                {
                    // Get column index
                    auto col = (interior == true) ? cast_ext->mat_.col[j] : cast_col->vec_[j];

                    // Fill / merge
                    sorted_row[col] += cast_ext->mat_.val[j];
                }

                for(auto it = sorted_row.begin(); it != sorted_row.end(); ++it)
                {
                    if(interior == true)
                    {
                        csr_col_ind[k] = it->first;
                    }
                    else
                    {
                        // Store in extra vector in ghost case
                        cast_glo->vec_[k] = it->first;
                    }

                    csr_val[k] = it->second;
                    ++k;
                }
            }

            // Update matrix
            this->Clear();
            this->SetDataPtrCSR(&csr_row_ptr, &csr_col_ind, &csr_val, nnz, nrow, ncol);
        }

        return true;
    }

    // LCOV_EXCL_STOP

    template class HostMatrixCSR<double>;
    template class HostMatrixCSR<float>;
#ifdef SUPPORT_COMPLEX
    template class HostMatrixCSR<std::complex<double>>;
    template class HostMatrixCSR<std::complex<float>>;
#endif

} // namespace rocalution

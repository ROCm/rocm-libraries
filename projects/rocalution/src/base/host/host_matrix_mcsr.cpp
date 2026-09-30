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

#include "host_matrix_mcsr.hpp"
#include "../../utils/allocate_free.hpp"
#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "../matrix_formats_ind.hpp"
#include "host_conversion.hpp"
#include "host_io.hpp"
#include "host_matrix_csr.hpp"
#include "host_vector.hpp"

#include <complex>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_set_num_threads(num) ;
#endif

namespace rocalution
{

    // LCOV_EXCL_START
    template <typename ValueType>
    HostMatrixMCSR<ValueType>::HostMatrixMCSR()
    {
        // no default constructors
        LOG_INFO("no default constructor");
        FATAL_ERROR(__FILE__, __LINE__);
    }
    // LCOV_EXCL_STOP

    template <typename ValueType>
    HostMatrixMCSR<ValueType>::HostMatrixMCSR(const Rocalution_Backend_Descriptor& local_backend)
    {
        log_debug(this, "HostMatrixMCSR::HostMatrixMCSR()", "constructor with local_backend");

        this->mat_.row_offset = NULL;
        this->mat_.val        = NULL;
        this->mat_.col        = NULL;

        this->set_backend(local_backend);
    }

    template <typename ValueType>
    HostMatrixMCSR<ValueType>::~HostMatrixMCSR()
    {
        log_debug(this, "HostMatrixMCSR::~HostMatrixMCSR()", "destructor");

        this->Clear();
    }

    template <typename ValueType>
    void HostMatrixMCSR<ValueType>::Info(void) const
    {
        LOG_INFO("HostMatrixMCSR<ValueType>");
    }

    template <typename ValueType>
    void HostMatrixMCSR<ValueType>::Clear()
    {
        free_host(&this->mat_.row_offset);
        free_host(&this->mat_.col);
        free_host(&this->mat_.val);

        this->nrow_ = 0;
        this->ncol_ = 0;
        this->nnz_  = 0;
    }

    template <typename ValueType>
    void HostMatrixMCSR<ValueType>::AllocateMCSR(int64_t nnz, int nrow, int ncol)
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
    void HostMatrixMCSR<ValueType>::SetDataPtrMCSR(
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

        this->mat_.row_offset = *row_offset;
        this->mat_.col        = *col;
        this->mat_.val        = *val;
    }

    template <typename ValueType>
    void HostMatrixMCSR<ValueType>::LeaveDataPtrMCSR(int** row_offset, int** col, ValueType** val)
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
    void HostMatrixMCSR<ValueType>::CopyFrom(const BaseMatrix<ValueType>& mat)
    {
        // copy only in the same format
        assert(this->GetMatFormat() == mat.GetMatFormat());

        if(const HostMatrixMCSR<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixMCSR<ValueType>*>(&mat))
        {
            if(this->nnz_ == 0)
            {
                this->AllocateMCSR(cast_mat->nnz_, cast_mat->nrow_, cast_mat->ncol_);
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
    void HostMatrixMCSR<ValueType>::CopyTo(BaseMatrix<ValueType>* mat) const
    {
        mat->CopyFrom(*this);
    }

    template <typename ValueType>
    bool HostMatrixMCSR<ValueType>::ConvertFrom(const BaseMatrix<ValueType>& mat)
    {
        this->Clear();

        // Empty matrix
        if(mat.GetNnz() == 0)
        {
            this->AllocateMCSR(mat.GetNnz(), mat.GetM(), mat.GetN());

            return true;
        }

        if(const HostMatrixMCSR<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixMCSR<ValueType>*>(&mat))
        {
            this->CopyFrom(*cast_mat);
            return true;
        }

        if(const HostMatrixCSR<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixCSR<ValueType>*>(&mat))
        {
            this->Clear();

            if(csr_to_mcsr(this->local_backend_.OpenMP_threads,
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

        return false;
    }

    template <typename ValueType>
    bool HostMatrixMCSR<ValueType>::ILU0Factorize(void)
    {
        assert(this->nrow_ == this->ncol_);
        assert(this->nnz_ > 0);

        int* diag_offset = NULL;
        int* nnz_entries = NULL;

        allocate_host(this->nrow_, &diag_offset);
        allocate_host(this->nrow_, &nnz_entries);

        set_to_zero_host(this->nrow_, nnz_entries);

        // ai = 0 to N loop over all rows
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            int aj;
            int row_start = this->mat_.row_offset[ai];
            int row_end   = this->mat_.row_offset[ai + 1];

            for(aj = row_start; aj < row_end; ++aj)
            {
                nnz_entries[this->mat_.col[aj]] = aj;
            }

            // Diagonal position
            nnz_entries[ai] = ai;

            // loop over ai-th row nnz entries
            for(aj = row_start; aj < row_end; ++aj)
            {
                if(this->mat_.col[aj] < ai)
                {
                    int col_j = this->mat_.col[aj];

                    this->mat_.val[aj] /= this->mat_.val[col_j];

                    // loop over upper offset pointer and do linear combination for nnz entry
                    for(int ak = diag_offset[col_j]; ak < this->mat_.row_offset[col_j + 1]; ++ak)
                    {
                        // if nnz at this position do linear combination
                        if(nnz_entries[this->mat_.col[ak]] != 0)
                        {
                            this->mat_.val[nnz_entries[this->mat_.col[ak]]]
                                -= this->mat_.val[aj] * this->mat_.val[ak];
                        }
                    }
                }
                else
                {
                    break;
                }
            }

            // set diagonal pointer to diagonal element
            diag_offset[ai] = aj;

            // clear nnz entries
            for(aj = row_start; aj < row_end; ++aj)
            {
                nnz_entries[this->mat_.col[aj]] = 0;
            }

            nnz_entries[ai] = 0;
        }

        free_host(&diag_offset);
        free_host(&nnz_entries);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixMCSR<ValueType>::LUSolve(const BaseVector<ValueType>& in,
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
            for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
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

        // Solve U
        for(int ai = this->nrow_ - 1; ai >= 0; --ai)
        {
            for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
            {
                if(this->mat_.col[aj] > ai)
                {
                    cast_out->vec_[ai] -= this->mat_.val[aj] * cast_out->vec_[this->mat_.col[aj]];
                }
            }

            cast_out->vec_[ai] /= this->mat_.val[ai];
        }

        return true;
    }

    template <typename ValueType>
    void HostMatrixMCSR<ValueType>::LUAnalyse(void)
    {
        // do nothing
    }

    template <typename ValueType>
    void HostMatrixMCSR<ValueType>::LUAnalyseClear(void)
    {
        // do nothing
    }

    template <typename ValueType>
    bool HostMatrixMCSR<ValueType>::ReadFileRSIO(const std::string& filename)
    {
        int64_t nrow;
        int64_t ncol;
        int64_t nnz;

        PtrType*   ptr = NULL;
        int*       col = NULL;
        ValueType* val = NULL;

        if(read_matrix_mcsr_rocsparseio(nrow, ncol, nnz, &ptr, &col, &val, filename.c_str())
           != true)
        {
            return false;
        }

        // Number of rows and columns are expected to be within 32 bits locally
        assert(nrow <= std::numeric_limits<int>::max());
        assert(ncol <= std::numeric_limits<int>::max());

        this->Clear();
        this->SetDataPtrMCSR(&ptr, &col, &val, nnz, static_cast<int>(nrow), static_cast<int>(ncol));

        return true;
    }

    template <typename ValueType>
    bool HostMatrixMCSR<ValueType>::WriteFileRSIO(const std::string& filename) const
    {
        return write_matrix_mcsr_rocsparseio(this->nrow_,
                                             this->ncol_,
                                             this->nnz_,
                                             this->mat_.row_offset,
                                             this->mat_.col,
                                             this->mat_.val,
                                             filename.c_str());
    }

    template <typename ValueType>
    void HostMatrixMCSR<ValueType>::Apply(const BaseVector<ValueType>& in,
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

            assert(this->nrow_ == this->ncol_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int ai = 0; ai < this->nrow_; ++ai)
            {
                ValueType sum = this->mat_.val[ai] * cast_in->vec_[ai];

                for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
                {
                    sum += this->mat_.val[aj] * cast_in->vec_[this->mat_.col[aj]];
                }

                cast_out->vec_[ai] = sum;
            }
        }
    }

    template <typename ValueType>
    void HostMatrixMCSR<ValueType>::ApplyAdd(const BaseVector<ValueType>& in,
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

            assert(this->nrow_ == this->ncol_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int ai = 0; ai < this->nrow_; ++ai)
            {
                cast_out->vec_[ai] += scalar * this->mat_.val[ai] * cast_in->vec_[ai];

                for(int aj = this->mat_.row_offset[ai]; aj < this->mat_.row_offset[ai + 1]; ++aj)
                {
                    cast_out->vec_[ai]
                        += scalar * this->mat_.val[aj] * cast_in->vec_[this->mat_.col[aj]];
                }
            }
        }
    }

    template class HostMatrixMCSR<double>;
    template class HostMatrixMCSR<float>;
#ifdef SUPPORT_COMPLEX
    template class HostMatrixMCSR<std::complex<double>>;
    template class HostMatrixMCSR<std::complex<float>>;
#endif

} // namespace rocalution

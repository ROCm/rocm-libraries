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

#include "host_matrix_ell.hpp"
#include "../../utils/allocate_free.hpp"
#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "../matrix_formats_ind.hpp"
#include "host_conversion.hpp"
#include "host_io.hpp"
#include "host_matrix_csr.hpp"
#include "host_vector.hpp"

#include <complex>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_set_num_threads(num) ;
#endif

namespace rocalution
{

    // LCOV_EXCL_START
    template <typename ValueType>
    HostMatrixELL<ValueType>::HostMatrixELL()
    {
        // no default constructors
        LOG_INFO("no default constructor");
        FATAL_ERROR(__FILE__, __LINE__);
    }
    // LCOV_EXCL_STOP

    template <typename ValueType>
    HostMatrixELL<ValueType>::HostMatrixELL(const Rocalution_Backend_Descriptor& local_backend)
    {
        log_debug(this, "HostMatrixELL::HostMatrixELL()", "constructor with local_backend");

        this->mat_.val     = NULL;
        this->mat_.col     = NULL;
        this->mat_.max_row = 0;

        this->set_backend(local_backend);
    }

    template <typename ValueType>
    HostMatrixELL<ValueType>::~HostMatrixELL()
    {
        log_debug(this, "HostMatrixELL::~HostMatrixELL()", "destructor");

        this->Clear();
    }

    template <typename ValueType>
    void HostMatrixELL<ValueType>::Info(void) const
    {
        LOG_INFO("HostMatrixELL<ValueType>");
    }

    template <typename ValueType>
    void HostMatrixELL<ValueType>::Clear()
    {
        free_host(&this->mat_.val);
        free_host(&this->mat_.col);

        this->nrow_ = 0;
        this->ncol_ = 0;
        this->nnz_  = 0;
    }

    template <typename ValueType>
    void HostMatrixELL<ValueType>::AllocateELL(int64_t nnz, int nrow, int ncol, int max_row)
    {
        assert(nnz >= 0);
        assert(ncol >= 0);
        assert(nrow >= 0);
        assert(max_row >= 0);
        assert(nnz == max_row * nrow);

        this->Clear();

        allocate_host(nnz, &this->mat_.val);
        allocate_host(nnz, &this->mat_.col);

        set_to_zero_host(nnz, this->mat_.val);
        set_to_zero_host(nnz, this->mat_.col);

        this->mat_.max_row = max_row;
        this->nrow_        = nrow;
        this->ncol_        = ncol;
        this->nnz_         = nnz;
    }

    template <typename ValueType>
    void HostMatrixELL<ValueType>::SetDataPtrELL(
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

        this->mat_.max_row = max_row;
        this->nrow_        = nrow;
        this->ncol_        = ncol;
        this->nnz_         = nnz;

        this->mat_.col = *col;
        this->mat_.val = *val;
    }

    template <typename ValueType>
    void HostMatrixELL<ValueType>::LeaveDataPtrELL(int** col, ValueType** val, int& max_row)
    {
        assert(this->nrow_ >= 0);
        assert(this->ncol_ >= 0);
        assert(this->nnz_ >= 0);
        assert(this->mat_.max_row >= 0);
        assert(this->mat_.max_row * this->nrow_ == this->nnz_);

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
    void HostMatrixELL<ValueType>::CopyFrom(const BaseMatrix<ValueType>& mat)
    {
        // copy only in the same format
        assert(this->GetMatFormat() == mat.GetMatFormat());

        if(const HostMatrixELL<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixELL<ValueType>*>(&mat))
        {
            if(this->nnz_ == 0)
            {
                this->AllocateELL(
                    cast_mat->nnz_, cast_mat->nrow_, cast_mat->ncol_, cast_mat->mat_.max_row);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

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
    void HostMatrixELL<ValueType>::CopyTo(BaseMatrix<ValueType>* mat) const
    {
        mat->CopyFrom(*this);
    }

    template <typename ValueType>
    bool HostMatrixELL<ValueType>::ConvertFrom(const BaseMatrix<ValueType>& mat)
    {
        this->Clear();

        // Empty matrix
        if(mat.GetNnz() == 0)
        {
            this->AllocateELL(0, mat.GetM(), mat.GetN(), 0);

            return true;
        }

        if(const HostMatrixELL<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixELL<ValueType>*>(&mat))
        {
            this->CopyFrom(*cast_mat);
            return true;
        }

        if(const HostMatrixCSR<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixCSR<ValueType>*>(&mat))
        {
            this->Clear();
            int64_t nnz = 0;

            if(csr_to_ell(this->local_backend_.OpenMP_threads,
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

        return false;
    }

    template <typename ValueType>
    bool HostMatrixELL<ValueType>::ReadFileRSIO(const std::string& filename)
    {
        int64_t nrow;
        int64_t ncol;
        int64_t nnz;
        int64_t width;

        int*       col = NULL;
        ValueType* val = NULL;

        if(read_matrix_ell_rocsparseio(nrow, ncol, nnz, width, &col, &val, filename.c_str())
           != true)
        {
            return false;
        }

        this->Clear();
        this->SetDataPtrELL(&col, &val, nnz, nrow, ncol, width);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixELL<ValueType>::WriteFileRSIO(const std::string& filename) const
    {
        return write_matrix_ell_rocsparseio(this->nrow_,
                                            this->ncol_,
                                            this->mat_.max_row,
                                            this->mat_.col,
                                            this->mat_.val,
                                            filename.c_str());
    }

    template <typename ValueType>
    void HostMatrixELL<ValueType>::Apply(const BaseVector<ValueType>& in,
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
                ValueType sum = static_cast<ValueType>(0);

                for(int n = 0; n < this->mat_.max_row; ++n)
                {
                    int aj     = ELL_IND(ai, n, this->nrow_, this->mat_.max_row);
                    int col_aj = this->mat_.col[aj];

                    if(col_aj >= 0)
                    {
                        sum += this->mat_.val[aj] * cast_in->vec_[col_aj];
                    }
                    else
                    {
                        break;
                    }
                }

                cast_out->vec_[ai] = sum;
            }
        }
    }

    template <typename ValueType>
    void HostMatrixELL<ValueType>::ApplyAdd(const BaseVector<ValueType>& in,
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
                for(int n = 0; n < this->mat_.max_row; ++n)
                {
                    int aj     = ELL_IND(ai, n, this->nrow_, this->mat_.max_row);
                    int col_aj = this->mat_.col[aj];

                    if(col_aj >= 0)
                    {
                        cast_out->vec_[ai] += scalar * this->mat_.val[aj] * cast_in->vec_[col_aj];
                    }
                    else
                    {
                        break;
                    }
                }
            }
        }
    }

    template class HostMatrixELL<double>;
    template class HostMatrixELL<float>;
#ifdef SUPPORT_COMPLEX
    template class HostMatrixELL<std::complex<double>>;
    template class HostMatrixELL<std::complex<float>>;
#endif

} // namespace rocalution

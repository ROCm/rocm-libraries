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

#include "host_matrix_dia.hpp"
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
    HostMatrixDIA<ValueType>::HostMatrixDIA()
    {
        // no default constructors
        LOG_INFO("no default constructor");
        FATAL_ERROR(__FILE__, __LINE__);
    }
    // LCOV_EXCL_STop

    template <typename ValueType>
    HostMatrixDIA<ValueType>::HostMatrixDIA(const Rocalution_Backend_Descriptor& local_backend)
    {
        log_debug(this, "HostMatrixDIA::HostMatrixDIA()", "constructor with local_backend");

        this->mat_.val      = NULL;
        this->mat_.offset   = NULL;
        this->mat_.num_diag = 0;
        this->set_backend(local_backend);
    }

    template <typename ValueType>
    HostMatrixDIA<ValueType>::~HostMatrixDIA()
    {
        log_debug(this, "HostMatrixDIA::~HostMatrixDIA()", "destructor");

        this->Clear();
    }

    template <typename ValueType>
    void HostMatrixDIA<ValueType>::Info(void) const
    {
        LOG_INFO("HostMatrixDIA<ValueType>, diag = " << this->mat_.num_diag
                                                     << " nnz=" << this->nnz_);
    }

    template <typename ValueType>
    void HostMatrixDIA<ValueType>::Clear()
    {
        free_host(&this->mat_.val);
        free_host(&this->mat_.offset);

        this->nrow_         = 0;
        this->ncol_         = 0;
        this->nnz_          = 0;
        this->mat_.num_diag = 0;
    }

    template <typename ValueType>
    void HostMatrixDIA<ValueType>::AllocateDIA(int64_t nnz, int nrow, int ncol, int ndiag)
    {
        assert(nnz >= 0);
        assert(ncol >= 0);
        assert(nrow >= 0);
        assert(ndiag >= 0);

        this->Clear();

        allocate_host(nnz, &this->mat_.val);
        allocate_host(ndiag, &this->mat_.offset);

        set_to_zero_host(nnz, mat_.val);
        set_to_zero_host(ndiag, mat_.offset);

        this->nrow_         = nrow;
        this->ncol_         = ncol;
        this->nnz_          = nnz;
        this->mat_.num_diag = ndiag;
    }

    template <typename ValueType>
    void HostMatrixDIA<ValueType>::SetDataPtrDIA(
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

        this->mat_.num_diag = num_diag;
        this->nrow_         = nrow;
        this->ncol_         = ncol;
        this->nnz_          = nnz;

        this->mat_.offset = *offset;
        this->mat_.val    = *val;
    }

    template <typename ValueType>
    void HostMatrixDIA<ValueType>::LeaveDataPtrDIA(int** offset, ValueType** val, int& num_diag)
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
    void HostMatrixDIA<ValueType>::CopyFrom(const BaseMatrix<ValueType>& mat)
    {
        // copy only in the same format
        assert(this->GetMatFormat() == mat.GetMatFormat());

        if(const HostMatrixDIA<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixDIA<ValueType>*>(&mat))
        {
            if(this->nnz_ == 0)
            {
                this->AllocateDIA(
                    cast_mat->nnz_, cast_mat->nrow_, cast_mat->ncol_, cast_mat->mat_.num_diag);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

            copy_h2h(this->nnz_, cast_mat->mat_.val, this->mat_.val);
            copy_h2h(this->mat_.num_diag, cast_mat->mat_.offset, this->mat_.offset);
        }
        else
        {
            // Host matrix knows only host matrices
            // -> dispatching
            mat.CopyTo(this);
        }
    }

    template <typename ValueType>
    void HostMatrixDIA<ValueType>::CopyTo(BaseMatrix<ValueType>* mat) const
    {
        mat->CopyFrom(*this);
    }

    template <typename ValueType>
    bool HostMatrixDIA<ValueType>::ConvertFrom(const BaseMatrix<ValueType>& mat)
    {
        this->Clear();

        // Empty matrix
        if(mat.GetNnz() == 0)
        {
            this->AllocateDIA(0, mat.GetM(), mat.GetN(), 0);

            return true;
        }

        if(const HostMatrixDIA<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixDIA<ValueType>*>(&mat))
        {
            this->CopyFrom(*cast_mat);
            return true;
        }

        if(const HostMatrixCSR<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixCSR<ValueType>*>(&mat))
        {
            this->Clear();
            int64_t nnz = 0;

            if(csr_to_dia(this->local_backend_.OpenMP_threads,
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
    bool HostMatrixDIA<ValueType>::ReadFileRSIO(const std::string& filename)
    {
        int64_t nrow;
        int64_t ncol;
        int64_t nnz;
        int64_t ndiag;

        int*       offset = NULL;
        ValueType* val    = NULL;

        if(read_matrix_dia_rocsparseio(nrow, ncol, nnz, ndiag, &offset, &val, filename.c_str())
           != true)
        {
            return false;
        }

        this->Clear();
        this->SetDataPtrDIA(&offset, &val, nnz, nrow, ncol, ndiag);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixDIA<ValueType>::WriteFileRSIO(const std::string& filename) const
    {
        return write_matrix_dia_rocsparseio(this->nrow_,
                                            this->ncol_,
                                            this->mat_.num_diag,
                                            this->mat_.offset,
                                            this->mat_.val,
                                            filename.c_str());
    }

    template <typename ValueType>
    void HostMatrixDIA<ValueType>::Apply(const BaseVector<ValueType>& in,
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
            for(int i = 0; i < this->nrow_; ++i)
            {
                ValueType sum = static_cast<ValueType>(0);

                for(int j = 0; j < this->mat_.num_diag; ++j)
                {
                    int start    = 0;
                    int end      = this->nrow_;
                    int v_offset = 0;
                    int offset   = this->mat_.offset[j];

                    if(offset < 0)
                    {
                        start -= offset;
                        v_offset = -start;
                    }
                    else
                    {
                        end -= offset;
                        v_offset = offset;
                    }

                    if((i >= start) && (i < end))
                    {
                        sum += this->mat_.val[DIA_IND(i, j, this->nrow_, this->mat_.num_diag)]
                               * cast_in->vec_[i + v_offset];
                    }
                    else if(i >= end)
                    {
                        break;
                    }
                }

                cast_out->vec_[i] = sum;
            }
        }
    }

    template <typename ValueType>
    void HostMatrixDIA<ValueType>::ApplyAdd(const BaseVector<ValueType>& in,
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
            for(int i = 0; i < this->nrow_; ++i)
            {
                for(int j = 0; j < this->mat_.num_diag; ++j)
                {
                    int start    = 0;
                    int end      = this->nrow_;
                    int v_offset = 0;
                    int offset   = this->mat_.offset[j];

                    if(offset < 0)
                    {
                        start -= offset;
                        v_offset = -start;
                    }
                    else
                    {
                        end -= offset;
                        v_offset = offset;
                    }

                    if((i >= start) && (i < end))
                    {
                        cast_out->vec_[i]
                            += scalar
                               * this->mat_.val[DIA_IND(i, j, this->nrow_, this->mat_.num_diag)]
                               * cast_in->vec_[i + v_offset];
                    }
                    else if(i >= end)
                    {
                        break;
                    }
                }
            }
        }
    }

    template class HostMatrixDIA<double>;
    template class HostMatrixDIA<float>;
#ifdef SUPPORT_COMPLEX
    template class HostMatrixDIA<std::complex<double>>;
    template class HostMatrixDIA<std::complex<float>>;
#endif

} // namespace rocalution

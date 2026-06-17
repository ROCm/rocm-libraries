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

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) \
    || defined(__WIN64) && !defined(__CYGWIN__)
#define _CRT_SECURE_NO_DEPRECATE
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "host_matrix_coo.hpp"
#include "../../utils/allocate_free.hpp"
#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "host_conversion.hpp"
#include "host_io.hpp"
#include "host_matrix_csr.hpp"
#include "host_vector.hpp"

#include <algorithm>
#include <complex>
#include <cstdio>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_set_num_threads(num) ;
#endif

namespace rocalution
{

    // LCOV_EXCL_START
    template <typename ValueType>
    HostMatrixCOO<ValueType>::HostMatrixCOO()
    {
        // no default constructors
        LOG_INFO("no default constructor");
        FATAL_ERROR(__FILE__, __LINE__);
    }
    // LCOV_EXCL_STOP

    template <typename ValueType>
    HostMatrixCOO<ValueType>::HostMatrixCOO(const Rocalution_Backend_Descriptor& local_backend)
    {
        log_debug(this, "HostMatrixCOO::HostMatrixCOO()", "constructor with local_backend");

        this->mat_.row = NULL;
        this->mat_.col = NULL;
        this->mat_.val = NULL;
        this->set_backend(local_backend);
    }

    template <typename ValueType>
    HostMatrixCOO<ValueType>::~HostMatrixCOO()
    {
        log_debug(this, "HostMatrixCOO::~HostMatrixCOO()", "destructor");

        this->Clear();
    }
    template <typename ValueType>
    void HostMatrixCOO<ValueType>::Info(void) const
    {
        LOG_INFO("HostMatrixCOO<ValueType>");
    }

    template <typename ValueType>
    void HostMatrixCOO<ValueType>::Clear()
    {
        free_host(&this->mat_.row);
        free_host(&this->mat_.col);
        free_host(&this->mat_.val);

        this->nrow_ = 0;
        this->ncol_ = 0;
        this->nnz_  = 0;
    }

    template <typename ValueType>
    void HostMatrixCOO<ValueType>::AllocateCOO(int64_t nnz, int nrow, int ncol)
    {
        assert(nnz >= 0);
        assert(ncol >= 0);
        assert(nrow >= 0);

        this->Clear();

        allocate_host(nnz, &this->mat_.row);
        allocate_host(nnz, &this->mat_.col);
        allocate_host(nnz, &this->mat_.val);

        set_to_zero_host(nnz, this->mat_.row);
        set_to_zero_host(nnz, this->mat_.col);
        set_to_zero_host(nnz, this->mat_.val);

        this->nrow_ = nrow;
        this->ncol_ = ncol;
        this->nnz_  = nnz;
    }

    template <typename ValueType>
    void HostMatrixCOO<ValueType>::SetDataPtrCOO(
        int** row, int** col, ValueType** val, int64_t nnz, int nrow, int ncol)
    {
        assert(nnz >= 0);
        assert(nrow >= 0);
        assert(ncol >= 0);

        if(nnz > 0)
        {
            assert(*row != NULL);
            assert(*col != NULL);
            assert(*val != NULL);
        }

        this->Clear();

        this->nrow_ = nrow;
        this->ncol_ = ncol;
        this->nnz_  = nnz;

        this->mat_.row = *row;
        this->mat_.col = *col;
        this->mat_.val = *val;
    }

    template <typename ValueType>
    void HostMatrixCOO<ValueType>::LeaveDataPtrCOO(int** row, int** col, ValueType** val)
    {
        assert(this->nrow_ >= 0);
        assert(this->ncol_ >= 0);
        assert(this->nnz_ >= 0);

        *row = this->mat_.row;
        *col = this->mat_.col;
        *val = this->mat_.val;

        this->mat_.row = NULL;
        this->mat_.col = NULL;
        this->mat_.val = NULL;

        this->nrow_ = 0;
        this->ncol_ = 0;
        this->nnz_  = 0;
    }

    template <typename ValueType>
    void HostMatrixCOO<ValueType>::CopyFromCOO(const int* row, const int* col, const ValueType* val)
    {
        if(this->nnz_ > 0)
        {
            assert(this->nrow_ > 0);
            assert(this->ncol_ > 0);
            assert(row != NULL);
            assert(col != NULL);
            assert(val != NULL);

            copy_h2h(this->nnz_, row, this->mat_.row);
            copy_h2h(this->nnz_, col, this->mat_.col);
            copy_h2h(this->nnz_, val, this->mat_.val);
        }
    }

    template <typename ValueType>
    void HostMatrixCOO<ValueType>::CopyToCOO(int* row, int* col, ValueType* val) const
    {
        if(this->nnz_ > 0)
        {
            assert(this->nrow_ > 0);
            assert(this->ncol_ > 0);
            assert(row != NULL);
            assert(col != NULL);
            assert(val != NULL);

            copy_h2h(this->nnz_, this->mat_.row, row);
            copy_h2h(this->nnz_, this->mat_.col, col);
            copy_h2h(this->nnz_, this->mat_.val, val);
        }
    }

    template <typename ValueType>
    void HostMatrixCOO<ValueType>::CopyFrom(const BaseMatrix<ValueType>& mat)
    {
        // copy only in the same format
        assert(this->GetMatFormat() == mat.GetMatFormat());

        if(const HostMatrixCOO<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixCOO<ValueType>*>(&mat))
        {
            if(this->nnz_ == 0)
            {
                this->AllocateCOO(cast_mat->nnz_, cast_mat->nrow_, cast_mat->ncol_);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

            copy_h2h(this->nnz_, cast_mat->mat_.row, this->mat_.row);
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
    void HostMatrixCOO<ValueType>::CopyTo(BaseMatrix<ValueType>* mat) const
    {
        mat->CopyFrom(*this);
    }

    template <typename ValueType>
    bool HostMatrixCOO<ValueType>::ConvertFrom(const BaseMatrix<ValueType>& mat)
    {
        this->Clear();

        // Empty matrix
        if(mat.GetNnz() == 0)
        {
            this->AllocateCOO(mat.GetNnz(), mat.GetM(), mat.GetN());

            return true;
        }

        if(const HostMatrixCOO<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixCOO<ValueType>*>(&mat))
        {
            this->CopyFrom(*cast_mat);
            return true;
        }

        if(const HostMatrixCSR<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixCSR<ValueType>*>(&mat))
        {
            this->Clear();

            if(csr_to_coo(this->local_backend_.OpenMP_threads,
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
    bool HostMatrixCOO<ValueType>::ReadFileMTX(const std::string& filename)
    {
        int     nrow;
        int     ncol;
        int64_t nnz;

        int*       row = NULL;
        int*       col = NULL;
        ValueType* val = NULL;

        if(read_matrix_mtx(nrow, ncol, nnz, &row, &col, &val, filename.c_str()) != true)
        {
            return false;
        }

        this->Clear();
        this->SetDataPtrCOO(&row, &col, &val, nnz, nrow, ncol);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCOO<ValueType>::WriteFileMTX(const std::string& filename) const
    {
        if(write_matrix_mtx(this->nrow_,
                            this->ncol_,
                            this->nnz_,
                            this->mat_.row,
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
    bool HostMatrixCOO<ValueType>::ReadFileRSIO(const std::string& filename)
    {
        int64_t nrow;
        int64_t ncol;
        int64_t nnz;

        int*       row = NULL;
        int*       col = NULL;
        ValueType* val = NULL;

        if(read_matrix_coo_rocsparseio(nrow, ncol, nnz, &row, &col, &val, filename.c_str()) != true)
        {
            return false;
        }

        this->Clear();
        this->SetDataPtrCOO(&row, &col, &val, nnz, nrow, ncol);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCOO<ValueType>::WriteFileRSIO(const std::string& filename) const
    {
        return write_matrix_coo_rocsparseio(this->nrow_,
                                            this->ncol_,
                                            this->nnz_,
                                            this->mat_.row,
                                            this->mat_.col,
                                            this->mat_.val,
                                            filename.c_str());
    }

    template <typename ValueType>
    void HostMatrixCOO<ValueType>::Apply(const BaseVector<ValueType>& in,
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

        _set_omp_backend_threads(this->local_backend_, this->nnz_);

        set_to_zero_host(this->nrow_, cast_out->vec_);

        for(int64_t i = 0; i < this->nnz_; ++i)
        {
            cast_out->vec_[this->mat_.row[i]]
                += this->mat_.val[i] * cast_in->vec_[this->mat_.col[i]];
        }
    }

    template <typename ValueType>
    void HostMatrixCOO<ValueType>::ApplyAdd(const BaseVector<ValueType>& in,
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

            for(int64_t i = 0; i < this->nnz_; ++i)
            {
                cast_out->vec_[this->mat_.row[i]]
                    += scalar * this->mat_.val[i] * cast_in->vec_[this->mat_.col[i]];
            }
        }
    }

    template <typename ValueType>
    bool HostMatrixCOO<ValueType>::Sort(void)
    {
        if(this->nnz_ > 0)
        {
            // Sort by row and column index
            std::vector<int64_t> perm(this->nnz_);

            // Create identity permutation
            std::iota(perm.begin(), perm.end(), 0);

            int*       row = this->mat_.row;
            int*       col = this->mat_.col;
            ValueType* val = this->mat_.val;

            this->mat_.row = NULL;
            this->mat_.col = NULL;
            this->mat_.val = NULL;

            allocate_host(this->nnz_, &this->mat_.row);
            allocate_host(this->nnz_, &this->mat_.col);
            allocate_host(this->nnz_, &this->mat_.val);

            // Compare function object to sort by row first, then by column
            std::sort(perm.begin(), perm.end(), [&](const int& a, const int& b) {
                if(row[a] < row[b])
                    return true;
                if(row[a] == row[b])
                    return col[a] < col[b];
                return false;
            });

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int64_t i = 0; i < this->nnz_; ++i)
            {
                this->mat_.row[i] = row[perm[i]];
                this->mat_.col[i] = col[perm[i]];
                this->mat_.val[i] = val[perm[i]];
            }

            free_host(&row);
            free_host(&col);
            free_host(&val);
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCOO<ValueType>::Permute(const BaseVector<int>& permutation)
    {
        // symmetric permutation only
        assert((permutation.GetSize() == this->nrow_) && (permutation.GetSize() == this->ncol_));

        const HostVector<int>* cast_perm = dynamic_cast<const HostVector<int>*>(&permutation);
        assert(cast_perm != NULL);

        HostMatrixCOO<ValueType> src(this->local_backend_);
        src.AllocateCOO(this->nnz_, this->nrow_, this->ncol_);
        src.CopyFrom(*this);

        _set_omp_backend_threads(this->local_backend_, this->nnz_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int64_t i = 0; i < this->nnz_; ++i)
        {
            this->mat_.row[i] = cast_perm->vec_[src.mat_.row[i]];
            this->mat_.col[i] = cast_perm->vec_[src.mat_.col[i]];
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCOO<ValueType>::PermuteBackward(const BaseVector<int>& permutation)
    {
        // symmetric permutation only
        assert((permutation.GetSize() == this->nrow_) && (permutation.GetSize() == this->ncol_));

        const HostVector<int>* cast_perm = dynamic_cast<const HostVector<int>*>(&permutation);
        assert(cast_perm != NULL);

        HostMatrixCOO<ValueType> src(this->local_backend_);
        src.AllocateCOO(this->nnz_, this->nrow_, this->ncol_);
        src.CopyFrom(*this);

        _set_omp_backend_threads(this->local_backend_, this->nnz_);

        // TODO
        // Is there a better way?
        int* pb = NULL;
        allocate_host(this->nrow_, &pb);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int i = 0; i < this->nrow_; ++i)
        {
            pb[cast_perm->vec_[i]] = i;
        }

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int64_t i = 0; i < this->nnz_; ++i)
        {
            this->mat_.row[i] = pb[src.mat_.row[i]];
            this->mat_.col[i] = pb[src.mat_.col[i]];
        }

        free_host(&pb);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCOO<ValueType>::Scale(ValueType alpha)
    {
        _set_omp_backend_threads(this->local_backend_, this->nnz_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int64_t i = 0; i < this->nnz_; ++i)
        {
            this->mat_.val[i] *= alpha;
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCOO<ValueType>::ScaleDiagonal(ValueType alpha)
    {
        _set_omp_backend_threads(this->local_backend_, this->nnz_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int64_t i = 0; i < this->nnz_; ++i)
        {
            if(this->mat_.row[i] == this->mat_.col[i])
            {
                this->mat_.val[i] *= alpha;
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCOO<ValueType>::ScaleOffDiagonal(ValueType alpha)
    {
        _set_omp_backend_threads(this->local_backend_, this->nnz_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int64_t i = 0; i < this->nnz_; ++i)
        {
            if(this->mat_.row[i] != this->mat_.col[i])
            {
                this->mat_.val[i] *= alpha;
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCOO<ValueType>::AddScalar(ValueType alpha)
    {
        _set_omp_backend_threads(this->local_backend_, this->nnz_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int64_t i = 0; i < this->nnz_; ++i)
        {
            this->mat_.val[i] += alpha;
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCOO<ValueType>::AddScalarDiagonal(ValueType alpha)
    {
        _set_omp_backend_threads(this->local_backend_, this->nnz_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int64_t i = 0; i < this->nnz_; ++i)
        {
            if(this->mat_.row[i] == this->mat_.col[i])
            {
                this->mat_.val[i] += alpha;
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixCOO<ValueType>::AddScalarOffDiagonal(ValueType alpha)
    {
        _set_omp_backend_threads(this->local_backend_, this->nnz_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int64_t i = 0; i < this->nnz_; ++i)
        {
            if(this->mat_.row[i] != this->mat_.col[i])
            {
                this->mat_.val[i] += alpha;
            }
        }

        return true;
    }

    template class HostMatrixCOO<double>;
    template class HostMatrixCOO<float>;
#ifdef SUPPORT_COMPLEX
    template class HostMatrixCOO<std::complex<double>>;
    template class HostMatrixCOO<std::complex<float>>;
#endif

} // namespace rocalution

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

#include "host_matrix_dense.hpp"
#include "../../utils/allocate_free.hpp"
#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "../../utils/math_functions.hpp"
#include "../matrix_formats_ind.hpp"
#include "host_conversion.hpp"
#include "host_io.hpp"
#include "host_matrix_csr.hpp"
#include "host_vector.hpp"

#include <complex>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_set_num_threads(num) ;
#endif

namespace rocalution
{

    // LCOV_EXCL_START
    template <typename ValueType>
    HostMatrixDENSE<ValueType>::HostMatrixDENSE()
    {
        // no default constructors
        LOG_INFO("no default constructor");
        FATAL_ERROR(__FILE__, __LINE__);
    }
    // LCOV_EXCL_STOP

    template <typename ValueType>
    HostMatrixDENSE<ValueType>::HostMatrixDENSE(const Rocalution_Backend_Descriptor& local_backend)
    {
        log_debug(this, "HostMatrixDENSE::HostMatrixDENSE()", "constructor with local_backend");

        this->mat_.val = NULL;
        this->set_backend(local_backend);
    }

    template <typename ValueType>
    HostMatrixDENSE<ValueType>::~HostMatrixDENSE()
    {
        log_debug(this, "HostMatrixDENSE::~HostMatrixDENSE()", "destructor");

        this->Clear();
    }

    template <typename ValueType>
    void HostMatrixDENSE<ValueType>::Info(void) const
    {
        LOG_INFO("HostMatrixDENSE<ValueType>");

        if(DENSE_IND_BASE == 0)
        {
            LOG_INFO("Dense matrix - row-based");
        }
        else
        {
            assert(DENSE_IND_BASE == 1);
            LOG_INFO("Dense matrix - column-based");
        }
    }

    template <typename ValueType>
    void HostMatrixDENSE<ValueType>::Clear()
    {
        free_host(&this->mat_.val);

        this->nrow_ = 0;
        this->ncol_ = 0;
        this->nnz_  = 0;
    }

    template <typename ValueType>
    void HostMatrixDENSE<ValueType>::AllocateDENSE(int nrow, int ncol)
    {
        assert(ncol >= 0);
        assert(nrow >= 0);

        this->Clear();

        allocate_host(nrow * ncol, &this->mat_.val);
        set_to_zero_host(nrow * ncol, mat_.val);

        this->nrow_ = nrow;
        this->ncol_ = ncol;
        this->nnz_  = nrow * ncol;
    }

    template <typename ValueType>
    void HostMatrixDENSE<ValueType>::SetDataPtrDENSE(ValueType** val, int nrow, int ncol)
    {
        assert(nrow >= 0);
        assert(ncol >= 0);

        if(nrow * ncol > 0)
        {
            assert(*val != NULL);
        }

        this->Clear();

        this->nrow_ = nrow;
        this->ncol_ = ncol;
        this->nnz_  = nrow * ncol;

        this->mat_.val = *val;
    }

    template <typename ValueType>
    void HostMatrixDENSE<ValueType>::LeaveDataPtrDENSE(ValueType** val)
    {
        assert(this->nrow_ >= 0);
        assert(this->ncol_ >= 0);
        assert(this->nnz_ >= 0);
        assert(this->nnz_ == this->nrow_ * this->ncol_);

        *val = this->mat_.val;

        this->mat_.val = NULL;

        this->nrow_ = 0;
        this->ncol_ = 0;
        this->nnz_  = 0;
    }

    template <typename ValueType>
    void HostMatrixDENSE<ValueType>::CopyFrom(const BaseMatrix<ValueType>& mat)
    {
        // copy only in the same format
        assert(this->GetMatFormat() == mat.GetMatFormat());

        if(const HostMatrixDENSE<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixDENSE<ValueType>*>(&mat))
        {
            if(this->nnz_ == 0)
            {
                this->AllocateDENSE(cast_mat->nrow_, cast_mat->ncol_);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);

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
    void HostMatrixDENSE<ValueType>::CopyTo(BaseMatrix<ValueType>* mat) const
    {
        mat->CopyFrom(*this);
    }

    template <typename ValueType>
    bool HostMatrixDENSE<ValueType>::ConvertFrom(const BaseMatrix<ValueType>& mat)
    {
        this->Clear();

        // Empty matrix
        if(mat.GetNnz() == 0)
        {
            this->AllocateDENSE(mat.GetM(), mat.GetN());

            return true;
        }

        if(const HostMatrixDENSE<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixDENSE<ValueType>*>(&mat))
        {
            this->CopyFrom(*cast_mat);
            return true;
        }

        if(const HostMatrixCSR<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixCSR<ValueType>*>(&mat))
        {
            this->Clear();

            if(csr_to_dense(this->local_backend_.OpenMP_threads,
                            cast_mat->nnz_,
                            cast_mat->nrow_,
                            cast_mat->ncol_,
                            cast_mat->mat_,
                            &this->mat_)
               == true)
            {
                this->nrow_ = cast_mat->nrow_;
                this->ncol_ = cast_mat->ncol_;
                this->nnz_  = this->nrow_ * this->ncol_;

                return true;
            }
        }

        return false;
    }

    template <typename ValueType>
    bool HostMatrixDENSE<ValueType>::ReadFileRSIO(const std::string& filename)
    {
        int64_t nrow;
        int64_t ncol;

        ValueType* val = NULL;

        if(read_matrix_dense_rocsparseio(nrow, ncol, &val, filename.c_str()) != true)
        {
            return false;
        }

        this->Clear();
        this->SetDataPtrDENSE(&val, nrow, ncol);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixDENSE<ValueType>::WriteFileRSIO(const std::string& filename) const
    {
        return write_matrix_dense_rocsparseio(
            this->nrow_, this->ncol_, this->mat_.val, filename.c_str());
    }

    template <typename ValueType>
    void HostMatrixDENSE<ValueType>::Apply(const BaseVector<ValueType>& in,
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

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int ai = 0; ai < this->nrow_; ++ai)
        {
            cast_out->vec_[ai] = static_cast<ValueType>(0);
            for(int aj = 0; aj < this->ncol_; ++aj)
            {
                cast_out->vec_[ai] += this->mat_.val[DENSE_IND(ai, aj, this->nrow_, this->ncol_)]
                                      * cast_in->vec_[aj];
            }
        }
    }

    template <typename ValueType>
    void HostMatrixDENSE<ValueType>::ApplyAdd(const BaseVector<ValueType>& in,
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

            _set_omp_backend_threads(this->local_backend_, this->nnz_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int ai = 0; ai < this->nrow_; ++ai)
            {
                for(int aj = 0; aj < this->ncol_; ++aj)
                {
                    cast_out->vec_[ai]
                        += scalar * this->mat_.val[DENSE_IND(ai, aj, this->nrow_, this->ncol_)]
                           * cast_in->vec_[aj];
                }
            }
        }
    }

    template <typename ValueType>
    bool HostMatrixDENSE<ValueType>::MatMatMult(const BaseMatrix<ValueType>& A,
                                                const BaseMatrix<ValueType>& B)
    {
        assert((this != &A) && (this != &B));

        const HostMatrixDENSE<ValueType>* cast_mat_A
            = dynamic_cast<const HostMatrixDENSE<ValueType>*>(&A);
        const HostMatrixDENSE<ValueType>* cast_mat_B
            = dynamic_cast<const HostMatrixDENSE<ValueType>*>(&B);

        assert(cast_mat_A != NULL);
        assert(cast_mat_B != NULL);
        assert(cast_mat_A->ncol_ == cast_mat_B->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int i = 0; i < cast_mat_A->nrow_; ++i)
        {
            for(int j = 0; j < cast_mat_B->ncol_; ++j)
            {
                ValueType sum = static_cast<ValueType>(0);

                for(int k = 0; k < cast_mat_A->ncol_; ++k)
                {
                    sum += cast_mat_A->mat_
                               .val[DENSE_IND(i, k, cast_mat_A->nrow_, cast_mat_A->ncol_)]
                           * cast_mat_B->mat_
                                 .val[DENSE_IND(k, j, cast_mat_B->nrow_, cast_mat_B->ncol_)];
                }

                this->mat_.val[DENSE_IND(i, j, cast_mat_A->nrow_, cast_mat_B->ncol_)] = sum;
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixDENSE<ValueType>::Householder(int                    idx,
                                                 ValueType&             beta,
                                                 BaseVector<ValueType>* vec) const
    {
        HostVector<ValueType>* cast_vec = dynamic_cast<HostVector<ValueType>*>(vec);
        assert(cast_vec != NULL);
        assert(cast_vec->GetSize() >= this->nrow_ - idx);

        double s = 0.0;

        for(int i = 1; i < this->nrow_ - idx; ++i)
        {
            cast_vec->vec_[i] = this->mat_.val[DENSE_IND(i + idx, idx, this->nrow_, this->ncol_)];
        }

        for(int i = idx + 1; i < this->nrow_; ++i)
        {
            s += std::norm(cast_vec->vec_[i - idx]);
        }

        if(s == 0)
        {
            beta = static_cast<ValueType>(0);
        }
        else
        {
            ValueType aii = this->mat_.val[DENSE_IND(idx, idx, this->nrow_, this->ncol_)];
            ValueType nrm = std::sqrt(std::norm(aii) + s);

            if(aii == ValueType(0))
            {
                aii += nrm;
            }
            else
            {
                aii += aii / std::sqrt(std::norm(aii)) * nrm;
            }

            double squared = std::norm(aii);
            beta           = static_cast<ValueType>(2.0 * squared / (s + squared));

            cast_vec->vec_[0] = static_cast<ValueType>(1);
            aii               = static_cast<ValueType>(1) / aii;
            for(int i = 1; i < this->nrow_ - idx; ++i)
            {
                cast_vec->vec_[i] *= aii;
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixDENSE<ValueType>::QRDecompose(void)
    {
        assert(this->nrow_ > 0);
        assert(this->ncol_ > 0);
        assert(this->nnz_ > 0);

        int                   size = (this->nrow_ < this->ncol_) ? this->nrow_ : this->ncol_;
        ValueType             beta;
        HostVector<ValueType> v(this->local_backend_);
        v.Allocate(this->nrow_);

        for(int i = 0; i < size; ++i)
        {
            this->Householder(i, beta, &v);

            if(beta != static_cast<ValueType>(0))
            {
                for(int aj = i; aj < this->ncol_; ++aj)
                {
                    ValueType sum = this->mat_.val[DENSE_IND(i, aj, this->nrow_, this->ncol_)];
                    for(int ai = i + 1; ai < this->nrow_; ++ai)
                    {
                        sum += rocalution_conj(v.vec_[ai - i])
                               * this->mat_.val[DENSE_IND(ai, aj, this->nrow_, this->ncol_)];
                    }

                    sum *= beta;

                    this->mat_.val[DENSE_IND(i, aj, this->nrow_, this->ncol_)] -= sum;

                    for(int ai = i + 1; ai < this->nrow_; ++ai)
                    {
                        this->mat_.val[DENSE_IND(ai, aj, this->nrow_, this->ncol_)]
                            -= sum * v.vec_[ai - i];
                    }
                }

                for(int k = i + 1; k < this->nrow_; ++k)
                {
                    this->mat_.val[DENSE_IND(k, i, this->nrow_, this->ncol_)] = v.vec_[k - i];
                }
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixDENSE<ValueType>::QRSolve(const BaseVector<ValueType>& in,
                                             BaseVector<ValueType>*       out) const
    {
        assert(in.GetSize() >= 0);
        assert(out->GetSize() >= 0);
        assert(in.GetSize() == this->nrow_);
        assert(out->GetSize() == this->ncol_);

        HostVector<ValueType>* cast_out = dynamic_cast<HostVector<ValueType>*>(out);

        assert(cast_out != NULL);

        HostVector<ValueType> copy_in(this->local_backend_);
        copy_in.CopyFrom(in);

        int size = (this->nrow_ < this->ncol_) ? this->nrow_ : this->ncol_;

        // Apply Q^T on copy_in
        for(int i = 0; i < size; ++i)
        {
            ValueType sum = static_cast<ValueType>(1);
            for(int j = i + 1; j < this->nrow_; ++j)
            {
                sum += this->mat_.val[DENSE_IND(j, i, this->nrow_, this->ncol_)]
                       * this->mat_.val[DENSE_IND(j, i, this->nrow_, this->ncol_)];
            }

            sum = static_cast<ValueType>(2) / sum;

            if(sum != static_cast<ValueType>(2))
            {
                ValueType sum2 = copy_in.vec_[i];
                for(int j = i + 1; j < this->nrow_; ++j)
                {
                    sum2 += this->mat_.val[DENSE_IND(j, i, this->nrow_, this->ncol_)]
                            * copy_in.vec_[j];
                }

                sum2 *= sum;
                copy_in.vec_[i] -= sum2;

                for(int j = i + 1; j < this->nrow_; ++j)
                {
                    copy_in.vec_[j]
                        -= sum2 * this->mat_.val[DENSE_IND(j, i, this->nrow_, this->ncol_)];
                }
            }
        }

        // Backsolve Rx = Q^T b
        for(int i = size - 1; i >= 0; --i)
        {
            ValueType sum = static_cast<ValueType>(0);
            for(int j = i + 1; j < this->ncol_; ++j)
            {
                sum += this->mat_.val[DENSE_IND(i, j, this->nrow_, this->ncol_)]
                       * cast_out->vec_[j];
            }

            cast_out->vec_[i] = (copy_in.vec_[i] - sum)
                                / this->mat_.val[DENSE_IND(i, i, this->nrow_, this->ncol_)];
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixDENSE<ValueType>::Invert(void)
    {
        assert(this->nrow_ > 0);
        assert(this->ncol_ > 0);
        assert(this->nnz_ > 0);
        assert(this->nrow_ == this->ncol_);

        ValueType* val = NULL;
        allocate_host(this->nrow_ * this->ncol_, &val);

        this->QRDecompose();

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int i = 0; i < this->nrow_; ++i)
        {
            HostVector<ValueType> sol(this->local_backend_);
            HostVector<ValueType> rhs(this->local_backend_);
            sol.Allocate(this->nrow_);
            rhs.Allocate(this->nrow_);

            rhs.vec_[i] = static_cast<ValueType>(1);

            this->QRSolve(rhs, &sol);

            for(int j = 0; j < this->ncol_; ++j)
            {
                val[DENSE_IND(j, i, this->nrow_, this->ncol_)] = sol.vec_[j];
            }
        }

        free_host(&this->mat_.val);
        this->mat_.val = val;

        return true;
    }

    template <typename ValueType>
    bool HostMatrixDENSE<ValueType>::LUFactorize(void)
    {
        assert(this->nrow_ > 0);
        assert(this->ncol_ > 0);
        assert(this->nnz_ > 0);
        assert(this->nrow_ == this->ncol_);

        for(int i = 0; i < this->nrow_ - 1; ++i)
        {
            for(int j = i + 1; j < this->nrow_; ++j)
            {
                this->mat_.val[DENSE_IND(j, i, this->nrow_, this->ncol_)]
                    /= this->mat_.val[DENSE_IND(i, i, this->nrow_, this->ncol_)];

                for(int k = i + 1; k < this->ncol_; ++k)
                {
                    this->mat_.val[DENSE_IND(j, k, this->nrow_, this->ncol_)]
                        -= this->mat_.val[DENSE_IND(j, i, this->nrow_, this->ncol_)]
                           * this->mat_.val[DENSE_IND(i, k, this->nrow_, this->ncol_)];
                }
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixDENSE<ValueType>::LUSolve(const BaseVector<ValueType>& in,
                                             BaseVector<ValueType>*       out) const
    {
        assert(in.GetSize() >= 0);
        assert(out->GetSize() >= 0);
        assert(in.GetSize() == this->nrow_);
        assert(out->GetSize() == this->ncol_);

        HostVector<ValueType>*       cast_out = dynamic_cast<HostVector<ValueType>*>(out);
        const HostVector<ValueType>* cast_in  = dynamic_cast<const HostVector<ValueType>*>(&in);

        assert(cast_out != NULL);

        // fill solution vector
        copy_h2h(this->nrow_, cast_in->vec_, cast_out->vec_);

        // forward sweeps
        for(int i = 0; i < this->nrow_ - 1; ++i)
        {
            for(int j = i + 1; j < this->nrow_; ++j)
            {
                cast_out->vec_[j] -= cast_out->vec_[i]
                                     * this->mat_.val[DENSE_IND(j, i, this->nrow_, this->ncol_)];
            }
        }

        // backward sweeps
        for(int i = this->nrow_ - 1; i >= 0; --i)
        {
            cast_out->vec_[i] /= this->mat_.val[DENSE_IND(i, i, this->nrow_, this->ncol_)];
            for(int j = 0; j < i; ++j)
            {
                cast_out->vec_[j] -= cast_out->vec_[i]
                                     * this->mat_.val[DENSE_IND(j, i, this->nrow_, this->ncol_)];
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixDENSE<ValueType>::ReplaceColumnVector(int idx, const BaseVector<ValueType>& vec)
    {
        assert(vec.GetSize() == this->nrow_);

        if(this->GetNnz() > 0)
        {
            const HostVector<ValueType>* cast_vec
                = dynamic_cast<const HostVector<ValueType>*>(&vec);
            assert(cast_vec != NULL);

            _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 0; i < this->nrow_; ++i)
            {
                this->mat_.val[DENSE_IND(i, idx, this->nrow_, this->ncol_)] = cast_vec->vec_[i];
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixDENSE<ValueType>::ReplaceRowVector(int idx, const BaseVector<ValueType>& vec)
    {
        assert(vec.GetSize() == this->ncol_);

        if(this->GetNnz() > 0)
        {
            const HostVector<ValueType>* cast_vec
                = dynamic_cast<const HostVector<ValueType>*>(&vec);
            assert(cast_vec != NULL);

            _set_omp_backend_threads(this->local_backend_, this->ncol_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 0; i < this->ncol_; ++i)
            {
                this->mat_.val[DENSE_IND(idx, i, this->nrow_, this->ncol_)] = cast_vec->vec_[i];
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixDENSE<ValueType>::ExtractColumnVector(int idx, BaseVector<ValueType>* vec) const
    {
        assert(vec != NULL);
        assert(vec->GetSize() == this->nrow_);

        if(this->GetNnz() > 0)
        {
            HostVector<ValueType>* cast_vec = dynamic_cast<HostVector<ValueType>*>(vec);
            assert(cast_vec != NULL);

            _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 0; i < this->nrow_; ++i)
            {
                cast_vec->vec_[i] = this->mat_.val[DENSE_IND(i, idx, this->nrow_, this->ncol_)];
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostMatrixDENSE<ValueType>::ExtractRowVector(int idx, BaseVector<ValueType>* vec) const
    {
        assert(vec != NULL);
        assert(vec->GetSize() == this->ncol_);

        if(this->GetNnz() > 0)
        {
            HostVector<ValueType>* cast_vec = dynamic_cast<HostVector<ValueType>*>(vec);
            assert(cast_vec != NULL);

            _set_omp_backend_threads(this->local_backend_, this->nrow_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int i = 0; i < this->nrow_; ++i)
            {
                cast_vec->vec_[i] = this->mat_.val[DENSE_IND(idx, i, this->nrow_, this->ncol_)];
            }
        }

        return true;
    }

    template class HostMatrixDENSE<double>;
    template class HostMatrixDENSE<float>;
#ifdef SUPPORT_COMPLEX
    template class HostMatrixDENSE<std::complex<double>>;
    template class HostMatrixDENSE<std::complex<float>>;
#endif

} // namespace rocalution

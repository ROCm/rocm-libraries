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

#include "host_matrix_hyb.hpp"
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
    HostMatrixHYB<ValueType>::HostMatrixHYB()
    {
        // no default constructors
        LOG_INFO("no default constructor");
        FATAL_ERROR(__FILE__, __LINE__);
    }
    // LCOV_EXCL_STOP

    template <typename ValueType>
    HostMatrixHYB<ValueType>::HostMatrixHYB(const Rocalution_Backend_Descriptor& local_backend)
    {
        log_debug(this, "HostMatrixHYB::HostMatrixHYB()", "constructor with local_backend");

        this->mat_.ELL.val     = NULL;
        this->mat_.ELL.col     = NULL;
        this->mat_.ELL.max_row = 0;

        this->mat_.COO.row = NULL;
        this->mat_.COO.col = NULL;
        this->mat_.COO.val = NULL;

        this->ell_nnz_ = 0;
        this->coo_nnz_ = 0;

        this->set_backend(local_backend);
    }

    template <typename ValueType>
    HostMatrixHYB<ValueType>::~HostMatrixHYB()
    {
        log_debug(this, "HostMatrixHYB::~HostMatrixHYB()", "destructor");

        this->Clear();
    }

    template <typename ValueType>
    void HostMatrixHYB<ValueType>::Info(void) const
    {
        LOG_INFO("HostMatrixHYB<ValueType>"
                 << " ELL nnz=" << this->ell_nnz_ << " ELL max row=" << this->mat_.ELL.max_row
                 << " COO nnz=" << this->coo_nnz_);
    }

    template <typename ValueType>
    void HostMatrixHYB<ValueType>::Clear()
    {
        free_host(&this->mat_.ELL.val);
        free_host(&this->mat_.ELL.col);
        free_host(&this->mat_.COO.row);
        free_host(&this->mat_.COO.col);
        free_host(&this->mat_.COO.val);

        this->mat_.ELL.max_row = 0;
        this->ell_nnz_         = 0;
        this->coo_nnz_         = 0;

        this->nrow_ = 0;
        this->ncol_ = 0;
        this->nnz_  = 0;
    }

    template <typename ValueType>
    void HostMatrixHYB<ValueType>::AllocateHYB(
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

        allocate_host(ell_nnz, &this->mat_.ELL.val);
        allocate_host(ell_nnz, &this->mat_.ELL.col);

        set_to_zero_host(ell_nnz, this->mat_.ELL.val);
        set_to_zero_host(ell_nnz, this->mat_.ELL.col);

        this->mat_.ELL.max_row = ell_max_row;
        this->ell_nnz_         = ell_nnz;
        this->nnz_ += ell_nnz;

        // COO
        allocate_host(coo_nnz, &this->mat_.COO.row);
        allocate_host(coo_nnz, &this->mat_.COO.col);
        allocate_host(coo_nnz, &this->mat_.COO.val);

        set_to_zero_host(coo_nnz, this->mat_.COO.row);
        set_to_zero_host(coo_nnz, this->mat_.COO.col);
        set_to_zero_host(coo_nnz, this->mat_.COO.val);

        this->coo_nnz_ = coo_nnz;
        this->nnz_ += coo_nnz;
    }

    template <typename ValueType>
    void HostMatrixHYB<ValueType>::CopyFrom(const BaseMatrix<ValueType>& mat)
    {
        // copy only in the same format
        assert(this->GetMatFormat() == mat.GetMatFormat());

        if(const HostMatrixHYB<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixHYB<ValueType>*>(&mat))
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

            copy_h2h(this->ell_nnz_, cast_mat->mat_.ELL.col, this->mat_.ELL.col);
            copy_h2h(this->ell_nnz_, cast_mat->mat_.ELL.val, this->mat_.ELL.val);
            copy_h2h(this->coo_nnz_, cast_mat->mat_.COO.row, this->mat_.COO.row);
            copy_h2h(this->coo_nnz_, cast_mat->mat_.COO.col, this->mat_.COO.col);
            copy_h2h(this->coo_nnz_, cast_mat->mat_.COO.val, this->mat_.COO.val);
        }
        else
        {
            // Host matrix knows only host matrices
            // -> dispatching
            mat.CopyTo(this);
        }
    }

    template <typename ValueType>
    void HostMatrixHYB<ValueType>::CopyTo(BaseMatrix<ValueType>* mat) const
    {
        mat->CopyFrom(*this);
    }

    template <typename ValueType>
    bool HostMatrixHYB<ValueType>::ConvertFrom(const BaseMatrix<ValueType>& mat)
    {
        this->Clear();

        // Empty matrix
        if(mat.GetNnz() == 0)
        {
            this->AllocateHYB(0, 0, 0, mat.GetM(), mat.GetN());

            return true;
        }

        if(const HostMatrixHYB<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixHYB<ValueType>*>(&mat))
        {
            this->CopyFrom(*cast_mat);
            return true;
        }

        if(const HostMatrixCSR<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixCSR<ValueType>*>(&mat))
        {
            this->Clear();

            int64_t nnz     = 0;
            int64_t coo_nnz = 0;
            int64_t ell_nnz = 0;

            if(csr_to_hyb(this->local_backend_.OpenMP_threads,
                          cast_mat->nnz_,
                          cast_mat->nrow_,
                          cast_mat->ncol_,
                          cast_mat->mat_,
                          &this->mat_,
                          &nnz,
                          &ell_nnz,
                          &coo_nnz)
               == true)
            {
                this->nrow_    = cast_mat->nrow_;
                this->ncol_    = cast_mat->ncol_;
                this->nnz_     = nnz;
                this->ell_nnz_ = ell_nnz;
                this->coo_nnz_ = coo_nnz;

                return true;
            }
        }

        return false;
    }

    template <typename ValueType>
    bool HostMatrixHYB<ValueType>::ReadFileRSIO(const std::string& filename)
    {
        int64_t nrow;
        int64_t ncol;
        int64_t nnz;
        int64_t coo_nnz;
        int64_t ell_nnz;
        int64_t ell_width;

        int*       coo_row = NULL;
        int*       coo_col = NULL;
        ValueType* coo_val = NULL;
        int*       ell_col = NULL;
        ValueType* ell_val = NULL;

        if(read_matrix_hyb_rocsparseio(nrow,
                                       ncol,
                                       nnz,
                                       coo_nnz,
                                       &coo_row,
                                       &coo_col,
                                       &coo_val,
                                       ell_nnz,
                                       ell_width,
                                       &ell_col,
                                       &ell_val,
                                       filename.c_str())
           != true)
        {
            return false;
        }

        this->Clear();

        this->nrow_ = nrow;
        this->ncol_ = ncol;
        this->nnz_  = nnz;

        // COO
        this->coo_nnz_     = coo_nnz;
        this->mat_.COO.row = coo_row;
        this->mat_.COO.col = coo_col;
        this->mat_.COO.val = coo_val;

        // ELL
        this->ell_nnz_         = ell_nnz;
        this->mat_.ELL.max_row = ell_width;
        this->mat_.ELL.col     = ell_col;
        this->mat_.ELL.val     = ell_val;

        return true;
    }

    template <typename ValueType>
    bool HostMatrixHYB<ValueType>::WriteFileRSIO(const std::string& filename) const
    {
        return write_matrix_hyb_rocsparseio(this->nrow_,
                                            this->ncol_,
                                            this->coo_nnz_,
                                            this->mat_.COO.row,
                                            this->mat_.COO.col,
                                            this->mat_.COO.val,
                                            this->mat_.ELL.max_row,
                                            this->mat_.ELL.col,
                                            this->mat_.ELL.val,
                                            filename.c_str());
    }

    template <typename ValueType>
    void HostMatrixHYB<ValueType>::Apply(const BaseVector<ValueType>& in,
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

            // ELL
            if(this->ell_nnz_ > 0)
            {
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for(int ai = 0; ai < this->nrow_; ++ai)
                {
                    cast_out->vec_[ai] = static_cast<ValueType>(0);

                    for(int n = 0; n < this->mat_.ELL.max_row; ++n)
                    {
                        int aj = ELL_IND(ai, n, this->nrow_, this->mat_.ELL.max_row);

                        if((this->mat_.ELL.col[aj] >= 0) && (this->mat_.ELL.col[aj] < this->ncol_))
                        {
                            cast_out->vec_[ai]
                                += this->mat_.ELL.val[aj] * cast_in->vec_[this->mat_.ELL.col[aj]];
                        }
                    }
                }
            }

            // COO
            if(this->coo_nnz_ > 0)
            {
                for(int64_t i = 0; i < this->coo_nnz_; ++i)
                {
                    cast_out->vec_[this->mat_.COO.row[i]]
                        += this->mat_.COO.val[i] * cast_in->vec_[this->mat_.COO.col[i]];
                }
            }
        }
    }

    template <typename ValueType>
    void HostMatrixHYB<ValueType>::ApplyAdd(const BaseVector<ValueType>& in,
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

            // ELL
            if(this->ell_nnz_ > 0)
            {
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for(int ai = 0; ai < this->nrow_; ++ai)
                {
                    for(int n = 0; n < this->mat_.ELL.max_row; ++n)
                    {
                        int aj = ELL_IND(ai, n, this->nrow_, this->mat_.ELL.max_row);

                        if((this->mat_.ELL.col[aj] >= 0) && (this->mat_.ELL.col[aj] < this->ncol_))
                        {
                            cast_out->vec_[ai] += scalar * this->mat_.ELL.val[aj]
                                                  * cast_in->vec_[this->mat_.ELL.col[aj]];
                        }
                    }
                }
            }

            // COO
            if(this->coo_nnz_ > 0)
            {
                for(int64_t i = 0; i < this->coo_nnz_; ++i)
                {
                    cast_out->vec_[this->mat_.COO.row[i]]
                        += scalar * this->mat_.COO.val[i] * cast_in->vec_[this->mat_.COO.col[i]];
                }
            }
        }
    }

    template class HostMatrixHYB<double>;
    template class HostMatrixHYB<float>;
#ifdef SUPPORT_COMPLEX
    template class HostMatrixHYB<std::complex<float>>;
    template class HostMatrixHYB<std::complex<double>>;
#endif

} // namespace rocalution

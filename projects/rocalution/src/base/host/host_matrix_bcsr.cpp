/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "host_matrix_bcsr.hpp"
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
    HostMatrixBCSR<ValueType>::HostMatrixBCSR()
    {
        // no default constructors
        LOG_INFO("no default constructor");
        FATAL_ERROR(__FILE__, __LINE__);
    }
    // LCOV_EXCL_STOP

    template <typename ValueType>
    HostMatrixBCSR<ValueType>::HostMatrixBCSR(const Rocalution_Backend_Descriptor& local_backend,
                                              int                                  blockdim)
    {
        log_debug(this, "HostMatrixBCSR::HostMatrixBCSR()", "constructor with local_backend");

        this->mat_.row_offset = NULL;
        this->mat_.col        = NULL;
        this->mat_.val        = NULL;
        this->mat_.blockdim   = blockdim;

        this->set_backend(local_backend);
    }

    template <typename ValueType>
    HostMatrixBCSR<ValueType>::~HostMatrixBCSR()
    {
        log_debug(this, "HostMatrixBCSR::~HostMatrixBCSR()", "destructor");

        this->Clear();
    }

    template <typename ValueType>
    void HostMatrixBCSR<ValueType>::Info(void) const
    {
        LOG_INFO("HostMatrixBCSR<ValueType>");
    }

    template <typename ValueType>
    void HostMatrixBCSR<ValueType>::Clear()
    {
        free_host(&this->mat_.row_offset);
        free_host(&this->mat_.col);
        free_host(&this->mat_.val);

        this->nrow_ = 0;
        this->ncol_ = 0;
        this->nnz_  = 0;
    }

    template <typename ValueType>
    void HostMatrixBCSR<ValueType>::AllocateBCSR(int64_t nnzb, int nrowb, int ncolb, int blockdim)
    {
        assert(nnzb >= 0);
        assert(ncolb >= 0);
        assert(nrowb >= 0);
        assert(blockdim > 1);

        this->Clear();

        allocate_host(nrowb + 1, &this->mat_.row_offset);
        allocate_host(nnzb, &this->mat_.col);
        allocate_host(nnzb * blockdim * blockdim, &this->mat_.val);

        set_to_zero_host(nrowb + 1, this->mat_.row_offset);
        set_to_zero_host(nnzb, this->mat_.col);
        set_to_zero_host(nnzb * blockdim * blockdim, this->mat_.val);

        this->nrow_ = nrowb * blockdim;
        this->ncol_ = ncolb * blockdim;
        this->nnz_  = nnzb * blockdim * blockdim;

        this->mat_.nrowb = nrowb;
        this->mat_.ncolb = ncolb;
        this->mat_.nnzb  = nnzb;

        this->mat_.blockdim = blockdim;
    }

    template <typename ValueType>
    void HostMatrixBCSR<ValueType>::SetDataPtrBCSR(int**       row_offset,
                                                   int**       col,
                                                   ValueType** val,
                                                   int64_t     nnzb,
                                                   int         nrowb,
                                                   int         ncolb,
                                                   int         blockdim)
    {
        assert(nnzb >= 0);
        assert(nrowb >= 0);
        assert(ncolb >= 0);
        assert(blockdim > 1);
        assert(*row_offset != NULL);

        if(nnzb > 0)
        {
            assert(*col != NULL);
            assert(*val != NULL);
        }

        this->Clear();

        this->nrow_ = nrowb * blockdim;
        this->ncol_ = ncolb * blockdim;
        this->nnz_  = nnzb * blockdim * blockdim;

        this->mat_.nrowb    = nrowb;
        this->mat_.ncolb    = ncolb;
        this->mat_.nnzb     = nnzb;
        this->mat_.blockdim = blockdim;

        this->mat_.row_offset = *row_offset;
        this->mat_.col        = *col;
        this->mat_.val        = *val;
    }

    template <typename ValueType>
    void HostMatrixBCSR<ValueType>::LeaveDataPtrBCSR(int**       row_offset,
                                                     int**       col,
                                                     ValueType** val,
                                                     int&        blockdim)
    {
        assert(this->nrow_ >= 0);
        assert(this->ncol_ >= 0);
        assert(this->nnz_ >= 0);
        assert(this->mat_.blockdim > 1);

        *row_offset = this->mat_.row_offset;
        *col        = this->mat_.col;
        *val        = this->mat_.val;

        this->mat_.row_offset = NULL;
        this->mat_.col        = NULL;
        this->mat_.val        = NULL;

        blockdim = this->mat_.blockdim;

        this->mat_.blockdim = 0;

        this->nrow_ = 0;
        this->ncol_ = 0;
        this->nnz_  = 0;
    }

    template <typename ValueType>
    void HostMatrixBCSR<ValueType>::CopyFrom(const BaseMatrix<ValueType>& mat)
    {
        // copy only in the same format
        assert(this->GetMatFormat() == mat.GetMatFormat());

        if(const HostMatrixBCSR<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixBCSR<ValueType>*>(&mat))
        {
            if(this->nnz_ == 0)
            {
                this->AllocateBCSR(cast_mat->mat_.nnzb,
                                   cast_mat->mat_.nrowb,
                                   cast_mat->mat_.ncolb,
                                   cast_mat->mat_.blockdim);
            }

            assert(this->nnz_ == cast_mat->nnz_);
            assert(this->nrow_ == cast_mat->nrow_);
            assert(this->ncol_ == cast_mat->ncol_);
            assert(this->mat_.nrowb == cast_mat->mat_.nrowb);
            assert(this->mat_.ncolb == cast_mat->mat_.ncolb);
            assert(this->mat_.nnzb == cast_mat->mat_.nnzb);
            assert(this->mat_.blockdim == cast_mat->mat_.blockdim);

            // Copy only if initialized
            if(cast_mat->mat_.row_offset != NULL)
            {
                copy_h2h(this->mat_.nrowb + 1, cast_mat->mat_.row_offset, this->mat_.row_offset);
            }

            copy_h2h(this->mat_.nnzb, cast_mat->mat_.col, this->mat_.col);
            copy_h2h(this->mat_.nnzb * this->mat_.blockdim * this->mat_.blockdim,
                     cast_mat->mat_.val,
                     this->mat_.val);
        }
        else
        {
            // Host matrix knows only host matrices
            // -> dispatching
            mat.CopyTo(this);
        }
    }

    template <typename ValueType>
    void HostMatrixBCSR<ValueType>::CopyTo(BaseMatrix<ValueType>* mat) const
    {
        mat->CopyFrom(*this);
    }

    template <typename ValueType>
    bool HostMatrixBCSR<ValueType>::ConvertFrom(const BaseMatrix<ValueType>& mat)
    {
        this->Clear();

        // Empty matrix
        if(mat.GetNnz() == 0)
        {
            int mb = (mat.GetM() + 1) / 2;
            int nb = (mat.GetN() + 1) / 2;

            this->AllocateBCSR(0, mb, nb, 2);

            return true;
        }

        if(const HostMatrixBCSR<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixBCSR<ValueType>*>(&mat))
        {
            this->CopyFrom(*cast_mat);
            return true;
        }

        if(const HostMatrixCSR<ValueType>* cast_mat
           = dynamic_cast<const HostMatrixCSR<ValueType>*>(&mat))
        {
            this->Clear();

            if(csr_to_bcsr(this->local_backend_.OpenMP_threads,
                           cast_mat->nnz_,
                           cast_mat->nrow_,
                           cast_mat->ncol_,
                           cast_mat->mat_,
                           &this->mat_)
               == true)
            {
                this->nrow_ = this->mat_.nrowb * this->mat_.blockdim;
                this->ncol_ = this->mat_.ncolb * this->mat_.blockdim;
                this->nnz_  = this->mat_.nnzb * this->mat_.blockdim * this->mat_.blockdim;

                return true;
            }
        }

        return false;
    }

    template <typename ValueType>
    bool HostMatrixBCSR<ValueType>::ReadFileRSIO(const std::string& filename)
    {
        int64_t nrowb;
        int64_t ncolb;
        int64_t nnzb;
        int64_t blockdim;

        PtrType*   ptr = NULL;
        int*       col = NULL;
        ValueType* val = NULL;

        if(read_matrix_bcsr_rocsparseio(
               nrowb, ncolb, nnzb, blockdim, &ptr, &col, &val, filename.c_str())
           != true)
        {
            return false;
        }

        // Number of rows and columns are expected to be within 32 bits locally
        assert(nrowb <= std::numeric_limits<int>::max());
        assert(ncolb <= std::numeric_limits<int>::max());
        assert(blockdim <= std::numeric_limits<int>::max());

        this->Clear();
        this->SetDataPtrBCSR(
            &ptr, &col, &val, nnzb, static_cast<int>(nrowb), static_cast<int>(ncolb), blockdim);

        return true;
    }

    template <typename ValueType>
    bool HostMatrixBCSR<ValueType>::WriteFileRSIO(const std::string& filename) const
    {
        return write_matrix_bcsr_rocsparseio(this->mat_.nrowb,
                                             this->mat_.ncolb,
                                             this->mat_.nnzb,
                                             this->mat_.blockdim,
                                             this->mat_.row_offset,
                                             this->mat_.col,
                                             this->mat_.val,
                                             filename.c_str());
    }

    template <typename ValueType>
    void HostMatrixBCSR<ValueType>::Apply(const BaseVector<ValueType>& in,
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

            _set_omp_backend_threads(this->local_backend_, this->mat_.nrowb);

            int bsrdim = this->mat_.blockdim;

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int ai = 0; ai < this->mat_.nrowb; ++ai)
            {
                for(int bi = 0; bi < bsrdim; ++bi)
                {
                    int row_begin = this->mat_.row_offset[ai];
                    int row_end   = this->mat_.row_offset[ai + 1];

                    ValueType sum = static_cast<ValueType>(0);

                    for(int aj = row_begin; aj < row_end; ++aj)
                    {
                        int col = this->mat_.col[aj];

                        for(int bj = 0; bj < bsrdim; ++bj)
                        {
                            sum += this->mat_.val[BCSR_IND(aj, bi, bj, bsrdim)]
                                   * cast_in->vec_[bsrdim * col + bj];
                        }
                    }

                    cast_out->vec_[ai * bsrdim + bi] = sum;
                }
            }
        }
    }

    template <typename ValueType>
    void HostMatrixBCSR<ValueType>::ApplyAdd(const BaseVector<ValueType>& in,
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

            int bsrdim = this->mat_.blockdim;
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int ai = 0; ai < this->mat_.nrowb; ++ai)
            {
                for(int bi = 0; bi < bsrdim; ++bi)
                {
                    int row_begin = this->mat_.row_offset[ai];
                    int row_end   = this->mat_.row_offset[ai + 1];

                    ValueType sum = static_cast<ValueType>(0);

                    for(int aj = row_begin; aj < row_end; ++aj)
                    {
                        int col = this->mat_.col[aj];

                        for(int bj = 0; bj < bsrdim; ++bj)
                        {
                            sum += this->mat_.val[BCSR_IND(aj, bi, bj, bsrdim)]
                                   * cast_in->vec_[bsrdim * col + bj];
                        }
                    }

                    cast_out->vec_[ai * bsrdim + bi] += scalar * sum;
                }
            }
        }
    }

    template class HostMatrixBCSR<double>;
    template class HostMatrixBCSR<float>;
#ifdef SUPPORT_COMPLEX
    template class HostMatrixBCSR<std::complex<double>>;
    template class HostMatrixBCSR<std::complex<float>>;
#endif

} // namespace rocalution

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

#include "preconditioner_multicolored_gs.hpp"
#include "../../utils/def.hpp"
#include "../solver.hpp"
#include "preconditioner.hpp"
#include "preconditioner_multicolored.hpp"

#include "../../base/local_matrix.hpp"

#include "../../base/local_vector.hpp"

#include "../../utils/log.hpp"

#include <complex>

namespace rocalution
{

    template <class OperatorType, class VectorType, typename ValueType>
    MultiColoredSGS<OperatorType, VectorType, ValueType>::MultiColoredSGS()
    {
        log_debug(this, "MultiColoredSGS::MultiColoredSGS()", "default constructor");

        this->omega_ = static_cast<ValueType>(1);
    }

    template <class OperatorType, class VectorType, typename ValueType>
    MultiColoredSGS<OperatorType, VectorType, ValueType>::~MultiColoredSGS()
    {
        log_debug(this, "MultiColoredSGS::~MultiColoredSGS()", "destructor");

        this->Clear();
    }

    // TODO
    // not optimal implementation - scale the diagonal vectors in the building phase
    template <class OperatorType, class VectorType, typename ValueType>
    void MultiColoredSGS<OperatorType, VectorType, ValueType>::SetRelaxation(ValueType omega)
    {
        log_debug(this, "MultiColoredSGS::SetRelaxation()", omega);

        this->omega_ = omega;
    }

    // LCOV_EXCL_START
    template <class OperatorType, class VectorType, typename ValueType>
    void MultiColoredSGS<OperatorType, VectorType, ValueType>::Print(void) const
    {
        LOG_INFO("Multicolored Symmetric Gauss-Seidel (SGS) preconditioner");

        if(this->build_ == true)
        {
            LOG_INFO("number of colors = " << this->num_blocks_);
        }
    }
    // LCOV_EXCL_STOP

    template <class OperatorType, class VectorType, typename ValueType>
    void MultiColoredSGS<OperatorType, VectorType, ValueType>::ReBuildNumeric(void)
    {
        log_debug(this, "MultiColoredSGS::ReBuildNumeric()", this->build_);

        if(this->preconditioner_ != NULL)
        {
            this->preconditioner_->Clear();
            delete this->preconditioner_;
        }

        for(int i = 0; i < this->num_blocks_; ++i)
        {
            delete this->x_block_[i];
            delete this->diag_block_[i];
            delete this->diag_solver_[i];

            for(int j = 0; j < this->num_blocks_; ++j)
            {
                delete this->preconditioner_block_[i][j];
            }

            delete[] this->preconditioner_block_[i];
        }

        delete[] this->preconditioner_block_;
        delete[] this->x_block_;
        delete[] this->diag_block_;
        delete[] this->diag_solver_;

        this->preconditioner_ = new OperatorType;
        this->preconditioner_->CloneFrom(*this->op_);

        this->Permute_();
        this->Factorize_();
        this->Decompose_();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void MultiColoredSGS<OperatorType, VectorType, ValueType>::PostAnalyse_(void)
    {
        log_debug(this, "MultiColoredSGS::PostAnalyse_()", this->build_);

        assert(this->build_ == true);
        this->preconditioner_->LAnalyse(false);
        this->preconditioner_->UAnalyse(false);
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void MultiColoredSGS<OperatorType, VectorType, ValueType>::SolveL_(void)
    {
        log_debug(this, "MultiColoredSGS::SolveL_()");

        assert(this->build_ == true);

        for(int i = 0; i < this->num_blocks_; ++i)
        {
            for(int j = 0; j < i; ++j)
            {
                if(this->preconditioner_block_[i][j]->GetNnz() > 0)
                {
                    this->preconditioner_block_[i][j]->ApplyAdd(
                        *this->x_block_[j], static_cast<ValueType>(-1), this->x_block_[i]);
                }
            }

            this->diag_solver_[i]->Solve(*this->x_block_[i], this->x_block_[i]);

            // SSOR
            if(this->omega_ != static_cast<ValueType>(1))
            {
                this->x_block_[i]->Scale(static_cast<ValueType>(1) / this->omega_);
            }
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void MultiColoredSGS<OperatorType, VectorType, ValueType>::SolveD_(void)
    {
        log_debug(this, "MultiColoredSGS::SolveD_()");

        assert(this->build_ == true);

        for(int i = 0; i < this->num_blocks_; ++i)
        {
            this->x_block_[i]->PointWiseMult(*this->diag_block_[i]);

            // SSOR
            if(this->omega_ != static_cast<ValueType>(1))
            {
                this->x_block_[i]->Scale(this->omega_ / (static_cast<ValueType>(2) - this->omega_));
            }
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void MultiColoredSGS<OperatorType, VectorType, ValueType>::SolveR_(void)
    {
        log_debug(this, "MultiColoredSGS::SolveR_()");

        assert(this->build_ == true);

        for(int i = this->num_blocks_ - 1; i >= 0; --i)
        {
            for(int j = this->num_blocks_ - 1; j > i; --j)
            {
                if(this->preconditioner_block_[i][j]->GetNnz() > 0)
                {
                    this->preconditioner_block_[i][j]->ApplyAdd(
                        *this->x_block_[j], static_cast<ValueType>(-1), this->x_block_[i]);
                }
            }

            this->diag_solver_[i]->Solve(*this->x_block_[i], this->x_block_[i]);

            // SSOR
            if(this->omega_ != static_cast<ValueType>(1))
            {
                this->x_block_[i]->Scale(static_cast<ValueType>(1) / this->omega_);
            }
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void MultiColoredSGS<OperatorType, VectorType, ValueType>::Solve_(const VectorType& rhs,
                                                                      VectorType*       x)
    {
        log_debug(this, "MultiColoredSGS::Solve_()", (const void*&)rhs, x);

        this->x_.CopyFromPermute(rhs, this->permutation_);

        this->preconditioner_->LSolve(this->x_, x);

        x->PointWiseMult(this->diag_);

        this->preconditioner_->USolve(*x, &this->x_);

        x->CopyFromPermuteBackward(this->x_, this->permutation_);
    }

    template <class OperatorType, class VectorType, typename ValueType>
    MultiColoredGS<OperatorType, VectorType, ValueType>::MultiColoredGS()
    {
    }

    template <class OperatorType, class VectorType, typename ValueType>
    MultiColoredGS<OperatorType, VectorType, ValueType>::~MultiColoredGS()
    {
        this->Clear();
    }

    // LCOV_EXCL_START
    template <class OperatorType, class VectorType, typename ValueType>
    void MultiColoredGS<OperatorType, VectorType, ValueType>::Print(void) const
    {
        LOG_INFO("Multicolored Gauss-Seidel (GS) preconditioner");

        if(this->build_ == true)
        {
            LOG_INFO("number of colors = " << this->num_blocks_);
        }
    }
    // LCOV_EXCL_STOP

    template <class OperatorType, class VectorType, typename ValueType>
    void MultiColoredGS<OperatorType, VectorType, ValueType>::PostAnalyse_(void)
    {
        assert(this->build_ == true);
        this->preconditioner_->UAnalyse(false);
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void MultiColoredGS<OperatorType, VectorType, ValueType>::SolveL_(void)
    {
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void MultiColoredGS<OperatorType, VectorType, ValueType>::SolveD_(void)
    {
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void MultiColoredGS<OperatorType, VectorType, ValueType>::SolveR_(void)
    {
        assert(this->build_ == true);

        for(int i = this->num_blocks_ - 1; i >= 0; --i)
        {
            for(int j = this->num_blocks_ - 1; j > i; --j)
            {
                if(this->preconditioner_block_[i][j]->GetNnz() > 0)
                {
                    this->preconditioner_block_[i][j]->ApplyAdd(
                        *this->x_block_[j], static_cast<ValueType>(-1), this->x_block_[i]);
                }
            }

            this->diag_solver_[i]->Solve(*this->x_block_[i], this->x_block_[i]);

            // SSOR
            if(this->omega_ != static_cast<ValueType>(1))
            {
                this->x_block_[i]->Scale(static_cast<ValueType>(1) / this->omega_);
            }
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void MultiColoredGS<OperatorType, VectorType, ValueType>::Solve_(const VectorType& rhs,
                                                                     VectorType*       x)
    {
        LOG_INFO("No implemented yet");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template class MultiColoredSGS<LocalMatrix<double>, LocalVector<double>, double>;
    template class MultiColoredSGS<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class MultiColoredSGS<LocalMatrix<std::complex<double>>,
                                   LocalVector<std::complex<double>>,
                                   std::complex<double>>;
    template class MultiColoredSGS<LocalMatrix<std::complex<float>>,
                                   LocalVector<std::complex<float>>,
                                   std::complex<float>>;
#endif

    template class MultiColoredGS<LocalMatrix<double>, LocalVector<double>, double>;
    template class MultiColoredGS<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class MultiColoredGS<LocalMatrix<std::complex<double>>,
                                  LocalVector<std::complex<double>>,
                                  std::complex<double>>;
    template class MultiColoredGS<LocalMatrix<std::complex<float>>,
                                  LocalVector<std::complex<float>>,
                                  std::complex<float>>;
#endif

} // namespace rocalution

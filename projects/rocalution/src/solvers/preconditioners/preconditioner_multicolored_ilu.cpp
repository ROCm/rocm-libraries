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

#include "preconditioner_multicolored_ilu.hpp"
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
    MultiColoredILU<OperatorType, VectorType, ValueType>::MultiColoredILU()
    {
        log_debug(this, "MultiColoredILU::MultiColoredILU()", "default constructor");

        this->q_     = 1;
        this->p_     = 0;
        this->level_ = true;
        this->nnz_   = 0;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    MultiColoredILU<OperatorType, VectorType, ValueType>::~MultiColoredILU()
    {
        log_debug(this, "MultiColoredILU::~MultiColoredILU()", "destructor");

        this->Clear();
    }

    // LCOV_EXCL_START
    template <class OperatorType, class VectorType, typename ValueType>
    void MultiColoredILU<OperatorType, VectorType, ValueType>::Print(void) const
    {
        LOG_INFO("Multicolored ILU preconditioner (power(q)-pattern method), ILU("
                 << this->p_ << "," << this->q_ << ")");

        if(this->build_ == true)
        {
            LOG_INFO("number of colors = " << this->num_blocks_ << "; ILU nnz = " << this->nnz_);
        }
    }
    // LCOV_EXCL_STOP

    template <class OperatorType, class VectorType, typename ValueType>
    void MultiColoredILU<OperatorType, VectorType, ValueType>::Set(int p)
    {
        log_debug(this, "MultiColoredILU::Set()", p);

        assert(this->build_ == false);
        assert(p >= 0);

        this->p_ = p;
        this->q_ = p + 1;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void MultiColoredILU<OperatorType, VectorType, ValueType>::Set(int p, int q, bool level)
    {
        log_debug(this, "MultiColoredILU::Set()", p, q, level);

        assert(this->build_ == false);
        assert(p >= 0);
        assert(q >= 1);

        this->p_     = p;
        this->q_     = q;
        this->level_ = level;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void MultiColoredILU<OperatorType, VectorType, ValueType>::Build_Analyser_(void)
    {
        log_debug(this, "MultiColoredILU::Build_Analyser_()", this->build_);

        assert(this->op_ != NULL);

        if(this->q_ > 1)
        {
            this->analyzer_op_ = new OperatorType;
            this->analyzer_op_->CloneFrom(*this->op_);

            this->analyzer_op_->SymbolicPower(this->q_);
        }
        else
        {
            this->analyzer_op_ = NULL;
        }

        this->preconditioner_ = new OperatorType;
        this->preconditioner_->CloneFrom(*this->op_);

        this->permutation_.CloneBackend(*this->op_);
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void MultiColoredILU<OperatorType, VectorType, ValueType>::PostAnalyse_(void)
    {
        log_debug(this, "MultiColoredILU::PostAnalyse_()", this->build_);

        assert(this->build_ == true);
        this->preconditioner_->LUAnalyse();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void MultiColoredILU<OperatorType, VectorType, ValueType>::Factorize_(void)
    {
        log_debug(this, "MultiColoredILU::Factorize_()", this->build_);

        this->preconditioner_->ILUpFactorize(this->p_, this->level_);

        this->nnz_ = this->preconditioner_->GetLocalNnz();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void MultiColoredILU<OperatorType, VectorType, ValueType>::ReBuildNumeric(void)
    {
        log_debug(this, "MultiColoredILU::ReBuildNumeric()", this->build_);

        if(this->decomp_ == false)
        {
            this->preconditioner_->PermuteBackward(this->permutation_);

            this->preconditioner_->Zeros();
            this->preconditioner_->MatrixAdd(
                *this->op_, static_cast<ValueType>(0), static_cast<ValueType>(1), false);

            this->preconditioner_->Permute(this->permutation_);

            this->preconditioner_->ILU0Factorize();
            this->preconditioner_->LUAnalyse();
        }
        else
        {
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
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void MultiColoredILU<OperatorType, VectorType, ValueType>::SolveL_(void)
    {
        log_debug(this, "MultiColoredILU::SolveL_()");

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
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void MultiColoredILU<OperatorType, VectorType, ValueType>::SolveD_(void)
    {
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void MultiColoredILU<OperatorType, VectorType, ValueType>::SolveR_(void)
    {
        log_debug(this, "MultiColoredILU::SolveR_()");

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
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void MultiColoredILU<OperatorType, VectorType, ValueType>::Solve_(const VectorType& rhs,
                                                                      VectorType*       x)
    {
        log_debug(this, "MultiColoredILU::Solve_()");

        x->CopyFromPermute(rhs, this->permutation_);

        this->preconditioner_->LUSolve(*x, &this->x_);

        x->CopyFromPermuteBackward(this->x_, this->permutation_);
    }

    template class MultiColoredILU<LocalMatrix<double>, LocalVector<double>, double>;
    template class MultiColoredILU<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class MultiColoredILU<LocalMatrix<std::complex<double>>,
                                   LocalVector<std::complex<double>>,
                                   std::complex<double>>;
    template class MultiColoredILU<LocalMatrix<std::complex<float>>,
                                   LocalVector<std::complex<float>>,
                                   std::complex<float>>;
#endif

} // namespace rocalution

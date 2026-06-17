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

#include "pairwise_amg.hpp"
#include "../../utils/def.hpp"

#include "../../base/global_matrix.hpp"
#include "../../base/global_vector.hpp"
#include "../../base/local_matrix.hpp"
#include "../../base/local_vector.hpp"

#include "../preconditioners/preconditioner.hpp"

#include "../../utils/allocate_free.hpp"
#include "../../utils/log.hpp"
#include "../../utils/math_functions.hpp"

#include <complex>
#include <list>

namespace rocalution
{

    template <class OperatorType, class VectorType, typename ValueType>
    PairwiseAMG<OperatorType, VectorType, ValueType>::PairwiseAMG()
    {
        log_debug(this, "PairwiseAMG::PairwiseAMG()", "default constructor");

        this->beta_        = static_cast<ValueType>(0.25);
        this->coarse_size_ = 300;

        // target coarsening factor
        this->coarsening_factor_ = 4.0;

        // number of pre- and post-smoothing steps
        this->iter_pre_smooth_  = 1;
        this->iter_post_smooth_ = 2;

        // set K-cycle to default
        this->cycle_ = 2;

        // disable scaling
        this->scaling_ = false;

        // disable ordering by default
        this->aggregation_ordering_ = NoOrdering;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    PairwiseAMG<OperatorType, VectorType, ValueType>::~PairwiseAMG()
    {
        log_debug(this, "PairwiseAMG::PairwiseAMG()", "destructor");

        this->Clear();
    }

    // LCOV_EXCL_START
    template <class OperatorType, class VectorType, typename ValueType>
    void PairwiseAMG<OperatorType, VectorType, ValueType>::Print(void) const
    {
        LOG_INFO("AMG solver");
        LOG_INFO("AMG number of levels " << this->levels_);
        LOG_INFO("AMG using pairwise aggregation");
        LOG_INFO("AMG coarsest operator size = " << this->op_level_[this->levels_ - 2]->GetM());
        int64_t global_nnz = this->op_level_[this->levels_ - 2]->GetNnz();
        LOG_INFO("AMG coarsest level nnz = " << global_nnz);
        LOG_INFO("AMG with smoother:");
        this->smoother_level_[0]->Print();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void PairwiseAMG<OperatorType, VectorType, ValueType>::PrintStart_(void) const
    {
        assert(this->levels_ > 0);

        LOG_INFO("AMG solver starts");
        LOG_INFO("AMG number of levels " << this->levels_);
        LOG_INFO("AMG using pairwise aggregation");
        LOG_INFO("AMG coarsest operator size = " << this->op_level_[this->levels_ - 2]->GetM());
        int64_t global_nnz = this->op_level_[this->levels_ - 2]->GetNnz();
        LOG_INFO("AMG coarsest level nnz = " << global_nnz);
        LOG_INFO("AMG with smoother:");
        this->smoother_level_[0]->Print();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void PairwiseAMG<OperatorType, VectorType, ValueType>::PrintEnd_(void) const
    {
        LOG_INFO("AMG ends");
    }
    // LCOV_EXCL_STOP

    template <class OperatorType, class VectorType, typename ValueType>
    void PairwiseAMG<OperatorType, VectorType, ValueType>::SetBeta(ValueType beta)
    {
        log_debug(this, "PairwiseAMG::SetBeta()", beta);

        assert(beta > static_cast<ValueType>(0));
        assert(beta < static_cast<ValueType>(1));

        this->beta_ = beta;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void PairwiseAMG<OperatorType, VectorType, ValueType>::SetCoarseningFactor(double factor)
    {
        log_debug(this, "PairwiseAMG::SetCoarseningFactor()", factor);

        assert(factor > 0.0);
        assert(factor < 20.0);

        this->coarsening_factor_ = factor;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void PairwiseAMG<OperatorType, VectorType, ValueType>::SetOrdering(unsigned int ordering)
    {
        log_debug(this, "PairwiseAMG::SetOrdering()", ordering);

        assert(ordering <= 5);

        this->aggregation_ordering_ = ordering;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void PairwiseAMG<OperatorType, VectorType, ValueType>::ReBuildNumeric(void)
    {
        log_debug(this, "PairwiseAMG::ReBuildNumeric()", " #*# begin");

        assert(this->levels_ > 1);
        assert(this->build_ == true);
        assert(this->op_ != NULL);

        this->op_level_[0]->Clear();
        this->op_level_[0]->CloneBackend(*this->op_);
        this->op_level_[0]->ConvertToCSR();

        this->trans_level_[0]->CloneBackend(*this->op_);

        this->op_->CoarsenOperator(this->op_level_[0],
                                   this->dim_level_[0],
                                   this->dim_level_[0],
                                   *this->trans_level_[0],
                                   this->Gsize_level_[0],
                                   this->rG_level_[0],
                                   this->rGsize_level_[0]);

        for(int i = 1; i < this->levels_ - 1; ++i)
        {
            this->op_level_[i]->Clear();
            this->op_level_[i]->ConvertToCSR();

            this->trans_level_[i]->CloneBackend(*this->op_level_[i]);

            if(i == this->levels_ - this->host_level_ - 1)
            {
                this->op_level_[i - 1]->MoveToHost();
            }

            this->op_level_[i - 1]->CoarsenOperator(this->op_level_[i],
                                                    this->dim_level_[i],
                                                    this->dim_level_[i],
                                                    *this->trans_level_[i],
                                                    this->Gsize_level_[i],
                                                    this->rG_level_[i],
                                                    this->rGsize_level_[i]);

            if(i == this->levels_ - this->host_level_ - 1)
            {
                this->op_level_[i - 1]->CloneBackend(*this->restrict_op_level_[i - 1]);
            }
        }

        this->smoother_level_[0]->ResetOperator(*this->op_);
        this->smoother_level_[0]->ReBuildNumeric();
        this->smoother_level_[0]->Verbose(0);

        for(int i = 1; i < this->levels_ - 1; ++i)
        {
            this->smoother_level_[i]->ResetOperator(*this->op_level_[i - 1]);
            this->smoother_level_[i]->ReBuildNumeric();
            this->smoother_level_[i]->Verbose(0);
        }

        this->solver_coarse_->ResetOperator(*this->op_level_[this->levels_ - 2]);
        this->solver_coarse_->ReBuildNumeric();
        this->solver_coarse_->Verbose(0);

        // Convert operator to op_format
        if(this->op_format_ != CSR)
        {
            for(int i = 0; i < this->levels_ - 1; ++i)
            {
                this->op_level_[i]->ConvertTo(this->op_format_, this->op_blockdim_);
            }
        }

        log_debug(this, "PairwiseAMG::ReBuildNumeric()", " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void PairwiseAMG<OperatorType, VectorType, ValueType>::ClearLocal(void)
    {
        log_debug(this, "PairwiseAMG::ClearLocal()", this->build_);

        if(this->build_ == true)
        {
            for(int i = 0; i < this->levels_ - 1; ++i)
            {
                free_host(&this->rG_level_[i]);
            }

            this->dim_level_.clear();
            this->Gsize_level_.clear();
            this->rGsize_level_.clear();
            this->rG_level_.clear();
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    bool PairwiseAMG<OperatorType, VectorType, ValueType>::Aggregate_(const OperatorType& op,
                                                                      OperatorType*       pro,
                                                                      OperatorType*       res,
                                                                      OperatorType*       coarse,
                                                                      LocalVector<int>*   trans)
    {
        log_debug(this, "PairwiseAMG::Aggregate_()", (const void*&)op, pro, res, coarse, trans);

        assert(pro != NULL);
        assert(res != NULL);
        assert(coarse != NULL);
        assert(trans != NULL);

        int  nc;
        int* rG = NULL;
        int  Gsize;
        int  rGsize;

        // Allocate transfer mapping for current level
        trans->Allocate("transfer map", op.GetLocalM());

        op.InitialPairwiseAggregation(
            this->beta_, nc, trans, Gsize, &rG, rGsize, this->aggregation_ordering_);
        op.CoarsenOperator(coarse, nc, nc, *trans, Gsize, rG, rGsize);

        int cycle = 0;

        while(static_cast<double>(op.GetM()) / static_cast<double>(coarse->GetM())
              < this->coarsening_factor_)
        {
            coarse->FurtherPairwiseAggregation(
                this->beta_, nc, trans, Gsize, &rG, rGsize, this->aggregation_ordering_);
            op.CoarsenOperator(coarse, nc, nc, *trans, Gsize, rG, rGsize);

            ++cycle;

            if(cycle > 8)
            {
                LOG_VERBOSE_INFO(2,
                                 "*** warning: PairwiseAMG::Build() Coarsening cannot obtain "
                                 "satisfying coarsening factor");
            }
        }

        res->CreateFromMap(*trans, op.GetLocalM(), nc, pro);

        // Store data for possible coarse operator rebuild
        this->dim_level_.push_back(nc);
        this->Gsize_level_.push_back(Gsize);
        this->rGsize_level_.push_back(rGsize);
        this->rG_level_.push_back(rG);

        return true;
    }

    template class PairwiseAMG<LocalMatrix<float>, LocalVector<float>, float>;
    template class PairwiseAMG<GlobalMatrix<float>, GlobalVector<float>, float>;
    template class PairwiseAMG<LocalMatrix<double>, LocalVector<double>, double>;
    template class PairwiseAMG<GlobalMatrix<double>, GlobalVector<double>, double>;
#ifdef SUPPORT_COMPLEX
    template class PairwiseAMG<LocalMatrix<std::complex<float>>,
                               LocalVector<std::complex<float>>,
                               std::complex<float>>;
    template class PairwiseAMG<GlobalMatrix<std::complex<float>>,
                               GlobalVector<std::complex<float>>,
                               std::complex<float>>;
    template class PairwiseAMG<LocalMatrix<std::complex<double>>,
                               LocalVector<std::complex<double>>,
                               std::complex<double>>;
    template class PairwiseAMG<GlobalMatrix<std::complex<double>>,
                               GlobalVector<std::complex<double>>,
                               std::complex<double>>;
#endif

} // namespace rocalution

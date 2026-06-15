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

#include "unsmoothed_amg.hpp"
#include "../../utils/def.hpp"

#include "../../base/global_matrix.hpp"
#include "../../base/global_vector.hpp"
#include "../../base/local_matrix.hpp"
#include "../../base/local_vector.hpp"

#include "../../solvers/preconditioners/preconditioner.hpp"

#include "../../utils/log.hpp"
#include "../../utils/math_functions.hpp"

namespace rocalution
{

    template <class OperatorType, class VectorType, typename ValueType>
    UAAMG<OperatorType, VectorType, ValueType>::UAAMG()
    {
        log_debug(this, "UAAMG::UAAMG()", "default constructor");

        // parameter for strong couplings in smoothed aggregation
        this->eps_         = static_cast<ValueType>(0.01f);
        this->over_interp_ = static_cast<ValueType>(1.5f);
        this->strat_       = CoarseningStrategy::Greedy;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    UAAMG<OperatorType, VectorType, ValueType>::~UAAMG()
    {
        log_debug(this, "UAAMG::UAAMG()", "destructor");

        this->Clear();
    }

    // LCOV_EXCL_START
    template <class OperatorType, class VectorType, typename ValueType>
    void UAAMG<OperatorType, VectorType, ValueType>::Print(void) const
    {
        LOG_INFO("UAAMG solver");
        LOG_INFO("UAAMG number of levels " << this->levels_);
        LOG_INFO("UAAMG using unsmoothed aggregation");
        LOG_INFO("UAAMG coarsest operator size = " << this->op_level_[this->levels_ - 2]->GetM());
        LOG_INFO("UAAMG coarsest level nnz = " << this->op_level_[this->levels_ - 2]->GetNnz());
        LOG_INFO("UAAMG with smoother:");
        this->smoother_level_[0]->Print();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void UAAMG<OperatorType, VectorType, ValueType>::PrintStart_(void) const
    {
        assert(this->levels_ > 0);

        LOG_INFO("UAAMG solver starts");
        LOG_INFO("UAAMG number of levels " << this->levels_);
        LOG_INFO("UAAMG using unsmoothed aggregation");
        LOG_INFO("UAAMG coarsest operator size = " << this->op_level_[this->levels_ - 2]->GetM());
        LOG_INFO("UAAMG coarsest level nnz = " << this->op_level_[this->levels_ - 2]->GetNnz());
        LOG_INFO("UAAMG with smoother:");
        this->smoother_level_[0]->Print();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void UAAMG<OperatorType, VectorType, ValueType>::PrintEnd_(void) const
    {
        LOG_INFO("UAAMG ends");
    }
    // LCOV_EXCL_STOP

    template <class OperatorType, class VectorType, typename ValueType>
    void UAAMG<OperatorType, VectorType, ValueType>::SetOverInterp(ValueType overInterp)
    {
        log_debug(this, "UAAMG::SetOverInterp()", overInterp);

        this->over_interp_ = overInterp;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void UAAMG<OperatorType, VectorType, ValueType>::SetCouplingStrength(ValueType eps)
    {
        log_debug(this, "UAAMG::SetCouplingStrength()", eps);

        this->eps_ = eps;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void UAAMG<OperatorType, VectorType, ValueType>::SetCoarseningStrategy(CoarseningStrategy strat)
    {
        log_debug(this, "UAAMG::SetCoarseningStrategy()", strat);

        this->strat_ = strat;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void UAAMG<OperatorType, VectorType, ValueType>::ReBuildNumeric(void)
    {
        log_debug(this, "UAAMG::ReBuildNumeric()", " #*# begin");

        assert(this->levels_ > 1);
        assert(this->build_);
        assert(this->op_ != NULL);

        // Create coarse operator
        this->op_level_[0]->Clear();
        this->op_level_[0]->ConvertToCSR();
        this->op_level_[0]->CloneBackend(*this->op_);

        assert(this->restrict_op_level_[0] != NULL);
        assert(this->prolong_op_level_[0] != NULL);

        if(this->op_->GetFormat() != CSR)
        {
            OperatorType op_csr;
            op_csr.CloneFrom(*this->op_);
            op_csr.ConvertToCSR();

            this->op_level_[0]->TripleMatrixProduct(
                *this->restrict_op_level_[0], op_csr, *this->prolong_op_level_[0]);
        }
        else
        {
            this->op_level_[0]->TripleMatrixProduct(
                *this->restrict_op_level_[0], *this->op_, *this->prolong_op_level_[0]);
        }

        for(int i = 1; i < this->levels_ - 1; ++i)
        {
            this->op_level_[i]->Clear();
            this->op_level_[i]->ConvertToCSR();

            // Create coarse operator
            OperatorType tmp;
            tmp.CloneBackend(*this->op_);
            this->op_level_[i]->CloneBackend(*this->op_);

            assert(this->restrict_op_level_[i] != NULL);
            assert(this->prolong_op_level_[i] != NULL);

            if(i == this->levels_ - this->host_level_ - 1)
            {
                this->op_level_[i - 1]->MoveToHost();
            }

            this->op_level_[i]->TripleMatrixProduct(
                *this->restrict_op_level_[i], *this->op_level_[i - 1], *this->prolong_op_level_[i]);

            if(i == this->levels_ - this->host_level_ - 1)
            {
                this->op_level_[i - 1]->CloneBackend(*this->restrict_op_level_[i - 1]);
            }
        }

        for(int i = 0; i < this->levels_ - 1; ++i)
        {
            if(i > 0)
            {
                this->smoother_level_[i]->ResetOperator(*this->op_level_[i - 1]);
            }
            else
            {
                this->smoother_level_[i]->ResetOperator(*this->op_);
            }

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
    }

    template <class OperatorType, class VectorType, typename ValueType>
    bool UAAMG<OperatorType, VectorType, ValueType>::Aggregate_(const OperatorType& op,
                                                                OperatorType*       pro,
                                                                OperatorType*       res,
                                                                OperatorType*       coarse,
                                                                LocalVector<int>*   trans)
    {
        log_debug(this, "UAAMG::Aggregate_()", this->build_);

        assert(pro != NULL);
        assert(res != NULL);
        assert(coarse != NULL);

        LocalVector<bool>    connections;
        LocalVector<int64_t> aggregates;
        LocalVector<int64_t> aggregate_root_nodes;

        connections.CloneBackend(op);
        aggregates.CloneBackend(op);
        aggregate_root_nodes.CloneBackend(op);

        ValueType eps = this->eps_;
        for(int i = 0; i < this->levels_ - 1; ++i)
        {
            eps *= static_cast<ValueType>(0.5);
        }
        switch(strat_)
        {
        case Greedy:
            op.AMGGreedyAggregate(eps, &connections, &aggregates, &aggregate_root_nodes);
            break;
        case PMIS:
            op.AMGPMISAggregate(eps, &connections, &aggregates, &aggregate_root_nodes);
            break;
        }

        op.AMGUnsmoothedAggregation(aggregates, aggregate_root_nodes, pro);

        // Free unused vectors
        connections.Clear();
        aggregates.Clear();
        aggregate_root_nodes.Clear();

        // Transpose P to obtain R
        pro->Transpose(res);

        // Triple matrix product
        coarse->CloneBackend(op);
        coarse->TripleMatrixProduct(*res, op, *pro);

        if(this->over_interp_ > static_cast<ValueType>(1))
        {
            coarse->Scale(static_cast<ValueType>(1) / this->over_interp_);
        }

        return true;
    }

    template class UAAMG<LocalMatrix<double>, LocalVector<double>, double>;
    template class UAAMG<GlobalMatrix<double>, GlobalVector<double>, double>;
    template class UAAMG<LocalMatrix<float>, LocalVector<float>, float>;
    template class UAAMG<GlobalMatrix<float>, GlobalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class UAAMG<LocalMatrix<std::complex<double>>,
                         LocalVector<std::complex<double>>,
                         std::complex<double>>;
    template class UAAMG<GlobalMatrix<std::complex<double>>,
                         GlobalVector<std::complex<double>>,
                         std::complex<double>>;
    template class UAAMG<LocalMatrix<std::complex<float>>,
                         LocalVector<std::complex<float>>,
                         std::complex<float>>;
    template class UAAMG<GlobalMatrix<std::complex<float>>,
                         GlobalVector<std::complex<float>>,
                         std::complex<float>>;
#endif

} // namespace rocalution

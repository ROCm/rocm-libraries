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

#include "preconditioner_blockjacobi.hpp"
#include "../../base/global_matrix.hpp"
#include "../../base/local_matrix.hpp"
#include "../../utils/def.hpp"
#include "../solver.hpp"

#include "../../base/global_vector.hpp"
#include "../../base/local_vector.hpp"

#include "../../utils/log.hpp"

#include "preconditioner.hpp"

#include <complex>
#include <math.h>

namespace rocalution
{

    template <class OperatorType, class VectorType, typename ValueType>
    BlockJacobi<OperatorType, VectorType, ValueType>::BlockJacobi()
    {
        log_debug(this, "BlockJacobi::BlockJacobi()", "default constructor");

        this->local_precond_ = NULL;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    BlockJacobi<OperatorType, VectorType, ValueType>::~BlockJacobi()
    {
        log_debug(this, "BlockJacobi::~BlockJacobi()", "destructor");

        this->Clear();
    }

    // LCOV_EXCL_START
    template <class OperatorType, class VectorType, typename ValueType>
    void BlockJacobi<OperatorType, VectorType, ValueType>::Print(void) const
    {
        LOG_INFO("BlockJacobi preconditioner");

        this->local_precond_->Print();
    }
    // LCOV_EXCL_STOP

    template <class OperatorType, class VectorType, typename ValueType>
    void BlockJacobi<OperatorType, VectorType, ValueType>::Set(
        Solver<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>& precond)
    {
        log_debug(this, "BlockJacobi::Set()", this->build_, (const void*&)precond);

        assert(this->local_precond_ == NULL);
        assert(this->build_ == false);

        this->local_precond_ = &precond;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BlockJacobi<OperatorType, VectorType, ValueType>::Build(void)
    {
        log_debug(this, "BlockJacobi::Build()", this->build_, " #*# begin");

        if(this->build_ == true)
        {
            this->Clear();
        }

        assert(this->build_ == false);
        this->build_ = true;

        assert(this->op_ != NULL);
        assert(this->local_precond_ != NULL);

        this->local_precond_->SetOperator(this->op_->GetInterior());
        this->local_precond_->Build();

        log_debug(this, "BlockJacobi::Build()", this->build_, " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BlockJacobi<OperatorType, VectorType, ValueType>::ReBuildNumeric(void)
    {
        log_debug(this, "BlockJacobi::ReBuildNumeric()", this->build_);

        if(this->build_ == true)
        {
            this->local_precond_->ReBuildNumeric();
        }
        else
        {
            this->Clear();
            this->Build();
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BlockJacobi<OperatorType, VectorType, ValueType>::Clear(void)
    {
        log_debug(this, "BlockJacobi::Clear()", this->build_);

        if(this->local_precond_ != NULL)
        {
            this->local_precond_->Clear();
        }

        this->local_precond_ = NULL;

        this->build_ = false;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BlockJacobi<OperatorType, VectorType, ValueType>::Solve(const VectorType& rhs,
                                                                 VectorType*       x)
    {
        log_debug(this, "BlockJacobi::Solve()", " #*# begin", (const void*&)rhs, x);

        this->local_precond_->Solve(rhs.GetInterior(), &x->GetInterior());

        log_debug(this, "BlockJacobi::Solve()", " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BlockJacobi<OperatorType, VectorType, ValueType>::SolveZeroSol(const VectorType& rhs,
                                                                        VectorType*       x)
    {
        log_debug(this, "BlockJacobi::SolveZeroSol()", " #*# begin", (const void*&)rhs, x);

        this->local_precond_->SolveZeroSol(rhs.GetInterior(), &x->GetInterior());

        log_debug(this, "BlockJacobi::SolveZeroSol()", " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BlockJacobi<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void)
    {
        log_debug(this, "BlockJacobi::MoveToHostLocalData_()", this->build_);

        this->local_precond_->MoveToHost();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BlockJacobi<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void)
    {
        log_debug(this, "BlockJacobi::MoveToAcceleratorLocalData_()", this->build_);

        this->local_precond_->MoveToAccelerator();
    }

    template class BlockJacobi<GlobalMatrix<double>, GlobalVector<double>, double>;
    template class BlockJacobi<GlobalMatrix<float>, GlobalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class BlockJacobi<GlobalMatrix<std::complex<double>>,
                               GlobalVector<std::complex<double>>,
                               std::complex<double>>;
    template class BlockJacobi<GlobalMatrix<std::complex<float>>,
                               GlobalVector<std::complex<float>>,
                               std::complex<float>>;
#endif

} // namespace rocalution

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

#include "chebyshev.hpp"
#include "../utils/def.hpp"
#include "iter_ctrl.hpp"

#include "../base/local_matrix.hpp"
#include "../base/local_stencil.hpp"
#include "../base/local_vector.hpp"

#include "../base/global_matrix.hpp"
#include "../base/global_vector.hpp"

#include "../utils/log.hpp"
#include "../utils/math_functions.hpp"

#include <complex>
#include <math.h>

namespace rocalution
{

    template <class OperatorType, class VectorType, typename ValueType>
    Chebyshev<OperatorType, VectorType, ValueType>::Chebyshev()
    {
        log_debug(this, "Chebyshev::Chebyshev()");

        this->init_lambda_ = false;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    Chebyshev<OperatorType, VectorType, ValueType>::~Chebyshev()
    {
        log_debug(this, "Chebyshev::~Chebyshev()");

        this->Clear();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void Chebyshev<OperatorType, VectorType, ValueType>::Set(ValueType lambda_min,
                                                             ValueType lambda_max)
    {
        log_debug(this, "Chebyshev::Set()", lambda_min, lambda_max);

        this->lambda_min_ = lambda_min;
        this->lambda_max_ = lambda_max;

        this->init_lambda_ = true;
    }

    // LCOV_EXCL_START
    template <class OperatorType, class VectorType, typename ValueType>
    void Chebyshev<OperatorType, VectorType, ValueType>::Print(void) const
    {
        if(this->precond_ == NULL)
        {
            LOG_INFO("Chebyshev solver");
        }
        else
        {
            LOG_INFO("PChebyshev solver, with preconditioner:");
            this->precond_->Print();
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void Chebyshev<OperatorType, VectorType, ValueType>::PrintStart_(void) const
    {
        if(this->precond_ == NULL)
        {
            LOG_INFO("Chebyshev (non-precond) linear solver starts");
        }
        else
        {
            LOG_INFO("PChebyshev solver starts, with preconditioner:");
            this->precond_->Print();
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void Chebyshev<OperatorType, VectorType, ValueType>::PrintEnd_(void) const
    {
        if(this->precond_ == NULL)
        {
            LOG_INFO("Chebyshev (non-precond) ends");
        }
        else
        {
            LOG_INFO("PChebyshev ends");
        }
    }
    // LCOV_EXCL_STOP

    template <class OperatorType, class VectorType, typename ValueType>
    void Chebyshev<OperatorType, VectorType, ValueType>::Build(void)
    {
        log_debug(this, "Chebyshev::Build()");

        if(this->build_ == true)
        {
            this->Clear();
        }

        assert(this->build_ == false);
        this->build_ = true;

        assert(this->op_ != NULL);

        assert(this->op_->GetM() == this->op_->GetN());
        assert(this->op_->GetM() > 0);

        if(this->precond_ != NULL)
        {
            this->precond_->SetOperator(*this->op_);
            this->precond_->Build();

            this->z_.CloneBackend(*this->op_);
            this->z_.Allocate("z", this->op_->GetM());
        }

        this->r_.CloneBackend(*this->op_);
        this->r_.Allocate("r", this->op_->GetM());

        this->p_.CloneBackend(*this->op_);
        this->p_.Allocate("p", this->op_->GetM());
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void Chebyshev<OperatorType, VectorType, ValueType>::Clear(void)
    {
        log_debug(this, "Chebyshev::Clear()");

        if(this->build_ == true)
        {
            if(this->precond_ != NULL)
            {
                this->precond_->Clear();
                this->precond_ = NULL;
            }

            this->r_.Clear();
            this->z_.Clear();
            this->p_.Clear();

            this->iter_ctrl_.Clear();

            this->build_       = false;
            this->init_lambda_ = false;
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void Chebyshev<OperatorType, VectorType, ValueType>::ReBuildNumeric(void)
    {
        log_debug(this, "Chebyshev::ReBuildNumeric()");

        if(this->build_ == true)
        {
            this->r_.Zeros();
            this->z_.Zeros();
            this->p_.Zeros();

            this->iter_ctrl_.Clear();

            this->build_       = false;
            this->init_lambda_ = false;

            if(this->precond_ != NULL)
            {
                this->precond_->ReBuildNumeric();
            }
        }
        else
        {
            this->Build();
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void Chebyshev<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void)
    {
        log_debug(this, "Chebyshev::MoveToHostLocalData_()");

        if(this->build_ == true)
        {
            this->r_.MoveToHost();
            this->p_.MoveToHost();

            if(this->precond_ != NULL)
            {
                this->z_.MoveToHost();
            }
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void Chebyshev<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void)
    {
        log_debug(this, "Chebyshev::MoveToAcceleratorLocalData_()");

        if(this->build_ == true)
        {
            this->r_.MoveToAccelerator();
            this->p_.MoveToAccelerator();

            if(this->precond_ != NULL)
            {
                this->z_.MoveToAccelerator();
            }
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void Chebyshev<OperatorType, VectorType, ValueType>::SolveNonPrecond_(const VectorType& rhs,
                                                                          VectorType*       x)
    {
        log_debug(this, "Chebyshev::SolveNonPrecond_()", " #*# begin", (const void*&)rhs, x);

        assert(x != NULL);
        assert(x != &rhs);
        assert(this->op_ != NULL);
        assert(this->precond_ == NULL);
        assert(this->build_ == true);
        assert(this->init_lambda_ == true);

        const OperatorType* op = this->op_;

        VectorType* r = &this->r_;
        VectorType* p = &this->p_;

        ValueType two = static_cast<ValueType>(2);
        ValueType alpha, beta;
        ValueType d = (this->lambda_max_ + this->lambda_min_) / two;
        ValueType c = (this->lambda_max_ - this->lambda_min_) / two;

        // initial residual = b - Ax
        op->Apply(*x, r);
        r->ScaleAdd(static_cast<ValueType>(-1), rhs);

        ValueType res = this->Norm_(*r);

        if(this->iter_ctrl_.InitResidual(std::abs(res)) == false)
        {
            log_debug(this, "Chebyshev::SolveNonPrecond_()", " #*# end");

            return;
        }

        // p = r
        p->CopyFrom(*r);

        alpha = two / d;

        // x = x + alpha*p
        x->AddScale(*p, alpha);

        // compute residual = b - Ax
        op->Apply(*x, r);
        r->ScaleAdd(static_cast<ValueType>(-1), rhs);

        res = this->Norm_(*r);
        while(!this->iter_ctrl_.CheckResidual(std::abs(res), this->index_))
        {
            beta = (c * alpha / two) * (c * alpha / two);

            alpha = static_cast<ValueType>(1) / (d - beta);

            // p = beta*p + r
            p->ScaleAdd(beta, *r);

            // x = x + alpha*p
            x->AddScale(*p, alpha);

            // compute residual = b - Ax
            op->Apply(*x, r);
            r->ScaleAdd(static_cast<ValueType>(-1), rhs);
            res = this->Norm_(*r);
        }

        log_debug(this, "Chebyshev::SolveNonPrecond_()", " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void Chebyshev<OperatorType, VectorType, ValueType>::SolvePrecond_(const VectorType& rhs,
                                                                       VectorType*       x)
    {
        log_debug(this, "Chebyshev::SolvePrecond_()", " #*# begin", (const void*&)rhs, x);

        assert(x != NULL);
        assert(x != &rhs);
        assert(this->op_ != NULL);
        assert(this->precond_ != NULL);
        assert(this->build_ == true);
        assert(this->init_lambda_ == true);

        const OperatorType* op = this->op_;

        VectorType* r = &this->r_;
        VectorType* z = &this->z_;
        VectorType* p = &this->p_;

        ValueType two = static_cast<ValueType>(2);
        ValueType alpha, beta;
        ValueType d = (this->lambda_max_ + this->lambda_min_) / two;
        ValueType c = (this->lambda_max_ - this->lambda_min_) / two;

        // initial residual = b - Ax
        op->Apply(*x, r);
        r->ScaleAdd(static_cast<ValueType>(-1), rhs);

        ValueType res = this->Norm_(*r);

        if(this->iter_ctrl_.InitResidual(std::abs(res)) == false)
        {
            log_debug(this, "Chebyshev::SolvePrecond_()", " #*# end");

            return;
        }

        // Solve Mz=r
        this->precond_->SolveZeroSol(*r, z);

        // p = z
        p->CopyFrom(*z);

        alpha = two / d;

        // x = x + alpha*p
        x->AddScale(*p, alpha);

        // compute residual = b - Ax
        op->Apply(*x, r);
        r->ScaleAdd(static_cast<ValueType>(-1), rhs);
        res = this->Norm_(*r);

        while(!this->iter_ctrl_.CheckResidual(std::abs(res), this->index_))
        {
            // Solve Mz=r
            this->precond_->SolveZeroSol(*r, z);

            beta = (c * alpha / two) * (c * alpha / two);

            alpha = static_cast<ValueType>(1) / (d - beta);

            // p = beta*p + z
            p->ScaleAdd(beta, *z);

            // x = x + alpha*p
            x->AddScale(*p, alpha);

            // compute residual = b - Ax
            op->Apply(*x, r);
            r->ScaleAdd(static_cast<ValueType>(-1), rhs);
            res = this->Norm_(*r);
        }

        log_debug(this, "Chebyshev::SolvePrecond_()", " #*# end");
    }

    template class Chebyshev<LocalMatrix<double>, LocalVector<double>, double>;
    template class Chebyshev<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class Chebyshev<LocalMatrix<std::complex<double>>,
                             LocalVector<std::complex<double>>,
                             std::complex<double>>;
    template class Chebyshev<LocalMatrix<std::complex<float>>,
                             LocalVector<std::complex<float>>,
                             std::complex<float>>;
#endif

    template class Chebyshev<GlobalMatrix<double>, GlobalVector<double>, double>;
    template class Chebyshev<GlobalMatrix<float>, GlobalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class Chebyshev<GlobalMatrix<std::complex<double>>,
                             GlobalVector<std::complex<double>>,
                             std::complex<double>>;
    template class Chebyshev<GlobalMatrix<std::complex<float>>,
                             GlobalVector<std::complex<float>>,
                             std::complex<float>>;
#endif

    template class Chebyshev<LocalStencil<double>, LocalVector<double>, double>;
    template class Chebyshev<LocalStencil<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class Chebyshev<LocalStencil<std::complex<double>>,
                             LocalVector<std::complex<double>>,
                             std::complex<double>>;
    template class Chebyshev<LocalStencil<std::complex<float>>,
                             LocalVector<std::complex<float>>,
                             std::complex<float>>;
#endif

} // namespace rocalution

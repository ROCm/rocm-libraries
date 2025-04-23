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

#include "fcg.hpp"
#include "../../utils/def.hpp"
#include "../iter_ctrl.hpp"

#include "../../base/local_matrix.hpp"
#include "../../base/local_stencil.hpp"
#include "../../base/local_vector.hpp"

#include "../../base/global_matrix.hpp"
#include "../../base/global_vector.hpp"

#include "../../utils/log.hpp"
#include "../../utils/math_functions.hpp"

#include <complex>
#include <math.h>

namespace rocalution
{

    template <class OperatorType, class VectorType, typename ValueType>
    FCG<OperatorType, VectorType, ValueType>::FCG()
    {
        log_debug(this, "FCG::FCG()", "default constructor");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    FCG<OperatorType, VectorType, ValueType>::~FCG()
    {
        log_debug(this, "FCG::~FCG()", "destructor");

        this->Clear();
    }

    // LCOV_EXCL_START
    template <class OperatorType, class VectorType, typename ValueType>
    void FCG<OperatorType, VectorType, ValueType>::Print(void) const
    {
        if(this->precond_ == NULL)
        {
            LOG_INFO("Flexible CG solver");
        }
        else
        {
            LOG_INFO("Flexible PCG solver, with preconditioner:");
            this->precond_->Print();
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void FCG<OperatorType, VectorType, ValueType>::PrintStart_(void) const
    {
        if(this->precond_ == NULL)
        {
            LOG_INFO("Flexible CG (non-precond) linear solver starts");
        }
        else
        {
            LOG_INFO("Flexible PCG solver starts, with preconditioner:");
            this->precond_->Print();
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void FCG<OperatorType, VectorType, ValueType>::PrintEnd_(void) const
    {
        if(this->precond_ == NULL)
        {
            LOG_INFO("Flexible CG (non-precond) ends");
        }
        else
        {
            LOG_INFO("Flexible PCG ends");
        }
    }
    // LCOV_EXCL_STOP

    template <class OperatorType, class VectorType, typename ValueType>
    void FCG<OperatorType, VectorType, ValueType>::Build(void)
    {
        log_debug(this, "FCG::Build()", this->build_, " #*# begin");

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

        this->w_.CloneBackend(*this->op_);
        this->w_.Allocate("w", this->op_->GetM());

        this->p_.CloneBackend(*this->op_);
        this->p_.Allocate("p", this->op_->GetM());

        this->q_.CloneBackend(*this->op_);
        this->q_.Allocate("q", this->op_->GetM());

        log_debug(this, "FCG::Build()", this->build_, " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void FCG<OperatorType, VectorType, ValueType>::Clear(void)
    {
        log_debug(this, "FCG::Clear()", this->build_);

        if(this->build_ == true)
        {
            if(this->precond_ != NULL)
            {
                this->precond_->Clear();
                this->precond_ = NULL;
            }

            this->r_.Clear();
            this->w_.Clear();
            this->z_.Clear();
            this->p_.Clear();
            this->q_.Clear();

            this->iter_ctrl_.Clear();

            this->build_ = false;
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void FCG<OperatorType, VectorType, ValueType>::ReBuildNumeric(void)
    {
        log_debug(this, "FCG::Clear()", this->build_);

        if(this->build_ == true)
        {
            this->r_.Zeros();
            this->w_.Zeros();
            this->z_.Zeros();
            this->p_.Zeros();
            this->q_.Zeros();

            this->iter_ctrl_.Clear();

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
    void FCG<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void)
    {
        log_debug(this, "FCG::MoveToHostLocalData_()", this->build_);

        if(this->build_ == true)
        {
            this->r_.MoveToHost();
            this->w_.MoveToHost();
            this->p_.MoveToHost();
            this->q_.MoveToHost();

            if(this->precond_ != NULL)
            {
                this->z_.MoveToHost();
                this->precond_->MoveToHost();
            }
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void FCG<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void)
    {
        log_debug(this, "FCG::MoveToAcceleratorLocalData_()", this->build_);

        if(this->build_ == true)
        {
            this->r_.MoveToAccelerator();
            this->w_.MoveToAccelerator();
            this->p_.MoveToAccelerator();
            this->q_.MoveToAccelerator();

            if(this->precond_ != NULL)
            {
                this->z_.MoveToAccelerator();
                this->precond_->MoveToAccelerator();
            }
        }
    }

    // TODO
    // re-orthogonalization and
    // residual - re-computed % iter
    template <class OperatorType, class VectorType, typename ValueType>
    void FCG<OperatorType, VectorType, ValueType>::SolveNonPrecond_(const VectorType& rhs,
                                                                    VectorType*       x)
    {
        log_debug(this, "FCG::SolveNonPrecond_()", " #*# begin", (const void*&)rhs, x);

        assert(x != NULL);
        assert(x != &rhs);
        assert(this->op_ != NULL);
        assert(this->precond_ == NULL);
        assert(this->build_ == true);

        const OperatorType* op = this->op_;

        VectorType* r = &this->r_;
        VectorType* w = &this->w_;
        VectorType* p = &this->p_;
        VectorType* q = &this->q_;

        ValueType alpha, beta;
        ValueType rho;
        ValueType gamma, gamma_rho;

        // initial residual = b - Ax
        op->Apply(*x, r);
        r->ScaleAdd(static_cast<ValueType>(-1), rhs);

        // initial residual norm
        ValueType res = this->Norm_(*r);
        this->iter_ctrl_.InitResidual(std::abs(res));

        // w = Ar
        op->Apply(*r, w);

        // alpha = (r,r)
        alpha = r->Dot(*r);

        // beta = (r,w)
        beta = r->Dot(*w);

        // p = r
        p->CopyFrom(*r);

        // q = w
        q->CopyFrom(*w);

        // rho = beta
        rho = beta;

        // x = x + alpha/rho * p
        x->AddScale(*p, alpha / rho);

        // r = r - alpha/rho * q
        r->AddScale(*q, -alpha / rho);

        res = this->Norm_(*r);

        while(!this->iter_ctrl_.CheckResidual(std::abs(res), this->index_))
        {
            // w = Ar
            op->Apply(*r, w);

            // beta = (r,w)
            beta = r->Dot(*w);

            // gamma = (r,q)
            gamma     = r->Dot(*q);
            gamma_rho = -gamma / rho;

            // p = r - gamma/rho * p
            p->ScaleAdd(gamma_rho, *r);

            // q = w - gamma/rho * q
            q->ScaleAdd(gamma_rho, *w);

            // rho = beta + gamma^2 / rho
            rho = beta + gamma * gamma_rho;

            // alpha = (r,r) / rho
            alpha = r->Dot(*r) / rho;

            // x = x + alpha*p
            x->AddScale(*p, alpha);

            // r = r - alpha*q
            r->AddScale(*q, -alpha);

            res = this->Norm_(*r);
        }

        log_debug(this, "FCG::SolveNonPrecond_()", " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void FCG<OperatorType, VectorType, ValueType>::SolvePrecond_(const VectorType& rhs,
                                                                 VectorType*       x)
    {
        log_debug(this, "FCG::SolvePrecond_()", " #*# begin", (const void*&)rhs, x);

        assert(x != NULL);
        assert(x != &rhs);
        assert(this->op_ != NULL);
        assert(this->precond_ != NULL);
        assert(this->build_ == true);

        const OperatorType* op = this->op_;

        VectorType* r = &this->r_;
        VectorType* w = &this->w_;
        VectorType* z = &this->z_;
        VectorType* p = &this->p_;
        VectorType* q = &this->q_;

        ValueType alpha, beta;
        ValueType rho;
        ValueType gamma, gamma_rho;

        // initial residual = b - Ax
        op->Apply(*x, r);
        r->ScaleAdd(static_cast<ValueType>(-1), rhs);

        // initial residual norm
        ValueType res = this->Norm_(*r);
        this->iter_ctrl_.InitResidual(std::abs(res));

        // Mz = r
        this->precond_->SolveZeroSol(*r, z);

        // w = Az
        op->Apply(*z, w);

        // alpha = (z,r)
        alpha = z->Dot(*r);

        // beta = (z,w)
        beta = z->Dot(*w);

        // p = z
        p->CopyFrom(*z);

        // q = w
        q->CopyFrom(*w);

        // rho = beta
        rho = beta;

        // x = x + alpha/rho * p
        x->AddScale(*p, alpha / rho);

        // r = r - alpha/rho * q
        r->AddScale(*q, -alpha / rho);

        res = this->Norm_(*r);

        while(!this->iter_ctrl_.CheckResidual(std::abs(res), this->index_))
        {
            // Mz = r
            this->precond_->SolveZeroSol(*r, z);

            // w = Az
            op->Apply(*z, w);

            // beta = (r,w)
            beta = z->Dot(*w);

            // gamma = (z,q)
            gamma     = z->Dot(*q);
            gamma_rho = -gamma / rho;

            // p = z - gamma/rho * p
            p->ScaleAdd(gamma_rho, *z);

            // q = w - gamma/rho * q
            q->ScaleAdd(gamma_rho, *w);

            // rho = beta + gamma^2 / rho
            rho = beta + gamma * gamma_rho;

            // alpha = (z,r) / rho
            alpha = z->Dot(*r) / rho;

            // x = x + alpha*p
            x->AddScale(*p, alpha);

            // r = r - alpha*q
            r->AddScale(*q, -alpha);

            res = this->Norm_(*r);
        }

        log_debug(this, "FCG::SolvePrecond_()", " #*# end");
    }

    template class FCG<LocalMatrix<double>, LocalVector<double>, double>;
    template class FCG<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class FCG<LocalMatrix<std::complex<double>>,
                       LocalVector<std::complex<double>>,
                       std::complex<double>>;
    template class FCG<LocalMatrix<std::complex<float>>,
                       LocalVector<std::complex<float>>,
                       std::complex<float>>;
#endif

    template class FCG<GlobalMatrix<double>, GlobalVector<double>, double>;
    template class FCG<GlobalMatrix<float>, GlobalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class FCG<GlobalMatrix<std::complex<double>>,
                       GlobalVector<std::complex<double>>,
                       std::complex<double>>;
    template class FCG<GlobalMatrix<std::complex<float>>,
                       GlobalVector<std::complex<float>>,
                       std::complex<float>>;
#endif

} // namespace rocalution

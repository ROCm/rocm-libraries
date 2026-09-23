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

#include "cr.hpp"
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
    CR<OperatorType, VectorType, ValueType>::CR()
    {
        log_debug(this, "CR::CR()", "default constructor");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    CR<OperatorType, VectorType, ValueType>::~CR()
    {
        log_debug(this, "CR::~CR()", "destructor");

        this->Clear();
    }

    // LCOV_EXCL_START
    template <class OperatorType, class VectorType, typename ValueType>
    void CR<OperatorType, VectorType, ValueType>::Print(void) const
    {
        if(this->precond_ == NULL)
        {
            LOG_INFO("CR solver");
        }
        else
        {
            LOG_INFO("PCR solver, with preconditioner:");
            this->precond_->Print();
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void CR<OperatorType, VectorType, ValueType>::PrintStart_(void) const
    {
        if(this->precond_ == NULL)
        {
            LOG_INFO("CR (non-precond) linear solver starts");
        }
        else
        {
            LOG_INFO("PCR solver starts, with preconditioner:");
            this->precond_->Print();
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void CR<OperatorType, VectorType, ValueType>::PrintEnd_(void) const
    {
        if(this->precond_ == NULL)
        {
            LOG_INFO("CR (non-precond) ends");
        }
        else
        {
            LOG_INFO("PCR ends");
        }
    }
    // LCOV_EXCL_STOP

    template <class OperatorType, class VectorType, typename ValueType>
    void CR<OperatorType, VectorType, ValueType>::Build(void)
    {
        log_debug(this, "CRG::Build()", this->build_, " #*# begin");

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

            this->t_.CloneBackend(*this->op_);
            this->t_.Allocate("t", this->op_->GetM());
        }

        this->r_.CloneBackend(*this->op_);
        this->r_.Allocate("r", this->op_->GetM());

        this->p_.CloneBackend(*this->op_);
        this->p_.Allocate("p", this->op_->GetM());

        this->q_.CloneBackend(*this->op_);
        this->q_.Allocate("q", this->op_->GetM());

        this->v_.CloneBackend(*this->op_);
        this->v_.Allocate("v", this->op_->GetM());

        log_debug(this, "CR::Build()", this->build_, " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void CR<OperatorType, VectorType, ValueType>::Clear(void)
    {
        log_debug(this, "CR::Clear()", this->build_);

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
            this->q_.Clear();
            this->v_.Clear();
            this->t_.Clear();

            this->iter_ctrl_.Clear();

            this->build_ = false;
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void CR<OperatorType, VectorType, ValueType>::ReBuildNumeric(void)
    {
        log_debug(this, "CR::ReBuildNumeric()", this->build_);

        if(this->build_ == true)
        {
            this->r_.Zeros();
            this->z_.Zeros();
            this->p_.Zeros();
            this->q_.Zeros();
            this->v_.Zeros();
            this->t_.Zeros();

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
    void CR<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void)
    {
        log_debug(this, "CR::MoveToHostLocalData_()", this->build_);

        if(this->build_ == true)
        {
            this->r_.MoveToHost();
            this->p_.MoveToHost();
            this->q_.MoveToHost();
            this->v_.MoveToHost();

            if(this->precond_ != NULL)
            {
                this->z_.MoveToHost();
                this->t_.MoveToHost();
                this->precond_->MoveToHost();
            }
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void CR<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void)
    {
        log_debug(this, "CR::MoveToAcceleratorLocalData_()", this->build_);

        if(this->build_ == true)
        {
            this->r_.MoveToAccelerator();
            this->p_.MoveToAccelerator();
            this->q_.MoveToAccelerator();
            this->v_.MoveToAccelerator();

            if(this->precond_ != NULL)
            {
                this->z_.MoveToAccelerator();
                this->t_.MoveToAccelerator();
                this->precond_->MoveToAccelerator();
            }
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void CR<OperatorType, VectorType, ValueType>::SolveNonPrecond_(const VectorType& rhs,
                                                                   VectorType*       x)
    {
        log_debug(this, "CR::SolveNonPrecond_()", " #*# begin", (const void*&)rhs, x);

        assert(x != NULL);
        assert(x != &rhs);
        assert(this->op_ != NULL);
        assert(this->precond_ == NULL);
        assert(this->build_ == true);

        const OperatorType* op = this->op_;

        VectorType* r = &this->r_;
        VectorType* p = &this->p_;
        VectorType* q = &this->q_;
        VectorType* v = &this->v_;

        ValueType alpha, beta;
        ValueType rho, rho_old;

        // initial residual = b - Ax
        op->Apply(*x, r);
        r->ScaleAdd(static_cast<ValueType>(-1), rhs);

        // p = r
        p->CopyFrom(*r);

        // use for |b-Ax0|
        ValueType res_norm = this->Norm_(*r);

        if(this->iter_ctrl_.InitResidual(std::abs(res_norm)) == false)
        {
            log_debug(this, "CR::SolveNonPrecond_()", " #*# end");

            return;
        }

        // use for |b|
        //  this->iter_ctrl_.InitResidual(rhs.Norm_());

        // v=Ar
        op->Apply(*r, v);

        // rho = (r,v)
        rho = r->DotNonConj(*v);

        // q=Ap
        op->Apply(*p, q);

        // alpha = rho / (q,q)
        alpha = rho / q->DotNonConj(*q);

        // x = x + alpha * p
        x->AddScale(*p, alpha);

        // r = r - alpha * q
        r->AddScale(*q, -alpha);

        res_norm = this->Norm_(*r);

        while(!this->iter_ctrl_.CheckResidual(std::abs(res_norm), this->index_))
        {
            rho_old = rho;

            // v=Ar
            op->Apply(*r, v);

            // rho = (r,v)
            rho = r->DotNonConj(*v);

            beta = rho / rho_old;

            // p = beta*p + r
            p->ScaleAdd(beta, *r);

            // q = beta*q + v
            q->ScaleAdd(beta, *v);

            // alpha = rho / (q,q)
            alpha = rho / q->DotNonConj(*q);

            // x = x + alpha * p
            x->AddScale(*p, alpha);

            // r = r - alpha * q
            r->AddScale(*q, -alpha);

            res_norm = this->Norm_(*r);
        }

        log_debug(this, "CR::SolveNonPrecond_()", " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void CR<OperatorType, VectorType, ValueType>::SolvePrecond_(const VectorType& rhs,
                                                                VectorType*       x)
    {
        log_debug(this, "CR::SolvePrecond_()", " #*# begin", (const void*&)rhs, x);

        assert(x != NULL);
        assert(x != &rhs);
        assert(this->op_ != NULL);
        assert(this->precond_ != NULL);
        assert(this->build_ == true);

        const OperatorType* op = this->op_;

        VectorType* r = &this->r_;
        VectorType* z = &this->z_;
        VectorType* p = &this->p_;
        VectorType* q = &this->q_;
        VectorType* v = &this->v_;
        VectorType* t = &this->t_;

        ValueType alpha, beta;
        ValueType rho, rho_old;

        // initial residual = b - Ax
        op->Apply(*x, z);
        z->ScaleAdd(static_cast<ValueType>(-1), rhs);

        // Solve Mr=z
        this->precond_->SolveZeroSol(*z, r);

        // p = r
        p->CopyFrom(*r);

        // t = z
        t->CopyFrom(*z);

        // use for |b-Ax0|
        ValueType res_norm = this->Norm_(*t);

        if(this->iter_ctrl_.InitResidual(std::abs(res_norm)) == false)
        {
            log_debug(this, "CR::SolvePrecond_()", " #*# end");

            return;
        }

        // use for |b|
        //  this->iter_ctrl_.InitResidual(rhs.Norm_());

        // v=Ar
        op->Apply(*r, v);

        // rho = (r,v)
        rho = r->DotNonConj(*v);

        // q=Ap
        op->Apply(*p, q);

        // Mz=q
        this->precond_->SolveZeroSol(*q, z);

        // alpha = rho / (q,z)
        alpha = rho / q->DotNonConj(*z);

        // x = x + alpha * p
        x->AddScale(*p, alpha);

        // r = r - alpha * z
        r->AddScale(*z, -alpha);

        // t = t - alpha * q
        t->AddScale(*q, -alpha);

        res_norm = this->Norm_(*t);

        while(!this->iter_ctrl_.CheckResidual(std::abs(res_norm), this->index_))
        {
            rho_old = rho;

            // v=Ar
            op->Apply(*r, v);

            // rho = (r,v)
            rho = r->DotNonConj(*v);

            beta = rho / rho_old;

            // p = beta*p + r
            p->ScaleAdd(beta, *r);

            // q = beta*q + v
            q->ScaleAdd(beta, *v);

            // Mz=q
            this->precond_->SolveZeroSol(*q, z);

            // alpha = rho / (q,z)
            alpha = rho / q->DotNonConj(*z);

            // x = x + alpha * p
            x->AddScale(*p, alpha);

            // r = r - alpha * z
            r->AddScale(*z, -alpha);

            // t = t - alpha * q
            t->AddScale(*q, -alpha);

            res_norm = this->Norm_(*t);
        }

        log_debug(this, "CR::SolvePrecond_()", " #*# end");
    }

    template class CR<LocalMatrix<double>, LocalVector<double>, double>;
    template class CR<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class CR<LocalMatrix<std::complex<double>>,
                      LocalVector<std::complex<double>>,
                      std::complex<double>>;
    template class CR<LocalMatrix<std::complex<float>>,
                      LocalVector<std::complex<float>>,
                      std::complex<float>>;
#endif

    template class CR<GlobalMatrix<double>, GlobalVector<double>, double>;
    template class CR<GlobalMatrix<float>, GlobalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class CR<GlobalMatrix<std::complex<double>>,
                      GlobalVector<std::complex<double>>,
                      std::complex<double>>;
    template class CR<GlobalMatrix<std::complex<float>>,
                      GlobalVector<std::complex<float>>,
                      std::complex<float>>;
#endif

    template class CR<LocalStencil<double>, LocalVector<double>, double>;
    template class CR<LocalStencil<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class CR<LocalStencil<std::complex<double>>,
                      LocalVector<std::complex<double>>,
                      std::complex<double>>;
    template class CR<LocalStencil<std::complex<float>>,
                      LocalVector<std::complex<float>>,
                      std::complex<float>>;
#endif

} // namespace rocalution

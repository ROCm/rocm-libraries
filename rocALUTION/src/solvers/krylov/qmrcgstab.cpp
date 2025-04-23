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

#include "qmrcgstab.hpp"
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
    QMRCGStab<OperatorType, VectorType, ValueType>::QMRCGStab()
    {
        log_debug(this, "QMRCGStab::QMRCGStab()", "default constructor");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    QMRCGStab<OperatorType, VectorType, ValueType>::~QMRCGStab()
    {
        log_debug(this, "QMRCGStab::~QMRCGStab()", "destructor");

        this->Clear();
    }

    // LCOV_EXCL_START
    template <class OperatorType, class VectorType, typename ValueType>
    void QMRCGStab<OperatorType, VectorType, ValueType>::Print(void) const
    {
        if(this->precond_ == NULL)
        {
            LOG_INFO("QMRCGStab solver");
        }
        else
        {
            LOG_INFO("PQMRCGStab solver, with preconditioner:");
            this->precond_->Print();
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void QMRCGStab<OperatorType, VectorType, ValueType>::PrintStart_(void) const
    {
        if(this->precond_ == NULL)
        {
            LOG_INFO("QMRCGStab (non-precond) linear solver starts");
        }
        else
        {
            LOG_INFO("PQMRCGStab solver starts, with preconditioner:");
            this->precond_->Print();
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void QMRCGStab<OperatorType, VectorType, ValueType>::PrintEnd_(void) const
    {
        if(this->precond_ == NULL)
        {
            LOG_INFO("QMRCGStab (non-precond) ends");
        }
        else
        {
            LOG_INFO("PQMRCGStab ends");
        }
    }
    // LCOV_EXCL_STOP

    template <class OperatorType, class VectorType, typename ValueType>
    void QMRCGStab<OperatorType, VectorType, ValueType>::Build(void)
    {
        log_debug(this, "QMRCGStab::Build()", this->build_, " #*# begin");

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

        this->r0_.CloneBackend(*this->op_);
        this->r0_.Allocate("r0", this->op_->GetM());

        this->r_.CloneBackend(*this->op_);
        this->r_.Allocate("r", this->op_->GetM());

        this->p_.CloneBackend(*this->op_);
        this->p_.Allocate("p", this->op_->GetM());

        this->t_.CloneBackend(*this->op_);
        this->t_.Allocate("t", this->op_->GetM());

        this->v_.CloneBackend(*this->op_);
        this->v_.Allocate("v", this->op_->GetM());

        this->d_.CloneBackend(*this->op_);
        this->d_.Allocate("d", this->op_->GetM());

        log_debug(this, "QMRCGStab::Build()", this->build_, " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void QMRCGStab<OperatorType, VectorType, ValueType>::Clear(void)
    {
        log_debug(this, "QMRCGStab::Clear()", this->build_);

        if(this->build_ == true)
        {
            this->r0_.Clear();
            this->r_.Clear();
            this->p_.Clear();
            this->t_.Clear();
            this->v_.Clear();
            this->d_.Clear();

            if(this->precond_ != NULL)
            {
                this->precond_->Clear();
                this->precond_ = NULL;

                this->z_.Clear();
            }

            this->iter_ctrl_.Clear();

            this->build_ = false;
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void QMRCGStab<OperatorType, VectorType, ValueType>::ReBuildNumeric(void)
    {
        log_debug(this, "QMRCGStab::ReBuildNumeric()", this->build_);

        if(this->build_ == true)
        {
            this->r0_.Zeros();
            this->r_.Zeros();
            this->p_.Zeros();
            this->t_.Zeros();
            this->v_.Zeros();
            this->d_.Zeros();

            this->iter_ctrl_.Clear();

            if(this->precond_ != NULL)
            {
                this->precond_->ReBuildNumeric();
                this->z_.Zeros();
            }
        }
        else
        {
            this->Build();
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void QMRCGStab<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void)
    {
        log_debug(this, "QMRCGStab::MoveToHostLocalData_()", this->build_);

        if(this->build_ == true)
        {
            this->r0_.MoveToHost();
            this->r_.MoveToHost();
            this->p_.MoveToHost();
            this->t_.MoveToHost();
            this->v_.MoveToHost();
            this->d_.MoveToHost();

            if(this->precond_ != NULL)
            {
                this->z_.MoveToHost();
            }
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void QMRCGStab<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void)
    {
        log_debug(this, "QMRCGStab::MoveToAcceleratorLocalData_()", this->build_);

        if(this->build_ == true)
        {
            this->r0_.MoveToAccelerator();
            this->r_.MoveToAccelerator();
            this->p_.MoveToAccelerator();
            this->t_.MoveToAccelerator();
            this->v_.MoveToAccelerator();
            this->d_.MoveToAccelerator();

            if(this->precond_ != NULL)
            {
                this->z_.MoveToAccelerator();
            }
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void QMRCGStab<OperatorType, VectorType, ValueType>::SolveNonPrecond_(const VectorType& rhs,
                                                                          VectorType*       x)
    {
        log_debug(this, "QMRCGStab::SolveNonPrecond_()", " #*# begin", (const void*&)rhs, x);

        assert(x != NULL);
        assert(x != &rhs);
        assert(this->op_ != NULL);
        assert(this->precond_ == NULL);
        assert(this->build_ == true);

        const OperatorType* op = this->op_;

        VectorType* r0 = &this->r0_;
        VectorType* r  = &this->r_;
        VectorType* p  = &this->p_;
        VectorType* t  = &this->t_;
        VectorType* v  = &this->v_;
        VectorType* d  = &this->d_;

        ValueType alpha, beta, omega;
        ValueType theta1, theta1sq, theta2, theta2sq;
        ValueType eta1, eta2, tau1, tau2;
        ValueType rho, rho_old, c;

        // inital residual r0 = b - Ax
        op->Apply(*x, r0);
        r0->ScaleAdd(static_cast<ValueType>(-1), rhs);

        // r = r0
        r->CopyFrom(*r0);

        // initial residual
        tau2            = this->Norm_(*r0);
        double res_norm = std::abs(tau2);

        this->iter_ctrl_.InitResidual(res_norm);

        // rho = (r0,r)
        rho = r0->Dot(*r);

        // beta = rho
        beta = rho;

        // p = p + r
        p->AddScale(*r, static_cast<ValueType>(1));

        // v = Ap
        op->Apply(*p, v);

        // rho_old = (r0,v)
        rho_old = r0->Dot(*v);

        // alpha = (r0,r) / (r0,v)
        alpha = rho / rho_old;

        // r = r - alpha * v
        r->AddScale(*v, -alpha);

        // First quasi-minimization and update iterate

        // theta1 = ||r|| / tau2
        theta1   = this->Norm_(*r) / tau2;
        theta1sq = theta1 * theta1;

        // c = 1 / sqrt(1 + theta1 * theta1)
        c = static_cast<ValueType>(1) / sqrt(static_cast<ValueType>(1) + theta1sq);

        // tau1 = tau2 * theta1 * c
        tau1 = tau2 * theta1 * c;

        // eta1 = c * c * alpha
        eta1 = c * c * alpha;

        // d = p
        d->CopyFrom(*p);

        // x = x + eta1 * d
        x->AddScale(*d, eta1);

        // Compute t_k, omega and update r_k

        // t = Ar
        op->Apply(*r, t);

        // omega = (r,t) / (t,t)
        omega = r->Dot(*t) / t->Dot(*t);

        // d = theta1 * theta1 * eta1 / omega * d + r
        d->ScaleAdd(theta1sq * eta1 / omega, *r);

        // r = r - omega * t
        r->AddScale(*t, -omega);

        // Second quasi-minimization and update iterate

        // theta2 = ||r|| / tau1
        theta2   = this->Norm_(*r) / tau1;
        theta2sq = theta2 * theta2;

        // c = 1 / sqrt(1 + theta2 * theta2)
        c = static_cast<ValueType>(1) / sqrt(static_cast<ValueType>(1) + theta2sq);

        // tau2 = tau1 * theta2 * c
        tau2 = tau1 * theta2 * c;

        // eta2 = c * c * omega
        eta2 = c * c * omega;

        // x = x + eta2 * d
        x->AddScale(*d, eta2);

        // residual <= sqrt(#iter+1) * |tau2|
        res_norm
            = sqrt(static_cast<double>(this->iter_ctrl_.GetIterationCount() + 1)) * std::abs(tau2);

        while(!this->iter_ctrl_.CheckResidual(res_norm, this->index_))
        {
            // rho_old = rho
            rho_old = rho;

            // rho = (r0,r)
            rho = r0->Dot(*r);

            // beta = (rho * alpha) / (rho_old * omega)
            beta = (rho * alpha) / (rho_old * omega);

            // p = r + beta * (p - omega * v)
            p->AddScale(*v, -omega);
            p->Scale(beta);
            p->AddScale(*r, static_cast<ValueType>(1));

            // v = Ap
            op->Apply(*p, v);

            // rho_old = (r0,v)
            rho_old = r0->Dot(*v);

            if(rho_old == static_cast<ValueType>(0))
            {
                LOG_INFO("QMRCGStab break rho_old == 0 !!!");
                break;
            }

            // alpha = (r0,r) / (r0,v)
            alpha = rho / rho_old;

            // r = r - alpha * v
            r->AddScale(*v, -alpha);

            // First quasi-minimization and update iterate

            // theta1 = ||r|| / tau2
            theta1   = this->Norm_(*r) / tau2;
            theta1sq = theta1 * theta1;

            // c = 1 / sqrt(1 + theta1* theta1)
            c = static_cast<ValueType>(1) / sqrt(static_cast<ValueType>(1) + theta1sq);

            // tau1 = tau2 * theta1 * c
            tau1 = tau2 * theta1 * c;

            // eta1 = c * c * alpha
            eta1 = c * c * alpha;

            // d = p + theta2 * theta2 * eta2 / alpha * d
            d->ScaleAdd(theta2sq * eta2 / alpha, *p);

            // x = x + eta1 * d
            x->AddScale(*d, eta1);

            // Compute t_k, omega and update r_k

            // t = Ar
            op->Apply(*r, t);

            // omega = (t,t)
            omega = t->Dot(*t);

            if(omega == static_cast<ValueType>(0))
            {
                LOG_INFO("QMRCGStab omega == 0 !!!");
                break;
            }

            // omega = (r,t) / (t,t)
            omega = r->Dot(*t) / omega;

            // d = r + theta1 * theta1 * eta1 / omega * d
            d->ScaleAdd(theta1sq * eta1 / omega, *r);

            // r = r - omega * t
            r->AddScale(*t, -omega);

            // Second quasi-minimization and update iterate

            // theta2 = ||r|| / tau
            theta2   = this->Norm_(*r) / tau1;
            theta2sq = theta2 * theta2;

            // c = 1 / sqrt(1 + theta2 * theta2)
            c = static_cast<ValueType>(1) / sqrt(static_cast<ValueType>(1) + theta2sq);

            // tau2 = tau1 * theta2 * c
            tau2 = tau1 * theta2 * c;

            // eta2 = c * c * omega
            eta2 = c * c * omega;

            // x = x + eta2 * d
            x->AddScale(*d, eta2);

            // residual <= sqrt(#iter+1) * |tau2|
            res_norm = sqrt(static_cast<double>(this->iter_ctrl_.GetIterationCount() + 1))
                       * std::abs(tau2);
        }

        // Compute final residual
        op->Apply(*x, r0);
        r0->ScaleAdd(static_cast<ValueType>(-1), rhs);

        this->iter_ctrl_.CheckResidual(std::abs(this->Norm_(*r0)));

        log_debug(this, "QMRCGStab::SolveNonPrecond_()", " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void QMRCGStab<OperatorType, VectorType, ValueType>::SolvePrecond_(const VectorType& rhs,
                                                                       VectorType*       x)
    {
        log_debug(this, "QMRCGStab::SolvePrecond_()", " #*# begin", (const void*&)rhs, x);

        assert(x != NULL);
        assert(x != &rhs);
        assert(this->op_ != NULL);
        assert(this->precond_ != NULL);
        assert(this->build_ == true);

        const OperatorType* op = this->op_;

        VectorType* r0 = &this->r0_;
        VectorType* r  = &this->r_;
        VectorType* p  = &this->p_;
        VectorType* t  = &this->t_;
        VectorType* v  = &this->v_;
        VectorType* d  = &this->d_;
        VectorType* z  = &this->z_;

        ValueType alpha, beta, omega;
        ValueType theta1, theta1sq, theta2, theta2sq;
        ValueType eta1, eta2, tau1, tau2;
        ValueType rho, rho_old, c;

        // inital residual r0 = b - Ax
        op->Apply(*x, r0);
        r0->ScaleAdd(static_cast<ValueType>(-1), rhs);

        // r = r0
        r->CopyFrom(*r0);

        // initial residual
        tau2            = this->Norm_(*r0);
        double res_norm = std::abs(tau2);

        this->iter_ctrl_.InitResidual(res_norm);

        // rho = (r0,r)
        rho = r0->Dot(*r);

        // beta = rho
        beta = rho;

        // p = p + r
        p->AddScale(*r, static_cast<ValueType>(1));

        // Mz = p
        this->precond_->SolveZeroSol(*p, z);

        // v = Az
        op->Apply(*z, v);

        // rho_old = (r0,v)
        rho_old = r0->Dot(*v);

        // alpha = (r0,r) / (r0,v)
        alpha = rho / rho_old;

        // r = r - alpha * v
        r->AddScale(*v, -alpha);

        // First quasi-minimization and update iterate

        // theta1 = ||r|| / tau2
        theta1   = this->Norm_(*r) / tau2;
        theta1sq = theta1 * theta1;

        // c = 1 / sqrt(1 + theta1 * theta1)
        c = static_cast<ValueType>(1) / sqrt(static_cast<ValueType>(1) + theta1sq);

        // tau1 = tau2 * theta1 * c
        tau1 = tau2 * theta1 * c;

        // eta1 = c * c * alpha
        eta1 = c * c * alpha;

        // d = p
        d->CopyFrom(*z);

        // x = x + eta1 * d
        x->AddScale(*d, eta1);

        // Compute t_k, omega and update r_k

        // Mz = r
        this->precond_->SolveZeroSol(*r, z);

        // t = Az
        op->Apply(*z, t);

        // omega = (r,t) / (t,t)
        omega = t->Dot(*r) / t->Dot(*t);

        // d = theta1 * theta1 * eta1 / omega * d + r
        d->ScaleAdd(theta1sq * eta1 / omega, *z);

        // r = r - omega * t
        r->AddScale(*t, -omega);

        // Second quasi-minimization and update iterate

        // theta2 = ||r|| / tau1
        theta2   = this->Norm_(*r) / tau1;
        theta2sq = theta2 * theta2;

        // c = 1 / sqrt(1 + theta2 * theta2)
        c = static_cast<ValueType>(1) / sqrt(static_cast<ValueType>(1) + theta2sq);

        // tau2 = tau1 * theta2 * c
        tau2 = tau1 * theta2 * c;

        // eta2 = c * c * omega
        eta2 = c * c * omega;

        // x = x + eta2 * d
        x->AddScale(*d, eta2);

        // residual <= sqrt(#iter+1) * |tau2|
        res_norm
            = sqrt(static_cast<double>(this->iter_ctrl_.GetIterationCount() + 1)) * std::abs(tau2);

        while(!this->iter_ctrl_.CheckResidual(res_norm, this->index_))
        {
            // rho_old = rho
            rho_old = rho;

            // rho = (r0,r)
            rho = r0->Dot(*r);

            // beta = (rho * alpha) / (rho_old * omega)
            beta = (rho * alpha) / (rho_old * omega);

            // p = r + beta * (p - omega * v)
            p->AddScale(*v, -omega);
            p->Scale(beta);
            p->AddScale(*r, static_cast<ValueType>(1));

            // Mz = p
            this->precond_->SolveZeroSol(*p, z);

            // v = Ap
            op->Apply(*z, v);

            // rho_old = (r0,v)
            rho_old = r0->Dot(*v);

            if(rho_old == static_cast<ValueType>(0))
            {
                LOG_INFO("QMRCGStab break rho_old == 0 !!!");
                break;
            }

            // alpha = (r0,r) / (r0,v)
            alpha = rho / rho_old;

            // r = r - alpha * v
            r->AddScale(*v, -alpha);

            // First quasi-minimization and update iterate

            // theta1 = ||r|| / tau2
            theta1   = this->Norm_(*r) / tau2;
            theta1sq = theta1 * theta1;

            // c = 1 / sqrt(1 + theta1* theta1)
            c = static_cast<ValueType>(1) / sqrt(static_cast<ValueType>(1) + theta1sq);

            // tau1 = tau2 * theta1 * c
            tau1 = tau2 * theta1 * c;

            // eta1 = c * c * alpha
            eta1 = c * c * alpha;

            // d = p + theta2 * theta2 * eta2 / alpha * d
            d->ScaleAdd(theta2sq * eta2 / alpha, *z);

            // x = x + eta1 * d
            x->AddScale(*d, eta1);

            // Compute t_k, omega and update r_k

            // Mz = r
            this->precond_->SolveZeroSol(*r, z);

            // t = Ar
            op->Apply(*z, t);

            // omega = (t,t)
            omega = t->Dot(*t);

            if(omega == static_cast<ValueType>(0))
            {
                LOG_INFO("QMRCGStab omega == 0 !!!");
                break;
            }

            // omega = (r,t) / (t,t)
            omega = t->Dot(*r) / omega;

            // d = r + theta1 * theta1 * eta1 / omega * d
            d->ScaleAdd(theta1sq * eta1 / omega, *z);

            // r = r - omega * t
            r->AddScale(*t, -omega);

            // Second quasi-minimization and update iterate

            // theta2 = ||r|| / tau
            theta2   = this->Norm_(*r) / tau1;
            theta2sq = theta2 * theta2;

            // c = 1 / sqrt(1 + theta2 * theta2)
            c = static_cast<ValueType>(1) / sqrt(static_cast<ValueType>(1) + theta2sq);

            // tau2 = tau1 * theta2 * c
            tau2 = tau1 * theta2 * c;

            // eta2 = c * c * omega
            eta2 = c * c * omega;

            // x = x + eta2 * d
            x->AddScale(*d, eta2);

            // residual <= sqrt(#iter+1) * |tau2|
            res_norm = sqrt(static_cast<double>(this->iter_ctrl_.GetIterationCount() + 1))
                       * std::abs(tau2);
        }

        // Compute final residual
        op->Apply(*x, r0);
        r0->ScaleAdd(static_cast<ValueType>(-1), rhs);

        this->iter_ctrl_.CheckResidual(std::abs(this->Norm_(*r0)));

        log_debug(this, "QMRCGStab::SolvePrecond_()", " #*# end");
    }

    template class QMRCGStab<LocalMatrix<double>, LocalVector<double>, double>;
    template class QMRCGStab<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class QMRCGStab<LocalMatrix<std::complex<double>>,
                             LocalVector<std::complex<double>>,
                             std::complex<double>>;
    template class QMRCGStab<LocalMatrix<std::complex<float>>,
                             LocalVector<std::complex<float>>,
                             std::complex<float>>;
#endif

    template class QMRCGStab<GlobalMatrix<double>, GlobalVector<double>, double>;
    template class QMRCGStab<GlobalMatrix<float>, GlobalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class QMRCGStab<GlobalMatrix<std::complex<double>>,
                             GlobalVector<std::complex<double>>,
                             std::complex<double>>;
    template class QMRCGStab<GlobalMatrix<std::complex<float>>,
                             GlobalVector<std::complex<float>>,
                             std::complex<float>>;
#endif

} // namespace rocalution

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

#include "idr.hpp"
#include "../../utils/def.hpp"
#include "../iter_ctrl.hpp"

#include "../../base/local_matrix.hpp"
#include "../../base/local_stencil.hpp"
#include "../../base/local_vector.hpp"

#include "../../base/global_matrix.hpp"
#include "../../base/global_vector.hpp"

#include "../../base/matrix_formats_ind.hpp"

#include "../../utils/allocate_free.hpp"
#include "../../utils/log.hpp"
#include "../../utils/math_functions.hpp"

#include <limits>
#include <math.h>
#include <time.h>

namespace rocalution
{

    template <class OperatorType, class VectorType, typename ValueType>
    IDR<OperatorType, VectorType, ValueType>::IDR()
    {
        log_debug(this, "IDR::IDR()", "default constructor");

        this->s_    = 4;
        this->seed_ = time(NULL);

        this->kappa_ = static_cast<ValueType>(0.7f);

        this->c_ = NULL;
        this->f_ = NULL;
        this->M_ = NULL;

        this->G_ = NULL;
        this->U_ = NULL;
        this->P_ = NULL;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    IDR<OperatorType, VectorType, ValueType>::~IDR()
    {
        log_debug(this, "IDR::~IDR()", "destructor");

        this->Clear();
    }

    // LCOV_EXCL_START
    template <class OperatorType, class VectorType, typename ValueType>
    void IDR<OperatorType, VectorType, ValueType>::Print(void) const
    {
        if(this->precond_ == NULL)
        {
            LOG_INFO("IDR(" << this->s_ << ") solver");
        }
        else
        {
            LOG_INFO("IDR(" << this->s_ << ") solver, with preconditioner:");
            this->precond_->Print();
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void IDR<OperatorType, VectorType, ValueType>::PrintStart_(void) const
    {
        if(this->precond_ == NULL)
        {
            LOG_INFO("IDR(" << this->s_ << ") (non-precond) linear solver starts");
        }
        else
        {
            LOG_INFO("PIDR(" << this->s_ << ") solver starts, with preconditioner:");
            this->precond_->Print();
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void IDR<OperatorType, VectorType, ValueType>::PrintEnd_(void) const
    {
        if(this->precond_ == NULL)
        {
            LOG_INFO("IDR(" << this->s_ << ") (non-precond) ends");
        }
        else
        {
            LOG_INFO("PIDR(" << this->s_ << ") ends");
        }
    }
    // LCOV_EXCL_STOP

    template <class OperatorType, class VectorType, typename ValueType>
    void IDR<OperatorType, VectorType, ValueType>::Build(void)
    {
        log_debug(this, "IDR::Build()", this->build_, " #*# begin");

        if(this->build_ == true)
        {
            this->Clear();
        }

        assert(this->build_ == false);
        assert(this->op_ != NULL);
        assert(this->op_->GetM() == this->op_->GetN());
        assert(this->op_->GetM() > 0);
        assert(static_cast<int64_t>(this->s_) <= this->op_->GetM());

        this->r_.CloneBackend(*this->op_);
        this->v_.CloneBackend(*this->op_);

        this->r_.Allocate("r", this->op_->GetM());
        this->v_.Allocate("v", this->op_->GetM());

        allocate_host(this->s_, &this->c_);
        allocate_host(this->s_, &this->f_);
        allocate_host(this->s_ * this->s_, &this->M_);

        this->G_ = new VectorType*[this->s_];
        this->U_ = new VectorType*[this->s_];
        this->P_ = new VectorType*[this->s_];

        for(int i = 0; i < this->s_; ++i)
        {
            this->G_[i] = new VectorType;
            this->U_[i] = new VectorType;
            this->P_[i] = new VectorType;

            this->G_[i]->CloneBackend(*this->op_);
            this->U_[i]->CloneBackend(*this->op_);
            this->P_[i]->CloneBackend(*this->op_);

            this->G_[i]->Allocate("g", this->op_->GetM());
            this->U_[i]->Allocate("u", this->op_->GetM());
            this->P_[i]->Allocate("P", this->op_->GetM());

            this->P_[i]->SetRandomNormal((i + 1) * this->seed_, 0.0, 1.0);
        }

        if(this->precond_ != NULL)
        {
            this->precond_->SetOperator(*this->op_);

            this->precond_->Build();

            this->t_.CloneBackend(*this->op_);
            this->t_.Allocate("t", this->op_->GetM());
        }

        // Build ONB out of P using modified Gram-Schmidt algorithm
        for(int k = 0; k < this->s_; ++k)
        {
            this->P_[k]->Scale(static_cast<ValueType>(1) / this->P_[k]->Norm());
            ValueType invdotk = static_cast<ValueType>(1) / this->P_[k]->Dot(*this->P_[k]);
            for(int j = k + 1; j < this->s_; ++j)
            {
                this->P_[j]->AddScale(*this->P_[k], -this->P_[j]->Dot(*this->P_[k]) * invdotk);
            }
        }

        this->build_ = true;

        log_debug(this, "IDR::Build()", this->build_, " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void IDR<OperatorType, VectorType, ValueType>::Clear(void)
    {
        log_debug(this, "IDR::Clear()", this->build_);

        if(this->build_ == true)
        {
            this->r_.Clear();
            this->v_.Clear();

            for(int i = 0; i < this->s_; ++i)
            {
                delete this->U_[i];
                delete this->G_[i];
                delete this->P_[i];
            }

            delete[] this->U_;
            delete[] this->G_;
            delete[] this->P_;

            this->U_ = NULL;
            this->G_ = NULL;
            this->P_ = NULL;

            free_host(&this->c_);
            free_host(&this->f_);
            free_host(&this->M_);

            if(this->precond_ != NULL)
            {
                this->precond_->Clear();
                this->precond_ = NULL;

                this->t_.Clear();
            }

            this->iter_ctrl_.Clear();

            this->build_ = false;
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void IDR<OperatorType, VectorType, ValueType>::ReBuildNumeric(void)
    {
        log_debug(this, "IDR::ReBuildNumeric()", this->build_);

        if(this->build_ == true)
        {
            this->r_.Zeros();
            this->v_.Zeros();

            for(int i = 0; i < this->s_; ++i)
            {
                this->U_[i]->Zeros();
                this->G_[i]->Zeros();
                this->P_[i]->Zeros();
            }

            if(this->precond_ != NULL)
            {
                this->precond_->ReBuildNumeric();
                this->t_.Zeros();
            }

            this->iter_ctrl_.Clear();
        }
        else
        {
            this->Build();
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void IDR<OperatorType, VectorType, ValueType>::SetShadowSpace(int s)
    {
        log_debug(this, "IDR:SetShadowSpace()", s);

        assert(this->build_ == false);
        assert(s > 0);
        assert(this->op_ != NULL);
        assert(static_cast<int64_t>(s) <= this->op_->GetM());

        this->s_ = s;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void IDR<OperatorType, VectorType, ValueType>::SetRandomSeed(unsigned long long seed)
    {
        log_debug(this, "IDR::SetRandomSeed()", seed);

        assert(this->build_ == false);
        assert(seed > 0ULL);

        this->seed_ = seed;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void IDR<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void)
    {
        log_debug(this, "IDR::MoveToHostLocalData_()", this->build_);

        if(this->build_ == true)
        {
            this->r_.MoveToHost();
            this->v_.MoveToHost();

            for(int i = 0; i < this->s_; ++i)
            {
                this->U_[i]->MoveToHost();
                this->G_[i]->MoveToHost();
                this->P_[i]->MoveToHost();
            }

            if(this->precond_ != NULL)
            {
                this->t_.MoveToHost();
            }
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void IDR<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void)
    {
        log_debug(this, "IDR::MoveToAcceleratorLocalData_()", this->build_);

        if(this->build_ == true)
        {
            this->r_.MoveToAccelerator();
            this->v_.MoveToAccelerator();

            for(int i = 0; i < this->s_; ++i)
            {
                this->U_[i]->MoveToAccelerator();
                this->G_[i]->MoveToAccelerator();
                this->P_[i]->MoveToAccelerator();
            }

            if(this->precond_ != NULL)
            {
                this->t_.MoveToAccelerator();
            }
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void IDR<OperatorType, VectorType, ValueType>::SolveNonPrecond_(const VectorType& rhs,
                                                                    VectorType*       x)
    {
        log_debug(this, "IDR::SolveNonPrecond_()", " #*# begin", (const void*&)rhs, x);

        assert(x != NULL);
        assert(x != &rhs);
        assert(this->op_ != NULL);
        assert(this->precond_ == NULL);
        assert(this->build_ == true);

        const OperatorType* op = this->op_;

        VectorType*  r = &this->r_;
        VectorType*  v = &this->v_;
        VectorType** G = this->G_;
        VectorType** U = this->U_;
        VectorType** P = this->P_;

        int       s     = this->s_;
        ValueType zero  = static_cast<ValueType>(0);
        ValueType one   = static_cast<ValueType>(1);
        ValueType kappa = this->kappa_;

        ValueType* c = this->c_;
        ValueType* f = this->f_;
        ValueType* M = this->M_;

        ValueType alpha;
        ValueType beta;
        ValueType rho;
        ValueType omega = one;

        // Initial residual r = b - Ax
        op->Apply(*x, r);
        r->ScaleAdd(-one, rhs);

        // use for |b-Ax0|
        ValueType res_norm = this->Norm_(*r);

        if(this->iter_ctrl_.InitResidual(std::abs(res_norm)) == false)
        {
            log_debug(this, "::SolvePrecond_()", " #*# end");
            return;
        }

        // G = U = 0
        for(int i = 0; i < s; ++i)
        {
            G[i]->Zeros();
            U[i]->Zeros();

            for(int j = 0; j < s; ++j)
            {
                M[DENSE_IND(i, j, s, s)] = (i == j) ? one : zero;
            }
        }

        // IDR(s) iteration
        while(true)
        {
            // Generate rhs for small system
            // f = P^T * r
            for(int i = 0; i < s; ++i)
            {
                f[i] = P[i]->Dot(*r);
            }

            // Loop over shadow spaces
            for(int k = 0; k < s; ++k)
            {
                // v = r
                v->CopyFrom(*r);

                // Solve lower triangular system Mc = f
                for(int i = k; i < s; ++i)
                {
                    c[i] = f[i];

                    for(int j = k; j < i; ++j)
                    {
                        c[i] -= M[DENSE_IND(i, j, s, s)] * c[j];
                    }

                    c[i] /= M[DENSE_IND(i, i, s, s)];
                    v->AddScale(*G[i], -c[i]);
                }

                // U_k = omega * v + sum(c_i * U_i), i=k,...,s-1
                U[k]->ScaleAddScale(c[k], *v, omega);

                for(int i = k + 1; i < s; ++i)
                {
                    U[k]->AddScale(*U[i], c[i]);
                }

                // G_k = A U_k
                op->Apply(*U[k], G[k]);

                // Make G orthogonal to P
                for(int i = 0; i < k; ++i)
                {
                    // alpha = P^T_i * G_k / M_ii
                    alpha = P[i]->Dot(*G[k]) / M[DENSE_IND(i, i, s, s)];

                    // G_k = G_k - alpha * G_i
                    G[k]->AddScale(*G[i], -alpha);

                    // U_k = U_k - alpha * U_i
                    U[k]->AddScale(*U[i], -alpha);
                }

                // Update column k of M
                for(int i = k; i < s; ++i)
                {
                    // M_ik = P^T_i * G_k
                    M[DENSE_IND(i, k, s, s)] = P[i]->Dot(*G[k]);
                }

                // Check M_kk for zero
                if(M[DENSE_IND(k, k, s, s)] == zero)
                {
                    LOG_INFO("IDR(s) break down ; M(k,k) == 0.0");
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                // Check M_kk for NaN
                if(M[DENSE_IND(k, k, s, s)] != M[DENSE_IND(k, k, s, s)])
                {
                    LOG_INFO("IDR(s) break down ; M(k,k) == NaN");
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                // Check M_kk for zero
                if(M[DENSE_IND(k, k, s, s)] == std::numeric_limits<ValueType>::infinity())
                {
                    LOG_INFO("IDR(s) break down ; M(k,k) == inf");
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                // Make residual r orthogonal to P

                // beta = f_k / M_k_k
                beta = f[k] / M[DENSE_IND(k, k, s, s)];

                // r = r - beta * G_k
                r->AddScale(*G[k], -beta);

                // x = x + beta * U_k
                x->AddScale(*U[k], beta);

                // Residual norm
                res_norm = this->Norm_(*r);

                // Check inner loop for convergence
                if(this->iter_ctrl_.CheckResidualNoCount(std::abs(res_norm)))
                {
                    break;
                }

                // f_i = f_i - beta * M_ik
                for(int i = k + 1; i < s; ++i)
                {
                    f[i] -= beta * M[DENSE_IND(i, k, s, s)];
                }
            }

            // Check convergence
            if(this->iter_ctrl_.CheckResidual(std::abs(res_norm), this->index_))
            {
                break;
            }

            // Enter the dimension reduction step

            // v = Ar
            op->Apply(*r, v);

            // omega = (v,r) / ||v||^2
            ValueType rt = v->Dot(*r);
            ValueType nt = v->Norm();

            rt /= nt;

            rho   = std::abs(rt / res_norm);
            omega = rt / nt;

            if(rho < kappa)
            {
                omega *= kappa / rho;
            }

            // Check omega for zero
            if(omega == zero)
            {
                LOG_INFO("IDR(s) break down ; w == 0.0");
                FATAL_ERROR(__FILE__, __LINE__);
            }

            // Check omega for NaN
            if(omega != omega)
            {
                LOG_INFO("IDR(s) break down ; w == NaN");
                FATAL_ERROR(__FILE__, __LINE__);
            }

            // Check omega for inf
            if(omega == std::numeric_limits<ValueType>::infinity())
            {
                LOG_INFO("IDR(s) break down ; w == inf");
                FATAL_ERROR(__FILE__, __LINE__);
            }

            // x = x + omega * r
            x->AddScale(*r, omega);

            // r = r - omega * v
            r->AddScale(*v, -omega);

            // Residual norm to check outer loop convergence
            res_norm = this->Norm_(*r);
        }

        log_debug(this, "IDR::SolveNonPrecond_()", " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void IDR<OperatorType, VectorType, ValueType>::SolvePrecond_(const VectorType& rhs,
                                                                 VectorType*       x)
    {
        log_debug(this, "IDR::SolvePrecond_()", " #*# begin", (const void*&)rhs, x);

        assert(x != NULL);
        assert(x != &rhs);
        assert(this->op_ != NULL);
        assert(this->precond_ != NULL);
        assert(this->build_ == true);

        const OperatorType* op = this->op_;

        VectorType*  r = &this->r_;
        VectorType*  v = &this->v_;
        VectorType*  t = &this->t_;
        VectorType** G = this->G_;
        VectorType** U = this->U_;
        VectorType** P = this->P_;

        int       s     = this->s_;
        ValueType zero  = static_cast<ValueType>(0);
        ValueType one   = static_cast<ValueType>(1);
        ValueType kappa = this->kappa_;

        ValueType* c = this->c_;
        ValueType* f = this->f_;
        ValueType* M = this->M_;

        ValueType alpha;
        ValueType beta;
        ValueType rho;
        ValueType omega = one;

        // Initial residual r = b - Ax
        op->Apply(*x, r);
        r->ScaleAdd(-one, rhs);

        // use for |b-Ax0|
        ValueType res_norm = this->Norm_(*r);

        if(this->iter_ctrl_.InitResidual(std::abs(res_norm)) == false)
        {
            log_debug(this, "::SolvePrecond_()", " #*# end");
            return;
        }

        // G = U = 0
        for(int i = 0; i < s; ++i)
        {
            G[i]->Zeros();
            U[i]->Zeros();

            for(int j = 0; j < s; ++j)
            {
                M[DENSE_IND(i, j, s, s)] = (i == j) ? one : zero;
            }
        }

        // IDR(s) iteration
        while(true)
        {
            // Generate rhs for small system
            // f = P^T * r
            for(int i = 0; i < s; ++i)
            {
                f[i] = P[i]->Dot(*r);
            }

            // Loop over shadow spaces
            for(int k = 0; k < s; ++k)
            {
                // v = r
                v->CopyFrom(*r);

                // Solve lower triangular system Mc = f
                for(int i = k; i < s; ++i)
                {
                    c[i] = f[i];

                    for(int j = k; j < i; ++j)
                    {
                        c[i] -= M[DENSE_IND(i, j, s, s)] * c[j];
                    }

                    c[i] /= M[DENSE_IND(i, i, s, s)];
                    v->AddScale(*G[i], -c[i]);
                }

                // Apply preconditioner Mt = v
                this->precond_->SolveZeroSol(*v, t);

                // U_k = omega * t + sum(c_i * U_i), i=k,...,s-1
                U[k]->ScaleAddScale(c[k], *t, omega);

                for(int i = k + 1; i < s; ++i)
                {
                    U[k]->AddScale(*U[i], c[i]);
                }

                // G_k = A U_k
                op->Apply(*U[k], G[k]);

                // Make G orthogonal to P
                for(int i = 0; i < k; ++i)
                {
                    // alpha = P^T_i * G_k / M_ii
                    alpha = P[i]->Dot(*G[k]) / M[DENSE_IND(i, i, s, s)];

                    // G_k = G_k - alpha * G_i
                    G[k]->AddScale(*G[i], -alpha);

                    // U_k = U_k - alpha * U_i
                    U[k]->AddScale(*U[i], -alpha);
                }

                // Update column k of M
                for(int i = k; i < s; ++i)
                {
                    // M_ik = P^T_i * G_k
                    M[DENSE_IND(i, k, s, s)] = P[i]->Dot(*G[k]);
                }

                // Check M_kk for zero
                if(M[DENSE_IND(k, k, s, s)] == zero)
                {
                    LOG_INFO("IDR(s) break down ; M(k,k) == 0.0");
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                // Check M_kk for NaN
                if(M[DENSE_IND(k, k, s, s)] != M[DENSE_IND(k, k, s, s)])
                {
                    LOG_INFO("IDR(s) break down ; M(k,k) == NaN");
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                // Check M_kk for zero
                if(M[DENSE_IND(k, k, s, s)] == std::numeric_limits<ValueType>::infinity())
                {
                    LOG_INFO("IDR(s) break down ; M(k,k) == inf");
                    FATAL_ERROR(__FILE__, __LINE__);
                }

                // Make residual r orthogonal to P

                // beta = f_k / M_k_k
                beta = f[k] / M[DENSE_IND(k, k, s, s)];

                // r = r - beta * G_k
                r->AddScale(*G[k], -beta);

                // x = x + beta * U_k
                x->AddScale(*U[k], beta);

                // Residual norm
                res_norm = this->Norm_(*r);

                // Check inner loop for convergence
                if(this->iter_ctrl_.CheckResidualNoCount(std::abs(res_norm)))
                {
                    break;
                }

                // f_i = f_i - beta * M_ik
                for(int i = k + 1; i < s; ++i)
                {
                    f[i] -= beta * M[DENSE_IND(i, k, s, s)];
                }
            }

            // Check convergence
            if(this->iter_ctrl_.CheckResidual(std::abs(res_norm), this->index_))
            {
                break;
            }

            // Enter the dimension reduction step

            // Mv = r
            this->precond_->SolveZeroSol(*r, v);

            // t = Av
            op->Apply(*v, t);

            // omega = (t,r) / ||t||^2
            ValueType rt = t->Dot(*r);
            ValueType nt = t->Norm();

            rt /= nt;

            rho   = std::abs(rt / res_norm);
            omega = rt / nt;

            if(rho < kappa)
            {
                omega *= kappa / rho;
            }

            // Check omega for zero
            if(omega == zero)
            {
                LOG_INFO("IDR(s) break down ; w == 0.0");
                FATAL_ERROR(__FILE__, __LINE__);
            }

            // Check omega for NaN
            if(omega != omega)
            {
                LOG_INFO("IDR(s) break down ; w == NaN");
                FATAL_ERROR(__FILE__, __LINE__);
            }

            // Check omega for inf
            if(omega == std::numeric_limits<ValueType>::infinity())
            {
                LOG_INFO("IDR(s) break down ; w == inf");
                FATAL_ERROR(__FILE__, __LINE__);
            }

            // r = r - omega * t
            r->AddScale(*t, -omega);

            // x = x + omega * v
            x->AddScale(*v, omega);

            // Residual norm to check outer loop convergence
            res_norm = this->Norm_(*r);
        }

        log_debug(this, "::SolvePrecond_()", " #*# end");
    }

    template class IDR<LocalMatrix<double>, LocalVector<double>, double>;
    template class IDR<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class IDR<LocalMatrix<std::complex<double>>,
                       LocalVector<std::complex<double>>,
                       std::complex<double>>;
    template class IDR<LocalMatrix<std::complex<float>>,
                       LocalVector<std::complex<float>>,
                       std::complex<float>>;
#endif

    template class IDR<GlobalMatrix<double>, GlobalVector<double>, double>;
    template class IDR<GlobalMatrix<float>, GlobalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class IDR<GlobalMatrix<std::complex<double>>,
                       GlobalVector<std::complex<double>>,
                       std::complex<double>>;
    template class IDR<GlobalMatrix<std::complex<float>>,
                       GlobalVector<std::complex<float>>,
                       std::complex<float>>;
#endif

    template class IDR<LocalStencil<double>, LocalVector<double>, double>;
    template class IDR<LocalStencil<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class IDR<LocalStencil<std::complex<double>>,
                       LocalVector<std::complex<double>>,
                       std::complex<double>>;
    template class IDR<LocalStencil<std::complex<float>>,
                       LocalVector<std::complex<float>>,
                       std::complex<float>>;
#endif

} // namespace rocalution

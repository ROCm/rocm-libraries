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

#include "gmres.hpp"
#include "../../utils/def.hpp"
#include "../iter_ctrl.hpp"

#include "../../base/local_matrix.hpp"
#include "../../base/local_stencil.hpp"
#include "../../base/local_vector.hpp"
#include "../../base/matrix_formats_ind.hpp"

#include "../../base/global_matrix.hpp"
#include "../../base/global_vector.hpp"

#include "../../utils/allocate_free.hpp"
#include "../../utils/log.hpp"
#include "../../utils/math_functions.hpp"

#include <complex>
#include <math.h>

namespace rocalution
{

    template <class OperatorType, class VectorType, typename ValueType>
    GMRES<OperatorType, VectorType, ValueType>::GMRES()
    {
        log_debug(this, "GMRES::GMRES()", "default constructor");

        this->size_basis_ = 30;

        this->c_ = NULL;
        this->s_ = NULL;
        this->r_ = NULL;
        this->H_ = NULL;
        this->v_ = NULL;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    GMRES<OperatorType, VectorType, ValueType>::~GMRES()
    {
        log_debug(this, "GMRES::~GMRES()", "destructor");

        this->Clear();
    }

    // LCOV_EXCL_START
    template <class OperatorType, class VectorType, typename ValueType>
    void GMRES<OperatorType, VectorType, ValueType>::Print(void) const
    {
        if(this->precond_ == NULL)
        {
            LOG_INFO("GMRES solver");
        }
        else
        {
            LOG_INFO("GMRES solver, with preconditioner:");
            this->precond_->Print();
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void GMRES<OperatorType, VectorType, ValueType>::PrintStart_(void) const
    {
        if(this->precond_ == NULL)
        {
            LOG_INFO("GMRES(" << this->size_basis_ << ") (non-precond) linear solver starts");
        }
        else
        {
            LOG_INFO("GMRES(" << this->size_basis_ << ") solver starts, with preconditioner:");
            this->precond_->Print();
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void GMRES<OperatorType, VectorType, ValueType>::PrintEnd_(void) const
    {
        if(this->precond_ == NULL)
        {
            LOG_INFO("GMRES(" << this->size_basis_ << ") (non-precond) ends");
        }
        else
        {
            LOG_INFO("GMRES(" << this->size_basis_ << ") ends");
        }
    }
    // LCOV_EXCL_STOP

    template <class OperatorType, class VectorType, typename ValueType>
    void GMRES<OperatorType, VectorType, ValueType>::Build(void)
    {
        log_debug(this, "GMRES::Build()", this->build_, " #*# begin");

        if(this->build_ == true)
        {
            this->Clear();
        }

        assert(this->build_ == false);
        assert(this->op_ != NULL);
        assert(this->op_->GetM() > 0);
        assert(this->op_->GetM() == this->op_->GetN());
        assert(this->size_basis_ > 0);

        if(this->res_norm_type_ != 2)
        {
            LOG_INFO(
                "GMRES solver supports only L2 residual norm. The solver is switching to L2 norm");
            this->res_norm_type_ = 2;
        }

        allocate_host(this->size_basis_, &this->c_);
        allocate_host(this->size_basis_, &this->s_);
        allocate_host(this->size_basis_ + 1, &this->r_);
        allocate_host((this->size_basis_ + 1) * this->size_basis_, &this->H_);

        this->v_ = new VectorType*[this->size_basis_ + 1];

        for(int i = 0; i < this->size_basis_ + 1; ++i)
        {
            this->v_[i] = new VectorType;
            this->v_[i]->CloneBackend(*this->op_);
            this->v_[i]->Allocate("v", this->op_->GetM());
        }

        if(this->precond_ != NULL)
        {
            this->z_.CloneBackend(*this->op_);
            this->z_.Allocate("z", this->op_->GetM());

            this->precond_->SetOperator(*this->op_);
            this->precond_->Build();
        }

        this->build_ = true;

        log_debug(this, "GMRES::Build()", this->build_, " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void GMRES<OperatorType, VectorType, ValueType>::Clear(void)
    {
        log_debug(this, "GMRES::Clear()", this->build_);

        if(this->build_ == true)
        {
            if(this->precond_ != NULL)
            {
                this->z_.Clear();
                this->precond_->Clear();
                this->precond_ = NULL;
            }

            free_host(&this->c_);
            free_host(&this->s_);
            free_host(&this->r_);
            free_host(&this->H_);

            for(int i = 0; i < this->size_basis_ + 1; ++i)
            {
                this->v_[i]->Clear();
                delete this->v_[i];
            }
            delete[] this->v_;
            this->v_ = NULL;

            this->iter_ctrl_.Clear();

            this->build_ = false;
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void GMRES<OperatorType, VectorType, ValueType>::ReBuildNumeric(void)
    {
        log_debug(this, "GMRES::ReBuildNumeric()", this->build_);

        if(this->build_ == true)
        {
            for(int i = 0; i < this->size_basis_ + 1; ++i)
            {
                this->v_[i]->Zeros();
            }

            this->iter_ctrl_.Clear();

            if(this->precond_ != NULL)
            {
                this->z_.Zeros();
                this->precond_->ReBuildNumeric();
            }
        }
        else
        {
            this->Build();
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void GMRES<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void)
    {
        log_debug(this, "GMRES::MoveToHostLocalData_()", this->build_);

        if(this->build_ == true)
        {
            for(int i = 0; i < this->size_basis_ + 1; ++i)
            {
                this->v_[i]->MoveToHost();
            }

            if(this->precond_ != NULL)
            {
                this->z_.MoveToHost();
                this->precond_->MoveToHost();
            }
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void GMRES<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void)
    {
        log_debug(this, "GMRES::MoveToAcceleratorLocalData_()", this->build_);

        if(this->build_ == true)
        {
            for(int i = 0; i < this->size_basis_ + 1; ++i)
            {
                this->v_[i]->MoveToAccelerator();
            }

            if(this->precond_ != NULL)
            {
                this->z_.MoveToAccelerator();
                this->precond_->MoveToAccelerator();
            }
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void GMRES<OperatorType, VectorType, ValueType>::SetBasisSize(int size_basis)
    {
        log_debug(this, "GMRES:SetBasisSize()", size_basis);

        assert(size_basis > 0);
        assert(this->build_ == false);

        this->size_basis_ = size_basis;
    }

    // GMRES implementation is based on the algorithm described in the book
    // 'Templates for the Solution of Linear Systems: Building Blocks for Iterative Methods'
    // by SIAM on page 18 and modified to fit rocalution structures.
    template <class OperatorType, class VectorType, typename ValueType>
    void GMRES<OperatorType, VectorType, ValueType>::SolveNonPrecond_(const VectorType& rhs,
                                                                      VectorType*       x)
    {
        log_debug(this, "GMRES::SolveNonPrecond_()", " #*# begin", (const void*&)rhs, x);

        assert(x != NULL);
        assert(x != &rhs);
        assert(this->op_ != NULL);
        assert(this->precond_ == NULL);
        assert(this->build_ == true);
        assert(this->size_basis_ > 0);
        assert(this->res_norm_type_ == 2);

        const OperatorType* op = this->op_;

        VectorType** v = this->v_;

        ValueType* c = this->c_;
        ValueType* s = this->s_;
        ValueType* r = this->r_;
        ValueType* H = this->H_;

        ValueType one = static_cast<ValueType>(1);

        int size = this->size_basis_;

        // Initial residual
        op->Apply(*x, v[0]);
        v[0]->ScaleAdd(-one, rhs);

        // r = 0
        set_to_zero_host(size + 1, r);

        // r_0 = ||v_0||
        r[0] = this->Norm_(*v[0]);

        // Initial residual
        if(this->iter_ctrl_.InitResidual(std::abs(r[0])) == false)
        {
            log_debug(this, "GMRES::SolveNonPrecond_()", " #*# end");
            return;
        }

        while(true)
        {
            // Normalize v_0
            v[0]->Scale(one / r[0]);

            // Arnoldi iteration
            int i = 0;
            while(i < size)
            {
                // v_i+1 = Av_i
                op->Apply(*v[i], v[i + 1]);

                // Build Hessenberg matrix H
                for(int k = 0; k <= i; ++k)
                {
                    int idx = DENSE_IND(k, i, size + 1, size);
                    // H_ki = <v_k,v_i+1>
                    H[idx] = v[k]->Dot(*v[i + 1]);
                    // v_i+1 -= H_ki * v_k
                    v[i + 1]->AddScale(*v[k], -H[idx]);
                }

                // Precompute some indices
                int ii   = DENSE_IND(i, i, size + 1, size);
                int ip1i = DENSE_IND(i + 1, i, size + 1, size);

                // H_i+1i = ||v_i+1||
                H[ip1i] = this->Norm_(*v[i + 1]);

                // v_i+1 /= H_i+1i
                v[i + 1]->Scale(one / H[ip1i]);

                // Apply Givens rotation J(0),...,J(j-1) on (H(0,i),...,H(i,i))
                for(int k = 0; k < i; ++k)
                {
                    int ki   = DENSE_IND(k, i, size + 1, size);
                    int kp1i = DENSE_IND(k + 1, i, size + 1, size);
                    this->ApplyGivensRotation_(c[k], s[k], H[ki], H[kp1i]);
                }

                // Construct J(i)
                this->GenerateGivensRotation_(H[ii], H[ip1i], c[i], s[i]);

                // Apply J(i) to H(i,i) and H(i,i+1) such that H(i,i+1) = 0
                this->ApplyGivensRotation_(c[i], s[i], H[ii], H[ip1i]);

                // Apply J(i) to the norm of the residual sg[i]
                this->ApplyGivensRotation_(c[i], s[i], r[i], r[i + 1]);

                // Check convergence
                if(this->iter_ctrl_.CheckResidual(std::abs(r[++i])))
                {
                    break;
                }
            }

            // Solve upper triangular system
            for(int j = i - 1; j >= 0; --j)
            {
                r[j] /= H[DENSE_IND(j, j, size + 1, size)];

                for(int k = 0; k < j; ++k)
                {
                    r[k] -= H[DENSE_IND(k, j, size + 1, size)] * r[j];
                }
            }

            // Update solution
            x->AddScale(*v[0], r[0]);

            for(int j = 1; j < i; ++j)
            {
                x->AddScale(*v[j], r[j]);
            }

            // Compute residual v_0 = b - Ax
            op->Apply(*x, v[0]);
            v[0]->ScaleAdd(-one, rhs);

            // r = 0
            set_to_zero_host(size + 1, r);

            // r_0 = ||v_0||
            r[0] = this->Norm_(*v[0]);

            // Check convergence
            if(this->iter_ctrl_.CheckResidualNoCount(std::abs(r[0])))
            {
                break;
            }
        }

        log_debug(this, "GMRES::SolveNonPrecond_()", " #*# end");
    }

    // GMRES implementation is based on the algorithm described in the book
    // 'Templates for the Solution of Linear Systems: Building Blocks for Iterative Methods'
    // by SIAM on page 18 and modified to fit rocalution structures.
    template <class OperatorType, class VectorType, typename ValueType>
    void GMRES<OperatorType, VectorType, ValueType>::SolvePrecond_(const VectorType& rhs,
                                                                   VectorType*       x)
    {
        log_debug(this, "GMRES::SolvePrecond_()", " #*# begin", (const void*&)rhs, x);

        assert(x != NULL);
        assert(x != &rhs);
        assert(this->op_ != NULL);
        assert(this->precond_ != NULL);
        assert(this->build_ == true);
        assert(this->size_basis_ > 0);
        assert(this->res_norm_type_ == 2);

        const OperatorType* op = this->op_;

        VectorType*  z = &this->z_;
        VectorType** v = this->v_;

        ValueType* c = this->c_;
        ValueType* s = this->s_;
        ValueType* r = this->r_;
        ValueType* H = this->H_;

        ValueType one = static_cast<ValueType>(1);

        int size = this->size_basis_;

        // Initial residual
        op->Apply(*x, z);
        z->ScaleAdd(-one, rhs);

        // Solve Mv_0 = z
        this->precond_->SolveZeroSol(*z, v[0]);

        // r = 0
        set_to_zero_host(size + 1, r);

        // r_0 = ||v_0||
        r[0] = this->Norm_(*v[0]);

        // Initial residual
        if(this->iter_ctrl_.InitResidual(std::abs(r[0])) == false)
        {
            log_debug(this, "GMRES::SolvePrecond_()", " #*# end");
            return;
        }

        while(true)
        {
            // Normalize v_0
            v[0]->Scale(one / r[0]);

            // Arnoldi iteration
            int i = 0;
            while(i < size)
            {
                // z = Av_i
                op->Apply(*v[i], z);

                // Solve M v_i+1 = z
                this->precond_->SolveZeroSol(*z, v[i + 1]);

                // Build Hessenberg matrix H
                for(int k = 0; k <= i; ++k)
                {
                    int idx = DENSE_IND(k, i, size + 1, size);
                    // H_ki = <v_k,v_i+1>
                    H[idx] = v[k]->Dot(*v[i + 1]);
                    // v_i+1 -= H_ki * v_k
                    v[i + 1]->AddScale(*v[k], -H[idx]);
                }

                // Precompute some indices
                int ii   = DENSE_IND(i, i, size + 1, size);
                int ip1i = DENSE_IND(i + 1, i, size + 1, size);

                // H_i+1i = ||v_i+1||
                H[ip1i] = this->Norm_(*v[i + 1]);

                // v_i+1 /= H_i+1i
                v[i + 1]->Scale(one / H[ip1i]);

                // Apply Givens rotation J(0),...,J(j-1) on (H(0,i),...,H(i,i))
                for(int k = 0; k < i; ++k)
                {
                    int ki   = DENSE_IND(k, i, size + 1, size);
                    int kp1i = DENSE_IND(k + 1, i, size + 1, size);
                    this->ApplyGivensRotation_(c[k], s[k], H[ki], H[kp1i]);
                }

                // Construct J(i)
                this->GenerateGivensRotation_(H[ii], H[ip1i], c[i], s[i]);

                // Apply J(i) to H(i,i) and H(i,i+1) such that H(i,i+1) = 0
                this->ApplyGivensRotation_(c[i], s[i], H[ii], H[ip1i]);

                // Apply J(i) to the norm of the residual sg[i]
                this->ApplyGivensRotation_(c[i], s[i], r[i], r[i + 1]);

                // Check convergence
                if(this->iter_ctrl_.CheckResidual(std::abs(r[++i])))
                {
                    break;
                }
            }

            // Solve upper triangular system
            for(int j = i - 1; j >= 0; --j)
            {
                r[j] /= H[DENSE_IND(j, j, size + 1, size)];

                for(int k = 0; k < j; ++k)
                {
                    r[k] -= H[DENSE_IND(k, j, size + 1, size)] * r[j];
                }
            }

            // Update solution
            x->AddScale(*v[0], r[0]);

            for(int j = 1; j < i; ++j)
            {
                x->AddScale(*v[j], r[j]);
            }

            // Compute residual z = b - Ax
            op->Apply(*x, z);
            z->ScaleAdd(-one, rhs);

            // Solve Mv_0 = z
            this->precond_->SolveZeroSol(*z, v[0]);

            // r = 0
            set_to_zero_host(size + 1, r);

            // r_0 = ||v_0||
            r[0] = this->Norm_(*v[0]);

            // Check convergence
            if(this->iter_ctrl_.CheckResidualNoCount(std::abs(r[0])))
            {
                break;
            }
        }

        log_debug(this, "GMRES::SolvePrecond_()", " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void GMRES<OperatorType, VectorType, ValueType>::GenerateGivensRotation_(ValueType  dx,
                                                                             ValueType  dy,
                                                                             ValueType& c,
                                                                             ValueType& s)
    {
        ValueType zero = static_cast<ValueType>(0);
        ValueType one  = static_cast<ValueType>(1);

        if(dy == zero)
        {
            c = one;
            s = zero;
        }
        else if(dx == zero)
        {
            c = zero;
            s = one;
        }
        else if(std::abs(dy) > std::abs(dx))
        {
            ValueType tmp = dx / dy;
            s             = one / sqrt(one + tmp * tmp);
            c             = tmp * s;
        }
        else
        {
            ValueType tmp = dy / dx;
            c             = one / sqrt(one + tmp * tmp);
            s             = tmp * c;
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void GMRES<OperatorType, VectorType, ValueType>::ApplyGivensRotation_(ValueType  c,
                                                                          ValueType  s,
                                                                          ValueType& dx,
                                                                          ValueType& dy)
    {
        ValueType temp = dx;
        dx             = rocalution_conj(c) * dx + rocalution_conj(s) * dy;
        dy             = -s * temp + c * dy;
    }

    template class GMRES<LocalMatrix<double>, LocalVector<double>, double>;
    template class GMRES<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class GMRES<LocalMatrix<std::complex<double>>,
                         LocalVector<std::complex<double>>,
                         std::complex<double>>;
    template class GMRES<LocalMatrix<std::complex<float>>,
                         LocalVector<std::complex<float>>,
                         std::complex<float>>;
#endif

    template class GMRES<GlobalMatrix<double>, GlobalVector<double>, double>;
    template class GMRES<GlobalMatrix<float>, GlobalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class GMRES<GlobalMatrix<std::complex<double>>,
                         GlobalVector<std::complex<double>>,
                         std::complex<double>>;
    template class GMRES<GlobalMatrix<std::complex<float>>,
                         GlobalVector<std::complex<float>>,
                         std::complex<float>>;
#endif

    template class GMRES<LocalStencil<double>, LocalVector<double>, double>;
    template class GMRES<LocalStencil<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class GMRES<LocalStencil<std::complex<double>>,
                         LocalVector<std::complex<double>>,
                         std::complex<double>>;
    template class GMRES<LocalStencil<std::complex<float>>,
                         LocalVector<std::complex<float>>,
                         std::complex<float>>;
#endif

} // namespace rocalution

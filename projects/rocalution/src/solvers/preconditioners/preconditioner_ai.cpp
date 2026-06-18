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

#include "preconditioner_ai.hpp"
#include "../../utils/def.hpp"
#include "../solver.hpp"

#include "../../base/local_matrix.hpp"

#include "../../base/local_vector.hpp"

#include "../../utils/log.hpp"

#include <complex>
#include <math.h>

namespace rocalution
{

    template <class OperatorType, class VectorType, typename ValueType>
    AIChebyshev<OperatorType, VectorType, ValueType>::AIChebyshev()
    {
        log_debug(this, "AIChebyshev::AIChebyshev()", "default constructor");

        this->p_          = 0;
        this->lambda_min_ = static_cast<ValueType>(0);
        this->lambda_max_ = static_cast<ValueType>(0);
    }

    template <class OperatorType, class VectorType, typename ValueType>
    AIChebyshev<OperatorType, VectorType, ValueType>::~AIChebyshev()
    {
        log_debug(this, "AIChebyshev::~AIChebyshev()", "destructor");

        this->Clear();
    }

    // LCOV_EXCL_START
    template <class OperatorType, class VectorType, typename ValueType>
    void AIChebyshev<OperatorType, VectorType, ValueType>::Print(void) const
    {
        LOG_INFO("Approximate Inverse Chebyshev(" << this->p_ << ") preconditioner");

        if(this->build_ == true)
        {
            LOG_INFO("AI matrix nnz = " << this->AIChebyshev_.GetNnz());
        }
    }
    // LCOV_EXCL_STOP

    template <class OperatorType, class VectorType, typename ValueType>
    void AIChebyshev<OperatorType, VectorType, ValueType>::Set(int       p,
                                                               ValueType lambda_min,
                                                               ValueType lambda_max)
    {
        log_debug(this, "AIChebyshev::Set()", p, lambda_min, lambda_max);

        assert(p > 0);
        assert(lambda_min != static_cast<ValueType>(0));
        assert(lambda_max != static_cast<ValueType>(0));
        assert(this->build_ == false);

        this->p_          = p;
        this->lambda_min_ = lambda_min;
        this->lambda_max_ = lambda_max;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void AIChebyshev<OperatorType, VectorType, ValueType>::Build(void)
    {
        log_debug(this, "AIChebyshev::Build()", this->build_, " #*# begin");

        if(this->build_ == true)
        {
            this->Clear();
        }

        assert(this->build_ == false);
        this->build_ = true;

        assert(this->op_ != NULL);

        this->AIChebyshev_.CloneFrom(*this->op_);

        ValueType q = (static_cast<ValueType>(1) - sqrt(this->lambda_min_ / this->lambda_max_))
                      / (static_cast<ValueType>(1) + sqrt(this->lambda_min_ / this->lambda_max_));
        ValueType c = static_cast<ValueType>(1) / sqrt(this->lambda_min_ * this->lambda_max_);

        // Shifting
        // Z = 2/(beta-alpha) [A-(beta+alpha)/2]
        OperatorType Z;
        Z.CloneFrom(*this->op_);

        Z.AddScalarDiagonal(static_cast<ValueType>(-1) * (this->lambda_max_ + this->lambda_min_)
                            / (static_cast<ValueType>(2)));
        Z.ScaleDiagonal(static_cast<ValueType>(2) / (this->lambda_max_ - this->lambda_min_));

        // Chebyshev formula/series
        // ai = I c_0 / 2 + sum c_k T_k
        // Tk = 2 Z T_k-1 - T_k-2

        // 1st term
        // T_0 = I
        // ai = I c_0 / 2
        this->AIChebyshev_.AddScalarDiagonal(c / static_cast<ValueType>(2));

        OperatorType Tkm2;
        Tkm2.CloneFrom(Z);
        // 2nd term
        // T_1 = Z
        // ai = ai + c_1 Z
        c = c * static_cast<ValueType>(-1) * q;
        this->AIChebyshev_.MatrixAdd(Tkm2, static_cast<ValueType>(1), c, true);

        // T_2 = 2*Z*Z - I
        // + c (2*Z*Z - I)
        OperatorType Tkm1;
        Tkm1.CloneBackend(*this->op_);
        Tkm1.MatrixMult(Z, Z);
        Tkm1.Scale(static_cast<ValueType>(2));
        Tkm1.AddScalarDiagonal(static_cast<ValueType>(-1));

        c = c * static_cast<ValueType>(-1) * q;
        this->AIChebyshev_.MatrixAdd(Tkm1, static_cast<ValueType>(1), c, true);

        // T_k = 2 Z T_k-1 - T_k-2
        OperatorType Tk;
        Tk.CloneBackend(*this->op_);

        for(int i = 2; i <= this->p_; ++i)
        {
            Tk.MatrixMult(Z, Tkm1);
            Tk.MatrixAdd(Tkm2, static_cast<ValueType>(2), static_cast<ValueType>(-1), true);

            c = c * static_cast<ValueType>(-1) * q;
            this->AIChebyshev_.MatrixAdd(Tk, static_cast<ValueType>(1), c, true);

            if(i + 1 <= this->p_)
            {
                Tkm2.CloneFrom(Tkm1);
                Tkm1.CloneFrom(Tk);
            }
        }

        log_debug(this, "AIChebyshev::Build()", this->build_, " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void AIChebyshev<OperatorType, VectorType, ValueType>::Clear(void)
    {
        log_debug(this, "AIChebyshev::Clear()", this->build_);

        this->AIChebyshev_.Clear();
        this->build_ = false;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void AIChebyshev<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void)
    {
        log_debug(this, "AIChebyshev::MoveToHostLocalData_()", this->build_);

        this->AIChebyshev_.MoveToHost();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void AIChebyshev<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void)
    {
        log_debug(this, "AIChebyshev::MoveToAcceleratorLocalData_()", this->build_);

        this->AIChebyshev_.MoveToAccelerator();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void AIChebyshev<OperatorType, VectorType, ValueType>::Solve(const VectorType& rhs,
                                                                 VectorType*       x)
    {
        log_debug(this, "AIChebyshev::Solve()", " #*# begin");

        assert(this->build_ == true);
        assert(x != NULL);
        assert(x != &rhs);

        this->AIChebyshev_.Apply(rhs, x);

        log_debug(this, "AIChebyshev::Solve()", " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    FSAI<OperatorType, VectorType, ValueType>::FSAI()
    {
        log_debug(this, "FSAI::FSAI()", "default constructor");

        this->op_mat_format_      = false;
        this->precond_mat_format_ = CSR;
        this->format_block_dim_   = 0;

        this->matrix_power_     = 1;
        this->external_pattern_ = false;
        this->matrix_pattern_   = NULL;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    FSAI<OperatorType, VectorType, ValueType>::~FSAI()
    {
        log_debug(this, "FSAI::~FSAI()", "destructor");

        this->Clear();
        this->matrix_pattern_ = NULL;
    }

    // LCOV_EXCL_START
    template <class OperatorType, class VectorType, typename ValueType>
    void FSAI<OperatorType, VectorType, ValueType>::Print(void) const
    {
        LOG_INFO("Factorized Sparse Approximate Inverse preconditioner");

        if(this->build_ == true)
        {
            LOG_INFO("FSAI matrix nnz = " << this->FSAI_L_.GetNnz() + this->FSAI_LT_.GetNnz()
                                                 - this->FSAI_L_.GetM());
        }
    }
    // LCOV_EXCL_STOP

    template <class OperatorType, class VectorType, typename ValueType>
    void FSAI<OperatorType, VectorType, ValueType>::Set(int power)
    {
        log_debug(this, "FSAI::Set()", power);

        assert(this->build_ == false);
        assert(power > 0);

        this->matrix_power_ = power;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void FSAI<OperatorType, VectorType, ValueType>::Set(const OperatorType& pattern)
    {
        log_debug(this, "FSAI::Set()", "");

        assert(this->build_ == false);

        this->matrix_pattern_ = &pattern;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void FSAI<OperatorType, VectorType, ValueType>::Build(void)
    {
        log_debug(this, "FSAI::Build()", this->build_, " #*# begin");

        if(this->build_ == true)
        {
            this->Clear();
        }

        assert(this->build_ == false);
        this->build_ = true;

        assert(this->op_ != NULL);

        this->FSAI_L_.CloneFrom(*this->op_);
        this->FSAI_L_.FSAI(this->matrix_power_, this->matrix_pattern_);

        this->FSAI_LT_.CloneBackend(*this->op_);
        this->FSAI_L_.Transpose(&this->FSAI_LT_);

        this->t_.CloneBackend(*this->op_);
        this->t_.Allocate("temporary", this->op_->GetM());

        if(this->op_mat_format_ == true)
        {
            this->FSAI_L_.ConvertTo(this->precond_mat_format_, this->format_block_dim_);
            this->FSAI_LT_.ConvertTo(this->precond_mat_format_, this->format_block_dim_);
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void FSAI<OperatorType, VectorType, ValueType>::Clear(void)
    {
        log_debug(this, "FSAI::Clear()", this->build_);

        if(this->build_ == true)
        {
            this->FSAI_L_.Clear();
            this->FSAI_LT_.Clear();

            this->t_.Clear();

            this->op_mat_format_      = false;
            this->precond_mat_format_ = CSR;
            this->format_block_dim_   = 0;

            this->build_ = false;
        }

        log_debug(this, "FSAI::Build()", this->build_, " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void FSAI<OperatorType, VectorType, ValueType>::SetPrecondMatrixFormat(unsigned int mat_format,
                                                                           int          blockdim)
    {
        log_debug(this, "FSAI::SetPrecondMatrixFormat()", mat_format);

        this->op_mat_format_      = true;
        this->precond_mat_format_ = mat_format;
        this->format_block_dim_   = blockdim;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void FSAI<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void)
    {
        log_debug(this, "FSAI::MoveToHostLocalData_()", this->build_);

        this->FSAI_L_.MoveToHost();
        this->FSAI_LT_.MoveToHost();

        this->t_.MoveToHost();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void FSAI<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void)
    {
        log_debug(this, "FSAI::MoveToAcceleratorLocalData_()", this->build_);

        this->FSAI_L_.MoveToAccelerator();
        this->FSAI_LT_.MoveToAccelerator();

        this->t_.MoveToAccelerator();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void FSAI<OperatorType, VectorType, ValueType>::Solve(const VectorType& rhs, VectorType* x)
    {
        log_debug(this, "FSAI::Solve()", " #*# begin");

        assert(this->build_ == true);
        assert(x != NULL);
        assert(x != &rhs);

        this->FSAI_L_.Apply(rhs, &this->t_);
        this->FSAI_LT_.Apply(this->t_, x);

        log_debug(this, "FSAI::Solve()", " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    SPAI<OperatorType, VectorType, ValueType>::SPAI()
    {
        log_debug(this, "SPAI::SPAI()", "default constructor");

        this->op_mat_format_      = false;
        this->precond_mat_format_ = CSR;
        this->format_block_dim_   = 0;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    SPAI<OperatorType, VectorType, ValueType>::~SPAI()
    {
        log_debug(this, "SPAI::~SPAI()", "destructor");

        this->Clear();
    }

    // LCOV_EXCL_START
    template <class OperatorType, class VectorType, typename ValueType>
    void SPAI<OperatorType, VectorType, ValueType>::Print(void) const
    {
        LOG_INFO("SParse Approximate Inverse preconditioner");

        if(this->build_ == true)
        {
            LOG_INFO("SPAI matrix nnz = " << this->SPAI_.GetNnz());
        }
    }
    // LCOV_EXCL_STOP

    template <class OperatorType, class VectorType, typename ValueType>
    void SPAI<OperatorType, VectorType, ValueType>::Build(void)
    {
        log_debug(this, "SPAI::Build()", this->build_, " #*# begin");

        if(this->build_ == true)
        {
            this->Clear();
        }

        assert(this->build_ == false);
        this->build_ = true;

        assert(this->op_ != NULL);

        this->SPAI_.CloneFrom(*this->op_);
        this->SPAI_.SPAI();

        if(this->op_mat_format_ == true)
        {
            this->SPAI_.ConvertTo(this->precond_mat_format_, this->format_block_dim_);
        }

        log_debug(this, "SPAI::Build()", this->build_, " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void SPAI<OperatorType, VectorType, ValueType>::Clear(void)
    {
        log_debug(this, "SPAI::Clear()", this->build_);

        if(this->build_ == true)
        {
            this->SPAI_.Clear();

            this->op_mat_format_      = false;
            this->precond_mat_format_ = CSR;
            this->format_block_dim_   = 0;

            this->build_ = false;
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void SPAI<OperatorType, VectorType, ValueType>::SetPrecondMatrixFormat(unsigned int mat_format,
                                                                           int          blockdim)
    {
        log_debug(this, "SPAI::SetPrecondMatrixFormat()", mat_format);

        this->op_mat_format_      = true;
        this->precond_mat_format_ = mat_format;
        this->format_block_dim_   = blockdim;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void SPAI<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void)
    {
        log_debug(this, "SPAI::MoveToHostLocalData_()", this->build_);

        this->SPAI_.MoveToHost();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void SPAI<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void)
    {
        log_debug(this, "SPAI::MoveToAcceleratorLocalData_()", this->build_);

        this->SPAI_.MoveToAccelerator();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void SPAI<OperatorType, VectorType, ValueType>::Solve(const VectorType& rhs, VectorType* x)
    {
        log_debug(this, "SPAI::Solve()", " #*# begin");

        assert(this->build_ == true);
        assert(x != NULL);
        assert(x != &rhs);

        this->SPAI_.Apply(rhs, x);

        log_debug(this, "SPAI::Solve()", " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    TNS<OperatorType, VectorType, ValueType>::TNS()
    {
        log_debug(this, "TNS::TNS()", "default constructor");

        this->op_mat_format_      = false;
        this->precond_mat_format_ = CSR;
        this->format_block_dim_   = 0;

        this->impl_ = true;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    TNS<OperatorType, VectorType, ValueType>::~TNS()
    {
        log_debug(this, "TNS::~TNS()", "destructor");

        this->Clear();
    }

    // LCOV_EXCL_START
    template <class OperatorType, class VectorType, typename ValueType>
    void TNS<OperatorType, VectorType, ValueType>::Print(void) const
    {
        LOG_INFO("Truncated Neumann Series (TNS) Preconditioner");

        if(this->build_ == true)
        {
            if(this->impl_ == true)
            {
                LOG_INFO("Implicit TNS L matrix nnz = " << this->L_.GetNnz());
            }
            else
            {
                LOG_INFO("Explicit TNS matrix nnz = " << this->TNS_.GetNnz());
            }
        }
    }
    // LCOV_EXCL_STOP

    template <class OperatorType, class VectorType, typename ValueType>
    void TNS<OperatorType, VectorType, ValueType>::Set(bool imp)
    {
        assert(this->build_ != true);

        this->impl_ = imp;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void TNS<OperatorType, VectorType, ValueType>::Build(void)
    {
        log_debug(this, "TNS::Build()", this->build_, " #*# begin");

        if(this->build_ == true)
        {
            this->Clear();
        }

        assert(this->build_ == false);
        this->build_ = true;

        assert(this->op_ != NULL);

        if(this->impl_ == true)
        {
            // implicit computation

            this->L_.CloneBackend(*this->op_);
            this->LT_.CloneBackend(*this->op_);

            this->tmp1_.CloneBackend(*this->op_);
            this->tmp2_.CloneBackend(*this->op_);
            this->Dinv_.CloneBackend(*this->op_);

            this->op_->ExtractInverseDiagonal(&this->Dinv_);

            this->op_->ExtractL(&this->L_, false);
            this->L_.DiagonalMatrixMultR(this->Dinv_);
            this->L_.Transpose(&this->LT_);

            this->tmp1_.Allocate("tmp1 vec for TNS", this->op_->GetM());
            this->tmp2_.Allocate("tmp2 vec for TNS", this->op_->GetM());
        }
        else
        {
            // explicit computation

            OperatorType K, KT;

            this->L_.CloneBackend(*this->op_);
            this->Dinv_.CloneBackend(*this->op_);
            this->TNS_.CloneBackend(*this->op_);

            K.CloneBackend(*this->op_);
            KT.CloneBackend(*this->op_);

            this->op_->ExtractInverseDiagonal(&this->Dinv_);

            // get the diagonal but flash them to zero
            // keep the structure
            this->op_->ExtractL(&this->L_, true);
            this->L_.ScaleDiagonal(static_cast<ValueType>(0));
            this->L_.DiagonalMatrixMultR(this->Dinv_);

            K.MatrixMult(this->L_, this->L_);

            // add -I
            this->L_.AddScalarDiagonal(static_cast<ValueType>(-1));

            K.MatrixAdd(this->L_,
                        static_cast<ValueType>(1), // for L^2
                        static_cast<ValueType>(-1), // for (-I+L)
                        true);

            K.Transpose(&KT);

            KT.DiagonalMatrixMultR(this->Dinv_);

            this->TNS_.MatrixMult(KT, K);

            K.Clear();
            KT.Clear();

            this->L_.Clear();
            this->Dinv_.Clear();
        }

        if(this->op_mat_format_ == true)
        {
            this->TNS_.ConvertTo(this->precond_mat_format_, this->format_block_dim_);
            this->L_.ConvertTo(this->precond_mat_format_, this->format_block_dim_);
            this->LT_.ConvertTo(this->precond_mat_format_, this->format_block_dim_);
        }

        log_debug(this, "TNS::Build()", this->build_, " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void TNS<OperatorType, VectorType, ValueType>::Clear(void)
    {
        log_debug(this, "TNS::Clear()", this->build_);

        if(this->build_ == true)
        {
            this->TNS_.Clear();

            this->L_.Clear();
            this->LT_.Clear();
            this->Dinv_.Clear();

            this->tmp1_.Clear();
            this->tmp2_.Clear();

            this->op_mat_format_      = false;
            this->precond_mat_format_ = CSR;
            this->format_block_dim_   = 0;

            this->build_ = false;
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void TNS<OperatorType, VectorType, ValueType>::SetPrecondMatrixFormat(unsigned int mat_format,
                                                                          int          blockdim)
    {
        log_debug(this, "TNS::SetPrecondMatrixFormat()", mat_format);

        this->op_mat_format_      = true;
        this->precond_mat_format_ = mat_format;
        this->format_block_dim_   = blockdim;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void TNS<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void)
    {
        log_debug(this, "TNS::MoveToHostLocalData_()", this->build_);

        this->TNS_.MoveToHost();
        this->L_.MoveToHost();
        this->LT_.MoveToHost();
        this->Dinv_.MoveToHost();
        this->tmp1_.MoveToHost();
        this->tmp2_.MoveToHost();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void TNS<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void)
    {
        log_debug(this, "TNS::MoveToAcceleratorLocalData_()", this->build_);

        this->TNS_.MoveToHost();
        this->L_.MoveToAccelerator();
        this->LT_.MoveToAccelerator();
        this->Dinv_.MoveToAccelerator();
        this->tmp1_.MoveToAccelerator();
        this->tmp2_.MoveToAccelerator();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void TNS<OperatorType, VectorType, ValueType>::Solve(const VectorType& rhs, VectorType* x)
    {
        log_debug(this, "TNS::Solve()", " #*# begin");

        assert(this->build_ == true);
        assert(x != NULL);
        assert(x != &rhs);

        if(this->impl_ == true)
        {
            // implicit

            this->L_.Apply(rhs, &this->tmp1_);
            this->L_.Apply(this->tmp1_, &this->tmp2_);
            this->tmp1_.AddScale(this->tmp2_, static_cast<ValueType>(-1));

            x->CopyFrom(rhs);
            x->AddScale(this->tmp1_, static_cast<ValueType>(-1));

            x->PointWiseMult(this->Dinv_);

            this->LT_.Apply(*x, &this->tmp1_);
            this->LT_.Apply(this->tmp1_, &this->tmp2_);

            x->ScaleAdd2(static_cast<ValueType>(1),
                         this->tmp1_,
                         static_cast<ValueType>(-1),
                         this->tmp2_,
                         static_cast<ValueType>(1));
        }
        else
        {
            // explicit

            this->TNS_.Apply(rhs, x);
        }

        //  LOG_INFO(x->Norm());

        log_debug(this, "TNS::Solve()", " #*# end");
    }

    template class AIChebyshev<LocalMatrix<double>, LocalVector<double>, double>;
    template class AIChebyshev<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class AIChebyshev<LocalMatrix<std::complex<double>>,
                               LocalVector<std::complex<double>>,
                               std::complex<double>>;
    template class AIChebyshev<LocalMatrix<std::complex<float>>,
                               LocalVector<std::complex<float>>,
                               std::complex<float>>;
#endif

    template class FSAI<LocalMatrix<double>, LocalVector<double>, double>;
    template class FSAI<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class FSAI<LocalMatrix<std::complex<double>>,
                        LocalVector<std::complex<double>>,
                        std::complex<double>>;
    template class FSAI<LocalMatrix<std::complex<float>>,
                        LocalVector<std::complex<float>>,
                        std::complex<float>>;
#endif

    template class SPAI<LocalMatrix<double>, LocalVector<double>, double>;
    template class SPAI<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class SPAI<LocalMatrix<std::complex<double>>,
                        LocalVector<std::complex<double>>,
                        std::complex<double>>;
    template class SPAI<LocalMatrix<std::complex<float>>,
                        LocalVector<std::complex<float>>,
                        std::complex<float>>;
#endif

    template class TNS<LocalMatrix<double>, LocalVector<double>, double>;
    template class TNS<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class TNS<LocalMatrix<std::complex<double>>,
                       LocalVector<std::complex<double>>,
                       std::complex<double>>;
    template class TNS<LocalMatrix<std::complex<float>>,
                       LocalVector<std::complex<float>>,
                       std::complex<float>>;
#endif

} // namespace rocalution

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

#include "mixed_precision.hpp"
#include "../utils/def.hpp"
#include "iter_ctrl.hpp"
#include "rocalution/utils/types.hpp"

#include "../base/local_matrix.hpp"

#include "../base/local_vector.hpp"

#include "../utils/allocate_free.hpp"
#include "../utils/log.hpp"

#include <math.h>

namespace rocalution
{

    template <class OperatorTypeH,
              class VectorTypeH,
              typename ValueTypeH,
              class OperatorTypeL,
              class VectorTypeL,
              typename ValueTypeL>
    MixedPrecisionDC<OperatorTypeH,
                     VectorTypeH,
                     ValueTypeH,
                     OperatorTypeL,
                     VectorTypeL,
                     ValueTypeL>::MixedPrecisionDC()
    {
        log_debug(this, "MixedPrecisionDC::MixedPrecisionDC()");

        this->op_l_     = NULL;
        this->Solver_L_ = NULL;
    }

    template <class OperatorTypeH,
              class VectorTypeH,
              typename ValueTypeH,
              class OperatorTypeL,
              class VectorTypeL,
              typename ValueTypeL>
    MixedPrecisionDC<OperatorTypeH,
                     VectorTypeH,
                     ValueTypeH,
                     OperatorTypeL,
                     VectorTypeL,
                     ValueTypeL>::~MixedPrecisionDC()
    {
        log_debug(this, "MixedPrecisionDC::~MixedPrecisionDC()");

        this->Clear();
    }

    template <class OperatorTypeH,
              class VectorTypeH,
              typename ValueTypeH,
              class OperatorTypeL,
              class VectorTypeL,
              typename ValueTypeL>
    void MixedPrecisionDC<OperatorTypeH,
                          VectorTypeH,
                          ValueTypeH,
                          OperatorTypeL,
                          VectorTypeL,
                          ValueTypeL>::Set(Solver<OperatorTypeL, VectorTypeL, ValueTypeL>& Solver_L)
    {
        log_debug(this, "MixedPrecisionDC::Set()", (const void*&)Solver_L);

        this->Solver_L_ = &Solver_L;
    }

    // LCOV_EXCL_START
    template <class OperatorTypeH,
              class VectorTypeH,
              typename ValueTypeH,
              class OperatorTypeL,
              class VectorTypeL,
              typename ValueTypeL>
    void MixedPrecisionDC<OperatorTypeH,
                          VectorTypeH,
                          ValueTypeH,
                          OperatorTypeL,
                          VectorTypeL,
                          ValueTypeL>::Print(void) const
    {
        if(this->Solver_L_ == NULL)
        {
            LOG_INFO("MixedPrecisionDC solver");
        }
        else
        {
            LOG_INFO("MixedPrecisionDC solver, with solver:");
            this->Solver_L_->Print();
        }
    }

    template <class OperatorTypeH,
              class VectorTypeH,
              typename ValueTypeH,
              class OperatorTypeL,
              class VectorTypeL,
              typename ValueTypeL>
    void MixedPrecisionDC<OperatorTypeH,
                          VectorTypeH,
                          ValueTypeH,
                          OperatorTypeL,
                          VectorTypeL,
                          ValueTypeL>::PrintStart_(void) const
    {
        assert(this->Solver_L_ != NULL);

        LOG_INFO("MixedPrecisionDC [" << 8 * sizeof(ValueTypeH) << "bit-" << 8 * sizeof(ValueTypeL)
                                      << "bit] solver starts, with solver:");
        this->Solver_L_->Print();
    }

    template <class OperatorTypeH,
              class VectorTypeH,
              typename ValueTypeH,
              class OperatorTypeL,
              class VectorTypeL,
              typename ValueTypeL>
    void MixedPrecisionDC<OperatorTypeH,
                          VectorTypeH,
                          ValueTypeH,
                          OperatorTypeL,
                          VectorTypeL,
                          ValueTypeL>::PrintEnd_(void) const
    {
        LOG_INFO("MixedPrecisionDC ends");
    }
    // LCOV_EXCL_STOP

    template <class OperatorTypeH,
              class VectorTypeH,
              typename ValueTypeH,
              class OperatorTypeL,
              class VectorTypeL,
              typename ValueTypeL>
    void MixedPrecisionDC<OperatorTypeH,
                          VectorTypeH,
                          ValueTypeH,
                          OperatorTypeL,
                          VectorTypeL,
                          ValueTypeL>::Build(void)
    {
        log_debug(this, "MixedPrecisionDC::Build()", " #*# begin");

        if(this->build_ == true)
        {
            this->Clear();
        }

        assert(this->build_ == false);
        this->build_ = true;

        assert(this->Solver_L_ != NULL);
        assert(this->op_ != NULL);

        this->op_h_ = this->op_;

        assert(this->op_->GetM() == this->op_->GetN());
        assert(this->op_->GetM() > 0);

        assert(this->op_l_ == NULL);
        this->op_l_ = new OperatorTypeL;

        this->r_l_.Allocate("r_l", this->op_l_->GetM());
        this->r_h_.Allocate("r_h", this->op_h_->GetM());

        this->d_h_.Allocate("d_h", this->op_h_->GetM());
        this->d_l_.Allocate("d_l", this->op_h_->GetM());

        // TODO - ugly
        // copy the matrix

        // CSR H
        PtrType*    row_offset = NULL;
        int*        col        = NULL;
        ValueTypeH* val_h      = NULL;

        // CSR L
        ValueTypeL* val_l = NULL;

        allocate_host(this->op_h_->GetLocalM() + 1, &row_offset);
        allocate_host(this->op_h_->GetLocalNnz(), &col);
        allocate_host(this->op_h_->GetLocalNnz(), &val_l);
        allocate_host(this->op_h_->GetLocalNnz(), &val_h);

        this->op_h_->CopyToCSR(row_offset, col, val_h);

        for(int64_t i = 0; i < this->op_h_->GetNnz(); ++i)
        {
            val_l[i] = static_cast<ValueTypeL>(val_h[i]);
        }

        this->op_l_->SetDataPtrCSR(&row_offset,
                                   &col,
                                   &val_l,
                                   "Low prec Matrix",
                                   this->op_h_->GetLocalNnz(),
                                   this->op_h_->GetLocalM(),
                                   this->op_h_->GetLocalN());

        // free only the h prec values
        free_host(&val_h);

        this->Solver_L_->SetOperator(*this->op_l_);
        this->Solver_L_->Build();

        this->op_l_->MoveToAccelerator();
        this->Solver_L_->MoveToAccelerator();

        log_debug(this, "MixedPrecisionDC::Build()", " #*# end");
    }

    template <class OperatorTypeH,
              class VectorTypeH,
              typename ValueTypeH,
              class OperatorTypeL,
              class VectorTypeL,
              typename ValueTypeL>
    void MixedPrecisionDC<OperatorTypeH,
                          VectorTypeH,
                          ValueTypeH,
                          OperatorTypeL,
                          VectorTypeL,
                          ValueTypeL>::ReBuildNumeric(void)
    {
        log_debug(this, "MixedPrecisionDC::ReBuildNumeric()");

        if(this->build_ == true)
        {
            this->r_l_.Zeros();
            this->r_h_.Zeros();

            this->d_l_.Zeros();
            this->d_h_.Zeros();

            this->iter_ctrl_.Clear();

            if(this->Solver_L_ != NULL)
            {
                this->Solver_L_->ReBuildNumeric();
            }
        }
        else
        {
            this->Build();
        }
    }

    template <class OperatorTypeH,
              class VectorTypeH,
              typename ValueTypeH,
              class OperatorTypeL,
              class VectorTypeL,
              typename ValueTypeL>
    void MixedPrecisionDC<OperatorTypeH,
                          VectorTypeH,
                          ValueTypeH,
                          OperatorTypeL,
                          VectorTypeL,
                          ValueTypeL>::Clear(void)
    {
        log_debug(this, "MixedPrecisionDC::Clear()");

        if(this->build_ == true)
        {
            if(this->Solver_L_ != NULL)
            {
                this->Solver_L_->Clear();
                this->Solver_L_ = NULL;
            }

            if(this->op_l_ != NULL)
            {
                delete this->op_l_;
                this->op_l_ = NULL;
            }

            this->r_l_.Clear();
            this->r_h_.Clear();

            this->d_l_.Clear();
            this->d_h_.Clear();

            this->iter_ctrl_.Clear();

            this->build_ = false;
        }
    }

    template <class OperatorTypeH,
              class VectorTypeH,
              typename ValueTypeH,
              class OperatorTypeL,
              class VectorTypeL,
              typename ValueTypeL>
    void MixedPrecisionDC<OperatorTypeH,
                          VectorTypeH,
                          ValueTypeH,
                          OperatorTypeL,
                          VectorTypeL,
                          ValueTypeL>::MoveToHostLocalData_(void)
    {
        if(this->build_ == true)
        {
            LOG_VERBOSE_INFO(2,
                             "MixedPrecisionDC: the inner solver is always performed on the accel; "
                             "this function does nothing");
        }
    }

    template <class OperatorTypeH,
              class VectorTypeH,
              typename ValueTypeH,
              class OperatorTypeL,
              class VectorTypeL,
              typename ValueTypeL>
    void MixedPrecisionDC<OperatorTypeH,
                          VectorTypeH,
                          ValueTypeH,
                          OperatorTypeL,
                          VectorTypeL,
                          ValueTypeL>::MoveToAcceleratorLocalData_(void)
    {
        if(this->build_ == true)
        {
            LOG_VERBOSE_INFO(2,
                             "MixedPrecisionDC: the inner solver is always performed on the accel; "
                             "this function does nothing");
        }
    }

    template <class OperatorTypeH,
              class VectorTypeH,
              typename ValueTypeH,
              class OperatorTypeL,
              class VectorTypeL,
              typename ValueTypeL>
    void MixedPrecisionDC<OperatorTypeH,
                          VectorTypeH,
                          ValueTypeH,
                          OperatorTypeL,
                          VectorTypeL,
                          ValueTypeL>::SolveNonPrecond_(const VectorTypeH& rhs, VectorTypeH* x)
    {
        log_debug(this, "MixedPrecisionDC::SolveNonPrecond_()", " #*# begin", (const void*&)rhs, x);

        assert(x != NULL);
        assert(x != &rhs);
        assert(this->op_ != NULL);
        assert(this->Solver_L_ != NULL);
        assert(this->build_ == true);

        this->x_h_ = x;

        // initial residual = b - Ax
        this->op_h_->Apply(*this->x_h_, &this->r_h_);
        this->r_h_.ScaleAdd(static_cast<ValueTypeH>(-1), rhs);

        ValueTypeH res = this->Norm_(this->r_h_);

        if(this->iter_ctrl_.InitResidual(res) == false)
        {
            log_debug(this, "MixedPrecisionDC::SolveNonPrecond_()", " #*# end");
            return;
        }

        while(!this->iter_ctrl_.CheckResidual(res, this->index_))
        {
            // cast to lower precision

            // TODO
            // use template
            this->r_l_.CopyFromDouble(this->r_h_);

            this->r_l_.MoveToAccelerator();

            this->d_l_.Clear();
            this->d_l_.MoveToAccelerator();

            LOG_VERBOSE_INFO(2,
                             "MixedPrecisionDC: starting the internal solver ["
                                 << 8 * sizeof(ValueTypeL) << "bit]");

            // set the initial solution to zero
            this->d_l_.Allocate("d_l", this->r_l_.GetSize());
            this->d_l_.Zeros();
            // solver the inner problem (low)
            this->Solver_L_->Solve(this->r_l_, &this->d_l_);

            this->r_l_.Clear();
            this->r_l_.MoveToHost();
            this->d_l_.MoveToHost();

            LOG_VERBOSE_INFO(2,
                             "MixedPrecisionDC: defect correcting on the host ["
                                 << 8 * sizeof(ValueTypeH) << "bit]");

            // TODO
            // use template
            this->d_h_.CopyFromFloat(this->d_l_);

            this->x_h_->AddScale(this->d_h_, static_cast<ValueTypeH>(1));

            // initial residual = b - Ax
            this->op_h_->Apply(*this->x_h_, &this->r_h_);
            this->r_h_.ScaleAdd(static_cast<ValueTypeH>(-1), rhs);
            res = this->Norm_(this->r_h_);
        }

        log_debug(this, "MixedPrecisionDC::SolveNonPrecond_()", " #*# end");
    }

    template <class OperatorTypeH,
              class VectorTypeH,
              typename ValueTypeH,
              class OperatorTypeL,
              class VectorTypeL,
              typename ValueTypeL>
    void MixedPrecisionDC<OperatorTypeH,
                          VectorTypeH,
                          ValueTypeH,
                          OperatorTypeL,
                          VectorTypeL,
                          ValueTypeL>::SolvePrecond_(const VectorTypeH& rhs, VectorTypeH* x)
    {
        assert(x != NULL);
        assert(x != &rhs);
        assert(this->op_ != NULL);
        assert(this->build_ == true);

        LOG_INFO(
            "MixedPrecisionDC solver does not work with preconditioner. Perhaps you want to set "
            "the preconditioner to the internal solver?");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template class MixedPrecisionDC<LocalMatrix<double>,
                                    LocalVector<double>,
                                    double,
                                    LocalMatrix<float>,
                                    LocalVector<float>,
                                    float>;

} // namespace rocalution

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

#include "preconditioner_as.hpp"
#include "../../base/local_matrix.hpp"
#include "../../base/local_vector.hpp"
#include "../../utils/def.hpp"
#include "../solver.hpp"

#include "../../utils/log.hpp"

#include "preconditioner.hpp"

#include <complex>
#include <limits>

namespace rocalution
{

    template <class OperatorType, class VectorType, typename ValueType>
    AS<OperatorType, VectorType, ValueType>::AS()
    {
        log_debug(this, "AS::AS()", "default constructor");

        this->pos_        = NULL;
        this->sizes_      = NULL;
        this->num_blocks_ = 0;
        this->overlap_    = -1;

        this->local_precond_ = NULL;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    AS<OperatorType, VectorType, ValueType>::~AS()
    {
        log_debug(this, "AS::~AS()", "destructor");

        this->Clear();
    }

    // LCOV_EXCL_START
    template <class OperatorType, class VectorType, typename ValueType>
    void AS<OperatorType, VectorType, ValueType>::Print(void) const
    {
        if(this->build_ == true)
        {
            LOG_INFO("Additive Schwarz preconditioner"
                     << " number of blocks = " << this->num_blocks_
                     << "; overlap = " << this->overlap_ << "; block preconditioner:");

            this->local_precond_[0]->Print();
        }
        else
        {
            LOG_INFO("Additive Schwarz preconditioner");
        }
    }
    // LCOV_EXCL_STOP

    template <class OperatorType, class VectorType, typename ValueType>
    void AS<OperatorType, VectorType, ValueType>::Set(
        int nb, int overlap, Solver<OperatorType, VectorType, ValueType>** preconds)
    {
        log_debug(this, "AS::Set()", nb, overlap, preconds);

        if((this->build_ == true) || (this->local_precond_ != NULL))
        {
            this->Clear();
        }

        assert(nb > 0);
        assert(overlap >= 0);
        assert(preconds != NULL);

        this->num_blocks_ = nb;
        this->overlap_    = overlap;

        this->local_precond_ = new Solver<OperatorType, VectorType, ValueType>*[nb];
        this->pos_           = new int[nb];
        this->sizes_         = new int[nb];

        for(int i = 0; i < this->num_blocks_; ++i)
        {
            this->local_precond_[i] = preconds[i];
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void AS<OperatorType, VectorType, ValueType>::Build(void)
    {
        log_debug(this, "AS::Build()", this->build_, " #*# begin");

        assert(this->op_ != NULL);
        assert(this->num_blocks_ > 0);
        assert(this->overlap_ >= 0);
        assert(this->local_precond_ != NULL);
        assert(this->op_->GetLocalM() / this->num_blocks_ <= std::numeric_limits<int>::max());

        int size   = static_cast<int>(this->op_->GetLocalM() / this->num_blocks_);
        int offset = 0;

        for(int i = 0; i < this->num_blocks_; ++i)
        {
            this->pos_[i] = offset - this->overlap_;
            offset += size;
            this->sizes_[i] = size + 2 * this->overlap_;
        }

        // Built for AS and RAS
        // correct fist and last
        this->pos_[0]                       = 0;
        this->sizes_[0]                     = size + this->overlap_;
        this->sizes_[this->num_blocks_ - 1] = size + this->overlap_;

        this->weight_.MoveToHost();
        this->weight_.Allocate("Overlapping weights", this->op_->GetM());
        this->weight_.Ones();

        ValueType* ptr_w = NULL;
        this->weight_.LeaveDataPtr(&ptr_w);

        for(int i = 0; i < this->num_blocks_; ++i)
        {
            for(int j = 0; j < this->overlap_; ++j)
            {
                if(i != 0)
                {
                    ptr_w[this->pos_[i] + j] = 0.5;
                }

                if(i != this->num_blocks_ - 1)
                {
                    ptr_w[this->pos_[i] + size + j] = 0.5;
                }
            }
        }

        this->weight_.SetDataPtr(&ptr_w, "Overlapping weights", this->op_->GetLocalM());
        this->weight_.CloneBackend(*this->op_);

        this->local_mat_ = new OperatorType*[this->num_blocks_];
        this->r_         = new VectorType*[this->num_blocks_];
        this->z_         = new VectorType*[this->num_blocks_];

        for(int i = 0; i < this->num_blocks_; ++i)
        {
            this->r_[i] = new VectorType;
            this->r_[i]->CloneBackend(*this->op_);
            this->r_[i]->Allocate("AS residual vector", this->sizes_[i]);

            this->z_[i] = new VectorType;
            this->z_[i]->CloneBackend(*this->op_);
            this->z_[i]->Allocate("AS residual vector", this->sizes_[i]);

            this->local_mat_[i] = new OperatorType;
            this->local_mat_[i]->CloneBackend(*this->op_);

            this->op_->ExtractSubMatrix(this->pos_[i],
                                        this->pos_[i],
                                        this->sizes_[i],
                                        this->sizes_[i],
                                        this->local_mat_[i]);

            this->local_precond_[i]->SetOperator(*this->local_mat_[i]);
            this->local_precond_[i]->Build();
        }

        this->build_ = true;

        log_debug(this, "AS::Build()", this->build_, " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void AS<OperatorType, VectorType, ValueType>::Clear(void)
    {
        log_debug(this, "AS::Clear()", this->build_);

        if(this->build_ == true)
        {
            this->weight_.Clear();

            for(int i = 0; i < this->num_blocks_; ++i)
            {
                if(this->local_precond_[i] != NULL)
                {
                    this->local_precond_[i]->Clear();
                    this->local_precond_[i] = NULL;
                }

                this->r_[i]->Clear();
                delete this->r_[i];

                this->z_[i]->Clear();
                delete this->z_[i];

                this->local_mat_[i]->Clear();
                delete this->local_mat_[i];
            }

            delete[] this->local_precond_;
            delete[] this->r_;
            delete[] this->z_;
            delete[] this->local_mat_;

            delete[] this->pos_;
            delete[] this->sizes_;

            this->pos_   = NULL;
            this->sizes_ = NULL;

            this->num_blocks_    = 0;
            this->overlap_       = -1;
            this->local_precond_ = NULL;

            this->build_ = false;
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void AS<OperatorType, VectorType, ValueType>::Solve(const VectorType& rhs, VectorType* x)
    {
        log_debug(this, "AS::Solve_()", " #*# begin", (const void*&)rhs, x);

        assert(this->build_ == true);
        assert(x != NULL);
        assert(x != &rhs);

        for(int i = 0; i < this->num_blocks_; ++i)
        {
            this->r_[i]->CopyFrom(rhs, this->pos_[i], 0, this->sizes_[i]);
        }

        // Solve
        for(int i = 0; i < this->num_blocks_; ++i)
        {
            this->local_precond_[i]->SolveZeroSol(*this->r_[i], // rhs
                                                  this->z_[i]); // x
        }

        x->Zeros();
        for(int i = 0; i < this->num_blocks_; ++i)
        {
            x->ScaleAddScale(static_cast<ValueType>(1),
                             *this->z_[i],
                             static_cast<ValueType>(1),
                             0,
                             this->pos_[i],
                             this->sizes_[i]);
        }

        x->PointWiseMult(this->weight_);

        log_debug(this, "AS::Solve_()", " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void AS<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void)
    {
        log_debug(this, "AS::MoveToAcceleratorLocalData_()", this->build_);

        if(this->build_ == true)
        {
            this->weight_.MoveToHost();

            for(int i = 0; i < this->num_blocks_; ++i)
            {
                this->local_precond_[i]->MoveToHost();
                this->r_[i]->MoveToHost();
                this->z_[i]->MoveToHost();
                this->local_mat_[i]->MoveToHost();
            }
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void AS<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void)
    {
        log_debug(this, "AS::MoveToHostLocalData_()", this->build_);

        if(this->build_ == true)
        {
            this->weight_.MoveToAccelerator();

            for(int i = 0; i < this->num_blocks_; ++i)
            {
                this->local_precond_[i]->MoveToAccelerator();
                this->r_[i]->MoveToAccelerator();
                this->z_[i]->MoveToAccelerator();
                this->local_mat_[i]->MoveToAccelerator();
            }
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    RAS<OperatorType, VectorType, ValueType>::RAS()
    {
        log_debug(this, "RAS::RAS()", "default constructor");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    RAS<OperatorType, VectorType, ValueType>::~RAS()
    {
        log_debug(this, "RAS::~RAS()", "destructor");
    }

    // LCOV_EXCL_START
    template <class OperatorType, class VectorType, typename ValueType>
    void RAS<OperatorType, VectorType, ValueType>::Print(void) const
    {
        if(this->build_ == true)
        {
            LOG_INFO("Restricted Additive Schwarz preconditioner"
                     << " number of blocks = " << this->num_blocks_
                     << "; overlap = " << this->overlap_ << "; block preconditioner:");

            this->local_precond_[0]->Print();
        }
        else
        {
            LOG_INFO("Additive Schwarz preconditioner");
        }
    }
    // LCOV_EXCL_STOP

    template <class OperatorType, class VectorType, typename ValueType>
    void RAS<OperatorType, VectorType, ValueType>::Solve(const VectorType& rhs, VectorType* x)
    {
        log_debug(this, "RAS::Solve_()", " #*# begin", (const void*&)rhs, x);

        assert(this->build_ == true);
        assert(x != NULL);
        assert(x != &rhs);
        assert(this->op_->GetLocalM() / this->num_blocks_ <= std::numeric_limits<int>::max());

        for(int i = 0; i < this->num_blocks_; ++i)
        {
            this->r_[i]->CopyFrom(rhs, this->pos_[i], 0, this->sizes_[i]);
        }

        // Solve
        for(int i = 0; i < this->num_blocks_; ++i)
        {
            this->local_precond_[i]->SolveZeroSol(*this->r_[i], // rhs
                                                  this->z_[i]); // x
        }

        int size     = static_cast<int>(this->op_->GetLocalM() / this->num_blocks_);
        int z_offset = 0;
        for(int i = 0; i < this->num_blocks_; ++i)
        {
            x->CopyFrom(*this->z_[i], z_offset, this->pos_[i] + z_offset, size);

            z_offset = this->overlap_;
        }

        log_debug(this, "RAS::Solve_()", " #*# end");
    }

    template class AS<LocalMatrix<double>, LocalVector<double>, double>;
    template class AS<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class AS<LocalMatrix<std::complex<double>>,
                      LocalVector<std::complex<double>>,
                      std::complex<double>>;
    template class AS<LocalMatrix<std::complex<float>>,
                      LocalVector<std::complex<float>>,
                      std::complex<float>>;
#endif

    template class RAS<LocalMatrix<double>, LocalVector<double>, double>;
    template class RAS<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class RAS<LocalMatrix<std::complex<double>>,
                       LocalVector<std::complex<double>>,
                       std::complex<double>>;
    template class RAS<LocalMatrix<std::complex<float>>,
                       LocalVector<std::complex<float>>,
                       std::complex<float>>;
#endif

} // namespace rocalution

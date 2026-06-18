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

#include "base_multigrid.hpp"
#include "../../utils/def.hpp"
#include "../iter_ctrl.hpp"

#include "../../base/local_matrix.hpp"
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
    BaseMultiGrid<OperatorType, VectorType, ValueType>::BaseMultiGrid()
    {
        log_debug(this, "BaseMultiGrid::BaseMultiGrid()", "default constructor");

        this->levels_        = -1;
        this->current_level_ = 0;

        this->iter_pre_smooth_  = 1;
        this->iter_post_smooth_ = 1;

        this->scaling_ = false;

        this->op_level_ = NULL;

        this->restrict_op_level_ = NULL;
        this->prolong_op_level_  = NULL;

        this->d_level_ = NULL;
        this->r_level_ = NULL;
        this->t_level_ = NULL;
        this->s_level_ = NULL;
        this->q_level_ = NULL;

        this->solver_coarse_  = NULL;
        this->smoother_level_ = NULL;

        this->cycle_      = Vcycle;
        this->host_level_ = 0;

        this->kcycle_full_ = true;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    BaseMultiGrid<OperatorType, VectorType, ValueType>::~BaseMultiGrid()
    {
        log_debug(this, "BaseMultiGrid::~BaseMultiGrid()", "destructor");

        this->Clear();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseMultiGrid<OperatorType, VectorType, ValueType>::InitLevels(int levels)
    {
        log_debug(this, "BaseMultiGrid::InitLevels()", levels);

        assert(this->build_ == false);
        assert(levels > 0);

        this->levels_ = levels;
    }

    // LCOV_EXCL_START
    template <class OperatorType, class VectorType, typename ValueType>
    void BaseMultiGrid<OperatorType, VectorType, ValueType>::SetPreconditioner(
        Solver<OperatorType, VectorType, ValueType>& precond)
    {
        LOG_INFO("BaseMultiGrid::SetPreconditioner() Perhaps you want to set the smoothers on all "
                 "levels? use SetSmootherLevel() instead of SetPreconditioner!");
        FATAL_ERROR(__FILE__, __LINE__);
    }
    // LCOV_EXCL_STOP

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseMultiGrid<OperatorType, VectorType, ValueType>::SetSmoother(
        IterativeLinearSolver<OperatorType, VectorType, ValueType>** smoother)
    {
        log_debug(this, "BaseMultiGrid::SetSmoother()", smoother);

        //  assert(this->build_ == false); not possible due to AMG
        assert(smoother != NULL);

        this->smoother_level_ = smoother;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseMultiGrid<OperatorType, VectorType, ValueType>::SetSmootherPreIter(int iter)
    {
        log_debug(this, "BaseMultiGrid::SetSmootherPreIter()", iter);

        this->iter_pre_smooth_ = iter;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseMultiGrid<OperatorType, VectorType, ValueType>::SetSmootherPostIter(int iter)
    {
        log_debug(this, "BaseMultiGrid::SetSmootherPostIter()", iter);

        this->iter_post_smooth_ = iter;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseMultiGrid<OperatorType, VectorType, ValueType>::SetSolver(
        Solver<OperatorType, VectorType, ValueType>& solver)
    {
        log_debug(this, "BaseMultiGrid::SetSolver()", (const void*&)solver);

        //  assert(this->build_ == false); not possible due to AMG

        this->solver_coarse_ = &solver;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseMultiGrid<OperatorType, VectorType, ValueType>::SetScaling(bool scaling)
    {
        log_debug(this, "BaseMultiGrid::SetScaling()", scaling);

        // Scaling can only be enabled pre-building, as it requires additional temporary
        // storage
        if(this->build_ == true)
        {
            LOG_VERBOSE_INFO(2, "*** warning: Scaling must be set before building");
        }
        else
        {
            this->scaling_ = scaling;
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseMultiGrid<OperatorType, VectorType, ValueType>::SetHostLevels(int levels)
    {
        log_debug(this, "BaseMultiGrid::SetHostLevels()", levels);

        assert(this->build_ == true);
        assert(levels > 0);

        if(levels > this->levels_)
        {
            LOG_VERBOSE_INFO(2,
                             "*** warning: Specified number of host levels is larger than "
                             "the total number of levels");
        }

        this->host_level_ = std::min(levels, this->levels_ - 1);
        this->MoveHostLevels_();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseMultiGrid<OperatorType, VectorType, ValueType>::SetCycle(unsigned int cycle)
    {
        log_debug(this, "BaseMultiGrid::SetCycle()", cycle);

        this->cycle_ = cycle;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseMultiGrid<OperatorType, VectorType, ValueType>::SetKcycleFull(bool kcycle_full)
    {
        log_debug(this, "BaseMultiGrid::SetKcycleFull()", kcycle_full);

        this->kcycle_full_ = kcycle_full;
    }

    // LCOV_EXCL_START
    template <class OperatorType, class VectorType, typename ValueType>
    void BaseMultiGrid<OperatorType, VectorType, ValueType>::Print(void) const
    {
        LOG_INFO("MultiGrid solver");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseMultiGrid<OperatorType, VectorType, ValueType>::PrintStart_(void) const
    {
        assert(this->levels_ > 0);

        LOG_INFO("MultiGrid solver starts");
        LOG_INFO("MultiGrid Number of levels " << this->levels_);
        LOG_INFO("MultiGrid with smoother:");
        this->smoother_level_[0]->Print();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseMultiGrid<OperatorType, VectorType, ValueType>::PrintEnd_(void) const
    {
        LOG_INFO("MultiGrid ends");
    }
    // LCOV_EXCL_STOP

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseMultiGrid<OperatorType, VectorType, ValueType>::Initialize(void)
    {
        log_debug(this, "BaseMultiGrid::Initialize()", " #*# begin");

        assert(this->build_ == false);

        // Initialize smoothers
        assert(this->smoother_level_ != NULL);

        // Finest level 0
        assert(this->smoother_level_[0] != NULL);

        this->smoother_level_[0]->SetOperator(*this->op_);
        this->smoother_level_[0]->Build();
        this->smoother_level_[0]->FlagSmoother();

        // Coarse levels
        for(int i = 1; i < this->levels_ - 1; ++i)
        {
            assert(this->smoother_level_[i] != NULL);

            this->smoother_level_[i]->SetOperator(*this->op_level_[i - 1]);
            this->smoother_level_[i]->Build();
            this->smoother_level_[i]->FlagSmoother();
        }

        // Initialize coarse grid solver
        assert(this->solver_coarse_ != NULL);

        this->solver_coarse_->SetOperator(*op_level_[this->levels_ - 2]);
        this->solver_coarse_->Build();

        // Setup all temporary vectors for the cycles - needed on all levels
        this->d_level_ = new VectorType*[this->levels_];
        this->r_level_ = new VectorType*[this->levels_];
        this->t_level_ = new VectorType*[this->levels_];

        // Extra vector for scaling
        if(this->scaling_)
        {
            this->s_level_ = new VectorType*[this->levels_];

            this->s_level_[0] = new VectorType;
            this->s_level_[0]->CloneBackend(*this->op_);
            this->s_level_[0]->Allocate("temporary", this->op_->GetM());

            for(int i = 1; i < this->levels_; ++i)
            {
                this->s_level_[i] = new VectorType;
                this->s_level_[i]->CloneBackend(*this->op_level_[i - 1]);
                this->s_level_[i]->Allocate("temporary", this->op_level_[i - 1]->GetM());
            }
        }

        // Extra vector for K-cycle
        if(this->cycle_ == Kcycle)
        {
            this->q_level_ = new VectorType*[this->levels_ - 2];

            for(int i = 0; i < this->levels_ - 2; ++i)
            {
                this->q_level_[i] = new VectorType;
                this->q_level_[i]->CloneBackend(*this->op_level_[i]);
                this->q_level_[i]->Allocate("q", this->op_level_[i]->GetM());
            }
        }

        for(int i = 1; i < this->levels_; ++i)
        {
            // On finest level, we need to get the size from this->op_ instead
            this->d_level_[i] = new VectorType;
            this->d_level_[i]->CloneBackend(*this->op_level_[i - 1]);
            this->d_level_[i]->Allocate("defect correction", this->op_level_[i - 1]->GetM());

            this->r_level_[i] = new VectorType;
            this->r_level_[i]->CloneBackend(*this->op_level_[i - 1]);
            this->r_level_[i]->Allocate("residual", this->op_level_[i - 1]->GetM());

            this->t_level_[i] = new VectorType;
            this->t_level_[i]->CloneBackend(*this->op_level_[i - 1]);
            this->t_level_[i]->Allocate("temporary", this->op_level_[i - 1]->GetM());
        }

        this->r_level_[0] = new VectorType;
        this->r_level_[0]->CloneBackend(*this->op_);
        this->r_level_[0]->Allocate("residual", this->op_->GetM());

        this->t_level_[0] = new VectorType;
        this->t_level_[0]->CloneBackend(*this->op_);
        this->t_level_[0]->Allocate("temporary", this->op_->GetM());

        log_debug(this, "BaseMultiGrid::Initialize()", " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseMultiGrid<OperatorType, VectorType, ValueType>::Build(void)
    {
        log_debug(this, "BaseMultiGrid::Build()", this->build_, " #*# begin");

        if(this->build_ == true)
        {
            this->Clear();
        }

        assert(this->build_ == false);

        // Check if all required structures are filled by the user
        for(int i = 0; i < this->levels_ - 1; ++i)
        {
            assert(this->op_level_[i] != NULL);
            assert(this->smoother_level_[i] != NULL);
            assert(this->restrict_op_level_[i] != NULL);
            assert(this->prolong_op_level_[i] != NULL);
        }

        assert(this->op_ != NULL);
        assert(this->solver_coarse_ != NULL);
        assert(this->levels_ > 0);

        this->Initialize();

        this->build_ = true;

        log_debug(this, "BaseMultiGrid::Build()", this->build_, " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseMultiGrid<OperatorType, VectorType, ValueType>::Clear(void)
    {
        log_debug(this, "BaseMultiGrid::Clear()", this->build_);

        if(this->build_ == true)
        {
            this->Finalize();

            this->levels_ = -1;
            this->build_  = false;
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseMultiGrid<OperatorType, VectorType, ValueType>::Finalize(void)
    {
        log_debug(this, "BaseMultiGrid::Finalize()", this->build_, " #*# begin");

        if(this->build_ == true)
        {
            // Clear transfer mapping
            for(int i = 0; i < this->levels_ - 1; ++i)
            {
                delete this->trans_level_[i];
            }

            delete[] this->trans_level_;

            // Clear temporary VectorTypes
            for(int i = 0; i < this->levels_; ++i)
            {
                if(i > 0)
                {
                    delete this->d_level_[i];
                }
                delete this->r_level_[i];
                delete this->t_level_[i];
            }

            delete[] this->d_level_;
            delete[] this->r_level_;
            delete[] this->t_level_;

            // Clear structure for scaling
            if(this->scaling_)
            {
                for(int i = 0; i < this->levels_; ++i)
                {
                    delete this->s_level_[i];
                }

                delete[] this->s_level_;
            }

            // Clear structure for K-cycle
            if(this->cycle_ == Kcycle)
            {
                for(int i = 0; i < this->levels_ - 2; ++i)
                {
                    delete this->q_level_[i];
                }

                delete[] this->q_level_;
            }

            // Clear smoothers
            for(int i = 0; i < this->levels_ - 1; ++i)
            {
                this->smoother_level_[i]->Clear();
            }

            // Clear coarse grid solver
            this->solver_coarse_->Clear();

            // Reset iteration control
            this->iter_ctrl_.Clear();
        }

        log_debug(this, "BaseMultiGrid::Finalize()", this->build_, " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseMultiGrid<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void)
    {
        log_debug(this, "BaseMultiGrid::MoveToHostLocalData_()", this->build_);

        if(this->build_ == true)
        {
            this->r_level_[this->levels_ - 1]->MoveToHost();
            this->d_level_[this->levels_ - 1]->MoveToHost();
            this->t_level_[this->levels_ - 1]->MoveToHost();
            this->solver_coarse_->MoveToHost();

            for(int i = 0; i < this->levels_ - 1; ++i)
            {
                this->op_level_[i]->MoveToHost();
                this->smoother_level_[i]->MoveToHost();
                this->r_level_[i]->MoveToHost();
                if(i > 0)
                {
                    this->d_level_[i]->MoveToHost();
                }
                this->t_level_[i]->MoveToHost();

                this->restrict_op_level_[i]->MoveToHost();
                this->prolong_op_level_[i]->MoveToHost();
            }

            // Extra structure for scaling
            if(this->scaling_)
            {
                this->s_level_[this->levels_ - 1]->MoveToHost();

                for(int i = 0; i < this->levels_ - 1; ++i)
                {
                    this->s_level_[i]->MoveToHost();
                }
            }

            // Extra structure for K-cycle
            if(this->cycle_ == Kcycle)
            {
                for(int i = 0; i < this->levels_ - 2; ++i)
                {
                    this->q_level_[i]->MoveToHost();
                }
            }

            if(this->precond_ != NULL)
            {
                this->precond_->MoveToHost();
            }
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseMultiGrid<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void)
    {
        log_debug(this, "BaseMultiGrid::MoveToAcceleratorLocalData_()", this->build_);

        if(this->build_ == true)
        {
            // If coarsest level on accelerator
            if(this->host_level_ == 0)
            {
                this->solver_coarse_->MoveToAccelerator();
            }

            // Move operators
            for(int i = 0; i < this->levels_ - 1; ++i)
            {
                if(i < this->levels_ - this->host_level_ - 1)
                {
                    this->op_level_[i]->MoveToAccelerator();
                    this->restrict_op_level_[i]->MoveToAccelerator();
                    this->prolong_op_level_[i]->MoveToAccelerator();
                }
            }

            // Move smoothers
            for(int i = 0; i < this->levels_ - 1; ++i)
            {
                if(i < this->levels_ - this->host_level_)
                {
                    this->smoother_level_[i]->MoveToAccelerator();
                }
            }

            // Move temporary vectors
            for(int i = 0; i < this->levels_; ++i)
            {
                if(i < this->levels_ - this->host_level_)
                {
                    this->r_level_[i]->MoveToAccelerator();
                    if(i > 0)
                    {
                        this->d_level_[i]->MoveToAccelerator();
                    }
                    this->t_level_[i]->MoveToAccelerator();
                }
            }

            // Extra structure for scaling
            if(this->scaling_)
            {
                for(int i = 0; i < this->levels_; ++i)
                {
                    if(i < this->levels_ - this->host_level_)
                    {
                        this->s_level_[i]->MoveToAccelerator();
                    }
                }
            }

            // Extra structure for K-cycle
            if(this->cycle_ == Kcycle)
            {
                for(int i = 0; i < this->levels_ - 2; ++i)
                {
                    if(i < this->levels_ - this->host_level_ - 1)
                    {
                        this->q_level_[i]->MoveToAccelerator();
                    }
                }
            }

            if(this->precond_ != NULL)
            {
                this->precond_->MoveToAccelerator();
            }
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseMultiGrid<OperatorType, VectorType, ValueType>::MoveHostLevels_(void)
    {
        log_debug(this, "BaseMultiGrid::MoveHostLevels_()", this->build_);

        // Move coarse grid solver
        if(this->host_level_ != 0)
        {
            this->solver_coarse_->MoveToHost();
        }

        // Move operators
        for(int i = 0; i < this->host_level_; ++i)
        {
            int level = this->levels_ - i;

            this->op_level_[level - 2]->MoveToHost();
            this->restrict_op_level_[level - 2]->MoveToHost();
            this->prolong_op_level_[level - 2]->MoveToHost();

            // Move temporary vectors
            this->t_level_[level - 1]->MoveToHost();
            this->r_level_[level - 1]->MoveToHost();
            this->d_level_[level - 1]->MoveToHost();

            // Extra structure for scaling
            if(this->scaling_)
            {
                this->s_level_[level - 1]->MoveToHost();
            }

            if(i > 0)
            {
                // Move smoothers
                this->smoother_level_[level - 1]->MoveToHost();

                // Move K-cycle temporary vectors
                if(this->cycle_ == Kcycle)
                {
                    this->q_level_[level - 2]->MoveToHost();
                }
            }
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseMultiGrid<OperatorType, VectorType, ValueType>::Solve(const VectorType& rhs,
                                                                   VectorType*       x)
    {
        log_debug(this, "BaseMultiGrid::Solve()", " #*# begin", (const void*&)rhs, x);

        assert(this->levels_ > 1);
        assert(x != NULL);
        assert(x != &rhs);
        assert(this->op_ != NULL);
        assert(this->build_ == true);
        assert(this->precond_ == NULL);
        assert(this->solver_coarse_ != NULL);

        for(int i = 0; i < this->levels_; ++i)
        {
            if(i > 0)
            {
                assert(this->d_level_[i] != NULL);
            }
            assert(this->r_level_[i] != NULL);
            assert(this->t_level_[i] != NULL);

            if(this->scaling_)
            {
                assert(this->s_level_[i] != NULL);
            }
        }

        if(this->cycle_ == Kcycle)
        {
            for(int i = 0; i < this->levels_ - 2; ++i)
            {
                assert(this->q_level_[i] != NULL);
            }
        }

        for(int i = 0; i < this->levels_ - 1; ++i)
        {
            if(i > 0)
            {
                assert(this->op_level_[i] != NULL);
            }
            assert(this->smoother_level_[i] != NULL);

            assert(this->restrict_op_level_[i] != NULL);
            assert(this->prolong_op_level_[i] != NULL);
        }

        if(this->verb_ > 0)
        {
            this->PrintStart_();
            this->iter_ctrl_.PrintInit();
        }

        // Skip residual, if preconditioner
        if(this->is_precond_ == false)
        {
            // initial residual = b - Ax
            this->op_->Apply(*x, this->r_level_[0]);
            this->r_level_[0]->ScaleAdd(static_cast<ValueType>(-1), rhs);

            this->res_norm_ = std::abs(this->Norm_(*this->r_level_[0]));

            if(this->iter_ctrl_.InitResidual(this->res_norm_) == false)
            {
                log_debug(this, "BaseMultiGrid::Solve()", " #*# end");

                return;
            }
        }
        else
        {
            // Initialize dummy residual
            this->iter_ctrl_.InitResidual(1.0);
        }

        this->Vcycle_(rhs, x);

        // If no preconditioner, compute until convergence
        if(this->is_precond_ == false)
        {
            while(!this->iter_ctrl_.CheckResidual(this->res_norm_, this->index_))
            {
                this->Vcycle_(rhs, x);
            }
        }

        if(this->verb_ > 0)
        {
            this->iter_ctrl_.PrintStatus();
            this->PrintEnd_();
        }

        log_debug(this, "BaseMultiGrid::Solve()", " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseMultiGrid<OperatorType, VectorType, ValueType>::Restrict_(const VectorType& fine,
                                                                       VectorType*       coarse)
    {
        log_debug(this, "BaseMultiGrid::Restrict_()", (const void*&)fine, coarse);

        this->restrict_op_level_[this->current_level_]->Apply(fine, coarse);
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseMultiGrid<OperatorType, VectorType, ValueType>::Prolong_(const VectorType& coarse,
                                                                      VectorType*       fine)
    {
        log_debug(this, "BaseMultiGrid::Prolong_()", (const void*&)coarse, fine);

        this->prolong_op_level_[this->current_level_]->Apply(coarse, fine);
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseMultiGrid<OperatorType, VectorType, ValueType>::Vcycle_(const VectorType& rhs,
                                                                     VectorType*       x)
    {
        log_debug(this, "BaseMultiGrid::Vcycle_()", " #*# begin", (const void*&)rhs, x);

        // Run coarse grid solver, if coarsest grid has been reached
        if(this->current_level_ == this->levels_ - 1)
        {
            this->solver_coarse_->SolveZeroSol(rhs, x);
            return;
        }

        // Smoother on the current level
        IterativeLinearSolver<OperatorType, VectorType, ValueType>* smoother
            = this->smoother_level_[this->current_level_];

        // Operator on the current level
        const OperatorType* op
            = (this->current_level_ == 0) ? this->op_ : this->op_level_[this->current_level_ - 1];

        // Temporary vectors on the current level
        VectorType* r  = this->r_level_[this->current_level_];
        VectorType* rc = this->t_level_[this->current_level_ + 1];
        VectorType* rf = this->t_level_[this->current_level_];
        VectorType* xc = this->d_level_[this->current_level_ + 1];
        VectorType* s  = (this->scaling_) ? this->s_level_[this->current_level_] : NULL;

        // Perform cycle
        ValueType factor;
        ValueType divisor;

        // Pre-smoothing
        smoother->InitMaxIter(this->iter_pre_smooth_);
        if(this->is_precond_ || this->current_level_ != 0)
        {
            // When this AMG is a preconditioner or if we are not on the finest level,
            // we have to use a zero initial guess
            smoother->SolveZeroSol(rhs, x);
        }
        else
        {
            // For AMG as a solver, x cannot be zero
            smoother->Solve(rhs, x);
        }

        // Scaling
        if(this->scaling_ == true)
        {
            if(this->current_level_ > 0 && this->current_level_ < this->levels_ - 2
               && this->iter_pre_smooth_ > 0)
            {
                s->PointWiseMult(rhs, *x);
                factor = s->Reduce();
                op->Apply(*x, s);
                s->PointWiseMult(*x);

                divisor = s->Reduce();

                if(divisor == static_cast<ValueType>(0))
                {
                    factor = static_cast<ValueType>(1);
                }
                else
                {
                    factor /= divisor;
                }

                x->Scale(factor);
            }
        }

        // Update residual r = b - Ax
        op->Apply(*x, r);
        r->ScaleAdd(static_cast<ValueType>(-1), rhs);

        // Copy s when scaling is enabled
        if(this->scaling_ && this->current_level_ == 0)
        {
            s->CopyFrom(*r);
        }

        // Check if 'continue computation on host' flag is set for this new level
        if(this->current_level_ + 1 == this->levels_ - this->host_level_)
        {
            r->MoveToHost();
        }

        // Restrict residual vector on finest level
        this->Restrict_(*r, rc);

        if(this->current_level_ + 1 == this->levels_ - this->host_level_)
        {
            r->CloneBackend(*op);
        }

        ++this->current_level_;

        // Recursive call dependent on the
        // cycle
        switch(this->cycle_)
        {
        // V-cycle
        case 0:
            this->Vcycle_(*rc, xc);
            break;

        // W-cycle
        case 1:
            this->Wcycle_(*rc, xc);
            break;

        // K-cycle
        case 2:
            this->Kcycle_(*rc, xc);
            break;

        // LCOV_EXCL_START
        // F-cycle
        case 3:
            this->Fcycle_(*rc, xc);
            break;

        default:
            FATAL_ERROR(__FILE__, __LINE__);
            break;
            // LCOV_EXCL_STOP
        }

        --this->current_level_;

        if(this->current_level_ + 1 == this->levels_ - this->host_level_)
        {
            r->MoveToHost();
        }

        // Prolong solution vector on finest level
        this->Prolong_(*xc, r);

        // Check if 'continue computation on host' flag is set for this level
        if(this->current_level_ + 1 == this->levels_ - this->host_level_)
        {
            r->CloneBackend(*op);
        }

        // Scaling
        if(this->scaling_ == true && this->current_level_ < this->levels_ - 2)
        {
            if(this->current_level_ == 0)
            {
                s->PointWiseMult(*r);
            }
            else
            {
                s->PointWiseMult(*r, *rf);
            }

            factor = s->Reduce();

            op->Apply(*r, s);

            s->PointWiseMult(*r);

            // Check for division by zero
            divisor = s->Reduce();

            if(divisor == static_cast<ValueType>(0))
            {
                factor = static_cast<ValueType>(1);
            }
            else
            {
                factor /= divisor;
            }

            // Defect correction
            x->AddScale(*r, factor);
        }
        else
        {
            // Defect correction
            x->AddScale(*r, static_cast<ValueType>(1));
        }

        // Post-smoothing on finest level
        smoother->InitMaxIter(this->iter_post_smooth_);
        smoother->Solve(rhs, x);

        // Only update the residual, if this is not a preconditioner
        if(this->current_level_ == 0 && this->is_precond_ == false)
        {
            // Update residual
            op->Apply(*x, r);
            r->ScaleAdd(static_cast<ValueType>(-1), rhs);

            this->res_norm_ = std::abs(this->Norm_(*r));
        }

        log_debug(this, "BaseMultiGrid::Vcycle_()", " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseMultiGrid<OperatorType, VectorType, ValueType>::Wcycle_(const VectorType& rhs,
                                                                     VectorType*       x)
    {
        // gamma = 2 hardcoded
        for(int i = 0; i < 2; ++i)
        {
            this->Vcycle_(rhs, x);
        }
    }

    // LCOV_EXCL_START
    template <class OperatorType, class VectorType, typename ValueType>
    void BaseMultiGrid<OperatorType, VectorType, ValueType>::Fcycle_(const VectorType& rhs,
                                                                     VectorType*       x)
    {
        LOG_INFO("BaseMultiGrid:Fcycle_() not implemented yet");
        FATAL_ERROR(__FILE__, __LINE__);
    }
    // LCOV_EXCL_STOP

    template <class OperatorType, class VectorType, typename ValueType>
    void BaseMultiGrid<OperatorType, VectorType, ValueType>::Kcycle_(const VectorType& rhs,
                                                                     VectorType*       x)
    {
        if(this->current_level_ != 1 && this->kcycle_full_ == false)
        {
            this->Vcycle_(rhs, x);
        }
        else if(this->current_level_ < this->levels_ - 1)
        {
            VectorType* q = this->q_level_[this->current_level_ - 1];
            VectorType* r = this->t_level_[this->current_level_];

            const OperatorType* op = this->op_level_[this->current_level_ - 1];

            // Start 2 CG iterations

            ValueType rho;
            ValueType rho_old;
            ValueType alpha;

            // Cycle
            this->Vcycle_(rhs, x);

            // r = rhs
            if(r != &rhs)
            {
                r->CopyFrom(rhs);
            }

            // rho = (r,x)
            rho = r->DotNonConj(*x);

            // q = Ax
            op->Apply(*x, q);

            // alpha = rho / (x,q)
            alpha = rho / x->DotNonConj(*q);

            // r = r - alpha * q
            r->AddScale(*q, -alpha);

            // Cycle
            this->Vcycle_(*r, q);

            // rho_old = rho
            rho_old = rho;

            // rho = (r,q)
            rho = r->DotNonConj(*q);

            // r = x
            r->CopyFrom(*x);

            // r = r * rho / rho_old + q
            r->ScaleAdd(rho / rho_old, *q);

            // q = Ar
            op->Apply(*r, q);

            // x = x * alpha
            x->Scale(alpha);

            // alpha = rho / (r,q)
            alpha = rho / r->DotNonConj(*q);

            // x = x + alpha * r
            x->AddScale(*r, alpha);
        }
        else
        {
            this->solver_coarse_->SolveZeroSol(rhs, x);
        }
    }

    // LCOV_EXCL_START
    // do nothing
    template <class OperatorType, class VectorType, typename ValueType>
    void BaseMultiGrid<OperatorType, VectorType, ValueType>::SolveNonPrecond_(const VectorType& rhs,
                                                                              VectorType*       x)
    {
        LOG_INFO(
            "BaseMultiGrid:SolveNonPrecond_() this function is disabled - something is very wrong "
            "if you are calling it ...");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    // do nothing
    template <class OperatorType, class VectorType, typename ValueType>
    void BaseMultiGrid<OperatorType, VectorType, ValueType>::SolvePrecond_(const VectorType& rhs,
                                                                           VectorType*       x)
    {
        LOG_INFO(
            "BaseMultiGrid:SolvePrecond_() this function is disabled - something is very wrong if "
            "you are calling it ...");
        FATAL_ERROR(__FILE__, __LINE__);
    }
    // LCOV_EXCL_STOP

    template class BaseMultiGrid<LocalMatrix<double>, LocalVector<double>, double>;
    template class BaseMultiGrid<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class BaseMultiGrid<LocalMatrix<std::complex<double>>,
                                 LocalVector<std::complex<double>>,
                                 std::complex<double>>;
    template class BaseMultiGrid<LocalMatrix<std::complex<float>>,
                                 LocalVector<std::complex<float>>,
                                 std::complex<float>>;
#endif

    template class BaseMultiGrid<GlobalMatrix<double>, GlobalVector<double>, double>;
    template class BaseMultiGrid<GlobalMatrix<float>, GlobalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class BaseMultiGrid<GlobalMatrix<std::complex<double>>,
                                 GlobalVector<std::complex<double>>,
                                 std::complex<double>>;
    template class BaseMultiGrid<GlobalMatrix<std::complex<float>>,
                                 GlobalVector<std::complex<float>>,
                                 std::complex<float>>;
#endif

} // namespace rocalution

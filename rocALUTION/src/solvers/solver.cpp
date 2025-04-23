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

#include "solver.hpp"
#include "../utils/def.hpp"

#include "../base/local_matrix.hpp"
#include "../base/local_stencil.hpp"
#include "../base/local_vector.hpp"

#include "../base/global_matrix.hpp"
#include "../base/global_vector.hpp"

#include "../utils/log.hpp"
#include "../utils/math_functions.hpp"

#include <complex>

namespace rocalution
{
    SolverDescr::SolverDescr() {}

    SolverDescr::SolverDescr(const SolverDescr& other)
        : tri_solver_alg_(other.tri_solver_alg_)
        , itsolver_max_iter_(other.itsolver_max_iter_)
        , itsolver_tol_(other.itsolver_tol_)
        , itsolver_use_tol_(other.itsolver_use_tol_)
    {
    }

    SolverDescr::~SolverDescr() {}

    SolverDescr& SolverDescr::operator=(const SolverDescr& rhs)
    {
        if(this != &rhs)
        {
            this->tri_solver_alg_    = rhs.tri_solver_alg_;
            this->itsolver_max_iter_ = rhs.itsolver_max_iter_;
            this->itsolver_tol_      = rhs.itsolver_tol_;
            this->itsolver_use_tol_  = rhs.itsolver_use_tol_;
        }

        return *this;
    }

    void SolverDescr::SetTriSolverAlg(TriSolverAlg alg)
    {
        this->tri_solver_alg_ = alg;
    }

    TriSolverAlg SolverDescr::GetTriSolverAlg() const
    {
        return this->tri_solver_alg_;
    }

    void SolverDescr::SetIterativeSolverMaxIteration(int max_iter)
    {
        this->itsolver_max_iter_ = max_iter;
    }

    int SolverDescr::GetIterativeSolverMaxIteration() const
    {
        return this->itsolver_max_iter_;
    }

    void SolverDescr::SetIterativeSolverTolerance(double tol)
    {
        this->itsolver_tol_ = tol;
    }

    double SolverDescr::GetIterativeSolverTolerance() const
    {
        return this->itsolver_tol_;
    }

    // LCOV_EXCL_START
    void SolverDescr::Print(void) const
    {
        switch(this->tri_solver_alg_)
        {
        case TriSolverAlg_Default:
            // We do not print anything in default direct case
            break;
        case TriSolverAlg_Iterative:
            if(this->itsolver_use_tol_ == true)
            {
                LOG_INFO("TriSolverAlg = iterative (" << this->itsolver_max_iter_ << ", "
                                                      << this->itsolver_tol_ << ")");
            }
            else
            {
                LOG_INFO("TriSolverAlg = iterative (" << this->itsolver_max_iter_ << ")");
            }
            break;
        }
    }
    // LCOV_EXCL_STOP

    void SolverDescr::EnableIterativeSolverTolerance(void)
    {
        this->itsolver_use_tol_ = true;
    }

    void SolverDescr::DisableIterativeSolverTolerance(void)
    {
        this->itsolver_use_tol_ = false;
    }

    bool SolverDescr::GetIterativeSolverUseTolerance(void) const
    {
        return this->itsolver_use_tol_;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    Solver<OperatorType, VectorType, ValueType>::Solver()
    {
        log_debug(this, "Solver::Solver()");

        this->op_      = NULL;
        this->precond_ = NULL;

        this->is_precond_  = false;
        this->is_smoother_ = false;
        this->build_       = false;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    Solver<OperatorType, VectorType, ValueType>::~Solver()
    {
        log_debug(this, "Solver::~Solver()");

        // the preconditioner is defined outsite
        this->op_      = NULL;
        this->precond_ = NULL;

        this->build_ = false;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void Solver<OperatorType, VectorType, ValueType>::SetOperator(const OperatorType& op)
    {
        log_debug(this, "Solver::SetOperator()", (const void*&)op);

        assert(this->build_ == false);

        this->op_ = &op;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void Solver<OperatorType, VectorType, ValueType>::ResetOperator(const OperatorType& op)
    {
        log_debug(this, "Solver::ResetOperator()", (const void*&)op);

        // TODO
        //  assert(this->build_ != false);

        this->op_ = &op;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void Solver<OperatorType, VectorType, ValueType>::SolveZeroSol(const VectorType& rhs,
                                                                   VectorType*       x)
    {
        log_debug(this, "Solver::SolveZeroSol()", (const void*&)rhs, x);

        x->Zeros();
        this->Solve(rhs, x);
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void Solver<OperatorType, VectorType, ValueType>::Build(void)
    {
        log_debug(this, "Solver::Build()");

        // by default - nothing to build

        if(this->build_ == true)
        {
            this->Clear();
        }

        this->build_ = true;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void Solver<OperatorType, VectorType, ValueType>::BuildMoveToAcceleratorAsync(void)
    {
        // default, normal build + move to accelerator

        this->Build();
        this->MoveToAccelerator();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void Solver<OperatorType, VectorType, ValueType>::Sync(void)
    {
        // default, do nothing
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void Solver<OperatorType, VectorType, ValueType>::ReBuildNumeric(void)
    {
        log_debug(this, "Solver::ReBuildNumeric()");

        assert(this->build_ == true);

        // by default - just rebuild everything
        this->Clear();
        this->Build();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void Solver<OperatorType, VectorType, ValueType>::Clear(void)
    {
        log_debug(this, "Solver::Clear()");

        if(this->precond_ != NULL)
        {
            delete this->precond_;
        }

        this->op_      = NULL;
        this->precond_ = NULL;

        this->build_ = false;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void Solver<OperatorType, VectorType, ValueType>::MoveToHost(void)
    {
        log_debug(this, "Solver::MoveToHost()");

        if(this->permutation_.GetSize() > 0)
        {
            this->permutation_.MoveToHost();
        }

        if(this->precond_ != NULL)
        {
            this->precond_->MoveToHost();
        }

        // move all local data too
        this->MoveToHostLocalData_();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void Solver<OperatorType, VectorType, ValueType>::MoveToAccelerator(void)
    {
        log_debug(this, "Solver::MoveToAccelerator()");

        if(this->permutation_.GetSize() > 0)
        {
            this->permutation_.MoveToAccelerator();
        }

        if(this->precond_ != NULL)
        {
            this->precond_->MoveToAccelerator();
        }

        // move all local data too
        this->MoveToAcceleratorLocalData_();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void Solver<OperatorType, VectorType, ValueType>::Verbose(int verb)
    {
        log_debug(this, "Solver::Verbose()", verb);

        this->verb_ = verb;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void Solver<OperatorType, VectorType, ValueType>::SetSolverDescriptor(const SolverDescr& descr)
    {
        log_debug(this, "Solver::SetSolverDescriptor()", (const void*&)descr);

        // Cannot change solver strategy after solver has been built
        assert(this->build_ == false);

        this->solver_descr_ = descr;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    IterativeLinearSolver<OperatorType, VectorType, ValueType>::IterativeLinearSolver()
    {
        log_debug(this, "IterativeLinearSolver::IterativeLinearSolver()");

        this->verb_ = 1;

        this->res_norm_type_ = 2;
        this->index_         = -1;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    IterativeLinearSolver<OperatorType, VectorType, ValueType>::~IterativeLinearSolver()
    {
        log_debug(this, "IterativeLinearSolver::~IterativeLinearSolver()");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void IterativeLinearSolver<OperatorType, VectorType, ValueType>::Init(double abs_tol,
                                                                          double rel_tol,
                                                                          double div_tol,
                                                                          int    max_iter)
    {
        log_debug(this, "IterativeLinearSolver::Init()", abs_tol, rel_tol, div_tol, max_iter);

        this->iter_ctrl_.Init(abs_tol, rel_tol, div_tol, max_iter);
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void IterativeLinearSolver<OperatorType, VectorType, ValueType>::Init(
        double abs_tol, double rel_tol, double div_tol, int min_iter, int max_iter)
    {
        log_debug(
            this, "IterativeLinearSolver::Init()", abs_tol, rel_tol, div_tol, min_iter, max_iter);

        this->iter_ctrl_.Init(abs_tol, rel_tol, div_tol, min_iter, max_iter);
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void IterativeLinearSolver<OperatorType, VectorType, ValueType>::InitMinIter(int min_iter)
    {
        log_debug(this, "IterativeLinearSolver::InitMinIter()", min_iter);

        this->iter_ctrl_.InitMinimumIterations(min_iter);
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void IterativeLinearSolver<OperatorType, VectorType, ValueType>::InitMaxIter(int max_iter)
    {
        log_debug(this, "IterativeLinearSolver::InitMaxIter()", max_iter);

        this->iter_ctrl_.InitMaximumIterations(max_iter);
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void IterativeLinearSolver<OperatorType, VectorType, ValueType>::InitTol(double abs,
                                                                             double rel,
                                                                             double div)
    {
        log_debug(this, "IterativeLinearSolver::Init()", abs, rel, div);

        this->iter_ctrl_.InitTolerance(abs, rel, div);
    }

    template <class OperatorType, class VectorType, typename ValueType>
    int IterativeLinearSolver<OperatorType, VectorType, ValueType>::GetIterationCount(void)
    {
        log_debug(this, "IterativeLinearSolver::GetIterationCount()");

        return this->iter_ctrl_.GetIterationCount();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    double IterativeLinearSolver<OperatorType, VectorType, ValueType>::GetCurrentResidual(void)
    {
        log_debug(this, "IterativeLinearSolver::GetCurrentResidual()");

        return this->iter_ctrl_.GetCurrentResidual();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    int IterativeLinearSolver<OperatorType, VectorType, ValueType>::GetSolverStatus(void)
    {
        log_debug(this, "IterativeLinearSolver::GetSolverStatus()");

        return this->iter_ctrl_.GetSolverStatus();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    int64_t IterativeLinearSolver<OperatorType, VectorType, ValueType>::GetAmaxResidualIndex(void)
    {
        int64_t ind = this->iter_ctrl_.GetAmaxResidualIndex();
        log_debug(this, "IterativeLinearSolver::GetAmaxResidualIndex()", ind);

        if(this->res_norm_type_ != 3)
        {
            LOG_INFO(
                "Absolute maximum index of residual vector is only available when using Linf norm");
        }

        return ind;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void IterativeLinearSolver<OperatorType, VectorType, ValueType>::RecordResidualHistory(void)
    {
        log_debug(this, "IterativeLinearSolver::RecordResidualHistory()");

        this->iter_ctrl_.RecordHistory();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void IterativeLinearSolver<OperatorType, VectorType, ValueType>::RecordHistory(
        const std::string& filename) const
    {
        log_debug(this, "IterativeLinearSolver::RecordHistory()", filename);

        this->iter_ctrl_.WriteHistoryToFile(filename);
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void IterativeLinearSolver<OperatorType, VectorType, ValueType>::Verbose(int verb)
    {
        log_debug(this, "IterativeLinearSolver::Verbose()", verb);

        this->verb_ = verb;
        this->iter_ctrl_.Verbose(verb);
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void IterativeLinearSolver<OperatorType, VectorType, ValueType>::SetResidualNorm(int resnorm)
    {
        log_debug(this, "IterativeLinearSolver::SetResidualNorm()", resnorm);

        assert(resnorm == 1 || resnorm == 2 || resnorm == 3);

        this->res_norm_type_ = resnorm;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    ValueType
        IterativeLinearSolver<OperatorType, VectorType, ValueType>::Norm_(const VectorType& vec)
    {
        log_debug(this, "IterativeLinearSolver::Norm_()", (const void*&)vec, this->res_norm_type_);

        // L1 norm
        if(this->res_norm_type_ == 1)
        {
            return vec.Asum();
        }

        // L2 norm
        if(this->res_norm_type_ == 2)
        {
            return vec.Norm();
        }

        // Infinity norm
        if(this->res_norm_type_ == 3)
        {
            ValueType amax;
            this->index_ = vec.Amax(amax);
            return amax;
        }

        return 0;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void IterativeLinearSolver<OperatorType, VectorType, ValueType>::Solve(const VectorType& rhs,
                                                                           VectorType*       x)
    {
        log_debug(this, "IterativeLinearSolver::Solve()", (const void*&)rhs, x);

        assert(x != NULL);
        assert(x != &rhs);
        assert(this->op_ != NULL);
        assert(this->build_ == true);

        if(this->verb_ > 0)
        {
            this->PrintStart_();
            this->iter_ctrl_.PrintInit();
        }

        if(this->precond_ == NULL)
        {
            this->SolveNonPrecond_(rhs, x);
        }
        else
        {
            this->SolvePrecond_(rhs, x);
        }

        if(this->verb_ > 0)
        {
            this->iter_ctrl_.PrintStatus();
            this->PrintEnd_();
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void IterativeLinearSolver<OperatorType, VectorType, ValueType>::SetPreconditioner(
        Solver<OperatorType, VectorType, ValueType>& precond)
    {
        log_debug(this, "IterativeLinearSolver::SetPreconditioner()", (const void*&)precond);

        assert(this != &precond);
        this->precond_ = &precond;

        this->precond_->FlagPrecond();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    FixedPoint<OperatorType, VectorType, ValueType>::FixedPoint()
    {
        log_debug(this, "FixedPoint::FixedPoint()");

        this->omega_ = 1.0;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    FixedPoint<OperatorType, VectorType, ValueType>::~FixedPoint()
    {
        log_debug(this, "FixedPoint::~FixedPoint()");

        this->Clear();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void FixedPoint<OperatorType, VectorType, ValueType>::SetRelaxation(ValueType omega)
    {
        log_debug(this, "FixedPoint::SetRelaxation()", omega);

        this->omega_ = omega;
    }

    // LCOV_EXCL_START
    template <class OperatorType, class VectorType, typename ValueType>
    void FixedPoint<OperatorType, VectorType, ValueType>::Print(void) const
    {
        if(this->precond_ == NULL)
        {
            LOG_INFO("Fixed Point Iteration solver");
        }
        else
        {
            LOG_INFO("Fixed Point Iteration solver, with preconditioner:");
            this->precond_->Print();
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void FixedPoint<OperatorType, VectorType, ValueType>::PrintStart_(void) const
    {
        assert(this->precond_ != NULL);
        LOG_INFO("Fixed Point Iteration solver starts with");
        this->precond_->Print();
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void FixedPoint<OperatorType, VectorType, ValueType>::PrintEnd_(void) const
    {
        LOG_INFO("Fixed Point Iteration solver ends");
    }
    // LCOV_EXCL_STOP

    template <class OperatorType, class VectorType, typename ValueType>
    void FixedPoint<OperatorType, VectorType, ValueType>::ReBuildNumeric(void)
    {
        log_debug(this, "FixedPoint::ReBuildNumeric()");

        if(this->build_ == true)
        {
            this->x_old_.Zeros();
            this->x_res_.Zeros();

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
    void FixedPoint<OperatorType, VectorType, ValueType>::Clear(void)
    {
        log_debug(this, "FixedPoint::Clear()");

        if(this->build_ == true)
        {
            if(this->precond_ != NULL)
            {
                this->precond_->Clear();
                this->precond_ = NULL;
            }

            this->x_old_.Clear();
            this->x_res_.Clear();

            this->iter_ctrl_.Clear();

            this->build_ = false;
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void FixedPoint<OperatorType, VectorType, ValueType>::Build(void)
    {
        log_debug(this, "FixedPoint::Build()", "#*# begin");

        if(this->build_ == true)
        {
            this->Clear();
        }

        assert(this->build_ == false);
        assert(this->precond_ != NULL);
        assert(this->op_ != NULL);
        assert(this->op_->GetM() == this->op_->GetN());

        this->build_ = true;

        this->x_old_.CloneBackend(*this->op_);
        this->x_old_.Allocate("x_old", this->op_->GetM());

        this->x_res_.CloneBackend(*this->op_);
        this->x_res_.Allocate("x_res", this->op_->GetM());

        this->precond_->SetOperator(*this->op_);

        this->precond_->Build();

        log_debug(this, "FixedPoint::Build()", "#*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void FixedPoint<OperatorType, VectorType, ValueType>::SolveZeroSol(const VectorType& rhs,
                                                                       VectorType*       x)
    {
        log_debug(this, "FixedPoint::SolveZeroSol()", (const void*&)rhs, x);

        assert(x != NULL);
        assert(x != &rhs);
        assert(this->op_ != NULL);
        assert(this->precond_ != NULL);
        assert(this->build_ == true);

        if(this->verb_ > 0)
        {
            this->PrintStart_();
            this->iter_ctrl_.PrintInit();
        }

        this->SolveZeroSol_(rhs, x);

        if(this->verb_ > 0)
        {
            this->iter_ctrl_.PrintStatus();
            this->PrintEnd_();
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void FixedPoint<OperatorType, VectorType, ValueType>::SolveNonPrecond_(const VectorType& rhs,
                                                                           VectorType*       x)
    {
        LOG_INFO("Preconditioner for the Fixed Point method is required");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void FixedPoint<OperatorType, VectorType, ValueType>::SolvePrecond_(const VectorType& rhs,
                                                                        VectorType*       x)
    {
        log_debug(this, "FixedPoint::SolvePrecond_()", " #*# begin", (const void*&)rhs, x);

        assert(x != NULL);
        assert(x != &rhs);
        assert(this->op_ != NULL);
        assert(this->precond_ != NULL);
        assert(this->build_ == true);

        // Differentiate between smoothing and non-smoothing, as we can skip norm
        // computation in smoothing case
        if(this->is_smoother_)
        {
            // Number of smoothing steps to perform
            int steps = this->iter_ctrl_.GetMaximumIterations();

            if(steps < 1)
            {
                return;
            }

            // Feed some dummy residual to initialize IterationControl class
            this->iter_ctrl_.InitResidual(1.0);

            // Modified Richardson Iteration
            // x^(k+1) = x^k + omega * (b - Ax^k)

            // inital residual x_res = b - Ax
            this->op_->Apply(*x, &this->x_res_);
            this->x_res_.ScaleAdd(static_cast<ValueType>(-1), rhs);

            // Solve M x_old = x_res
            this->precond_->SolveZeroSol(this->x_res_, &this->x_old_);

            // x = x + omega * x_old
            x->AddScale(this->x_old_, this->omega_);

            for(int iter = 1; iter < steps; ++iter)
            {
                // x_res = b - Ax
                this->op_->Apply(*x, &this->x_res_);
                this->x_res_.ScaleAdd(static_cast<ValueType>(-1), rhs);

                // Solve M x_old = x_res
                this->precond_->SolveZeroSol(this->x_res_, &this->x_old_);

                // x = x + x_old
                x->AddScale(this->x_old_, this->omega_);
            }
        }
        else
        {
            if(this->iter_ctrl_.GetMaximumIterations() < 1)
            {
                return;
            }

            // Modified Richardson Iteration
            // x^(k+1) = x^k + omega * (b - Ax^k)

            // inital residual x_res = b - Ax
            this->op_->Apply(*x, &this->x_res_);
            this->x_res_.ScaleAdd(static_cast<ValueType>(-1), rhs);

            ValueType res = this->Norm_(this->x_res_);

            if(this->iter_ctrl_.InitResidual(std::abs(res)) == false)
            {
                log_debug(this, "FixedPoint::SolvePrecond_()", " #*# end");
                return;
            }

            while(true)
            {
                // Solve M x_old = x_res
                this->precond_->SolveZeroSol(this->x_res_, &this->x_old_);

                // x = x + omega * x_old
                x->AddScale(this->x_old_, this->omega_);

                // Check if maximum number of iterations have been reached,
                // as we do not need to compute the residual in that case.
                // This will significantly improve smoother applications.
                // Note: This check will not increment iteration count!
                if(this->iter_ctrl_.CheckMaximumIterNoCount() == true)
                {
                    break;
                }

                // x_res = b - Ax
                this->op_->Apply(*x, &this->x_res_);
                this->x_res_.ScaleAdd(static_cast<ValueType>(-1), rhs);

                res = this->Norm_(this->x_res_);

                if(this->iter_ctrl_.CheckResidual(std::abs(res), this->index_) == true)
                {
                    break;
                }
            }
        }

        log_debug(this, "FixedPoint::SolvePrecond_()", " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void FixedPoint<OperatorType, VectorType, ValueType>::SolveZeroSol_(const VectorType& rhs,
                                                                        VectorType*       x)
    {
        log_debug(this, "FixedPoint::SolveZeroSol_()", " #*# begin", (const void*&)rhs, x);

        assert(x != NULL);
        assert(x != &rhs);
        assert(this->op_ != NULL);
        assert(this->precond_ != NULL);
        assert(this->build_ == true);

        // Differentiate between smoothing and non-smoothing, as we can skip norm
        // computation in smoothing case
        if(this->is_smoother_)
        {
            // Number of smoothing steps to perform
            int steps = this->iter_ctrl_.GetMaximumIterations();

            if(steps < 1)
            {
                return;
            }

            // Feed some dummy residual to initialize IterationControl class
            this->iter_ctrl_.InitResidual(1.0);

            // Modified Richardson Iteration
            // x^(k+1) = x^k + omega * (b - Ax^k)

            // Solve M x = rhs
            this->precond_->Solve(rhs, x);

            // x *= omega
            x->Scale(this->omega_);

            // Do remaining smoothing steps
            for(int iter = 1; iter < steps; ++iter)
            {
                // x_res = rhs - Ax
                this->op_->Apply(*x, &this->x_res_);
                this->x_res_.ScaleAdd(static_cast<ValueType>(-1), rhs);

                // Solve M x_old = x_res
                this->precond_->SolveZeroSol(this->x_res_, &this->x_old_);

                // x = x + omega * x_old
                x->AddScale(this->x_old_, this->omega_);
            }
        }
        else
        {
            if(this->iter_ctrl_.GetMaximumIterations() < 1)
            {
                return;
            }

            // Modified Richardson Iteration
            // x^(k+1) = x^k + omega * (rhs - Ax^k)

            // inital residual x_res = rhs
            ValueType res = this->Norm_(rhs);

            if(this->iter_ctrl_.InitResidual(std::abs(res)) == false)
            {
                log_debug(this, "FixedPoint::SolveZeroSol_()", " #*# end");
                return;
            }

            // Solve M x_old = rhs
            this->precond_->Solve(rhs, x);

            // x *= omega
            x->Scale(this->omega_);

            if(this->iter_ctrl_.CheckMaximumIterNoCount() != true)
            {
                while(true)
                {
                    // x_res = rhs - Ax
                    this->op_->Apply(*x, &this->x_res_);
                    this->x_res_.ScaleAdd(static_cast<ValueType>(-1), rhs);

                    res = this->Norm_(this->x_res_);

                    if(this->iter_ctrl_.CheckResidual(std::abs(res), this->index_) == true)
                    {
                        break;
                    }

                    // Solve M x_old = x_res
                    this->precond_->SolveZeroSol(this->x_res_, &this->x_old_);

                    // x = x + omega * x_old
                    x->AddScale(this->x_old_, this->omega_);

                    if(this->iter_ctrl_.CheckMaximumIterNoCount() == true)
                    {
                        break;
                    }
                }
            }
        }

        log_debug(this, "FixedPoint::SolveZeroSol_()", " #*# end");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void FixedPoint<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void)
    {
        log_debug(this, "FixedPoint::MoveToHostLocalData_()");

        if(this->build_ == true)
        {
            this->x_old_.MoveToHost();
            this->x_res_.MoveToHost();
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void FixedPoint<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void)
    {
        log_debug(this, "FixedPoint::MoveToAcceleratorLocalData__()");

        if(this->build_ == true)
        {
            this->x_old_.MoveToAccelerator();
            this->x_res_.MoveToAccelerator();
        }
    }

    template <class OperatorType, class VectorType, typename ValueType>
    DirectLinearSolver<OperatorType, VectorType, ValueType>::DirectLinearSolver()
    {
        log_debug(this, "DirectLinearSolver::DirectLinearSolver()");

        this->verb_ = 1;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    DirectLinearSolver<OperatorType, VectorType, ValueType>::~DirectLinearSolver()
    {
        log_debug(this, "DirectLinearSolver::~DirectLinearSolver()");
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void DirectLinearSolver<OperatorType, VectorType, ValueType>::Verbose(int verb)
    {
        log_debug(this, "DirectLinearSolver::Verbose()", verb);

        this->verb_ = verb;
    }

    template <class OperatorType, class VectorType, typename ValueType>
    void DirectLinearSolver<OperatorType, VectorType, ValueType>::Solve(const VectorType& rhs,
                                                                        VectorType*       x)
    {
        log_debug(this, "DirectLinearSolver::Solve()", (const void*&)rhs, x);

        assert(x != NULL);
        assert(x != &rhs);
        assert(this->op_ != NULL);
        assert(this->build_ == true);

        if(this->verb_ > 0)
        {
            this->PrintStart_();
        }

        this->Solve_(rhs, x);

        if(this->verb_ > 0)
        {
            this->PrintEnd_();
        }
    }

    template class Solver<LocalMatrix<double>, LocalVector<double>, double>;
    template class Solver<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class Solver<LocalMatrix<std::complex<double>>,
                          LocalVector<std::complex<double>>,
                          std::complex<double>>;
    template class Solver<LocalMatrix<std::complex<float>>,
                          LocalVector<std::complex<float>>,
                          std::complex<float>>;
#endif

    template class Solver<GlobalMatrix<double>, GlobalVector<double>, double>;
    template class Solver<GlobalMatrix<float>, GlobalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class Solver<GlobalMatrix<std::complex<double>>,
                          GlobalVector<std::complex<double>>,
                          std::complex<double>>;
    template class Solver<GlobalMatrix<std::complex<float>>,
                          GlobalVector<std::complex<float>>,
                          std::complex<float>>;
#endif

    template class IterativeLinearSolver<LocalMatrix<double>, LocalVector<double>, double>;
    template class IterativeLinearSolver<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class IterativeLinearSolver<LocalMatrix<std::complex<double>>,
                                         LocalVector<std::complex<double>>,
                                         std::complex<double>>;
    template class IterativeLinearSolver<LocalMatrix<std::complex<float>>,
                                         LocalVector<std::complex<float>>,
                                         std::complex<float>>;
#endif

    template class IterativeLinearSolver<GlobalMatrix<double>, GlobalVector<double>, double>;
    template class IterativeLinearSolver<GlobalMatrix<float>, GlobalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class IterativeLinearSolver<GlobalMatrix<std::complex<double>>,
                                         GlobalVector<std::complex<double>>,
                                         std::complex<double>>;
    template class IterativeLinearSolver<GlobalMatrix<std::complex<float>>,
                                         GlobalVector<std::complex<float>>,
                                         std::complex<float>>;
#endif

    template class FixedPoint<LocalMatrix<double>, LocalVector<double>, double>;
    template class FixedPoint<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class FixedPoint<LocalMatrix<std::complex<double>>,
                              LocalVector<std::complex<double>>,
                              std::complex<double>>;
    template class FixedPoint<LocalMatrix<std::complex<float>>,
                              LocalVector<std::complex<float>>,
                              std::complex<float>>;
#endif

    template class FixedPoint<GlobalMatrix<double>, GlobalVector<double>, double>;
    template class FixedPoint<GlobalMatrix<float>, GlobalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class FixedPoint<GlobalMatrix<std::complex<double>>,
                              GlobalVector<std::complex<double>>,
                              std::complex<double>>;
    template class FixedPoint<GlobalMatrix<std::complex<float>>,
                              GlobalVector<std::complex<float>>,
                              std::complex<float>>;
#endif

    template class DirectLinearSolver<LocalMatrix<double>, LocalVector<double>, double>;
    template class DirectLinearSolver<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class DirectLinearSolver<LocalMatrix<std::complex<double>>,
                                      LocalVector<std::complex<double>>,
                                      std::complex<double>>;
    template class DirectLinearSolver<LocalMatrix<std::complex<float>>,
                                      LocalVector<std::complex<float>>,
                                      std::complex<float>>;
#endif

    template class Solver<LocalStencil<double>, LocalVector<double>, double>;
    template class Solver<LocalStencil<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class Solver<LocalStencil<std::complex<double>>,
                          LocalVector<std::complex<double>>,
                          std::complex<double>>;
    template class Solver<LocalStencil<std::complex<float>>,
                          LocalVector<std::complex<float>>,
                          std::complex<float>>;
#endif

    template class IterativeLinearSolver<LocalStencil<double>, LocalVector<double>, double>;
    template class IterativeLinearSolver<LocalStencil<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class IterativeLinearSolver<LocalStencil<std::complex<double>>,
                                         LocalVector<std::complex<double>>,
                                         std::complex<double>>;
    template class IterativeLinearSolver<LocalStencil<std::complex<float>>,
                                         LocalVector<std::complex<float>>,
                                         std::complex<float>>;
#endif

    template class FixedPoint<LocalStencil<double>, LocalVector<double>, double>;
    template class FixedPoint<LocalStencil<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class FixedPoint<LocalStencil<std::complex<double>>,
                              LocalVector<std::complex<double>>,
                              std::complex<double>>;
    template class FixedPoint<LocalStencil<std::complex<float>>,
                              LocalVector<std::complex<float>>,
                              std::complex<float>>;
#endif

    template class DirectLinearSolver<LocalStencil<double>, LocalVector<double>, double>;
    template class DirectLinearSolver<LocalStencil<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
    template class DirectLinearSolver<LocalStencil<std::complex<double>>,
                                      LocalVector<std::complex<double>>,
                                      std::complex<double>>;
    template class DirectLinearSolver<LocalStencil<std::complex<float>>,
                                      LocalVector<std::complex<float>>,
                                      std::complex<float>>;
#endif

} // namespace rocalution

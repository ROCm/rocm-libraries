/* ************************************************************************
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#pragma once

#include "rocalution_driver_itsolver.hpp"

template <typename T>
struct rocalution_driver_itsolver<rocalution_enum_itsolver::uaamg, T>
    : rocalution_driver_itsolver_base<rocalution_enum_itsolver::uaamg, T>
{
    static constexpr auto ITSOLVER = rocalution_enum_itsolver::uaamg;
    using params_t                 = rocalution_bench_solver_parameters;
    using traits_t                 = rocalution_driver_itsolver_traits<ITSOLVER, T>;
    using solver_t                 = typename traits_t::solver_t;
    LocalVector<T>                                             b2;
    LocalVector<T>                                             e;
    FCG<LocalMatrix<T>, LocalVector<T>, T>                     cgs;
    IterativeLinearSolver<LocalMatrix<T>, LocalVector<T>, T>** sm;
    Preconditioner<LocalMatrix<T>, LocalVector<T>, T>**        smooth;

    virtual bool
        PreprocessLinearSolve(LocalMatrix<T>&                           A,
                              LocalVector<T>&                           B,
                              LocalVector<T>&                           X,
                              solver_t&                                 solver,
                              const rocalution_bench_solver_parameters& parameters) override
    {

        const auto rebuild_numeric = parameters.Get(params_t::rebuild_numeric);
        const auto format          = parameters.Get(params_t::format);
        if(rebuild_numeric)
        {

            b2.MoveToAccelerator();
            e.MoveToAccelerator();

            b2.Allocate("b2", A.GetM());
            e.Allocate("e", A.GetN());

            if(this->cache_csr_val)
            {
                A.UpdateValuesCSR(this->cache_csr_val);
                delete[] this->cache_csr_val;
                this->cache_csr_val = nullptr;
            }

            // b2 = A * 1
            e.Ones();
            A.Apply(e, &b2);

            solver.ReBuildNumeric();
        }

        // Matrix format
        const auto blockdim = parameters.Get(params_t::blockdim);
        A.ConvertTo(format, format == BCSR ? blockdim : 1);
        return true;
    }

    virtual bool
        PostprocessLinearSolve(LocalMatrix<T>&                           A,
                               LocalVector<T>&                           B,
                               LocalVector<T>&                           X,
                               solver_t&                                 solver,
                               const rocalution_bench_solver_parameters& parameters) override
    {
        return true;
    }

    virtual bool CreatePreconditioner(LocalMatrix<T>&                           A,
                                      LocalVector<T>&                           B,
                                      LocalVector<T>&                           X,
                                      const rocalution_bench_solver_parameters& parameters) override
    {

        const auto pre_smooth  = parameters.Get(params_t::solver_pre_smooth);
        const auto post_smooth = parameters.Get(params_t::solver_post_smooth);
        const auto cycle       = parameters.Get(params_t::cycle);
        const auto scaling     = parameters.Get(params_t::solver_ordering);
        const auto format      = parameters.Get(params_t::format);

        const auto couplingStrength = parameters.Get(params_t::solver_coupling_strength);
        const auto overInterp       = parameters.Get(params_t::solver_over_interp);
        const auto coarsestLevel    = parameters.Get(params_t::solver_coarsest_level);

        const auto enum_coarsening_strategy = parameters.GetEnumCoarseningStrategy();
        if(enum_coarsening_strategy.is_invalid())
        {
            rocalution_bench_errmsg << "coarsening strategy is invalid" << std::endl;
            return false;
        }

        const auto enum_smoother = parameters.GetEnumSmoother();
        if(enum_smoother.is_invalid())
        {
            rocalution_bench_errmsg << "smoother is invalid" << std::endl;
            return false;
        }

        auto* preconditioner = new UAAMG<LocalMatrix<T>, LocalVector<T>, T>();

        preconditioner->SetCoarsestLevel(coarsestLevel);
        preconditioner->SetCycle(cycle);
        preconditioner->SetOperator(A);
        preconditioner->SetManualSmoothers(true);
        preconditioner->SetManualSolver(true);
        preconditioner->SetScaling(scaling);

        switch(enum_coarsening_strategy.value)
        {
        case rocalution_enum_coarsening_strategy::Greedy:
        {
            preconditioner->SetCoarseningStrategy(CoarseningStrategy::Greedy);
            break;
        }
        case rocalution_enum_coarsening_strategy::PMIS:
        {
            preconditioner->SetCoarseningStrategy(CoarseningStrategy::PMIS);
            break;
        }
        }

        preconditioner->SetCouplingStrength(couplingStrength);
        preconditioner->SetOverInterp(overInterp);
        preconditioner->BuildHierarchy();

        // Get number of hierarchy levels
        int levels = preconditioner->GetNumLevels();

        // Coarse grid solver
        this->cgs.Verbose(0);

        // Smoother for each level
        this->sm = new IterativeLinearSolver<LocalMatrix<T>, LocalVector<T>, T>*[levels - 1];

        this->smooth = new Preconditioner<LocalMatrix<T>, LocalVector<T>, T>*[levels - 1];

        for(int i = 0; i < levels - 1; ++i)
        {
            this->sm[i] = new FixedPoint<LocalMatrix<T>, LocalVector<T>, T>;

            switch(enum_smoother.value)
            {
            case rocalution_enum_smoother::FSAI:
            {
                this->smooth[i] = new FSAI<LocalMatrix<T>, LocalVector<T>, T>;
                break;
            }
            case rocalution_enum_smoother::ILU:
            {
                this->smooth[i] = new ILU<LocalMatrix<T>, LocalVector<T>, T>;
                break;
            }
            }
            this->sm[i]->SetPreconditioner(*(this->smooth[i]));
            this->sm[i]->Verbose(0);
        }

        preconditioner->SetSmoother(sm);
        preconditioner->SetSolver(cgs);
        preconditioner->SetSmootherPreIter(pre_smooth);
        preconditioner->SetSmootherPostIter(post_smooth);
        preconditioner->SetOperatorFormat(format, parameters.Get(params_t::blockdim));
        preconditioner->InitMaxIter(1);
        preconditioner->Verbose(0);

        this->SetPreconditioner(preconditioner);

        return true;
    }
};

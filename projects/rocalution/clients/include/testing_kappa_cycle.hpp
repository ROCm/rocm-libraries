/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TESTING_KAPPA_CYCLE_HPP
#define TESTING_KAPPA_CYCLE_HPP

#include "utility.hpp"

#include <gtest/gtest.h>
#include <memory>
#include <rocalution/rocalution.hpp>
#include <vector>

using namespace rocalution;

template <typename T>
static std::unique_ptr<BaseAMG<LocalMatrix<T>, LocalVector<T>, T>>
    kappa_cycle_create_amg(const std::string& amg)
{
    if(amg == "UAAMG")
    {
        return std::make_unique<UAAMG<LocalMatrix<T>, LocalVector<T>, T>>();
    }

    if(amg == "SAAMG")
    {
        return std::make_unique<SAAMG<LocalMatrix<T>, LocalVector<T>, T>>();
    }

    auto rs = std::make_unique<RugeStuebenAMG<LocalMatrix<T>, LocalVector<T>, T>>();
    rs->SetCoarseningStrategy(PMIS);
    rs->SetInterpolationType(ExtPI);
    rs->SetInterpolationFF1Limit(false);
    return rs;
}

template <typename T>
static void kappa_cycle_setup_amg(BaseAMG<LocalMatrix<T>, LocalVector<T>, T>& p,
                                  const LocalMatrix<T>&                       A,
                                  unsigned int                                cycle,
                                  int                                         kappa)
{
    p.SetOperator(A);
    p.SetCoarsestLevel(100);
    p.SetCycle(cycle);
    if(cycle == Kappacycle)
    {
        p.SetKappa(kappa);
    }
    p.Verbose(0);
}

template <typename T>
static void testing_kappa_cycle_run(const Arguments& argus)
{
    int         ndim                = argus.size;
    std::string amg                 = argus.precond;
    bool        disable_accelerator = !argus.use_acc;

    const double tol     = std::is_same<T, float>::value ? 1e-5 : 1e-10;
    const double err_tol = std::is_same<T, float>::value ? 5e-2 : 1e-5;

    LocalMatrix<T> A;
    LocalVector<T> x;
    LocalVector<T> b;
    LocalVector<T> e;

    int* csr_ptr = NULL;
    int* csr_col = NULL;
    T*   csr_val = NULL;

    int nrow = gen_2d_laplacian(ndim, &csr_ptr, &csr_col, &csr_val);
    int nnz  = csr_ptr[nrow];

    A.SetDataPtrCSR(&csr_ptr, &csr_col, &csr_val, "A", nnz, nrow, nrow);

    if(!disable_accelerator)
    {
        A.MoveToAccelerator();
        x.MoveToAccelerator();
        b.MoveToAccelerator();
        e.MoveToAccelerator();
    }

    x.Allocate("x", A.GetN());
    b.Allocate("b", A.GetM());
    e.Allocate("e", A.GetN());

    // b = A * 1
    e.Ones();
    A.Apply(e, &b);

    const double e_nrm = static_cast<double>(e.Norm());

    // Stand-alone AMG solve, returns the number of cycles until convergence
    int    levels = 0;
    double res    = 0.0;
    auto   solve  = [&](unsigned int cycle, int kappa) {
        auto p = kappa_cycle_create_amg<T>(amg);
        kappa_cycle_setup_amg(*p, A, cycle, kappa);
        p->Init(0.0, tol, 1e+8, 1000);
        p->Build();

        levels = p->GetNumLevels();

        x.SetRandomUniform(12345ULL, -4.0, 6.0);
        p->Solve(b, &x);

        int iter = p->GetIterationCount();
        res      = p->GetCurrentResidual();

        x.ScaleAdd(static_cast<T>(-1), e);
        EXPECT_LT(static_cast<double>(x.Norm()) / e_nrm, err_tol)
            << "cycle " << cycle << " kappa " << kappa;

        p->Clear();

        return iter;
    };

    int    iter_v = solve(Vcycle, 0);
    double res_v  = res;
    int    iter_f = solve(Fcycle, 0);
    int    iter_w = solve(Wcycle, 0);
    double res_w  = res;

    // W-cycle only differs from the V-cycle with at least 3 levels
    ASSERT_GE(levels, 3);

    // Kappa-cycle with kappa >= levels is the W-cycle
    std::vector<int> iter_kappa(levels + 1);
    for(int kappa = 1; kappa <= levels; ++kappa)
    {
        iter_kappa[kappa] = solve(Kappacycle, kappa);
    }

    EXPECT_EQ(iter_kappa[1], iter_v);
    EXPECT_EQ(iter_kappa[2], iter_f);
    EXPECT_EQ(iter_kappa[levels], iter_w);
    EXPECT_EQ(solve(Kappacycle, levels + 3), iter_w);

    for(int kappa = 2; kappa <= levels; ++kappa)
    {
        EXPECT_LE(iter_kappa[kappa], iter_kappa[kappa - 1]) << "kappa " << kappa;
    }

    // W-cycle converges in fewer cycles, or reaches a lower residual in as many cycles
    EXPECT_TRUE(iter_w < iter_v || (iter_w == iter_v && res_w < res_v))
        << "W-cycle " << iter_w << " iterations, residual " << res_w << ", V-cycle " << iter_v
        << " iterations, residual " << res_v;

    // Kappa-cycle as preconditioner of a flexible Krylov solver
    auto p = kappa_cycle_create_amg<T>(amg);
    kappa_cycle_setup_amg(*p, A, Kappacycle, 3);
    p->InitMaxIter(1);

    FCG<LocalMatrix<T>, LocalVector<T>, T> ls;
    ls.SetOperator(A);
    ls.SetPreconditioner(*p);
    ls.Init(0.0, tol, 1e+8, 1000);
    ls.Verbose(0);
    ls.Build();

    x.SetRandomUniform(12345ULL, -4.0, 6.0);
    ls.Solve(b, &x);

    x.ScaleAdd(static_cast<T>(-1), e);
    EXPECT_LT(static_cast<double>(x.Norm()) / e_nrm, err_tol);
    EXPECT_LE(ls.GetIterationCount(), iter_v);

    ls.Clear();
}

template <typename T>
void testing_kappa_cycle(Arguments argus)
{
    // Initialize rocALUTION platform
    disable_accelerator_rocalution(!argus.use_acc);
    set_device_rocalution(device);
    init_rocalution();

    testing_kappa_cycle_run<T>(argus);

    // Stop rocALUTION platform
    stop_rocalution();
    disable_accelerator_rocalution(false);
}

#endif // TESTING_KAPPA_CYCLE_HPP

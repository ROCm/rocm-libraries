/* ************************************************************************
 * Copyright (C) 2018-2022 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TESTING_PAIRWISE_AMG_HPP
#define TESTING_PAIRWISE_AMG_HPP

#include "utility.hpp"

#include <rocalution/rocalution.hpp>

using namespace rocalution;

static bool check_residual(float res)
{
    return (res < 1e-2f);
}

static bool check_residual(double res)
{
    return (res < 1e-5);
}

template <typename T>
bool testing_pairwise_amg(Arguments argus)
{
    int          ndim           = argus.size;
    int          pre_iter       = argus.pre_smooth;
    int          post_iter      = argus.post_smooth;
    std::string  smoother       = argus.smoother;
    unsigned int format         = argus.format;
    unsigned int ordering       = argus.ordering;
    bool         rebuildnumeric = argus.rebuildnumeric;

    // Initialize rocALUTION platform
    set_device_rocalution(device);
    init_rocalution();

    // rocALUTION structures
    LocalMatrix<T> A;
    LocalVector<T> x;
    LocalVector<T> b;
    LocalVector<T> b2;
    LocalVector<T> e;

    // Generate A
    int* csr_ptr = NULL;
    int* csr_col = NULL;
    T*   csr_val = NULL;

    int nrow = gen_2d_laplacian(ndim, &csr_ptr, &csr_col, &csr_val);
    int nnz  = csr_ptr[nrow];

    T* csr_val2 = NULL;
    if(rebuildnumeric)
    {
        csr_val2 = new T[nnz];
        for(int i = 0; i < nnz; i++)
        {
            csr_val2[i] = csr_val[i];
        }
    }

    A.SetDataPtrCSR(&csr_ptr, &csr_col, &csr_val, "A", nnz, nrow, nrow);

    assert(csr_ptr == NULL);
    assert(csr_col == NULL);
    assert(csr_val == NULL);

    // Move data to accelerator
    A.MoveToAccelerator();
    x.MoveToAccelerator();
    b.MoveToAccelerator();
    b2.MoveToAccelerator();
    e.MoveToAccelerator();

    // Allocate x, b and e
    x.Allocate("x", A.GetN());
    b.Allocate("b", A.GetM());
    b2.Allocate("b2", A.GetM());
    e.Allocate("e", A.GetN());

    // b = A * 1
    e.Ones();
    A.Apply(e, &b);

    // Random initial guess
    x.SetRandomUniform(12345ULL, -4.0, 6.0);

    // Solver
    CG<LocalMatrix<T>, LocalVector<T>, T> ls;

    // AMG
    PairwiseAMG<LocalMatrix<T>, LocalVector<T>, T> p;

    // Setup AMG
    p.SetOrdering(ordering);
    p.SetCoarsestLevel(300);
    p.SetCycle(Kcycle);
    p.SetOperator(A);
    p.SetManualSmoothers(true);
    p.SetManualSolver(true);
    p.BuildHierarchy();

    // Get number of hierarchy levels
    int levels = p.GetNumLevels();

    // Coarse grid solver
    CG<LocalMatrix<T>, LocalVector<T>, T> cgs;
    cgs.Verbose(0);

    // Smoother for each level
    IterativeLinearSolver<LocalMatrix<T>, LocalVector<T>, T>** sm
        = new IterativeLinearSolver<LocalMatrix<T>, LocalVector<T>, T>*[levels - 1];

    Preconditioner<LocalMatrix<T>, LocalVector<T>, T>** smooth
        = new Preconditioner<LocalMatrix<T>, LocalVector<T>, T>*[levels - 1];

    for(int i = 0; i < levels - 1; ++i)
    {
        FixedPoint<LocalMatrix<T>, LocalVector<T>, T>* fp
            = new FixedPoint<LocalMatrix<T>, LocalVector<T>, T>;
        sm[i] = fp;

        if(smoother == "Jacobi")
        {
            smooth[i] = new Jacobi<LocalMatrix<T>, LocalVector<T>, T>;
            fp->SetRelaxation(0.67);
        }
        else if(smoother == "MCGS")
        {
            smooth[i] = new MultiColoredGS<LocalMatrix<T>, LocalVector<T>, T>;
            fp->SetRelaxation(1.3);
        }
        else if(smoother == "MCILU")
            smooth[i] = new MultiColoredILU<LocalMatrix<T>, LocalVector<T>, T>;
        else
            return false;

        sm[i]->SetPreconditioner(*(smooth[i]));
        sm[i]->Verbose(0);
    }

    p.SetSmoother(sm);
    p.SetSolver(cgs);
    p.SetSmootherPreIter(pre_iter);
    p.SetSmootherPostIter(post_iter);
    p.SetOperatorFormat(format, (format == BCSR ? argus.blockdim : 1));
    p.InitMaxIter(1);
    p.Verbose(0);

    ls.Verbose(0);
    ls.SetOperator(A);
    ls.SetPreconditioner(p);

    ls.Init(1e-8, 0.0, 1e+8, 10000);
    ls.Build();

    if(rebuildnumeric)
    {
        A.UpdateValuesCSR(csr_val2);
        delete[] csr_val2;

        // b2 = A * 1
        A.Apply(e, &b2);

        ls.ReBuildNumeric();
    }

    // Matrix format
    A.ConvertTo(format, format == BCSR ? argus.blockdim : 1);

    ls.Solve(rebuildnumeric ? b2 : b, &x);

    // Verify solution
    x.ScaleAdd(-1.0, e);
    T nrm2 = x.Norm();

    bool success = check_residual(nrm2);

    // Clean up
    ls.Clear(); // TODO

    // Stop rocALUTION platform
    stop_rocalution();

    for(int i = 0; i < levels - 1; ++i)
    {
        delete smooth[i];
        delete sm[i];
    }
    delete[] smooth;
    delete[] sm;

    return success;
}

#endif // TESTING_PAIRWISE_AMG_HPP

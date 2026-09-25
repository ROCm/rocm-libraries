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
#ifndef TESTING_HYBRID_AMG_HPP
#define TESTING_HYBRID_AMG_HPP

#include "utility.hpp"

#include <algorithm>
#include <rocalution/rocalution.hpp>

using namespace rocalution;

static bool check_residual(float res)
{
    return (res < 2e-2f);
}

static bool check_residual(double res)
{
    return (res < 1e-5);
}

// Unsmoothed aggregation AMG on the finest levels (limited by SetMaxLevels), with a
// Ruge-Stueben AMG as coarse grid solver for the remaining levels.
template <typename T>
bool testing_hybrid_amg(Arguments argus)
{
    int         ndim                = argus.size;
    int         max_levels          = argus.max_levels;
    std::string coarsening_strategy = argus.coarsening_strategy;
    std::string matrix_type         = argus.matrix_type;
    int         cycle               = argus.cycle;
    bool        rebuildnumeric      = argus.rebuildnumeric;
    bool        disable_accelerator = !argus.use_acc;

    const int ua_coarse_size = 1000;
    const int rs_coarse_size = 20;

    // Initialize rocALUTION platform
    disable_accelerator_rocalution(disable_accelerator);
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

    int nrow = 0;
    if(matrix_type == "Laplacian2D")
    {
        nrow = gen_2d_laplacian(ndim, &csr_ptr, &csr_col, &csr_val);
    }
    else if(matrix_type == "Laplacian3D")
    {
        nrow = gen_3d_laplacian(ndim, &csr_ptr, &csr_col, &csr_val);
    }
    else
    {
        return false;
    }
    int nnz = csr_ptr[nrow];

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
    if(!disable_accelerator)
    {
        A.MoveToAccelerator();
        x.MoveToAccelerator();
        b.MoveToAccelerator();
        b2.MoveToAccelerator();
        e.MoveToAccelerator();
    }

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

    CoarseningStrategy strat;
    if(coarsening_strategy == "Greedy")
    {
        strat = CoarseningStrategy::Greedy;
    }
    else if(coarsening_strategy == "PMIS")
    {
        strat = CoarseningStrategy::PMIS;
    }
    else
    {
        return false;
    }

    // Number of levels the unsmoothed aggregation AMG builds without level limit
    int ref_levels;
    {
        UAAMG<LocalMatrix<T>, LocalVector<T>, T> ref;

        ref.SetCoarsestLevel(ua_coarse_size);
        ref.SetCoarseningStrategy(strat);
        ref.SetOperator(A);
        ref.BuildHierarchy();

        ref_levels = ref.GetNumLevels();
    }

    // Coarse levels: Ruge-Stueben AMG, one V-cycle per coarse grid solve
    RugeStuebenAMG<LocalMatrix<T>, LocalVector<T>, T> rs;

    rs.SetCoarseningStrategy(CoarseningStrategy::PMIS);
    rs.SetInterpolationType(InterpolationType::ExtPI);
    rs.SetCoarsestLevel(rs_coarse_size);
    rs.FlagPrecond();
    rs.Verbose(0);

    // Fine levels: unsmoothed aggregation AMG
    UAAMG<LocalMatrix<T>, LocalVector<T>, T> p;

    p.SetCoarsestLevel(ua_coarse_size);
    p.SetMaxLevels(max_levels);
    p.SetCoarseningStrategy(strat);
    p.SetCycle(cycle);
    p.SetManualSolver(true);
    p.SetSolver(rs);
    p.InitMaxIter(1);
    p.Verbose(0);

    // Solver
    FCG<LocalMatrix<T>, LocalVector<T>, T> ls;

    ls.Verbose(0);
    ls.SetOperator(A);
    ls.SetPreconditioner(p);

    ls.Init(1e-8, 0.0, 1e+8, 10000);
    ls.Build();

    bool success = true;

    // The level limit is respected and does not affect the coarsening otherwise
    success &= (p.GetNumLevels() == std::min(ref_levels, max_levels));

    // The Ruge-Stueben AMG has been built on the coarsest unsmoothed aggregation level
    success &= (rs.GetNumLevels() > 1);

    if(rebuildnumeric)
    {
        A.UpdateValuesCSR(csr_val2);
        delete[] csr_val2;

        // b2 = A * 1
        A.Apply(e, &b2);

        ls.ReBuildNumeric();
    }

    ls.Solve(rebuildnumeric ? b2 : b, &x);

    // Verify solution
    x.ScaleAdd(-1.0, e);
    T nrm2 = x.Norm();

    success &= check_residual(nrm2);

    // Clean up
    ls.Clear();

    // Stop rocALUTION platform
    stop_rocalution();
    disable_accelerator_rocalution(false);

    return success;
}

#endif // TESTING_HYBRID_AMG_HPP

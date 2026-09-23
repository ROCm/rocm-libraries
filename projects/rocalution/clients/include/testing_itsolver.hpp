/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "utility.hpp"

#include <rocalution/rocalution.hpp>

template <typename T>
bool testing_itsolver(Arguments argus)
{
    using namespace rocalution;

    int          ndim                = argus.size;
    unsigned int format              = argus.format;
    std::string  matrix_type         = argus.matrix_type;
    bool         disable_accelerator = !argus.use_acc;

    // Initialize rocALUTION platform
    disable_accelerator_rocalution(disable_accelerator);
    set_device_rocalution(device);
    init_rocalution();

    // rocALUTION structures
    LocalMatrix<T> A;
    LocalVector<T> x;
    LocalVector<T> b;
    LocalVector<T> e;

    // Generate A
    int* csr_ptr = NULL;
    int* csr_col = NULL;
    T*   csr_val = NULL;

    int nrow = 0;
    int ncol = 0;
    if(matrix_type == "Laplacian2D")
    {
        nrow = gen_2d_laplacian(ndim, &csr_ptr, &csr_col, &csr_val);
        ncol = nrow;
    }
    else if(matrix_type == "PermutedIdentity")
    {
        nrow = gen_permuted_identity(ndim, &csr_ptr, &csr_col, &csr_val);
        ncol = nrow;
    }
    else
    {
        stop_rocalution();
        disable_accelerator_rocalution(false);
        return true;
    }
    int nnz = csr_ptr[nrow];

    A.SetDataPtrCSR(&csr_ptr, &csr_col, &csr_val, "A", nnz, nrow, nrow);

    // Move data to accelerator
    if(!disable_accelerator)
    {
        A.MoveToAccelerator();
        x.MoveToAccelerator();
        b.MoveToAccelerator();
        e.MoveToAccelerator();
    }

    // Allocate x, b and e
    x.Allocate("x", A.GetN());
    b.Allocate("b", A.GetM());
    e.Allocate("e", A.GetN());

    // Linear Solver
    FixedPoint<LocalMatrix<T>, LocalVector<T>, T> fp;

    // Preconditioner
    ItILU0<LocalMatrix<T>, LocalVector<T>, T> p;

    // Set iterative ILU stopping criteria
    p.SetTolerance(1e-8);
    p.SetMaxIter(50);

    p.SetAlgorithm(ItILU0Algorithm::SyncSplit);

    // Set up iterative triangular solve
    SolverDescr descr;
    descr.SetTriSolverAlg(TriSolverAlg_Iterative);
    descr.SetIterativeSolverMaxIteration(30);
    descr.SetIterativeSolverTolerance(1e-8);

    descr.DisableIterativeSolverTolerance();
    descr.EnableIterativeSolverTolerance();
    SolverDescr descr_new(descr); // Copy the descriptor

    p.SetSolverDescriptor(descr_new);

    // Initialize b such that A 1 = b
    e.Ones();
    A.Apply(e, &b);

    // Initial zero guess
    x.Zeros();

    // Set solver operator
    fp.SetOperator(A);
    // Set solver preconditioner
    fp.SetPreconditioner(p);

    // Build solver
    fp.Build();

    // Verbosity output
    fp.Verbose(1);

    fp.InitMinIter(1);
    fp.InitMaxIter(1000);
    fp.InitTol(1e-8, 1e-8, 1e-8);

    // Print matrix info
    A.Info();

    // Solve A x = b
    fp.Solve(b, &x);

    int           niter_preconditioner;
    const double* history = p.GetConvergenceHistory(&niter_preconditioner);

    auto res_final = fp.GetCurrentResidual();
    //auto res_init      = fp.GetInitialResidual();
    //auto niter         = fp.GetNumIterations();
    auto status_solver = fp.GetSolverStatus();
    auto ind           = fp.GetAmaxResidualIndex();
    // Clear solver
    fp.Clear();

    // Compute error L2 norm
    e.ScaleAdd(-1.0, x);
    T error = e.Norm();
    std::cout << "||e - x||_2 = " << error << std::endl;

    // Stop rocALUTION platform
    stop_rocalution();
    disable_accelerator_rocalution(false);

    return true;
}
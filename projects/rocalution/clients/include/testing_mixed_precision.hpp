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
bool testing_mixed_precision(Arguments argus)
{
    using namespace rocalution;

    int          ndim           = argus.size;
    std::string  precond        = argus.precond;
    unsigned int format         = argus.format;
    std::string  matrix_type    = argus.matrix_type;
    bool         rebuildnumeric = argus.rebuildnumeric;

    // Initialize rocALUTION platform
    set_device_rocalution(device);
    init_rocalution();

    // rocALUTION structures
    LocalMatrix<T> A;
    LocalVector<T> x;
    LocalVector<T> e;
    LocalVector<T> rhs;

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
    else
    {
        stop_rocalution();
        disable_accelerator_rocalution(false);
        return true;
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

    // Allocate x, b and e
    x.Allocate("x", A.GetN());
    rhs.Allocate("b", A.GetM());
    e.Allocate("e", A.GetN());

    // Linear Solver
    MixedPrecisionDC<LocalMatrix<T>,
                     LocalVector<T>,
                     T,
                     LocalMatrix<float>,
                     LocalVector<float>,
                     float>
                                                                   mp;
    CG<LocalMatrix<float>, LocalVector<float>, float>              cg;
    MultiColoredILU<LocalMatrix<float>, LocalVector<float>, float> p;

    // Initialize rhs such that A 1 = rhs
    e.Ones();
    A.Apply(e, &rhs);

    // Initial zero guess
    x.Zeros();

    // setup a lower tol for the inner solver
    cg.SetPreconditioner(p);
    cg.Init(1e-5, 1e-2, 1e+20, 100000);

    // setup the mixed-precision DC
    mp.SetOperator(A);
    mp.Set(cg);

    // Build solver
    mp.Build();

    // Verbosity output
    mp.Verbose(1);

    // Solve A x = rhs
    mp.Solve(rhs, &x);

    // Compute error L2 norm
    e.ScaleAdd(-1.0, x);
    T error = e.Norm();
    std::cout << "||e - x||_2 = " << error << std::endl;

    // Clear solver
    mp.Clear();

    // Stop rocALUTION platform
    stop_rocalution();

    return true;
}
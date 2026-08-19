/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TESTING_QR_HPP
#define TESTING_QR_HPP

#include "utility.hpp"

#include <rocalution/rocalution.hpp>

using namespace rocalution;

static bool check_residual(float res)
{
    return (res < 1e-3f);
}

static bool check_residual(double res)
{
    return (res < 1e-6);
}

template <typename T>
bool testing_qr(Arguments argus)
{
    int          ndim             = argus.size;
    unsigned int format           = argus.format;
    std::string  matrix_type      = argus.matrix_type;
    const bool   use_host_and_acc = argus.use_acc;

    // Initialize rocALUTION platform
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
        return false;
    }
    int nnz = csr_ptr[nrow];

    A.SetDataPtrCSR(&csr_ptr, &csr_col, &csr_val, "A", nnz, nrow, nrow);

    // Allocate x, b and e
    x.Allocate("x", A.GetN());
    b.Allocate("b", A.GetM());
    e.Allocate("e", A.GetN());

    // b = A * 1
    e.Ones();
    A.Apply(e, &b);

    // Random initial guess
    x.SetRandomUniform(12345ULL, -4.0, 6.0);

    // Solver
    QR<LocalMatrix<T>, LocalVector<T>, T> dls;

    dls.Verbose(0);
    dls.SetOperator(A);

    dls.Build();

    // Matrix format
    A.ConvertTo(format, format == BCSR ? argus.blockdim : 1);

    // Move data to accelerator
    dls.MoveToAccelerator();
    A.MoveToAccelerator();
    x.MoveToAccelerator();
    b.MoveToAccelerator();
    e.MoveToAccelerator();

    dls.Solve(b, &x);

    // Verify solution
    x.ScaleAdd(-1.0, e);
    T nrm2_acc = x.Norm();

    bool success = check_residual(nrm2_acc);

    if(use_host_and_acc)
    {
        dls.MoveToHost();
        A.MoveToHost();
        x.MoveToHost();
        e.MoveToHost();
        b.MoveToHost();

        dls.Solve(b, &x);

        // Verify solution
        x.ScaleAdd(-1.0, e);
        T nrm2_host = x.Norm();

        success = success && check_residual(nrm2_host);
    }
    // Clean up
    dls.Clear();

    // Stop rocALUTION platform
    stop_rocalution();

    return success;
}

#endif // TESTING_QR_HPP

/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include <gtest/gtest.h>
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
bool testing_local_matrix_lusolve(Arguments argus)
{
    const int          size   = argus.size;
    const unsigned int format = argus.format;

    // Initialize rocALUTION
    set_device_rocalution(device);
    init_rocalution();

    // rocALUTION structures
    LocalMatrix<T> A;
    LocalVector<T> base_x;
    LocalVector<T> test_x;
    LocalVector<T> b;
    LocalVector<T> e;

    // Generate A
    int* csr_ptr = NULL;
    int* csr_col = NULL;
    T*   csr_val = NULL;

    int nrow = 0;
    int ncol = 0;

    nrow = gen_2d_laplacian(size, &csr_ptr, &csr_col, &csr_val);
    ncol = nrow;

    int nnz = csr_ptr[nrow];

    assert(csr_ptr != NULL);
    assert(csr_col != NULL);
    assert(csr_val != NULL);

    A.SetDataPtrCSR(&csr_ptr, &csr_col, &csr_val, "A", nnz, nrow, nrow);

    // Move data to accelerator
    A.MoveToAccelerator();
    base_x.MoveToAccelerator();
    b.MoveToAccelerator();
    e.MoveToAccelerator();

    // Allocate x, b and e
    base_x.Allocate("x", A.GetN());
    b.Allocate("b", A.GetM());
    e.Allocate("e", A.GetN());

    // b = A * 1
    e.Ones();
    A.Apply(e, &b);

    // Random initial guess
    base_x.SetRandomUniform(12345ULL, -4.0, 6.0);
    test_x.CloneFrom(base_x);

    // Matrix format
    A.ConvertTo(format, format == BCSR ? argus.blockdim : 1);

    // Factor
    A.LUFactorize();

    // Compare Solves
    bool success = true;
    T    nrm2    = 1;

    // LUSolve on device
    A.LUAnalyse();
    A.LUSolve(b, &test_x);
    A.LUAnalyseClear();

    test_x.ScaleAdd(-1.0, e);
    nrm2 = test_x.Norm();

    success &= check_residual(nrm2);

    // LUSolve on host
    A.MoveToHost();
    b.MoveToHost();
    test_x.CopyFrom(base_x);
    test_x.MoveToHost();

    A.LUAnalyse();
    A.LUSolve(b, &test_x);
    A.LUAnalyseClear();

    test_x.MoveToAccelerator();
    test_x.ScaleAdd(-1.0, e);
    nrm2 = test_x.Norm();

    success &= check_residual(nrm2);

    // Stop rocALUTION platform
    stop_rocalution();

    return success;
}

template <typename T>
bool testing_local_matrix_llsolve(Arguments argus)
{
    const int          size   = argus.size;
    const unsigned int format = argus.format;

    // Initialize rocALUTION
    set_device_rocalution(device);
    init_rocalution();

    // rocALUTION structures
    LocalMatrix<T> A;
    LocalMatrix<T> L;
    LocalVector<T> base_x;
    LocalVector<T> test_x;
    LocalVector<T> b;
    LocalVector<T> e;

    // Generate A
    int* csr_ptr = NULL;
    int* csr_col = NULL;
    T*   csr_val = NULL;

    int nrow = 0;
    int ncol = 0;

    nrow = gen_2d_laplacian(size, &csr_ptr, &csr_col, &csr_val);
    ncol = nrow;

    int nnz = csr_ptr[nrow];

    assert(csr_ptr != NULL);
    assert(csr_col != NULL);
    assert(csr_val != NULL);

    A.SetDataPtrCSR(&csr_ptr, &csr_col, &csr_val, "A", nnz, nrow, nrow);

    // Move data to accelerator
    A.MoveToAccelerator();
    L.MoveToAccelerator();
    base_x.MoveToAccelerator();
    b.MoveToAccelerator();
    e.MoveToAccelerator();

    // Allocate x, b and e
    base_x.Allocate("x", A.GetM());
    b.Allocate("b", A.GetM());
    e.Allocate("e", A.GetM());

    // Set up L
    A.ExtractL(&L, true);

    // b = LL^T * 1
    e.Ones();
    {
        LocalMatrix<T> Lt;
        LocalVector<T> tmp;

        Lt.MoveToAccelerator();
        tmp.MoveToAccelerator();

        tmp.Allocate("tmp", A.GetN());
        L.Transpose(&Lt);

        Lt.Apply(e, &tmp);
        L.Apply(tmp, &b);
    }

    // Random initial guess
    base_x.SetRandomUniform(12345ULL, -4.0, 6.0);
    test_x.CloneFrom(base_x);

    // Matrix format
    L.ConvertTo(format, format == BCSR ? argus.blockdim : 1);

    // Compare Solves
    bool success = true;
    T    nrm2    = 1;

    // LLSolve on device
    L.LLAnalyse();
    L.LLSolve(b, &test_x);
    L.LAnalyseClear();

    test_x.ScaleAdd(-1.0, e);
    nrm2 = test_x.Norm();

    success &= check_residual(nrm2);

    // LLSolve on host
    L.MoveToHost();
    b.MoveToHost();
    test_x.CopyFrom(base_x);
    test_x.MoveToHost();

    L.LLAnalyse();
    L.LLSolve(b, &test_x);
    L.LLAnalyseClear();

    test_x.MoveToAccelerator();
    test_x.ScaleAdd(-1.0, e);
    nrm2 = test_x.Norm();

    success &= check_residual(nrm2);

    // Stop rocALUTION platform
    stop_rocalution();

    return success;
}

template <typename T>
bool testing_local_matrix_lsolve(Arguments argus)
{
    const int          size      = argus.size;
    const unsigned int format    = argus.format;
    const bool         unit_diag = argus.unit_diag;

    // Initialize rocALUTION
    set_device_rocalution(device);
    init_rocalution();

    // rocALUTION structures
    LocalMatrix<T> A;
    LocalMatrix<T> L;
    LocalVector<T> base_x;
    LocalVector<T> test_x;
    LocalVector<T> b;
    LocalVector<T> e;

    // Generate A
    int* csr_ptr = NULL;
    int* csr_col = NULL;
    T*   csr_val = NULL;

    int nrow = 0;
    int ncol = 0;

    nrow = gen_2d_laplacian(size, &csr_ptr, &csr_col, &csr_val);
    ncol = nrow;

    int nnz = csr_ptr[nrow];

    assert(csr_ptr != NULL);
    assert(csr_col != NULL);
    assert(csr_val != NULL);

    A.SetDataPtrCSR(&csr_ptr, &csr_col, &csr_val, "A", nnz, nrow, nrow);

    // Move data to accelerator
    A.MoveToAccelerator();
    L.MoveToAccelerator();
    base_x.MoveToAccelerator();
    b.MoveToAccelerator();
    e.MoveToAccelerator();

    // Allocate x, b and e
    base_x.Allocate("x", A.GetN());
    b.Allocate("b", A.GetM());
    e.Allocate("e", A.GetN());

    // Set up L
    A.ExtractL(&L, !unit_diag);

    // b = L * 1
    e.Ones();
    L.Apply(e, &b);

    if(unit_diag)
    {
        // b + 1 * e
        b.ScaleAdd(1, e);
    }

    // Random initial guess
    base_x.SetRandomUniform(12345ULL, -4.0, 6.0);
    test_x.CloneFrom(base_x);

    // Matrix format
    L.ConvertTo(format, format == BCSR ? argus.blockdim : 1);

    // Compare Solves
    bool success = true;
    T    nrm2    = 1;

    // LSolve on device
    L.LAnalyse(unit_diag);
    L.LSolve(b, &test_x);
    L.LAnalyseClear();

    test_x.ScaleAdd(-1.0, e);
    nrm2 = test_x.Norm();

    success &= check_residual(nrm2);

    // LSolve on host
    L.MoveToHost();
    b.MoveToHost();
    test_x.CopyFrom(base_x);
    test_x.MoveToHost();

    L.LAnalyse(unit_diag);
    L.LSolve(b, &test_x);
    L.LAnalyseClear();

    test_x.MoveToAccelerator();
    test_x.ScaleAdd(-1.0, e);
    nrm2 = test_x.Norm();

    success &= check_residual(nrm2);

    // Stop rocALUTION platform
    stop_rocalution();

    return success;
}

template <typename T>
bool testing_local_matrix_usolve(Arguments argus)
{
    const int          size      = argus.size;
    const unsigned int format    = argus.format;
    const bool         unit_diag = argus.unit_diag;

    // Initialize rocALUTION
    set_device_rocalution(device);
    init_rocalution();

    // rocALUTION structures
    LocalMatrix<T> A;
    LocalMatrix<T> U;
    LocalVector<T> base_x;
    LocalVector<T> test_x;
    LocalVector<T> b;
    LocalVector<T> e;

    // Generate A
    int* csr_ptr = NULL;
    int* csr_col = NULL;
    T*   csr_val = NULL;

    int nrow = 0;
    int ncol = 0;

    nrow = gen_2d_laplacian(size, &csr_ptr, &csr_col, &csr_val);
    ncol = nrow;

    int nnz = csr_ptr[nrow];

    assert(csr_ptr != NULL);
    assert(csr_col != NULL);
    assert(csr_val != NULL);

    A.SetDataPtrCSR(&csr_ptr, &csr_col, &csr_val, "A", nnz, nrow, nrow);

    // Move data to accelerator
    A.MoveToAccelerator();
    U.MoveToAccelerator();
    base_x.MoveToAccelerator();
    b.MoveToAccelerator();
    e.MoveToAccelerator();

    // Allocate x, b and e
    base_x.Allocate("x", A.GetN());
    b.Allocate("b", A.GetM());
    e.Allocate("e", A.GetN());

    // Set up U
    A.ExtractU(&U, !unit_diag);

    // b = U * 1
    e.Ones();
    U.Apply(e, &b);

    if(unit_diag)
    {
        // b + 1 * e
        b.ScaleAdd(1, e);
    }

    // Random initial guess
    base_x.SetRandomUniform(12345ULL, -4.0, 6.0);
    test_x.CloneFrom(base_x);

    // Matrix format
    U.ConvertTo(format, format == BCSR ? argus.blockdim : 1);

    // Compare Solves
    bool success = true;
    T    nrm2    = 1;

    // USolve on device
    U.UAnalyse(unit_diag);
    U.USolve(b, &test_x);
    U.UAnalyseClear();

    test_x.ScaleAdd(-1.0, e);
    nrm2 = test_x.Norm();

    success &= check_residual(nrm2);

    // USolve on host
    U.MoveToHost();
    b.MoveToHost();
    test_x.CopyFrom(base_x);
    test_x.MoveToHost();

    U.UAnalyse(unit_diag);
    U.USolve(b, &test_x);
    U.UAnalyseClear();

    test_x.MoveToAccelerator();
    test_x.ScaleAdd(-1.0, e);
    nrm2 = test_x.Norm();

    success &= check_residual(nrm2);

    // Stop rocALUTION platform
    stop_rocalution();

    return success;
}

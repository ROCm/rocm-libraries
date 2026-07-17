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
bool testing_chebyshev(Arguments argus)
{
    using namespace rocalution;

    int          ndim                = argus.size;
    std::string  precond             = argus.precond;
    unsigned int format              = argus.format;
    std::string  matrix_type         = argus.matrix_type;
    bool         rebuildnumeric      = argus.rebuildnumeric;
    bool         disable_accelerator = !argus.use_acc;

    // Initialize rocALUTION platform
    disable_accelerator_rocalution(disable_accelerator);
    set_device_rocalution(device);
    init_rocalution();

    // rocALUTION structures
    LocalMatrix<T>  A;
    LocalVector<T>  x;
    LocalVector<T>  b;
    LocalVector<T>  b_old;
    LocalVector<T>* b_k;
    LocalVector<T>* b_k1;
    LocalVector<T>* b_tmp;
    LocalVector<T>  e;
    LocalVector<T>  rhs;

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

    // Move data to accelerator
    if(!disable_accelerator)
    {
        A.MoveToAccelerator();
        x.MoveToAccelerator();
        rhs.MoveToAccelerator();
        e.MoveToAccelerator();
    }

    // Allocate x, b and e
    x.Allocate("x", A.GetN());
    rhs.Allocate("b", A.GetM());
    e.Allocate("e", A.GetN());

    T lambda_min;
    T lambda_max;

    A.Gershgorin(lambda_min, lambda_max);

    // Chebyshev iteration
    Chebyshev<LocalMatrix<T>, LocalVector<T>, T> ls;

    // Initialize rhs such that A 1 = rhs
    e.Ones();
    A.Apply(e, &rhs);

    // Initial zero guess
    x.Zeros();

    // Preconditioner
    Preconditioner<LocalMatrix<T>, LocalVector<T>, T>* p;

    if(precond == "None")
        p = NULL;
    else if(precond == "Chebyshev")
    {
        // Chebyshev preconditioner

        // Determine min and max eigenvalues
        T lambda_min;
        T lambda_max;

        A.Gershgorin(lambda_min, lambda_max);

        AIChebyshev<LocalMatrix<T>, LocalVector<T>, T>* cheb
            = new AIChebyshev<LocalMatrix<T>, LocalVector<T>, T>;
        cheb->Set(3, lambda_max / 7.0, lambda_max);

        p = cheb;
    }
    else if(precond == "FSAI")
        p = new FSAI<LocalMatrix<T>, LocalVector<T>, T>;
    else if(precond == "SPAI")
        p = new SPAI<LocalMatrix<T>, LocalVector<T>, T>;
    else if(precond == "TNS")
        p = new TNS<LocalMatrix<T>, LocalVector<T>, T>;
    else if(precond == "Jacobi")
        p = new Jacobi<LocalMatrix<T>, LocalVector<T>, T>;
    else if(precond == "GS")
        p = new GS<LocalMatrix<T>, LocalVector<T>, T>;
    else if(precond == "SGS")
        p = new SGS<LocalMatrix<T>, LocalVector<T>, T>;
    else if(precond == "ILU")
        p = new ILU<LocalMatrix<T>, LocalVector<T>, T>;
    else if(precond == "ItILU0")
        p = new ItILU0<LocalMatrix<T>, LocalVector<T>, T>;
    else if(precond == "ILUT")
        p = new ILUT<LocalMatrix<T>, LocalVector<T>, T>;
    else if(precond == "IC")
        p = new IC<LocalMatrix<T>, LocalVector<T>, T>;
    else if(precond == "MCGS")
        p = new MultiColoredGS<LocalMatrix<T>, LocalVector<T>, T>;
    else if(precond == "MCSGS")
        p = new MultiColoredSGS<LocalMatrix<T>, LocalVector<T>, T>;
    else if(precond == "MCILU")
        p = new MultiColoredILU<LocalMatrix<T>, LocalVector<T>, T>;
    else
        return false;

    // Set solver operator
    ls.SetOperator(A);

    ls.Verbose(0);
    ls.SetOperator(A);

    // Set preconditioner
    if(p != NULL)
    {
        ls.SetPreconditioner(*p);
    }

    // Set eigenvalues
    ls.Set(lambda_min, lambda_max);

    // Build solver
    ls.Build();

    if(rebuildnumeric)
    {
        A.UpdateValuesCSR(csr_val2);
        delete[] csr_val2;

        A.Apply(e, &rhs);

        ls.ReBuildNumeric();
        ls.Set(lambda_min, lambda_max);
    }

    // Solve A x = rhs
    ls.Solve(rhs, &x);

    // Clear solver
    ls.Clear();
    if(p != NULL)
    {
        delete p;
    }

    // Compute error L2 norm
    e.ScaleAdd(-1.0, x);
    T error = e.Norm();
    std::cout << "Chebyshev iteration ||e - x||_2 = " << error << std::endl;

    // Stop rocALUTION platform
    stop_rocalution();
    disable_accelerator_rocalution(false);

    return true;
}
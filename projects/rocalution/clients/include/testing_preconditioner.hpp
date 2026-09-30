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

#include <gtest/gtest.h>
#include <rocalution/rocalution.hpp>

using namespace rocalution;

template <typename T>
void getTestMatrix(Arguments argus, LocalMatrix<T>& matrix)
{
    int         size        = argus.size;
    std::string matrix_type = argus.matrix_type;

    // Generate A
    int* csr_ptr = NULL;
    int* csr_col = NULL;
    T*   csr_val = NULL;

    int nrow = gen_2d_laplacian(size, &csr_ptr, &csr_col, &csr_val);
    int ncol = nrow;

    int nnz = csr_ptr[nrow];

    matrix.SetDataPtrCSR(&csr_ptr, &csr_col, &csr_val, "TestMatrix", nnz, nrow, ncol);
}

template <typename T>
void testing_preconditioner_all(Arguments argus)
{
    std::string precond = argus.precond;

    LocalMatrix<T> matrix;
    getTestMatrix<T>(argus, matrix);
    GMRES<LocalMatrix<T>, LocalVector<T>, T> ls;

    std::cout << "Testing preconditioner: " << precond << std::endl;
    std::cout << "Matrix size: " << matrix.GetM() << " x " << matrix.GetN() << std::endl;

    Preconditioner<LocalMatrix<T>, LocalVector<T>, T>*  p;
    MultiColoredILU<LocalMatrix<T>, LocalVector<T>, T>* mcilu_p = NULL;
    Solver<LocalMatrix<T>, LocalVector<T>, T>**         vp      = NULL;

    ls.Verbose(0);
    ls.SetOperator(matrix);

    if(precond == "Jacobi")
    {
        p = new Jacobi<LocalMatrix<T>, LocalVector<T>, T>;
    }
    else if(precond == "GS")
    {
        p = new GS<LocalMatrix<T>, LocalVector<T>, T>;
    }
    else if(precond == "SGS")
    {
        p = new SGS<LocalMatrix<T>, LocalVector<T>, T>;
    }
    else if(precond == "ILU")
    {
        p = new ILU<LocalMatrix<T>, LocalVector<T>, T>;
        static_cast<ILU<LocalMatrix<T>, LocalVector<T>, T>*>(p)->Set(0, true);
    }
    else if(precond == "ItILU0")
    {
        p = new ItILU0<LocalMatrix<T>, LocalVector<T>, T>;
        static_cast<ItILU0<LocalMatrix<T>, LocalVector<T>, T>*>(p)->SetAlgorithm(
            ItILU0Algorithm::Default);
        static_cast<ItILU0<LocalMatrix<T>, LocalVector<T>, T>*>(p)->SetOptions(
            ItILU0Option::Verbose);
        static_cast<ItILU0<LocalMatrix<T>, LocalVector<T>, T>*>(p)->SetMaxIter(10);
        static_cast<ItILU0<LocalMatrix<T>, LocalVector<T>, T>*>(p)->SetTolerance(1e-6);
    }
    else if(precond == "ILUT")
    {
        p = new ILUT<LocalMatrix<T>, LocalVector<T>, T>;
        static_cast<ILUT<LocalMatrix<T>, LocalVector<T>, T>*>(p)->Set(1.);
        static_cast<ILUT<LocalMatrix<T>, LocalVector<T>, T>*>(p)->Set(1., 10);
    }
    else if(precond == "IC")
    {
        p = new IC<LocalMatrix<T>, LocalVector<T>, T>;
    }
    else if(precond == "AIChebyshev")
    {
        p = new AIChebyshev<LocalMatrix<T>, LocalVector<T>, T>;
        T lambda_min;
        T lambda_max;

        matrix.Gershgorin(lambda_min, lambda_max);

        static_cast<AIChebyshev<LocalMatrix<T>, LocalVector<T>, T>*>(p)->Set(
            3, lambda_max / 7.0, lambda_max);
    }
    else if(precond == "FSAI")
    {
        p = new FSAI<LocalMatrix<T>, LocalVector<T>, T>;
        static_cast<FSAI<LocalMatrix<T>, LocalVector<T>, T>*>(p)->Set(2);
        static_cast<FSAI<LocalMatrix<T>, LocalVector<T>, T>*>(p)->Set(matrix);
        static_cast<FSAI<LocalMatrix<T>, LocalVector<T>, T>*>(p)->SetPrecondMatrixFormat(CSR);
    }
    else if(precond == "SPAI")
    {
        p = new SPAI<LocalMatrix<T>, LocalVector<T>, T>;
        static_cast<SPAI<LocalMatrix<T>, LocalVector<T>, T>*>(p)->SetPrecondMatrixFormat(CSR);
    }
    else if(precond == "TNS")
    {
        p = new TNS<LocalMatrix<T>, LocalVector<T>, T>;
        static_cast<TNS<LocalMatrix<T>, LocalVector<T>, T>*>(p)->Set(true);
        static_cast<TNS<LocalMatrix<T>, LocalVector<T>, T>*>(p)->SetPrecondMatrixFormat(CSR);
    }
    else if(precond == "AS")
    {
        p = new AS<LocalMatrix<T>, LocalVector<T>, T>;

        // Second level preconditioners
        Solver<LocalMatrix<T>, LocalVector<T>, T>** p2;

        int n = 2;
        p2    = new Solver<LocalMatrix<T>, LocalVector<T>, T>*[n];

        for(int i = 0; i < n; ++i)
        {
            MultiColoredILU<LocalMatrix<T>, LocalVector<T>, T>* mc;
            mc    = new MultiColoredILU<LocalMatrix<T>, LocalVector<T>, T>;
            p2[i] = mc;
        }

        // Initialize preconditioner
        static_cast<AS<LocalMatrix<T>, LocalVector<T>, T>*>(p)->Set(n, 4, p2);
    }
    else if(precond == "RAS")
    {
        p = new RAS<LocalMatrix<T>, LocalVector<T>, T>;

        // Second level preconditioners
        Solver<LocalMatrix<T>, LocalVector<T>, T>** p2;

        int n = 2;
        p2    = new Solver<LocalMatrix<T>, LocalVector<T>, T>*[n];

        for(int i = 0; i < n; ++i)
        {
            MultiColoredILU<LocalMatrix<T>, LocalVector<T>, T>* mc;
            mc    = new MultiColoredILU<LocalMatrix<T>, LocalVector<T>, T>;
            p2[i] = mc;
        }

        // Initialize preconditioner
        static_cast<RAS<LocalMatrix<T>, LocalVector<T>, T>*>(p)->Set(n, 4, p2);
    }
    else if(precond == "MultiColoredSGS")
    {
        p = new MultiColoredSGS<LocalMatrix<T>, LocalVector<T>, T>;
        static_cast<MultiColoredSGS<LocalMatrix<T>, LocalVector<T>, T>*>(p)->SetRelaxation(0.9);
        static_cast<MultiColoredSGS<LocalMatrix<T>, LocalVector<T>, T>*>(p)->SetPrecondMatrixFormat(
            CSR);
    }
    else if(precond == "MultiColoredGS")
    {
        p = new MultiColoredGS<LocalMatrix<T>, LocalVector<T>, T>;
        static_cast<MultiColoredGS<LocalMatrix<T>, LocalVector<T>, T>*>(p)->SetRelaxation(0.9);
        static_cast<MultiColoredGS<LocalMatrix<T>, LocalVector<T>, T>*>(p)->SetPrecondMatrixFormat(
            CSR);
    }
    else if(precond == "MultiColoredILU")
    {
        p = new MultiColoredILU<LocalMatrix<T>, LocalVector<T>, T>;
        static_cast<MultiColoredILU<LocalMatrix<T>, LocalVector<T>, T>*>(p)->Set(0);
        static_cast<MultiColoredILU<LocalMatrix<T>, LocalVector<T>, T>*>(p)->Set(0, 1, true);
        static_cast<MultiColoredILU<LocalMatrix<T>, LocalVector<T>, T>*>(p)->SetPrecondMatrixFormat(
            CSR);
    }
    else if(precond == "MultiElimination")
    {
        p = new MultiElimination<LocalMatrix<T>, LocalVector<T>, T>;

        mcilu_p = new MultiColoredILU<LocalMatrix<T>, LocalVector<T>, T>;
        mcilu_p->Set(0);
        static_cast<MultiElimination<LocalMatrix<T>, LocalVector<T>, T>*>(p)->Set(*mcilu_p, 2, 0.4);
    }
    else if(precond == "VariablePreconditioner")
    {
        p = new VariablePreconditioner<LocalMatrix<T>, LocalVector<T>, T>;

        vp = new Solver<LocalMatrix<T>, LocalVector<T>, T>*[3];

        // Preconditioners
        vp[0] = new MultiColoredSGS<LocalMatrix<T>, LocalVector<T>, T>;
        vp[1] = new MultiColoredILU<LocalMatrix<T>, LocalVector<T>, T>;
        vp[2] = new ILU<LocalMatrix<T>, LocalVector<T>, T>;

        // Set variable preconditioners
        static_cast<VariablePreconditioner<LocalMatrix<T>, LocalVector<T>, T>*>(p)
            ->SetPreconditioner(3, vp);
    }
    else
    {
        std::cerr << "Unknown preconditioner: " << precond << std::endl;
        exit(1);
    }

    ls.SetPreconditioner(*p);

    ls.Init(1e-6, 0.0, 1e+8, 10000);

    ls.Build();

    if(precond == "Jacobi")
    {
        p->Build(); // trigger a clear of the preconditioner followed by a build
        p->ResetOperator(matrix);
    }
    else if(precond == "GS")
    {
        p->Build(); // trigger a clear of the preconditioner followed by a build
        p->ResetOperator(matrix);
    }
    else if(precond == "SGS")
    {
        p->Build(); // trigger a clear of the preconditioner followed by a build
        p->ResetOperator(matrix);
    }
    else if(precond == "ILU")
    {
        p->Build(); // trigger a clear of the preconditioner followed by a build
        p->ResetOperator(matrix);
    }
    else if(precond == "ItILU0")
    {
        p->Build(); // trigger a clear of the preconditioner followed by a build
        p->ResetOperator(matrix);
    }
    else if(precond == "ILUT")
    {
        p->Build(); // trigger a clear of the preconditioner followed by a build
        p->ResetOperator(matrix);
    }
    else if(precond == "IC")
    {
        p->Build(); // trigger a clear of the preconditioner followed by a build
        p->ResetOperator(matrix);
    }
    else if(precond == "AIChebyshev")
    {
        p->Build(); // trigger a clear of the preconditioner followed by a build
        p->ResetOperator(matrix);
    }
    else if(precond == "FSAI")
    {
        p->Build(); // trigger a clear of the preconditioner followed by a build
        p->ResetOperator(matrix);
    }
    else if(precond == "SPAI")
    {
        p->Build(); // trigger a clear of the preconditioner followed by a build
        p->ResetOperator(matrix);
    }
    else if(precond == "TNS")
    {
        p->Build(); // trigger a clear of the preconditioner followed by a build
        p->ResetOperator(matrix);
    }
    else if(precond == "AS")
    {
        p->ResetOperator(matrix);
    }
    else if(precond == "RAS")
    {
        p->ResetOperator(matrix);
    }
    else if(precond == "MultiColoredSGS")
    {
        p->ResetOperator(matrix);
        p->ReBuildNumeric();
    }
    else if(precond == "MultiColoredGS")
    {
        p->ResetOperator(matrix);
        p->ReBuildNumeric();
    }
    else if(precond == "MultiColoredILU")
    {
        p->ResetOperator(matrix);
        p->ReBuildNumeric();
    }
    else if(precond == "MultiElimination")
    {
        p->ResetOperator(matrix);
    }

    p->Print();

    // Test the preconditioner
    LocalVector<T> x_acc, x_host, b, e;

    x_acc.Allocate("x", matrix.GetM());
    x_host.Allocate("x", matrix.GetM());
    b.Allocate("b", matrix.GetM());
    e.Allocate("e", matrix.GetM());

    ls.MoveToAccelerator();
    p->MoveToAccelerator();
    matrix.MoveToAccelerator();
    x_acc.MoveToAccelerator();
    b.MoveToAccelerator();
    e.MoveToAccelerator();

    if(precond != "ILUT" && precond != "IC")
    {
        e.Ones();
        matrix.Apply(e, &b);
        p->Solve(b, &x_acc);

        // Compute error L2 norm
        e.ScaleAdd(-1.0, x_acc);
        T error_acc = e.Norm();
        std::cout << "error_acc =" << error_acc << std::endl;
    }

    ls.MoveToHost();
    p->MoveToHost();
    matrix.MoveToHost();
    x_acc.MoveToHost();
    e.MoveToHost();
    b.MoveToHost();

    e.Ones();
    matrix.Apply(e, &b);
    p->Solve(b, &x_host);

    // Compute error L2 norm
    e.ScaleAdd(-1.0, x_host);
    T error_host = e.Norm();
    std::cout << "error_host = " << error_host << std::endl;

    if(precond != "ILUT" && precond != "IC")
    {
        x_host.ScaleAdd(-1.0, x_acc);
        T diff = x_host.Norm();
        std::cout << "diff = " << diff << std::endl;
    }

    ls.Clear();
    delete p;
    if(mcilu_p != NULL)
    {
        delete mcilu_p;
    }

    if(vp != NULL)
    {
        delete[] vp;
    }
}

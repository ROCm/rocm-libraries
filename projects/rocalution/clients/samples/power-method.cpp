/* ************************************************************************
 * Copyright (C) 2018-2020 Advanced Micro Devices, Inc. All rights Reserved.
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

#include <cstdlib>
#include <iostream>
#include <rocalution/rocalution.hpp>

using namespace rocalution;

int main(int argc, char* argv[])
{
    // Check command line parameters
    if(argc == 1)
    {
        std::cerr << argv[0] << " <matrix> [Num threads]" << std::endl;
        exit(1);
    }

    // Initialize rocALUTION
    init_rocalution();

    // Check command line parameters for number of OMP threads
    if(argc > 2)
    {
        set_omp_threads_rocalution(atoi(argv[2]));
    }

    // Print rocALUTION info
    info_rocalution();

    LocalVector<double>  b;
    LocalVector<double>  b_old;
    LocalVector<double>* b_k;
    LocalVector<double>* b_k1;
    LocalVector<double>* b_tmp;
    LocalMatrix<double>  mat;

    // Read matrix from MTX file
    mat.ReadFileMTX(std::string(argv[1]));

    // Gershgorin spectrum approximation
    double glambda_min, glambda_max;

    // Power method spectrum approximation
    double plambda_min, plambda_max;

    // Maximum number of iteration for the power method
    int iter_max = 10000;

    // Gershgorin approximation of the eigenvalues
    mat.Gershgorin(glambda_min, glambda_max);
    std::cout << "Gershgorin : Lambda min = " << glambda_min << "; Lambda max = " << glambda_max
              << std::endl;

    // Move objects to accelerator
    mat.MoveToAccelerator();
    b.MoveToAccelerator();
    b_old.MoveToAccelerator();

    // Allocate vectors
    b.Allocate("b_k+1", mat.GetM());
    b_k1 = &b;

    b_old.Allocate("b_k", mat.GetM());
    b_k = &b_old;

    // Set b_k to 1
    b_k->Ones();

    // Print matrix info
    mat.Info();

    // Start time measurement
    double tick, tack;
    tick = rocalution_time();

    // compute lambda max
    for(int i = 0; i <= iter_max; ++i)
    {
        mat.Apply(*b_k, b_k1);

        //    std::cout << b_k1->Dot(*b_k) << std::endl;
        b_k1->Scale(double(1.0) / b_k1->Norm());

        b_tmp = b_k1;
        b_k1  = b_k;
        b_k   = b_tmp;
    }

    // get lambda max (Rayleigh quotient)
    mat.Apply(*b_k, b_k1);
    plambda_max = b_k1->Dot(*b_k);

    tack = rocalution_time();
    std::cout << "Power method (lambda max) execution:" << (tack - tick) / 1e6 << " sec"
              << std::endl;

    mat.AddScalarDiagonal(double(-1.0) * plambda_max);

    b_k->Ones();

    tick = rocalution_time();

    // compute lambda min
    for(int i = 0; i <= iter_max; ++i)
    {
        mat.Apply(*b_k, b_k1);

        //    std::cout << b_k1->Dot(*b_k) + plambda_max << std::endl;
        b_k1->Scale(double(1.0) / b_k1->Norm());

        b_tmp = b_k1;
        b_k1  = b_k;
        b_k   = b_tmp;
    }

    // get lambda min (Rayleigh quotient)
    mat.Apply(*b_k, b_k1);
    plambda_min = (b_k1->Dot(*b_k) + plambda_max);

    // back to the original matrix
    mat.AddScalarDiagonal(plambda_max);

    tack = rocalution_time();
    std::cout << "Power method (lambda min) execution:" << (tack - tick) / 1e6 << " sec"
              << std::endl;

    std::cout << "Power method Lambda min = " << plambda_min << "; Lambda max = " << plambda_max
              << "; iter=2x" << iter_max << std::endl;

    LocalVector<double> x;
    LocalVector<double> e;
    LocalVector<double> rhs;

    x.CloneBackend(mat);
    e.CloneBackend(mat);
    rhs.CloneBackend(mat);

    x.Allocate("x", mat.GetN());
    e.Allocate("e", mat.GetN());
    rhs.Allocate("rhs", mat.GetM());

    // Chebyshev iteration
    Chebyshev<LocalMatrix<double>, LocalVector<double>, double> ls;

    // Initialize rhs such that A 1 = rhs
    e.Ones();
    mat.Apply(e, &rhs);

    // Initial zero guess
    x.Zeros();

    // Set solver operator
    ls.SetOperator(mat);

    // Set eigenvalues
    ls.Set(plambda_min, plambda_max);

    // Build solver
    ls.Build();

    // Start time measurement
    tick = rocalution_time();

    // Solve A x = rhs
    ls.Solve(rhs, &x);

    // Stop time measurement
    tack = rocalution_time();
    std::cout << "Solver execution:" << (tack - tick) / 1e6 << " sec" << std::endl;

    // Clear solver
    ls.Clear();

    // Compute error L2 norm
    e.ScaleAdd(-1.0, x);
    double error = e.Norm();
    std::cout << "Chebyshev iteration ||e - x||_2 = " << error << std::endl;

    // PCG + Chebyshev polynomial
    CG<LocalMatrix<double>, LocalVector<double>, double>          cg;
    AIChebyshev<LocalMatrix<double>, LocalVector<double>, double> p;

    // damping factor
    plambda_min = plambda_max / 7;

    // Set preconditioner
    p.Set(3, plambda_min, plambda_max);

    // Initialize rhs such that A 1 = rhs
    e.Ones();
    mat.Apply(e, &rhs);

    // Initial zero guess
    x.Zeros();

    // Set solver operator
    cg.SetOperator(mat);
    // Set solver preconditioner
    cg.SetPreconditioner(p);

    // Build solver
    cg.Build();

    // Start time measurement
    tick = rocalution_time();

    // Solve A x = rhs
    cg.Solve(rhs, &x);

    // Stop time measurement
    tack = rocalution_time();
    std::cout << "Solver execution:" << (tack - tick) / 1e6 << " sec" << std::endl;

    // Clear solver
    cg.Clear();

    // Compute error L2 norm
    e.ScaleAdd(-1.0, x);
    error = e.Norm();
    std::cout << "CG + AIChebyshev ||e - x||_2 = " << error << std::endl;

    // Stop rocALUTION platform
    stop_rocalution();

    return 0;
}

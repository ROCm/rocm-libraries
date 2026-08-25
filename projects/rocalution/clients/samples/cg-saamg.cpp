/* ************************************************************************
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
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

    // rocALUTION objects
    LocalVector<double> x;
    LocalVector<double> rhs;
    LocalVector<double> e;
    LocalMatrix<double> mat;

    // Move objects to accelerator
    mat.MoveToAccelerator();
    x.MoveToAccelerator();
    rhs.MoveToAccelerator();
    e.MoveToAccelerator();

    // Read matrix from MTX file
    mat.ReadFileMTX(std::string(argv[1]));

    // Start time measurement
    double tick, tack, start, end;
    start = rocalution_time();

    // Allocate vectors
    x.Allocate("x", mat.GetN());
    rhs.Allocate("rhs", mat.GetM());
    e.Allocate("e", mat.GetN());

    // Initialize rhs such that A 1 = rhs
    e.Ones();
    mat.Apply(e, &rhs);

    // Initial zero guess
    x.Zeros();

    // Start time measurement
    tick = rocalution_time();

    // Linear Solver
    CG<LocalMatrix<double>, LocalVector<double>, double> ls;

    // AMG Preconditioner
    SAAMG<LocalMatrix<double>, LocalVector<double>, double> p;
    p.SetCoarseningStrategy(CoarseningStrategy::PMIS);
    p.SetLumpingStrategy(LumpingStrategy::AddWeakConnections);
    p.SetCoarsestLevel(200);
    p.SetCouplingStrength(0.001);

    // Disable verbosity output of AMG preconditioner
    p.Verbose(0);

    // Set solver preconditioner
    ls.SetPreconditioner(p);
    // Set solver operator
    ls.SetOperator(mat);

    // Build solver
    ls.Build();

    // Compute 2 coarsest levels on the host
    p.SetHostLevels(2);

    // Stop time measurement
    tack = rocalution_time();
    std::cout << "Building took: " << (tack - tick) / 1e6 << " sec" << std::endl;

    // Print matrix info
    mat.Info();

    // Initialize solver tolerances
    ls.Init(1e-8, 1e-8, 1e+8, 10000);

    // Set verbosity output
    ls.Verbose(2);

    // Start time measurement
    tick = rocalution_time();

    // Solve A x = rhs
    ls.Solve(rhs, &x);

    // Stop time measurement
    tack = rocalution_time();
    std::cout << "Solver took: " << (tack - tick) / 1e6 << " sec" << std::endl;

    // Clear solver
    ls.Clear();

    // Compute error L2 norm
    e.ScaleAdd(-1.0, x);
    double error = e.Norm();
    std::cout << "||e - x||_2 = " << error << std::endl;

    // Stop time measurement
    end = rocalution_time();
    std::cout << "Total runtime: " << (end - start) / 1e6 << " sec" << std::endl;

    // Stop rocALUTION platform
    stop_rocalution();

    return 0;
}

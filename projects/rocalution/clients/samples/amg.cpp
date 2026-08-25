/* ************************************************************************
 * Copyright (C) 2018-2021 Advanced Micro Devices, Inc. All rights Reserved.
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
    // Time measurement
    double tick, tack, start, end;

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

    // Read matrix from MTX file
    mat.ReadFileMTX(std::string(argv[1]));

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
    tick  = rocalution_time();
    start = rocalution_time();

    // Linear Solver
    SAAMG<LocalMatrix<double>, LocalVector<double>, double> ls;

    // Set solver operator
    ls.SetOperator(mat);

    // Set coupling strength
    ls.SetCouplingStrength(0.001);
    // Set maximal number of unknowns on coarsest level
    ls.SetCoarsestLevel(300);
    // Set relaxation parameter for smoothed interpolation aggregation
    ls.SetInterpRelax(2. / 3.);
    // Set manual smoothers
    ls.SetManualSmoothers(true);
    // Set manual course grid solver
    ls.SetManualSolver(true);
    // Set grid transfer scaling
    ls.SetScaling(true);

    // Build AMG hierarchy
    ls.BuildHierarchy();

    // Obtain number of AMG levels
    int levels = ls.GetNumLevels();

    // Coarse Grid Solver
    CG<LocalMatrix<double>, LocalVector<double>, double> cgs;
    cgs.Verbose(0);

    // Smoother for each level
    IterativeLinearSolver<LocalMatrix<double>, LocalVector<double>, double>** sm
        = new IterativeLinearSolver<LocalMatrix<double>, LocalVector<double>, double>*[levels - 1];
    MultiColoredGS<LocalMatrix<double>, LocalVector<double>, double>** gs
        = new MultiColoredGS<LocalMatrix<double>, LocalVector<double>, double>*[levels - 1];

    // Coarse grid solver with preconditioner
    //  MultiColoredILU<LocalMatrix<double>, LocalVector<double>, double > p;
    //  cgs->SetPreconditioner(p);

    // Initialize smoother for each level
    for(int i = 0; i < levels - 1; ++i)
    {
        // Smooth with MultiColoredGS using Fixed-Point iteration scheme
        FixedPoint<LocalMatrix<double>, LocalVector<double>, double>* fp;
        fp = new FixedPoint<LocalMatrix<double>, LocalVector<double>, double>;
        fp->SetRelaxation(1.3);
        sm[i] = fp;

        gs[i] = new MultiColoredGS<LocalMatrix<double>, LocalVector<double>, double>;

        sm[i]->SetPreconditioner(*gs[i]);
        sm[i]->Verbose(0);
    }

    // Pass smoother and coarse grid solver to AMG
    ls.SetSmoother(sm);
    ls.SetSolver(cgs);

    // Set number of pre and post smoothing steps
    ls.SetSmootherPreIter(1);
    ls.SetSmootherPostIter(2);

    // Initialize solver tolerances
    ls.Init(1e-10, 1e-8, 1e+8, 10000);

    // Verbosity output
    ls.Verbose(2);

    // Build solver
    ls.Build();

    // Move solver to the accelerator
    mat.MoveToAccelerator();
    x.MoveToAccelerator();
    rhs.MoveToAccelerator();
    e.MoveToAccelerator();
    ls.MoveToAccelerator();

    // Print matrix info
    mat.Info();

    // Stop building time measurement
    tack = rocalution_time();
    std::cout << "Building took: " << (tack - tick) / 1e6 << " sec" << std::endl;

    // Start solving time measurement
    tick = rocalution_time();

    // Solve A x = rhs
    ls.Solve(rhs, &x);

    // Stop solving time measurement
    tack = rocalution_time();
    std::cout << "Solving took: " << (tack - tick) / 1e6 << " sec" << std::endl;

    // Clear the solver
    ls.Clear();

    // Free all allocated data
    for(int i = 0; i < levels - 1; ++i)
    {
        delete gs[i];
        delete sm[i];
    }

    delete[] gs;
    delete[] sm;

    // End time measurement
    end = rocalution_time();
    std::cout << "Total runtime: " << (end - start) / 1e6 << " sec" << std::endl;

    // Compute error L2 norm
    e.ScaleAdd(-1.0, x);
    double error = e.Norm();
    std::cout << "||e - x||_2 = " << error << std::endl;

    // Stop rocALUTION platform
    stop_rocalution();

    return 0;
}

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

#include "common.hpp"

#include <iostream>
#include <mpi.h>
#include <rocalution/rocalution.hpp>

using namespace rocalution;

int main(int argc, char* argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank;
    int num_procs;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    if(argc < 2)
    {
        std::cerr << argv[0] << " <global_matrix>" << std::endl;
        return -1;
    }

    // Disable OpenMP thread affinity
    set_omp_affinity_rocalution(false);

    // Initialize platform with rank and # of accelerator devices in the node
    init_rocalution(rank, 8);

    // Disable OpenMP
    set_omp_threads_rocalution(1);

    // Print rocALUTION info
    info_rocalution();

    // Load undistributed matrix
    LocalMatrix<double> lmat;
    lmat.ReadFileMTX(argv[1]);

    // Global structures
    ParallelManager      manager;
    GlobalMatrix<double> mat;

    // Distribute matrix - lmat will be destroyed
    distribute_matrix(&comm, &lmat, &mat, &manager);

    // rocALUTION vectors
    GlobalVector<double> rhs(manager);
    GlobalVector<double> x(manager);
    GlobalVector<double> e(manager);

    // Move objects to accelerator
    mat.MoveToAccelerator();
    x.MoveToAccelerator();
    rhs.MoveToAccelerator();
    e.MoveToAccelerator();

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
    CG<GlobalMatrix<double>, GlobalVector<double>, double> ls;

    // AMG Preconditioner
    RugeStuebenAMG<GlobalMatrix<double>, GlobalVector<double>, double> p;

    p.SetCoarseningStrategy(CoarseningStrategy::PMIS);
    p.SetInterpolationType(InterpolationType::ExtPI);
    p.SetCoarsestLevel(20);

    // Limit operator complexity
    p.SetInterpolationFF1Limit(false);

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

    if(rank == 0)
    {
        std::cout << "Building took: " << (tack - tick) / 1e6 << " sec" << std::endl;
    }

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

    if(rank == 0)
    {
        std::cout << "Solver took: " << (tack - tick) / 1e6 << " sec" << std::endl;
    }

    // Clear solver
    ls.Clear();

    // Compute error L2 norm
    e.ScaleAdd(-1.0, x);
    double error = e.Norm();

    if(rank == 0)
    {
        std::cout << "||e - x||_2 = " << error << std::endl;
    }

    // Stop rocALUTION platform
    stop_rocalution();

    MPI_Finalize();

    return 0;
}

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

#include "common.hpp"

#include <iostream>
#include <mpi.h>
#include <rocalution/rocalution.hpp>

#define ValueType double

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

    if(argc < 4)
    {
        std::cerr << argv[0] << " <x> <y> <z>" << std::endl;
        return -1;
    }

    // Disable OpenMP thread affinity
    set_omp_affinity_rocalution(false);

    // Initialize platform with rank and # of accelerator devices in the node
    init_rocalution(rank, 8);

    // Disable OpenMP
    set_omp_threads_rocalution(1);

    // Print platform
    info_rocalution();

    // Global structures
    ParallelManager         manager;
    GlobalMatrix<ValueType> mat;

    // Generate distributed 3D laplacian
    if(rank == 0)
    {
        std::cout << "Generating 3D laplacian..." << std::endl;
    }

    generate_3d_laplacian(
        atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), &comm, &mat, &manager, rank, num_procs);

    // rocALUTION vectors
    GlobalVector<ValueType> rhs(manager);
    GlobalVector<ValueType> x(manager);
    GlobalVector<ValueType> e(manager);

    // Move structures to accelerator, if available
    mat.MoveToAccelerator();
    rhs.MoveToAccelerator();
    x.MoveToAccelerator();
    e.MoveToAccelerator();

    // Allocate memory
    rhs.Allocate("rhs", mat.GetM());
    x.Allocate("x", mat.GetN());
    e.Allocate("sol", mat.GetN());

    // Initialize rhs such that A 1 = rhs
    e.Ones();
    mat.Apply(e, &rhs);

    // Initial zero guess
    x.Zeros();

    // Print matrix info
    mat.Info();

    // Linear solver
    CG<GlobalMatrix<ValueType>, GlobalVector<ValueType>, ValueType> ls;

    // Preconditioner
    RugeStuebenAMG<GlobalMatrix<ValueType>, GlobalVector<ValueType>, ValueType> p;

    // Disable AMG preconditioner verbosity output
    p.Verbose(0);

    p.SetCoarseningStrategy(CoarseningStrategy::PMIS);
    p.SetInterpolationType(InterpolationType::ExtPI);
    p.SetCoarsestLevel(20);

    // Limit operator complexity
    p.SetInterpolationFF1Limit(false);

    // Set solver preconditioner
    ls.SetPreconditioner(p);

    // Set solver operator
    ls.SetOperator(mat);

    // Build solver
    ls.Build();

    // Set verbosity output
    ls.Verbose(2);

    // Set host levels (requires solver built)
    p.SetHostLevels(2);

    // Start time measurement
    double time = rocalution_time();

    // Solve A x = rhs
    ls.Solve(rhs, &x);

    // Stop time measurement
    time = rocalution_time() - time;
    if(rank == 0)
    {
        std::cout << "Solving: " << time / 1e6 << " sec" << std::endl;
    }

    // Compute error L2 norm
    e.ScaleAdd(-1.0, x);
    ValueType nrm2 = e.Norm();
    if(rank == 0)
    {
        std::cout << "||e - x||_2 = " << nrm2 << std::endl;
    }

    // Clear solver
    ls.Clear();

    // Stop rocALUTION platform
    stop_rocalution();

    MPI_Finalize();

    return 0;
}

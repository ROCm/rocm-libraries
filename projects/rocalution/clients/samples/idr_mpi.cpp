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

    // Check command line parameters
    if(num_procs < 2)
    {
        std::cerr << "Expecting at least 2 MPI processes" << std::endl;
        return -1;
    }

    if(argc < 2)
    {
        std::cerr << argv[0] << " <global_matrix>" << std::endl;
        return -1;
    }

    // Disable OpenMP thread affinity
    set_omp_affinity_rocalution(false);

    // Initialize platform with rank and # of accelerator devices in the node
    init_rocalution(rank, 2);

    // Disable OpenMP
    set_omp_threads_rocalution(1);

    // Print platform
    info_rocalution();

    // Load undistributed matrix
    LocalMatrix<ValueType> lmat;
    lmat.ReadFileMTX(argv[1]);

    // Global structures
    ParallelManager         manager;
    GlobalMatrix<ValueType> mat;

    // Distribute matrix - lmat will be destroyed
    distribute_matrix(&comm, &lmat, &mat, &manager);

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

    e.Ones();
    mat.Apply(e, &rhs);
    x.Zeros();

    IDR<GlobalMatrix<double>, GlobalVector<double>, double>         ls;
    BlockJacobi<GlobalMatrix<double>, GlobalVector<double>, double> bj;
    FSAI<LocalMatrix<double>, LocalVector<double>, double>          p;

    bj.Set(p);

    ls.SetPreconditioner(bj);
    ls.SetOperator(mat);
    ls.Build();
    ls.Verbose(1);

    mat.Info();

    double time = rocalution_time();

    ls.Solve(rhs, &x);

    time = rocalution_time() - time;
    if(rank == 0)
    {
        std::cout << "Solving: " << time / 1e6 << " sec" << std::endl;
    }

    e.ScaleAdd(-1.0, x);
    double nrm2 = e.Norm();
    if(rank == 0)
    {
        std::cout << "||e - x||_2 = " << nrm2 << std::endl;
    }

    ls.Clear();

    stop_rocalution();

    MPI_Finalize();

    return 0;
}

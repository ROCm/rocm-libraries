/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TESTING_GLOBAL_AMG_HPP
#define TESTING_GLOBAL_AMG_HPP

#include "testing_global_matrix.hpp"
#include "utility.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <rocalution/rocalution.hpp>
#include <string>

using namespace rocalution;

// Two ranks with n rows each, coupled in one direction only: rank 1's first row has an entry
// in rank 0's last column, but rank 0's last row has no entry in rank 1's first column.
// So rank 0 sends a boundary row to rank 1 without having any ghost entries itself. Rank 0 sets
// only its local data, rank 1 sets local and ghost data. Both ranks call one setter, so they
// run the same collectives.
//
// amg is one of "SAAMG", "UAAMG", "RSAMG-Direct", "RSAMG-ExtPI". With a nonzero
// host_levels, the solve is repeated after SetHostLevels() and ReBuildNumeric(); a negative
// host_levels moves all coarse levels to the host.
template <typename T>
void testing_global_amg_one_sided_coupling(const std::string& amg,
                                           bool               coo,
                                           bool               use_acc,
                                           int                host_levels = 0)
{
    if(!testing_mpi_launched())
    {
        GTEST_SKIP() << "needs exactly 2 MPI ranks (mpirun -np 2)";
    }

    const MPI_Comm* comm = testing_mpi_comm();

    int rank;
    int num_procs;
    MPI_Comm_rank(*comm, &rank);
    MPI_Comm_size(*comm, &num_procs);

    if(num_procs != 2)
    {
        GTEST_SKIP() << "needs exactly 2 MPI ranks (mpirun -np 2)";
    }

    // See testing_global_matrix_local_data_only() for why the accelerator stays enabled
    disable_accelerator_rocalution(false);
    set_device_rocalution(device);
    init_rocalution();

    // The exact check after ReBuildNumeric() needs the host reductions in a fixed order, and
    // other tests in this binary change the OpenMP threshold without restoring it
    if(!use_acc)
    {
        set_omp_threads_rocalution(1);
    }

    // The HIP backend reads past the end of a missing ghost row pointer instead of
    // dereferencing NULL, which only faults once the boundary row is far enough out.
    int n   = use_acc ? 100000 : 1000;
    int nnz = 3 * n - 2;

    int* row = nullptr;
    int* col = nullptr;
    T*   val = nullptr;

    allocate_host(coo ? nnz : n + 1, &row);
    allocate_host(nnz, &col);
    allocate_host(nnz, &val);

    int k = 0;
    for(int i = 0; i < n; ++i)
    {
        if(!coo)
        {
            row[i] = k;
        }

        for(int j = std::max(i - 1, 0); j <= std::min(i + 1, n - 1); ++j)
        {
            if(coo)
            {
                row[k] = i;
            }

            col[k] = j;
            val[k] = (i == j) ? static_cast<T>(3) : static_cast<T>(-1);
            ++k;
        }
    }

    if(!coo)
    {
        row[n] = k;
    }

    int zero_offset[1] = {0};
    int one_offset[2]  = {0, 1};
    int boundary[1]    = {n - 1};
    int neighbor[1]    = {1 - rank};

    ParallelManager pm;
    pm.SetMPICommunicator(comm);
    pm.SetGlobalNrow(2 * static_cast<int64_t>(n));
    pm.SetGlobalNcol(2 * static_cast<int64_t>(n));
    pm.SetLocalNrow(n);
    pm.SetLocalNcol(n);

    if(rank == 0)
    {
        pm.SetBoundaryIndex(1, boundary);
        pm.SetReceivers(0, nullptr, zero_offset);
        pm.SetSenders(1, neighbor, one_offset);
    }
    else
    {
        pm.SetBoundaryIndex(0, nullptr);
        pm.SetReceivers(1, neighbor, one_offset);
        pm.SetSenders(0, nullptr, zero_offset);
    }

    GlobalMatrix<T> mat;
    mat.SetParallelManager(pm);

    if(rank == 0)
    {
        if(coo)
        {
            mat.SetLocalDataPtrCOO(&row, &col, &val, "mat", nnz);
        }
        else
        {
            mat.SetLocalDataPtrCSR(&row, &col, &val, "mat", nnz);
        }
    }
    else
    {
        // Ghost entry: local row 0, ghost column 0 (global column n - 1), value -1
        int* gst_row = nullptr;
        int* gst_col = nullptr;
        T*   gst_val = nullptr;

        allocate_host(coo ? 1 : n + 1, &gst_row);
        allocate_host(1, &gst_col);
        allocate_host(1, &gst_val);

        gst_row[0] = 0;

        if(!coo)
        {
            for(int i = 1; i <= n; ++i)
            {
                gst_row[i] = 1;
            }
        }

        gst_col[0] = 0;
        gst_val[0] = static_cast<T>(-1);

        if(coo)
        {
            mat.SetDataPtrCOO(&row, &col, &val, &gst_row, &gst_col, &gst_val, "mat", nnz, 1);
        }
        else
        {
            mat.SetDataPtrCSR(&row, &col, &val, &gst_row, &gst_col, &gst_val, "mat", nnz, 1);
        }
    }

    EXPECT_EQ(mat.GetGhost().GetM(), n);
    EXPECT_EQ(mat.GetGhost().GetN(), rank == 0 ? 0 : 1);
    EXPECT_EQ(mat.GetGhost().GetNnz(), rank == 0 ? 0 : 1);

    GlobalVector<T> x(pm);
    GlobalVector<T> b(pm);
    GlobalVector<T> e(pm);

    if(use_acc)
    {
        mat.MoveToAccelerator();
        x.MoveToAccelerator();
        b.MoveToAccelerator();
        e.MoveToAccelerator();
    }

    x.Allocate("x", mat.GetN());
    b.Allocate("b", mat.GetM());
    e.Allocate("e", mat.GetN());

    // b = A * 1
    e.Ones();
    mat.Apply(e, &b);

    BiCGStab<GlobalMatrix<T>, GlobalVector<T>, T> ls;
    ls.Verbose(0);
    ls.SetOperator(mat);

    SAAMG<GlobalMatrix<T>, GlobalVector<T>, T>          sa;
    UAAMG<GlobalMatrix<T>, GlobalVector<T>, T>          ua;
    RugeStuebenAMG<GlobalMatrix<T>, GlobalVector<T>, T> rs;
    BaseAMG<GlobalMatrix<T>, GlobalVector<T>, T>*       p = nullptr;

    if(amg == "SAAMG")
    {
        sa.SetCoarseningStrategy(CoarseningStrategy::PMIS);
        p = &sa;
    }
    else if(amg == "UAAMG")
    {
        ua.SetCoarseningStrategy(CoarseningStrategy::PMIS);
        p = &ua;
    }
    else
    {
        rs.SetCoarseningStrategy(CoarseningStrategy::PMIS);
        rs.SetInterpolationType(amg == "RSAMG-Direct" ? InterpolationType::Direct
                                                      : InterpolationType::ExtPI);
        p = &rs;
    }
    p->Verbose(0);
    ls.SetPreconditioner(*p);

    ls.Build();
    ls.Init(1e-10, 1e-10, 1e+8, 1000);

    // x should be the vector of ones
    const auto expect_solved = [&]() {
        x.Zeros();
        ls.Solve(b, &x);

        e.Ones();
        e.ScaleAdd(static_cast<T>(-1), x);
        T error = e.Norm() / std::sqrt(static_cast<T>(2 * n));

        EXPECT_LT(error, static_cast<T>(1e-6));
        EXPECT_LT(ls.GetIterationCount(), 1000);
    };

    expect_solved();

    if(host_levels != 0)
    {
        // Fewer levels put all coarse levels on the host, and miss the boundary between the
        // levels on the backend of the finest and the host levels
        if(host_levels > 0)
        {
            EXPECT_GT(p->GetNumLevels(), host_levels + 1);
        }

        const int    iter = ls.GetIterationCount();
        const double res  = ls.GetCurrentResidual();

        // Doubling all values is exact. On the host, with one thread, it leaves the iterates of
        // the solve unchanged and doubles its residuals
        mat.Scale(static_cast<T>(2));
        e.Ones();
        mat.Apply(e, &b);

        const int h = host_levels < 0 ? p->GetNumLevels() - 1 : host_levels;

        p->SetHostLevels(h);
        p->ReBuildNumeric();

        SCOPED_TRACE("SetHostLevels(" + std::to_string(h) + ") and ReBuildNumeric");
        expect_solved();

        // A wrong or stale coarse operator or smoother can still converge. On the host, the
        // rebuilt hierarchy must give exactly the same iterates. With use_acc, the host levels
        // are now computed on the host and only agree up to rounding, so this mainly checks
        // their backends and parallel managers
        if(use_acc)
        {
            EXPECT_NEAR(ls.GetIterationCount(), iter, 1);
        }
        else
        {
            EXPECT_EQ(ls.GetIterationCount(), iter);
            EXPECT_EQ(ls.GetCurrentResidual(), 2 * res);
        }
    }

    ls.Clear();
    mat.Clear();
    x.Clear();
    b.Clear();
    e.Clear();

    stop_rocalution();
}

#endif // TESTING_GLOBAL_AMG_HPP

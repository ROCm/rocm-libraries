/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TESTING_GLOBAL_MATRIX_HPP
#define TESTING_GLOBAL_MATRIX_HPP

#include "utility.hpp"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <gtest/gtest.h>
#include <mpi.h>
#include <rocalution/rocalution.hpp>

using namespace rocalution;

// rocalution-test does not initialize MPI in main, so that death tests can re-execute the binary.
// Tests that need MPI initialize it on first use, and only when an MPI launcher started the
// process: singleton MPI_Init can abort, e.g. when hwloc picks up the zlib that ships with ROCm.
// Tests that need more than one rank must be started through mpirun with a --gtest_filter, so
// that all ranks run the same tests in the same order.
inline bool testing_mpi_launched()
{
    return std::getenv("OMPI_COMM_WORLD_SIZE") != nullptr || std::getenv("PMIX_RANK") != nullptr
           || std::getenv("PMI_RANK") != nullptr;
}

inline const MPI_Comm* testing_mpi_comm()
{
    static MPI_Comm comm = MPI_COMM_WORLD;

    int initialized = 0;
    MPI_Initialized(&initialized);

    if(!initialized)
    {
        assert(testing_mpi_launched());
        MPI_Init(NULL, NULL);
        std::atexit([] {
            int finalized = 0;
            MPI_Finalized(&finalized);

            if(!finalized)
            {
                MPI_Finalize();
            }
        });
    }

    return &comm;
}

template <typename T>
void testing_global_matrix_bad_args(void)
{
    int safe_size = 100;

    // Initialize rocALUTION
    set_device_rocalution(device);
    init_rocalution();

    GlobalMatrix<T> mat;
    GlobalVector<T> vec;

    int* idata = nullptr;
    T*   data  = nullptr;

    allocate_host(safe_size, &idata);
    allocate_host(safe_size, &data);

    // SetDataPtrCSR
    {
        int* null_int  = nullptr;
        T*   null_data = nullptr;
        ASSERT_DEATH(mat.SetDataPtrCSR(
                         nullptr, &idata, &data, &idata, &idata, &data, "", safe_size, safe_size),
                     ".*Assertion.*local_row_offset != (NULL|__null)*");
        ASSERT_DEATH(mat.SetDataPtrCSR(
                         &idata, nullptr, &data, &idata, &idata, &data, "", safe_size, safe_size),
                     ".*Assertion.*local_col != (NULL|__null)*");
        ASSERT_DEATH(mat.SetDataPtrCSR(
                         &idata, &idata, nullptr, &idata, &idata, &data, "", safe_size, safe_size),
                     ".*Assertion.*local_val != (NULL|__null)*");
        ASSERT_DEATH(mat.SetDataPtrCSR(
                         &idata, &idata, &data, nullptr, &idata, &data, "", safe_size, safe_size),
                     ".*Assertion.*ghost_row_offset != (NULL|__null)*");
        ASSERT_DEATH(mat.SetDataPtrCSR(
                         &idata, &idata, &data, &idata, nullptr, &data, "", safe_size, safe_size),
                     ".*Assertion.*ghost_col != (NULL|__null)*");
        ASSERT_DEATH(mat.SetDataPtrCSR(
                         &idata, &idata, &data, &idata, &idata, nullptr, "", safe_size, safe_size),
                     ".*Assertion.*ghost_val != (NULL|__null)*");
        ASSERT_DEATH(mat.SetDataPtrCSR(
                         &null_int, &idata, &data, &idata, &idata, &data, "", safe_size, safe_size),
                     ".*Assertion.*local_row_offset != (NULL|__null)*");
        ASSERT_DEATH(mat.SetDataPtrCSR(
                         &idata, &null_int, &data, &idata, &idata, &data, "", safe_size, safe_size),
                     ".*Assertion.*local_col != (NULL|__null)*");
        ASSERT_DEATH(
            mat.SetDataPtrCSR(
                &idata, &idata, &null_data, &idata, &idata, &data, "", safe_size, safe_size),
            ".*Assertion.*local_val != (NULL|__null)*");
        ASSERT_DEATH(mat.SetDataPtrCSR(
                         &idata, &idata, &data, &null_int, &idata, &data, "", safe_size, safe_size),
                     ".*Assertion.*ghost_row_offset != (NULL|__null)*");
        ASSERT_DEATH(mat.SetDataPtrCSR(
                         &idata, &idata, &data, &idata, &null_int, &data, "", safe_size, safe_size),
                     ".*Assertion.*ghost_col != (NULL|__null)*");
        ASSERT_DEATH(
            mat.SetDataPtrCSR(
                &idata, &idata, &data, &idata, &idata, &null_data, "", safe_size, safe_size),
            ".*Assertion.*ghost_val != (NULL|__null)*");
    }

    // SetDataPtrCOO
    {
        int* null_int  = nullptr;
        T*   null_data = nullptr;
        ASSERT_DEATH(mat.SetDataPtrCOO(
                         nullptr, &idata, &data, &idata, &idata, &data, "", safe_size, safe_size),
                     ".*Assertion.*local_row != (NULL|__null)*");
        ASSERT_DEATH(mat.SetDataPtrCOO(
                         &idata, nullptr, &data, &idata, &idata, &data, "", safe_size, safe_size),
                     ".*Assertion.*local_col != (NULL|__null)*");
        ASSERT_DEATH(mat.SetDataPtrCOO(
                         &idata, &idata, nullptr, &idata, &idata, &data, "", safe_size, safe_size),
                     ".*Assertion.*local_val != (NULL|__null)*");
        ASSERT_DEATH(mat.SetDataPtrCOO(
                         &idata, &idata, &data, nullptr, &idata, &data, "", safe_size, safe_size),
                     ".*Assertion.*ghost_row != (NULL|__null)*");
        ASSERT_DEATH(mat.SetDataPtrCOO(
                         &idata, &idata, &data, &idata, nullptr, &data, "", safe_size, safe_size),
                     ".*Assertion.*ghost_col != (NULL|__null)*");
        ASSERT_DEATH(mat.SetDataPtrCOO(
                         &idata, &idata, &data, &idata, &idata, nullptr, "", safe_size, safe_size),
                     ".*Assertion.*ghost_val != (NULL|__null)*");
        ASSERT_DEATH(mat.SetDataPtrCOO(
                         &null_int, &idata, &data, &idata, &idata, &data, "", safe_size, safe_size),
                     ".*Assertion.*local_row != (NULL|__null)*");
        ASSERT_DEATH(mat.SetDataPtrCOO(
                         &idata, &null_int, &data, &idata, &idata, &data, "", safe_size, safe_size),
                     ".*Assertion.*local_col != (NULL|__null)*");
        ASSERT_DEATH(
            mat.SetDataPtrCOO(
                &idata, &idata, &null_data, &idata, &idata, &data, "", safe_size, safe_size),
            ".*Assertion.*local_val != (NULL|__null)*");
        ASSERT_DEATH(mat.SetDataPtrCOO(
                         &idata, &idata, &data, &null_int, &idata, &data, "", safe_size, safe_size),
                     ".*Assertion.*ghost_row != (NULL|__null)*");
        ASSERT_DEATH(mat.SetDataPtrCOO(
                         &idata, &idata, &data, &idata, &null_int, &data, "", safe_size, safe_size),
                     ".*Assertion.*ghost_col != (NULL|__null)*");
        ASSERT_DEATH(
            mat.SetDataPtrCOO(
                &idata, &idata, &data, &idata, &idata, &null_data, "", safe_size, safe_size),
            ".*Assertion.*ghost_val != (NULL|__null)*");
    }

    // SetLocalDataPtrCSR
    {
        int* null_int  = nullptr;
        T*   null_data = nullptr;
        ASSERT_DEATH(mat.SetLocalDataPtrCSR(nullptr, &idata, &data, "", safe_size),
                     ".*Assertion.*row_offset != (NULL|__null)*");
        ASSERT_DEATH(mat.SetLocalDataPtrCSR(&idata, nullptr, &data, "", safe_size),
                     ".*Assertion.*col != (NULL|__null)*");
        ASSERT_DEATH(mat.SetLocalDataPtrCSR(&idata, &idata, nullptr, "", safe_size),
                     ".*Assertion.*val != (NULL|__null)*");
        ASSERT_DEATH(mat.SetLocalDataPtrCSR(&null_int, &idata, &data, "", safe_size),
                     ".*Assertion.*row_offset != (NULL|__null)*");
        ASSERT_DEATH(mat.SetLocalDataPtrCSR(&idata, &null_int, &data, "", safe_size),
                     ".*Assertion.*col != (NULL|__null)*");
        ASSERT_DEATH(mat.SetLocalDataPtrCSR(&idata, &idata, &null_data, "", safe_size),
                     ".*Assertion.*val != (NULL|__null)*");
    }

    // SetGhostDataPtrCSR
    {
        int* null_int  = nullptr;
        T*   null_data = nullptr;
        ASSERT_DEATH(mat.SetGhostDataPtrCSR(nullptr, &idata, &data, "", safe_size),
                     ".*Assertion.*row_offset != (NULL|__null)*");
        ASSERT_DEATH(mat.SetGhostDataPtrCSR(&idata, nullptr, &data, "", safe_size),
                     ".*Assertion.*col != (NULL|__null)*");
        ASSERT_DEATH(mat.SetGhostDataPtrCSR(&idata, &idata, nullptr, "", safe_size),
                     ".*Assertion.*val != (NULL|__null)*");
        ASSERT_DEATH(mat.SetGhostDataPtrCSR(&null_int, &idata, &data, "", safe_size),
                     ".*Assertion.*row_offset != (NULL|__null)*");
        ASSERT_DEATH(mat.SetGhostDataPtrCSR(&idata, &null_int, &data, "", safe_size),
                     ".*Assertion.*col != (NULL|__null)*");
        ASSERT_DEATH(mat.SetGhostDataPtrCSR(&idata, &idata, &null_data, "", safe_size),
                     ".*Assertion.*val != (NULL|__null)*");
    }

    // SetLocalDataPtrCOO
    {
        int* null_int  = nullptr;
        T*   null_data = nullptr;
        ASSERT_DEATH(mat.SetLocalDataPtrCOO(nullptr, &idata, &data, "", safe_size),
                     ".*Assertion.*row != (NULL|__null)*");
        ASSERT_DEATH(mat.SetLocalDataPtrCOO(&idata, nullptr, &data, "", safe_size),
                     ".*Assertion.*col != (NULL|__null)*");
        ASSERT_DEATH(mat.SetLocalDataPtrCOO(&idata, &idata, nullptr, "", safe_size),
                     ".*Assertion.*val != (NULL|__null)*");
        ASSERT_DEATH(mat.SetLocalDataPtrCOO(&null_int, &idata, &data, "", safe_size),
                     ".*Assertion.*row != (NULL|__null)*");
        ASSERT_DEATH(mat.SetLocalDataPtrCOO(&idata, &null_int, &data, "", safe_size),
                     ".*Assertion.*col != (NULL|__null)*");
        ASSERT_DEATH(mat.SetLocalDataPtrCOO(&idata, &idata, &null_data, "", safe_size),
                     ".*Assertion.*val != (NULL|__null)*");
    }

    // SetGhostDataPtrCOO
    {
        int* null_int  = nullptr;
        T*   null_data = nullptr;
        ASSERT_DEATH(mat.SetGhostDataPtrCOO(nullptr, &idata, &data, "", safe_size),
                     ".*Assertion.*row != (NULL|__null)*");
        ASSERT_DEATH(mat.SetGhostDataPtrCOO(&idata, nullptr, &data, "", safe_size),
                     ".*Assertion.*col != (NULL|__null)*");
        ASSERT_DEATH(mat.SetGhostDataPtrCOO(&idata, &idata, nullptr, "", safe_size),
                     ".*Assertion.*val != (NULL|__null)*");
        ASSERT_DEATH(mat.SetGhostDataPtrCOO(&null_int, &idata, &data, "", safe_size),
                     ".*Assertion.*row != (NULL|__null)*");
        ASSERT_DEATH(mat.SetGhostDataPtrCOO(&idata, &null_int, &data, "", safe_size),
                     ".*Assertion.*col != (NULL|__null)*");
        ASSERT_DEATH(mat.SetGhostDataPtrCOO(&idata, &idata, &null_data, "", safe_size),
                     ".*Assertion.*val != (NULL|__null)*");
    }

    // LeaveDataPtrCSR
    {
        int* null_int  = nullptr;
        T*   null_data = nullptr;
        ASSERT_DEATH(
            mat.LeaveDataPtrCSR(&idata, &null_int, &null_data, &null_int, &null_int, &null_data),
            ".*Assertion.*local_row_offset == (NULL|__null)*");
        ASSERT_DEATH(
            mat.LeaveDataPtrCSR(&null_int, &idata, &null_data, &null_int, &null_int, &null_data),
            ".*Assertion.*local_col == (NULL|__null)*");
        ASSERT_DEATH(
            mat.LeaveDataPtrCSR(&null_int, &null_int, &data, &null_int, &null_int, &null_data),
            ".*Assertion.*local_val == (NULL|__null)*");
        ASSERT_DEATH(
            mat.LeaveDataPtrCSR(&null_int, &null_int, &null_data, &idata, &null_int, &null_data),
            ".*Assertion.*ghost_row_offset == (NULL|__null)*");
        ASSERT_DEATH(
            mat.LeaveDataPtrCSR(&null_int, &null_int, &null_data, &null_int, &idata, &null_data),
            ".*Assertion.*ghost_col == (NULL|__null)*");
        ASSERT_DEATH(
            mat.LeaveDataPtrCSR(&null_int, &null_int, &null_data, &null_int, &null_int, &data),
            ".*Assertion.*ghost_val == (NULL|__null)*");
    }

    // LeaveDataPtrCOO
    {
        int* null_int  = nullptr;
        T*   null_data = nullptr;
        ASSERT_DEATH(
            mat.LeaveDataPtrCOO(&idata, &null_int, &null_data, &null_int, &null_int, &null_data),
            ".*Assertion.*local_row == (NULL|__null)*");
        ASSERT_DEATH(
            mat.LeaveDataPtrCOO(&null_int, &idata, &null_data, &null_int, &null_int, &null_data),
            ".*Assertion.*local_col == (NULL|__null)*");
        ASSERT_DEATH(
            mat.LeaveDataPtrCOO(&null_int, &null_int, &data, &null_int, &null_int, &null_data),
            ".*Assertion.*local_val == (NULL|__null)*");
        ASSERT_DEATH(
            mat.LeaveDataPtrCOO(&null_int, &null_int, &null_data, &idata, &null_int, &null_data),
            ".*Assertion.*ghost_row == (NULL|__null)*");
        ASSERT_DEATH(
            mat.LeaveDataPtrCOO(&null_int, &null_int, &null_data, &null_int, &idata, &null_data),
            ".*Assertion.*ghost_col == (NULL|__null)*");
        ASSERT_DEATH(
            mat.LeaveDataPtrCOO(&null_int, &null_int, &null_data, &null_int, &null_int, &data),
            ".*Assertion.*ghost_val == (NULL|__null)*");
    }

    // LeaveLocalDataPtrCSR
    {
        int* null_int  = nullptr;
        T*   null_data = nullptr;
        ASSERT_DEATH(mat.LeaveLocalDataPtrCSR(&idata, &null_int, &null_data),
                     ".*Assertion.*row_offset == (NULL|__null)*");
        ASSERT_DEATH(mat.LeaveLocalDataPtrCSR(&null_int, &idata, &null_data),
                     ".*Assertion.*col == (NULL|__null)*");
        ASSERT_DEATH(mat.LeaveLocalDataPtrCSR(&null_int, &null_int, &data),
                     ".*Assertion.*val == (NULL|__null)*");
    }

    // LeaveGhostDataPtrCSR
    {
        int* null_int  = nullptr;
        T*   null_data = nullptr;
        ASSERT_DEATH(mat.LeaveGhostDataPtrCSR(&idata, &null_int, &null_data),
                     ".*Assertion.*row_offset == (NULL|__null)*");
        ASSERT_DEATH(mat.LeaveGhostDataPtrCSR(&null_int, &idata, &null_data),
                     ".*Assertion.*col == (NULL|__null)*");
        ASSERT_DEATH(mat.LeaveGhostDataPtrCSR(&null_int, &null_int, &data),
                     ".*Assertion.*val == (NULL|__null)*");
    }

    // LeaveLocalDataPtrCOO
    {
        int* null_int  = nullptr;
        T*   null_data = nullptr;
        ASSERT_DEATH(mat.LeaveLocalDataPtrCOO(&idata, &null_int, &null_data),
                     ".*Assertion.*row == (NULL|__null)*");
        ASSERT_DEATH(mat.LeaveLocalDataPtrCOO(&null_int, &idata, &null_data),
                     ".*Assertion.*col == (NULL|__null)*");
        ASSERT_DEATH(mat.LeaveLocalDataPtrCOO(&null_int, &null_int, &data),
                     ".*Assertion.*val == (NULL|__null)*");
    }

    // LeaveGhostDataPtrCOO
    {
        int* null_int  = nullptr;
        T*   null_data = nullptr;
        ASSERT_DEATH(mat.LeaveGhostDataPtrCOO(&idata, &null_int, &null_data),
                     ".*Assertion.*row == (NULL|__null)*");
        ASSERT_DEATH(mat.LeaveGhostDataPtrCOO(&null_int, &idata, &null_data),
                     ".*Assertion.*col == (NULL|__null)*");
        ASSERT_DEATH(mat.LeaveGhostDataPtrCOO(&null_int, &null_int, &data),
                     ".*Assertion.*val == (NULL|__null)*");
    }

    // Apply
    {
        GlobalVector<T>* null_vec = nullptr;
        ASSERT_DEATH(mat.Apply(vec, null_vec), ".*Assertion.*out != (NULL|__null)*");
    }

    // ApplyAdd
    {
        GlobalVector<T>* null_vec = nullptr;
        ASSERT_DEATH(mat.ApplyAdd(vec, 1.0, null_vec), ".*Assertion.*out != (NULL|__null)*");
    }

    // ExtractInverseDiagonal
    {
        GlobalVector<T>* null_vec = nullptr;
        ASSERT_DEATH(mat.ExtractInverseDiagonal(null_vec),
                     ".*Assertion.*vec_inv_diag != (NULL|__null)*");
    }

    // InitialPairwiseAggregation
    {
        int               val;
        int*              null_int = nullptr;
        LocalVector<int>  lvint;
        LocalVector<int>* null_vec = nullptr;
        ASSERT_DEATH(mat.InitialPairwiseAggregation(0.1, val, null_vec, val, &null_int, val, 0),
                     ".*Assertion.*G != (NULL|__null)*");
        ASSERT_DEATH(mat.InitialPairwiseAggregation(0.1, val, &lvint, val, &idata, val, 0),
                     ".*Assertion.*rG == (NULL|__null)*");
    }

    // FurtherPairwiseAggregation
    {
        int               val;
        int*              null_int = nullptr;
        LocalVector<int>  lvint;
        LocalVector<int>* null_vec = nullptr;
        ASSERT_DEATH(mat.FurtherPairwiseAggregation(0.1, val, null_vec, val, &idata, val, 0),
                     ".*Assertion.*G != (NULL|__null)*");
        ASSERT_DEATH(mat.FurtherPairwiseAggregation(0.1, val, &lvint, val, &null_int, val, 0),
                     ".*Assertion.*rG != (NULL|__null)*");
    }

    // CoarsenOperator
    {
        GlobalMatrix<T>  Ac;
        ParallelManager  pm;
        LocalVector<int> lvint;
        GlobalMatrix<T>* null_mat = nullptr;
        int*             null_int = nullptr;
        ASSERT_DEATH(
            mat.CoarsenOperator(null_mat, safe_size, safe_size, lvint, safe_size, idata, safe_size),
            ".*Assertion.*Ac != (NULL|__null)*");
        ASSERT_DEATH(
            mat.CoarsenOperator(&Ac, safe_size, safe_size, lvint, safe_size, null_int, safe_size),
            ".*Assertion.*rG != (NULL|__null)*");
    }

    free_host(&idata);
    free_host(&data);

    // Stop rocALUTION
    stop_rocalution();
}

// Each rank owns a tridiagonal block without neighbors and sets only its local data.
// The ghost part must still have one empty row per local row.
template <typename T>
void testing_global_matrix_local_data_only(bool coo, bool use_acc)
{
    if(!testing_mpi_launched())
    {
        GTEST_SKIP() << "needs an MPI launcher (mpirun)";
    }

    const MPI_Comm* comm = testing_mpi_comm();

    int num_procs;
    MPI_Comm_size(*comm, &num_procs);

    // GlobalMatrix::Apply() synchronizes the accelerator even when it is disabled, so the host
    // runs keep it enabled and leave the objects on the host.
    disable_accelerator_rocalution(false);
    set_device_rocalution(device);
    init_rocalution();

    int n   = 100;
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

    ParallelManager pm;
    pm.SetMPICommunicator(comm);
    pm.SetGlobalNrow(static_cast<int64_t>(n) * num_procs);
    pm.SetGlobalNcol(static_cast<int64_t>(n) * num_procs);
    pm.SetLocalNrow(n);
    pm.SetLocalNcol(n);
    pm.SetBoundaryIndex(0, nullptr);
    pm.SetReceivers(0, nullptr, zero_offset);
    pm.SetSenders(0, nullptr, zero_offset);

    GlobalMatrix<T> mat;
    mat.SetParallelManager(pm);

    if(coo)
    {
        mat.SetLocalDataPtrCOO(&row, &col, &val, "mat", nnz);
    }
    else
    {
        mat.SetLocalDataPtrCSR(&row, &col, &val, "mat", nnz);
    }

    EXPECT_EQ(mat.GetGhost().GetM(), n);
    EXPECT_EQ(mat.GetGhost().GetN(), 0);
    EXPECT_EQ(mat.GetGhost().GetNnz(), 0);

    GlobalVector<T> x(pm);
    GlobalVector<T> y(pm);

    if(use_acc)
    {
        mat.MoveToAccelerator();
        x.MoveToAccelerator();
        y.MoveToAccelerator();

        EXPECT_EQ(mat.GetGhost().GetM(), n);
    }

    x.Allocate("x", mat.GetN());
    y.Allocate("y", mat.GetM());
    x.Ones();
    mat.Apply(x, &y);

    T* hy = nullptr;
    allocate_host(n, &hy);
    y.GetInterior().CopyToHostData(hy);

    for(int i = 0; i < n; ++i)
    {
        EXPECT_EQ(hy[i], (i == 0 || i == n - 1) ? static_cast<T>(2) : static_cast<T>(1));
    }

    free_host(&hy);

    mat.Clear();
    x.Clear();
    y.Clear();

    stop_rocalution();
}

#endif // TESTING_GLOBAL_MATRIX_HPP

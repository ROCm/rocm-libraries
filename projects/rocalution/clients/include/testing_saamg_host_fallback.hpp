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
#ifndef TESTING_SAAMG_HOST_FALLBACK_HPP
#define TESTING_SAAMG_HOST_FALLBACK_HPP

#include "utility.hpp"

#include <gtest/gtest.h>
#include <memory>
#include <rocalution/rocalution.hpp>
#include <vector>

using namespace rocalution;

// The smoothed aggregation fill kernel keeps one row of the prolongation operator in LDS and
// hands the operator back to the host as soon as a row needs 512 entries or more. The nnz count
// that runs before it has a much wider limit of 1024 entries per row of A, so a row of A that
// stays below 1024 can still drive the fill over its own limit.
static constexpr int saamg_fill_capacity = 512;

// Build a star matrix with num_rays rays and put every node into an aggregate of its own. The
// tentative prolongation operator is then the identity and the smoothed operator is
// I - relax * D^-1 * A, so the centre row holds one entry per ray plus its own. num_rays alone
// decides whether the fill fits into LDS.
//
//   row 0     : [num_rays, -1, -1, ... , -1]
//   row i > 0 : [-1, 0, ... , 2 (at column i), ... , 0]
template <typename T>
static int gen_saamg_wide_prolong_row(int                   num_rays,
                                      std::vector<PtrType>& csr_ptr,
                                      std::vector<int>&     csr_col,
                                      std::vector<T>&       csr_val,
                                      std::vector<int64_t>& aggregates,
                                      std::vector<int64_t>& roots,
                                      std::vector<uint8_t>& connections)
{
    int nrow = num_rays + 1;
    int nnz  = 3 * num_rays + 1;

    csr_ptr.resize(nrow + 1);
    csr_col.resize(nnz);
    csr_val.resize(nnz);
    aggregates.resize(nrow);
    roots.resize(nrow);
    connections.resize(nnz);

    PtrType idx = 0;

    // Centre row, diagonal first so that the columns stay ascending
    csr_ptr[0]       = idx;
    csr_col[idx]     = 0;
    csr_val[idx]     = static_cast<T>(num_rays);
    connections[idx] = 0;
    ++idx;

    for(int i = 1; i <= num_rays; ++i)
    {
        csr_col[idx]     = i;
        csr_val[idx]     = static_cast<T>(-1);
        connections[idx] = 1;
        ++idx;
    }

    // Ray rows
    for(int i = 1; i <= num_rays; ++i)
    {
        csr_ptr[i] = idx;

        csr_col[idx]     = 0;
        csr_val[idx]     = static_cast<T>(-1);
        connections[idx] = 1;
        ++idx;

        csr_col[idx]     = i;
        csr_val[idx]     = static_cast<T>(2);
        connections[idx] = 0;
        ++idx;
    }

    csr_ptr[nrow] = idx;

    // Every node is the root of its own aggregate
    for(int i = 0; i < nrow; ++i)
    {
        aggregates[i] = i;
        roots[i]      = i;
    }

    return nrow;
}

template <typename T>
static T saamg_tolerance()
{
    return std::is_same<T, float>::value ? static_cast<T>(1e-5) : static_cast<T>(1e-12);
}

// Regression test for the host fallback of the smoothed aggregation prolongation. The fill ran
// in a branch whose return value was discarded, so a row that did not fit into LDS left the
// prolongation operator empty instead of falling back to the host. Running the same aggregation
// on the host and on the accelerator and comparing the two operators catches that.
template <typename T>
void testing_saamg_host_fallback(Arguments argus)
{
    int num_rays = argus.size;
    T   relax    = static_cast<T>(argus.alpha);

    std::vector<PtrType> csr_ptr;
    std::vector<int>     csr_col;
    std::vector<T>       csr_val;
    std::vector<int64_t> aggregates;
    std::vector<int64_t> roots;
    std::vector<uint8_t> connections;

    int nrow = gen_saamg_wide_prolong_row(
        num_rays, csr_ptr, csr_col, csr_val, aggregates, roots, connections);
    int nnz = csr_ptr[nrow];

    // LocalVector<bool> needs a contiguous bool array, which std::vector<bool> cannot provide
    std::unique_ptr<bool[]> connection_data(new bool[nnz]);
    for(int i = 0; i < nnz; ++i)
    {
        connection_data[i] = (connections[i] != 0);
    }

    // Initialize rocALUTION platform
    disable_accelerator_rocalution(false);
    set_device_rocalution(device);
    init_rocalution();

    LocalMatrix<T>       A;
    LocalVector<bool>    conn;
    LocalVector<int64_t> agg;
    LocalVector<int64_t> agg_roots;

    A.AllocateCSR("A", nnz, nrow, nrow);
    A.CopyFromCSR(csr_ptr.data(), csr_col.data(), csr_val.data());

    conn.Allocate("connections", nnz);
    conn.CopyFromData(connection_data.get());

    agg.Allocate("aggregates", nrow);
    agg.CopyFromData(aggregates.data());

    agg_roots.Allocate("aggregate root nodes", nrow);
    agg_roots.CopyFromData(roots.data());

    // Reference operator, computed entirely on the host
    LocalMatrix<T> P_host;
    A.AMGSmoothedAggregation(relax, conn, agg, agg_roots, &P_host);
    P_host.Sort();

    std::vector<PtrType> host_ptr(P_host.GetM() + 1);
    std::vector<int>     host_col(P_host.GetNnz());
    std::vector<T>       host_val(P_host.GetNnz());
    P_host.CopyToCSR(host_ptr.data(), host_col.data(), host_val.data());

    // The centre row is what drives the fill across the LDS capacity. Assert it, so that the
    // test reports loudly if a future change stops covering the fallback.
    PtrType widest_row = 0;
    for(int i = 0; i < P_host.GetM(); ++i)
    {
        widest_row = std::max(widest_row, host_ptr[i + 1] - host_ptr[i]);
    }

    if(num_rays >= saamg_fill_capacity)
    {
        EXPECT_GE(widest_row, saamg_fill_capacity)
            << "the prolongation operator no longer exceeds the LDS capacity, so this test "
               "stopped covering the host fallback";
    }
    else
    {
        EXPECT_LT(widest_row, saamg_fill_capacity);
    }

    // Same aggregation on the accelerator, which takes the host fallback for the wide row
    A.MoveToAccelerator();
    conn.MoveToAccelerator();
    agg.MoveToAccelerator();
    agg_roots.MoveToAccelerator();

    LocalMatrix<T> P_acc;
    P_acc.CloneBackend(A);

    A.AMGSmoothedAggregation(relax, conn, agg, agg_roots, &P_acc);

    P_acc.MoveToHost();
    P_acc.Sort();

    EXPECT_EQ(P_acc.GetM(), P_host.GetM());
    EXPECT_EQ(P_acc.GetN(), P_host.GetN());
    EXPECT_EQ(P_acc.GetNnz(), P_host.GetNnz());

    if(P_acc.GetM() == P_host.GetM() && P_acc.GetNnz() == P_host.GetNnz())
    {
        std::vector<PtrType> acc_ptr(P_acc.GetM() + 1);
        std::vector<int>     acc_col(P_acc.GetNnz());
        std::vector<T>       acc_val(P_acc.GetNnz());
        P_acc.CopyToCSR(acc_ptr.data(), acc_col.data(), acc_val.data());

        int bad_rows = 0;
        for(int i = 0; i <= P_acc.GetM(); ++i)
        {
            if(acc_ptr[i] != host_ptr[i])
            {
                if(bad_rows++ == 0)
                {
                    EXPECT_EQ(acc_ptr[i], host_ptr[i]) << "row offset " << i << " differs";
                }
            }
        }
        EXPECT_EQ(bad_rows, 0);

        int bad_entries = 0;
        for(int64_t j = 0; j < P_acc.GetNnz(); ++j)
        {
            bool equal = acc_col[j] == host_col[j]
                         && std::abs(acc_val[j] - host_val[j])
                                <= saamg_tolerance<T>() * std::abs(host_val[j]);

            if(!equal)
            {
                if(bad_entries++ == 0)
                {
                    EXPECT_EQ(acc_col[j], host_col[j]) << "column of entry " << j << " differs";
                    EXPECT_NEAR(
                        acc_val[j], host_val[j], saamg_tolerance<T>() * std::abs(host_val[j]))
                        << "value of entry " << j << " differs";
                }
            }
        }
        EXPECT_EQ(bad_entries, 0);
    }

    stop_rocalution();
}

#endif // TESTING_SAAMG_HOST_FALLBACK_HPP

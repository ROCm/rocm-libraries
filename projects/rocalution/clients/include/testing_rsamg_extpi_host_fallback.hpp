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
#ifndef TESTING_RSAMG_EXTPI_HOST_FALLBACK_HPP
#define TESTING_RSAMG_EXTPI_HOST_FALLBACK_HPP

#include "utility.hpp"

#include <gtest/gtest.h>
#include <memory>
#include <rocalution/rocalution.hpp>
#include <vector>

using namespace rocalution;

// The extended+i fill kernel keeps the interpolatory set of a row in LDS and hands the whole
// operator back to the host as soon as one row needs 4096 entries or more.
static constexpr int extpi_lds_capacity = 4096;

// Build a star matrix whose centre is the only fine point and whose num_coarse rays are all
// coarse points. Every off-diagonal entry is a strong connection, so the interpolatory set of
// the centre is the complete set of rays and row 0 of the interpolation operator ends up with
// num_coarse entries. That is the only knob the test needs to move the fill across the LDS
// capacity.
//
//   row 0     : [num_coarse, -1, -1, ... , -1]
//   row i > 0 : [-1, 0, ... , 2 (at column i), ... , 0]
//
// The matrix is a weakly diagonally dominant M-matrix, so the interpolation weights are well
// defined: the centre interpolates each ray with weight 1 / num_coarse.
template <typename T>
static int gen_extpi_wide_fine_row(int                   num_coarse,
                                   std::vector<PtrType>& csr_ptr,
                                   std::vector<int>&     csr_col,
                                   std::vector<T>&       csr_val,
                                   std::vector<int>&     cf_map,
                                   std::vector<uint8_t>& strong)
{
    int nrow = num_coarse + 1;
    int nnz  = 3 * num_coarse + 1;

    csr_ptr.resize(nrow + 1);
    csr_col.resize(nnz);
    csr_val.resize(nnz);
    cf_map.resize(nrow);
    strong.resize(nnz);

    PtrType idx = 0;

    // Centre row, diagonal first so that the columns stay ascending
    csr_ptr[0]   = idx;
    csr_col[idx] = 0;
    csr_val[idx] = static_cast<T>(num_coarse);
    strong[idx]  = 0;
    ++idx;

    for(int i = 1; i <= num_coarse; ++i)
    {
        csr_col[idx] = i;
        csr_val[idx] = static_cast<T>(-1);
        strong[idx]  = 1;
        ++idx;
    }

    // Ray rows
    for(int i = 1; i <= num_coarse; ++i)
    {
        csr_ptr[i] = idx;

        csr_col[idx] = 0;
        csr_val[idx] = static_cast<T>(-1);
        strong[idx]  = 1;
        ++idx;

        csr_col[idx] = i;
        csr_val[idx] = static_cast<T>(2);
        strong[idx]  = 0;
        ++idx;
    }

    csr_ptr[nrow] = idx;

    // The centre is fine, every ray is coarse
    cf_map[0] = 0;
    for(int i = 1; i <= num_coarse; ++i)
    {
        cf_map[i] = 1;
    }

    return nrow;
}

template <typename T>
static T extpi_tolerance()
{
    return std::is_same<T, float>::value ? static_cast<T>(1e-5) : static_cast<T>(1e-12);
}

// Regression test for the host fallback of the extended+i interpolation. The fill used to scan
// the row offsets of the interpolation operator in place before deciding that the row did not
// fit into LDS. The host then scanned the already scanned offsets a second time and produced a
// garbage operator. Running the same interpolation on the host and on the accelerator and
// comparing the two operators catches that.
template <typename T>
void testing_rsamg_extpi_host_fallback(Arguments argus)
{
    int num_coarse = argus.size;

    std::vector<PtrType> csr_ptr;
    std::vector<int>     csr_col;
    std::vector<T>       csr_val;
    std::vector<int>     cf_map;
    std::vector<uint8_t> strong;

    int nrow = gen_extpi_wide_fine_row(num_coarse, csr_ptr, csr_col, csr_val, cf_map, strong);
    int nnz  = csr_ptr[nrow];

    // LocalVector<bool> needs a contiguous bool array, which std::vector<bool> cannot provide
    std::unique_ptr<bool[]> strong_data(new bool[nnz]);
    for(int i = 0; i < nnz; ++i)
    {
        strong_data[i] = (strong[i] != 0);
    }

    // Initialize rocALUTION platform
    disable_accelerator_rocalution(false);
    set_device_rocalution(device);
    init_rocalution();

    LocalMatrix<T>    A;
    LocalVector<int>  CFmap;
    LocalVector<bool> S;

    A.AllocateCSR("A", nnz, nrow, nrow);
    A.CopyFromCSR(csr_ptr.data(), csr_col.data(), csr_val.data());

    CFmap.Allocate("CFmap", nrow);
    CFmap.CopyFromData(cf_map.data());

    S.Allocate("S", nnz);
    S.CopyFromData(strong_data.get());

    // Reference operator, computed entirely on the host
    LocalMatrix<T> P_host;
    A.RSExtPIInterpolation(CFmap, S, false, &P_host);
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

    if(num_coarse >= extpi_lds_capacity)
    {
        EXPECT_GE(widest_row, extpi_lds_capacity)
            << "the interpolation operator no longer exceeds the LDS capacity, so this test "
               "stopped covering the host fallback";
    }
    else
    {
        EXPECT_LT(widest_row, extpi_lds_capacity);
    }

    // Same interpolation on the accelerator, which takes the host fallback for the wide row
    A.MoveToAccelerator();
    CFmap.MoveToAccelerator();
    S.MoveToAccelerator();

    LocalMatrix<T> P_acc;
    P_acc.CloneBackend(A);

    A.RSExtPIInterpolation(CFmap, S, false, &P_acc);

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
                                <= extpi_tolerance<T>() * std::abs(host_val[j]);

            if(!equal)
            {
                if(bad_entries++ == 0)
                {
                    EXPECT_EQ(acc_col[j], host_col[j]) << "column of entry " << j << " differs";
                    EXPECT_NEAR(
                        acc_val[j], host_val[j], extpi_tolerance<T>() * std::abs(host_val[j]))
                        << "value of entry " << j << " differs";
                }
            }
        }
        EXPECT_EQ(bad_entries, 0);
    }

    stop_rocalution();
}

#endif // TESTING_RSAMG_EXTPI_HOST_FALLBACK_HPP

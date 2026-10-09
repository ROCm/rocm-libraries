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
#ifndef TESTING_RSAMG_TRUNCATION_HPP
#define TESTING_RSAMG_TRUNCATION_HPP

#include "utility.hpp"

#include <rocalution/rocalution.hpp>
#include <vector>

using namespace rocalution;

static float truncation_tolerance(float)
{
    return 1e-4f;
}

static double truncation_tolerance(double)
{
    return 1e-10;
}

// A prolongation-like operator: every row holds the same set of distinct positive
// magnitudes, so which entries truncation has to drop is unambiguous and the row sum is
// far from zero, which means the rescaling is always exercised.
template <typename T>
static void gen_truncation_operator(int nrow, PtrType** csr_ptr, int** csr_col, T** csr_val)
{
    constexpr int row_nnz = 6;

    // rocALUTION takes ownership of these below, so they have to come from the same
    // allocator that free_host and the library's internal deallocation use
    allocate_host(nrow + 1, csr_ptr);
    allocate_host(nrow * row_nnz, csr_col);
    allocate_host(nrow * row_nnz, csr_val);

    for(int i = 0; i < nrow; ++i)
    {
        (*csr_ptr)[i] = i * row_nnz;

        for(int j = 0; j < row_nnz; ++j)
        {
            (*csr_col)[i * row_nnz + j] = j;
            (*csr_val)[i * row_nnz + j] = static_cast<T>(j + 1);
        }
    }

    (*csr_ptr)[nrow] = nrow * row_nnz;
}

// Truncate the operator on the requested backend and hand the result back on the host
template <typename T>
static void run_truncation(float                 trunc_factor,
                           int                   max_elmts,
                           bool                  use_acc,
                           int                   nrow,
                           std::vector<PtrType>& out_ptr,
                           std::vector<int>&     out_col,
                           std::vector<T>&       out_val)
{
    disable_accelerator_rocalution(!use_acc);
    set_device_rocalution(device);
    init_rocalution();

    PtrType* csr_ptr = NULL;
    int*     csr_col = NULL;
    T*       csr_val = NULL;

    gen_truncation_operator(nrow, &csr_ptr, &csr_col, &csr_val);

    int64_t nnz = csr_ptr[nrow];

    {
        LocalMatrix<T> P;
        P.SetDataPtrCSR(&csr_ptr, &csr_col, &csr_val, "P", nnz, nrow, 6);

        if(use_acc)
        {
            P.MoveToAccelerator();
        }

        P.RSInterpolationTruncation(trunc_factor, max_elmts);

        P.MoveToHost();

        int64_t nnz_trunc = P.GetNnz();

        P.LeaveDataPtrCSR(&csr_ptr, &csr_col, &csr_val);

        out_ptr.assign(csr_ptr, csr_ptr + nrow + 1);
        out_col.assign(csr_col, csr_col + nnz_trunc);
        out_val.assign(csr_val, csr_val + nnz_trunc);

        free_host(&csr_ptr);
        free_host(&csr_col);
        free_host(&csr_val);
    }

    stop_rocalution();
    disable_accelerator_rocalution(false);
}

// The cap on the number of entries per row, the preservation of the row sum and the
// ascending column order are the full specification of the truncation post-pass
template <typename T>
bool testing_rsamg_truncation(Arguments argus)
{
    float trunc_factor = argus.trunc_factor;
    int   max_elmts    = argus.p_max_elmts;
    bool  use_acc      = argus.use_acc;
    int   nrow         = argus.size;

    std::vector<PtrType> ptr;
    std::vector<int>     col;
    std::vector<T>       val;

    run_truncation<T>(trunc_factor, max_elmts, use_acc, nrow, ptr, col, val);

    if(static_cast<int>(ptr.size()) != nrow + 1)
    {
        return false;
    }

    // Row sum of the untruncated operator, which the rescaling has to reproduce
    T expected_sum = static_cast<T>(0);

    for(int j = 0; j < 6; ++j)
    {
        expected_sum += static_cast<T>(j + 1);
    }

    T tol = static_cast<T>(truncation_tolerance(T())) * expected_sum;

    for(int i = 0; i < nrow; ++i)
    {
        int count = static_cast<int>(ptr[i + 1] - ptr[i]);

        if(count <= 0)
        {
            return false;
        }

        if(max_elmts > 0 && count > max_elmts)
        {
            return false;
        }

        // Nothing may be dropped when no truncation was requested
        if(trunc_factor <= 0.0f && max_elmts <= 0 && count != 6)
        {
            return false;
        }

        T sum = static_cast<T>(0);

        for(PtrType j = ptr[i]; j < ptr[i + 1]; ++j)
        {
            if(j > ptr[i] && col[j] <= col[j - 1])
            {
                return false;
            }

            sum += val[j];
        }

        if(sum - expected_sum > tol || expected_sum - sum > tol)
        {
            return false;
        }
    }

    return true;
}

// Host and accelerator must agree entry for entry, which is the property the two
// implementations of the post-pass are written to guarantee
template <typename T>
bool testing_rsamg_truncation_parity(Arguments argus)
{
    float trunc_factor = argus.trunc_factor;
    int   max_elmts    = argus.p_max_elmts;
    int   nrow         = argus.size;

    std::vector<PtrType> host_ptr;
    std::vector<int>     host_col;
    std::vector<T>       host_val;

    std::vector<PtrType> acc_ptr;
    std::vector<int>     acc_col;
    std::vector<T>       acc_val;

    run_truncation<T>(trunc_factor, max_elmts, false, nrow, host_ptr, host_col, host_val);
    run_truncation<T>(trunc_factor, max_elmts, true, nrow, acc_ptr, acc_col, acc_val);

    if(host_ptr != acc_ptr || host_col != acc_col)
    {
        return false;
    }

    if(host_val.size() != acc_val.size())
    {
        return false;
    }

    T tol = static_cast<T>(truncation_tolerance(T()));

    for(size_t i = 0; i < host_val.size(); ++i)
    {
        T diff = host_val[i] - acc_val[i];

        if(diff > tol || -diff > tol)
        {
            return false;
        }
    }

    return true;
}

#endif // TESTING_RSAMG_TRUNCATION_HPP

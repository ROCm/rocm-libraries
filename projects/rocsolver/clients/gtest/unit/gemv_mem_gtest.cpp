/* **************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#include <algorithm>
#include <gtest/gtest.h>

// The helper under test. This file is compiled with the internal API enabled (see
// clients/gtest/CMakeLists.txt) so that the tests exercise the routine rocSOLVER
// actually ships, rather than a copy of it.
#include "rocblas.hpp"

// The bound produced by the shipped routine.
template <typename T>
static size_t
    gemv_mem_max(rocblas_operation transA, rocblas_int m, rocblas_int n, rocblas_int batch_count)
{
    size_t w = 0;
    rocsolver::rocblasCall_gemv_mem_max<false, T>(transA, m, n, batch_count, &w);
    return w;
}

// Exhaustive requirement over every smaller call (m' <= m, n' <= n).
template <typename T>
static size_t
    gemv_mem_brute(rocblas_operation transA, rocblas_int m, rocblas_int n, rocblas_int batch_count)
{
    size_t worst = 0;
    for(rocblas_int mm = 1; mm <= m; ++mm)
        for(rocblas_int nn = 1; nn <= n; ++nn)
            worst = std::max(
                worst, rocblas_internal_gemv_kernel_workspace_size<T>(transA, mm, nn, batch_count));
    return worst;
}

// The bound must cover every smaller call. A factorization sizes this workspace once
// from the full (m, n) and then calls larf over a shrinking trailing submatrix, so
// anything the bound misses is a buffer overrun at run time.
template <typename T>
static void expect_covers_all(rocblas_operation transA,
                              rocblas_int m,
                              rocblas_int n,
                              rocblas_int bc)
{
    EXPECT_GE(gemv_mem_max<T>(transA, m, n, bc), gemv_mem_brute<T>(transA, m, n, bc))
        << "transA=" << transA << " m=" << m << " n=" << n << " batch_count=" << bc;
}

// Left-side larf issues a transposed gemv; right-side issues a non-transposed one.
constexpr rocblas_operation trans_op = rocblas_operation_transpose;
constexpr rocblas_operation none_op = rocblas_operation_none;

TEST(checkin_lapack, GEMV_MEM_covers_subproblems_transpose)
{
    for(rocblas_int m : {64, 600, 5000, 40000})
        for(rocblas_int n : {1, 7, 40, 130, 300})
        {
            expect_covers_all<float>(trans_op, m, n, 1);
            expect_covers_all<double>(trans_op, m, n, 1);
        }
}

TEST(checkin_lapack, GEMV_MEM_covers_subproblems_no_transpose)
{
    for(rocblas_int m : {64, 600, 5000})
        for(rocblas_int n : {1, 40, 300, 4096})
        {
            expect_covers_all<float>(none_op, m, n, 1);
            expect_covers_all<double>(none_op, m, n, 1);
        }
}

TEST(checkin_lapack, GEMV_MEM_covers_subproblems_batched)
{
    for(rocblas_int bc : {1, 3})
    {
        expect_covers_all<double>(trans_op, 5000, 130, bc);
        expect_covers_all<double>(none_op, 600, 4096, bc);
    }
}

// The requirement is not monotone in n: rocBLAS selects its optimized kernel only below
// a threshold on n, so a tall problem whose n sits above it reports nothing while a
// narrower call on the same matrix needs scratch. Querying (m, n) alone therefore
// under-reserves. This is the case that regressed; keep a direct guard on it.
TEST(checkin_lapack, GEMV_MEM_non_monotonic_in_n)
{
    const rocblas_int m = 40000;
    const rocblas_int n_wide = 300;

    size_t at_corner = rocblas_internal_gemv_kernel_workspace_size<double>(trans_op, m, n_wide, 1);
    size_t over_box = gemv_mem_max<double>(trans_op, m, n_wide, 1);

    // Largest requirement among the narrower calls on the same matrix.
    size_t narrowest = 0;
    for(rocblas_int nn = 1; nn <= n_wide; ++nn)
        narrowest = std::max(
            narrowest, rocblas_internal_gemv_kernel_workspace_size<double>(trans_op, m, nn, 1));

    EXPECT_GE(over_box, narrowest);
    if(narrowest > at_corner)
    {
        // Confirms the hazard is live in this rocBLAS, and that the bound closes it.
        EXPECT_GT(over_box, at_corner);
    }
}

TEST(checkin_lapack, GEMV_MEM_degenerate_sizes)
{
    EXPECT_EQ(gemv_mem_max<double>(trans_op, 0, 10, 1), 0u);
    EXPECT_EQ(gemv_mem_max<double>(trans_op, 10, 0, 1), 0u);
    EXPECT_EQ(gemv_mem_max<double>(trans_op, 10, 10, 0), 0u);
    EXPECT_EQ(gemv_mem_max<double>(none_op, -1, 10, 1), 0u);
}

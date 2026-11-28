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
#include <vector>

// Compiled with the internal API enabled (see clients/gtest/CMakeLists.txt) so that the
// bound is the one rocSOLVER uses.
#include "rocblas.hpp"

// rocblasCall_gemv_mem_max must be at least the rocBLAS gemv scratch of every smaller
// shape, since larf callers size their workspace once and then run on smaller matrices.
// It copies constants from rocBLAS, so this compares it with the rocBLAS query, maximized
// over all grid points (m', n') <= (m, n). The grid is dense for small sizes and includes
// the sizes where the rocBLAS kernel selection changes.
static std::vector<rocblas_int> grid()
{
    std::vector<rocblas_int> g;
    for(rocblas_int v = 1; v <= 1100; ++v)
        g.push_back(v);
    for(rocblas_int k = 1; k <= 300; ++k)
    {
        g.insert(g.end(), {1024 * k + 1, 2048 * k - 1, 2048 * k, 4096 * k, 4096 * k + 1});
    }
    std::sort(g.begin(), g.end());
    g.erase(std::unique(g.begin(), g.end()), g.end());
    return g;
}

template <bool BATCHED, typename T>
static void expect_bound(rocblas_operation transA, rocblas_int batch_count)
{
    const std::vector<rocblas_int> g = grid();
    std::vector<size_t> box(g.size(), 0); // max of the query over m' <= m, n' <= g[j]
    for(rocblas_int m : g)
    {
        size_t row = 0; // max over n' <= g[j] for this m
        for(size_t j = 0; j < g.size(); ++j)
        {
            row = std::max(
                row, rocblas_internal_gemv_kernel_workspace_size<T>(transA, m, g[j], batch_count));
            box[j] = std::max(box[j], row);

            size_t bound;
            rocsolver::rocblasCall_gemv_mem_max<BATCHED, T>(transA, m, g[j], batch_count, &bound);
            ASSERT_GE(bound, box[j]) << "transA=" << transA << " m=" << m << " n=" << g[j]
                                     << " batch_count=" << batch_count;
        }
    }
}

// non-batched with one matrix, and batched with a few batch counts
template <typename T>
static void expect_bound_all(rocblas_operation transA)
{
    expect_bound<false, T>(transA, 1);
    expect_bound<true, T>(transA, 3);
    expect_bound<true, T>(transA, 1000);
}

TEST(checkin_lapack, GEMV_MEM_MAX_transposed)
{
    expect_bound_all<float>(rocblas_operation_transpose);
    expect_bound_all<double>(rocblas_operation_transpose);
    expect_bound_all<rocblas_float_complex>(rocblas_operation_conjugate_transpose);
    expect_bound_all<rocblas_double_complex>(rocblas_operation_conjugate_transpose);
}

TEST(checkin_lapack, GEMV_MEM_MAX_not_transposed)
{
    expect_bound_all<float>(rocblas_operation_none);
    expect_bound_all<double>(rocblas_operation_none);
    expect_bound_all<rocblas_float_complex>(rocblas_operation_none);
    expect_bound_all<rocblas_double_complex>(rocblas_operation_none);
}

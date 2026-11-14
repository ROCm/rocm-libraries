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

#include <gtest/gtest.h>

// Compiled with the internal API enabled (see clients/gtest/CMakeLists.txt) so that the
// size query is the one rocSOLVER uses.
#include "rocblas.hpp"

// The larf callers size their workspace by querying each (m, n) their own loop will use
// (rocsolver_larf_getMemorySize_seq). A blocked routine such as geqrf sizes geqr2 once and
// then runs it on a smaller trailing matrix, so the actual larf calls can be smaller in one
// dimension than the shapes that were queried: m for a left larf, whose gemv is transposed,
// and n for a right larf. That is only safe if the rocBLAS gemv scratch does not decrease
// as that dimension grows, which these tests check.
template <typename T>
static void expect_nondecreasing(rocblas_operation transA, bool vary_m, rocblas_int fixed)
{
    size_t prev = 0;
    for(rocblas_int k = 1; k <= 300000; ++k)
    {
        size_t w = 0;
        rocsolver::rocblasCall_gemv_mem<false, T>(transA, vary_m ? k : fixed, vary_m ? fixed : k, 1,
                                                  &w);
        ASSERT_GE(w, prev) << "transA=" << transA << (vary_m ? " n=" : " m=") << fixed
                           << " decreases at " << (vary_m ? "m=" : "n=") << k;
        prev = w;
    }
}

TEST(checkin_lapack, GEMV_MEM_left_larf_nondecreasing_in_m)
{
    for(rocblas_int n : {1, 8, 127, 128, 300})
    {
        expect_nondecreasing<float>(rocblas_operation_transpose, true, n);
        constexpr rocblas_operation conj_trans = rocblas_operation_conjugate_transpose;
        expect_nondecreasing<rocblas_double_complex>(conj_trans, true, n);
    }
}

TEST(checkin_lapack, GEMV_MEM_right_larf_nondecreasing_in_n)
{
    for(rocblas_int m : {1, 64, 512, 1024, 4096})
    {
        expect_nondecreasing<float>(rocblas_operation_none, false, m);
        expect_nondecreasing<rocblas_double_complex>(rocblas_operation_none, false, m);
    }
}

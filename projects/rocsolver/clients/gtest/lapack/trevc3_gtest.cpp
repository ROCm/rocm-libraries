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

#include "common/lapack/testing_trevc3.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<char>> trevc3_tuple;

// each matrix_size_range vector is a {n, ldt, ldv, mtype[, gap]}
// (ldv is the leading dimension of the computed eigenvectors; the one not
// referenced, if any, is set to 1; with gap > 0 the strides are ld*n + gap
// instead of ld*n, and the gaps between the members must not be changed)
// mtype = 0: random upper triangular T,
// mtype = 1: T with clustered diagonal entries (strong growth, overflow control),
// mtype = 2: T with repeated and zero diagonal entries (small pivots),
// mtype = 3: Schur form of a random upper Hessenberg matrix,
// mtype = 4: bidiagonal T whose last vector reaches the bound of the solve, with Q = 32 *
//            a reflector (Q x overflows unless the vectors are scaled first),
// mtype = 5: T with huge off-diagonal entries (the row and column sums overflow)
// (see trevc3_initData)

// each op_range vector is a {side, howmny}

// case when n = 0, side = B and howmny = B will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<vector<char>> op_range
    = {{'R', 'A'}, {'L', 'A'}, {'B', 'A'}, {'R', 'B'}, {'L', 'B'}, {'B', 'B'}};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1, 1, 0},
    // invalid
    {-1, 1, 1, 0},
    {10, 5, 10, 0},
    {10, 10, 5, 0},
    // normal (valid) samples
    {1, 1, 1, 0},
    {2, 2, 2, 0},
    {10, 10, 10, 0},
    {33, 40, 35, 0},
    {64, 64, 64, 1},
    {65, 65, 70, 2},
    {100, 100, 100, 3},
    {130, 130, 130, 1},
    {257, 257, 257, 0},
    // overflow in the back-transformation, and in the bound on T
    {16, 16, 16, 4},
    {64, 64, 70, 4},
    {40, 40, 40, 5},
    {300, 300, 300, 5},
    // strides larger than ld*n
    {33, 40, 35, 0, 7},
    {100, 100, 100, 3, 7},
};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {500, 500, 500, 1},
    {600, 600, 600, 2},
    {1000, 1000, 1000, 3},
};

Arguments trevc3_setup_arguments(trevc3_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    vector<char> op = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", matrix_size[0]);
    arg.set<rocblas_int>("ldt", matrix_size[1]);
    arg.set<rocblas_int>("ldvl", op[0] == 'R' ? 1 : matrix_size[2]);
    arg.set<rocblas_int>("ldvr", op[0] == 'L' ? 1 : matrix_size[2]);
    arg.set<rocblas_int>("mtype", matrix_size[3]);
    arg.set<char>("side", op[0]);
    arg.set<char>("howmny", op[1]);

    // strides: defaults (ld*n), or with a gap between the members
    if(matrix_size.size() > 4 && matrix_size[4] > 0)
    {
        const rocblas_int n = matrix_size[0], gap = matrix_size[4];
        const rocblas_stride ldvl = arg.peek<rocblas_int>("ldvl");
        const rocblas_stride ldvr = arg.peek<rocblas_int>("ldvr");
        arg.set<rocblas_stride>("strideT", rocblas_stride(matrix_size[1]) * n + gap);
        arg.set<rocblas_stride>("strideVL", ldvl * n + gap);
        arg.set<rocblas_stride>("strideVR", ldvr * n + gap);
    }

    arg.timing = 0;

    return arg;
}

class TREVC3 : public ::TestWithParam<trevc3_tuple>
{
protected:
    void TearDown() override
    {
        ASSERT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = trevc3_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<char>("side") == 'B'
           && arg.peek<char>("howmny") == 'B')
            testing_trevc3_bad_arg<BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_trevc3<BATCHED, STRIDED, T>(arg);
    }
};

// non-batch tests

TEST_P(TREVC3, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(TREVC3, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(TREVC3, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(TREVC3, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched tests

TEST_P(TREVC3, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(TREVC3, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         TREVC3,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         TREVC3,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(op_range)));

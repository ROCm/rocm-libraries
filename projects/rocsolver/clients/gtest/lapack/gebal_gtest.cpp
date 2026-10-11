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

#include "common/lapack/testing_gebal.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, char> gebal_tuple;

// each matrix_size_range vector is a {n, lda, mtype}
// mtype = 0: random matrix,
// mtype = 1: badly scaled random matrix,
// mtype = 2: badly scaled random matrix with isolated eigenvalues,
// mtype = 3: permuted triangular matrix (all eigenvalues isolated),
// mtype = 4: badly scaled random matrix with a zero row and column,
// mtype = 5: random matrix with extreme scaling,
// mtype = 6: lower triangular matrix
// (see gebal_genMatrix)

// each job_range value is the balancing job (N, P, S or B)

// case when n = 0 and job = B will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<char> job_range = {'N', 'P', 'S', 'B'};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1, 2},
    // invalid
    {-1, 1, 2},
    {20, 5, 2},
    // normal (valid) samples
    {1, 1, 2},
    {2, 2, 2},
    {10, 10, 0},
    {15, 20, 1},
    {40, 40, 2},
    {100, 100, 2},
    {257, 300, 2},
    {600, 600, 1},
    {5, 5, 3},
    {300, 300, 3},
    {50, 50, 4},
    {300, 320, 4},
    {60, 60, 5},
    {300, 300, 5},
    {5, 5, 6},
    {130, 130, 6},
};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {1000, 1000, 2},
    {1000, 1024, 0},
    {2000, 2000, 1},
};

Arguments gebal_setup_arguments(gebal_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    char job = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", matrix_size[0]);
    arg.set<rocblas_int>("lda", matrix_size[1]);
    arg.set<rocblas_int>("mtype", matrix_size[2]);
    arg.set<char>("job", job);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

class GEBAL : public ::TestWithParam<gebal_tuple>
{
protected:
    void TearDown() override
    {
        ASSERT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = gebal_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<char>("job") == 'B')
            testing_gebal_bad_arg<BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_gebal<BATCHED, STRIDED, T>(arg);
    }
};

// non-batch tests

TEST_P(GEBAL, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GEBAL, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GEBAL, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GEBAL, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(GEBAL, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GEBAL, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GEBAL, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GEBAL, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched tests

TEST_P(GEBAL, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GEBAL, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GEBAL, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GEBAL, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GEBAL,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(job_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GEBAL,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(job_range)));

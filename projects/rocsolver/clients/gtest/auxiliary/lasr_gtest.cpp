/* **************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common/auxiliary/testing_lasr.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<char>> lasr_tuple;

// each matrix_size_range vector is a {M,N,lda}

// each ops_case vector is a {side, pivot, direct}

// case when m = 0, n = 0, side = 'L',  pivot = 'V' and direct = 'F' will also execute the
// bad arguments test (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 2, 2},
    {2, 0, 2},
    {1, 1, 2},
    // invalid
    {-1, 1, 1},
    {1, -1, 1},
    {15, 15, 5},
    // normal (valid) samples
    {15, 15, 15},
    {18, 18, 30},
    {25, 18, 25},
    {30, 25, 40},
    {30, 35, 30},
    {40, 50, 50}};

const vector<vector<char>> ops_case
    = {{'L', 'V', 'F'}, {'L', 'V', 'B'}, {'L', 'T', 'F'}, {'L', 'T', 'B'},
       {'L', 'B', 'F'}, {'L', 'B', 'B'}, {'R', 'V', 'F'}, {'R', 'V', 'B'},
       {'R', 'T', 'F'}, {'R', 'T', 'B'}, {'R', 'B', 'F'}, {'R', 'B', 'B'}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range
    = {{180, 250, 200}, {500, 350, 500}, {700, 700, 800}, {1024, 1000, 1024}};

Arguments lasr_setup_arguments(lasr_tuple tup)
{
    vector<int> size = std::get<0>(tup);
    vector<char> ops = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("m", size[0]);
    arg.set<rocblas_int>("n", size[1]);
    arg.set<rocblas_int>("lda", size[2]);
    arg.set<char>("side", ops[0]);
    arg.set<char>("pivot", ops[1]);
    arg.set<char>("direct", ops[2]);

    arg.timing = 0;

    return arg;
}

class LASR : public ::TestWithParam<lasr_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = lasr_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("m") == 0 && arg.peek<rocblas_int>("n") == 0
           && arg.peek<char>("side") == 'L' && arg.peek<char>("pivot") == 'V'
           && arg.peek<char>("direct") == 'F')
            testing_lasr_bad_arg<T>();

        testing_lasr<T>(arg);
    }
};

// non-batch tests

TEST_P(LASR, __float)
{
    run_tests<float>();
}

TEST_P(LASR, __double)
{
    run_tests<double>();
}

TEST_P(LASR, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(LASR, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         LASR,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(ops_case)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         LASR,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(ops_case)));

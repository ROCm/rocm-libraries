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

#include "common/lapack/testing_gebak.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<char>> gebak_tuple;

// each matrix_size_range vector is a {n, m, ldv, mtype}
// (mtype is the type of the matrix balanced with the host GEBAL, see gebal_genMatrix)

// each op_range vector is a {job, side}

// case when n = 0, job = B and side = R will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<vector<char>> op_range = {{'N', 'R'}, {'P', 'R'}, {'S', 'R'}, {'B', 'R'},
                                       {'N', 'L'}, {'P', 'L'}, {'S', 'L'}, {'B', 'L'}};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 5, 1, 2},
    {5, 0, 5, 2},
    // invalid
    {-1, 1, 1, 2},
    {5, -1, 5, 2},
    {5, 5, 4, 2},
    // normal (valid) samples
    {1, 1, 1, 2},
    {10, 5, 10, 2},
    {40, 40, 50, 2},
    {100, 30, 100, 2},
    {300, 120, 300, 1},
    {50, 20, 50, 3},
    {60, 30, 60, 4},
    {60, 30, 60, 5},
};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {1000, 1000, 1000, 2},
    {2000, 500, 2048, 2},
};

Arguments gebak_setup_arguments(gebak_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    vector<char> op = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", matrix_size[0]);
    arg.set<rocblas_int>("m", matrix_size[1]);
    arg.set<rocblas_int>("ldv", matrix_size[2]);
    arg.set<rocblas_int>("mtype", matrix_size[3]);
    arg.set<char>("job", op[0]);
    arg.set<char>("side", op[1]);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

class GEBAK : public ::TestWithParam<gebak_tuple>
{
protected:
    void TearDown() override
    {
        ASSERT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = gebak_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<char>("job") == 'B'
           && arg.peek<char>("side") == 'R')
            testing_gebak_bad_arg<BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_gebak<BATCHED, STRIDED, T>(arg);
    }
};

// non-batch tests

TEST_P(GEBAK, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GEBAK, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GEBAK, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GEBAK, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(GEBAK, batched__float)
{
    run_tests<true, true, float>();
}

TEST_P(GEBAK, batched__double)
{
    run_tests<true, true, double>();
}

TEST_P(GEBAK, batched__float_complex)
{
    run_tests<true, true, rocblas_float_complex>();
}

TEST_P(GEBAK, batched__double_complex)
{
    run_tests<true, true, rocblas_double_complex>();
}

// strided_batched tests

TEST_P(GEBAK, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(GEBAK, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(GEBAK, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(GEBAK, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GEBAK,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GEBAK,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(op_range)));

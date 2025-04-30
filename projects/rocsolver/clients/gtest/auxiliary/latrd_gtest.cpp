/* **************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common/auxiliary/testing_latrd.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> latrd_tuple;

// each matrix_size_range is a {n, lda, ldw}

// each op_range is a {k, ul}
// if ul = 0, then uplo = 'L'
// if ul = 1, then uplo = 'U'

// case when n = 0 and k = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1, 1},
    // invalid
    {-1, 1, 1},
    {20, 5, 20},
    {20, 20, 5},
    // normal (valid) samples
    {35, 50, 50},
    {70, 100, 70},
    {130, 130, 150},
    {150, 150, 150}};

const vector<vector<int>> op_range = {
    // quick return
    {0, 0},
    // invalid
    {-1, 0},
    {180, 0},
    // normal (valid) samples
    {10, 0},
    {25, 1},
    {30, 0},
    {30, 1}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range
    = {{152, 152, 152}, {640, 640, 656}, {1000, 1024, 1000}};

const vector<vector<int>> large_op_range = {{64, 0}, {98, 1}, {130, 0}, {150, 1}};

Arguments latrd_setup_arguments(latrd_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    vector<int> op_size = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", matrix_size[0]);
    arg.set<rocblas_int>("lda", matrix_size[1]);
    arg.set<rocblas_int>("ldw", matrix_size[2]);

    arg.set<rocblas_int>("k", op_size[0]);
    arg.set<char>("uplo", op_size[1] ? 'U' : 'L');

    arg.timing = 0;

    return arg;
}

class LATRD : public ::TestWithParam<latrd_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = latrd_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("k") == 0 && arg.peek<rocblas_int>("n") == 0)
            testing_latrd_bad_arg<T>();

        testing_latrd<T>(arg);
    }
};

// non-batch tests

TEST_P(LATRD, __float)
{
    run_tests<float>();
}

TEST_P(LATRD, __double)
{
    run_tests<double>();
}

TEST_P(LATRD, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(LATRD, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         LATRD,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         LATRD,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(op_range)));

/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING IN ANY WAY OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * ************************************************************************ */

#include "testing_trtri.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<char>> trtri_tuple;

// each matrix_size_range vector is a {n, lda, diag}
// if diag = 0, then the used matrix for the tests is triangular unit
// if diag = 1, then the used matrix for the tests is triangular non-unit

// each op_range vector is a {uplo}

// case when n = 0 and uplo = L will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<vector<char>> op_range = {{'L'}, {'U'}};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1, 0},
    // normal (valid) samples
    {20, 32, 0},
    {30, 30, 1},
    {40, 60, 2},
    {80, 80, 2},
    {90, 100, 1},
    {100, 150, 0}};

// // for daily_lapack tests
// const vector<vector<int>> large_matrix_size_range
//     = {{192, 192, 1}, {500, 600, 2}, {640, 640, 0}, {1000, 1024, 1}, {1200, 1230, 2}};

Arguments trtri_setup_arguments(trtri_tuple tup)
{
    vector<int>  matrix_size = std::get<0>(tup);
    vector<char> op          = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", matrix_size[0]);
    arg.set<rocblas_int>("lda", matrix_size[1]);

    arg.set<char>("uplo", op[0]);

    if(matrix_size[2] == 0)
        arg.set<char>("diag", 'U');
    else
        arg.set<char>("diag", 'N');

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <testAPI_t API, typename I, typename SIZE>
class TRTRI_BASE : public ::TestWithParam<trtri_tuple>
{
protected:
    void TearDown() override
    {
        ASSERT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = trtri_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<char>("uplo") == 'L')
            testing_trtri_bad_arg<API, BATCHED, STRIDED, T, I, SIZE>();

        arg.batch_count = 1;
        testing_trtri<API, BATCHED, STRIDED, T, I, SIZE>(arg);
    }
};

class TRTRI_COMPAT_64 : public TRTRI_BASE<API_COMPAT, int64_t, size_t>
{
};

// non-batch tests

TEST_P(TRTRI_COMPAT_64, __float)
{
    run_tests<false, false, float>();
}

TEST_P(TRTRI_COMPAT_64, __double)
{
    run_tests<false, false, double>();
}

TEST_P(TRTRI_COMPAT_64, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(TRTRI_COMPAT_64, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          TRTRI_COMPAT_64,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         TRTRI_COMPAT_64,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(op_range)));

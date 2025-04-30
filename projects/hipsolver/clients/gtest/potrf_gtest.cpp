/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * ************************************************************************ */

#include "testing_potrf.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, char> potrf_tuple;

// each size_range vector is a {N, lda}
// if singular = 1, then the used matrix for the tests is not positive definite

// each uplo_range is a {uplo}

// case when n = -1 and uplo = L will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<char> uplo_range = {'L', 'U'};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // invalid
    {-1, 1},
    {10, 2},
    // normal (valid) samples
    {10, 10},
    {20, 30},
    {50, 50},
    {70, 80}};

// // for daily_lapack tests
// const vector<vector<int>> large_matrix_size_range = {
//     {192, 192},
//     {640, 960},
//     {1000, 1000},
//     {1024, 1024},
//     {2000, 2000},
// };

Arguments potrf_setup_arguments(potrf_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    char        uplo        = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", matrix_size[0]);
    arg.set<rocblas_int>("lda", matrix_size[1]);

    arg.set<char>("uplo", uplo);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <testAPI_t API>
class POTRF_BASE : public ::TestWithParam<potrf_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = potrf_setup_arguments(GetParam());

        if(arg.peek<char>("uplo") == 'L' && arg.peek<int>("n") == -1)
            testing_potrf_bad_arg<API, BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_potrf<API, BATCHED, STRIDED, T>(arg);
    }
};

class POTRF : public POTRF_BASE<API_NORMAL>
{
};

class POTRF_FORTRAN : public POTRF_BASE<API_FORTRAN>
{
};

class POTRF_COMPAT : public POTRF_BASE<API_COMPAT>
{
};

// non-batch tests

TEST_P(POTRF, __float)
{
    run_tests<false, false, float>();
}

TEST_P(POTRF, __double)
{
    run_tests<false, false, double>();
}

TEST_P(POTRF, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(POTRF, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(POTRF_FORTRAN, __float)
{
    run_tests<false, false, float>();
}

TEST_P(POTRF_FORTRAN, __double)
{
    run_tests<false, false, double>();
}

TEST_P(POTRF_FORTRAN, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(POTRF_FORTRAN, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(POTRF_COMPAT, __float)
{
    run_tests<false, false, float>();
}

TEST_P(POTRF_COMPAT, __double)
{
    run_tests<false, false, double>();
}

TEST_P(POTRF_COMPAT, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(POTRF_COMPAT, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(POTRF, batched__float)
{
    run_tests<true, false, float>();
}

TEST_P(POTRF, batched__double)
{
    run_tests<true, false, double>();
}

TEST_P(POTRF, batched__float_complex)
{
    run_tests<true, false, rocblas_float_complex>();
}

TEST_P(POTRF, batched__double_complex)
{
    run_tests<true, false, rocblas_double_complex>();
}

TEST_P(POTRF_FORTRAN, batched__float)
{
    run_tests<true, false, float>();
}

TEST_P(POTRF_FORTRAN, batched__double)
{
    run_tests<true, false, double>();
}

TEST_P(POTRF_FORTRAN, batched__float_complex)
{
    run_tests<true, false, rocblas_float_complex>();
}

TEST_P(POTRF_FORTRAN, batched__double_complex)
{
    run_tests<true, false, rocblas_double_complex>();
}

TEST_P(POTRF_COMPAT, batched__float)
{
    run_tests<true, false, float>();
}

TEST_P(POTRF_COMPAT, batched__double)
{
    run_tests<true, false, double>();
}

TEST_P(POTRF_COMPAT, batched__float_complex)
{
    run_tests<true, false, rocblas_float_complex>();
}

TEST_P(POTRF_COMPAT, batched__double_complex)
{
    run_tests<true, false, rocblas_double_complex>();
}

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          POTRF,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         POTRF,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          POTRF_FORTRAN,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         POTRF_FORTRAN,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          POTRF_COMPAT,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         POTRF_COMPAT,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));

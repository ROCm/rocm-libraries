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

#include "testing_getrs.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> getrs_tuple;

// each A_range vector is a {N, lda, ldb};

// each B_range vector is a {nrhs, trans};
// if trans = 0 then no transpose
// if trans = 1 then transpose
// if trans = 2 then conjugate transpose

// case when N = nrhs = -1 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> matrix_sizeA_range = {
    // invalid
    {-1, 1, 1},
    {10, 2, 10},
    {10, 10, 2},
    /// normal (valid) samples
    {20, 20, 20},
    {30, 50, 30},
    {30, 30, 50},
    {50, 60, 60}};

const vector<vector<int>> matrix_sizeB_range = {
    // invalid
    {-1, 0},
    // normal (valid) samples
    {10, 0},
    {20, 1},
    {30, 2},
};

// // for daily_lapack tests
// const vector<vector<int>> large_matrix_sizeA_range
//     = {{70, 70, 100}, {192, 192, 192}, {600, 700, 645}, {1000, 1000, 1000}, {1000, 2000, 2000}};

// const vector<vector<int>> large_matrix_sizeB_range = {
//     {100, 0},
//     {150, 0},
//     {200, 1},
//     {524, 2},
//     {1000, 2},
// };

Arguments getrs_setup_arguments(getrs_tuple tup)
{
    vector<int> matrix_sizeA = std::get<0>(tup);
    vector<int> matrix_sizeB = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", matrix_sizeA[0]);
    arg.set<rocblas_int>("nrhs", matrix_sizeB[0]);
    arg.set<rocblas_int>("lda", matrix_sizeA[1]);
    arg.set<rocblas_int>("ldb", matrix_sizeA[2]);

    if(matrix_sizeB[1] == 0)
        arg.set<char>("trans", 'N');
    else if(matrix_sizeB[1] == 1)
        arg.set<char>("trans", 'T');
    else
        arg.set<char>("trans", 'C');

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <testAPI_t API, typename I, typename SIZE>
class GETRS_BASE : public ::TestWithParam<getrs_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = getrs_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == -1 && arg.peek<rocblas_int>("nrhs") == -1)
            testing_getrs_bad_arg<API, BATCHED, STRIDED, T, I, SIZE>();

        arg.batch_count = 1;
        testing_getrs<API, BATCHED, STRIDED, T, I, SIZE>(arg);
    }
};

class GETRS : public GETRS_BASE<API_NORMAL, int, int>
{
};

class GETRS_FORTRAN : public GETRS_BASE<API_FORTRAN, int, int>
{
};

class GETRS_COMPAT : public GETRS_BASE<API_COMPAT, int, int>
{
};

class GETRS_COMPAT_64 : public GETRS_BASE<API_COMPAT, int64_t, size_t>
{
};

// non-batch tests

TEST_P(GETRS, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GETRS, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GETRS, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GETRS, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GETRS_FORTRAN, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GETRS_FORTRAN, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GETRS_FORTRAN, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GETRS_FORTRAN, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GETRS_COMPAT, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GETRS_COMPAT, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GETRS_COMPAT, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GETRS_COMPAT, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GETRS_COMPAT_64, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GETRS_COMPAT_64, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GETRS_COMPAT_64, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GETRS_COMPAT_64, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          GETRS,
//                          Combine(ValuesIn(large_matrix_sizeA_range),
//                                  ValuesIn(large_matrix_sizeB_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GETRS,
                         Combine(ValuesIn(matrix_sizeA_range), ValuesIn(matrix_sizeB_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          GETRS_FORTRAN,
//                          Combine(ValuesIn(large_matrix_sizeA_range),
//                                  ValuesIn(large_matrix_sizeB_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GETRS_FORTRAN,
                         Combine(ValuesIn(matrix_sizeA_range), ValuesIn(matrix_sizeB_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          GETRS_COMPAT,
//                          Combine(ValuesIn(large_matrix_sizeA_range),
//                                  ValuesIn(large_matrix_sizeB_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GETRS_COMPAT,
                         Combine(ValuesIn(matrix_sizeA_range), ValuesIn(matrix_sizeB_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          GETRS_COMPAT_64,
//                          Combine(ValuesIn(large_matrix_sizeA_range),
//                                  ValuesIn(large_matrix_sizeB_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GETRS_COMPAT_64,
                         Combine(ValuesIn(matrix_sizeA_range), ValuesIn(matrix_sizeB_range)));

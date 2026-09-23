/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "testing_sytrs.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>, char> sytrs_tuple;

// each A_range vector is a {n, lda, ldb}

// each B_range vector is a {nrhs}

// each uplo_range is a {uplo}

// case when n = -1 and nrhs = -1 and uplo = L will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<char> uplo_range = {'L', 'U'};

// for checkin_lapack tests
const vector<vector<int>> matrix_sizeA_range = {
    // invalid
    {-1, 1, 1},
    {20, 5, 20},
    {20, 20, 5},
    // valid
    {20, 20, 20},
    {30, 50, 30},
    {30, 30, 50},
    {50, 60, 60}};

const vector<vector<int>> matrix_sizeB_range = {
    // invalid
    {-1},
    // valid
    {10},
    {20},
    {30}};

Arguments sytrs_setup_arguments(sytrs_tuple tup)
{
    vector<int> matrix_sizeA = std::get<0>(tup);
    vector<int> matrix_sizeB = std::get<1>(tup);
    char        uplo         = std::get<2>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", matrix_sizeA[0]);
    arg.set<rocblas_int>("lda", matrix_sizeA[1]);
    arg.set<rocblas_int>("ldb", matrix_sizeA[2]);
    arg.set<rocblas_int>("nrhs", matrix_sizeB[0]);

    arg.set<char>("uplo", uplo);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <testAPI_t API, typename I, typename SIZE>
class SYTRS_BASE : public ::TestWithParam<sytrs_tuple>
{
protected:
    void TearDown() override
    {
        ASSERT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = sytrs_setup_arguments(GetParam());

        if(arg.peek<char>("uplo") == 'L' && arg.peek<rocblas_int>("n") == -1
           && arg.peek<rocblas_int>("nrhs") == -1)
            testing_sytrs_bad_arg<API, BATCHED, STRIDED, T, I, SIZE>();

        arg.batch_count = 1;
        testing_sytrs<API, BATCHED, STRIDED, T, I, SIZE>(arg);
    }
};

class SYTRS_COMPAT : public SYTRS_BASE<API_COMPAT, int64_t, size_t>
{
};

// non-batch tests

TEST_P(SYTRS_COMPAT, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYTRS_COMPAT, __double)
{
    run_tests<false, false, double>();
}

TEST_P(SYTRS_COMPAT, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(SYTRS_COMPAT, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          SYTRS_COMPAT,
//                          Combine(ValuesIn(large_matrix_sizeA_range),
//                                  ValuesIn(large_matrix_sizeB_range),
//                                  ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYTRS_COMPAT,
                         Combine(ValuesIn(matrix_sizeA_range),
                                 ValuesIn(matrix_sizeB_range),
                                 ValuesIn(uplo_range)));

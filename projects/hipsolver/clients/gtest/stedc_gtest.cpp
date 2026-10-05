/* ************************************************************************
 * Copyright (C) 2021-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "testing_stedc.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, char> stedc_tuple;

// each size_range vector is a {N, ldc}

// each op_range vector is a {e}

// case when n == 0 and jobz == N will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<char> op_range = {'N', 'I', 'V'};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // quick return
    {0, 1},
    // invalid
    {-1, 1},
    // invalid for case evect != N
    {2, 1},
    // normal (valid) samples
    {12, 12},
    {20, 30},
    {35, 40}};

// // for daily_lapack tests
// const vector<vector<int>> large_matrix_size_range = {{192, 192}, {250, 250}, {256, 270}, {300, 300}};

Arguments stedc_setup_arguments(stedc_tuple tup)
{
    vector<int> size = std::get<0>(tup);
    char        op   = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", size[0]);
    arg.set<rocblas_int>("ldc", size[1]);

    arg.set<char>("jobz", op);

    arg.timing = 0;

    return arg;
}

template <testAPI_t API, typename I, typename SIZE>
class STEDC_BASE : public ::TestWithParam<stedc_tuple>
{
protected:
    void TearDown() override
    {
        ASSERT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = stedc_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 0 && arg.peek<char>("jobz") == 'N')
            testing_stedc_bad_arg<API, BATCHED, STRIDED, T, I, SIZE>();

        testing_stedc<API, BATCHED, STRIDED, T, I, SIZE>(arg);
    }
};

class STEDC_COMPAT_64 : public STEDC_BASE<API_COMPAT, int64_t, size_t>
{
};

// non-batch tests

TEST_P(STEDC_COMPAT_64, __float)
{
    run_tests<false, false, float>();
}

TEST_P(STEDC_COMPAT_64, __double)
{
    run_tests<false, false, double>();
}

TEST_P(STEDC_COMPAT_64, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(STEDC_COMPAT_64, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          STEDC_COMPAT_64,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         STEDC_COMPAT_64,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(op_range)));

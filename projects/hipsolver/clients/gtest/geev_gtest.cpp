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
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * ************************************************************************ */

#include "testing_geev.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<char>> geev_tuple;

// each matrix_size_range vector is a {n, lda, ldvl, ldvr}

// each op_range vector is a {jobvl, jobvr}

// case when n = -1, jobvl = N, and jobvr = N will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<vector<char>> op_range = {{'N', 'N'}, {'N', 'V'}};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // invalid
    {-1, 1, 1, 1},
    {20, 5, 20, 20},
    {20, 20, -1, 20},
    {20, 20, 20, -1},
    // normal (valid) samples
    {32, 32, 36, 32},
    {50, 50, 50, 56},
    {70, 100, 70, 70},
};

// // for daily_lapack tests
// const vector<vector<int>> large_matrix_size_range = {
//     {192, 192, 198, 192},
//     {640, 640, 640, 646},
//     {1000, 1024, 1000, 1000},
// };

Arguments geev_setup_arguments(geev_tuple tup)
{
    vector<int>  matrix_size = std::get<0>(tup);
    vector<char> op          = std::get<1>(tup);

    Arguments arg;

    arg.set<int>("n", matrix_size[0]);
    arg.set<int>("lda", matrix_size[1]);
    arg.set<int>("ldvl", matrix_size[2]);
    arg.set<int>("ldvr", matrix_size[3]);

    arg.set<char>("jobvl", op[0]);
    arg.set<char>("jobvr", op[1]);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <testAPI_t API, typename I, typename SIZE>
class GEEV_BASE : public ::TestWithParam<geev_tuple>
{
protected:
    void TearDown() override
    {
        ASSERT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T, typename W>
    void run_tests()
    {
        Arguments arg = geev_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == -1 && arg.peek<char>("jobvl") == 'N'
           && arg.peek<char>("jobvr") == 'N')
            testing_geev_bad_arg<API, BATCHED, STRIDED, T, W, I, SIZE>();

        arg.batch_count = 1;
        testing_geev<API, BATCHED, STRIDED, T, W, I, SIZE>(arg);
    }
};

class GEEV_COMPAT_64 : public GEEV_BASE<API_COMPAT, int64_t, size_t>
{
};

// non-batch tests

TEST_P(GEEV_COMPAT_64, __float)
{
    run_tests<false, false, float, float>();
}

TEST_P(GEEV_COMPAT_64, __float_alt)
{
    run_tests<false, false, float, hipsolverComplex>();
}

TEST_P(GEEV_COMPAT_64, __double)
{
    run_tests<false, false, double, double>();
}

TEST_P(GEEV_COMPAT_64, __double_alt)
{
    run_tests<false, false, double, hipsolverDoubleComplex>();
}

TEST_P(GEEV_COMPAT_64, __float_complex)
{
    run_tests<false, false, hipsolverComplex, hipsolverComplex>();
}

TEST_P(GEEV_COMPAT_64, __double_complex)
{
    run_tests<false, false, hipsolverDoubleComplex, hipsolverDoubleComplex>();
}

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          GEEV_COMPAT_64,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GEEV_COMPAT_64,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(op_range)));

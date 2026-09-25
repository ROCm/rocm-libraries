/* ************************************************************************
 * Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "testing_larft.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<char>> larft_tuple;

// each size_range is a {n, k, ldv, ldt}
// if n = -1 and k = -1, then the bad arguments test will be run

// each op_range vector is a {direct, storev}

// case when n == -1, k == -1, direct == F, and storev == C will also execute the bad arguments test
// (null handle, null pointers and invalid values)

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
const vector<vector<char>> op_range = {{'F', 'C'}, {'F', 'R'}, {'B', 'C'}, {'B', 'R'}};
#else
const vector<vector<char>> op_range = {{'F', 'C'}, {'B', 'C'}};
#endif

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // normal (valid) samples
    {10, 5, 10, 5},
    {20, 10, 20, 10},
    {30, 15, 30, 15},
    {50, 25, 50, 25}};

// for daily_lapack tests
// const vector<vector<int>> large_size_range
//     = {{100, 50, 100, 50}, {200, 100, 200, 100}, {500, 250, 500, 250}, {1000, 500, 1000, 500}};

Arguments larft_setup_arguments(larft_tuple tup)
{
    vector<int>  size = std::get<0>(tup);
    vector<char> op   = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", size[0]);
    arg.set<rocblas_int>("k", size[1]);
    arg.set<rocblas_int>("ldv", size[2]);
    arg.set<rocblas_int>("ldt", size[3]);

    arg.set<char>("direct", op[0]);
    arg.set<char>("storev", op[1]);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <testAPI_t API, typename I, typename SIZE>
class LARFT_BASE : public ::TestWithParam<larft_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = larft_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == -1 && arg.peek<rocblas_int>("k") == -1
           && arg.peek<char>("direct") == 'F' && arg.peek<char>("storev") == 'C')
            testing_larft_bad_arg<API, BATCHED, STRIDED, T, I, SIZE>();

        arg.batch_count = 1;
        testing_larft<API, BATCHED, STRIDED, T, I, SIZE>(arg);
    }
};

class LARFT_COMPAT_64 : public LARFT_BASE<API_COMPAT, int64_t, size_t>
{
};

// non-batch tests

TEST_P(LARFT_COMPAT_64, __float)
{
    run_tests<false, false, float>();
}

TEST_P(LARFT_COMPAT_64, __double)
{
    run_tests<false, false, double>();
}

TEST_P(LARFT_COMPAT_64, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(LARFT_COMPAT_64, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          LARFT_COMPAT_64,
//                          Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         LARFT_COMPAT_64,
                         Combine(ValuesIn(size_range), ValuesIn(op_range)));

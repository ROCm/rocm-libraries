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

#include "testing_syev_heev.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
typedef std::tuple<std::vector<int>, std::vector<char>> syev_heev_tuple;

// each size_range vector is a {n, lda}

// each op_range vector is a {jobz, uplo}

// case when n == 1, jobz == N, and uplo = L will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const std::vector<std::vector<char>> op_range = {{'N', 'L'}, {'N', 'U'}, {'V', 'L'}, {'V', 'U'}};

// for checkin_lapack tests
const std::vector<std::vector<int>> size_range = {
    // normal (valid) samples
    {1, 1},
    {12, 12},
    {20, 30},
    {35, 35},
    {50, 60}};

// for daily_lapack tests
// const std::vector<std::vector<int>> large_size_range = {
//     {192, 192},
//     {500, 600},
//     {640, 640},
//     {1000, 1024}};

template <typename T>
Arguments syev_heev_setup_arguments(syev_heev_tuple tup)
{
    std::vector<int>  size = std::get<0>(tup);
    std::vector<char> op   = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", size[0]);
    arg.set<rocblas_int>("lda", size[1]);

    arg.set<char>("jobz", op[0]);
    arg.set<char>("uplo", op[1]);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <testAPI_t API, typename I, typename SIZE>
class SYEV_HEEV_BASE : public ::TestWithParam<syev_heev_tuple>
{
protected:
    void TearDown() override
    {
        ASSERT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = syev_heev_setup_arguments<T>(GetParam());

        if(arg.peek<rocblas_int>("n") == 1 && arg.peek<char>("jobz") == 'N'
           && arg.peek<char>("uplo") == 'L')
            testing_syev_heev_bad_arg<API, BATCHED, STRIDED, T, I, SIZE>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_syev_heev<API, BATCHED, STRIDED, T, I, SIZE>(arg);
    }
};

class SYEV_COMPAT_64 : public SYEV_HEEV_BASE<API_COMPAT, int64_t, size_t>
{
};

class HEEV_COMPAT_64 : public SYEV_HEEV_BASE<API_COMPAT, int64_t, size_t>
{
};

// Only strided_batched tests are instantiated here, as the underlying API
// (hipsolverDnXsyevBatched) operates on strided-batched data layouts.

TEST_P(SYEV_COMPAT_64, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(SYEV_COMPAT_64, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(HEEV_COMPAT_64, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(HEEV_COMPAT_64, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYEV_COMPAT_64,
                         Combine(ValuesIn(size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HEEV_COMPAT_64,
                         Combine(ValuesIn(size_range), ValuesIn(op_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          SYEV_COMPAT_64,
//                          Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          HEEV_COMPAT_64,
//                          Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

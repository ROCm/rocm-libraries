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

#include "testing_sygvdx_hegvdx.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<char>> sygvdx_tuple;

// each matrix_size_range is a {n, lda, ldb, vl, vu, il, iu}

// each type_range is a {itype, jobz, range, uplo}

// case when n = -1, itype = 1, jobz = 'N', range = 'A', and uplo = U will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<vector<char>> type_range = {{'1', 'N', 'A', 'U'},
                                         {'2', 'N', 'V', 'L'},
                                         {'3', 'N', 'I', 'U'},
                                         {'1', 'V', 'V', 'L'},
                                         {'2', 'V', 'I', 'U'},
                                         {'3', 'V', 'A', 'L'}};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // invalid
    {-1, 1, 1, 0, 10, 1, 1},
    {20, 5, 5, 0, 10, 1, 1},
    // valid only when erange=A
    {20, 20, 20, 10, 0, 10, 1},
    // normal (valid) samples
    {21, 30, 20, 5, 15, 1, 10},
    {35, 35, 35, -10, 10, 1, 35},
    {50, 50, 60, -15, -5, 25, 50},
};

// // for daily_lapack tests
// const vector<vector<int>> large_matrix_size_range = {
//     {192, 192, 192, 5, 15, 100, 150},
//     {256, 270, 256, -10, 10, 1, 100},
//     {300, 300, 310, -15, -5, 200, 300},
// };

template <typename T>
Arguments sygvdx_setup_arguments(sygvdx_tuple tup)
{
    using S = decltype(std::real(T{}));

    vector<int>  matrix_size = std::get<0>(tup);
    vector<char> type        = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", matrix_size[0]);
    arg.set<rocblas_int>("lda", matrix_size[1]);
    arg.set<rocblas_int>("ldb", matrix_size[2]);
    arg.set<double>("vl", matrix_size[3]);
    arg.set<double>("vu", matrix_size[4]);
    arg.set<rocblas_int>("il", matrix_size[5]);
    arg.set<rocblas_int>("iu", matrix_size[6]);

    arg.set<char>("itype", type[0]);
    arg.set<char>("jobz", type[1]);
    arg.set<char>("range", type[2]);
    arg.set<char>("uplo", type[3]);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <testAPI_t API>
class SYGVDX_HEGVDX : public ::TestWithParam<sygvdx_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = sygvdx_setup_arguments<T>(GetParam());

        if(arg.peek<char>("itype") == '1' && arg.peek<char>("jobz") == 'N'
           && arg.peek<char>("range") == 'A' && arg.peek<char>("uplo") == 'U'
           && arg.peek<rocblas_int>("n") == -1)
            testing_sygvdx_hegvdx_bad_arg<API, BATCHED, STRIDED, T>();

        arg.batch_count = 1;
        testing_sygvdx_hegvdx<API, BATCHED, STRIDED, T>(arg);
    }
};

class SYGVDX : public SYGVDX_HEGVDX<API_NORMAL>
{
};

class HEGVDX : public SYGVDX_HEGVDX<API_NORMAL>
{
};

// non-batch tests

TEST_P(SYGVDX, DISABLED__float)
{
    run_tests<false, false, float>();
}

TEST_P(SYGVDX, DISABLED__double)
{
    run_tests<false, false, double>();
}

TEST_P(HEGVDX, DISABLED__float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HEGVDX, DISABLED__double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          SYGVDX,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYGVDX,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(type_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          HEGVDX,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HEGVDX,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(type_range)));

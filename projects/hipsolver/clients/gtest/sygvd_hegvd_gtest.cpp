/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "testing_sygvd_hegvd.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<char>> sygvd_tuple;

// each matrix_size_range is a {n, lda, ldb}

// each type_range is a {itype, jobz, uplo}

// case when n = -1, itype = 1, jobz = 'N', and uplo = U will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<vector<char>> type_range = {{'1', 'N', 'U'},
                                         {'2', 'N', 'L'},
                                         {'3', 'N', 'U'},
                                         {'1', 'V', 'L'},
                                         {'2', 'V', 'U'},
                                         {'3', 'V', 'L'}};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // invalid
    {-1, 1, 1},
    {20, 5, 5},
    // normal (valid) samples
    {20, 30, 20},
    {35, 35, 35},
    {50, 50, 60}};

// // for daily_lapack tests
// const vector<vector<int>> large_matrix_size_range = {
//     {192, 192, 192},
//     {256, 270, 256},
//     {300, 300, 310},
// };

Arguments sygvd_setup_arguments(sygvd_tuple tup)
{
    vector<int>  matrix_size = std::get<0>(tup);
    vector<char> type        = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", matrix_size[0]);
    arg.set<rocblas_int>("lda", matrix_size[1]);
    arg.set<rocblas_int>("ldb", matrix_size[2]);

    arg.set<char>("itype", type[0]);
    arg.set<char>("jobz", type[1]);
    arg.set<char>("uplo", type[2]);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <testAPI_t API>
class SYGVD_HEGVD : public ::TestWithParam<sygvd_tuple>
{
protected:
    void TearDown() override
    {
        EXPECT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = sygvd_setup_arguments(GetParam());

        if(arg.peek<char>("itype") == '1' && arg.peek<char>("jobz") == 'N'
           && arg.peek<char>("uplo") == 'U' && arg.peek<rocblas_int>("n") == -1)
            testing_sygvd_hegvd_bad_arg<API, BATCHED, STRIDED, T>();

        arg.batch_count = 1;
        testing_sygvd_hegvd<API, BATCHED, STRIDED, T>(arg);
    }
};

class SYGVD : public SYGVD_HEGVD<API_NORMAL>
{
};

class HEGVD : public SYGVD_HEGVD<API_NORMAL>
{
};

class SYGVD_FORTRAN : public SYGVD_HEGVD<API_FORTRAN>
{
};

class HEGVD_FORTRAN : public SYGVD_HEGVD<API_FORTRAN>
{
};

class SYGVD_COMPAT : public SYGVD_HEGVD<API_COMPAT>
{
};

class HEGVD_COMPAT : public SYGVD_HEGVD<API_COMPAT>
{
};

// non-batch tests

TEST_P(SYGVD, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYGVD, __double)
{
    run_tests<false, false, double>();
}

TEST_P(HEGVD, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HEGVD, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(SYGVD_FORTRAN, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYGVD_FORTRAN, __double)
{
    run_tests<false, false, double>();
}

TEST_P(HEGVD_FORTRAN, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HEGVD_FORTRAN, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(SYGVD_COMPAT, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYGVD_COMPAT, __double)
{
    run_tests<false, false, double>();
}

TEST_P(HEGVD_COMPAT, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HEGVD_COMPAT, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          SYGVD,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYGVD,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(type_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          HEGVD,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HEGVD,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(type_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          SYGVD_FORTRAN,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYGVD_FORTRAN,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(type_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          HEGVD_FORTRAN,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HEGVD_FORTRAN,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(type_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          SYGVD_COMPAT,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYGVD_COMPAT,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(type_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          HEGVD_COMPAT,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(type_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HEGVD_COMPAT,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(type_range)));
